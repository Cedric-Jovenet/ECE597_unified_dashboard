import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import soundfile as sf
import inspect
import os
import shutil
import io
import pandas as pd
from pesq import pesq
import pystoi
import fitz

from src.catalogs import (
    get_algorithm_catalog,
    get_benchmark_catalog,
    get_dataset_catalog,
    get_knowledge_topics,
    get_reference_study_catalog,
)

# --- UTILITY FUNCTIONS FOR NOIZEUS ---
def get_noizeus_samples():
    """Return the list of available NOIZEUS conditions."""
    base_path = "assets/NOIZEUS"
    samples = []

    if not os.path.exists(base_path):
        return samples

    # List available noise conditions
    noise_types = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and not d.endswith('.zip')]

    for noise_type in sorted(noise_types):
        if noise_type == 'clean':
            continue
        # Extract the noise type and SNR
        parts = noise_type.split('_')
        if len(parts) >= 2:
            noise_name = parts[0]
            snr = parts[1]

            # Count available files
            noise_path = os.path.join(base_path, noise_type)
            if os.path.exists(noise_path):
                wav_files = [f for f in os.listdir(noise_path) if f.endswith('.wav')]
                samples.append({
                    'name': f"{noise_name.upper()} - {snr} SNR",
                    'noise_type': noise_type,
                    'noise_name': noise_name,
                    'snr': snr,
                    'count': len(wav_files)
                })

    return samples

def load_noizeus_sample(noise_type, sample_idx=1):
    """Load a NOIZEUS sample (noisy + clean)."""
    base_path = "assets/NOIZEUS"

    # Load the noisy signal
    noisy_file = f"sp{sample_idx:02d}_{noise_type.split('_')[0]}_sn{noise_type.split('_')[1][:-2]}.wav"
    noisy_path = os.path.join(base_path, noise_type, noisy_file)

    # Load the matching clean signal
    clean_file = f"sp{sample_idx:02d}.wav"
    clean_path = os.path.join(base_path, "clean", clean_file)

    if not os.path.exists(noisy_path) or not os.path.exists(clean_path):
        return None, None, None

    # Load signals
    y_noisy, sr_noisy = librosa.load(noisy_path, sr=None)
    y_clean, sr_clean = librosa.load(clean_path, sr=None)

    # Ensure the sample rates are consistent
    if sr_noisy != sr_clean:
        # Resample clean to match noisy
        y_clean = librosa.resample(y_clean, orig_sr=sr_clean, target_sr=sr_noisy)
        sr_clean = sr_noisy

    return y_noisy, y_clean, sr_noisy

def ensure_static_pdf(pdf_source_path, static_pdf_filename):
    """Expose a PDF through Streamlit's static folder and return its served URL."""
    if not os.path.exists(pdf_source_path):
        return None

    static_pdf_path = os.path.join("static", static_pdf_filename)
    static_pdf_url = f"/app/static/{static_pdf_filename}"

    if not os.path.exists(static_pdf_path):
        shutil.copyfile(pdf_source_path, static_pdf_path)

    return static_pdf_url

def render_pdf_preview(pdf_source_path, max_pages=3, zoom=1.5):
    """Render the first pages of a PDF as images to avoid browser PDF iframe issues."""
    if not os.path.exists(pdf_source_path):
        return False

    document = fitz.open(pdf_source_path)
    try:
        page_count = min(max_pages, len(document))
        matrix = fitz.Matrix(zoom, zoom)

        for page_index in range(page_count):
            page = document.load_page(page_index)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            page_image = io.BytesIO(pixmap.tobytes("png"))
            st.image(page_image, caption=f"PDF preview - page {page_index + 1}", use_column_width=True)
    finally:
        document.close()

    return True

def calculate_metrics(y_original, y_clean, y_enhanced, sr):
    """Compute objective metrics."""
    metrics = {}

    # Ensure all signals have the same length
    min_length = min(len(y_original), len(y_clean), len(y_enhanced))
    y_original = y_original[:min_length]
    y_clean = y_clean[:min_length]
    y_enhanced = y_enhanced[:min_length]

    # SNR before / after
    def calculate_snr(signal, noise):
        if len(signal) == 0 or len(noise) == 0:
            return 0.0
        signal_power = np.mean(signal**2)
        noise_power = np.mean(noise**2)
        if noise_power == 0:
            return float('inf') if signal_power > 0 else 0.0
        return 10 * np.log10(signal_power / noise_power)

    def calculate_segmental_snr(clean_signal, noisy_signal, frame_len, hop_len):
        """Compute segmental SNR averaged across short frames."""
        if frame_len <= 0 or hop_len <= 0 or len(clean_signal) < frame_len or len(noisy_signal) < frame_len:
            return 0.0

        snr_frames = []
        for start in range(0, len(clean_signal) - frame_len + 1, hop_len):
            clean_frame = clean_signal[start:start + frame_len]
            noisy_frame = noisy_signal[start:start + frame_len]
            noise_frame = noisy_frame - clean_frame

            clean_energy = np.sum(clean_frame ** 2)
            noise_energy = np.sum(noise_frame ** 2)
            frame_snr = 10 * np.log10((clean_energy + 1e-12) / (noise_energy + 1e-12))
            # Standard clipping often used in SSNR reporting.
            snr_frames.append(np.clip(frame_snr, -10.0, 35.0))

        if not snr_frames:
            return 0.0
        return float(np.mean(snr_frames))

    # Noisy signal SNR
    noise = y_original - y_clean
    snr_input = calculate_snr(y_clean, noise)

    # Enhanced signal SNR
    noise_enhanced = y_enhanced - y_clean
    snr_output = calculate_snr(y_clean, noise_enhanced)

    metrics['snr_improvement'] = snr_output - snr_input

    # Segmental SNR improvement (20 ms frames, 50% overlap)
    frame_len = max(1, int(0.02 * sr))
    hop_len = max(1, frame_len // 2)
    seg_snr_input = calculate_segmental_snr(y_clean, y_original, frame_len, hop_len)
    seg_snr_output = calculate_segmental_snr(y_clean, y_enhanced, frame_len, hop_len)
    metrics['seg_snr_improvement'] = seg_snr_output - seg_snr_input

    # PESQ (if available)
    try:
        if sr == 8000:
            pesq_sr = 8000
            pesq_mode = 'nb'
            y_clean_pesq = y_clean
            y_enhanced_pesq = y_enhanced
        else:
            pesq_sr = 16000
            pesq_mode = 'wb'
            if sr != 16000:
                y_clean_pesq = librosa.resample(y_clean, orig_sr=sr, target_sr=16000)
                y_enhanced_pesq = librosa.resample(y_enhanced, orig_sr=sr, target_sr=16000)
            else:
                y_clean_pesq = y_clean
                y_enhanced_pesq = y_enhanced

        # Normalize
        y_clean_pesq = y_clean_pesq / np.max(np.abs(y_clean_pesq)) if np.max(np.abs(y_clean_pesq)) > 0 else y_clean_pesq
        y_enhanced_pesq = y_enhanced_pesq / np.max(np.abs(y_enhanced_pesq)) if np.max(np.abs(y_enhanced_pesq)) > 0 else y_enhanced_pesq

        pesq_score = pesq(pesq_sr, y_clean_pesq, y_enhanced_pesq, pesq_mode)
        metrics['pesq'] = pesq_score
    except Exception as e:
        metrics['pesq'] = None

    # STOI
    try:
        stoi_score = pystoi.stoi(y_clean, y_enhanced, sr)
        metrics['stoi'] = stoi_score
    except Exception as e:
        metrics['stoi'] = None

    return metrics

def plot_signal_analysis(y_noisy, y_clean, y_enhanced, sr, title="Signal Analysis"):
    """Create a comprehensive signal analysis figure."""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(title, fontsize=16)

    # Time-domain waveform
    time = np.linspace(0, len(y_noisy) / sr, len(y_noisy))

    axes[0, 0].plot(time, y_noisy, alpha=0.7, label='Noisy')
    axes[0, 0].plot(time, y_clean, alpha=0.7, label='Clean')
    axes[0, 0].set_title('Waveform - Comparison')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(time, y_enhanced, color='green', label='Enhanced')
    axes[0, 1].plot(time, y_clean, alpha=0.7, color='orange', label='Clean')
    axes[0, 1].set_title('Waveform - Result')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Spectrograms
    D_noisy = librosa.amplitude_to_db(np.abs(librosa.stft(y_noisy)), ref=np.max)
    D_clean = librosa.amplitude_to_db(np.abs(librosa.stft(y_clean)), ref=np.max)
    D_enhanced = librosa.amplitude_to_db(np.abs(librosa.stft(y_enhanced)), ref=np.max)

    img1 = librosa.display.specshow(D_noisy, y_axis='hz', x_axis='time', sr=sr, ax=axes[1, 0], cmap='viridis')
    axes[1, 0].set_title('Spectrogram - Noisy signal')
    plt.colorbar(img1, ax=axes[1, 0], format='%+2.0f dB')

    img2 = librosa.display.specshow(D_enhanced, y_axis='hz', x_axis='time', sr=sr, ax=axes[1, 1], cmap='viridis')
    axes[1, 1].set_title('Spectrogram - Enhanced signal')
    plt.colorbar(img2, ax=axes[1, 1], format='%+2.0f dB')

    # Power spectra
    freqs = np.fft.rfftfreq(len(y_noisy), 1/sr)
    fft_noisy = np.abs(np.fft.rfft(y_noisy))
    fft_clean = np.abs(np.fft.rfft(y_clean))
    fft_enhanced = np.abs(np.fft.rfft(y_enhanced))

    axes[2, 0].semilogx(freqs, 20*np.log10(fft_noisy + 1e-10), alpha=0.7, label='Noisy')
    axes[2, 0].semilogx(freqs, 20*np.log10(fft_clean + 1e-10), alpha=0.7, label='Clean')
    axes[2, 0].set_title('Power Spectrum - Original')
    axes[2, 0].set_xlabel('Frequency (Hz)')
    axes[2, 0].set_ylabel('Magnitude (dB)')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    axes[2, 1].semilogx(freqs, 20*np.log10(fft_enhanced + 1e-10), color='green', label='Enhanced')
    axes[2, 1].semilogx(freqs, 20*np.log10(fft_clean + 1e-10), alpha=0.7, color='orange', label='Clean')
    axes[2, 1].set_title('Power Spectrum - Enhanced')
    axes[2, 1].set_xlabel('Frequency (Hz)')
    axes[2, 1].set_ylabel('Magnitude (dB)')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def render_markdown_list(items):
    """Render simple bullet lists from a Python list."""
    for item in items:
        st.markdown(f"- {item}")


FAMILY_COLORS = {
    "Spectral Subtraction": "#93c5fd",
    "Wiener Filtering": "#fca5a5",
    "Statistical Model-Based Enhancement": "#86efac",
    "Subspace Enhancement": "#fcd34d",
}


def get_family_segments(labels, family_lookup):
    """Return contiguous family spans for the ordered algorithm labels."""
    if not labels:
        return []

    segments = []
    start_index = 1
    current_family = family_lookup[labels[0]]

    for index, label in enumerate(labels, start=1):
        family = family_lookup[label]
        if family != current_family:
            segments.append((current_family, start_index, index - 1))
            start_index = index
            current_family = family

    segments.append((current_family, start_index, len(labels)))
    return segments


def style_benchmark_axis(axis, labels, family_lookup, boxplot_artists=None):
    """Rotate labels vertically and visually delimit algorithm families."""
    if not labels:
        return

    if boxplot_artists is not None:
        for index, box in enumerate(boxplot_artists['boxes']):
            family = family_lookup[labels[index]]
            box.set_facecolor(FAMILY_COLORS.get(family, "#d1d5db"))
            box.set_edgecolor("#4b5563")
            box.set_alpha(0.6)

        for median in boxplot_artists['medians']:
            median.set_color("#111827")
            median.set_linewidth(1.5)

    for family, start, end in get_family_segments(labels, family_lookup):
        family_color = FAMILY_COLORS.get(family, "#d1d5db")
        axis.axvspan(start - 0.5, end + 0.5, color=family_color, alpha=0.08, zorder=0)
        if end < len(labels):
            axis.axvline(end + 0.5, color="#6b7280", linestyle="--", linewidth=1, alpha=0.8)

    axis.set_xlim(0.5, len(labels) + 0.5)
    axis.set_xticks(range(1, len(labels) + 1))
    axis.set_xticklabels(labels, rotation=90, ha="center", va="top", fontsize=8)
    axis.tick_params(axis='x', pad=6)
    axis.set_xlabel("Algorithms grouped by family", labelpad=12)


def add_family_legend(fig, labels, family_lookup):
    """Add a shared family legend for the benchmark figure."""
    ordered_families = []
    for label in labels:
        family = family_lookup[label]
        if family not in ordered_families:
            ordered_families.append(family)

    handles = [
        Patch(
            facecolor=FAMILY_COLORS.get(family, "#d1d5db"),
            edgecolor="#4b5563",
            alpha=0.6,
            label=family,
        )
        for family in ordered_families
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=min(4, len(handles)), frameon=False)

def run_algorithm(algorithm_meta, y_noisy, sr):
    """Execute an algorithm based on the catalog metadata."""
    if not algorithm_meta['implemented'] or algorithm_meta['runner'] is None:
        return None

    if algorithm_meta['needs_sr']:
        return algorithm_meta['runner'](y_noisy, sr)

    return algorithm_meta['runner'](y_noisy)


def format_protocol_for_table(protocol):
    """Flatten protocol metadata for dataframe display."""
    return {
        'Sample rate': protocol['sample_rate'],
        'Subset': protocol['subset'],
        'SNR levels': ", ".join(protocol['snr_levels']) if protocol['snr_levels'] else "Not recorded",
        'Metrics': ", ".join(protocol['metrics']) if protocol['metrics'] else "Not recorded",
        'Preprocessing': protocol['preprocessing'],
    }

st.set_page_config(page_title="Speech Enhancement Research", layout="wide")

st.title("🎙️ Unified Speech Enhancement Research Dashboard")
st.markdown("This app presents classical speech enhancement algorithms, datasets, benchmark results, and literature notes in one place so the project stays rigorous, comparable, and easy to demonstrate. - Cédric JOVENET")

noizeus_samples = get_noizeus_samples()
algorithm_catalog = get_algorithm_catalog()
dataset_catalog = get_dataset_catalog(noizeus_samples)
benchmark_catalog = get_benchmark_catalog()
knowledge_topics = get_knowledge_topics()
reference_studies = get_reference_study_catalog()
reference_studies_by_lane = {}
for study in reference_studies:
    reference_studies_by_lane.setdefault(study['lane_id'], []).append(study)

spectral_subtraction_presentation_pdf_source = os.path.join(
    "assets",
    "pdf",
    "presentations",
    "specsub1.pdf"
)
spectral_subtraction_presentation_pdf_url = ensure_static_pdf(
    spectral_subtraction_presentation_pdf_source,
    "spectral_subtraction_presentation.pdf"
)

speech_enhancement_algorithms = algorithm_catalog["Speech Enhancement"]
track_algorithms = speech_enhancement_algorithms
available_families = sorted({algorithm['family'] for algorithm in track_algorithms})
selected_family = available_families[0]
family_algorithms = [algorithm for algorithm in track_algorithms if algorithm['family'] == selected_family]
selected_algorithm_label = family_algorithms[0]['label']
selected_algorithm = family_algorithms[0]

dataset_options = [label for label, meta in dataset_catalog.items() if meta['task_scope'] in {"exploration", "speech_enhancement"}]
dataset = dataset_options[0]
selected_dataset_meta = dataset_catalog[dataset]

selected_sample = None
sample_number = 1

run_processing = False

tab_workspace, tab_datasets, tab_benchmarks, tab_knowledge = st.tabs([
    "🧪 Experiment Workspace",
    "🗂️ Datasets",
    "📏 Benchmarks",
    "📚 Knowledge Base"
])

with tab_workspace:
    st.header("🧪 Experiment Workspace")
    st.markdown("Use this page for qualitative listening tests and single-sample inspection. The full benchmark study stays in its own page.")

    st.markdown("### Experiment Configuration")
    config_col1, config_col2 = st.columns(2)

    with config_col1:
        st.caption("Research objective: single-channel speech enhancement")
        track_algorithms = speech_enhancement_algorithms
        available_families = sorted({algorithm['family'] for algorithm in track_algorithms})
        selected_family = st.selectbox(
            "Algorithm family",
            available_families,
            key="workspace_algorithm_family"
        )

        family_algorithms = [algorithm for algorithm in track_algorithms if algorithm['family'] == selected_family]
        selected_algorithm_label = st.selectbox(
            "Algorithm",
            [algorithm['label'] for algorithm in family_algorithms],
            key="workspace_algorithm_label"
        )
        selected_algorithm = next(
            algorithm for algorithm in family_algorithms if algorithm['label'] == selected_algorithm_label
        )

    with config_col2:
        dataset = st.selectbox(
            "Dataset / source",
            dataset_options,
            key="workspace_dataset"
        )
        selected_dataset_meta = dataset_catalog[dataset]

        if dataset == "NOIZEUS Database" and selected_dataset_meta['available']:
            sample_options = [f"{sample['name']} ({sample['count']} samples)" for sample in noizeus_samples]
            selected_sample_idx = st.selectbox(
                "Noise condition",
                range(len(sample_options)),
                format_func=lambda index: sample_options[index],
                key="workspace_noise_condition"
            )
            selected_sample = noizeus_samples[selected_sample_idx]
            sample_number = st.selectbox(
                "Sample number",
                list(range(1, selected_sample['count'] + 1)),
                format_func=lambda value: f"Sample {value:02d}",
                key="workspace_sample_number"
            )
        elif dataset == "NOIZEUS Database":
            st.error("NOIZEUS dataset not found in assets/NOIZEUS/")
        elif dataset == "Custom Upload":
            st.info("Custom upload is a planned workflow for exploratory listening tests.")
        else:
            st.info("This dataset is documented as literature context for protocol comparison, not yet as a runnable local dataset.")

    st.caption(f"Purpose: {selected_algorithm['purpose']}")
    st.caption("Evaluation scope: matched-reference single-channel speech enhancement")
    if not selected_algorithm['implemented']:
        st.warning("This algorithm is documented in the app but not implemented yet.")

    run_processing = st.button("Run workspace test 🚀", key="workspace_run_processing")

    col_workspace_1, col_workspace_2, col_workspace_3 = st.columns(3)
    with col_workspace_1:
        st.metric("Research objective", "Speech enhancement")
    with col_workspace_2:
        st.metric("Algorithm family", selected_algorithm['family'])
    with col_workspace_3:
        st.metric("Benchmark dataset", "NOIZEUS-compatible")

    st.info(
        "Only compare algorithms when the objective, dataset, and clean-reference metrics are aligned. This workspace is therefore centered on the same speech-enhancement scope as the project benchmark."
    )

    if dataset == "NOIZEUS Database" and selected_sample:
        with st.spinner("Loading NOIZEUS sample..."):
            y_noisy, y_clean_ref, sr = load_noizeus_sample(selected_sample['noise_type'], sample_number)

        if y_noisy is None or y_clean_ref is None:
            st.error(f"Unable to load sample {sample_number} for {selected_sample['name']}")
            st.stop()

        st.success(f"Loaded sample: {selected_sample['name']} - Sample {sample_number:02d} (SR: {sr} Hz)")

        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("Noise type", selected_sample['noise_name'].upper())
        with col_info2:
            st.metric("Signal-to-noise ratio", selected_sample['snr'])
        with col_info3:
            st.metric("Sample", f"#{sample_number:02d}")
    else:
        sample_path = "assets/sample_audio/test_audio.wav"
        if os.path.exists(sample_path):
            y_noisy, sr = librosa.load(sample_path, sr=16000)
            y_clean_ref = y_noisy
            st.warning("Using fallback test audio because no benchmark-ready sample is selected.")
        else:
            y_noisy = None
            y_clean_ref = None
            sr = None
            st.error("No audio file found for the workspace demo.")

    st.markdown(f"### Current algorithm: {selected_algorithm['label']}")
    st.markdown(selected_algorithm['summary'])

    if y_noisy is not None and sr is not None:
        st.markdown("### Listen before / after")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Noisy signal (original)**")
            temp_noisy_path = "assets/sample_audio/temp_noisy.wav"
            sf.write(temp_noisy_path, y_noisy, sr)
            st.audio(temp_noisy_path)

        if run_processing:
            if not selected_algorithm['implemented']:
                st.warning("This algorithm is not implemented yet, so the workspace cannot process the sample.")
            else:
                with st.spinner("Processing signal..."):
                    y_enhanced = run_algorithm(selected_algorithm, y_noisy, sr)
                    temp_clean_path = "assets/sample_audio/temp_enhanced.wav"
                    sf.write(temp_clean_path, y_enhanced, sr)

                    with col2:
                        st.write(f"**Enhanced signal ({selected_algorithm['label']})**")
                        st.audio(temp_clean_path)

                    with st.expander("🔍 View algorithm source code (Python)"):
                        code_source = inspect.getsource(selected_algorithm['runner'])
                        st.code(code_source, language='python')

                    st.markdown("---")
                    st.markdown("### Sample-level metrics")

                    if y_clean_ref is not None:
                        metrics = calculate_metrics(y_noisy, y_clean_ref, y_enhanced, sr)
                        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)

                        with col_m1:
                            st.metric("Global SNR improvement", f"{metrics.get('snr_improvement', 0):+.2f} dB")
                        with col_m2:
                            st.metric("Segmental SNR improvement", f"{metrics.get('seg_snr_improvement', 0):+.2f} dB")
                        with col_m3:
                            st.metric("Perceptual Evaluation of Speech Quality (PESQ)", f"{metrics['pesq']:.3f}" if metrics.get('pesq') is not None else "N/A")
                        with col_m4:
                            st.metric("Short-Time Objective Intelligibility (STOI)", f"{metrics['stoi']:.3f}" if metrics.get('stoi') is not None else "N/A")
                        with col_m5:
                            st.metric("Mean squared error vs clean reference", f"{np.mean((y_enhanced - y_clean_ref) ** 2):.6f}")

                    st.markdown("### Signal visualization")
                    if y_clean_ref is not None:
                        fig = plot_signal_analysis(
                            y_noisy,
                            y_clean_ref,
                            y_enhanced,
                            sr,
                            f"Analysis: {selected_sample['name']} - Sample {sample_number:02d}" if selected_sample else selected_algorithm['label']
                        )
                    else:
                        fig, axes = plt.subplots(2, 2, figsize=(15, 8))
                        fig.suptitle("Signal analysis (no clean reference)", fontsize=16)
                        time = np.linspace(0, len(y_noisy) / sr, len(y_noisy))

                        axes[0, 0].plot(time, y_noisy, alpha=0.7)
                        axes[0, 0].set_title('Waveform - Original')
                        axes[0, 0].set_xlabel('Time (s)')
                        axes[0, 0].set_ylabel('Amplitude')
                        axes[0, 0].grid(True, alpha=0.3)

                        axes[0, 1].plot(time, y_enhanced, color='green')
                        axes[0, 1].set_title('Waveform - Result')
                        axes[0, 1].set_xlabel('Time (s)')
                        axes[0, 1].set_ylabel('Amplitude')
                        axes[0, 1].grid(True, alpha=0.3)

                        d_noisy = librosa.amplitude_to_db(np.abs(librosa.stft(y_noisy)), ref=np.max)
                        d_enhanced = librosa.amplitude_to_db(np.abs(librosa.stft(y_enhanced)), ref=np.max)
                        img1 = librosa.display.specshow(d_noisy, y_axis='hz', x_axis='time', sr=sr, ax=axes[1, 0], cmap='viridis')
                        axes[1, 0].set_title('Spectrogram - Original')
                        plt.colorbar(img1, ax=axes[1, 0], format='%+2.0f dB')

                        img2 = librosa.display.specshow(d_enhanced, y_axis='hz', x_axis='time', sr=sr, ax=axes[1, 1], cmap='viridis')
                        axes[1, 1].set_title('Spectrogram - Enhanced')
                        plt.colorbar(img2, ax=axes[1, 1], format='%+2.0f dB')
                        plt.tight_layout()

                    st.pyplot(fig)

                    st.markdown("### Additional statistics")
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.markdown("**Original signal**")
                        st.write(f"- RMS: {np.sqrt(np.mean(y_noisy**2)):.4f}")
                        st.write(f"- Max: {np.max(np.abs(y_noisy)):.4f}")
                    with col_stat2:
                        st.markdown("**Enhanced signal**")
                        st.write(f"- RMS: {np.sqrt(np.mean(y_enhanced**2)):.4f}")
                        st.write(f"- Max: {np.max(np.abs(y_enhanced)):.4f}")
                    with col_stat3:
                        st.markdown("**Comparison**")
                        if y_clean_ref is not None:
                            corr_orig = np.corrcoef(y_noisy, y_clean_ref)[0, 1]
                            corr_enh = np.corrcoef(y_enhanced, y_clean_ref)[0, 1]
                            st.write(f"- Original correlation: {corr_orig:.3f}")
                            st.write(f"- Enhanced correlation: {corr_enh:.3f}")
                            st.write(f"- Improvement: {corr_enh - corr_orig:+.3f}")
        else:
            st.info("Use the button above to run a single-sample workspace test.")

with tab_datasets:
    st.header("🗂️ Dataset Catalog")
    st.markdown("Datasets are described independently from experiments so the app can remain reusable while keeping the current benchmark scope clear.")

    dataset_rows = []
    for label, meta in dataset_catalog.items():
        dataset_rows.append({
            'Dataset': label,
            'Available': "Yes" if meta['available'] else "No",
            'Task scope': meta['task_scope'],
            'Benchmark-ready': "Yes" if meta['benchmark_ready'] else "No",
            'Description': meta['description']
        })
    st.dataframe(pd.DataFrame(dataset_rows), use_container_width=True)

    selected_dataset_name = st.selectbox("Inspect dataset details", list(dataset_catalog.keys()), key="dataset_catalog_select")
    dataset_details = dataset_catalog[selected_dataset_name]

    st.subheader(selected_dataset_name)
    st.markdown(dataset_details['description'])
    render_markdown_list(dataset_details['details'])
    if dataset_details['reference']:
        st.markdown(f"Reference: {dataset_details['reference']}")

    if selected_dataset_name == "NOIZEUS Database" and noizeus_samples:
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("Noise types", len({sample['noise_name'] for sample in noizeus_samples}))
        with col_stat2:
            st.metric("Signal-to-noise ratio levels", len({sample['snr'] for sample in noizeus_samples}))
        with col_stat3:
            st.metric("Total samples", sum(sample['count'] for sample in noizeus_samples))

        noise_types = {}
        for sample in noizeus_samples:
            noise_types.setdefault(sample['noise_name'], []).append(sample)

        st.markdown("### Browse NOIZEUS conditions")
        for noise_name, samples in noise_types.items():
            with st.expander(f"{noise_name.upper()} - {len(samples)} signal-to-noise ratio conditions"):
                cols = st.columns(len(samples))
                for index, sample in enumerate(samples):
                    with cols[index]:
                        st.markdown(f"**{sample['snr']}**")
                        st.write(f"{sample['count']} samples")
                        if st.button("Play sample 01", key=f"listen_{noise_name}_{sample['snr']}"):
                            y_ex, _, sr_ex = load_noizeus_sample(sample['noise_type'], 1)
                            if y_ex is not None:
                                temp_ex_path = f"assets/sample_audio/temp_example_{noise_name}_{sample['snr']}.wav"
                                sf.write(temp_ex_path, y_ex, sr_ex)
                                st.audio(temp_ex_path, format='audio/wav')

        st.markdown("### File structure")
        st.code("""
assets/NOIZEUS/
├── clean/                          # Clean reference signals
│   ├── sp01.wav to sp30.wav
├── airport_0dB/                    # Airport, SNR=0dB
│   ├── sp01_airport_sn0.wav
│   ├── sp02_airport_sn0.wav
│   └── ...
├── airport_5dB/                    # Airport, SNR=5dB
│   └── ...
├── babble_0dB/                     # Babble, SNR=0dB
│   └── ...
└── ... (7 other noise types)
        """)

with tab_benchmarks:
    st.header("📏 Benchmark Study")
    st.markdown("This section keeps the project benchmark, the protocol description, and the literature context clearly separated.")

    benchmark_overview_tab, benchmark_run_tab, benchmark_reference_tab = st.tabs([
        "Overview",
        "Run Benchmark",
        "Literature Context"
    ])

    with benchmark_overview_tab:
        for benchmark in benchmark_catalog:
            with st.expander(f"{benchmark['type']}: {benchmark['label']}", expanded=benchmark['type'] == "Project benchmark"):
                st.markdown(f"**Goal:** {benchmark['goal']}")
                st.markdown(f"**Scope:** {benchmark['scope']}")
                st.markdown(f"**Status:** {benchmark['status']}")
                st.markdown(f"**Datasets:** {', '.join(benchmark['datasets'])}")
                st.markdown(f"**Reference studies tracked:** {len(reference_studies_by_lane.get(benchmark['id'], []))}")
                st.markdown(f"**Comparison rule:** {benchmark['comparison_rule']}")
                render_markdown_list(benchmark['notes'])

    with benchmark_reference_tab:
        st.subheader("Literature context and protocol notes")
        st.markdown("Use this section to document external studies and benchmark protocols without mixing them with your computed project results.")

        reference_rows = []
        for benchmark in benchmark_catalog:
            if benchmark['type'] != "Project benchmark":
                reference_rows.append({
                    'Benchmark lane': benchmark['label'],
                    'Scope': benchmark['scope'],
                    'Status': benchmark['status'],
                    'Tracked studies': len(reference_studies_by_lane.get(benchmark['id'], [])),
                    'How to use it': benchmark['goal']
                })
        st.dataframe(pd.DataFrame(reference_rows), use_container_width=True)

        lane_options = [benchmark for benchmark in benchmark_catalog if benchmark['type'] != "Project benchmark"]
        selected_reference_lane_label = st.selectbox(
            "Reference lane",
            [benchmark['label'] for benchmark in lane_options],
            key="reference_lane_select"
        )
        selected_reference_lane = next(
            benchmark for benchmark in lane_options if benchmark['label'] == selected_reference_lane_label
        )
        selected_lane_studies = reference_studies_by_lane.get(selected_reference_lane['id'], [])

        st.markdown("### Lane protocol policy")
        st.markdown(selected_reference_lane['comparison_rule'])
    with benchmark_run_tab:
        st.subheader("Run your project benchmark")
        st.markdown("This runnable section is reserved for the benchmark that is directly comparable inside the app: single-channel speech enhancement on NOIZEUS.")

        enhancement_algorithms = [
            algorithm for algorithm in algorithm_catalog["Speech Enhancement"]
            if algorithm['implemented'] and algorithm['benchmark_group'] == "speech_enhancement"
        ]

        if not noizeus_samples:
            st.error("NOIZEUS dataset required for the current project benchmark.")
        else:
            available_analysis_noises = sorted({sample['noise_name'] for sample in noizeus_samples})
            available_analysis_snrs = sorted({sample['snr'] for sample in noizeus_samples})

            run_mode = st.radio(
                "Benchmark scope",
                ["Single condition", "Full dataset (all noise types × all SNR levels)"],
                horizontal=True,
                key="bench_mode"
            )

            if run_mode == "Single condition":
                col_sel1, col_sel2 = st.columns(2)
                with col_sel1:
                    analysis_noise = st.selectbox("Noise type", available_analysis_noises, key="analysis_noise")
                with col_sel2:
                    analysis_snr = st.selectbox("Signal-to-noise ratio level", available_analysis_snrs, key="analysis_snr")

                selected_analysis_sample = next(
                    (s for s in noizeus_samples if s['noise_name'] == analysis_noise and s['snr'] == analysis_snr),
                    None
                )

                if selected_analysis_sample and st.button("Run benchmark on this condition", key="run_analysis"):
                    with st.spinner("Analyzing all samples in this condition..."):
                        results = []
                        for sample_idx in range(1, selected_analysis_sample['count'] + 1):
                            y_noisy, y_clean, sr = load_noizeus_sample(selected_analysis_sample['noise_type'], sample_idx)
                            if y_noisy is None or y_clean is None:
                                continue
                            sample_results = {'Sample': sample_idx, 'noise': analysis_noise, 'snr': analysis_snr}
                            for algorithm in enhancement_algorithms:
                                try:
                                    y_enhanced = run_algorithm(algorithm, y_noisy, sr)
                                    sample_results[algorithm['label']] = calculate_metrics(y_noisy, y_clean, y_enhanced, sr)
                                except Exception as exc:
                                    sample_results[algorithm['label']] = {'error': str(exc)}
                            results.append(sample_results)

                        st.session_state['benchmark_results'] = results
                        st.session_state['benchmark_mode'] = 'single'
                        st.session_state['benchmark_noise'] = analysis_noise
                        st.session_state['benchmark_snr'] = analysis_snr
                        st.session_state['benchmark_algorithms'] = enhancement_algorithms
                        st.success(f"Benchmark completed on {len(results)} samples")

                elif 'benchmark_results' not in st.session_state:
                    st.info("Select a NOIZEUS condition, then run the benchmark.")

            else:
                total_utterances = sum(s['count'] for s in noizeus_samples)
                st.info(
                    f"Will process **{total_utterances} utterances** across "
                    f"**{len(available_analysis_noises)} noise types × {len(available_analysis_snrs)} SNR levels**. "
                    f"This takes several minutes."
                )
                if st.button("Run full dataset benchmark", key="run_full"):
                    results = []
                    progress = st.progress(0, text="Starting…")
                    total_conditions = len(noizeus_samples)
                    for cond_idx, sample_meta in enumerate(noizeus_samples):
                        progress.progress(
                            cond_idx / total_conditions,
                            text=f"Condition {cond_idx + 1}/{total_conditions}: {sample_meta['noise_name']} {sample_meta['snr']}"
                        )
                        for sample_idx in range(1, sample_meta['count'] + 1):
                            y_noisy, y_clean, sr = load_noizeus_sample(sample_meta['noise_type'], sample_idx)
                            if y_noisy is None or y_clean is None:
                                continue
                            sample_results = {
                                'Sample': sample_idx,
                                'noise': sample_meta['noise_name'],
                                'snr': sample_meta['snr'],
                            }
                            for algorithm in enhancement_algorithms:
                                try:
                                    y_enhanced = run_algorithm(algorithm, y_noisy, sr)
                                    sample_results[algorithm['label']] = calculate_metrics(y_noisy, y_clean, y_enhanced, sr)
                                except Exception as exc:
                                    sample_results[algorithm['label']] = {'error': str(exc)}
                            results.append(sample_results)
                    progress.progress(1.0, text="Done")

                    st.session_state['benchmark_results'] = results
                    st.session_state['benchmark_mode'] = 'full'
                    st.session_state['benchmark_algorithms'] = enhancement_algorithms
                    st.success(f"Full benchmark completed on {len(results)} utterances")

                elif 'benchmark_results' not in st.session_state:
                    st.info("Run the full dataset benchmark above to get a global algorithm ranking.")

            # ── Display cached results ──────────────────────────────────────────
            if 'benchmark_results' in st.session_state:
                results = st.session_state['benchmark_results']
                cached_algorithms = st.session_state['benchmark_algorithms']
                bench_mode = st.session_state.get('benchmark_mode', 'single')

                algo_labels = [a['label'] for a in cached_algorithms]
                family_lookup = {a['label']: a['family'] for a in cached_algorithms}
                metrics_names = ['snr_improvement', 'seg_snr_improvement', 'pesq', 'stoi']
                metric_display = {
                    'snr_improvement': "SNRi (dB)",
                    'seg_snr_improvement': "SegSNRi (dB)",
                    'pesq': "PESQ",
                    'stoi': "STOI",
                }

                if bench_mode == 'single':
                    cached_noise = st.session_state.get('benchmark_noise', '?')
                    cached_snr = st.session_state.get('benchmark_snr', '?')
                    st.caption(f"Results for **{cached_noise}** at **{cached_snr}** — switch to Full dataset or re-run to update.")
                else:
                    noises_covered = sorted({r['noise'] for r in results})
                    snrs_covered = sorted({r['snr'] for r in results})
                    st.caption(
                        f"Full dataset results — {len(noises_covered)} noise types × {len(snrs_covered)} SNR levels "
                        f"({len(results)} utterances total)."
                    )

                # ── Aggregate means table ──────────────────────────────────────
                agg_rows = []
                for algo in cached_algorithms:
                    lbl = algo['label']
                    row = {'Algorithm': lbl, 'Family': algo['family']}
                    for metric in metrics_names:
                        values = [
                            r[lbl].get(metric)
                            for r in results
                            if lbl in r and isinstance(r[lbl], dict) and r[lbl].get(metric) is not None
                        ]
                        row[metric_display[metric]] = round(float(np.mean(values)), 3) if values else None
                    agg_rows.append(row)

                agg_df = pd.DataFrame(agg_rows).set_index('Algorithm')

                if bench_mode == 'full':
                    # Composite score: normalise each metric to [0,1] then average
                    score_cols = list(metric_display.values())
                    normed = agg_df[score_cols].copy().astype(float)
                    for col in score_cols:
                        col_min, col_max = normed[col].min(), normed[col].max()
                        if col_max > col_min:
                            normed[col] = (normed[col] - col_min) / (col_max - col_min)
                        else:
                            normed[col] = 0.5
                    agg_df['Composite score'] = normed.mean(axis=1).round(3)
                    agg_df = agg_df.sort_values('Composite score', ascending=False)
                    agg_df.insert(0, 'Rank', range(1, len(agg_df) + 1))

                    st.markdown("### Global ranking")
                    st.markdown(
                        "Composite score = average of per-metric min-max normalised values across all conditions. "
                        "Higher is better. Rank 1 is the best overall algorithm."
                    )

                st.markdown("### Summary table" if bench_mode == 'single' else "### Per-algorithm aggregate (all conditions)")
                st.dataframe(agg_df, use_container_width=True)

                # ── Per-noise-type heatmap (full mode only) ───────────────────
                if bench_mode == 'full':
                    st.markdown("### PESQ by noise type — identify algorithm weak spots")
                    st.caption("Mean PESQ per noise type averaged across all SNR levels. Red = low quality, green = high quality.")
                    noise_types = sorted({r['noise'] for r in results})
                    heat_data = {}
                    for algo in cached_algorithms:
                        lbl = algo['label']
                        heat_data[lbl] = {}
                        for noise in noise_types:
                            vals = [
                                r[lbl].get('pesq')
                                for r in results
                                if r['noise'] == noise and lbl in r
                                   and isinstance(r[lbl], dict) and r[lbl].get('pesq') is not None
                            ]
                            heat_data[lbl][noise] = round(float(np.mean(vals)), 3) if vals else None
                    heat_df = pd.DataFrame(heat_data).T
                    heat_df.index.name = 'Algorithm'
                    st.dataframe(
                        heat_df.style.background_gradient(cmap='RdYlGn', axis=None, vmin=1.0, vmax=3.5),
                        use_container_width=True
                    )

                # ── Boxplots ───────────────────────────────────────────────────
                st.markdown("### Visual comparisons")
                fig, axes = plt.subplots(1, len(metric_display), figsize=(7 * len(metric_display), 8))

                if len(metric_display) == 1:
                    axes = [axes]

                for ax_idx, (metric_key, metric_label) in enumerate(metric_display.items()):
                    valid_labels = [
                        lbl for lbl in algo_labels
                        if any(
                            isinstance(r.get(lbl), dict) and r[lbl].get(metric_key) is not None
                            for r in results
                        )
                    ]
                    if not valid_labels:
                        axes[ax_idx].set_visible(False)
                        continue
                    data_per_algo = [
                        [r[lbl][metric_key] for r in results if isinstance(r.get(lbl), dict) and r[lbl].get(metric_key) is not None]
                        for lbl in valid_labels
                    ]
                    bp = axes[ax_idx].boxplot(data_per_algo, labels=valid_labels, patch_artist=True)
                    axes[ax_idx].set_title(metric_label)
                    axes[ax_idx].grid(True, alpha=0.3)
                    if metric_key == 'pesq':
                        axes[ax_idx].set_ylim(0, 4.5)
                        axes[ax_idx].set_ylabel('Score (0–4.5)')
                    elif metric_key == 'stoi':
                        axes[ax_idx].set_ylim(0, 1)
                        axes[ax_idx].set_ylabel('Score (0–1)')
                    else:
                        axes[ax_idx].set_ylabel('dB')
                    style_benchmark_axis(axes[ax_idx], valid_labels, family_lookup, bp)

                add_family_legend(fig, algo_labels, family_lookup)
                fig.subplots_adjust(bottom=0.42, top=0.82, wspace=0.3)
                st.caption("Boxplots grouped and colour-coded by algorithm family.")
                st.pyplot(fig)

                # ── CSV export ─────────────────────────────────────────────────
                export_rows = []
                for r in results:
                    base = {'Sample': r['Sample'], 'noise': r.get('noise', ''), 'snr': r.get('snr', '')}
                    for lbl in algo_labels:
                        if isinstance(r.get(lbl), dict):
                            for metric in metrics_names:
                                base[f"{lbl}_{metric}"] = r[lbl].get(metric, 'N/A')
                    export_rows.append(base)
                csv_data = pd.DataFrame(export_rows).to_csv(index=False)
                fname_suffix = "full" if bench_mode == 'full' else f"{st.session_state.get('benchmark_noise', 'cond')}_{st.session_state.get('benchmark_snr', '')}"
                st.download_button(
                    label="Download benchmark CSV",
                    data=csv_data,
                    file_name=f"benchmark_{fname_suffix}.csv",
                    mime="text/csv"
                )
