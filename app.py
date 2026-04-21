import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import inspect
import os
import shutil
import io
from pesq import pesq
import pystoi
import fitz

# Import algorithms
from src.algorithms.spectral import spectral_subtraction, spectral_subtraction_oversubtraction, multiband_spectral_subtraction
from src.algorithms.subspace import subspace_enhance

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

    # SNR before / afterter
    def calculate_snr(signal, noise):
        if len(signal) == 0 or len(noise) == 0:
            return 0.0
        signal_power = np.mean(signal**2)
        noise_power = np.mean(noise**2)
        if noise_power == 0:
            return float('inf') if signal_power > 0 else 0.0
        return 10 * np.log10(signal_power / noise_power)

    # Noisy signal SNR
    noise = y_original - y_clean
    snr_input = calculate_snr(y_clean, noise)

    # Enhanced signal SNR
    noise_enhanced = y_enhanced - y_clean
    snr_output = calculate_snr(y_clean, noise_enhanced)

    metrics['snr_improvement'] = snr_output - snr_input

    # PESQ (if available)
    try:
        # Resample to 16kHz for PESQ
        if sr != 16000:
            y_clean_16k = librosa.resample(y_clean, orig_sr=sr, target_sr=16000)
            y_enhanced_16k = librosa.resample(y_enhanced, orig_sr=sr, target_sr=16000)
        else:
            y_clean_16k = y_clean
            y_enhanced_16k = y_enhanced

        # Normalize
        y_clean_16k = y_clean_16k / np.max(np.abs(y_clean_16k)) if np.max(np.abs(y_clean_16k)) > 0 else y_clean_16k
        y_enhanced_16k = y_enhanced_16k / np.max(np.abs(y_enhanced_16k)) if np.max(np.abs(y_enhanced_16k)) > 0 else y_enhanced_16k

        pesq_score = pesq(16000, y_clean_16k, y_enhanced_16k, 'wb')
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

st.set_page_config(page_title="Speech Enhancement Research", layout="wide")

st.title("🎙️ Speech Enhancement & ANC Dashboard")
st.markdown("This dashboard presents the algorithms tested in my research project (ECE597 2025-2026). - Cédric JOVENET")

# --- SIDEBAR: Data and algorithm selection ---
st.sidebar.header("Configuration")

# Retrieve available NOIZEUS samples
noizeus_samples = get_noizeus_samples()

if noizeus_samples:
    # Dataset selection simulation (NOIZEUS)
    dataset = st.sidebar.selectbox("Select dataset", ["NOIZEUS Database", "Custom Upload"])

    if dataset == "NOIZEUS Database":
        sample_options = [f"{s['name']} ({s['count']} samples)" for s in noizeus_samples]
        selected_sample_idx = st.sidebar.selectbox("Noise condition", range(len(sample_options)), format_func=lambda x: sample_options[x])

        selected_sample = noizeus_samples[selected_sample_idx]
        sample_number = st.sidebar.selectbox("Sample number", list(range(1, selected_sample['count'] + 1)), format_func=lambda x: f"Sample {x:02d}")
    else:
        st.sidebar.warning("Custom upload not implemented yet.")
        selected_sample = None
        sample_number = 1
else:
    st.sidebar.error("NOIZEUS dataset not found in assets/NOIZEUS/")
    dataset = "Custom Upload"
    selected_sample = None
    sample_number = 1

# Algorithm menu
algo_family = st.sidebar.selectbox("Algorithm family", ["Spectral Subtraction", "Subspace Enhancement", "Active Noise Control (LMS)"])

if algo_family == "Spectral Subtraction":
    algo_choice = st.sidebar.selectbox("Specific algorithm", [
        "Spectral Subtraction (Classic)",
        "Spectral Subtraction (Over-subtraction)",
        "Multi-band Spectral Subtraction"
    ])
elif algo_family == "Subspace Enhancement":
    algo_choice = "Subspace Enhancement (SVD)"
else:
    algo_choice = st.sidebar.selectbox("Specific algorithm", ["Feedback LMS", "Hybrid LMS", "FxLMS"])
    st.sidebar.warning("LMS algorithms are under development.")

if st.sidebar.button("Run processing 🚀"):
    run_processing = True
else:
    run_processing = False

# --- MAIN TABS ---
tab_demo, tab_dataset, tab_analysis, tab_definitions = st.tabs(["🎧 Demo", "📂 Datasets (NOIZEUS)", "📊 Analysis & Metrics", "📘 Definitions & Personal Analysis"])

with tab_dataset:
    st.header("📂 Dataset: NOIZEUS")

    if noizeus_samples:
        st.markdown("""
        The **NOIZEUS** corpus is a benchmark dataset for single-channel speech enhancement.

        **Detailed composition (as used in this project):**
        - 30 clean IEEE sentences (`sp01.wav` to `sp30.wav`)
        - 8 real-world noise environments: airport, babble, car, exhibition, restaurant, station, street, train
        - 4 SNR levels per noise: 0 dB, 5 dB, 10 dB, 15 dB
        - 30 noisy files per condition, aligned with clean references
        - File naming convention: `spXX_<noise>_sn<snr>.wav` (example: `sp01_airport_sn0.wav`)

        This creates a controlled testbed to compare algorithms under identical speech content
        with progressively different noise difficulty.

        **Reference link (official source):**
        http://ecs.utdallas.edu/loizou/speech/noizeus/
        """)

        # General statistics
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("Noise types", len(set([s['noise_name'] for s in noizeus_samples])))
        with col_stat2:
            st.metric("SNR levels", len(set([s['snr'] for s in noizeus_samples])))
        with col_stat3:
            st.metric("Total samples", sum([s['count'] for s in noizeus_samples]))

        # Detailed sample list
        st.markdown("### Available samples")
        st.markdown("Each condition contains 30 audio samples (sp01.wav to sp30.wav)")

        # Group by noise type
        noise_types = {}
        for sample in noizeus_samples:
            noise = sample['noise_name']
            if noise not in noise_types:
                noise_types[noise] = []
            noise_types[noise].append(sample)

        for noise_name, samples in noise_types.items():
            with st.expander(f"🔊 {noise_name.upper()} - {len(samples)} SNR conditions"):
                cols = st.columns(len(samples))
                for i, sample in enumerate(samples):
                    with cols[i]:
                        st.markdown(f"**{sample['snr']}**")
                        st.write(f"{sample['count']} samples")
                        if st.button(f"Play sample 01", key=f"listen_{noise_name}_{sample['snr']}"):
                            y_ex, y_clean_ex, sr_ex = load_noizeus_sample(sample['noise_type'], 1)
                            if y_ex is not None:
                                temp_ex_path = f"assets/sample_audio/temp_example_{noise_name}_{sample['snr']}.wav"
                                sf.write(temp_ex_path, y_ex, sr_ex)
                                st.audio(temp_ex_path, format='audio/wav')

        # File structure information
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

    else:
        st.error("NOIZEUS dataset not found. Make sure files are extracted to assets/NOIZEUS/")

        # User instructions
        st.markdown("### Installation instructions:")
        st.markdown("""
        1. Download the NOIZEUS dataset from the official source
        2. Place the .zip files in `assets/NOIZEUS/`
        3. The files will be extracted and organized automatically
        """)

with tab_demo:
    st.subheader(f"Current test: {algo_choice}")

    # Load the selected sample
    if dataset == "NOIZEUS Database" and selected_sample:
        with st.spinner("Loading NOIZEUS sample..."):
            y_noisy, y_clean_ref, sr = load_noizeus_sample(selected_sample['noise_type'], sample_number)

        if y_noisy is None or y_clean_ref is None:
            st.error(f"Unable to load sample {sample_number} for {selected_sample['name']}")
            st.stop()

        st.success(f"✅ Loaded sample: {selected_sample['name']} - Sample {sample_number:02d} (SR: {sr} Hz)")

        # Display sample metadata
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("Noise type", selected_sample['noise_name'].upper())
        with col_info2:
            st.metric("SNR", f"{selected_sample['snr']}")
        with col_info3:
            st.metric("Sample", f"#{sample_number:02d}")

    else:
        # Fallback to the test sample
        sample_path = "assets/sample_audio/test_audio.wav"
        if os.path.exists(sample_path):
            y_noisy, sr = librosa.load(sample_path, sr=16000)
            y_clean_ref = y_noisy  # No clean reference available
            st.warning("⚠️ Using test sample (no clean reference available)")
        else:
            st.error("No audio file found!")
            st.stop()

    st.markdown("### Listen before / after")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Noisy signal (original)**")
        temp_noisy_path = "assets/sample_audio/temp_noisy.wav"
        sf.write(temp_noisy_path, y_noisy, sr)
        st.audio(temp_noisy_path)

    # Start processing
    if run_processing:
        with st.spinner("Processing signal..."):

            # Map chosen algorithm to the Python function
            if algo_choice == "Spectral Subtraction (Classic)":
                func = spectral_subtraction
                y_enhanced = func(y_noisy, sr)
            elif algo_choice == "Spectral Subtraction (Over-subtraction)":
                func = spectral_subtraction_oversubtraction
                y_enhanced = func(y_noisy, sr)
            elif algo_choice == "Multi-band Spectral Subtraction":
                func = multiband_spectral_subtraction
                y_enhanced = func(y_noisy, sr)
            elif algo_choice == "Subspace Enhancement (SVD)":
                func = subspace_enhance
                y_enhanced = func(y_noisy)
            else:
                st.warning("Algorithm not implemented for playback.")
                st.stop()

            temp_clean_path = "assets/sample_audio/temp_enhanced.wav"
            sf.write(temp_clean_path, y_enhanced, sr)

            with col2:
                st.write(f"**Enhanced signal ({algo_choice})**")
                st.audio(temp_clean_path)

            # --- DYNAMIC SOURCE CODE DISPLAY ---
            with st.expander("🔍 View algorithm source code (Python)"):
                code_source = inspect.getsource(func)
                st.code(code_source, language='python')

            st.markdown("---")
            st.markdown("### 📊 Full analysis")

            # Metric calculation
            if 'y_clean_ref' in locals() and y_clean_ref is not None:
                metrics = calculate_metrics(y_noisy, y_clean_ref, y_enhanced, sr)

                st.markdown("#### Objective metrics")
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)

                with col_m1:
                    if 'snr_improvement' in metrics:
                        st.metric(
                            label="SNR improvement",
                            value=f"{metrics['snr_improvement']:+.2f} dB",
                            delta="vs noisy signal"
                        )

                with col_m2:
                    if metrics.get('pesq') is not None:
                        st.metric(
                            label="PESQ score",
                            value=f"{metrics['pesq']:.3f}",
                            delta="Perceptual quality"
                        )
                    else:
                        st.metric(label="PESQ score", value="N/A")

                with col_m3:
                    if metrics.get('stoi') is not None:
                        st.metric(
                            label="STOI score",
                            value=f"{metrics['stoi']:.3f}",
                            delta="Intelligibility"
                        )
                    else:
                        st.metric(label="STOI score", value="N/A")

                with col_m4:
                    mse = np.mean((y_enhanced - y_clean_ref) ** 2)
                    st.metric(
                        label="MSE vs clean",
                        value=f"{mse:.6f}",
                        delta="Mean squared error"
                    )

            # Visual signal analysis
            st.markdown("#### Signal visualization")
            if 'y_clean_ref' in locals() and y_clean_ref is not None:
                fig = plot_signal_analysis(y_noisy, y_clean_ref, y_enhanced, sr,
                                         f"Analysis: {selected_sample['name']} - Sample {sample_number:02d}")
            else:
                # Simplified analysis without clean reference
                fig, axes = plt.subplots(2, 2, figsize=(15, 8))
                fig.suptitle("Signal analysis (no clean reference)", fontsize=16)

                time = np.linspace(0, len(y_noisy)/sr, len(y_noisy))

                # Waveform
                axes[0,0].plot(time, y_noisy, alpha=0.7, label='Noisy')
                axes[0,0].set_title('Waveform - Original')
                axes[0,0].set_xlabel('Time (s)')
                axes[0,0].set_ylabel('Amplitude')
                axes[0,0].grid(True, alpha=0.3)

                axes[0,1].plot(time, y_enhanced, color='green', label='Enhanced')
                axes[0,1].set_title('Waveform - Result')
                axes[0,1].set_xlabel('Time (s)')
                axes[0,1].set_ylabel('Amplitude')
                axes[0,1].grid(True, alpha=0.3)

                # Spectrograms
                D_noisy = librosa.amplitude_to_db(np.abs(librosa.stft(y_noisy)), ref=np.max)
                D_enhanced = librosa.amplitude_to_db(np.abs(librosa.stft(y_enhanced)), ref=np.max)

                img1 = librosa.display.specshow(D_noisy, y_axis='hz', x_axis='time', sr=sr, ax=axes[1,0], cmap='viridis')
                axes[1,0].set_title('Spectrogram - Original')
                plt.colorbar(img1, ax=axes[1,0], format='%+2.0f dB')

                img2 = librosa.display.specshow(D_enhanced, y_axis='hz', x_axis='time', sr=sr, ax=axes[1,1], cmap='viridis')
                axes[1,1].set_title('Spectrogram - Enhanced')
                plt.colorbar(img2, ax=axes[1,1], format='%+2.0f dB')

                plt.tight_layout()

            st.pyplot(fig)

            # Additional statistics
            st.markdown("#### Additional statistics")
            col_stat1, col_stat2, col_stat3 = st.columns(3)

            with col_stat1:
                st.markdown("**Original signal:**")
                st.write(f"- RMS: {np.sqrt(np.mean(y_noisy**2)):.4f}")
                st.write(f"- Max: {np.max(np.abs(y_noisy)):.4f}")
                st.write(f"- Estimated SNR: {10*np.log10(np.mean(y_noisy**2)/np.var(y_noisy - y_clean_ref)) if 'y_clean_ref' in locals() and y_clean_ref is not None else 'N/A':.1f} dB")

            with col_stat2:
                st.markdown("**Enhanced signal:**")
                st.write(f"- RMS: {np.sqrt(np.mean(y_enhanced**2)):.4f}")
                st.write(f"- Max: {np.max(np.abs(y_enhanced)):.4f}")
                st.write(f"- Estimated SNR: {10*np.log10(np.mean(y_enhanced**2)/np.var(y_enhanced - y_clean_ref)) if 'y_clean_ref' in locals() and y_clean_ref is not None else 'N/A':.1f} dB")

            with col_stat3:
                st.markdown("**Comparison:**")
                if 'y_clean_ref' in locals() and y_clean_ref is not None:
                    corr_orig = np.corrcoef(y_noisy, y_clean_ref)[0,1]
                    corr_enh = np.corrcoef(y_enhanced, y_clean_ref)[0,1]
                    st.write(f"- Original correlation: {corr_orig:.3f}")
                    st.write(f"- Enhanced correlation: {corr_enh:.3f}")
                    st.write(f"- Improvement: {corr_enh - corr_orig:+.3f}")

    else:
        st.info("👈 Select an algorithm and click 'Run processing' in the sidebar.")

with tab_analysis:
    st.header("📊 Advanced Analysis & Metrics")

    if not noizeus_samples:
        st.error("NOIZEUS dataset required for advanced analysis")
        st.stop()

    st.markdown("""
    This section provides detailed analysis and objective metrics to evaluate
    the performance of noise reduction algorithms.
    """)

    # Analysis parameter selection
    col_sel1, col_sel2 = st.columns(2)
    available_analysis_noises = sorted({sample['noise_name'] for sample in noizeus_samples})

    with col_sel1:
        analysis_noise = st.selectbox(
            "Noise type for analysis",
            available_analysis_noises,
            key="analysis_noise"
        )

    with col_sel2:
        analysis_snr = st.selectbox(
            "SNR level",
            ["0dB", "5dB", "10dB", "15dB"],
            key="analysis_snr"
        )

    # Find the matching sample
    selected_analysis_sample = None
    for s in noizeus_samples:
        if s['noise_name'] == analysis_noise and s['snr'] == analysis_snr:
            selected_analysis_sample = s
            break

    if selected_analysis_sample and st.button("🚀 Run full analysis", key="run_analysis"):

        with st.spinner("Analyzing all samples in this condition..."):

            # Analyze all samples in this condition
            results = []
            for sample_idx in range(1, selected_analysis_sample['count'] + 1):

                # Load signals
                y_noisy, y_clean, sr = load_noizeus_sample(selected_analysis_sample['noise_type'], sample_idx)
                if y_noisy is None or y_clean is None:
                    continue

                # Apply all available algorithms
                algorithms = {
                    "Spectral Subtraction": spectral_subtraction,
                    "Over-subtraction": spectral_subtraction_oversubtraction,
                    "Multi-band": multiband_spectral_subtraction,
                    "Subspace": subspace_enhance
                }

                sample_results = {'sample': sample_idx}
                for algo_name, algo_func in algorithms.items():
                    try:
                        if algo_name == "Subspace":
                            y_enhanced = algo_func(y_noisy)
                        else:
                            y_enhanced = algo_func(y_noisy, sr)

                        # Compute metrics
                        metrics = calculate_metrics(y_noisy, y_clean, y_enhanced, sr)
                        sample_results[algo_name] = metrics

                    except Exception as e:
                        sample_results[algo_name] = {'error': str(e)}

                results.append(sample_results)

            # Display results
            st.success(f"✅ Analysis completed on {len(results)} samples")

            # Performance summary
            st.markdown("### 📈 Performance summary")

            # Compute averages
            algo_names = list(algorithms.keys())
            metrics_names = ['snr_improvement', 'pesq', 'stoi']

            summary_data = {}
            for algo in algo_names:
                summary_data[algo] = {}
                for metric in metrics_names:
                    values = [r[algo].get(metric) for r in results if algo in r and metric in r[algo] and r[algo][metric] is not None]
                    if values:
                        summary_data[algo][metric] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'min': np.min(values),
                            'max': np.max(values)
                        }

            # Show metrics as a table
            import pandas as pd

            summary_df = pd.DataFrame(index=algo_names, columns=[f"{m.upper()} (mean ± std)" for m in metrics_names])

            for algo in algo_names:
                for metric in metrics_names:
                    if metric in summary_data[algo]:
                        val = summary_data[algo][metric]
                        summary_df.loc[algo, f"{metric.upper()} (mean ± std)"] = f"{val['mean']:.3f} ± {val['std']:.3f}"

            st.dataframe(summary_df)

            # Comparison plots
            st.markdown("### Visual comparisons")

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # SNR Improvement
            snr_data = {algo: [r[algo].get('snr_improvement', 0) for r in results if algo in r] for algo in algo_names}
            axes[0].boxplot([snr_data[algo] for algo in algo_names], labels=algo_names)
            axes[0].set_title('SNR Improvement (dB)')
            axes[0].set_ylabel('dB')
            axes[0].grid(True, alpha=0.3)

            # PESQ Scores
            pesq_data = {algo: [r[algo].get('pesq', 0) for r in results if algo in r and r[algo].get('pesq') is not None] for algo in algo_names}
            if any(pesq_data.values()):
                axes[1].boxplot([pesq_data[algo] for algo in algo_names if pesq_data[algo]], labels=[algo for algo in algo_names if pesq_data[algo]])
                axes[1].set_title('PESQ Scores')
                axes[1].set_ylabel('Score (0-4.5)')
                axes[1].set_ylim(0, 4.5)
                axes[1].grid(True, alpha=0.3)

            # STOI Scores
            stoi_data = {algo: [r[algo].get('stoi', 0) for r in results if algo in r and r[algo].get('stoi') is not None] for algo in algo_names}
            if any(stoi_data.values()):
                axes[2].boxplot([stoi_data[algo] for algo in algo_names if stoi_data[algo]], labels=[algo for algo in algo_names if stoi_data[algo]])
                axes[2].set_title('STOI Scores')
                axes[2].set_ylabel('Score (0-1)')
                axes[2].set_ylim(0, 1)
                axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)

            # Export results
            if st.button("📥 Export results (CSV)"):
                # Convert to DataFrame for export
                export_data = []
                for r in results:
                    row = {'Sample': r['sample']}
                    for algo in algo_names:
                        if algo in r:
                            for metric in metrics_names:
                                val = r[algo].get(metric)
                                row[f"{algo}_{metric}"] = val if val is not None else 'N/A'
                    export_data.append(row)

                export_df = pd.DataFrame(export_data)
                csv_data = export_df.to_csv(index=False)

                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"analysis_{analysis_noise}_{analysis_snr}.csv",
                    mime="text/csv"
                )

    else:
        st.info("👈 Select a noise type and SNR level, then click 'Run full analysis'")

with tab_definitions:
    st.header("📘 Definitions and Personal Analysis")
    st.markdown("""
    This section defines the main technical terms used in the dashboard and collects literature references.
    You can add your own personal comments and analysis below each definition.
    """)

    st.markdown("### Quick term links")
    st.markdown("[PESQ](#pesq) | [STOI](#stoi) | [SNR](#snr) | [Spectral Subtraction](#spectral-subtraction) | [Subspace Enhancement](#subspace-enhancement) | [ANC](#anc) | [NOIZEUS](#noizeus)")

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



    st.markdown("<a id='pesq'></a>", unsafe_allow_html=True)
    st.markdown("### PESQ")
    st.markdown("PESQ (Perceptual Evaluation of Speech Quality) is an objective metric that estimates perceived speech quality on a scale typically between 1.0 and 4.5. It is widely used to evaluate speech enhancement algorithms.")
    st.markdown("**References:** [ITU-T P.862](https://www.itu.int/rec/T-REC-P.862/en); [GitHub repository of the Python library](https://github.com/ludlows/PESQ/tree/v0.0.4)")
    st.markdown("**Personal comments:** In this project, PESQ is calculated using the `pesq` Python package, which implements the ITU-T standard. The [GitHub repository](https://github.com/ludlows/PESQ/tree/v0.0.4) provides the implementation used." \
    " The PESQ score is basically calculated using a mathematical model that simulates human hearing to compare the reference and degraded signals and return a standardized perceptual quality score.")

    st.markdown("<a id='stoi'></a>", unsafe_allow_html=True)
    st.markdown("### STOI")
    st.markdown("STOI (Short-Time Objective Intelligibility) estimates speech intelligibility by comparing clean and processed signals. It ranges from 0 to 1, with higher values indicating better intelligibility.")
    st.markdown("""
    **References:**
    1. C. H. Taal, R. C. Hendriks, R. Heusdens, J. Jensen, *A Short-Time Objective Intelligibility Measure for Time-Frequency Weighted Noisy Speech*, ICASSP 2010. [DOI:10.1109/ICASSP.2010.5495701](https://doi.org/10.1109/ICASSP.2010.5495701)
    2. C. H. Taal, R. C. Hendriks, R. Heusdens, J. Jensen, *An Algorithm for Intelligibility Prediction of Time-Frequency Weighted Noisy Speech*, IEEE Transactions on Audio, Speech, and Language Processing, 2011. [DOI:10.1109/TASL.2011.2114881](https://doi.org/10.1109/TASL.2011.2114881)
    3. J. Jensen, C. H. Taal, *An Algorithm for Predicting the Intelligibility of Speech Masked by Modulated Noise Maskers*, IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2016. [DOI:10.1109/TASLP.2016.2585878](https://doi.org/10.1109/TASLP.2016.2585878)
    """)
    st.markdown("**Personal comments:** In this project, STOI is calculated using the `pystoi` Python package, which provides an objective measure of speech intelligibility.")

    st.markdown("<a id='snr'></a>", unsafe_allow_html=True)
    st.markdown("### SNR")
    st.markdown("SNR (Signal-to-Noise Ratio) measures the ratio between clean signal power and noise power. It is often reported in decibels (dB) and is a primary objective measure for noise reduction performance.")
    st.markdown(r"**Equation:** $\mathrm{SNR}_{\mathrm{dB}} = 10\log_{10}\left(\frac{P_{\mathrm{signal}}}{P_{\mathrm{noise}}}\right)$")
    st.markdown(r"For discrete-time signals: $\mathrm{SNR}_{\mathrm{dB}} = 10\log_{10}\left(\frac{\sum_n s[n]^2}{\sum_n (x[n]-s[n])^2}\right)$, where $x[n]$ is noisy speech and $s[n]$ is clean speech.")
    st.markdown("**References:** [SNR on wikipedia](https://en.wikipedia.org/wiki/Signal-to-noise_ratio)")
    st.markdown("**Personal comments:** ")

    st.markdown("<a id='spectral-subtraction'></a>", unsafe_allow_html=True)
    st.markdown("### Spectral Subtraction")
    st.markdown("Spectral subtraction is a classic speech enhancement technique that estimates noise from non-speech segments and subtracts it from the noisy speech spectrum. Variants include over-subtraction and multi-band subtraction.")
    st.markdown("**References:** [Boll, 1979](https://ieeexplore.ieee.org/document/1163209); P. C. Loizou, *Speech Enhancement: Theory and Practice*, 2nd ed., CRC Press, 2013.")
    st.markdown("**Personal comments:**")

    if spectral_subtraction_presentation_pdf_url:
        st.markdown(
            f'Presentation PDF: <a href="{spectral_subtraction_presentation_pdf_url}" download="spectral_subtraction_presentation.pdf">Download</a>',
            unsafe_allow_html=True
        )

        if st.checkbox("Show PDF preview (Spectral Subtraction)", key="preview_spectral_subtraction_pdf"):
            if not render_pdf_preview(spectral_subtraction_presentation_pdf_source):
                st.warning("Unable to generate the Spectral Subtraction PDF preview.")
    else:
        st.info("The Spectral Subtraction presentation PDF was not found in assets/pdf/presentations/.")

    st.markdown("<a id='subspace-enhancement'></a>", unsafe_allow_html=True)
    st.markdown("### Subspace Enhancement")
    st.markdown("Subspace enhancement uses matrix decompositions to separate speech and noise subspaces. The algorithm then reconstructs the signal from the estimated speech subspace, improving noise suppression.")
    st.markdown("**References:** [Huang and Huang, 2001](https://ieeexplore.ieee.org/document/969612); P. C. Loizou, *Speech Enhancement: Theory and Practice*, 2nd ed., CRC Press, 2013.")
    st.markdown("**Personal comments:**")

    st.markdown("<a id='anc'></a>", unsafe_allow_html=True)
    st.markdown("### ANC")
    st.markdown("Active Noise Control (ANC) uses an anti-noise signal generated by a filter to cancel unwanted noise. It is commonly used in headphones and acoustic systems.")
    st.markdown("**References:** [Widrow et al., 1975](https://ieeexplore.ieee.org/document/1163658)")
    st.markdown("**Personal comments:** Add your notes here.")

    st.markdown("<a id='noizeus'></a>", unsafe_allow_html=True)
    st.markdown("### NOIZEUS")
    st.markdown("NOIZEUS is a speech corpus with noisy recordings at multiple SNR levels, commonly used to benchmark speech enhancement algorithms. It contains clean references and several real-world noise conditions.")
    st.markdown("**References:** [NOIZEUS dataset information](https://ecs.utdallas.edu/loizou/speech/noizeus/) ")
    st.markdown("**Personal comments:** See the dataset tab for more details on the NOIZEUS corpus, including its structure.")
