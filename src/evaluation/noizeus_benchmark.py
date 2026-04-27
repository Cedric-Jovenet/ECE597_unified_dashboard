import argparse
import os

import librosa
import numpy as np
import pandas as pd
import pystoi
from pesq import pesq

from src.catalogs import get_algorithm_catalog


def get_noizeus_conditions(base_path="assets/NOIZEUS"):
    conditions = []
    if not os.path.exists(base_path):
        return conditions

    for condition in sorted(os.listdir(base_path)):
        condition_path = os.path.join(base_path, condition)
        if not os.path.isdir(condition_path) or condition == "clean" or condition.endswith(".zip"):
            continue

        parts = condition.split("_")
        if len(parts) < 2:
            continue

        wav_files = [file_name for file_name in os.listdir(condition_path) if file_name.endswith(".wav")]
        conditions.append(
            {
                "noise_type": condition,
                "noise_name": parts[0],
                "snr": parts[1],
                "count": len(wav_files),
            }
        )

    return conditions


def load_noizeus_sample(noise_type, sample_idx=1, base_path="assets/NOIZEUS"):
    noisy_file = f"sp{sample_idx:02d}_{noise_type.split('_')[0]}_sn{noise_type.split('_')[1][:-2]}.wav"
    noisy_path = os.path.join(base_path, noise_type, noisy_file)

    clean_file = f"sp{sample_idx:02d}.wav"
    clean_path = os.path.join(base_path, "clean", clean_file)

    if not os.path.exists(noisy_path) or not os.path.exists(clean_path):
        return None, None, None

    y_noisy, sr_noisy = librosa.load(noisy_path, sr=None)
    y_clean, sr_clean = librosa.load(clean_path, sr=None)

    if sr_noisy != sr_clean:
        y_clean = librosa.resample(y_clean, orig_sr=sr_clean, target_sr=sr_noisy)
        sr_clean = sr_noisy

    return y_noisy, y_clean, sr_noisy


def calculate_metrics(y_original, y_clean, y_enhanced, sr):
    min_length = min(len(y_original), len(y_clean), len(y_enhanced))
    y_original = y_original[:min_length]
    y_clean = y_clean[:min_length]
    y_enhanced = y_enhanced[:min_length]

    def calculate_snr(signal, noise):
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        if noise_power == 0:
            return float("inf") if signal_power > 0 else 0.0
        return 10 * np.log10(signal_power / noise_power)

    def calculate_segmental_snr(clean_signal, noisy_signal, frame_len, hop_len):
        if frame_len <= 0 or hop_len <= 0 or len(clean_signal) < frame_len or len(noisy_signal) < frame_len:
            return 0.0

        frame_scores = []
        for start in range(0, len(clean_signal) - frame_len + 1, hop_len):
            clean_frame = clean_signal[start:start + frame_len]
            noisy_frame = noisy_signal[start:start + frame_len]
            noise_frame = noisy_frame - clean_frame

            clean_energy = np.sum(clean_frame ** 2)
            noise_energy = np.sum(noise_frame ** 2)
            frame_snr = 10 * np.log10((clean_energy + 1e-12) / (noise_energy + 1e-12))
            frame_scores.append(np.clip(frame_snr, -10.0, 35.0))

        if not frame_scores:
            return 0.0
        return float(np.mean(frame_scores))

    metrics = {}
    noise = y_original - y_clean
    noise_enhanced = y_enhanced - y_clean
    metrics["snr_improvement"] = calculate_snr(y_clean, noise_enhanced) - calculate_snr(y_clean, noise)

    frame_len = max(1, int(0.02 * sr))
    hop_len = max(1, frame_len // 2)
    seg_snr_input = calculate_segmental_snr(y_clean, y_original, frame_len, hop_len)
    seg_snr_output = calculate_segmental_snr(y_clean, y_enhanced, frame_len, hop_len)
    metrics["seg_snr_improvement"] = seg_snr_output - seg_snr_input

    try:
        if sr == 8000:
            pesq_sr = 8000
            pesq_mode = "nb"
            y_clean_pesq = y_clean
            y_enhanced_pesq = y_enhanced
        else:
            pesq_sr = 16000
            pesq_mode = "wb"
            if sr != 16000:
                y_clean_pesq = librosa.resample(y_clean, orig_sr=sr, target_sr=16000)
                y_enhanced_pesq = librosa.resample(y_enhanced, orig_sr=sr, target_sr=16000)
            else:
                y_clean_pesq = y_clean
                y_enhanced_pesq = y_enhanced

        if np.max(np.abs(y_clean_pesq)) > 0:
            y_clean_pesq = y_clean_pesq / np.max(np.abs(y_clean_pesq))
        if np.max(np.abs(y_enhanced_pesq)) > 0:
            y_enhanced_pesq = y_enhanced_pesq / np.max(np.abs(y_enhanced_pesq))

        metrics["pesq"] = pesq(pesq_sr, y_clean_pesq, y_enhanced_pesq, pesq_mode)
    except Exception:
        metrics["pesq"] = None

    try:
        metrics["stoi"] = pystoi.stoi(y_clean, y_enhanced, sr)
    except Exception:
        metrics["stoi"] = None

    return metrics


def get_project_algorithms():
    return [
        algorithm
        for algorithm in get_algorithm_catalog()["Speech Enhancement"]
        if algorithm["implemented"]
    ]


def run_noizeus_benchmark(noise_name=None, snr=None, sample_limit=None):
    records = []
    conditions = get_noizeus_conditions()
    algorithms = get_project_algorithms()

    for condition in conditions:
        if noise_name and condition["noise_name"].lower() != noise_name.lower():
            continue
        if snr and condition["snr"].lower() != snr.lower():
            continue

        condition_count = condition["count"] if sample_limit is None else min(condition["count"], sample_limit)
        for sample_idx in range(1, condition_count + 1):
            y_noisy, y_clean, sr_value = load_noizeus_sample(condition["noise_type"], sample_idx)
            if y_noisy is None or y_clean is None:
                continue

            for algorithm in algorithms:
                if algorithm["needs_sr"]:
                    y_enhanced = algorithm["runner"](y_noisy, sr_value)
                else:
                    y_enhanced = algorithm["runner"](y_noisy)

                metrics = calculate_metrics(y_noisy, y_clean, y_enhanced, sr_value)
                records.append(
                    {
                        "noise": condition["noise_name"],
                        "snr": condition["snr"],
                        "sample": sample_idx,
                        "algorithm": algorithm["label"],
                        **metrics,
                    }
                )

    return pd.DataFrame(records)


def summarize_results(results_df):
    if results_df.empty:
        return pd.DataFrame()

    summary = results_df.groupby("algorithm")[["snr_improvement", "seg_snr_improvement", "pesq", "stoi"]].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def main():
    parser = argparse.ArgumentParser(description="Run the project NOIZEUS benchmark outside Streamlit.")
    parser.add_argument("--noise", default=None, help="Noise type to benchmark, for example babble or airport.")
    parser.add_argument("--snr", default=None, help="Signal-to-noise ratio condition, for example 5dB.")
    parser.add_argument("--sample-limit", type=int, default=None, help="Optional cap on the number of samples per condition.")
    parser.add_argument("--output-dir", default=None, help="Optional directory where raw and summary CSV files will be written.")
    args = parser.parse_args()

    results_df = run_noizeus_benchmark(noise_name=args.noise, snr=args.snr, sample_limit=args.sample_limit)
    summary_df = summarize_results(results_df)

    if results_df.empty:
        print("No benchmark rows were produced. Check dataset availability and filter arguments.")
        return

    print(summary_df.to_string(index=False))

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        suffix_parts = [part for part in [args.noise, args.snr] if part]
        suffix = "_".join(suffix_parts) if suffix_parts else "all_conditions"
        raw_path = os.path.join(args.output_dir, f"noizeus_raw_{suffix}.csv")
        summary_path = os.path.join(args.output_dir, f"noizeus_summary_{suffix}.csv")
        results_df.to_csv(raw_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved raw rows to {raw_path}")
        print(f"Saved summary rows to {summary_path}")


if __name__ == "__main__":
    main()