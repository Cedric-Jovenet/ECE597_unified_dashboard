import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


REPRESENTATIVE_ALGORITHMS = [
    "Spectral Subtraction (Classic)",
    "Wiener Filtering",
    "MMSE-STSA",
    "Signal Subspace Enhancement (EVT)",
]

ALGORITHM_SHORT_LABELS = {
    "Spectral Subtraction (Classic)": "Spectral",
    "Wiener Filtering": "Wiener",
    "MMSE-STSA": "MMSE-STSA",
    "Signal Subspace Enhancement (EVT)": "EVT",
}

ALGORITHM_COLORS = {
    "Spectral Subtraction (Classic)": "#4C78A8",
    "Wiener Filtering": "#59A14F",
    "MMSE-STSA": "#F28E2B",
    "Signal Subspace Enhancement (EVT)": "#E15759",
}

NOISE_ORDER = [
    "airport",
    "babble",
    "car",
    "exhibition",
    "restaurant",
    "station",
    "street",
    "train",
]

METRICS = [
    ("snr_improvement", "Delta SNR"),
    ("pesq", "PESQ"),
    ("stoi", "STOI"),
]


def compute_representative_summary(raw_df):
    filtered = raw_df[raw_df["algorithm"].isin(REPRESENTATIVE_ALGORITHMS)].copy()
    summary = (
        filtered.groupby(["noise", "algorithm"])[["snr_improvement", "pesq", "stoi"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "noise",
        "algorithm",
        "snr_improvement_mean",
        "snr_improvement_std",
        "pesq_mean",
        "pesq_std",
        "stoi_mean",
        "stoi_std",
    ]

    summary["noise"] = pd.Categorical(summary["noise"], categories=NOISE_ORDER, ordered=True)
    summary["algorithm"] = pd.Categorical(summary["algorithm"], categories=REPRESENTATIVE_ALGORITHMS, ordered=True)
    summary = summary.sort_values(["noise", "algorithm"]).reset_index(drop=True)
    return summary


def compute_winners(summary_df):
    winner_rows = []
    for noise in NOISE_ORDER:
        noise_rows = summary_df[summary_df["noise"] == noise]
        if noise_rows.empty:
            continue
        winner_row = {"noise": noise}
        for metric_key, _ in METRICS:
            mean_column = f"{metric_key}_mean"
            best = noise_rows.sort_values(mean_column, ascending=False).iloc[0]
            winner_row[f"{metric_key}_winner"] = best["algorithm"]
            winner_row[f"{metric_key}_value"] = best[mean_column]
        winner_rows.append(winner_row)
    return pd.DataFrame(winner_rows)


def format_mean_std(mean_value, std_value):
    return f"{mean_value:.3f} +/- {std_value:.3f}"


def write_markdown_table(summary_df, markdown_path):
    lines = [
        "# Detailed Per-Scenario Representative Table",
        "",
        "Derived from `artifacts/noizeus_raw_all_conditions.csv`.",
        "",
        "| Scenario | Algorithm | Delta SNR (mean +/- std) | PESQ (mean +/- std) | STOI (mean +/- std) |",
        "| --- | --- | --- | --- | --- |",
    ]

    for _, row in summary_df.iterrows():
        lines.append(
            "| {noise} | {algorithm} | {snr} | {pesq} | {stoi} |".format(
                noise=str(row["noise"]).capitalize(),
                algorithm=row["algorithm"],
                snr=format_mean_std(row["snr_improvement_mean"], row["snr_improvement_std"]),
                pesq=format_mean_std(row["pesq_mean"], row["pesq_std"]),
                stoi=format_mean_std(row["stoi_mean"], row["stoi_std"]),
            )
        )

    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_winner_heatmap(winners_df, output_path):
    algorithm_order = REPRESENTATIVE_ALGORITHMS
    cmap = ListedColormap([ALGORITHM_COLORS[name] for name in algorithm_order])
    algorithm_to_index = {name: index for index, name in enumerate(algorithm_order)}

    data = np.zeros((len(winners_df), len(METRICS)), dtype=int)
    annotations = [["" for _ in METRICS] for _ in range(len(winners_df))]

    for row_index, (_, row) in enumerate(winners_df.iterrows()):
        for col_index, (metric_key, _) in enumerate(METRICS):
            winner = row[f"{metric_key}_winner"]
            value = row[f"{metric_key}_value"]
            data[row_index, col_index] = algorithm_to_index[winner]
            annotations[row_index][col_index] = f"{ALGORITHM_SHORT_LABELS[winner]}\n{value:.3f}"

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(data, cmap=cmap, aspect="auto", vmin=-0.5, vmax=len(algorithm_order) - 0.5)

    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels([label for _, label in METRICS], fontsize=11)
    ax.set_yticks(range(len(winners_df)))
    ax.set_yticklabels([str(noise).capitalize() for noise in winners_df["noise"]], fontsize=11)
    ax.set_title("NOIZEUS Scenario Winner Heatmap (Representative Algorithms)", fontsize=13, pad=14)

    ax.set_xticks(np.arange(-0.5, len(METRICS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(winners_df), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row_index in range(len(winners_df)):
        for col_index in range(len(METRICS)):
            ax.text(
                col_index,
                row_index,
                annotations[row_index][col_index],
                ha="center",
                va="center",
                color="black",
                fontsize=9,
                fontweight="bold",
            )

    legend_handles = [
        Patch(facecolor=ALGORITHM_COLORS[name], edgecolor="none", label=name)
        for name in algorithm_order
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate reporting assets from the NOIZEUS full benchmark export.")
    parser.add_argument(
        "--raw-csv",
        default="artifacts/noizeus_raw_all_conditions.csv",
        help="Path to the raw all-conditions benchmark CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/reporting",
        help="Directory where reporting assets will be written.",
    )
    args = parser.parse_args()

    raw_csv = Path(args.raw_csv)
    if not raw_csv.exists():
        raise FileNotFoundError(f"Raw benchmark CSV not found: {raw_csv}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(raw_csv)
    summary_df = compute_representative_summary(raw_df)
    winners_df = compute_winners(summary_df)

    summary_csv_path = output_dir / "noizeus_representative_scenario_summary.csv"
    winners_csv_path = output_dir / "noizeus_scenario_winners.csv"
    summary_md_path = output_dir / "noizeus_representative_scenario_summary.md"
    heatmap_path = output_dir / "noizeus_scenario_winner_heatmap.png"

    summary_df.to_csv(summary_csv_path, index=False)
    winners_df.to_csv(winners_csv_path, index=False)
    write_markdown_table(summary_df, summary_md_path)
    render_winner_heatmap(winners_df, heatmap_path)

    print(f"Saved representative summary CSV to {summary_csv_path}")
    print(f"Saved representative summary Markdown table to {summary_md_path}")
    print(f"Saved scenario winner CSV to {winners_csv_path}")
    print(f"Saved scenario winner heatmap to {heatmap_path}")


if __name__ == "__main__":
    main()