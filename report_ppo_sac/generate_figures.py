"""Generate report-specific figures from the archived experiment CSV files."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "scenario_summary.csv"
OUTPUT = ROOT / "images" / "baseline_metrics.png"


def main() -> None:
    with DATA.open(encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if row["scenario"] == "baseline"]

    by_algorithm = {row["algorithm"]: row for row in rows}
    metrics = ["precision_mean", "recall_mean", "f1_mean", "sustain_f1_mean"]
    labels = ["Precision", "Recall", "Note F1", "Sustain F1"]
    x = np.arange(len(metrics))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for offset, algorithm, color in (
        (-width / 2, "PPO", "#1f77b4"),
        (width / 2, "SAC", "#ff7f0e"),
    ):
        values = [float(by_algorithm[algorithm][metric]) for metric in metrics]
        bars = ax.bar(x + offset, values, width, label=algorithm, color=color)
        ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3)

    ax.set_title("PPO vs SAC: Baseline MIDI metrics")
    ax.set_ylabel("Score")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 1.10)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
