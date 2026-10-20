"""Build comparison tables and plots from PPO and SAC scenario evaluations."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = PROJECT_ROOT / "training_runs" / "full_experiment"
METRICS = (
    "episode_return",
    "precision",
    "recall",
    "f1",
    "sustain_f1",
    "length",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["algorithm"], row["scenario"])].append(row)
    summary: list[dict[str, object]] = []
    for (algorithm, scenario), group in groups.items():
        item: dict[str, object] = {
            "algorithm": algorithm,
            "scenario": scenario,
            "description_zh": group[0]["description_zh"],
            "episodes": len(group),
        }
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in group])
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_std"] = float(values.std())
        normalized_returns = np.asarray(
            [
                float(row["episode_return"]) / float(row["length"])
                for row in group
            ]
        )
        item["return_per_step_mean"] = float(normalized_returns.mean())
        item["return_per_step_std"] = float(normalized_returns.std())
        summary.append(item)
    return sorted(summary, key=lambda row: (str(row["scenario"]), str(row["algorithm"])))


def save_summary(path: Path, summary: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def plot_scenarios(summary: list[dict[str, object]], output_dir: Path) -> None:
    scenarios = list(dict.fromkeys(str(row["scenario"]) for row in summary))
    algorithms = ["PPO", "SAC"]
    x = np.arange(len(scenarios))
    width = 0.36
    for metric, ylabel in (
        ("f1", "Note F1"),
        ("episode_return", "Episode return"),
        ("return_per_step", "Return per step"),
    ):
        fig, ax = plt.subplots(figsize=(12, 5))
        for index, algorithm in enumerate(algorithms):
            by_scenario = {
                str(row["scenario"]): row
                for row in summary
                if row["algorithm"] == algorithm
            }
            means = [
                float(by_scenario[name][f"{metric}_mean"]) for name in scenarios
            ]
            stds = [
                float(by_scenario[name][f"{metric}_std"]) for name in scenarios
            ]
            offset = (index - 0.5) * width
            ax.bar(
                x + offset,
                means,
                width,
                yerr=stds,
                capsize=3,
                label=algorithm,
            )
        ax.set_xticks(x, scenarios, rotation=28, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(f"PPO vs SAC: {ylabel} across evaluation scenarios")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"scenario_{metric}.png", dpi=160)
        plt.close(fig)


def plot_checkpoint_sweep(results_dir: Path) -> None:
    ppo = read_csv(results_dir / "ppo" / "checkpoint_sweep.csv")
    sac = read_csv(results_dir / "sac" / "checkpoint_sweep.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for rows, algorithm in ((ppo, "PPO"), (sac, "SAC")):
        steps = [int(row["step"]) for row in rows]
        returns = [float(row["episode_return"]) for row in rows]
        f1_values = [float(row["f1"]) for row in rows]
        axes[0].plot(steps, returns, marker="o", label=algorithm)
        axes[1].plot(steps, f1_values, marker="o", label=algorithm)
    axes[0].set_ylabel("Episode return")
    axes[1].set_ylabel("Note F1")
    for axis in axes:
        axis.set_xlabel("Environment interactions")
        axis.grid(alpha=0.3)
        axis.legend()
    axes[0].set_title("Checkpoint evaluation return")
    axes[1].set_title("Checkpoint evaluation note F1")
    fig.tight_layout()
    fig.savefig(results_dir / "checkpoint_sweep.png", dpi=160)
    plt.close(fig)


def save_failures(
    path: Path, rows: list[dict[str, str]], quantile: float = 0.25
) -> None:
    failures: list[dict[str, str]] = []
    for algorithm in ("PPO", "SAC"):
        selected = [row for row in rows if row["algorithm"] == algorithm]
        threshold = float(
            np.quantile([float(row["f1"]) for row in selected], quantile)
        )
        for row in selected:
            if float(row["f1"]) <= threshold:
                failure = dict(row)
                failure["failure_rule"] = f"bottom_{int(quantile * 100)}pct_f1"
                failure["f1_threshold"] = str(threshold)
                failures.append(failure)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(failures[0]))
        writer.writeheader()
        writer.writerows(failures)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    ppo_rows = read_csv(args.results_dir / "ppo" / "scenario_results.csv")
    sac_rows = read_csv(args.results_dir / "sac" / "scenario_results.csv")
    rows = ppo_rows + sac_rows
    summary = summarize(rows)
    save_summary(args.results_dir / "scenario_summary.csv", summary)
    (args.results_dir / "scenario_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_failures(args.results_dir / "failure_cases.csv", rows)
    plot_scenarios(summary, args.results_dir)
    plot_checkpoint_sweep(args.results_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
