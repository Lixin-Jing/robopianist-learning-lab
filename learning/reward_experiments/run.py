"""Run round 2 independently, then compare it with baseline and round 1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ENV = "RoboPianist-debug-TwinkleTwinkleLittleStar-v0"
FIRST = ROOT / "training_runs/reward_margin_2026-10-26"
SEEDS = [20000, 20001, 20002]
CORE = ["learning/common.py", "learning/train_ppo.py", "learning/diagnose_ppo_failures.py", "robopianist/suite/tasks/piano_with_shadow_hands.py"]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def verify_original_sources():
    hashes = {}
    for path in CORE:
        original = subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=ROOT)
        current = (ROOT / path).read_bytes()
        assert current == original, f"Original file differs: {path}"
        hashes[path] = hashlib.sha256(current).hexdigest()
    return hashes


def train_one(output, seed):
    run = output / "models/shallow_eta_0p5" / f"{ENV}-seed{seed}"
    if (run / "ppo_final.zip").exists() and (run / "vecnormalize.pkl").exists():
        c = json.loads((run / "config.json").read_text())
        assert c["shallow_press_eta"] == 0.5 and c["seed"] == seed
        return {"seed": seed, "status": "reused_completed_round2"}
    run.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-B", "-u", "-m", "learning.reward_experiments.train",
        "--seed", str(seed), "--eta", "0.5", "--output-dir", str(output / "models/shallow_eta_0p5")]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
    with (run / "training_console.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    return {"seed": seed, "status": "trained"}


def verify_configs(output):
    from learning.train_ppo import Config
    from dataclasses import asdict
    from learning.common import task_kwargs
    for seed in SEEDS:
        old = json.loads((FIRST / "models/margin_0p1" / f"{ENV}-seed{seed}" / "config.json").read_text())
        new = json.loads((output / "models/shallow_eta_0p5" / f"{ENV}-seed{seed}" / "config.json").read_text())
        expected = asdict(Config(ENV, 20000, seed, "ignored"))
        for key, value in expected.items():
            if key != "output_dir":
                assert old[key] == new[key] == value, (key, old[key], new[key])
        options = old["task_kwargs"].copy()
        assert options.pop("fingering_reward_margin") == 0.1
        assert options == new["task_kwargs"] == task_kwargs()
        assert old["torch_threads"] == new["torch_threads"] == 1
        assert new["energy_penalty_coef"] == 0.005
    return True


def summarize(output, comparison_files=None):
    from learning.diagnose_ppo_failures import write_csv
    metrics = ["key_micro_recall", "key_micro_f1", "wrapper_f1", "episode_return", "unwanted_pressed_keys", "finger_distance_mean_m", "target_depression_mean", "finger_within_5cm_fraction", "reward_energy_reward_sum"]
    files = {
        "original": output / "evaluation/original/episode_results.csv",
        "round1_margin_0p3": FIRST / "evaluation/margin_0p3/episode_results.csv",
        "round1_margin_0p5": FIRST / "evaluation/margin_0p5/episode_results.csv",
        "round2_shallow_eta_0p5": output / "evaluation/round2_shallow_eta_0p5/episode_results.csv",
    }
    if comparison_files is not None:
        files = dict(comparison_files)
    seed_rows, summaries, scenario_rows = [], [], []
    for group, path in files.items():
        with path.open() as f: rows = list(csv.DictReader(f))
        assert len(rows) == 63
        for scope in ["baseline", "seven_scenario_mean"]:
            values = []
            for seed in SEEDS:
                rs = [r for r in rows if int(r["training_seed"]) == seed and (scope != "baseline" or r["scenario"] == "baseline")]
                scenarios = sorted({r["scenario"] for r in rs})
                v = {m: float(np.mean([np.mean([float(r[m]) for r in rs if r["scenario"] == s]) for s in scenarios])) for m in metrics}
                values.append(v)
                seed_rows.append({"group": group, "scope": scope, "training_seed": seed, **v})
            summaries.append({"group": group, "scope": scope, **{f"{m}_{stat}": float(fn([v[m] for v in values])) for m in metrics for stat, fn in [("mean", np.mean), ("sample_sd", lambda x: np.std(x, ddof=1))]}})
        for scenario in sorted({r["scenario"] for r in rows}):
            values = [{m: np.mean([float(r[m]) for r in rows if int(r["training_seed"]) == seed and r["scenario"] == scenario]) for m in metrics} for seed in SEEDS]
            scenario_rows.append({"group": group, "scenario": scenario, **{f"{m}_{stat}": float(fn([v[m] for v in values])) for m in metrics for stat, fn in [("mean", np.mean), ("sample_sd", lambda x: np.std(x, ddof=1))]}})
    deltas = []
    for r in summaries:
        b = next(b for b in summaries if b["group"] == "original" and b["scope"] == r["scope"])
        deltas.append({"group": r["group"], "scope": r["scope"], **{f"{m}_{kind}": (r[m+"_mean"]-b[m+"_mean"] if kind == "absolute_delta" else (100*(r[m+"_mean"]/b[m+"_mean"]-1) if b[m+"_mean"] else None)) for m in metrics for kind in ["absolute_delta", "relative_change_percent"]}})
    write_csv(output / "per_seed_summary.csv", seed_rows)
    write_csv(output / "group_summary.csv", summaries)
    write_csv(output / "scenario_summary.csv", scenario_rows)
    write_csv(output / "changes_vs_original.csv", deltas)
    write_json(output / "results.json", {"summary": summaries, "per_seed": seed_rows, "changes_vs_original": deltas, "comparison": "independent interventions against the SAME original-reward PPO control"})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (output / "figures").mkdir(exist_ok=True)
    for scope in ["baseline", "seven_scenario_mean"]:
        rs = [r for r in summaries if r["scope"] == scope]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
        for ax, metric, title in zip(axes, ["key_micro_recall", "key_micro_f1", "unwanted_pressed_keys"], ["Target-key recall", "Key-time micro F1", "Unwanted key-time cells / episode"]):
            x = np.arange(len(rs))
            ax.bar(x, [r[metric+"_mean"] for r in rs], yerr=[r[metric+"_sample_sd"] for r in rs], capsize=4, color=["#6b7280", "#16a085", "#e69f00", "#4888ce", "#ab67bd"][:len(rs)], alpha=.8)
            for i,r in enumerate(rs):
                points = [v[metric] for v in seed_rows if v["group"] == r["group"] and v["scope"] == scope]
                ax.scatter(i+np.linspace(-.12,.12,3), points, color="black", s=18, zorder=3)
            labels = {"original": "Original", "round1_margin_0p3": "R1: 0.30 m", "round1_margin_0p5": "R1: 0.50 m", "round2_shallow_eta_0p5": "R2: eta=0.5", "round3_energy_0p001": "R3: energy=0.001"}
            ax.set_xticks(x, [labels[r["group"]] for r in rs], rotation=20)
            ax.set_title(title)
            ax.set_ylim(bottom=0)
        fig.suptitle(f"Independent reward experiments: {scope}; 3 training seeds; mean +/- sample SD")
        fig.savefig(output / "figures" / f"{scope}.png", dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"started_at": datetime.now(ZoneInfo("Europe/Berlin")).isoformat(),
        "original_core_sha256": verify_original_sources(), "eta": 0.5,
        "fingering_margin_m": 0.1, "energy_penalty_coef": 0.005,
        "training_seeds": SEEDS, "evaluation_seeds": [40000,40001,40002],
        "requested_timesteps": 20000, "actual_timesteps": 20224,
        "control_models": str(FIRST / "models/margin_0p1"),
        "design": "Round 2 alone; no round-1 margin modification; original-reward evaluation"}
    if not (output / "manifest.json").exists(): write_json(output / "manifest.json", manifest)
    status = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(train_one, output, seed) for seed in SEEDS]
        for future in as_completed(jobs):
            status.append(future.result())
            write_json(output / "training_status.json", status)
            print("TRAIN_COMPLETE", status[-1], flush=True)
    verify_configs(output)
    from learning.reward_experiments.evaluation import evaluate_group, verify_baseline
    evaluate_group(FIRST / "models/margin_0p1", output, "original")
    check = verify_baseline(FIRST / "evaluation/margin_0p1/episode_results.csv", output / "evaluation/original/episode_results.csv")
    write_json(output / "baseline_regression.json", check)
    evaluate_group(output / "models/shallow_eta_0p5", output, "round2_shallow_eta_0p5")
    summarize(output)
    verify_original_sources()
    write_json(output / "complete.json", {"finished_at": datetime.now(ZoneInfo("Europe/Berlin")).isoformat(), "new_models": 3, "reevaluated_control_models": 3, "complete_episodes": 126, "baseline_regression_matches": True})
    print("EXPERIMENT_COMPLETE", output, flush=True)


if __name__ == "__main__":
    main()
