"""Run the independent third reward ablation: lower energy cost only."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from learning.reward_experiments.run import (
    ROOT, FIRST, ENV, SEEDS, verify_original_sources, write_json, summarize,
)

SECOND = ROOT / "training_runs/reward_shallow_press_2026-10-27"
GROUP = "round3_energy_0p001"


def train_one(output, seed):
    run = output / "models" / GROUP / f"{ENV}-seed{seed}"
    if (run / "ppo_final.zip").exists() and (run / "vecnormalize.pkl").exists():
        config = json.loads((run / "config.json").read_text())
        assert config["seed"] == seed and config["energy_penalty_coef"] == 0.001
        assert config["shallow_press_eta"] == 0
        return {"seed": seed, "status": "reused_completed_round3"}
    run.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-B", "-u", "-m", "learning.reward_experiments.energy_train",
        "--seed", str(seed), "--energy-coef", "0.001", "--output-dir", str(output / "models" / GROUP)]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
    with (run / "training_console.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    return {"seed": seed, "status": "trained"}


def verify_configs(output):
    from dataclasses import asdict
    from learning.train_ppo import Config
    from learning.common import task_kwargs
    for seed in SEEDS:
        control = json.loads((FIRST / "models/margin_0p1" / f"{ENV}-seed{seed}" / "config.json").read_text())
        current = json.loads((output / "models" / GROUP / f"{ENV}-seed{seed}" / "config.json").read_text())
        for key, value in asdict(Config(ENV, 20000, seed, "ignored")).items():
            if key != "output_dir": assert control[key] == current[key] == value, key
        old_options, new_options = control["task_kwargs"].copy(), current["task_kwargs"].copy()
        assert old_options.pop("fingering_reward_margin") == 0.1
        assert new_options.pop("energy_penalty_coef") == 0.001
        assert old_options == new_options == task_kwargs()
        assert control["torch_threads"] == current["torch_threads"] == 1
        assert current["fingering_margin_m"] == 0.1
        assert current["shallow_press_eta"] == 0
        assert current["reward_variant"] == "energy_penalty"
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started_at": datetime.now(ZoneInfo("Europe/Berlin")).isoformat(),
        "original_core_sha256": verify_original_sources(),
        "training_seeds": SEEDS, "evaluation_seeds": [40000, 40001, 40002],
        "requested_timesteps": 20000, "actual_timesteps": 20224,
        "original_energy_coefficient": 0.005, "training_energy_coefficient": 0.001,
        "evaluation_energy_coefficient": 0.005, "shallow_press_eta": 0,
        "fingering_margin_m": 0.1, "torch_threads": 1,
        "control_models": str(FIRST / "models/margin_0p1"),
        "design": "Round 3 alone; original fingering and key-press rewards; no round-1/2 intervention",
        "parameter_status": "0.001 is a single exploratory candidate, not a paper-proven optimum",
    }
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
    baseline_csv = output / "evaluation/original/episode_results.csv"
    checks = {
        "round1_control": verify_baseline(FIRST / "evaluation/margin_0p1/episode_results.csv", baseline_csv),
        "round2_control": verify_baseline(SECOND / "evaluation/original/episode_results.csv", baseline_csv),
    }
    write_json(output / "baseline_regression.json", checks)
    evaluate_group(output / "models" / GROUP, output, GROUP)
    summarize(output, comparison_files={
        "original": baseline_csv,
        "round1_margin_0p3": FIRST / "evaluation/margin_0p3/episode_results.csv",
        "round1_margin_0p5": FIRST / "evaluation/margin_0p5/episode_results.csv",
        "round2_shallow_eta_0p5": SECOND / "evaluation/round2_shallow_eta_0p5/episode_results.csv",
        GROUP: output / "evaluation" / GROUP / "episode_results.csv",
    })
    verify_original_sources()
    write_json(output / "complete.json", {
        "finished_at": datetime.now(ZoneInfo("Europe/Berlin")).isoformat(),
        "new_models": 3, "reevaluated_control_models": 3,
        "complete_episodes": 126, "baseline_regression_matches_rounds_1_and_2": True,
    })
    print("EXPERIMENT_COMPLETE", output, flush=True)


if __name__ == "__main__":
    main()
