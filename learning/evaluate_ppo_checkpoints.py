"""Evaluate every saved PPO checkpoint with its matching normalization state."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from learning.common import DEFAULT_ENVIRONMENT, DEFAULT_RUN_DIR, make_env
from learning.evaluate_ppo_scenarios import evaluate_one, load_scenarios


DEFAULT_SCENARIOS = Path(__file__).with_name("experiment_scenarios.json")
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "training_runs"
    / "full_experiment"
    / "ppo"
    / "checkpoint_sweep.csv"
)
CHECKPOINT_PATTERN = re.compile(r"ppo_(\d+)_steps\.zip")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", default=DEFAULT_ENVIRONMENT)
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--evaluation-seed", type=int, default=20_000)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    run_dir = args.run_dir or (
        DEFAULT_RUN_DIR / f"{args.environment}-seed{args.training_seed}"
    )
    scenario = load_scenarios(args.scenarios)[0]
    checkpoints: list[tuple[int, Path]] = []
    for path in (run_dir / "checkpoints").glob("ppo_*_steps.zip"):
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    checkpoints.sort()
    if not checkpoints:
        raise FileNotFoundError(f"No PPO checkpoints found under {run_dir}")

    rows: list[dict[str, object]] = []
    for step, checkpoint in checkpoints:
        normalization_env = DummyVecEnv(
            [lambda: Monitor(make_env(args.environment, args.evaluation_seed))]
        )
        normalization = VecNormalize.load(
            run_dir
            / "checkpoints"
            / f"ppo_vecnormalize_{step}_steps.pkl",
            normalization_env,
        )
        normalization.training = False
        normalization.norm_reward = False
        model = PPO.load(checkpoint)
        try:
            row = evaluate_one(
                model,
                normalization,
                args.environment,
                scenario,
                args.evaluation_seed,
            )
        finally:
            normalization.close()
        row["step"] = step
        rows.append(row)
        print(
            f"step={step} return={row['episode_return']:.3f} "
            f"f1={row['f1']:.6f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved PPO checkpoint sweep to {args.output}")


if __name__ == "__main__":
    main()
