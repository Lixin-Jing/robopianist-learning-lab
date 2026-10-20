"""Evaluate a saved PPO policy on a fixed multi-scenario protocol."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import dm_env_wrappers as wrappers
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from robopianist import suite
import robopianist.wrappers as robopianist_wrappers

from learning.common import DEFAULT_ENVIRONMENT, DEFAULT_RUN_DIR, make_env, task_kwargs
from learning.gymnasium_env import flatten_observation


DEFAULT_SCENARIOS = Path(__file__).with_name("experiment_scenarios.json")
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1] / "training_runs" / "full_experiment" / "ppo"
)


def load_scenarios(path: Path) -> list[dict[str, object]]:
    scenarios = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("Scenario file must contain a non-empty JSON list.")
    return scenarios


def evaluate_one(
    model: PPO,
    normalization: VecNormalize,
    environment_name: str,
    scenario: dict[str, object],
    seed: int,
) -> dict[str, object]:
    options = task_kwargs()
    options["randomize_hand_positions"] = bool(
        scenario["randomize_hand_positions"]
    )
    environment = suite.load(
        environment_name=environment_name,
        seed=seed,
        stretch=float(scenario["stretch"]),
        shift=int(scenario["shift"]),
        recompile_physics=False,
        task_kwargs=options,
    )
    environment = wrappers.EpisodeStatisticsWrapper(environment, deque_size=1)
    environment = robopianist_wrappers.MidiEvaluationWrapper(
        environment, deque_size=1
    )
    observation_keys = tuple(environment.observation_spec().keys())
    action_dtype = environment.action_spec().dtype

    episode_return = 0.0
    try:
        time_step = environment.reset()
        while not time_step.last():
            observation = flatten_observation(
                time_step.observation, observation_keys
            )
            normalized = normalization.normalize_obs(observation[None, :])
            action, _ = model.predict(normalized, deterministic=True)
            time_step = environment.step(
                np.asarray(action[0], dtype=action_dtype)
            )
            episode_return += float(time_step.reward or 0.0)
        metrics = {
            **environment.get_statistics(),
            **environment.get_musical_metrics(),
        }
    finally:
        environment.close()

    return {
        "algorithm": "PPO",
        "scenario": scenario["name"],
        "description_zh": scenario["description_zh"],
        "seed": seed,
        "stretch": float(scenario["stretch"]),
        "shift": int(scenario["shift"]),
        "randomize_hand_positions": bool(
            scenario["randomize_hand_positions"]
        ),
        "episode_return": episode_return,
        **{key: float(value) for key, value in metrics.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", default=DEFAULT_ENVIRONMENT)
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20_000, 20_001, 20_002])
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    run_dir = args.run_dir or (
        DEFAULT_RUN_DIR / f"{args.environment}-seed{args.training_seed}"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    normalization_env = DummyVecEnv(
        [lambda: Monitor(make_env(args.environment, args.training_seed + 20_000))]
    )
    normalization = VecNormalize.load(
        run_dir / "vecnormalize.pkl", normalization_env
    )
    normalization.training = False
    normalization.norm_reward = False
    model = PPO.load(run_dir / "ppo_final")

    rows: list[dict[str, object]] = []
    try:
        for scenario in load_scenarios(args.scenarios):
            for seed in args.seeds:
                row = evaluate_one(
                    model, normalization, args.environment, scenario, seed
                )
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False))
    finally:
        normalization.close()

    csv_path = args.output_dir / "scenario_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "scenario_results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(rows)} PPO evaluations to {csv_path}")


if __name__ == "__main__":
    main()
