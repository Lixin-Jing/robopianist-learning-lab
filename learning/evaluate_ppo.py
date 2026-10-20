"""Evaluate a trained PPO policy and report undiscounted episode returns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import dm_env_wrappers as wrappers
from robopianist import suite
import robopianist.wrappers as robopianist_wrappers

from learning.common import (
    DEFAULT_ENVIRONMENT,
    DEFAULT_RUN_DIR,
    make_env,
    task_kwargs,
)
from learning.gymnasium_env import flatten_observation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", default=DEFAULT_ENVIRONMENT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir or (
        DEFAULT_RUN_DIR / f"{args.environment}-seed{args.seed}"
    )
    normalization_env = DummyVecEnv(
        [lambda: Monitor(make_env(args.environment, args.seed + 20_000))]
    )
    normalization = VecNormalize.load(
        run_dir / "vecnormalize.pkl", normalization_env
    )
    normalization.training = False
    normalization.norm_reward = False
    model = PPO.load(run_dir / "ppo_final")

    environment = suite.load(
        environment_name=args.environment,
        seed=args.seed + 20_000,
        recompile_physics=False,
        task_kwargs=task_kwargs(),
    )
    environment = wrappers.EpisodeStatisticsWrapper(
        environment=environment, deque_size=args.episodes
    )
    environment = robopianist_wrappers.MidiEvaluationWrapper(
        environment=environment, deque_size=args.episodes
    )
    observation_keys = tuple(environment.observation_spec().keys())
    action_dtype = environment.action_spec().dtype

    returns: list[float] = []
    metrics: dict[str, float] = {}
    try:
        for episode in range(1, args.episodes + 1):
            time_step = environment.reset()
            episode_return = 0.0
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
            returns.append(episode_return)
            print(f"episode={episode} return={episode_return:.3f}")
        metrics = {
            **environment.get_statistics(),
            **environment.get_musical_metrics(),
        }
    finally:
        normalization.close()
        environment.close()

    print(f"mean_return={np.mean(returns):.3f} std_return={np.std(returns):.3f}")
    print(json.dumps({key: float(value) for key, value in metrics.items()}))


if __name__ == "__main__":
    main()
