"""Open the dm_control viewer and play a trained PPO policy."""

from __future__ import annotations

import argparse
from pathlib import Path

from dm_control import viewer
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from learning.common import DEFAULT_ENVIRONMENT, DEFAULT_RUN_DIR, make_env, task_kwargs
from learning.gymnasium_env import flatten_observation
from robopianist import suite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", default=DEFAULT_ENVIRONMENT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()
    run_dir = args.run_dir or (
        DEFAULT_RUN_DIR / f"{args.environment}-seed{args.seed}"
    )

    normalization_env = DummyVecEnv(
        [lambda: Monitor(make_env(args.environment, args.seed))]
    )
    normalization = VecNormalize.load(run_dir / "vecnormalize.pkl", normalization_env)
    normalization.training = False
    normalization.norm_reward = False
    model = PPO.load(run_dir / "ppo_final")

    environment = suite.load(
        environment_name=args.environment,
        seed=args.seed,
        recompile_physics=False,
        task_kwargs=task_kwargs(),
    )
    observation_keys = tuple(environment.observation_spec().keys())

    def policy(time_step):
        flat = flatten_observation(time_step.observation, observation_keys)
        normalized = normalization.normalize_obs(flat[None, :])
        action, _ = model.predict(normalized, deterministic=True)
        return np.asarray(action[0], dtype=environment.action_spec().dtype)

    try:
        viewer.launch(environment, policy=policy)
    finally:
        normalization.close()


if __name__ == "__main__":
    main()
