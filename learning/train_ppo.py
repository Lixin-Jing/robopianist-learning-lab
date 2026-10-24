"""Train a compact PPO baseline on a RoboPianist debug task."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from robopianist import suite

from learning.common import DEFAULT_ENVIRONMENT, DEFAULT_RUN_DIR, make_env, task_kwargs
from learning.gymnasium_env import RoboPianistGymEnv
from learning.tempo_curriculum import TempoCurriculum


@dataclass(frozen=True)
class Config:
    environment: str
    total_timesteps: int
    seed: int
    output_dir: str
    learning_rate: float = 3e-4
    n_steps: int = 256
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    target_kl: float | None = None
    slow_tempo_curriculum: bool = False
    curriculum_warmup_episodes: int = 32
    curriculum_slow_probability: float = 0.7
    curriculum_slow_stretch: float = 1.2


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", default=DEFAULT_ENVIRONMENT)
    parser.add_argument("--total-timesteps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--target-kl",
        type=float,
        default=None,
        help="Stop a PPO update early when the approximate KL is too large; "
        "omit to preserve the original baseline.",
    )
    parser.add_argument("--slow-tempo-curriculum", action="store_true")
    parser.add_argument("--curriculum-warmup-episodes", type=int, default=32)
    parser.add_argument("--curriculum-slow-probability", type=float, default=0.7)
    parser.add_argument("--curriculum-slow-stretch", type=float, default=1.2)
    args = parser.parse_args()
    if args.target_kl is not None and args.target_kl <= 0:
        parser.error("--target-kl must be positive")
    if args.slow_tempo_curriculum:
        try:
            TempoCurriculum(
                args.curriculum_warmup_episodes,
                args.curriculum_slow_probability,
                args.curriculum_slow_stretch,
            )
        except ValueError as error:
            parser.error(str(error))
    return Config(
        environment=args.environment,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        output_dir=str(args.output_dir.expanduser().resolve()),
        target_kl=args.target_kl,
        slow_tempo_curriculum=args.slow_tempo_curriculum,
        curriculum_warmup_episodes=args.curriculum_warmup_episodes,
        curriculum_slow_probability=args.curriculum_slow_probability,
        curriculum_slow_stretch=args.curriculum_slow_stretch,
    )


def main(config: Config) -> None:
    run_dir = Path(config.output_dir) / f"{config.environment}-seed{config.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps({**asdict(config), "task_kwargs": task_kwargs()}, indent=2),
        encoding="utf-8",
    )

    curriculum = None
    if config.slow_tempo_curriculum:
        curriculum = TempoCurriculum(
            config.curriculum_warmup_episodes,
            config.curriculum_slow_probability,
            config.curriculum_slow_stretch,
        )
        options = task_kwargs()
        options["augmentations"] = [curriculum]

        def training_env_factory() -> Monitor:
            dm_environment = suite.load(
                environment_name=config.environment,
                seed=config.seed,
                recompile_physics=False,
                task_kwargs=options,
            )
            return Monitor(RoboPianistGymEnv(dm_environment))
    else:
        def training_env_factory() -> Monitor:
            return Monitor(make_env(config.environment, config.seed))

    train_env = DummyVecEnv([training_env_factory])
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        gamma=config.gamma,
    )
    eval_env = DummyVecEnv(
        [lambda: Monitor(make_env(config.environment, config.seed + 10_000))]
    )
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=config.gamma,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(config.n_steps * 4, 1),
        save_path=str(run_dir / "checkpoints"),
        name_prefix="ppo",
        save_vecnormalize=True,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best"),
        log_path=str(run_dir / "eval"),
        eval_freq=max(config.n_steps * 4, 1),
        n_eval_episodes=1,
        deterministic=True,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        target_kl=config.target_kl,
        policy_kwargs={"net_arch": [256, 256]},
        tensorboard_log=str(run_dir / "tensorboard"),
        seed=config.seed,
        verbose=1,
        device="auto",
    )

    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=False,
        )
        model.save(run_dir / "ppo_final")
        train_env.save(run_dir / "vecnormalize.pkl")
        if curriculum is not None:
            with (run_dir / "curriculum_episodes.csv").open(
                "w", newline="", encoding="utf-8"
            ) as file:
                writer = csv.DictWriter(file, fieldnames=["episode", "stage", "stretch"])
                writer.writeheader()
                writer.writerows(curriculum.history)
    finally:
        train_env.close()
        eval_env.close()

    print(f"Saved PPO model and normalization statistics to: {run_dir}")


if __name__ == "__main__":
    main(parse_args())
