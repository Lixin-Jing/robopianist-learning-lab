"""Shared task settings for PPO training and evaluation."""

from __future__ import annotations

from pathlib import Path

from robopianist import suite

from learning.gymnasium_env import RoboPianistGymEnv

DEFAULT_ENVIRONMENT = "RoboPianist-debug-TwinkleTwinkleLittleStar-v0"
DEFAULT_RUN_DIR = Path(__file__).resolve().parents[1] / "training_runs" / "ppo"


def task_kwargs(fingering_reward_margin: float | None = None) -> dict[str, object]:
    """Task settings matching the public SAC recipe where possible."""
    options = {
        "n_steps_lookahead": 10,
        "trim_silence": True,
        "gravity_compensation": True,
        "reduced_action_space": True,
        "control_timestep": 0.05,
        "primitive_fingertip_collisions": True,
        "change_color_on_activation": True,
    }
    if fingering_reward_margin is not None:
        options["fingering_reward_margin"] = fingering_reward_margin
    return options


def make_env(
    environment_name: str, seed: int, fingering_reward_margin: float | None = None
) -> RoboPianistGymEnv:
    dm_environment = suite.load(
        environment_name=environment_name,
        seed=seed,
        recompile_physics=False,
        task_kwargs=task_kwargs(fingering_reward_margin),
    )
    return RoboPianistGymEnv(dm_environment)
