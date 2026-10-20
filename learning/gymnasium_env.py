"""Gymnasium adapter for RoboPianist's dm_env environments.

The adapter converts RoboPianist's ordered observation dictionary into one
float32 vector and preserves the distinction between terminal failures and the
normal end of a MIDI sequence.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import dm_env
from dm_env import specs
import gymnasium as gym
import numpy as np


def _bounds(spec: specs.Array) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(spec, specs.BoundedArray):
        low = np.broadcast_to(spec.minimum, spec.shape)
        high = np.broadcast_to(spec.maximum, spec.shape)
    else:
        low = np.full(spec.shape, -np.inf)
        high = np.full(spec.shape, np.inf)
    return low.reshape(-1).astype(np.float32), high.reshape(-1).astype(np.float32)


def flatten_observation(
    observation: Mapping[str, Any], observation_keys: tuple[str, ...]
) -> np.ndarray:
    """Flatten an observation in a stable, explicit key order."""
    return np.concatenate(
        [
            np.asarray(observation[key], dtype=np.float32).reshape(-1)
            for key in observation_keys
        ]
    )


class RoboPianistGymEnv(gym.Env[np.ndarray, np.ndarray]):
    """Expose a RoboPianist ``dm_env.Environment`` through Gymnasium.

    A LAST dm_env timestep with discount 0 is terminal. A LAST timestep with a
    non-zero discount is the normal end of the MIDI trajectory and is a
    Gymnasium truncation. PPO needs this distinction for value bootstrapping.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        environment: dm_env.Environment,
        render_mode: str | None = None,
        render_width: int = 640,
        render_height: int = 480,
        camera_id: str | int = "piano/back",
    ) -> None:
        super().__init__()
        if render_mode not in (None, "rgb_array"):
            raise ValueError(f"Unsupported render_mode: {render_mode!r}")

        observation_spec = environment.observation_spec()
        if not isinstance(observation_spec, Mapping):
            raise TypeError("RoboPianist observation_spec() must be a mapping.")
        action_spec = environment.action_spec()
        if not isinstance(action_spec, specs.BoundedArray):
            raise TypeError("RoboPianist action_spec() must be a BoundedArray.")

        self.dm_env = environment
        self.observation_keys = tuple(observation_spec.keys())
        observation_bounds = [
            _bounds(observation_spec[key]) for key in self.observation_keys
        ]
        obs_low = np.concatenate([low for low, _ in observation_bounds])
        obs_high = np.concatenate([high for _, high in observation_bounds])
        action_low = np.broadcast_to(action_spec.minimum, action_spec.shape).astype(
            np.float32
        )
        action_high = np.broadcast_to(action_spec.maximum, action_spec.shape).astype(
            np.float32
        )

        self.observation_space = gym.spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space = gym.spaces.Box(
            action_low, action_high, dtype=np.float32
        )
        self.render_mode = render_mode
        self._render_width = render_width
        self._render_height = render_height
        self._camera_id = camera_id
        self._action_dtype = action_spec.dtype

    def _observation(self, time_step: dm_env.TimeStep) -> np.ndarray:
        return flatten_observation(time_step.observation, self.observation_keys)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        super().reset(seed=seed)
        if seed is not None:
            random_state = getattr(self.dm_env, "random_state", None)
            if random_state is not None and hasattr(random_state, "seed"):
                random_state.seed(seed)
        time_step = self.dm_env.reset()
        info = {"dm_env_step_type": time_step.step_type.name}
        return self._observation(time_step), info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        clipped_action = np.clip(
            np.asarray(action, dtype=self._action_dtype),
            self.action_space.low,
            self.action_space.high,
        )
        time_step = self.dm_env.step(clipped_action)
        reward = 0.0 if time_step.reward is None else float(time_step.reward)
        discount = 1.0 if time_step.discount is None else float(time_step.discount)
        terminated = bool(time_step.last() and np.isclose(discount, 0.0))
        truncated = bool(time_step.last() and not terminated)
        info = {
            "discount": discount,
            "dm_env_step_type": time_step.step_type.name,
        }
        return self._observation(time_step), reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        physics = getattr(self.dm_env, "physics", None)
        if physics is None:
            raise RuntimeError("The wrapped environment does not expose MuJoCo physics.")
        return physics.render(
            height=self._render_height,
            width=self._render_width,
            camera_id=self._camera_id,
        )

    def close(self) -> None:
        close = getattr(self.dm_env, "close", None)
        if close is not None:
            close()
