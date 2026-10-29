"""Round 2: strengthen shallow target-key depression, with original fingering."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from dm_control.utils.rewards import tolerance

from learning.reward_experiments.diagnostics import measure_reward_state


def key_press_delta(goal, key_state, eta: float) -> float:
    """Replace only eta of the original 0.5-weight target-depression term.

    Original: g(1-q). Candidate: (1-eta)*g(1-q) + eta*clip(q/0.95, 0, 1).
    The no-false-positive half, silence handling, and all other terms stay intact.
    """
    if not np.isfinite(eta) or not 0 <= eta <= 1:
        raise ValueError("eta must be finite and in [0, 1]")
    on = np.flatnonzero(goal)
    if not on.size or eta == 0:
        return 0.0
    q = np.asarray(key_state)[on]
    original = tolerance(1.0 - q, bounds=(0, 0.05), margin=0.5, sigmoid="gaussian")
    linear = np.clip(q / 0.95, 0.0, 1.0)
    return float(0.5 * eta * np.mean(linear - original))


class ShallowPressReward(gym.Wrapper):
    """Apply the isolated reward change after physics, before vector auto-reset."""

    def __init__(self, env, eta: float = 0.5, diagnostics: bool = True):
        super().__init__(env)
        key_press_delta(np.zeros(1), np.zeros(1), eta)
        self.eta = eta
        self.diagnostics = diagnostics

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        dm = self.env.unwrapped.dm_env
        task = dm.task
        q = task.piano.state / task.piano._qpos_range[:, 1]
        delta = key_press_delta(task._goal_current[:-1], q, self.eta)
        if self.diagnostics:
            stats = measure_reward_state(task, dm.physics)
            stats["original_reward"] = reward
            stats["reward_key_press_reward"] += delta
            info["reward_diagnostics"] = stats
            info["shallow_press_delta"] = delta
        return obs, reward + delta, terminated, truncated, info
