"""Round 3: change only the original task's energy-penalty coefficient."""

import numpy as np
import gymnasium as gym

from robopianist import suite
from learning.common import task_kwargs
from learning.gymnasium_env import RoboPianistGymEnv
from learning.reward_experiments.diagnostics import measure_reward_state

ORIGINAL_ENERGY_COEFFICIENT = 0.005


def make_energy_env(environment_name: str, seed: int, coefficient: float):
    if not np.isfinite(coefficient) or coefficient <= 0:
        raise ValueError("energy coefficient must be finite and positive")
    options = task_kwargs()
    options["energy_penalty_coef"] = coefficient
    dm = suite.load(environment_name=environment_name, seed=seed,
        recompile_physics=False, task_kwargs=options)
    return RoboPianistGymEnv(dm)


class EnergyDiagnostics(gym.Wrapper):
    """Keep native reward intact; record the comparable original-coefficient sum."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        dm = self.env.unwrapped.dm_env
        task = dm.task
        stats = measure_reward_state(task, dm.physics)
        energy = task.reward_fn.reward_terms["energy_reward"]
        original_energy = energy * (ORIGINAL_ENERGY_COEFFICIENT / task._energy_penalty_coef)
        stats["original_reward"] = float(reward - energy + original_energy)
        info["reward_diagnostics"] = stats
        return obs, reward, terminated, truncated, info
