"""Fast unit tests for the dm_env to Gymnasium conversion."""

from collections import OrderedDict

import dm_env
from dm_env import specs
import numpy as np

from learning.gymnasium_env import RoboPianistGymEnv


class FakeDmEnv(dm_env.Environment):
    def __init__(self, terminal_discount: float = 1.0) -> None:
        self.terminal_discount = terminal_discount
        self.random_state = np.random.RandomState(0)

    def observation_spec(self):
        return OrderedDict(
            position=specs.BoundedArray((2,), np.float64, -1.0, 1.0),
            goal=specs.Array((1, 2), np.int32),
        )

    def action_spec(self):
        return specs.BoundedArray((2,), np.float64, [-2.0, -1.0], [2.0, 1.0])

    def reset(self):
        return dm_env.restart(
            OrderedDict(
                position=np.array([0.25, -0.5]),
                goal=np.array([[1, 0]]),
            )
        )

    def step(self, action):
        observation = OrderedDict(
            position=np.asarray(action),
            goal=np.array([[0, 1]]),
        )
        if self.terminal_discount == 0:
            return dm_env.termination(2.0, observation)
        return dm_env.truncation(2.0, observation, self.terminal_discount)


def test_spaces_and_flattening():
    env = RoboPianistGymEnv(FakeDmEnv())
    observation, info = env.reset(seed=7)
    np.testing.assert_allclose(observation, [0.25, -0.5, 1.0, 0.0])
    assert env.observation_space.contains(observation)
    assert env.action_space.shape == (2,)
    assert info["dm_env_step_type"] == "FIRST"


def test_normal_midi_end_is_truncated():
    env = RoboPianistGymEnv(FakeDmEnv(terminal_discount=1.0))
    env.reset()
    _, reward, terminated, truncated, info = env.step(np.array([99.0, -99.0]))
    assert reward == 2.0
    assert not terminated
    assert truncated
    assert info["discount"] == 1.0


def test_zero_discount_is_terminated():
    env = RoboPianistGymEnv(FakeDmEnv(terminal_discount=0.0))
    env.reset()
    observation, _, terminated, truncated, _ = env.step(np.zeros(2))
    assert observation.dtype == np.float32
    assert terminated
    assert not truncated
