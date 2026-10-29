"""Check reward isolation and original-reward evaluation for round 2."""

import numpy as np
import pytest
from dm_control.utils.rewards import tolerance

from learning.common import DEFAULT_ENVIRONMENT, make_env
from learning.reward_experiments.shallow_press import ShallowPressReward, key_press_delta


@pytest.mark.parametrize("q", [0, 0.1, 0.25, 0.5, 0.75, 0.95, 1.0])
def test_candidate_curve_matches_specification(q):
    original = 0.5 * tolerance(1-q, bounds=(0, 0.05), margin=0.5, sigmoid="gaussian")
    expected = 0.5 * (0.5 * (2 * original) + 0.5 * np.clip(q / 0.95, 0, 1))
    assert original + key_press_delta([1], [q], 0.5) == pytest.approx(expected)


def test_silence_has_no_extra_reward():
    assert key_press_delta([0, 0], [1, 0.5], 0.5) == 0


def test_wrong_keys_do_not_change_the_added_term():
    assert key_press_delta([1, 0], [0.25, 0], 0.5) == key_press_delta([1, 0], [0.25, 1], 0.5)


@pytest.mark.parametrize("eta", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_eta(eta):
    with pytest.raises(ValueError):
        key_press_delta([1], [0.2], eta)


@pytest.mark.parametrize("eta", [0.0, 0.5])
def test_wrapper_preserves_full_episode_physics_and_original_terms(eta):
    original = make_env(DEFAULT_ENVIRONMENT, 77)
    candidate = ShallowPressReward(make_env(DEFAULT_ENVIRONMENT, 77), eta)
    try:
        a, _ = original.reset(seed=77)
        b, _ = candidate.reset(seed=77)
        np.testing.assert_array_equal(a, b)
        task = candidate.unwrapped.dm_env.task
        assert not hasattr(task, "_fingering_reward_margin")
        assert task._energy_penalty_coef == 0.005
        action = (original.action_space.low + original.action_space.high) / 2
        for step in range(161):
            a = original.step(action)
            b = candidate.step(action)
            np.testing.assert_array_equal(a[0], b[0])
            assert a[2:4] == b[2:4]
            assert b[4]["reward_diagnostics"]["original_reward"] == a[1]
            assert b[1] == pytest.approx(a[1] + b[4]["shallow_press_delta"])
            if eta == 0:
                assert a[1] == b[1]
            assert original.dm_env.task.reward_fn.reward_terms == task.reward_fn.reward_terms
        assert a[2] is False and a[3] is True
    finally:
        original.close()
        candidate.close()
