"""Verify energy-only reward changes against a full original MIDI episode."""

import numpy as np
import pytest

from learning.common import DEFAULT_ENVIRONMENT, make_env
from learning.reward_experiments.energy import make_energy_env, EnergyDiagnostics


@pytest.mark.parametrize("coefficient", [0.005, 0.001])
def test_energy_change_leaves_everything_else_original(coefficient):
    original = make_env(DEFAULT_ENVIRONMENT, 77)
    candidate = EnergyDiagnostics(make_energy_env(DEFAULT_ENVIRONMENT, 77, coefficient))
    try:
        a, _ = original.reset(seed=77)
        b, _ = candidate.reset(seed=77)
        np.testing.assert_array_equal(a, b)
        task = candidate.unwrapped.dm_env.task
        assert not hasattr(task, "_fingering_reward_margin")
        assert task._energy_penalty_coef == coefficient
        action = (original.action_space.low + original.action_space.high) / 2
        observed_nonzero_energy = False
        for _ in range(161):
            a, b = original.step(action), candidate.step(action)
            np.testing.assert_array_equal(a[0], b[0])
            assert a[2:4] == b[2:4]
            x, y = original.dm_env.task.reward_fn.reward_terms, task.reward_fn.reward_terms
            for name in x:
                if name != "energy_reward": assert x[name] == y[name]
            assert y["energy_reward"] == pytest.approx(x["energy_reward"] * coefficient / 0.005)
            assert b[1] - a[1] == pytest.approx(y["energy_reward"] - x["energy_reward"], abs=1e-12)
            assert b[4]["reward_diagnostics"]["original_reward"] == pytest.approx(a[1], abs=1e-10)
            observed_nonzero_energy |= abs(x["energy_reward"]) > 1e-8
            if coefficient == 0.005: assert a[1] == b[1]
        assert observed_nonzero_energy
        assert a[2] is False and a[3] is True
    finally:
        original.close()
        candidate.close()


@pytest.mark.parametrize("coefficient", [0, -0.001, float("nan"), float("inf")])
def test_invalid_energy_coefficients(coefficient):
    with pytest.raises(ValueError, match="finite and positive"):
        make_energy_env(DEFAULT_ENVIRONMENT, 77, coefficient)
