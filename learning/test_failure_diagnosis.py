"""Small tests for the PPO failure-diagnosis arithmetic."""

import numpy as np
import pytest

from learning.diagnose_ppo_failures import (
    hand_offset_bucket,
    key_counts,
    key_micro_f1,
    song_quarter,
)


@pytest.mark.parametrize("length", [1, 3, 4, 5, 161, 193])
def test_song_quarters_cover_episode(length: int) -> None:
    quarters = [song_quarter(step, length) for step in range(length)]
    assert quarters[0] == 1
    assert quarters[-1] == 4 or length < 4
    assert all(1 <= quarter <= 4 for quarter in quarters)
    assert quarters == sorted(quarters)


def test_key_counts() -> None:
    target = np.array([1, 0, 1, 0], dtype=bool)
    actual = np.array([1, 1, 0, 0], dtype=bool)
    assert key_counts(target, actual) == (1, 1, 1)


def test_key_counts_rejects_mismatch() -> None:
    with pytest.raises(ValueError):
        key_counts(np.ones(2), np.ones(3))


@pytest.mark.parametrize(
    ("offset", "bucket"),
    [(0, "0-1cm"), (-0.012, "1-2.5cm"), (0.03, "2.5-4cm"), (-0.049, "4-5cm")],
)
def test_hand_offset_bucket(offset: float, bucket: str) -> None:
    assert hand_offset_bucket(offset) == bucket


def test_micro_f1_does_not_credit_silence() -> None:
    assert key_micro_f1(tp=0, fn=154, fp=45) == 0
    assert key_micro_f1(tp=1, fn=1, fp=1) == 0.5
    assert key_micro_f1(tp=0, fn=0, fp=0) is None
