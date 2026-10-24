"""Tests for the episode-level full-song tempo sampling rule."""

import numpy as np
import pytest

from learning.tempo_curriculum import TempoCurriculum


class DummyMidi:
    def __init__(self, factor: float = 1.0) -> None:
        self.factor = factor

    def stretch(self, factor: float) -> "DummyMidi":
        return DummyMidi(factor)


def test_warmup_then_slow_sampling() -> None:
    curriculum = TempoCurriculum(warmup_episodes=2, slow_probability=1.0)
    midi = DummyMidi()
    rng = np.random.RandomState(4)
    assert curriculum(initial_value=midi, random_state=rng) is midi
    assert curriculum(initial_value=midi, random_state=rng) is midi
    assert curriculum(initial_value=midi, random_state=rng).factor == 1.2
    assert [row["stage"] for row in curriculum.history] == [
        "warmup", "warmup", "targeted_sampling"
    ]


@pytest.mark.parametrize("probability", [-0.1, 1.1])
def test_probability_validation(probability: float) -> None:
    with pytest.raises(ValueError):
        TempoCurriculum(slow_probability=probability)
