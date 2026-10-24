"""Episode-level tempo curriculum for the full RoboPianist MIDI task."""

from __future__ import annotations

import numpy as np


class TempoCurriculum:
    """Keep full songs: warm up at normal tempo, then oversample slow songs."""

    def __init__(
        self,
        warmup_episodes: int = 32,
        slow_probability: float = 0.7,
        slow_stretch: float = 1.2,
    ) -> None:
        if warmup_episodes < 0:
            raise ValueError("warmup_episodes must not be negative")
        if not 0 <= slow_probability <= 1:
            raise ValueError("slow_probability must be between 0 and 1")
        if slow_stretch <= 1:
            raise ValueError("slow_stretch must be greater than 1")
        self.warmup_episodes = warmup_episodes
        self.slow_probability = slow_probability
        self.slow_stretch = slow_stretch
        self.history: list[dict[str, object]] = []

    def __call__(self, initial_value=None, current_value=None, random_state=None):
        del current_value
        if initial_value is None or not hasattr(initial_value, "stretch"):
            raise ValueError("TempoCurriculum requires a MIDI file with stretch()")
        rng = random_state if random_state is not None else np.random
        episode = len(self.history) + 1
        warmup = episode <= self.warmup_episodes
        slow = not warmup and rng.uniform(0.0, 1.0) < self.slow_probability
        factor = self.slow_stretch if slow else 1.0
        self.history.append({
            "episode": episode,
            "stage": "warmup" if warmup else "targeted_sampling",
            "stretch": factor,
        })
        return initial_value if factor == 1.0 else initial_value.stretch(factor)
