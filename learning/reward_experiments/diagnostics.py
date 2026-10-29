"""Shared read-only physical/reward diagnostics for independent ablations."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import numpy as np
from dm_control.utils.rewards import tolerance
from stable_baselines3.common.callbacks import BaseCallback


def measure_reward_state(task, physics) -> dict[str, float]:
    """Measure the completed control step using its cached, current MIDI goal."""
    if task._disable_fingering_reward:
        raise ValueError("This experiment requires the annotated fingering reward")
    distances = []
    for hand, keys in [
        (task.right_hand, task._rh_keys_current),
        (task.left_hand, task._lh_keys_current),
    ]:
        for key, finger in keys:
            geom = task.piano.keys[key].geom[0]
            key_pos = physics.bind(geom).xpos.copy()
            key_pos[-1] += 0.5 * physics.bind(geom).size[2]
            key_pos[0] += 0.35 * physics.bind(geom).size[0]
            tip_pos = physics.bind(hand.fingertip_sites[finger]).xpos
            distances.append(float(np.linalg.norm(key_pos - tip_pos)))
    distances = np.asarray(distances, dtype=float)
    goal = np.asarray(task._goal_current[:-1], dtype=bool)
    actual = np.asarray(task.piano.activation, dtype=bool)
    q = np.clip(task.piano.state / task.piano._qpos_range[:, 1], 0, 1)[goal]
    terms = {f"reward_{k}": float(v) for k, v in task.reward_fn.reward_terms.items()}
    legacy_fingering = float(np.mean(tolerance(
        distances, bounds=(0, 0.01), margin=0.1, sigmoid="gaussian"
    ))) if distances.size else 0.0
    return {
        **terms,
        "original_reward": sum(terms.values()) - terms["reward_fingering_reward"] + legacy_fingering,
        "finger_distance_sum_m": float(distances.sum()),
        "finger_target_count": int(distances.size),
        "fingers_within_1cm": int(np.sum(distances <= 0.01)),
        "fingers_within_5cm": int(np.sum(distances <= 0.05)),
        "target_depression_sum": float(q.sum()),
        "target_key_count": int(goal.sum()),
        "targets_depressed_ge_0p1": int(np.sum(q >= 0.1)),
        "true_positive_keys": int(np.sum(goal & actual)),
        "missed_target_keys": int(np.sum(goal & ~actual)),
        "unwanted_pressed_keys": int(np.sum(~goal & actual)),
    }


class RewardDiagnosticsWrapper(gym.Wrapper):
    """Capture diagnostics before vectorized auto-reset replaces the last state."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        dm = self.env.unwrapped.dm_env
        info["reward_diagnostics"] = measure_reward_state(dm.task, dm.physics)
        return obs, reward, terminated, truncated, info


class RewardDiagnosticsCallback(BaseCallback):
    """Write one aggregate per rollout, without changing observations or RNGs."""

    def __init__(self, path: Path):
        super().__init__()
        self.path = path
        self.sums = defaultdict(float)
        self.steps = 0
        self.rows = []

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            for name, value in info["reward_diagnostics"].items():
                self.sums[name] += float(value)
            self.steps += 1
        actions = self.locals["actions"]
        space = self.training_env.action_space
        self.sums["clipped_action_components"] += int(np.sum(
            (actions < space.low) | (actions > space.high)
        ))
        self.sums["action_components"] += actions.size
        return True

    def _on_rollout_end(self) -> None:
        s = self.sums
        tp, fn, fp = (s[k] for k in ("true_positive_keys", "missed_target_keys", "unwanted_pressed_keys"))
        row = {
            "timesteps": self.num_timesteps,
            "steps": self.steps,
            "finger_distance_mean_m": s["finger_distance_sum_m"] / max(s["finger_target_count"], 1),
            "finger_within_1cm_fraction": s["fingers_within_1cm"] / max(s["finger_target_count"], 1),
            "finger_within_5cm_fraction": s["fingers_within_5cm"] / max(s["finger_target_count"], 1),
            "target_depression_mean": s["target_depression_sum"] / max(s["target_key_count"], 1),
            "target_depression_ge_0p1_fraction": s["targets_depressed_ge_0p1"] / max(s["target_key_count"], 1),
            "target_recall": tp / max(tp + fn, 1),
            "key_micro_f1": 2 * tp / max(2 * tp + fn + fp, 1),
            "true_positive_keys": tp,
            "missed_target_keys": fn,
            "unwanted_pressed_keys": fp,
            "action_clip_fraction": s["clipped_action_components"] / max(s["action_components"], 1),
            "original_reward_per_step": s["original_reward"] / self.steps,
            **{f"{k}_per_step": v / self.steps for k, v in s.items() if k.startswith("reward_")},
        }
        self.rows.append(row)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(self.rows)
        self.sums.clear()
        self.steps = 0
