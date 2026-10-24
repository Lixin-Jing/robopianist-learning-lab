"""Record PPO key-activation failures across scenarios and song quarters."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import dm_env_wrappers as wrappers
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from robopianist import suite
import robopianist.wrappers as robopianist_wrappers

from learning.common import DEFAULT_ENVIRONMENT, DEFAULT_RUN_DIR, make_env, task_kwargs
from learning.evaluate_ppo_scenarios import DEFAULT_SCENARIOS, load_scenarios
from learning.gymnasium_env import flatten_observation


def song_quarter(step: int, length: int) -> int:
    """Return a 1-based quarter index, including short final episodes."""
    if length <= 0 or not 0 <= step < length:
        raise ValueError("step must be inside a non-empty episode")
    return min(4, 4 * step // length + 1)


def key_counts(target: np.ndarray, actual: np.ndarray) -> tuple[int, int, int]:
    target = np.asarray(target, dtype=bool)
    actual = np.asarray(actual, dtype=bool)
    if target.shape != actual.shape:
        raise ValueError("target and actual key vectors must have equal shape")
    return (
        int(np.count_nonzero(target & actual)),
        int(np.count_nonzero(target & ~actual)),
        int(np.count_nonzero(~target & actual)),
    )


def hand_offset_bucket(offset_m: float) -> str:
    """Describe the magnitude of the shared initial hand displacement."""
    magnitude = abs(offset_m)
    if magnitude < 0.01:
        return "0-1cm"
    if magnitude < 0.025:
        return "1-2.5cm"
    if magnitude < 0.04:
        return "2.5-4cm"
    return "4-5cm"


def key_micro_f1(tp: int, fn: int, fp: int) -> float | None:
    """F1 over target/played key-time cells, without silent-step credit."""
    denominator = 2 * tp + fn + fp
    return 2 * tp / denominator if denominator else None


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_activations(
    target: np.ndarray,
    actual: np.ndarray,
    title: str,
    path: Path,
    control_timestep: float,
) -> None:
    """Show target keys, played keys, and false/missed activations over time."""
    active = np.flatnonzero(np.any(target | actual, axis=0))
    low = max(0, int(active.min()) - 2) if active.size else 0
    high = min(target.shape[1], int(active.max()) + 3) if active.size else target.shape[1]
    target_view = target[:, low:high].T.astype(float)
    actual_view = actual[:, low:high].T.astype(float)
    errors = np.zeros_like(target_view)
    errors[(target & ~actual)[:, low:high].T] = 1.0
    errors[(~target & actual)[:, low:high].T] = -1.0

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    extent = (0, target.shape[0] * control_timestep, low - 0.5, high - 0.5)
    axes[0].imshow(target_view, aspect="auto", origin="lower", interpolation="nearest", cmap="Greys", vmin=0, vmax=1, extent=extent)
    axes[0].set_ylabel("Target key")
    axes[1].imshow(actual_view, aspect="auto", origin="lower", interpolation="nearest", cmap="Greys", vmin=0, vmax=1, extent=extent)
    axes[1].set_ylabel("Activated key")
    axes[2].imshow(errors, aspect="auto", origin="lower", interpolation="nearest", cmap="bwr", vmin=-1, vmax=1, extent=extent)
    axes[2].set_ylabel("Key index")
    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_title("Red: missed target; blue: unwanted press")
    for axis in axes:
        for quarter in (1, 2, 3):
            axis.axvline(quarter * target.shape[0] * control_timestep / 4, color="tab:green", alpha=0.5, linewidth=0.8)
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate_episode(
    model: PPO,
    normalization: VecNormalize,
    environment_name: str,
    scenario: dict[str, object],
    training_seed: int,
    evaluation_seed: int,
) -> tuple[dict[str, object], list[dict[str, object]], np.ndarray, np.ndarray]:
    options = task_kwargs()
    options["randomize_hand_positions"] = bool(scenario["randomize_hand_positions"])
    environment = suite.load(
        environment_name=environment_name,
        seed=evaluation_seed,
        stretch=float(scenario["stretch"]),
        shift=int(scenario["shift"]),
        recompile_physics=False,
        task_kwargs=options,
    )
    environment = wrappers.EpisodeStatisticsWrapper(environment, deque_size=1)
    environment = robopianist_wrappers.MidiEvaluationWrapper(environment, deque_size=1)
    observation_keys = tuple(environment.observation_spec().keys())
    action_dtype = environment.action_spec().dtype
    targets: list[np.ndarray] = []
    actuals: list[np.ndarray] = []
    step_rows: list[dict[str, object]] = []
    episode_return = 0.0
    try:
        time_step = environment.reset()
        right_hand = environment.task.right_hand
        left_hand = environment.task.left_hand
        right_offset = (
            float(environment.physics.bind(right_hand.root_body).xpos[1])
            - float(right_hand.root_body.pos[1])
        )
        left_offset = (
            float(environment.physics.bind(left_hand.root_body).xpos[1])
            - float(left_hand.root_body.pos[1])
        )
        if not np.isclose(right_offset, left_offset, atol=1e-4):
            raise RuntimeError("The two hands have different initial offsets")
        initial_hand_offset_y_m = (right_offset + left_offset) / 2
        offset_bucket = hand_offset_bucket(initial_hand_offset_y_m)
        note_sequence = environment.task._notes
        length = len(note_sequence)
        n_keys = environment.task.piano.n_keys
        for step in range(length):
            if time_step.last():
                raise RuntimeError(f"Episode stopped early at step {step}/{length}")
            target = np.zeros(n_keys, dtype=bool)
            target[[note.key for note in note_sequence[step]]] = True
            observation = flatten_observation(time_step.observation, observation_keys)
            normalized = normalization.normalize_obs(observation[None, :])
            action, _ = model.predict(normalized, deterministic=True)
            time_step = environment.step(np.asarray(action[0], dtype=action_dtype))
            actual = np.asarray(environment.task.piano.activation, dtype=bool).copy()
            tp, fn, fp = key_counts(target, actual)
            episode_return += float(time_step.reward or 0.0)
            targets.append(target)
            actuals.append(actual)
            step_rows.append({
                "training_seed": training_seed,
                "scenario": scenario["name"],
                "evaluation_seed": evaluation_seed,
                "stretch": float(scenario["stretch"]),
                "shift": int(scenario["shift"]),
                "randomize_hand_positions": bool(scenario["randomize_hand_positions"]),
                "initial_hand_offset_y_m": initial_hand_offset_y_m,
                "hand_offset_bucket": offset_bucket,
                "step": step,
                "time_seconds": round(step * float(options["control_timestep"]), 4),
                "song_quarter": song_quarter(step, length),
                "target_keys": json.dumps(np.flatnonzero(target).tolist()),
                "activated_keys": json.dumps(np.flatnonzero(actual).tolist()),
                "target_key_count": int(target.sum()),
                "activated_key_count": int(actual.sum()),
                "true_positive_keys": tp,
                "missed_target_keys": fn,
                "unwanted_pressed_keys": fp,
            })
        if not time_step.last():
            raise RuntimeError(f"Episode failed to finish after {length} MIDI steps")
        metrics = environment.get_musical_metrics()
    finally:
        environment.close()
    target_array = np.stack(targets)
    actual_array = np.stack(actuals)
    tp_total = sum(int(row["true_positive_keys"]) for row in step_rows)
    fn_total = sum(int(row["missed_target_keys"]) for row in step_rows)
    fp_total = sum(int(row["unwanted_pressed_keys"]) for row in step_rows)
    summary = {
        "training_seed": training_seed,
        "scenario": scenario["name"],
        "evaluation_seed": evaluation_seed,
        "initial_hand_offset_y_m": initial_hand_offset_y_m,
        "hand_offset_bucket": offset_bucket,
        "episode_length": length,
        "full_song_completed": True,
        "episode_return": episode_return,
        "wrapper_f1": float(metrics["f1"]),
        "wrapper_recall": float(metrics["recall"]),
        "target_key_activations": tp_total + fn_total,
        "true_positive_keys": tp_total,
        "missed_target_keys": fn_total,
        "unwanted_pressed_keys": fp_total,
        "key_micro_recall": tp_total / (tp_total + fn_total) if tp_total + fn_total else None,
        "key_micro_f1": key_micro_f1(tp_total, fn_total, fp_total),
    }
    return summary, step_rows, target_array, actual_array


def segment_summaries(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["scenario"]), int(row["song_quarter"]))].append(row)
    summaries = []
    for (scenario, quarter), group in sorted(groups.items()):
        tp = sum(int(row["true_positive_keys"]) for row in group)
        fn = sum(int(row["missed_target_keys"]) for row in group)
        fp = sum(int(row["unwanted_pressed_keys"]) for row in group)
        summaries.append({
            "scenario": scenario,
            "song_quarter": quarter,
            "steps": len(group),
            "true_positive_keys": tp,
            "missed_target_keys": fn,
            "unwanted_pressed_keys": fp,
            "target_key_recall": tp / (tp + fn) if tp + fn else None,
            "key_micro_f1": key_micro_f1(tp, fn, fp),
            "miss_rate": fn / (tp + fn) if tp + fn else None,
            "unwanted_presses_per_step": fp / len(group),
        })
    return summaries


def hand_offset_summaries(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["scenario"] == "random_hand_positions":
            groups[str(row["hand_offset_bucket"])].append(row)
    summaries = []
    for bucket, group in sorted(groups.items()):
        tp = sum(int(row["true_positive_keys"]) for row in group)
        fn = sum(int(row["missed_target_keys"]) for row in group)
        summaries.append({
            "hand_offset_bucket": bucket,
            "episodes": len(group),
            "unique_evaluation_seeds": len({row["evaluation_seed"] for row in group}),
            "mean_absolute_offset_m": float(np.mean([abs(float(row["initial_hand_offset_y_m"])) for row in group])),
            "mean_wrapper_f1": float(np.mean([float(row["wrapper_f1"]) for row in group])),
            "target_key_recall": tp / (tp + fn) if tp + fn else None,
        })
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", default=DEFAULT_ENVIRONMENT)
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[20000, 20001, 20002])
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", default=[40000, 40001, 40002])
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--scenario-names", nargs="+", default=None)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    scenarios = load_scenarios(args.scenarios)
    if args.scenario_names is not None:
        scenarios = [scenario for scenario in scenarios if scenario["name"] in args.scenario_names]
        if not scenarios:
            parser.error("No named scenarios found in the scenario file")
    episode_rows: list[dict[str, object]] = []
    step_rows: list[dict[str, object]] = []
    for training_seed in args.training_seeds:
        run_dir = args.runs_dir / f"{args.environment}-seed{training_seed}"
        dummy = DummyVecEnv([lambda seed=training_seed: Monitor(make_env(args.environment, seed + 20000))])
        normalization = VecNormalize.load(run_dir / "vecnormalize.pkl", dummy)
        normalization.training = False
        normalization.norm_reward = False
        model = PPO.load(run_dir / "ppo_final")
        try:
            for scenario in scenarios:
                for evaluation_seed in args.evaluation_seeds:
                    episode, steps, target, actual = evaluate_episode(
                        model, normalization, args.environment, scenario,
                        training_seed, evaluation_seed,
                    )
                    episode_rows.append(episode)
                    step_rows.extend(steps)
                    if training_seed == args.training_seeds[0] and evaluation_seed == args.evaluation_seeds[0]:
                        plot_activations(
                            target, actual,
                            f"PPO seed {training_seed} · {scenario['name']} · eval {evaluation_seed}",
                            args.output_dir / "activation_plots" / f"{scenario['name']}.png",
                            float(task_kwargs()["control_timestep"]),
                        )
                    print(f"train={training_seed} scenario={scenario['name']} eval={evaluation_seed} f1={episode['wrapper_f1']:.4f} missed={episode['missed_target_keys']} fp={episode['unwanted_pressed_keys']}")
        finally:
            normalization.close()
    write_csv(args.output_dir / "episode_failures.csv", episode_rows)
    write_csv(args.output_dir / "key_activation_steps.csv", step_rows)
    write_csv(args.output_dir / "song_quarter_summary.csv", segment_summaries(step_rows))
    hand_rows = hand_offset_summaries(episode_rows)
    if hand_rows:
        write_csv(args.output_dir / "hand_offset_summary.csv", hand_rows)
    print(f"Saved {len(episode_rows)} full-song episodes to {args.output_dir}")


if __name__ == "__main__":
    main()
