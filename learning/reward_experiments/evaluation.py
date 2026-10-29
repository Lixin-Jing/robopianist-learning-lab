"""Common ORIGINAL-reward evaluation for independent reward interventions."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from robopianist import suite
from robopianist.wrappers import MidiEvaluationWrapper
from learning.common import DEFAULT_ENVIRONMENT, make_env, task_kwargs
from learning.gymnasium_env import flatten_observation
from learning.diagnose_ppo_failures import key_counts, key_micro_f1, song_quarter, plot_activations, write_csv, segment_summaries
from learning.evaluate_ppo_scenarios import DEFAULT_SCENARIOS, load_scenarios
from learning.reward_experiments.diagnostics import measure_reward_state

SEEDS = [20000, 20001, 20002]
EVAL_SEEDS = [40000, 40001, 40002]


def evaluate_group(models_root: Path, output: Path, group: str):
    import torch
    torch.set_num_threads(1)
    dest = output / "evaluation" / group
    dest.mkdir(parents=True, exist_ok=True)
    episodes, steps_all = [], []
    for seed in SEEDS:
        run = models_root / f"{DEFAULT_ENVIRONMENT}-seed{seed}"
        norm = VecNormalize.load(run / "vecnormalize.pkl", DummyVecEnv([lambda: Monitor(make_env(DEFAULT_ENVIRONMENT, 40000))]))
        norm.training = False
        norm.norm_reward = False
        model = PPO.load(run / "ppo_final.zip", device="cpu")
        assert model.num_timesteps == 20224
        try:
            for scenario in load_scenarios(DEFAULT_SCENARIOS):
                for eval_seed in EVAL_SEEDS:
                    options = task_kwargs()
                    options["randomize_hand_positions"] = bool(scenario["randomize_hand_positions"])
                    env = suite.load(environment_name=DEFAULT_ENVIRONMENT, seed=eval_seed,
                        stretch=float(scenario["stretch"]), shift=int(scenario["shift"]),
                        recompile_physics=False, task_kwargs=options)
                    env = MidiEvaluationWrapper(env, deque_size=1)
                    local_steps, targets, actuals = [], [], []
                    total = 0.0
                    try:
                        ts = env.reset()
                        task = env.task
                        assert not hasattr(task, "_fingering_reward_margin")
                        assert task._energy_penalty_coef == 0.005
                        keys = tuple(env.observation_spec().keys())
                        length = len(task._notes)
                        for step in range(length):
                            assert not ts.last()
                            target = np.zeros(task.piano.n_keys, dtype=bool)
                            target[[note.key for note in task._notes[step]]] = True
                            obs = flatten_observation(ts.observation, keys)
                            action, _ = model.predict(norm.normalize_obs(obs[None, :]), deterministic=True)
                            ts = env.step(np.asarray(action[0], dtype=env.action_spec().dtype))
                            actual = task.piano.activation.astype(bool).copy()
                            np.testing.assert_array_equal(target, task._goal_current[:-1].astype(bool))
                            tp, fn, fp = key_counts(target, actual)
                            total += float(ts.reward or 0)
                            measurement = measure_reward_state(task, env.physics)
                            assert abs(measurement["original_reward"] - float(ts.reward or 0)) < 1e-10
                            local_steps.append({
                                "training_seed": seed, "scenario": scenario["name"],
                                "evaluation_seed": eval_seed, "step": step,
                                "time_seconds": round(step * 0.05, 4),
                                "song_quarter": song_quarter(step, length),
                                "target_keys": str(np.flatnonzero(target).tolist()),
                                "activated_keys": str(np.flatnonzero(actual).tolist()),
                                **measurement,
                            })
                            targets.append(target)
                            actuals.append(actual)
                        assert ts.last() and ts.discount == 1
                        metrics = env.get_musical_metrics()
                    finally:
                        env.close()
                    tp, fn, fp = [sum(r[k] for r in local_steps) for k in ["true_positive_keys", "missed_target_keys", "unwanted_pressed_keys"]]
                    count = sum(r["finger_target_count"] for r in local_steps)
                    key_count = tp + fn
                    episodes.append({
                        "group": group, "training_seed": seed,
                        "scenario": scenario["name"], "evaluation_seed": eval_seed,
                        "episode_length": length, "full_song_completed": True,
                        "actual_training_timesteps": model.num_timesteps,
                        "episode_return": total, "wrapper_f1": metrics["f1"],
                        "wrapper_recall": metrics["recall"],
                        "true_positive_keys": tp, "missed_target_keys": fn,
                        "unwanted_pressed_keys": fp, "target_key_activations": key_count,
                        "key_micro_recall": tp / max(key_count, 1),
                        "key_micro_f1": key_micro_f1(tp, fn, fp),
                        "finger_distance_mean_m": sum(r["finger_distance_sum_m"] for r in local_steps) / max(count, 1),
                        "finger_within_1cm_fraction": sum(r["fingers_within_1cm"] for r in local_steps) / max(count, 1),
                        "finger_within_5cm_fraction": sum(r["fingers_within_5cm"] for r in local_steps) / max(count, 1),
                        "target_depression_mean": sum(r["target_depression_sum"] for r in local_steps) / max(key_count, 1),
                        **{f"{k}_sum": sum(r[k] for r in local_steps) for k in local_steps[0] if k.startswith("reward_")},
                    })
                    steps_all.extend(local_steps)
                    if seed == SEEDS[0] and eval_seed == EVAL_SEEDS[0]:
                        plot_activations(np.stack(targets), np.stack(actuals),
                            f"{group} | train {seed} | eval {eval_seed} | {scenario['name']}",
                            dest / "activation_plots" / f"{scenario['name']}.png", 0.05)
                    print(f"EVAL {group} train={seed} {scenario['name']} eval={eval_seed} TP={tp} F1={episodes[-1]['key_micro_f1']:.6f}", flush=True)
        finally:
            norm.close()
    assert len(episodes) == 63
    write_csv(dest / "episode_results.csv", episodes)
    write_csv(dest / "key_activation_steps.csv", steps_all)
    write_csv(dest / "song_quarter_summary.csv", segment_summaries(steps_all))
    return episodes


def verify_baseline(previous_csv: Path, current_csv: Path):
    def read(path):
        with path.open() as f:
            return {(r["training_seed"], r["scenario"], r["evaluation_seed"]): r for r in csv.DictReader(f)}
    old, new = read(previous_csv), read(current_csv)
    assert old.keys() == new.keys()
    fields = ["episode_return", "wrapper_f1", "wrapper_recall", "true_positive_keys", "missed_target_keys", "unwanted_pressed_keys", "key_micro_recall", "key_micro_f1", "finger_distance_mean_m", "target_depression_mean"]
    for key in old:
        for field in fields:
            assert abs(float(old[key][field]) - float(new[key][field])) < 1e-8, (key, field)
    return {"compared_episodes": len(old), "fields": fields, "all_match": True}
