"""Train and evaluate a paired, three-seed PPO fingering-margin ablation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = "RoboPianist-debug-TwinkleTwinkleLittleStar-v0"
SEEDS = [20000, 20001, 20002]
EVAL_SEEDS = [40000, 40001, 40002]
GROUPS = {"margin_0p1": 0.1, "margin_0p3": 0.3, "margin_0p5": 0.5}


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def train_one(output, group, margin, seed, timesteps):
    # Keep physics/framework imports out of worker threads on macOS.
    run = output / "models" / group / f"{ENVIRONMENT}-seed{seed}"
    expected = {
        "seed": seed, "fingering_reward_margin": margin,
        "total_timesteps": timesteps, "torch_threads": 1,
        "reward_diagnostics": True,
    }
    if (run / "ppo_final.zip").exists() and (run / "vecnormalize.pkl").exists():
        config = json.loads((run / "config.json").read_text())
        if not all(config.get(k) == v for k, v in expected.items()):
            raise ValueError(f"Existing run configuration differs: {run}")
        return {"group": group, "seed": seed, "status": "reused_completed_run"}
    run.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-B", "-m", "learning.train_ppo",
        "--seed", str(seed), "--total-timesteps", str(timesteps),
        "--fingering-reward-margin", str(margin), "--reward-diagnostics",
        "--torch-threads", "1", "--output-dir", str(output / "models" / group),
    ]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
    with (run / "training_console.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    return {"group": group, "seed": seed, "status": "trained"}


def evaluate(output, group):
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from learning.common import DEFAULT_ENVIRONMENT, make_env, task_kwargs
    from learning.diagnose_ppo_failures import evaluate_episode, plot_activations, write_csv, segment_summaries
    from learning.evaluate_ppo_scenarios import DEFAULT_SCENARIOS, load_scenarios

    dest = output / "evaluation" / group
    dest.mkdir(parents=True, exist_ok=True)
    completed = dest / "complete.json"
    if completed.exists():
        with (dest / "episode_results.csv").open() as file:
            rows = list(csv.DictReader(file))
        if len(rows) != 63:
            raise ValueError("Completed evaluation must contain 63 episodes")
        return
    episodes, all_steps = [], []
    for seed in SEEDS:
        run = output / "models" / group / f"{DEFAULT_ENVIRONMENT}-seed{seed}"
        dummy = DummyVecEnv([lambda: Monitor(make_env(DEFAULT_ENVIRONMENT, 40000))])
        norm = VecNormalize.load(run / "vecnormalize.pkl", dummy)
        norm.training = False
        norm.norm_reward = False
        model = PPO.load(run / "ppo_final.zip", device="cpu")
        assert model.num_timesteps == 20224, model.num_timesteps
        try:
            for scenario in load_scenarios(DEFAULT_SCENARIOS):
                for eval_seed in EVAL_SEEDS:
                    episode, steps, target, actual = evaluate_episode(
                        model, norm, DEFAULT_ENVIRONMENT, scenario, seed, eval_seed,
                        collect_reward_diagnostics=True,
                    )
                    count = sum(s["finger_target_count"] for s in steps)
                    key_count = sum(s["target_key_count"] for s in steps)
                    episode.update({
                        "group": group,
                        "training_margin_m": GROUPS[group],
                        "evaluation_margin_m": 0.1,
                        "actual_training_timesteps": model.num_timesteps,
                        "finger_distance_mean_m": sum(s["finger_distance_sum_m"] for s in steps) / max(count, 1),
                        "finger_within_1cm_fraction": sum(s["fingers_within_1cm"] for s in steps) / max(count, 1),
                        "finger_within_5cm_fraction": sum(s["fingers_within_5cm"] for s in steps) / max(count, 1),
                        "target_depression_mean": sum(s["target_depression_sum"] for s in steps) / max(key_count, 1),
                        **{f"{k}_sum": sum(s[k] for s in steps) for k in steps[0] if k.startswith("reward_")},
                    })
                    assert abs(episode["episode_return"] - sum(s["original_reward"] for s in steps)) < 1e-8
                    episodes.append(episode)
                    all_steps.extend(steps)
                    if seed == SEEDS[0] and eval_seed == EVAL_SEEDS[0]:
                        plot_activations(target, actual,
                            f"{group} | train {seed} | eval {eval_seed} | {scenario['name']}",
                            dest / "activation_plots" / f"{scenario['name']}.png",
                            float(task_kwargs()["control_timestep"]),
                        )
                    print(f"EVAL {group} seed={seed} {scenario['name']} eval={eval_seed} TP={episode['true_positive_keys']} microF1={episode['key_micro_f1']:.6f}", flush=True)
        finally:
            norm.close()
    write_csv(dest / "episode_results.csv", episodes)
    write_csv(dest / "key_activation_steps.csv", all_steps)
    write_csv(dest / "song_quarter_summary.csv", segment_summaries(all_steps))
    save_json(completed, {"episodes": len(episodes), "full_songs": all(e["full_song_completed"] for e in episodes), "evaluation_reward": "original_margin_0.1"})


def summarize(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = ["wrapper_f1", "key_micro_recall", "key_micro_f1", "episode_return", "true_positive_keys", "missed_target_keys", "unwanted_pressed_keys", "finger_distance_mean_m", "finger_within_1cm_fraction", "finger_within_5cm_fraction", "target_depression_mean"]
    all_rows = []
    for group in GROUPS:
        with (output / "evaluation" / group / "episode_results.csv").open() as file:
            for row in csv.DictReader(file):
                for name in metrics: row[name] = float(row[name])
                row["training_seed"] = int(row["training_seed"])
                all_rows.append(row)
    seed_rows, summary = [], []
    for group in GROUPS:
        for scope in ["baseline", "seven_scenario_mean"]:
            vals = []
            for seed in SEEDS:
                rs = [r for r in all_rows if r["group"] == group and r["training_seed"] == seed and (scope != "baseline" or r["scenario"] == "baseline")]
                scenarios = sorted({r["scenario"] for r in rs})
                value = {k: float(np.mean([np.mean([r[k] for r in rs if r["scenario"] == scenario]) for scenario in scenarios])) for k in metrics}
                vals.append(value)
                seed_rows.append({"group": group, "scope": scope, "training_seed": seed, **value})
            summary.append({"group": group, "scope": scope, **{f"{k}_{stat}": float(fn([v[k] for v in vals])) for k in metrics for stat,fn in [("mean",np.mean),("sample_sd",lambda x: np.std(x,ddof=1))]}})
    from learning.diagnose_ppo_failures import write_csv
    write_csv(output / "per_seed_summary.csv", seed_rows)
    write_csv(output / "group_summary.csv", summary)
    scenario_rows = []
    for group in GROUPS:
        for scenario in sorted({r["scenario"] for r in all_rows}):
            seed_metrics = [{k: np.mean([r[k] for r in all_rows if r["group"] == group and r["scenario"] == scenario and r["training_seed"] == seed]) for k in metrics} for seed in SEEDS]
            scenario_rows.append({"group": group, "scenario": scenario, **{f"{k}_{stat}": float(fn([v[k] for v in seed_metrics])) for k in metrics for stat,fn in [("mean",np.mean),("sample_sd",lambda x: np.std(x,ddof=1))]}})
    write_csv(output / "scenario_summary.csv", scenario_rows)
    output.joinpath("figures").mkdir(exist_ok=True)
    for scope in ["baseline", "seven_scenario_mean"]:
        rs = [r for r in summary if r["scope"] == scope]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
        for ax, metric, title in zip(axes, ["key_micro_recall", "key_micro_f1", "finger_distance_mean_m"], ["Target-key recall", "Key-time micro F1", "Assigned finger distance (m)"]):
            xs = np.arange(3)
            ax.bar(xs, [r[f"{metric}_mean"] for r in rs], yerr=[r[f"{metric}_sample_sd"] for r in rs], capsize=4, color=["#6b7280", "#16a085", "#e69f00"], alpha=.75)
            for i,r in enumerate(rs):
                pts = [v[metric] for v in seed_rows if v["group"] == r["group"] and v["scope"] == scope]
                ax.scatter(i + np.linspace(-.12,.12,3), pts, color="black",s=18,zorder=3)
            ax.set_xticks(xs,["0.10 m", "0.30 m", "0.50 m"])
            ax.set_title(title)
            ax.set_xlabel("Training fingering reward margin")
            ax.set_ylim(bottom=0)
        fig.suptitle(f"{scope}: 3 independent training seeds; bars = mean +/- sample SD")
        fig.savefig(output / "figures" / f"{scope}.png", dpi=160)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for group in GROUPS:
        series=[]
        for seed in SEEDS:
            from learning.common import DEFAULT_ENVIRONMENT
            path=output/"models"/group/f"{DEFAULT_ENVIRONMENT}-seed{seed}"/"training_reward_diagnostics.csv"
            with path.open() as f: series.append(list(csv.DictReader(f)))
        for ax,metric in zip(axes,["finger_distance_mean_m","target_recall"]):
            y=np.asarray([[float(r[metric]) for r in rows] for rows in series])
            x=[int(r["timesteps"]) for r in series[0]]
            ax.plot(x,y.mean(axis=0),label=group)
            ax.fill_between(x,y.min(axis=0),y.max(axis=0),alpha=.15)
            ax.set_xlabel("Environment steps")
            ax.set_ylabel(metric)
    axes[0].legend()
    fig.suptitle("Training: mean across 3 seeds; shaded = min/max, not confidence interval")
    fig.savefig(output/"figures"/"training_diagnostics.png",dpi=160)
    plt.close(fig)
    save_json(output / "results.json", {"groups":summary,"per_seed":seed_rows,"episodes":len(all_rows),"evaluation_margin":0.1})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=3)
    parser.add_argument("--total-timesteps",type=int,default=20000,choices=[20000])
    args=parser.parse_args()
    output=args.output_dir.resolve()
    output.mkdir(parents=True,exist_ok=True)
    manifest={"created_at":datetime.now().astimezone().isoformat(),"groups":GROUPS,"training_seeds":SEEDS,"evaluation_seeds":EVAL_SEEDS,"requested_timesteps":args.total_timesteps,"actual_timesteps":20224,"torch_threads":1,"evaluation_margin":0.1,"selection":"all final models; no best-seed selection","design":"freshly train all 9 models; only reward margin differs between paired groups"}
    if not (output/"manifest.json").exists():save_json(output/"manifest.json",manifest)
    status=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs=[pool.submit(train_one,output,group,margin,seed,args.total_timesteps) for group,margin in GROUPS.items() for seed in SEEDS]
        for future in as_completed(jobs):
            result=future.result()
            status.append(result)
            save_json(output/"training_status.json",status)
            print("TRAIN_COMPLETE",json.dumps(result),flush=True)
    # Catch accidental hyperparameter differences before evaluating.
    from learning.common import DEFAULT_ENVIRONMENT
    configs=[]
    for group in GROUPS:
        for seed in SEEDS:
            c=json.loads((output/"models"/group/f"{DEFAULT_ENVIRONMENT}-seed{seed}"/"config.json").read_text())
            for k in ["seed","output_dir","fingering_reward_margin"]:c.pop(k)
            c["task_kwargs"].pop("fingering_reward_margin")
            configs.append(c)
    assert all(c==configs[0] for c in configs), "Non-margin configurations differ"
    for group in GROUPS:evaluate(output,group)
    summarize(output)
    save_json(output/"complete.json",{"finished_at":datetime.now().astimezone().isoformat(),"models":9,"evaluations":189})
    print("EXPERIMENT_COMPLETE",output,flush=True)


if __name__=="__main__":main()
