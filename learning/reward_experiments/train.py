"""Train an independent shallow-press PPO ablation from the original defaults."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from learning.common import DEFAULT_ENVIRONMENT, make_env, task_kwargs
from learning.train_ppo import Config
from learning.reward_experiments.diagnostics import RewardDiagnosticsCallback
from learning.reward_experiments.shallow_press import ShallowPressReward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cfg = Config(DEFAULT_ENVIRONMENT, 20000, args.seed, str(args.output_dir.resolve()))
    run = args.output_dir / f"{cfg.environment}-seed{cfg.seed}"
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps({
        **asdict(cfg), "task_kwargs": task_kwargs(), "torch_threads": 1,
        "reward_diagnostics": True, "reward_variant": "shallow_press",
        "shallow_press_eta": args.eta, "fingering_margin_m": 0.1,
        "energy_penalty_coef": 0.005,
    }, indent=2), encoding="utf-8")
    train = DummyVecEnv([lambda: Monitor(ShallowPressReward(make_env(cfg.environment, cfg.seed), args.eta))])
    train = VecNormalize(train, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=cfg.gamma)
    evaluate = DummyVecEnv([lambda: Monitor(ShallowPressReward(make_env(cfg.environment, cfg.seed + 10000), args.eta, diagnostics=False))])
    evaluate = VecNormalize(evaluate, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=cfg.gamma)
    callbacks = [
        CheckpointCallback(save_freq=cfg.n_steps * 4, save_path=str(run / "checkpoints"), name_prefix="ppo", save_vecnormalize=True),
        EvalCallback(evaluate, best_model_save_path=str(run / "best"), log_path=str(run / "eval"), eval_freq=cfg.n_steps * 4, n_eval_episodes=1, deterministic=True),
        RewardDiagnosticsCallback(run / "training_reward_diagnostics.csv"),
    ]
    model = PPO(
        "MlpPolicy", train, learning_rate=cfg.learning_rate, n_steps=cfg.n_steps,
        batch_size=cfg.batch_size, n_epochs=cfg.n_epochs, gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda, clip_range=cfg.clip_range, target_kl=cfg.target_kl,
        policy_kwargs={"net_arch": [256, 256]}, tensorboard_log=str(run / "tensorboard"),
        seed=cfg.seed, verbose=1, device="auto",
    )
    try:
        model.learn(total_timesteps=cfg.total_timesteps, callback=callbacks, progress_bar=False)
        model.save(run / "ppo_final")
        train.save(run / "vecnormalize.pkl")
    finally:
        train.close()
        evaluate.close()
    print("Saved isolated shallow-press experiment:", run)


if __name__ == "__main__":
    main()
