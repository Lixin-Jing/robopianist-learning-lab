# PPO baseline and official SAC comparison

These scripts provide a small PPO baseline using Gymnasium and
Stable-Baselines3. The task parameters mirror the public RoboPianist SAC recipe
where possible, so returns can be compared on the same debug MIDI task.

Run commands from the RoboPianist repository root with the `Robot_learning`
Conda environment selected.

```bash
python -m pip install -r learning/requirements.txt
```

## 1. Check the adapter

```bash
python -m pytest learning/test_gymnasium_env.py -q
```

## 2. Smoke-test PPO

```bash
python -m learning.train_ppo --total-timesteps 512
```

This checks the complete data path. It is not long enough to learn a song.

## 3. Train a small baseline

```bash
python -m learning.train_ppo --total-timesteps 20000
```

Models, normalization statistics, checkpoints, evaluation logs, and the exact
configuration are stored under `training_runs/ppo/`.

## 4. Evaluate or watch the policy

```bash
python -m learning.evaluate_ppo --episodes 3
python -m learning.play_ppo
```

Evaluation reports both episode return and the same MIDI precision, recall, and
F1 metrics used by the authors' SAC implementation.

## PPO quantities in the code

- `gamma`: discount factor in the return target.
- `gae_lambda`: bias/variance trade-off in generalized advantage estimation.
- `clip_range`: epsilon in the clipped PPO surrogate objective.
- `n_steps`: rollout length before one optimization phase.
- `n_epochs`: passes over each rollout during that optimization phase.

The normal end of a MIDI sequence has dm_env discount 1 and is represented as
`truncated=True`; a failure with discount 0 is represented as
`terminated=True`. This preserves the correct PPO bootstrap behavior.

## Official SAC reference

The authors' reference implementation lives in the sibling repository
`robopianist-rl`. Its `run.sh` is a five-million-step experiment, not a quick
smoke test. First verify it with `python train.py --help`; then start with a
small `--max-steps` value before launching the full recipe.
