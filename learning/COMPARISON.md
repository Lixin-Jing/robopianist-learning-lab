# PPO and SAC local comparison

Task: `RoboPianist-debug-TwinkleTwinkleLittleStar-v0`

Shared task settings: 10-step lookahead, silence trimming, gravity compensation,
reduced action space, 0.05 s control timestep, and primitive fingertip
collisions. Results use deterministic policies and three evaluation episodes.

| Method | Interactions | Return | Note precision | Note recall | Note F1 | Sustain F1 |
|---|---:|---:|---:|---:|---:|---:|
| PPO final | 20,224 | 142.497 | 0.6273 | 0.0435 | 0.0124 | 0.8944 |
| Official SAC | 1,000 | 124.426 | 1.0000 | 0.0435 | 0.0435 | 0.1553 |
| Official SAC | 10,000 | 450.917 | 0.9845 | 0.0994 | 0.0973 | 1.0000 |
| Official SAC | 20,000 | 452.259 | 0.7319 | 0.4472 | 0.4110 | 1.0000 |

At an approximately equal 20k interaction budget, SAC is clearly ahead: its
return is about 3.17 times PPO's, while note recall and note F1 are about 10.3
and 33.1 times higher respectively. The off-policy replay buffer lets SAC reuse
each interaction for gradient updates; PPO discards a rollout after its update
epochs. This is a useful local baseline, not evidence of general performance:
the task, MIDI, seed, and evaluation policy are fixed.

Re-run the 20k SAC comparison with:

```bash
cd /Users/jing/Local_Project/robopianist-rl
bash run_macos_small.sh
```

The authors' full recipe uses five million interactions and should be run on a
suitable GPU machine rather than treated as a quick local test.

## Artifacts

- PPO: `training_runs/ppo/RoboPianist-debug-TwinkleTwinkleLittleStar-v0-seed0/`
- SAC: `/Users/jing/Local_Project/robopianist-rl/training_runs/SAC-local-20k-seed42-v2/`
- SAC metrics: `metrics.jsonl`
- SAC model: `checkpoint_latest.msgpack`
