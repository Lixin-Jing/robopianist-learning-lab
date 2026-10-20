# RoboPianist PPO 与 SAC 完整实验开发日志

## 1. 文档信息

- 项目：RoboPianist
- 主仓库：/Users/jing/Local_Project/robopianist
- SAC 参考仓库：/Users/jing/Local_Project/robopianist-rl
- Python 环境：Robot_learning
- 实验整理日期：2026-10-20
- 训练任务：RoboPianist-debug-TwinkleTwinkleLittleStar-v0
- 本轮原则：不修改 PPO/SAC 的网络、损失函数和既有训练超参数；复用已保存的 20k 训练产物，补齐可复现评估、检查点选择、多场景泛化、失败分桶、图表和结果归档。

## 2. 实验目标

此前二维 reaching + pushing 项目形成了如下实验闭环：

1. 建立固定场景基线；
2. 保存训练日志和模型；
3. 在固定场景复现结果；
4. 扩展到多场景评估；
5. 保存并分析失败案例；
6. 按场景难度分桶；
7. 检查训练过程和早停点；
8. 通过消融实验定位泛化问题；
9. 记录可复现命令、结果和结论。

本实验把相同方法映射到 RoboPianist：

| 二维 pushing 流程 | RoboPianist 对应流程 |
|---|---|
| 固定物体与目标位置 | 原速度、原音高、固定手位的《小星星》 |
| 随机初始状态 | 随机双手初始位置 |
| 大/小 y 偏移分桶 | 曲速、移调、手位扰动分桶 |
| 到达目标成功率 | 音符 precision、recall、F1 |
| 最终物体距离 | episode return 与 return/step |
| 失败轨迹图 | 低 F1 场景清单与场景对比图 |
| 早停分析 | 逐检查点评估 |
| 定向采样消融 | 后续曲速/移调/手位课程学习候选实验 |

## 3. 可比性控制

### 3.1 两种算法共用的环境设置

    n_steps_lookahead = 10
    trim_silence = true
    gravity_compensation = true
    reduced_action_space = true
    control_timestep = 0.05
    primitive_fingertip_collisions = true
    change_color_on_activation = true

两种算法均使用确定性评估动作，并使用 RoboPianist 自带的 MidiEvaluationWrapper 计算音乐指标。

### 3.2 PPO 设置

来源：training_runs/ppo/RoboPianist-debug-TwinkleTwinkleLittleStar-v0-seed0/config.json

    训练 seed = 0
    请求交互数 = 20,000
    实际交互数 = 20,224（PPO rollout 长度导致向上取整）
    learning_rate = 3e-4
    n_steps = 256
    batch_size = 64
    n_epochs = 10
    gamma = 0.99
    gae_lambda = 0.95
    clip_range = 0.2
    policy network = [256, 256]
    观察归一化 = 开启
    奖励归一化 = 训练时开启

### 3.3 SAC 设置

来源：/Users/jing/Local_Project/robopianist-rl/training_runs/SAC-local-20k-seed42-v2/config.json

    训练 seed = 42
    交互数 = 20,000
    随机预热 = 1,000
    batch_size = 256
    replay_capacity = 50,000
    discount = 0.8
    actor_lr = 3e-4
    critic_lr = 3e-4
    temperature_lr = 3e-4
    critic 数量 = 2
    hidden_dims = [256, 256, 256]
    activation = GELU
    critic dropout = 0.01
    critic layer norm = 开启
    target update tau = 0.005

### 3.4 重要限制

本比较只有一个 PPO 训练种子和一个 SAC 训练种子，因此属于本机基线实验，不是可以进行统计显著性推断的多种子论文级比较。PPO 与 SAC 还使用了不同的折扣系数和网络结构；这些是现有运行配置的一部分，本轮没有改动。

## 4. 阶段一：接口验证

运行：

    python -m pytest learning/test_gymnasium_env.py -q

结果：

    3 passed in 0.22s

验证内容：

1. RoboPianist 观察字典能够稳定展平为 Gymnasium 向量；
2. 正常 MIDI 结束被标记为 truncated=True；
3. discount 为 0 的失败结束被标记为 terminated=True；
4. 动作被裁剪到合法连续范围。

## 5. 阶段二：固定场景基准复现

### 5.1 PPO

命令：

    python -m learning.evaluate_ppo --episodes 3

三回合结果完全一致：

    return = 142.497
    precision = 0.627329
    recall = 0.043478
    note F1 = 0.012422
    sustain F1 = 0.894410
    episode length = 161

### 5.2 SAC

命令：

    cd /Users/jing/Local_Project/robopianist-rl
    python evaluate_checkpoint.py \
      training_runs/SAC-local-20k-seed42-v2/checkpoint_latest.msgpack \
      --episodes 3

三回合结果完全一致：

    return = 452.259
    precision = 0.731884
    recall = 0.447205
    note F1 = 0.410973
    sustain F1 = 1.000000
    episode length = 161

### 5.3 基准结论

在约 20k 交互预算下：

- SAC return 约为 PPO 的 3.17 倍；
- SAC note F1 约为 PPO 的 33.1 倍；
- PPO precision 不低，但 recall 只有 0.0435，说明策略很少正确触发目标音符；
- SAC 同时取得较高 precision 和 recall，已经学到更有效的按键行为。

## 6. 阶段三：检查点扫描与早停诊断

### 6.1 PPO 检查点

每个 PPO 模型都与同一步数保存的 VecNormalize 状态配对，避免使用错误的观察统计量。

| 步数 | Return | Note F1 |
|---:|---:|---:|
| 1,024 | 326.521 | 0.043478 |
| 2,048 | 304.752 | 0.043478 |
| 3,072 | 300.555 | 0.043478 |
| 4,096 | 285.936 | 0.043478 |
| 5,120 | 227.796 | 0.043478 |
| 6,144 | 236.495 | 0.043478 |
| 7,168 | 222.529 | 0.043478 |
| 8,192 | 228.050 | 0.043478 |
| 9,216 | 221.272 | 0.043478 |
| 10,240 | 210.344 | 0.037267 |
| 11,264 | 205.806 | 0.043478 |
| 12,288 | 187.920 | 0.037267 |
| 13,312 | 163.927 | 0.031056 |
| 14,336 | 161.801 | 0.037267 |
| 15,360 | 149.218 | 0.037267 |
| 16,384 | 142.328 | 0.024845 |
| 17,408 | 119.949 | 0.006211 |
| 18,432 | 160.545 | 0.031056 |
| 19,456 | 133.114 | 0.031056 |
| 最终模型 | 142.497 | 0.012422 |

诊断：

- PPO 的评估回报总体随训练推进而下降；
- 1,024 步检查点具有最高回报，但 note F1 仍只有 0.0435；
- 最终模型的 note F1 低于绝大多数早期检查点；
- 这不是“训练不够久”，而是当前 PPO 配置在该任务上出现训练退化；
- 后续若继续 PPO，应使用早停、多个训练种子，并检查奖励归一化、rollout 长度和超参数，而不是盲目延长训练。

### 6.2 SAC 检查点

| 步数 | Return | Note F1 |
|---:|---:|---:|
| 5,000 | 334.523 | 0.043478 |
| 10,000 | 450.917 | 0.097308 |
| 15,000 | 388.596 | 0.114907 |
| 20,000 | 452.259 | 0.410973 |

诊断：

- SAC 的性能不是严格单调，15k return 低于 10k；
- note F1 从 5k 到 20k 总体上升；
- 20k 同时取得最高 return 和最高 note F1，因此当前最终检查点选择合理；
- replay buffer 对交互数据的重复利用，在这个小预算实验中明显优于当前 PPO 基线。

## 7. 阶段四：多场景泛化协议

场景定义保存在 learning/experiment_scenarios.json。

| 场景 | stretch | shift | 随机手位 | 目的 |
|---|---:|---:|---|---|
| baseline | 1.0 | 0 | 否 | 训练分布内基准 |
| tempo_slow_1.2 | 1.2 | 0 | 否 | 慢速泛化 |
| tempo_fast_0.8 | 0.8 | 0 | 否 | 快速泛化 |
| pitch_down_2 | 1.0 | -2 | 否 | 降调泛化 |
| pitch_up_2 | 1.0 | +2 | 否 | 升调泛化 |
| random_hand_positions | 1.0 | 0 | 是 | 初始手位泛化 |
| combined_hard | 0.8 | +2 | 是 | 组合困难场景 |

每个场景使用评估种子 20000、20001、20002，每种算法共运行 21 回合。

固定手位场景的三个种子结果完全相同。原因是这些场景没有启用随机手位，种子不会产生实际初态差异。随机手位场景出现了明显方差，说明评估协议成功覆盖了不同初始状态。

## 8. 阶段五：多场景结果

表中为三回合均值；括号中仅在存在实际随机差异时给出标准差。

| 算法 | 场景 | Return | Return/step | Precision | Recall | Note F1 | Sustain F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| PPO | baseline | 142.497 | 0.885 | 0.627 | 0.043 | 0.012 | 0.894 |
| SAC | baseline | 452.259 | 2.809 | 0.732 | 0.447 | 0.411 | 1.000 |
| PPO | tempo_slow_1.2 | 192.019 | 0.995 | 0.617 | 0.041 | 0.021 | 0.927 |
| SAC | tempo_slow_1.2 | 554.032 | 2.871 | 0.780 | 0.404 | 0.372 | 1.000 |
| PPO | tempo_fast_0.8 | 105.016 | 0.814 | 0.605 | 0.054 | 0.031 | 0.938 |
| SAC | tempo_fast_0.8 | 375.372 | 2.910 | 0.846 | 0.519 | 0.460 | 1.000 |
| PPO | pitch_down_2 | 191.096 | 1.187 | 0.571 | 0.043 | 0.019 | 0.907 |
| SAC | pitch_down_2 | 417.640 | 2.594 | 0.966 | 0.161 | 0.153 | 1.000 |
| PPO | pitch_up_2 | 120.389 | 0.748 | 0.735 | 0.050 | 0.040 | 0.882 |
| SAC | pitch_up_2 | 432.117 | 2.684 | 0.902 | 0.236 | 0.216 | 1.000 |
| PPO | random_hand_positions | 159.566 ± 29.862 | 0.991 ± 0.185 | 0.714 ± 0.162 | 0.043 | 0.029 ± 0.012 | 0.903 ± 0.020 |
| SAC | random_hand_positions | 446.535 ± 28.466 | 2.774 ± 0.177 | 0.761 ± 0.107 | 0.321 ± 0.176 | 0.305 ± 0.174 | 1.000 |
| PPO | combined_hard | 101.569 ± 1.429 | 0.787 ± 0.011 | 0.782 ± 0.028 | 0.059 ± 0.007 | 0.046 ± 0.011 | 0.922 ± 0.033 |
| SAC | combined_hard | 337.540 ± 18.524 | 2.617 ± 0.144 | 0.908 ± 0.021 | 0.198 ± 0.068 | 0.181 ± 0.074 | 1.000 |

## 9. 阶段六：失败分桶与原因诊断

为每种算法分别把 note F1 最低的 25% 回合作为失败案例。结果位于：

    training_runs/full_experiment/failure_cases.csv

### 9.1 PPO 失败模式

PPO 的低 F1 阈值为 0.0186335。主要失败项：

- baseline 的三个回合；
- pitch_down_2 的三个回合；
- 随机手位 seed 20002。

PPO 的典型表现是 precision 尚可、recall 极低。这表示它倾向于少按键：按下时有一部分是正确的，但漏掉了绝大多数目标音符。由于回报还包含指法、前臂、能量和踏板等 shaping 项，较高 return 不一定对应较高 note F1。例如 pitch_down_2 的 return 高于 baseline，但 note F1 仍然很低。

### 9.2 SAC 失败模式

SAC 的低 F1 阈值为 0.157623。主要失败项：

- pitch_down_2 的三个回合；
- 随机手位 seed 20001；
- combined_hard 的 seed 20001 和 20002。

pitch_down_2 中 SAC precision 高达 0.966，但 recall 只有 0.161。这说明策略仍然较少误按，却无法覆盖移调后的多数目标音符。随机手位与组合场景的方差说明 SAC 对初始手位仍然敏感。

### 9.3 曲速结果

跨曲速比较累计 return 会受到回合长度影响：

- 慢速回合长度为 193；
- 基准回合长度为 161；
- 快速回合长度为 129。

因此必须同时查看 return/step。SAC 在快速场景中的 return/step 为 2.910，高于基准的 2.809；其 note F1 也从 0.411 上升至 0.460。当前策略对快速版本并未退化。慢速场景的 note F1 略降至 0.372。

## 10. 图表与原始数据

结果目录：

    training_runs/full_experiment/

主要文件：

    checkpoint_sweep.png
    scenario_f1.png
    scenario_episode_return.png
    scenario_return_per_step.png
    scenario_summary.csv
    scenario_summary.json
    failure_cases.csv
    ppo/scenario_results.csv
    ppo/scenario_results.json
    ppo/checkpoint_sweep.csv
    sac/scenario_results.csv
    sac/scenario_results.json
    sac/checkpoint_sweep.csv

说明：

- scenario_results.csv：逐回合原始结果；
- scenario_summary.csv：按算法和场景汇总的均值与标准差；
- failure_cases.csv：每种算法 note F1 最低 25% 的回合；
- checkpoint_sweep.png：相同评估口径下的训练进展；
- 三张 scenario 图片：场景级 return、return/step 和 note F1 对比。

## 11. 复现实验命令

### PPO 多场景评估

    cd /Users/jing/Local_Project/robopianist
    conda activate Robot_learning
    python -m learning.evaluate_ppo_scenarios

### PPO 检查点扫描

    python -m learning.evaluate_ppo_checkpoints

### SAC 多场景评估

    cd /Users/jing/Local_Project/robopianist-rl
    conda activate Robot_learning
    python evaluate_scenarios.py \
      training_runs/SAC-local-20k-seed42-v2/checkpoint_latest.msgpack

### SAC 检查点扫描

    python evaluate_checkpoint_sweep.py \
      training_runs/SAC-local-20k-seed42-v2

### 重新生成汇总和图表

    cd /Users/jing/Local_Project/robopianist
    python -m learning.build_full_experiment_report

## 12. 本轮结论

1. 现有 PPO 和 SAC 基准结果可以稳定复现；
2. SAC 在当前 20k 预算下显著优于 PPO；
3. PPO 在训练后期明显退化，当前最终模型不是理想检查点；
4. SAC 在 20k 检查点取得当前最佳结果；
5. SAC 对曲速变化较稳健，但对移调和随机手位仍然敏感；
6. PPO 的主要问题是极低 recall，而不是单纯 precision 不足；
7. return 与音乐准确率可能不一致，不能只用 return 判断钢琴策略；
8. 不同曲速必须同时报告 return/step；
9. 固定手位下重复不同种子不会产生真正不同场景；
10. 下一轮最有价值的实验是多训练种子、PPO 早停，以及加入移调和随机手位的课程学习；本轮遵守“不修改算法”的要求，没有启动这些训练变体。

## 13. 非阻塞警告

运行中出现两项依赖弃用警告：

- Python audioop 将在 Python 3.13 移除；
- TensorFlow Probability 使用的部分 JAX 映射接口已弃用。

本次所有命令均以退出码 0 完成，警告没有影响当前实验结果。
