# SPO 接入说明

## 1. 可以把它理解成换一张“训练评分表”

飞机仍按原流程观察、选择动作、仿真飞行、收集经验。更新策略网络时，根据评分表决定参数往哪里调整：

- `clip`：原来的 PPO-Clip 评分，更新超过对应边界后，该样本的目标变平。
- `spo`：SPO 评分，增加“概率比变化过大”的二次扣分，更新过头时提供纠偏梯度。

SPO 只改变训练时的策略损失，GRU 和时序注意力都可以使用。推理时仍加载相同结构的 actor。

## 2. 代码是怎么接上的

1. `config.py` 增加 `--policy-loss clip/spo`，默认 `clip`。
2. `algorithms/ppo/ppo_trainer.py` 根据开关选择目标。新旧策略对完整动作的联合概率比为 `ratio`，优势为 `advantage`，SPO 最小化：

   ```python
   -(ratio * advantage - advantage.abs() / (2 * epsilon) * (ratio - 1).square()).mean()
   ```

3. `epsilon` 复用 `--clip-param`，默认 `0.2`，SPO 要求它为有限正数。这里没有同时叠加 Clip 与 SPO。
4. 采样、标准化优势、GAE、价值损失、熵项、优化器和网络结构沿用现有配置。完整动作的 log probability 已由动作层相加，trainer 对每个样本只算一个概率比。
5. `scripts/train/train_jsbsim.py` 拒绝把 SPO 用到 MAPPO / MultipleCombat，避免参数被静默忽略。

公式来源：Xie 等，*Simple Policy Optimization*，ICML 2025，公式（16）。[会议论文](https://proceedings.mlr.press/v267/xie25m.html)。这是已有目标函数的项目适配。

## 3. 怎么用

在项目根目录、已安装项目依赖的 Python 环境中运行。保留原来的 `--algorithm-name ppo`，通过 `--policy-loss` 切换评分方式。

以下是 GRU + SPO 的短训练命令，用来确认入口能运行；它不足以得出算法优劣结论：

```powershell
python scripts/train/train_jsbsim.py --env-name SingleCombat --algorithm-name ppo --scenario-name 1v1/NoWeapon/Selfplay --use-selfplay --selfplay-algorithm fsp --memory-encoder gru --policy-loss spo --clip-param 0.2 --experiment-name gru_spo --n-rollout-threads 1 --buffer-size 32 --data-chunk-length 8 --num-mini-batch 1 --ppo-epoch 2 --num-env-steps 64 --seed 1
```

短训练可能尚未结束完整对局；现有平均回合奖励统计此时会除零并显示 `inf`，正式评估需等待完整回合。

复制这条命令，只替换下面两项，就得到四组对照；其他训练参数及种子保持一致：

| 组别 | 编码器参数 | 策略目标参数 | 实验名建议 |
| --- | --- | --- | --- |
| GRU + PPO | `--memory-encoder gru` | `--policy-loss clip` | `gru_clip` |
| GRU + SPO | `--memory-encoder gru` | `--policy-loss spo` | `gru_spo` |
| 注意力 + PPO | `--memory-encoder temporal_attention` | `--policy-loss clip` | `attention_clip` |
| 注意力 + SPO | `--memory-encoder temporal_attention` | `--policy-loss spo` | `attention_spo` |

注意力组可额外指定 `--history-length 16 --attention-hidden-size 64 --attention-heads 4`。正式实验应预先统一网络规模、预算、种子、干净验证和固定对手协议，不能用上面极短的自博弈训练判断性能。

如果已有常用训练命令，只需追加 `--policy-loss spo` 并更换实验名。未指定该参数时仍使用 PPO-Clip。训练及定期验证使用原有干净观测，观测退化留给冻结模型后的独立测试。

## 4. 看哪些日志

原有策略损失、价值损失、熵项和 actor/critic 梯度范数继续记录；新增：

| 日志名 | 含义 |
| --- | --- |
| `ratio_deviation` | 概率比偏离 1 的绝对值均值 |
| `approx_kl` | `mean((ratio - 1) - log(ratio))`，衡量新旧策略差异 |
| `ratio_out_of_bounds` | 概率比超出 `[1-epsilon, 1+epsilon]` 的样本比例；Clip/SPO 共用口径 |
| `explained_variance` | rollout 价值预测对回报的解释方差；回报无方差时记为 NaN，表示该指标无定义 |

前三项汇总每个更新批次在执行该次梯度更新前的概率比。SPO 提供软惩罚，概率比仍可能越界。原有熵计算和批量设置在各组中保持一致。

这些指标沿用 Runner 现有的 W&B 记录通道。需要保存日志时，在训练命令中追加 `--use-wandb --log-interval 1`；离线使用可先执行：

```powershell
$env:WANDB_MODE = "offline"
```

离线模式将指标及包含 `policy_loss` 的运行配置保存在本地 W&B 运行目录。未启用 `--use-wandb` 时，现有 Runner 不会将这些详细指标写入文件。

运行针对性验证：

```powershell
python -m pytest tests/test_spo.py tests/test_temporal_attention.py -q
```

测试覆盖正/负/零优势、边界及越界纠偏梯度、Clip 基线、联合动作概率、GRU/MLP/注意力更新，以及真实 JSBSim 的训练、评估和保存恢复闭环。性能提升需要后续正式训练和对照实验验证。
