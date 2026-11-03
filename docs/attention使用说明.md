# 轻量因果时序注意力接入说明

本次在 `attention` 分支实现 `temporal_attention`，参考《第三阶段方案》第 4.2–4.3 节。支持 SingleControl/heading 和 SingleCombat/1v1 的 PPO 训练、定期验证和恢复权重，1v1 同时支持自博弈。默认仍使用原来的 GRU。

## 1. 把它想成一本能翻阅的小笔记

原来的 GRU 每步读一帧观测，把历史压缩成一份“脑内记忆”。现在的注意力每步保留最近 16 帧原始观测，再根据当前情况判断该重点参考哪几帧。例如，敌机相对距离正在变化，模型可以同时参考当前距离和几步前的距离。

```text
原来：当前观测 → MLP → GRU 的压缩记忆 → 动作头
现在：最近 16 帧观测 → 线性编码＋相对时间位置 → 一层注意力 → 动作头
```

单帧观测维数沿用各任务：heading 为 12 维，常规 1v1 为 15 维。16 帧包含当前帧和最多 15 帧过去观测。回合刚开始时，不足的历史用 padding 补齐并屏蔽。模型实际经历的重复观测仍是有效历史。

“因果”在这里表示只能看现在和过去，不能偷看未来；不表示因果推断。只用当前帧作 query、用窗口中的有效帧作 key/value，等价于单层因果自注意力最后一个时刻的输出，省去其他时刻的 query 计算。相对位置编码告诉模型“这是几步前”，没有输入目标信息质量或龄期标签。

默认配置为窗口 16、特征维数 64、4 个注意力头、1 层、Pre-LayerNorm、残差和两层小前馈网络，dropout 为 0。actor 负责动作、critic 负责估值，两者各有独立参数。

## 2. 代码怎么接进去的

| 文件 | 修改作用 |
| --- | --- |
| `algorithms/utils/temporal_attention.py` | 原始观测窗口、相对位置编码、padding 屏蔽和轻量注意力 |
| `ppo_actor.py`、`ppo_critic.py` | 按编码器参数选择 MLP、GRU 或注意力，继续接原有动作/价值头 |
| `algorithms/utils/buffer.py` | 按真实状态形状分配空间；新增逐时刻 mini-batch 生成器 |
| `ppo_trainer.py` | GRU 用原有序列训练，MLP/注意力用逐时刻训练；PPO 损失保持原样 |
| `ppo_policy.py`、`runner/base_runner.py` | 保存 `model_config.json`，恢复前校验结构 |
| `runner/selfplay_jsbsim_runner.py` | 修正评估 mask，使新回合第一步清空后能继续积累历史 |
| `config.py`、训练入口 | 新增参数，拦截旧开关冲突及暂不支持的 MAPPO 路径 |

为了少改 runner，继续使用名为 `rnn_states_actor/critic` 的现有状态通道。GRU 时里面仍是隐状态；注意力时里面变成原始观测和每个历史位置的有效标记。标记只区分回合开始的 padding，不判断传感器质量。原有的双方独立状态、并行环境状态、回合结束清空和换对手清空逻辑可以继续使用。

每个采样时刻的历史都随状态进入 buffer。训练打乱样本时，观测和对应历史一起取出；每次 PPO 更新都用当前参数重新编码整段原始历史。这样不会把旧网络算出的特征当成新特征，也不会把其他飞机或下一回合的观测混进来。

同一回合的窗口随 `after_update()` 跨 rollout 保留。注意力窗口由 `--history-length` 控制，独立于 GRU 的 `--data-chunk-length`。窗口中的上下文参与编码和反传，但每个采样时刻只计算一次自己的动作/价值损失。

另外修正了 critic 价值输出层的输入维数：它应接收前置 MLP 的实际输出维数。旧默认配置两者都是 128，原问题没有暴露；注意力的编码维数为 64 时需要正确衔接。

## 3. 怎么启动

先激活已安装项目依赖的 Python 环境，在仓库根目录运行。一个小规模检查命令如下，PowerShell 中整行复制：

```powershell
python scripts/train/train_jsbsim.py --env-name SingleCombat --algorithm-name ppo --scenario-name 1v1/NoWeapon/Selfplay --experiment-name attention_smoke --memory-encoder temporal_attention --use-selfplay --n-choose-opponents 1 --n-rollout-threads 1 --buffer-size 32 --num-env-steps 64 --ppo-epoch 1 --num-mini-batch 1 --use-eval --n-eval-rollout-threads 1 --eval-episodes 1
```

这个命令已实际跑通，使用真实场景的原始回合长度，完成 64 步自博弈训练、完整回合验证及权重/配置保存。它只检查链路能否运行，64 步不构成有效策略训练。训练采样尚未结束任何回合时，原有 `average_episode_rewards` 日志会因分母为零显示 `inf`；这不等于 PPO 损失发散，末尾验证奖励单独统计。

heading 场景也可以通过标准入口训练，下面使用 128 维注意力执行短检查：

```powershell
python scripts/train/train_jsbsim.py --env-name SingleControl --algorithm-name ppo --scenario-name 1/heading --experiment-name heading_attention_smoke --memory-encoder temporal_attention --history-length 16 --attention-hidden-size 128 --attention-heads 4 --n-rollout-threads 1 --buffer-size 32 --num-env-steps 64 --ppo-epoch 1 --num-mini-batch 1 --use-eval --n-eval-rollout-threads 1 --eval-episodes 1
```

已通过标准入口在 CPU 上完成 heading 的双环境短训练：128 个环境步、2 次 PPO 更新，使用原始场景回合长度完成验证，并保存 actor/critic 的 latest、best 权重及 `model_config.json`。heading 不添加自博弈参数；正式训练可沿用原 heading 的 PPO 参数和训练预算，并使用独立实验名。MAPPO/MultipleCombat 仍不支持注意力。

把原有 heading 或 1v1 训练命令改成注意力，通常只要增加：

```text
--memory-encoder temporal_attention
```

需要明确写出默认结构时，增加：

```text
--history-length 16 --attention-hidden-size 64 --attention-heads 4
```

头数必须整除特征维数。使用新编码器参数时，移除旧的 `--use-recurrent-policy`：旧参数的历史含义是“关闭 GRU”，与显式选择 GRU/注意力同时出现会报错。

三个编码器的选择方法：

| 选择 | 参数 |
| --- | --- |
| 原有 GRU | 不加新参数，或 `--memory-encoder gru` |
| 单帧 MLP | `--memory-encoder mlp` |
| 时序注意力 | `--memory-encoder temporal_attention` |

输出仍在 `scripts/results/` 原有训练目录中。每个目录新增 `model_config.json`；恢复时使用原有 `--model-dir`，并传入与该文件一致的网络参数。该入口仍只恢复 actor/critic 权重，不恢复优化器、自博弈池或中途历史。新启动从新回合开始。

注意力需要重新训练自己的权重。旧 GRU 权重仍按 GRU 配置加载；原有裸权重没有元数据时，GRU/MLP 保留旧加载行为。新注意力 checkpoint 必须保留旁边的配置文件。

## 4. 验证与使用边界

```powershell
python -m pytest tests/test_temporal_attention.py tests/test_ppo.py tests/test_eval_checkpoints.py -q
```

新测试覆盖 padding、因果前缀、重置与环境隔离、历史观测反传、跨 rollout 的历史、逐步采样与训练概率一致性、零更新概率比约为 1、三种常用动作空间的实际 PPO 更新、旧 GRU 配置兼容和 checkpoint 校验。另有训练入口的支持范围检查，以及真实 JSBSim 的 heading（64/128 维注意力）、1v1 普通/自博弈短闭环；回归测试临时缩短实例回合，不改 YAML 或物理数据。

本次在 CPU 上执行上述相关测试，109 项通过。检查环境为 `demo3.8`；其缺少的依赖单独装在被 Git 忽略的 `tmp/attention_test_deps`，未修改已有 Conda 环境。要复用这套本机检查环境，先执行：

```powershell
conda activate demo3.8
$env:PYTHONPATH = (Resolve-Path tmp/attention_test_deps).Path
```

再运行上面的训练或测试命令。这里的临时依赖只用于本次本机验证，不会随 Git 分支分发。

本次训练及定期验证仍使用环境原有干净观测。独立 E0–E3 退化评估入口尚未扩展为注意力模型加载器；正式对照实验与性能结论后续单独完成。SPO、对手采样算法、低层控制器和 Agent 本次没有扩展。

注意力缓存原始窗口，占用的 buffer 空间高于旧 128 维 GRU 状态。15 维观测、H=16 时，每个状态为 `16×(15+1)=256` 个 float32；actor/critic 各保存一份。设置大窗口、大 buffer 或大量并行环境前，应留意内存占用。
