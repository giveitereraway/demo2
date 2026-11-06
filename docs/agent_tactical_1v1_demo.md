# 1v1 LLM-Agent 战术调度演示

这个演示面向 `SingleCombat + 1v1/NoWeapon/TacticalHierarchySelfplay`，用于展示“自然语言战术指令 + 训练好的 tactical actor fallback”的人在回路闭环。LLM 只负责把一句中文指令解析为 12 类战术动作编号，不直接输出舵面、油门或连续飞控动作。

## 运行方式

```powershell
python scripts/agent/agent_tactical_1v1_demo.py `
  --max-steps 100 `
  --render-mode txt
```

固定动作敌机示例：

```powershell
python scripts/agent/agent_tactical_1v1_demo.py `
  --render-mode real_time `
  --enemy-action PURE_PURSUIT `
  --status-interval 0
```

指定敌机模型示例：

```powershell
python scripts/agent/agent_tactical_1v1_demo.py `
  --render-mode real_time `
  --enemy-path $env:ENEMY_ACTOR_PATH `
  --status-interval 0
```

`--actor-path` 默认使用项目下的训练目录。

```text
scripts/results/SingleCombat/1v1/NoWeapon/TacticalHierarchySelfplay/ppo/1v1_tactical_hierarchy_2/run1
```

它可以指向 `actor_latest.pt` 文件，也可以指向包含该文件的目录。该 actor 必须是 `1v1/NoWeapon/TacticalHierarchySelfplay` 训练出的高层 tactical actor，动作空间为 `Discrete(12)`，输出战术动作编号。

敌机现在也支持 actor 模型。`--enemy-path` 不传时会复用当前 `--actor-path`，因此默认情况下双方都由 tactical actor 推理；如果显式传入 `--enemy-action`，敌机改为固定战术动作，并且该固定动作优先于 `--enemy-path`。

常用参数：

- `--scenario-name`：默认 `1v1/NoWeapon/TacticalHierarchySelfplay`。
- `--actor-path`：默认使用上面的训练目录。传入目录时，脚本会自动读取目录下的 `actor_latest.pt`。
- `--enemy-path`：敌机 tactical actor 路径；不传时复用 `--actor-path`，同样可以是 `actor_latest.pt` 或包含该文件的目录。
- `--agent-id`：默认 `A0100`，表示接受人工指令和 actor fallback 的己方飞机。
- `--enemy-action`：敌机固定战术动作；显式传入时优先于 `--enemy-path`，例如 `PURE_PURSUIT`、`LEAD_PURSUIT` 或动作编号。
- `--hold-steps`：人工指令接管持续的环境步数，默认 10。
- `--render-mode`：`txt` 输出 ACMI 文件；`real_time` 连接 Tacview Advanced；`none` 只运行日志。
- `--step-sleep`：每个环境步后的真实等待时间，默认 0.2 秒，接近当前 YAML 的 0.2 秒仿真步长。
- `--status-interval`：终端状态打印间隔，默认每 25 步打印一次；设为 0 时只在人工指令或安全覆盖时打印。
- `--verbose-steps`：调试开关；打开后每个仿真步都打印状态，容易刷屏。
- `--disable-llm`：只使用关键词映射，不调用硅基流动模型。

## 交互逻辑

脚本启动后会开一个输入线程。用户不输入内容时，己方每步调用 `--actor-path` 指定的 actor，根据当前观测自主输出 tactical action id，日志中记录为 `source=actor_fallback`。敌机默认调用 `--enemy-path` 指定的 actor；如传入 `--enemy-action`，敌机每步使用固定动作。

用户输入中文短句并回车后，例如：

```text
爬升占位
减速避免冲过头
向左防御转弯
```

系统会先尝试使用 LLM 输出固定 JSON；如果没有 API Key、LLM 调用失败或 JSON 不合法，则回退到关键词解析。解析成功后，人工动作接管 `--hold-steps` 个环境步，日志中记录为 `source=manual`。接管结束后自动恢复 actor fallback。

终端默认不会逐步刷屏，完整逐步记录写入 JSONL 日志。实时操控时建议保持默认低频打印，或者进一步降低输出：

```powershell
python scripts/agent/agent_tactical_1v1_demo.py --render-mode real_time --status-interval 0
```

## 安全覆盖

人工指令和 actor 输出都会经过轻量安全规则：

- 高度接近低空阈值时，`DIVE_ACCELERATE`、`LOW_YOYO` 会被改为 `CLIMB_POSITION`。
- 近距高速追击时，`PURE_PURSUIT`、`LEAD_PURSUIT` 会被改为 `LEVEL_DECELERATE`。
- 明显不利姿态下，进攻类动作会被改为 `DISENGAGE`。
- 非法动作编号会回退到上一安全动作或默认 `PURE_PURSUIT`。

这些规则只约束战术动作边界，实际飞控仍由 `TacticalHierarchicalSingleCombatTask` 内部的低层 `actor_heading.pt` 完成。

## 输出

默认日志文件：

```text
output/agent_tactical_1v1/demo_log.jsonl
```

每个 step 会记录己方动作来源、己方 actor 输出、敌机动作来源、敌机动作、人工解析结果、最终动作、安全覆盖原因、奖励和 done 状态。默认 ACMI 文件：

```text
output/agent_tactical_1v1/demo.txt.acmi
```

如果使用 `--render-mode real_time`，需要 Tacview Advanced 打开 `Record -> Real-time Telemetry`，连接控制台打印的 IP 和端口。

## 论文表述边界

建议表述为“面向 1v1 空战的 LLM-Agent 战术调度层”。LLM 负责自然语言战术意图识别，规则模块负责动作合法性与安全边界校验，分层强化学习策略负责具体飞行动作执行。不要声称 LLM 提升了空战胜率；第一版重点是人机交互性、策略可调用性和演示可解释性。
