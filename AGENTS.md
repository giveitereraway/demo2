# AGENTS.md

## 基本约束

- 本仓库服务于硕士毕设《基于分层强化学习的飞行器智能自主决策技术》，所有修改和说明优先围绕飞行器自主决策、分层强化学习、空战对抗和 JSBSim 物理仿真组织。
- 生成代码时，注释使用中文；文档和项目解释默认使用中文。
- 禁止批量删除文件或目录。不要使用 `del /s`、`rd /s`、`rmdir /s`、`Remove-Item -Recurse`、`rm -rf`。需要删除时只能一次删除一个明确文件；若涉及批量清理，停止并让用户手动确认。
- 不要随意改动 `envs/JSBSim/data`，它是 JSBSim 飞机/发动机/系统数据资源。
- 密钥和本机路径放 `.env`，不要写进代码、测试或文档正文。

## 项目定位

这是一个 JSBSim 空战强化学习实验框架，主线是：

1. `SingleControl` 训练低层航向/高度/速度控制器。
2. `SingleCombat` / `MultipleCombat` 在 1v1、2v2 中训练高层空战决策。
3. 分层任务用 `envs/JSBSim/model/actor_heading.pt` 把高层指令转成底层飞控动作。
4. 自博弈、导弹攻防、人机交互和 Tacview 轨迹用于训练、评估和论文分析。
5. `agent_system/` 是当前新增的航电智能 Agent 原型，用于路由训练、评估、人机、可视化和 RAG 问答流程。

## 关键目录

- `config.py`：训练/评估通用命令行参数，正式参数是 `--clip-param`。
- `scripts/train/train_jsbsim.py`：JSBSim 主训练入口。
- `scripts/*.sh`：常用训练、人机、渲染脚本；部分旧脚本仍有 `--clip-params`，改命令时优先统一为 `--clip-param`。
- `experiments/1v1.py`：1v1 actor 对比评估入口。
- `scripts/run_agent_app.py`：启动 Streamlit Agent 应用。
- `agent_system/`：Agent 应用、命令构造、执行、路由、结果分析、RAG 适配。
- `envs/JSBSim/configs/`：场景 YAML，`--scenario-name` 与这里的相对路径对应。
- `envs/JSBSim/envs/`：`SingleControlEnv`、`SingleCombatEnv`、`MultipleCombatEnv`。
- `envs/JSBSim/tasks/`：任务定义，是修改观测、动作、奖励、终止和分层逻辑的核心位置。
- `envs/JSBSim/tasks/singlecombat_with_missle_task.py`：文件名保留历史拼写 `missle`，引用时不要改成 `missile`。
- `algorithms/ppo/`、`algorithms/mappo/`：PPO / MAPPO 实现。
- `runner/`：训练、评估、渲染、自博弈和 Tacview 通信。
- `tests/`：PPO、JSBSim 环境、Agent 系统测试。

## 核心链路

训练链路：

```text
scripts/train/train_jsbsim.py
-> config.py
-> make_train_env/make_eval_env
-> envs/JSBSim/configs/{scenario}.yaml
-> Env.load_task()
-> Task.normalize_action()/get_obs()/reward/termination
-> Runner.collect()/train()/eval()/save()
-> scripts/results/...
```

Agent 链路：

```text
scripts/run_agent_app.py
-> agent_system/app.py
-> routing.py 选择 train/evaluate_1v1/human_loop/visualize/rag_qa
-> commands.py 构造命令
-> executor.py 用 shell=False 执行
-> result_analysis.py / rag_adapter.py 输出分析或问答结果
```

## 当前重点实现

- 常规分层 1v1：`HierarchicalSingleCombatTask` 使用 `MultiDiscrete([3, 5, 3])` 表示高度、航向、速度增量。
- 战术分层 1v1：`TacticalHierarchicalSingleCombatTask` 是当前主动维护路径，动作空间为 `Discrete(12)`，包含纯追击、提前追击、滞后追击、脱离、爬升、俯冲加速、平飞加/减速、防御转弯、高/低 Yo-Yo。
- 战术导弹任务：`TacticalHierarchicalSingleCombatShootTask` 使用 `Tuple([Discrete(12), Discrete(2)])`，第一维是战术动作，第二维是发射决策。
- 战术动作最终仍会通过低层 `actor_heading.pt` 输出 4 维飞控动作；修改时要保留航向量化、安全高度约束和近距追击限速等保护逻辑。
- 2v2 使用 `MultipleCombatEnv` + MAPPO + 集中式 critic；不要把 1v1 PPO 假设直接套到 2v2。

## 修改建议

- 修改 Task 时同步检查 `load_observation_space()`、`load_action_space()`、`get_obs()`、`normalize_action()`、`reset()`、`step()` 和 YAML 的 `task` 字段。
- 修改动作空间时同步检查 `algorithms/utils/act.py`、`distributions.py`、buffer action shape、训练脚本和测试。
- 修改分层低层输入时同时检查 `HeadingTask.get_obs()` 与各 `Hierarchical*Task.normalize_action()` 拼接的 12 维输入。
- 修改导弹逻辑时同步检查 `MissileSimulator`、导弹任务 `step()`、`ShootPenaltyReward` 和 `--use-prior`。
- 修改自博弈时同步检查 `policy_pool`、`actor_{episode}.pt`、`actor_latest.pt`、评估对手选择和 ELO 更新。
- 修改 `agent_system/commands.py` 时保持命令为 `list[str]`，继续使用 `shell=False`，并在测试中覆盖命令参数。

## 常用命令

```powershell
python -m pytest tests/test_agent_system.py
python -m pytest tests/test_ppo.py
python -m pytest tests/test_jsbsim.py
python scripts/run_agent_app.py
python scripts/train/train_jsbsim.py --env-name SingleControl --algorithm-name ppo --scenario-name 1/heading
python experiments/1v1.py --help
```

Git Bash / WSL 中可运行：

```bash
cd scripts
bash train_heading.sh
bash train_selfplay.sh
bash train_tactical_selfplay.sh
bash train_tactical_selfplay_shoot.sh
```

## 产物和注意事项

- 训练结果默认在 `scripts/results/{env}/{scenario}/{algorithm}/{experiment}/runX/`。
- 对比评估和图表结果通常在 `experiments/results/`。
- Tacview 轨迹为 `.acmi` / `.txt.acmi`，可用于论文实验分析。
- 训练默认使用 SwanLab 离线记录，可用 `SWANLAB_MODE=online` 切换为在线记录。
- JSBSim 相关测试依赖 `torch`、`gymnasium`、`jsbsim`、`pymap3d` 等本机环境，失败时先区分依赖问题和代码问题。
- Windows 终端可能显示中文乱码；编辑文档和代码时保持 UTF-8。
