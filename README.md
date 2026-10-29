# Light Aircraft Game: A lightweight, scalable, gym-wrapped aircraft competitive environment with baseline reinforcement learning algorithms
We provide a competitive environment for red and blue aircrafts games, which includes single control setting, 1v1 setting and 2v2 setting. The flight dynamics based on JSBSIM, and missile dynamics based on our implementation of proportional guidance. We also provide ppo and mappo implementation for self-play or vs-baseline training. 

![fromework](assets/framework.jpg)

## Install 

```shell
# create python env
conda create -n jsbsim python=3.8
# install dependency
pip install torch pymap3d jsbsim==1.1.6 geographiclib gym==0.20.0 wandb icecream setproctitle. 

- Download Shapely‑1.7.1‑cp38‑cp38‑win_amd64.whl from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely), and `pip install shaply` from local file.

- Initialize submodules(*JSBSim-Team/jsbsim*): `git submodule init; git submodule update`
```
## Envs
We provide all task configs in  `envs/JSBSim/configs`, each config corresponds to a task.

### SingleControl
SingleControl env includes single agent heading task, whose goal is to train agent fly according to the given direction, altitude and velocity. The trained agent can be used to design baselines or become the low level policy of the following combat tasks. We can designed two baselines, as shown in the video:

![singlecontrol](assets/1_control.gif)

The red is manever_agent, flying in a triangular trajectory. The blue is pursue agent, constantly tracking the red agent. You can reproduce this by `python envs/JSBSim/test/test_baseline_use_env.py`.


### SingleCombat
SingleCombat env is for two agents 1v1 competitive tasks, including NoWeapon tasks and Missile tasks. We provide self-play setting and vs-baseline setting for each task. Due to the fact that learning to fly and combat simultaneously is non-trival, we also provide a hierarchical framework, where the upper level control gives the direction, altitude and velocity, the low level control use the model trained in SingleControl. 


- NoWeapon tasks require the agent to be in an posture advantage, which means the agent need to fly towards the tail of its opponent and maintain a proper distance. 
- Missile tasks require the agent learn to shoot down oppoents and dodge missiles. Missile engines are based on proportional guidance, we provide a document for our impletation [here](docs/missile_engine). We can futher divide missile tasks into into two categories:
  - Dodge missile task. Missile launches are controled by rules, train agent learn to dodge missile.
  - Shoot missile task. Missile launches are also learning goals. But training from scratch to learn launching missiles is not trival, we need to introduce some prior knowledge for policy learning. We use property that conjugate prior of binomial distribution is beta distribution to address this issue, refer to [here](docs/parameterized_shooting.md) for more details.  A demo for shoot missile task:

![1v1_missile](assets/1v1_missile.gif)


### MultiCombat
MultiCombat env is for four agents 2v2 competitive tasks. The setting is same as SingleCombat. A demo for non-weapon tasks: 

![2v2_posture](assets/2v2_posture.gif)

## Quick Start
### Training

```bash
cd scripts
bash train_*.sh
```
We have provide scripts for five tasks in `scripts/`.

- `train_heading.sh` is for SingleControl environment heading task.
- `train_vsbaseline.sh` is for SingleCombat vs-baseline tasks.
- `train_selfplay.sh` is for SingleCombat self-play tasks. 
- `train_selfplay_shoot.sh` is for SingleCombat self-play shoot missile tasks.
- `train_share_selfplay.sh` is for MultipleCombat self-play tasks.

It can be adapted to other tasks by modifying a few parameter settings. 

- `--env-name` includes options ['SingleControl', 'SingleCombat', 'MultipleCombat'].
- `--scenario` corresponds to yaml file in `envs/JBSim/configs` one by one.
- `--algorithm` includes options [ppo, mappo], ppo for SingleControl and SingleCombat, mappo for MultipleCombat

The description of parameter setting refers to `config.py`.
Note that we set parameters `--use-selfplay --selfplay-algorithm --n-choose-opponents --use-eval --n-eval-rollout-threads --eval-interval --eval-episodes` in selfplay-setting training. `--use-prior` is only set true for shoot missile tasks.
We use wandb to track the training process. If you set `--use-wandb`, please replace the `--wandb-name` with your name. 

训练时添加 `--use-eval --use-early-stop` 可启用工程早停，适用于普通 PPO、自博弈 PPO 和 MAPPO：

```bash
python scripts/train/train_jsbsim.py --env-name SingleControl --scenario-name 1/heading --num-env-steps 1e8 --use-eval --use-early-stop
```

默认在总环境步数预算的 20% 之后开始累计无提升次数，连续 10 次验证平均回合奖励未严格超过历史最佳值时停止；可用 `--early-stop-min-steps 2e7 --early-stop-patience 10` 调整。一次验证指完成 `--eval-episodes` 个回合，验证频率沿用 `--eval-interval`。

开启验证后，从第一次有效验证起保存最佳 `actor_best.pt` / `critic_best.pt`（与是否启用早停无关）；训练正常结束或早停时保存最后的 `actor_latest.pt` / `critic_latest.pt`，结束前也会验证最后模型。自博弈的历史快照和对手采样逻辑继续保留，验证奖励仍受抽取对手强弱影响。未加 `--use-early-stop` 时不提前停止；`--model-dir` 沿用原有权重加载方式，新一轮训练重新累计早停状态。

### Evaluate and Render
```bash
cd renders
python render*.py
```
This will generate a `*.acmi` file. We can use [**TacView**](https://www.tacview.net/), a universal flight analysis tool, to open the file and watch the render videos.

### Real-time Telemetry Rendering during Training

We have added support for real-time telemetry rendering using **Tacview Advanced**. This feature allows you to visualize the training process dynamically, providing a clearer understanding of the simulation data and agent behaviors.

#### How to Enable Real-time Rendering

**Note:** This feature is exclusively available with **Tacview Advanced**, as only the Advanced version supports real-time telemetry.

1. **Set `real_time`、`--use-eval` and `--eval-interval` Parameters**:
   To enable real-time rendering, include the parameter `--render-mode real_time` in the training script command. 
   Real-time rendering occurs during the evaluation process. Enable evaluation mode by setting the parameter `--use-eval`. Use `--eval-interval` to control how often evaluation and visualization are performed. For example:
   ```bash
   bash train_selfplay.sh --render-mode real_time --use-eval --eval-interval 32
   ```
   **Caution:** Real-time visualization may impact performance. Choose an appropriate `--eval-interval` value (e.g., `1` for every episode, `10` for every 10 episodes) based on your system capabilities.

2. **Configure Tacview**:
   - Ensure you have **Tacview Advanced** installed.
   - In Tacview, go to **Record -> Real-time Telemetry**, and input the IP address and port displayed in the training console output (e.g., `192.168.1.120:12345`).

#### Notes
- **Tacview Requirement**: This feature requires Tacview Advanced to be installed and properly configured.
- **Optional Use**: If `--render-mode` is not set to `real_time`, the training process will proceed without real-time rendering.
- **Evaluation-based Rendering**: Real-time rendering is tied to the evaluation process and controlled by `--eval-interval`.
- **Non-blocking**: Real-time rendering will not interfere with the training process. It is designed to work seamlessly alongside the existing pipeline.

This functionality enhances training visualization, making it easier to debug and analyze the agent's performance during the simulation.

## Citing
If you find this repo useful, pleased use the following citation:
````
@misc{liu2022light,
  author = {Qihan Liu and Yuhua Jiang and Xiaoteng Ma},
  title = {Light Aircraft Game: A lightweight, scalable, gym-wrapped aircraft competitive environment with baseline reinforcement learning algorithms},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/liuqh16/CloseAirCombat}},
}
