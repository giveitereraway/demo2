#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import contextlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple

HELP_REQUESTED = any(arg in {"-h", "--help"} for arg in sys.argv[1:])
RUNTIME_IMPORT_ERROR = None

try:
    import numpy as np
    import torch
    from gymnasium import spaces
except Exception as exc:
    if not HELP_REQUESTED:
        raise
    RUNTIME_IMPORT_ERROR = exc
    np = None
    spaces = None

    class _TorchHelpStub:
        def no_grad(self):
            def decorator(func):
                return func

            return decorator

    torch = _TorchHelpStub()

# matplotlib 只用于实验后处理，缺失时不影响对战评估。
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# 让脚本可以从 experiments/ 目录直接导入项目代码。
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from algorithms.ppo.ppo_actor import PPOActor
    from envs.JSBSim.core.catalog import Catalog as c
    from envs.JSBSim.envs import SingleCombatEnv
    from envs.JSBSim.utils.utils import NEU2LLA, get_AO_TA_R
except Exception as exc:
    if not HELP_REQUESTED:
        raise
    RUNTIME_IMPORT_ERROR = RUNTIME_IMPORT_ERROR or exc
    PPOActor = None
    c = None
    SingleCombatEnv = None
    NEU2LLA = None
    get_AO_TA_R = None


# =========================
# 可修改实验配置
# =========================

EXPERIMENT_NAME = "follow_2e8_A_vs_follow_1e8_B"

# 评估环境必须使用直接控制版 SingleCombat 场景。
EVAL_SCENARIO_NAME = "1v1/NoWeapon/Selfplay"

# 两个 actor 可以来自不同训练场景；脚本会按各自场景构造网络动作空间。
ACTOR_A_PATH = "E:/clone/demo2/scripts/results/SingleCombat/1v1/NoWeapon/Selfplay/ppo/1v1_follow_2e8/wandb/offline-run-20260617_150234-2fbfp5qn/files/actor_latest.pt"
ACTOR_A_SCENARIO_NAME = "1v1/NoWeapon/Selfplay"

ACTOR_B_PATH = "E:/clone/demo1/scripts/results/SingleCombat/1v1/NoWeapon/Selfplay/ppo/1v1_follow/wandb/offline-run-20260512_175151-yryla8wg/files/actor_latest.pt"
ACTOR_B_SCENARIO_NAME = "1v1/NoWeapon/Selfplay"

# 分层 actor 的低层控制器。
LOWLEVEL_ACTOR_PATH = REPO_ROOT / "envs/JSBSim/model/actor_heading.pt"

NUM_EPISODES = 100
SEED = 1
DEVICE = "cuda:0"  # auto / cpu / cuda:0
DETERMINISTIC = True
WIN_REWARD_MARGIN = 100.0

# 是否交替 actor A/B 控制 A0100 与 B0100，减少固定阵营偏差。
SWAP_ACTOR_ORDER = True

# 是否沿用 SingleCombatEnv 默认的初始状态随机换边。
RANDOM_SIDE_SWAP = True

# 可选：覆盖 YAML 初始状态。保持 None 表示使用场景 YAML。
CUSTOM_INITIAL_STATES = None
# CUSTOM_INITIAL_STATES = {
#     "A0100": {"ic_h_sl_ft": 20000, "ic_psi_true_deg": 0.0, "ic_u_fps": 800.0},
#     "B0100": {"ic_h_sl_ft": 20000, "ic_psi_true_deg": 180.0, "ic_u_fps": 800.0},
# }

# 初始状态模式：
# fixed：沿用 YAML 的两个初始状态，只可选择是否随机换边。
# random：每个 episode 重新采样双方距离、方位、高度、航向和速度。
INITIAL_STATE_MODE = "random"  # fixed / random

# 随机初始态范围。单位保持 JSBSim 初始条件习惯：高度 ft、速度 ft/s、角度 deg、距离 m。
RANDOM_INITIAL_STATE_CONFIG = {
    "distance_m": [18000.0, 22000.0], # 初始距离
    "center_north_m": [-1500.0, 1500.0], # 两机连线中心点相对战场中心的北向偏移
    "center_east_m": [-1500.0, 1500.0], # 两机连线中心点相对战场中心的东向偏移
    "altitude_ft": [16000.0, 26000.0], # 两机平均初始高度
    "altitude_difference_ft": [-3000.0, 3000.0], # 两机高度差
    "speed_fps": [650.0, 950.0], # 双方初始速度范围
    # mixed 会在迎头、同向追逐、交叉、随机航向中随机选一种。
    "heading_mode": "head_on",  # mixed / head_on / tail_chase / crossing / random
    "heading_noise_deg": 20.0, # 航向扰动角度
}

SAVE_ACMI = True
ACMI_EPISODES = {0}  # 只保存指定回合，避免大量轨迹文件。
OUTPUT_ROOT = REPO_ROOT / "experiments/results"
OUTPUT_DIR = None

# 是否在实验结束后保存图表，默认输出到结果目录下的 plots/。
SAVE_PLOTS = True
PLOT_DPI = 180
MOVING_AVERAGE_WINDOW = 5

# 绘图颜色约定：Actor A 固定红色，Actor B 固定蓝色，方便不同实验图表横向比较。
ACTOR_A_COLOR = "#d62728"
ACTOR_B_COLOR = "#1f77b4"
TIE_COLOR = "#7f7f7f"

PLOT_ONLY_RESULT_DIR = None
#PLOT_ONLY_RESULT_DIR = OUTPUT_ROOT / "selfA_vs_hierarchyselfB_20260513_104605"

# 当前 BetaShootBernoulli 内部有调试 print，射击任务中建议保持开启。
SUPPRESS_POLICY_DEBUG_PRINT = True


def str_to_bool(value) -> bool:
    """把命令行里的 true/false 字符串转成 bool。"""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"无法解析布尔值: {value}")


def cli_path(value: str | Path | None) -> Optional[Path]:
    if value is None:
        return None
    text = str(value).strip().strip('"').strip("'")
    if sys.platform == "win32":
        text = text.replace("/", "\\")
    if text == "":
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def parse_acmi_episodes(value: str | Iterable[int]) -> set[int]:
    """解析 0,1,2 形式的 ACMI 保存回合。"""
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in {"none", "false", "no"}:
            return set()
        return {int(part.strip()) for part in stripped.split(",") if part.strip()}
    return {int(item) for item in value}


def parse_custom_initial_states(value: str | None):
    """支持从 JSON 字符串或 JSON 文件读取初始状态覆盖。"""
    if value is None or not str(value).strip():
        return None
    candidate = cli_path(value)
    if candidate is not None and candidate.exists():
        return json.loads(candidate.read_text(encoding="utf-8"))
    return json.loads(str(value))


def parse_random_initial_state_config(value: str | None) -> Dict[str, object]:
    """读取随机初始态配置，允许用 JSON 字符串或 JSON 文件覆盖默认范围。"""
    config = dict(RANDOM_INITIAL_STATE_CONFIG)
    if value is None or not str(value).strip():
        return config
    candidate = cli_path(value)
    if candidate is not None and candidate.exists():
        override = json.loads(candidate.read_text(encoding="utf-8"))
    else:
        override = json.loads(str(value))
    config.update(override)
    return config


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="1v1 actor 对战评估脚本。")
    parser.add_argument("--experiment-name", default=EXPERIMENT_NAME)
    parser.add_argument("--eval-scenario-name", default=EVAL_SCENARIO_NAME)
    parser.add_argument("--actor-a-path", default=str(ACTOR_A_PATH))
    parser.add_argument("--actor-a-scenario-name", default=ACTOR_A_SCENARIO_NAME)
    parser.add_argument("--actor-b-path", default=str(ACTOR_B_PATH))
    parser.add_argument("--actor-b-scenario-name", default=ACTOR_B_SCENARIO_NAME)
    parser.add_argument("--lowlevel-actor-path", default=str(LOWLEVEL_ACTOR_PATH))
    parser.add_argument("--num-episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default=DEVICE, help="auto / cpu / cuda:0")
    parser.add_argument("--deterministic", type=str_to_bool, default=DETERMINISTIC)
    parser.add_argument("--win-reward-margin", type=float, default=WIN_REWARD_MARGIN)
    parser.add_argument("--swap-actor-order", type=str_to_bool, default=SWAP_ACTOR_ORDER)
    parser.add_argument("--random-side-swap", type=str_to_bool, default=RANDOM_SIDE_SWAP)
    parser.add_argument("--custom-initial-states-json", default=None)
    parser.add_argument("--initial-state-mode", choices=["fixed", "random"], default=INITIAL_STATE_MODE)
    parser.add_argument("--random-initial-state-config-json", default=None)
    parser.add_argument("--save-acmi", type=str_to_bool, default=SAVE_ACMI)
    parser.add_argument("--acmi-episodes", default=",".join(str(item) for item in sorted(ACMI_EPISODES)))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-plots", type=str_to_bool, default=SAVE_PLOTS)
    parser.add_argument("--plot-dpi", type=int, default=PLOT_DPI)
    parser.add_argument("--moving-average-window", type=int, default=MOVING_AVERAGE_WINDOW)
    parser.add_argument("--plot-only-result-dir", default=PLOT_ONLY_RESULT_DIR)
    parser.add_argument("--suppress-policy-debug-print", type=str_to_bool, default=SUPPRESS_POLICY_DEBUG_PRINT)
    return parser


def apply_cli_args(args: argparse.Namespace) -> None:
    """把 CLI 参数写回旧脚本使用的全局配置，尽量不扰动原有评估逻辑。"""
    global EXPERIMENT_NAME, EVAL_SCENARIO_NAME
    global ACTOR_A_PATH, ACTOR_A_SCENARIO_NAME, ACTOR_B_PATH, ACTOR_B_SCENARIO_NAME
    global LOWLEVEL_ACTOR_PATH, NUM_EPISODES, SEED, DEVICE, DETERMINISTIC
    global WIN_REWARD_MARGIN, SWAP_ACTOR_ORDER, RANDOM_SIDE_SWAP, CUSTOM_INITIAL_STATES
    global INITIAL_STATE_MODE, RANDOM_INITIAL_STATE_CONFIG
    global SAVE_ACMI, ACMI_EPISODES, OUTPUT_ROOT, OUTPUT_DIR, SAVE_PLOTS
    global PLOT_DPI, MOVING_AVERAGE_WINDOW, PLOT_ONLY_RESULT_DIR, SUPPRESS_POLICY_DEBUG_PRINT

    EXPERIMENT_NAME = args.experiment_name
    EVAL_SCENARIO_NAME = args.eval_scenario_name
    ACTOR_A_PATH = cli_path(args.actor_a_path)
    ACTOR_A_SCENARIO_NAME = args.actor_a_scenario_name
    ACTOR_B_PATH = cli_path(args.actor_b_path)
    ACTOR_B_SCENARIO_NAME = args.actor_b_scenario_name
    LOWLEVEL_ACTOR_PATH = cli_path(args.lowlevel_actor_path)
    NUM_EPISODES = int(args.num_episodes)
    SEED = int(args.seed)
    DEVICE = args.device
    DETERMINISTIC = bool(args.deterministic)
    WIN_REWARD_MARGIN = float(args.win_reward_margin)
    SWAP_ACTOR_ORDER = bool(args.swap_actor_order)
    RANDOM_SIDE_SWAP = bool(args.random_side_swap)
    CUSTOM_INITIAL_STATES = parse_custom_initial_states(args.custom_initial_states_json)
    INITIAL_STATE_MODE = args.initial_state_mode
    RANDOM_INITIAL_STATE_CONFIG = parse_random_initial_state_config(args.random_initial_state_config_json)
    SAVE_ACMI = bool(args.save_acmi)
    ACMI_EPISODES = parse_acmi_episodes(args.acmi_episodes)
    OUTPUT_ROOT = cli_path(args.output_root)
    OUTPUT_DIR = cli_path(args.output_dir)
    SAVE_PLOTS = bool(args.save_plots)
    PLOT_DPI = int(args.plot_dpi)
    MOVING_AVERAGE_WINDOW = int(args.moving_average_window)
    PLOT_ONLY_RESULT_DIR = cli_path(args.plot_only_result_dir)
    SUPPRESS_POLICY_DEBUG_PRINT = bool(args.suppress_policy_debug_print)


TASK_FAMILY = {
    "singlecombat": "NoWeapon",
    "hierarchical_singlecombat": "NoWeapon",
    "tactical_hierarchical_singlecombat": "NoWeapon",
    "singlecombat_dodge_missile": "DodgeMissile",
    "hierarchical_singlecombat_dodge_missile": "DodgeMissile",
    "tactical_hierarchical_singlecombat_dodge_missile": "DodgeMissile",
    "singlecombat_shoot": "ShootMissile",
    "hierarchical_singlecombat_shoot": "ShootMissile",
    "tactical_hierarchical_singlecombat_shoot": "ShootMissile",
}

DIRECT_EVAL_TASKS = {
    "singlecombat",
    "singlecombat_dodge_missile",
    "singlecombat_shoot",
}


def _t2n(x):
    return x.detach().cpu().numpy()


def make_actor_args(use_prior: bool) -> SimpleNamespace:
    """构造与训练脚本默认 PPOActor 一致的网络参数。"""
    return SimpleNamespace(
        gain=0.01,
        hidden_size="128 128",
        act_hidden_size="128 128",
        activation_id=1,
        use_feature_normalization=False,
        use_recurrent_policy=True,
        recurrent_hidden_size=128,
        recurrent_hidden_layers=1,
        use_prior=use_prior,
    )


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def task_family(task_name: str) -> str:
    if task_name not in TASK_FAMILY:
        raise ValueError(f"不支持的 SingleCombat task: {task_name}")
    return TASK_FAMILY[task_name]


def action_dim(action_space) -> int:
    """统计 actor 输出动作维度，用于基础兼容性检查。"""
    if isinstance(action_space, spaces.Discrete):
        return 1
    if isinstance(action_space, spaces.MultiDiscrete):
        return int(action_space.shape[0])
    if isinstance(action_space, spaces.Tuple):
        return int(sum(action_dim(space) for space in action_space.spaces))
    raise TypeError(f"不支持的动作空间: {action_space}")


@dataclass
class ActorController:
    name: str
    path: Path
    scenario_name: str
    task_name: str
    family: str
    is_hierarchical: bool
    is_tactical: bool
    is_shoot: bool
    obs_dim: int
    actor_action_dim: int
    actor: PPOActor
    lowlevel_policy: Optional[PPOActor]
    device: torch.device

    def reset(self) -> None:
        # PPO actor 与分层低层 actor 都是 GRU 策略，每局开始清空隐状态。
        self.rnn_states = np.zeros((1, 1, 128), dtype=np.float32)
        self.lowlevel_rnn_states = np.zeros((1, 1, 128), dtype=np.float32)
        self.masks = np.ones((1, 1), dtype=np.float32)

    @torch.no_grad()
    def act(self, obs: np.ndarray, env: Optional[SingleCombatEnv] = None, agent_id: Optional[str] = None) -> np.ndarray:
        obs_batch = np.expand_dims(obs, axis=0)
        if SUPPRESS_POLICY_DEBUG_PRINT:
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                actor_action, _, self.rnn_states = self.actor(
                    obs_batch,
                    self.rnn_states,
                    self.masks,
                    deterministic=DETERMINISTIC,
                )
        else:
            actor_action, _, self.rnn_states = self.actor(
                obs_batch,
                self.rnn_states,
                self.masks,
                deterministic=DETERMINISTIC,
            )

        raw_action = np.asarray(_t2n(actor_action).squeeze(0)).reshape(-1)
        self.rnn_states = _t2n(self.rnn_states)

        if not self.is_hierarchical:
            return raw_action

        if self.is_tactical:
            if env is None or agent_id is None:
                raise ValueError(f"{self.name} 是 tactical_hierarchical actor，act() 需要 env 和 agent_id。")
            lowlevel_action = self._tactical_to_lowlevel(obs, raw_action, env, agent_id)
        else:
            lowlevel_action = self._hierarchical_to_lowlevel(obs, raw_action)
        if self.is_shoot:
            return np.concatenate([lowlevel_action, raw_action[-1:]], axis=0)
        return lowlevel_action

    @torch.no_grad()
    def _hierarchical_to_lowlevel(self, obs: np.ndarray, high_action: np.ndarray) -> np.ndarray:
        if self.lowlevel_policy is None:
            raise RuntimeError(f"{self.name} 是分层 actor，但未加载低层控制器。")

        delta_altitude = np.array([0.1, 0.0, -0.1], dtype=np.float32)
        delta_heading = np.array(
            [-np.pi / 6, -np.pi / 12, 0.0, np.pi / 12, np.pi / 6],
            dtype=np.float32,
        )
        delta_velocity = np.array([0.05, 0.0, -0.05], dtype=np.float32)

        # 分层高层动作含义：高度变化、航向变化、速度变化。
        high_action = high_action.astype(np.int64)
        delta_control = np.array(
            [
                delta_altitude[high_action[0]],
                delta_heading[high_action[1]],
                delta_velocity[high_action[2]],
            ],
            dtype=np.float32,
        )
        return self._delta_control_to_lowlevel(obs, delta_control)

    def _delta_control_to_lowlevel(self, obs: np.ndarray, delta_control: np.ndarray) -> np.ndarray:
        lowlevel_obs = np.zeros(12, dtype=np.float32)
        lowlevel_obs[0:3] = np.asarray(delta_control, dtype=np.float32)
        lowlevel_obs[3:12] = obs[:9]

        lowlevel_action, _, self.lowlevel_rnn_states = self.lowlevel_policy(
            np.expand_dims(lowlevel_obs, axis=0),
            self.lowlevel_rnn_states,
            self.masks,
            deterministic=True,
        )
        self.lowlevel_rnn_states = _t2n(self.lowlevel_rnn_states)
        return _t2n(lowlevel_action).squeeze(0)

    # ---- Tactical 辅助方法（与 TacticalHierarchicalSingleCombatTask 保持一致） ----

    # 12 个战术动作常量
    PURE_PURSUIT = 0
    LEAD_PURSUIT = 1
    LAG_PURSUIT = 2
    DISENGAGE = 3
    CLIMB_POSITION = 4
    DIVE_ACCELERATE = 5
    LEVEL_ACCELERATE = 6
    LEVEL_DECELERATE = 7
    DEFENSIVE_TURN_LEFT = 8
    DEFENSIVE_TURN_RIGHT = 9
    HIGH_YOYO = 10
    LOW_YOYO = 11

    _norm_delta_heading = np.array([-np.pi / 6, -np.pi / 12, 0.0, np.pi / 12, np.pi / 6], dtype=np.float32)

    def _safe_unit_vector(self, vector, fallback=None):
        vector = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm > 1e-6:
            return vector / norm
        if fallback is None:
            return np.zeros_like(vector, dtype=np.float32)
        fallback = np.asarray(fallback, dtype=np.float32)
        fallback_norm = np.linalg.norm(fallback)
        if fallback_norm > 1e-6:
            return fallback / fallback_norm
        return np.zeros_like(vector, dtype=np.float32)

    def _ego_heading_vector(self, env: SingleCombatEnv, agent_id: str):
        ego = env.agents[agent_id]
        ego_xy = np.asarray(ego.get_velocity()[:2], dtype=np.float32)
        if np.linalg.norm(ego_xy) > 1e-6:
            return self._safe_unit_vector(ego_xy)
        heading = ego.get_property_value(c.attitude_heading_true_rad)
        return np.array([np.cos(heading), np.sin(heading)], dtype=np.float32)

    def _rotate_vector(self, vector, angle):
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        return np.array([
            vector[0] * cos_angle - vector[1] * sin_angle,
            vector[0] * sin_angle + vector[1] * cos_angle,
        ], dtype=np.float32)

    def _combat_geometry(self, env: SingleCombatEnv, agent_id: str):
        ego = env.agents[agent_id]
        enm = ego.enemies[0]
        ego_pos = np.asarray(ego.get_position(), dtype=np.float32)
        enm_pos = np.asarray(enm.get_position(), dtype=np.float32)
        ego_vel = np.asarray(ego.get_velocity(), dtype=np.float32)
        enm_vel = np.asarray(enm.get_velocity(), dtype=np.float32)

        relative = enm_pos - ego_pos
        relative_xy = relative[:2]
        distance_xy = np.linalg.norm(relative_xy)
        if distance_xy < 1e-6:
            relative_xy = self._ego_heading_vector(env, agent_id)
            distance_xy = 0.0

        return {
            "ego": ego,
            "enm": enm,
            "ego_vel_xy": ego_vel[:2],
            "enm_vel_xy": enm_vel[:2],
            "relative_xy": relative_xy.astype(np.float32),
            "distance_xy": float(distance_xy),
            "ego_heading_xy": self._ego_heading_vector(env, agent_id),
            "enm_heading_xy": self._safe_unit_vector(enm_vel[:2], fallback=relative_xy),
        }

    def _heading_error_to_vector(self, env: SingleCombatEnv, agent_id: str, target_xy: np.ndarray) -> float:
        target_xy = np.asarray(target_xy, dtype=np.float32)
        if np.linalg.norm(target_xy) < 1e-6:
            return 0.0
        ego_xy = self._ego_heading_vector(env, agent_id)
        target_xy = self._safe_unit_vector(target_xy, fallback=ego_xy)
        dot = np.clip(np.dot(ego_xy, target_xy), -1.0, 1.0)
        cross_z = ego_xy[0] * target_xy[1] - ego_xy[1] * target_xy[0]
        heading_error = np.arctan2(cross_z, dot)
        return float(np.clip(heading_error, -np.pi / 6, np.pi / 6))

    def _quantize_heading_error(self, heading_error: float) -> float:
        index = np.argmin(np.abs(self._norm_delta_heading - heading_error))
        return float(self._norm_delta_heading[index])

    def _altitude_step_to_enemy(self, env: SingleCombatEnv, agent_id: str) -> float:
        ego = env.agents[agent_id]
        enm = ego.enemies[0]
        delta_altitude = float(enm.get_position()[2] - ego.get_position()[2])
        if delta_altitude > 250:
            return 0.1
        if delta_altitude < -250:
            return -0.1
        return 0.0

    def _lead_target_vector(self, geometry):
        ego_speed = np.linalg.norm(geometry["ego_vel_xy"])
        lead_time = np.clip(geometry["distance_xy"] / (ego_speed + 1e-6), 1.0, 3.0)
        return geometry["relative_xy"] + geometry["enm_vel_xy"] * lead_time

    def _lag_target_vector(self, geometry):
        lag_distance = np.clip(geometry["distance_xy"] * 0.35, 300.0, 900.0)
        return geometry["relative_xy"] - geometry["enm_heading_xy"] * lag_distance

    def _blend_with_enemy_vector(self, geometry, enemy_weight=0.7):
        enemy_vector = self._safe_unit_vector(geometry["relative_xy"], fallback=geometry["ego_heading_xy"])
        ego_vector = geometry["ego_heading_xy"]
        return enemy_vector * enemy_weight + ego_vector * (1.0 - enemy_weight)

    def _incoming_missile_geometry(self, env: SingleCombatEnv, agent_id: str):
        ego = env.agents[agent_id]
        missile_sim = ego.check_missile_warning()
        if missile_sim is None:
            return None

        ego_pos = np.asarray(ego.get_position(), dtype=np.float32)
        missile_pos = np.asarray(missile_sim.get_position(), dtype=np.float32)
        ego_vel = np.asarray(ego.get_velocity(), dtype=np.float32)
        missile_vel = np.asarray(missile_sim.get_velocity(), dtype=np.float32)

        relative = missile_pos - ego_pos
        relative_xy = relative[:2]
        distance = float(np.linalg.norm(relative))
        los_xy = self._safe_unit_vector(relative_xy, fallback=self._ego_heading_vector(env, agent_id))
        relative_velocity = missile_vel - ego_vel
        closing_speed = max(0.0, -float(np.dot(relative, relative_velocity) / (distance + 1e-6)))

        danger_distance = getattr(env.config, "tactical_missile_danger_distance", 9000.0)
        watch_distance = getattr(env.config, "max_attack_distance", 14000.0)
        danger_closing_speed = getattr(env.config, "tactical_missile_danger_closing_speed", 150.0)
        dangerous = distance <= danger_distance or (
            distance <= watch_distance and closing_speed >= danger_closing_speed
        )

        return {
            "distance": distance,
            "closing_speed": closing_speed,
            "los_xy": los_xy,
            "away_xy": -los_xy,
            "left_beam_xy": self._rotate_vector(los_xy, np.pi / 2),
            "right_beam_xy": self._rotate_vector(los_xy, -np.pi / 2),
            "dangerous": dangerous,
        }

    def _mix_tactical_vectors(self, primary, secondary, primary_weight):
        primary_vector = self._safe_unit_vector(primary)
        secondary_vector = self._safe_unit_vector(secondary, fallback=primary_vector)
        return primary_vector * primary_weight + secondary_vector * (1.0 - primary_weight)

    def _preferred_missile_beam(self, missile_geometry, ego_heading_xy):
        left_score = np.dot(ego_heading_xy, missile_geometry["left_beam_xy"])
        right_score = np.dot(ego_heading_xy, missile_geometry["right_beam_xy"])
        return missile_geometry["left_beam_xy"] if left_score >= right_score else missile_geometry["right_beam_xy"]

    def _apply_tactical_safety(self, env: SingleCombatEnv, agent_id: str, action_id: int,
                               delta_altitude: float, delta_velocity: float, distance_xy: float):
        ego = env.agents[agent_id]
        altitude = ego.get_property_value(c.position_h_sl_m)
        altitude_limit = 2500  # 默认低高度限制

        if delta_altitude < 0.0 and altitude <= altitude_limit + 1000:
            delta_altitude = 0.1 if altitude <= altitude_limit + 500 else 0.0

        close_distance = min(3.0 * 1000 * 0.35, 1200.0)
        if action_id in (self.PURE_PURSUIT, self.LEAD_PURSUIT) and distance_xy < close_distance:
            delta_velocity = 0.0

        return delta_altitude, delta_velocity

    def _missile_aware_action_to_delta_control(
        self,
        env: SingleCombatEnv,
        agent_id: str,
        action_id: int,
        missile_geometry,
    ) -> np.ndarray:
        geometry = self._combat_geometry(env, agent_id)
        relative_xy = geometry["relative_xy"]
        ego_heading_xy = geometry["ego_heading_xy"]
        distance_xy = geometry["distance_xy"]
        preferred_beam_xy = self._preferred_missile_beam(missile_geometry, ego_heading_xy)
        close_threat_distance = getattr(env.config, "tactical_missile_close_distance", 3500.0)
        close_threat = missile_geometry["distance"] <= close_threat_distance

        delta_altitude = 0.0
        delta_velocity = 0.0

        if action_id in (self.PURE_PURSUIT, self.LEAD_PURSUIT, self.LAG_PURSUIT):
            # 来袭导弹较近时，追击类动作改为带横切角的 crank，避免持续正对敌机暴露。
            beam_weight = 0.8 if close_threat else 0.6
            target_xy = self._mix_tactical_vectors(preferred_beam_xy, relative_xy, beam_weight)
            delta_altitude = self._altitude_step_to_enemy(env, agent_id)
            delta_velocity = 0.0 if close_threat else 0.05
        elif action_id == self.DISENGAGE:
            # 脱离动作优先远离来袭导弹，同时保留少量当前航向惯性。
            target_xy = self._mix_tactical_vectors(missile_geometry["away_xy"], ego_heading_xy, 0.85)
            delta_velocity = 0.05
        elif action_id == self.CLIMB_POSITION:
            target_xy = self._mix_tactical_vectors(preferred_beam_xy, relative_xy, 0.7)
            delta_altitude = 0.1
        elif action_id == self.DIVE_ACCELERATE:
            target_xy = self._mix_tactical_vectors(missile_geometry["away_xy"], preferred_beam_xy, 0.75)
            delta_altitude = -0.1
            delta_velocity = 0.05
        elif action_id == self.LEVEL_ACCELERATE:
            target_xy = self._mix_tactical_vectors(missile_geometry["away_xy"], preferred_beam_xy, 0.6)
            delta_velocity = 0.05
        elif action_id == self.LEVEL_DECELERATE:
            # 导弹威胁下不主动减速，保留能量做横切。
            target_xy = preferred_beam_xy
        elif action_id == self.DEFENSIVE_TURN_LEFT:
            # 左/右防御转弯改为相对导弹视线的 beam/notch。
            target_xy = missile_geometry["left_beam_xy"]
            delta_velocity = 0.05
        elif action_id == self.DEFENSIVE_TURN_RIGHT:
            target_xy = missile_geometry["right_beam_xy"]
            delta_velocity = 0.05
        elif action_id == self.HIGH_YOYO:
            target_xy = self._mix_tactical_vectors(preferred_beam_xy, relative_xy, 0.65)
            delta_altitude = 0.1
        elif action_id == self.LOW_YOYO:
            target_xy = self._mix_tactical_vectors(preferred_beam_xy, missile_geometry["away_xy"], 0.55)
            delta_altitude = -0.1
            delta_velocity = 0.05
        else:
            raise ValueError(f"未知 tactical action id: {action_id}")

        delta_altitude, delta_velocity = self._apply_tactical_safety(
            env, agent_id, action_id, delta_altitude, delta_velocity, distance_xy
        )
        delta_heading = self._quantize_heading_error(self._heading_error_to_vector(env, agent_id, target_xy))
        return np.array([delta_altitude, delta_heading, delta_velocity], dtype=np.float32)

    def _tactical_action_to_delta_control(
        self,
        env: SingleCombatEnv,
        agent_id: str,
        action_id: int,
    ) -> np.ndarray:
        if self.is_shoot:
            missile_geometry = self._incoming_missile_geometry(env, agent_id)
            if missile_geometry is not None and missile_geometry["dangerous"]:
                return self._missile_aware_action_to_delta_control(
                    env, agent_id, action_id, missile_geometry
                )

        geometry = self._combat_geometry(env, agent_id)
        relative_xy = geometry["relative_xy"]
        ego_heading_xy = geometry["ego_heading_xy"]
        distance_xy = geometry["distance_xy"]

        delta_altitude = 0.0
        delta_velocity = 0.0

        if action_id == self.PURE_PURSUIT:
            target_xy = relative_xy
            delta_altitude = self._altitude_step_to_enemy(env, agent_id)
            delta_velocity = 0.05
        elif action_id == self.LEAD_PURSUIT:
            target_xy = self._lead_target_vector(geometry)
            delta_altitude = self._altitude_step_to_enemy(env, agent_id)
            delta_velocity = 0.05 if distance_xy > 2000.0 else 0.0
        elif action_id == self.LAG_PURSUIT:
            target_xy = self._lag_target_vector(geometry)
            delta_altitude = self._altitude_step_to_enemy(env, agent_id)
            delta_velocity = -0.05 if distance_xy < 1500.0 else 0.0
        elif action_id == self.DISENGAGE:
            target_xy = -relative_xy
            delta_velocity = 0.05
        elif action_id == self.CLIMB_POSITION:
            target_xy = relative_xy
            delta_altitude = 0.1
        elif action_id == self.DIVE_ACCELERATE:
            target_xy = relative_xy
            delta_altitude = -0.1
            delta_velocity = 0.05
        elif action_id == self.LEVEL_ACCELERATE:
            target_xy = ego_heading_xy
            delta_velocity = 0.05
        elif action_id == self.LEVEL_DECELERATE:
            target_xy = ego_heading_xy
            delta_velocity = -0.05
        elif action_id == self.DEFENSIVE_TURN_LEFT:
            target_xy = self._rotate_vector(ego_heading_xy, np.pi / 2)
        elif action_id == self.DEFENSIVE_TURN_RIGHT:
            target_xy = self._rotate_vector(ego_heading_xy, -np.pi / 2)
        elif action_id == self.HIGH_YOYO:
            target_xy = self._blend_with_enemy_vector(geometry, enemy_weight=0.7)
            delta_altitude = 0.1
            delta_velocity = -0.05
        elif action_id == self.LOW_YOYO:
            target_xy = self._blend_with_enemy_vector(geometry, enemy_weight=0.7)
            delta_altitude = -0.1
            delta_velocity = 0.05
        else:
            raise ValueError(f"未知 tactical action id: {action_id}")

        delta_altitude, delta_velocity = self._apply_tactical_safety(
            env, agent_id, action_id, delta_altitude, delta_velocity, distance_xy
        )
        delta_heading = self._quantize_heading_error(self._heading_error_to_vector(env, agent_id, target_xy))
        return np.array([delta_altitude, delta_heading, delta_velocity], dtype=np.float32)

    def _tactical_to_lowlevel(
        self,
        obs: np.ndarray,
        raw_action: np.ndarray,
        env: SingleCombatEnv,
        agent_id: str,
    ) -> np.ndarray:
        action_id = int(np.asarray(raw_action).reshape(-1)[0])
        delta_control = self._tactical_action_to_delta_control(env, agent_id, action_id)
        return self._delta_control_to_lowlevel(obs, delta_control)


def load_lowlevel_policy(device: torch.device) -> PPOActor:
    if not LOWLEVEL_ACTOR_PATH.exists():
        raise FileNotFoundError(f"找不到低层控制器: {LOWLEVEL_ACTOR_PATH}")

    obs_space = spaces.Box(low=-10, high=10.0, shape=(12,))
    act_space = spaces.MultiDiscrete([41, 41, 41, 30])
    policy = PPOActor(make_actor_args(use_prior=False), obs_space, act_space, device=device)
    policy.load_state_dict(torch.load(str(LOWLEVEL_ACTOR_PATH), map_location=device))
    policy.eval()
    return policy


def load_controller(
    name: str,
    actor_path: Path,
    scenario_name: str,
    lowlevel_policy: PPOActor,
    device: torch.device,
) -> ActorController:
    if not actor_path.exists():
        raise FileNotFoundError(f"{name} actor 文件不存在: {actor_path}")

    # 用 actor 自己的训练场景读取 observation/action space。
    actor_env = SingleCombatEnv(scenario_name)
    try:
        task_name = getattr(actor_env.config, "task", "")
        family = task_family(task_name)
        obs_dim = int(actor_env.observation_space.shape[0])
        actor_action_dim = action_dim(actor_env.action_space)
        actor = PPOActor(
            make_actor_args(use_prior=True),
            actor_env.observation_space,
            actor_env.action_space,
            device=device,
        )
        actor.load_state_dict(torch.load(str(actor_path), map_location=device))
        actor.eval()
    finally:
        actor_env.close()

    is_tactical = task_name.startswith("tactical_hierarchical_")
    controller = ActorController(
        name=name,
        path=actor_path,
        scenario_name=scenario_name,
        task_name=task_name,
        family=family,
        is_hierarchical=task_name.startswith("hierarchical_") or is_tactical,
        is_tactical=is_tactical,
        is_shoot="shoot" in task_name,
        obs_dim=obs_dim,
        actor_action_dim=actor_action_dim,
        actor=actor,
        lowlevel_policy=lowlevel_policy,
        device=device,
    )
    controller.reset()
    return controller


def validate_setup(env: SingleCombatEnv, controllers: Iterable[ActorController]) -> None:
    eval_task = getattr(env.config, "task", "")
    if eval_task not in DIRECT_EVAL_TASKS:
        raise ValueError(
            f"EVAL_SCENARIO_NAME 必须是直接控制版 SingleCombat 场景，当前 task={eval_task}。"
        )
    if env.num_agents != 2:
        raise ValueError(
            "评估场景必须有两个 RL agent；请不要使用 use_baseline: true 的 vsBaseline 场景。"
        )

    eval_family = task_family(eval_task)
    eval_obs_dim = int(env.observation_space.shape[0])
    eval_act_dim = action_dim(env.action_space)

    for controller in controllers:
        if controller.family != eval_family:
            raise ValueError(
                f"{controller.name} 的训练任务族是 {controller.family}，"
                f"但评估场景任务族是 {eval_family}。"
            )

        # 分层 actor 会被转换成直接底层动作，因此最终动作维度必须匹配评估环境。
        expected_dim = 5 if controller.is_shoot else 4
        if expected_dim != eval_act_dim:
            raise ValueError(
                f"{controller.name} 转换后的动作维度为 {expected_dim}，"
                f"但评估环境需要 {eval_act_dim}。"
            )

        # 同一任务族的直接/分层 actor 应共享观测维度。
        if controller.obs_dim != eval_obs_dim:
            raise ValueError(
                f"{controller.name} 观测维度与评估环境不一致: "
                f"actor={controller.obs_dim}, env={eval_obs_dim}"
            )


def config_range(config: Dict[str, object], key: str, default: Tuple[float, float]) -> Tuple[float, float]:
    """从配置中读取 [min, max] 范围；单个数值会被视为固定值。"""
    value = config.get(key, default)
    if isinstance(value, (int, float)):
        return float(value), float(value)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])
    raise ValueError(f"随机初始态配置 {key} 必须是单个数值或长度为 2 的范围，当前值={value}")


def sample_uniform(rng, config: Dict[str, object], key: str, default: Tuple[float, float]) -> float:
    low, high = config_range(config, key, default)
    if low > high:
        raise ValueError(f"随机初始态配置 {key} 下界不能大于上界: {low} > {high}")
    return float(rng.uniform(low, high))


def wrap_heading_deg(value: float) -> float:
    """把航向角约束到 [0, 360) deg。"""
    return float(value % 360.0)


def noisy_heading_deg(rng, base_heading_deg: float, noise_deg: float) -> float:
    return wrap_heading_deg(base_heading_deg + float(rng.uniform(-noise_deg, noise_deg)))


def sample_heading_pair(rng, bearing_deg: float, config: Dict[str, object]) -> Tuple[float, float, str]:
    """根据交战几何采样双方初始航向。"""
    mode = str(config.get("heading_mode", "mixed")).lower()
    if mode == "mixed":
        mode = str(rng.choice(["head_on", "tail_chase", "crossing", "random"]))

    noise_deg = float(config.get("heading_noise_deg", 20.0))
    if mode == "head_on":
        a_heading = noisy_heading_deg(rng, bearing_deg, noise_deg)
        b_heading = noisy_heading_deg(rng, bearing_deg + 180.0, noise_deg)
    elif mode == "tail_chase":
        a_heading = noisy_heading_deg(rng, bearing_deg, noise_deg)
        b_heading = noisy_heading_deg(rng, bearing_deg, noise_deg)
    elif mode == "crossing":
        sign = float(rng.choice([-1.0, 1.0]))
        a_heading = noisy_heading_deg(rng, bearing_deg + sign * 90.0, noise_deg)
        b_heading = noisy_heading_deg(rng, bearing_deg - sign * 90.0, noise_deg)
    elif mode == "random":
        a_heading = float(rng.uniform(0.0, 360.0))
        b_heading = float(rng.uniform(0.0, 360.0))
    else:
        raise ValueError(f"未知 heading_mode: {mode}")
    return a_heading, b_heading, mode


def sample_random_initial_states(
    env: SingleCombatEnv,
    base_init_states: List[Dict[str, object]],
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """为 1v1 每局生成新的初始交战几何。"""
    if len(base_init_states) != 2:
        raise ValueError("随机初始态采样目前只支持 1v1 两架飞机。")
    if NEU2LLA is None:
        raise RuntimeError("缺少 NEU2LLA，无法生成随机经纬度初始状态。")

    rng = env.np_random
    config = RANDOM_INITIAL_STATE_CONFIG
    horizontal_distance_m = sample_uniform(rng, config, "distance_m", (800.0, 6000.0))
    center_north_m = sample_uniform(rng, config, "center_north_m", (-1500.0, 1500.0))
    center_east_m = sample_uniform(rng, config, "center_east_m", (-1500.0, 1500.0))
    altitude_ft = sample_uniform(rng, config, "altitude_ft", (16000.0, 26000.0))
    altitude_difference_ft = sample_uniform(
        rng, config, "altitude_difference_ft", (-3000.0, 3000.0)
    )
    a_speed_fps = sample_uniform(rng, config, "speed_fps", (650.0, 950.0))
    b_speed_fps = sample_uniform(rng, config, "speed_fps", (650.0, 950.0))

    bearing_rad = float(rng.uniform(0.0, 2.0 * np.pi))
    bearing_deg = wrap_heading_deg(np.rad2deg(bearing_rad))
    direction_ne = np.array([np.cos(bearing_rad), np.sin(bearing_rad)], dtype=np.float64)
    half_distance = horizontal_distance_m / 2.0

    a_ne = np.array([center_north_m, center_east_m], dtype=np.float64) - direction_ne * half_distance
    b_ne = np.array([center_north_m, center_east_m], dtype=np.float64) + direction_ne * half_distance
    a_alt_ft = altitude_ft - altitude_difference_ft / 2.0
    b_alt_ft = altitude_ft + altitude_difference_ft / 2.0
    a_alt_m = a_alt_ft * 0.3048
    b_alt_m = b_alt_ft * 0.3048

    a_lon, a_lat, _ = NEU2LLA(
        a_ne[0], a_ne[1], a_alt_m, env.center_lon, env.center_lat, env.center_alt
    )
    b_lon, b_lat, _ = NEU2LLA(
        b_ne[0], b_ne[1], b_alt_m, env.center_lon, env.center_lat, env.center_alt
    )
    a_heading_deg, b_heading_deg, geometry_mode = sample_heading_pair(rng, bearing_deg, config)

    init_states = [state.copy() for state in base_init_states]
    sampled_values = [
        (a_lon, a_lat, a_alt_ft, a_heading_deg, a_speed_fps),
        (b_lon, b_lat, b_alt_ft, b_heading_deg, b_speed_fps),
    ]
    for init_state, (lon, lat, alt_ft, heading_deg, speed_fps) in zip(init_states, sampled_values):
        init_state.update(
            {
                "ic_long_gc_deg": float(lon),
                "ic_lat_geod_deg": float(lat),
                "ic_h_sl_ft": float(alt_ft),
                "ic_psi_true_deg": float(heading_deg),
                "ic_u_fps": float(speed_fps),
                "ic_v_fps": 0.0,
                "ic_w_fps": 0.0,
                "ic_p_rad_sec": 0.0,
                "ic_q_rad_sec": 0.0,
                "ic_r_rad_sec": 0.0,
                "ic_roc_fpm": 0.0,
            }
        )

    return init_states, {
        "init_geometry_mode": geometry_mode,
        "sampled_horizontal_distance_m": float(horizontal_distance_m),
        "init_bearing_deg": float(bearing_deg),
    }


def build_initial_state_info(
    env: SingleCombatEnv,
    assigned_init_states: List[Dict[str, object]],
    extra_info: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """记录本局初始态，便于后续按场景分组分析。"""
    agent_ids = list(env.agents.keys())
    info = {
        "initial_state_mode": INITIAL_STATE_MODE,
        "init_geometry_mode": "",
        "sampled_horizontal_distance_m": "",
        "init_bearing_deg": "",
        "init_distance_m": "",
        "agent_states": {},
    }
    if extra_info:
        info.update(extra_info)

    for agent_id, init_state in zip(agent_ids, assigned_init_states):
        info["agent_states"][agent_id] = {
            "init_altitude_ft": float(init_state.get("ic_h_sl_ft", 0.0)),
            "init_heading_deg": float(init_state.get("ic_psi_true_deg", 0.0)),
            "init_speed_fps": float(init_state.get("ic_u_fps", 0.0)),
        }

    if len(agent_ids) >= 2:
        first = np.asarray(env.agents[agent_ids[0]].get_position(), dtype=np.float64)
        second = np.asarray(env.agents[agent_ids[1]].get_position(), dtype=np.float64)
        info["init_distance_m"] = float(np.linalg.norm(second - first))
    return info


def reset_eval_env(env: SingleCombatEnv) -> np.ndarray:
    """按脚本配置重置 1v1 环境，支持固定或随机初始态。"""
    env.current_step = 0

    # 保存一份不随 sim.reload() 改变的基准初始状态，避免随机换边后下一局状态漂移。
    if not hasattr(env, "_experiment_base_init_states"):
        base_init_states = []
        for agent_id, sim in env.agents.items():
            init_state = sim.init_state.copy()
            if CUSTOM_INITIAL_STATES and agent_id in CUSTOM_INITIAL_STATES:
                init_state.update(CUSTOM_INITIAL_STATES[agent_id])
            base_init_states.append(init_state)
        env._experiment_base_init_states = base_init_states

    env.init_states = [state.copy() for state in env._experiment_base_init_states]

    if INITIAL_STATE_MODE == "random":
        init_states, initial_extra_info = sample_random_initial_states(env, env.init_states)
    elif INITIAL_STATE_MODE == "fixed":
        init_states = [state.copy() for state in env.init_states]
        initial_extra_info = None
    else:
        raise ValueError(f"未知 INITIAL_STATE_MODE: {INITIAL_STATE_MODE}")

    if INITIAL_STATE_MODE == "fixed" and RANDOM_SIDE_SWAP:
        env.np_random.shuffle(init_states)

    for sim, init_state in zip(env.agents.values(), init_states):
        sim.reload(init_state)
    env._tempsims.clear()
    env.task.reset(env)
    env._experiment_last_initial_state_info = build_initial_state_info(
        env, init_states, initial_extra_info
    )
    return env._pack(env.get_obs())


def status_name(sim) -> str:
    if sim.is_alive:
        return "alive"
    if sim.is_crash:
        return "crash"
    if sim.is_shotdown:
        return "shotdown"
    return "unknown"


def missile_stats(env: SingleCombatEnv, agent_id: str) -> Dict[str, int]:
    sim = env.agents[agent_id]
    launched = len(sim.launch_missiles)
    hits = sum(1 for missile in sim.launch_missiles if missile.is_success)
    remaining_by_task = getattr(env.task, "remaining_missiles", {})
    remaining = remaining_by_task.get(agent_id, max(int(sim.num_missiles) - launched, 0))
    return {
        "missiles_launched": int(launched),
        "missile_hits": int(hits),
        "missiles_remaining": int(remaining),
    }


def decide_winner(
    actor_a_reward: float,
    actor_b_reward: float,
    actor_a_status: str,
    actor_b_status: str,
) -> str:
    actor_a_alive = actor_a_status == "alive"
    actor_b_alive = actor_b_status == "alive"
    if actor_a_alive and not actor_b_alive:
        return "actor_a"
    if actor_b_alive and not actor_a_alive:
        return "actor_b"

    reward_margin = actor_a_reward - actor_b_reward
    if reward_margin > WIN_REWARD_MARGIN:
        return "actor_a"
    if reward_margin < -WIN_REWARD_MARGIN:
        return "actor_b"
    return "tie"


def make_output_dir() -> Path:
    if OUTPUT_DIR is not None:
        output_dir = Path(OUTPUT_DIR)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(OUTPUT_ROOT) / f"{EXPERIMENT_NAME}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    if SAVE_ACMI:
        (output_dir / "acmi").mkdir(parents=True, exist_ok=True)
    if SAVE_PLOTS:
        (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    return output_dir


def save_csv(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: Path) -> List[Dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def save_summary(summary: Dict[str, object], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def load_summary(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def average(rows: List[Dict[str, object]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows])) if rows else 0.0


def metric_values(rows: List[Dict[str, object]], key: str) -> List[float]:
    """读取某个指标列，兼容旧版 episodes.csv 中不存在新指标的情况。"""
    values = []
    for row in rows:
        if key not in row or row[key] in ("", None):
            continue
        values.append(float(row[key]))
    return values


def metric_mean(rows: List[Dict[str, object]], key: str) -> float:
    values = metric_values(rows, key)
    return float(np.mean(values)) if values else 0.0


def metric_std(rows: List[Dict[str, object]], key: str) -> float:
    """按回合样本计算标准差；只有 1 个样本时记为 0。"""
    values = metric_values(rows, key)
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def has_metric(rows: List[Dict[str, object]], key: str) -> bool:
    return bool(metric_values(rows, key))


def build_summary(rows: List[Dict[str, object]], output_dir: Path) -> Dict[str, object]:
    total = len(rows)
    actor_a_wins = sum(row["winner"] == "actor_a" for row in rows)
    actor_b_wins = sum(row["winner"] == "actor_b" for row in rows)
    ties = sum(row["winner"] == "tie" for row in rows)

    actor_a_launched = sum(int(row["actor_a_missiles_launched"]) for row in rows)
    actor_b_launched = sum(int(row["actor_b_missiles_launched"]) for row in rows)
    actor_a_hits = sum(int(row["actor_a_missile_hits"]) for row in rows)
    actor_b_hits = sum(int(row["actor_b_missile_hits"]) for row in rows)

    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "output_dir": str(output_dir),
        "eval_scenario_name": EVAL_SCENARIO_NAME,
        "actor_a_path": str(ACTOR_A_PATH),
        "actor_a_scenario_name": ACTOR_A_SCENARIO_NAME,
        "actor_b_path": str(ACTOR_B_PATH),
        "actor_b_scenario_name": ACTOR_B_SCENARIO_NAME,
        "num_episodes": total,
        "initial_state_mode": INITIAL_STATE_MODE,
        "random_side_swap": RANDOM_SIDE_SWAP,
        "random_initial_state_config": RANDOM_INITIAL_STATE_CONFIG,
        "actor_a_win_rate": actor_a_wins / total if total else 0.0,
        "actor_b_win_rate": actor_b_wins / total if total else 0.0,
        "tie_rate": ties / total if total else 0.0,
        "actor_a_avg_reward": average(rows, "actor_a_reward"),
        "actor_b_avg_reward": average(rows, "actor_b_reward"),
        "actor_a_reward_std": metric_std(rows, "actor_a_reward"),
        "actor_b_reward_std": metric_std(rows, "actor_b_reward"),
        "avg_reward_margin": average(rows, "reward_margin"),
        "reward_margin_std": metric_std(rows, "reward_margin"),
        "avg_steps": average(rows, "steps"),
        "avg_duration_sec": average(rows, "duration_sec"),
        "actor_a_crash_rate": sum(row["actor_a_status"] == "crash" for row in rows) / total if total else 0.0,
        "actor_b_crash_rate": sum(row["actor_b_status"] == "crash" for row in rows) / total if total else 0.0,
        "actor_a_shotdown_rate": sum(row["actor_a_status"] == "shotdown" for row in rows) / total if total else 0.0,
        "actor_b_shotdown_rate": sum(row["actor_b_status"] == "shotdown" for row in rows) / total if total else 0.0,
        "actor_a_missiles_launched": actor_a_launched,
        "actor_b_missiles_launched": actor_b_launched,
        "actor_a_missile_hits": actor_a_hits,
        "actor_b_missile_hits": actor_b_hits,
        "actor_a_missile_hit_rate": actor_a_hits / actor_a_launched if actor_a_launched else 0.0,
        "actor_b_missile_hit_rate": actor_b_hits / actor_b_launched if actor_b_launched else 0.0,
    }
    if has_metric(rows, "final_distance_m"):
        summary.update(
            {
                "final_distance_avg_m": metric_mean(rows, "final_distance_m"),
                "final_distance_std_m": metric_std(rows, "final_distance_m"),
                "final_distance_avg_km": metric_mean(rows, "final_distance_m") / 1000.0,
                "final_distance_std_km": metric_std(rows, "final_distance_m") / 1000.0,
            }
        )

    for prefix in ("actor_a", "actor_b"):
        metric_prefix = f"{prefix}_final"
        if has_metric(rows, f"{metric_prefix}_ao_deg"):
            summary.update(
                {
                    f"{metric_prefix}_ao_avg_deg": metric_mean(rows, f"{metric_prefix}_ao_deg"),
                    f"{metric_prefix}_ao_std_deg": metric_std(rows, f"{metric_prefix}_ao_deg"),
                    f"{metric_prefix}_ta_avg_deg": metric_mean(rows, f"{metric_prefix}_ta_deg"),
                    f"{metric_prefix}_ta_std_deg": metric_std(rows, f"{metric_prefix}_ta_deg"),
                }
            )

        metric_prefix = f"{prefix}_episode_mean"
        if has_metric(rows, f"{metric_prefix}_ao_deg"):
            summary.update(
                {
                    f"{metric_prefix}_ao_avg_deg": metric_mean(rows, f"{metric_prefix}_ao_deg"),
                    f"{metric_prefix}_ao_std_deg": metric_std(rows, f"{metric_prefix}_ao_deg"),
                    f"{metric_prefix}_ta_avg_deg": metric_mean(rows, f"{metric_prefix}_ta_deg"),
                    f"{metric_prefix}_ta_std_deg": metric_std(rows, f"{metric_prefix}_ta_deg"),
                }
            )
    return summary


def configure_plot_style() -> None:
    """配置适合中文实验图表的 matplotlib 样式。"""
    if plt is None:
        return
    plt.rcParams.update(
        {
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "figure.autolayout": True,
        }
    )


def numeric_series(rows: List[Dict[str, object]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def safe_numeric_series(rows: List[Dict[str, object]], key: str) -> Optional[np.ndarray]:
    values = metric_values(rows, key)
    if not values:
        return None
    return np.asarray(values, dtype=np.float64)


def simulator_feature(sim) -> np.ndarray:
    """提取计算相对态势所需的 3D 位置和速度。"""
    return np.hstack([sim.get_position(), sim.get_velocity()])


def relative_state_metrics(actor_a_sim, actor_b_sim) -> Dict[str, float]:
    """计算双方相对距离、AO 和 TA；角度同时从两个 actor 视角记录。"""
    actor_a_feature = simulator_feature(actor_a_sim)
    actor_b_feature = simulator_feature(actor_b_sim)

    actor_a_ao, actor_a_ta, distance_m = get_AO_TA_R(actor_a_feature, actor_b_feature)
    actor_b_ao, actor_b_ta, _ = get_AO_TA_R(actor_b_feature, actor_a_feature)
    return {
        "distance_m": float(distance_m),
        "actor_a_ao_deg": float(np.rad2deg(actor_a_ao)),
        "actor_a_ta_deg": float(np.rad2deg(actor_a_ta)),
        "actor_b_ao_deg": float(np.rad2deg(actor_b_ao)),
        "actor_b_ta_deg": float(np.rad2deg(actor_b_ta)),
    }


def append_metric_history(history: Dict[str, List[float]], metrics: Dict[str, float]) -> None:
    for key, value in metrics.items():
        history.setdefault(key, []).append(value)


def metric_history_mean(history: Dict[str, List[float]], key: str) -> float:
    values = history.get(key, [])
    return float(np.mean(values)) if values else 0.0


def metric_history_std(history: Dict[str, List[float]], key: str) -> float:
    values = history.get(key, [])
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def summarize_relative_history(history: Dict[str, List[float]]) -> Dict[str, float]:
    """汇总单局内的末态距离、末态 AO/TA 和整局平均 AO/TA。"""
    if not history:
        return {
            "final_distance_m": 0.0,
            "actor_a_final_ao_deg": 0.0,
            "actor_a_final_ta_deg": 0.0,
            "actor_b_final_ao_deg": 0.0,
            "actor_b_final_ta_deg": 0.0,
            "actor_a_episode_mean_ao_deg": 0.0,
            "actor_a_episode_mean_ta_deg": 0.0,
            "actor_b_episode_mean_ao_deg": 0.0,
            "actor_b_episode_mean_ta_deg": 0.0,
            "actor_a_episode_std_ao_deg": 0.0,
            "actor_a_episode_std_ta_deg": 0.0,
            "actor_b_episode_std_ao_deg": 0.0,
            "actor_b_episode_std_ta_deg": 0.0,
        }

    return {
        "final_distance_m": float(history["distance_m"][-1]),
        "actor_a_final_ao_deg": float(history["actor_a_ao_deg"][-1]),
        "actor_a_final_ta_deg": float(history["actor_a_ta_deg"][-1]),
        "actor_b_final_ao_deg": float(history["actor_b_ao_deg"][-1]),
        "actor_b_final_ta_deg": float(history["actor_b_ta_deg"][-1]),
        "actor_a_episode_mean_ao_deg": metric_history_mean(history, "actor_a_ao_deg"),
        "actor_a_episode_mean_ta_deg": metric_history_mean(history, "actor_a_ta_deg"),
        "actor_b_episode_mean_ao_deg": metric_history_mean(history, "actor_b_ao_deg"),
        "actor_b_episode_mean_ta_deg": metric_history_mean(history, "actor_b_ta_deg"),
        "actor_a_episode_std_ao_deg": metric_history_std(history, "actor_a_ao_deg"),
        "actor_a_episode_std_ta_deg": metric_history_std(history, "actor_a_ta_deg"),
        "actor_b_episode_std_ao_deg": metric_history_std(history, "actor_b_ao_deg"),
        "actor_b_episode_std_ta_deg": metric_history_std(history, "actor_b_ta_deg"),
    }


def initial_state_row(
    env: SingleCombatEnv,
    actor_a_agent_id: str,
    actor_b_agent_id: str,
) -> Dict[str, object]:
    """把本局初始条件转换为 CSV 字段，方便解释不同测试场景。"""
    info = getattr(env, "_experiment_last_initial_state_info", {})
    agent_states = info.get("agent_states", {})
    actor_a_state = agent_states.get(actor_a_agent_id, {})
    actor_b_state = agent_states.get(actor_b_agent_id, {})
    return {
        "initial_state_mode": info.get("initial_state_mode", INITIAL_STATE_MODE),
        "init_geometry_mode": info.get("init_geometry_mode", ""),
        "init_distance_m": info.get("init_distance_m", ""),
        "sampled_horizontal_distance_m": info.get("sampled_horizontal_distance_m", ""),
        "init_bearing_deg": info.get("init_bearing_deg", ""),
        "actor_a_init_altitude_ft": actor_a_state.get("init_altitude_ft", ""),
        "actor_b_init_altitude_ft": actor_b_state.get("init_altitude_ft", ""),
        "actor_a_init_heading_deg": actor_a_state.get("init_heading_deg", ""),
        "actor_b_init_heading_deg": actor_b_state.get("init_heading_deg", ""),
        "actor_a_init_speed_fps": actor_a_state.get("init_speed_fps", ""),
        "actor_b_init_speed_fps": actor_b_state.get("init_speed_fps", ""),
    }


def moving_average(values: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """计算滑动平均，用于在回合数较多时观察整体趋势。"""
    if window <= 1 or values.size < window:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    averaged = np.convolve(values, kernel, mode="valid")
    x_offset = np.arange(window - 1, values.size, dtype=np.int64)
    return x_offset, averaged


def save_plot(fig, plot_dir: Path, filename: str) -> Path:
    path = plot_dir / filename
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def add_line_with_average(
    ax,
    episode_ids: np.ndarray,
    values: np.ndarray,
    label: str,
    color: str,
) -> None:
    ax.plot(episode_ids, values, marker="o", linewidth=1.8, label=label, color=color)
    avg_x, avg_y = moving_average(values, MOVING_AVERAGE_WINDOW)
    if avg_y.size:
        ax.plot(
            episode_ids[avg_x],
            avg_y,
            linestyle="--",
            linewidth=2.0,
            label=f"{label} {MOVING_AVERAGE_WINDOW}局滑动平均",
            color=color,
            alpha=0.75,
        )


def plot_reward_curve(rows: List[Dict[str, object]], plot_dir: Path) -> Path:
    episode_ids = numeric_series(rows, "episode")
    actor_a_rewards = numeric_series(rows, "actor_a_reward")
    actor_b_rewards = numeric_series(rows, "actor_b_reward")

    fig, ax = plt.subplots(figsize=(10, 5))
    add_line_with_average(ax, episode_ids, actor_a_rewards, "Actor A", ACTOR_A_COLOR)
    add_line_with_average(ax, episode_ids, actor_b_rewards, "Actor B", ACTOR_B_COLOR)
    ax.set_title("各回合累计奖励曲线")
    ax.set_xlabel("Episode")
    ax.set_ylabel("累计奖励")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return save_plot(fig, plot_dir, "reward_curve.png")


def plot_reward_margin(rows: List[Dict[str, object]], plot_dir: Path) -> Path:
    episode_ids = numeric_series(rows, "episode")
    reward_margin = numeric_series(rows, "reward_margin")

    fig, ax = plt.subplots(figsize=(10, 5))
    add_line_with_average(ax, episode_ids, reward_margin, "A-B 奖励差", "#2ca02c")
    ax.axhline(0.0, color="#333333", linewidth=1.0)
    ax.axhline(WIN_REWARD_MARGIN, color="#999999", linestyle=":", linewidth=1.2)
    ax.axhline(-WIN_REWARD_MARGIN, color="#999999", linestyle=":", linewidth=1.2)
    ax.fill_between(
        episode_ids,
        -WIN_REWARD_MARGIN,
        WIN_REWARD_MARGIN,
        color="#999999",
        alpha=0.08,
        label="平局奖励差区间",
    )
    ax.set_title("各回合奖励差曲线")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Actor A 奖励 - Actor B 奖励")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return save_plot(fig, plot_dir, "reward_margin_curve.png")


def plot_outcome_curve(rows: List[Dict[str, object]], plot_dir: Path) -> Path:
    episode_ids = numeric_series(rows, "episode")
    winners = [str(row["winner"]) for row in rows]
    episode_count = np.arange(1, len(rows) + 1, dtype=np.float64)
    actor_a_rate = np.cumsum([winner == "actor_a" for winner in winners]) / episode_count
    actor_b_rate = np.cumsum([winner == "actor_b" for winner in winners]) / episode_count
    tie_rate = np.cumsum([winner == "tie" for winner in winners]) / episode_count

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    labels = ["Actor A胜", "Actor B胜", "平局"]
    counts = [
        winners.count("actor_a"),
        winners.count("actor_b"),
        winners.count("tie"),
    ]
    bars = axes[0].bar(labels, counts, color=[ACTOR_A_COLOR, ACTOR_B_COLOR, TIE_COLOR])
    axes[0].set_title("胜负结果统计")
    axes[0].set_ylabel("回合数")
    for bar in bars:
        height = bar.get_height()
        axes[0].annotate(
            f"{int(height)}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    axes[1].plot(episode_ids, actor_a_rate, marker="o", label="Actor A累计胜率", color=ACTOR_A_COLOR)
    axes[1].plot(episode_ids, actor_b_rate, marker="o", label="Actor B累计胜率", color=ACTOR_B_COLOR)
    axes[1].plot(episode_ids, tie_rate, marker="o", label="累计平局率", color=TIE_COLOR)
    axes[1].set_title("累计胜率/平局率")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("比例")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    return save_plot(fig, plot_dir, "outcome_curve.png")


def plot_terminal_metrics(rows: List[Dict[str, object]], plot_dir: Path) -> Path:
    episode_ids = numeric_series(rows, "episode")
    steps = numeric_series(rows, "steps")
    duration_sec = numeric_series(rows, "duration_sec")
    actor_a_bloods = numeric_series(rows, "actor_a_bloods")
    actor_b_bloods = numeric_series(rows, "actor_b_bloods")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(episode_ids, steps, marker="o", label="步数", color="#9467bd")
    axes[0].plot(episode_ids, duration_sec, marker="s", label="仿真时长(s)", color="#8c564b")
    axes[0].set_title("每回合终止步数与仿真时长")
    axes[0].set_ylabel("数值")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(episode_ids, actor_a_bloods, marker="o", label="Actor A血量", color=ACTOR_A_COLOR)
    axes[1].plot(episode_ids, actor_b_bloods, marker="o", label="Actor B血量", color=ACTOR_B_COLOR)
    axes[1].set_title("每回合终局血量")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("血量")
    axes[1].set_ylim(-5, max(105.0, float(np.max([actor_a_bloods.max(), actor_b_bloods.max()])) + 5))
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    return save_plot(fig, plot_dir, "terminal_metrics.png")


def plot_tactical_metrics(rows: List[Dict[str, object]], plot_dir: Path) -> Optional[Path]:
    if not has_metric(rows, "final_distance_m"):
        return None

    episode_ids = numeric_series(rows, "episode")
    final_distance_km = numeric_series(rows, "final_distance_m") / 1000.0
    actor_a_mean_ao = safe_numeric_series(rows, "actor_a_episode_mean_ao_deg")
    actor_b_mean_ao = safe_numeric_series(rows, "actor_b_episode_mean_ao_deg")
    actor_a_mean_ta = safe_numeric_series(rows, "actor_a_episode_mean_ta_deg")
    actor_b_mean_ta = safe_numeric_series(rows, "actor_b_episode_mean_ta_deg")

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    add_line_with_average(axes[0], episode_ids, final_distance_km, "最终距离", "#9467bd")
    axes[0].set_title("最终距离曲线")
    axes[0].set_ylabel("距离(km)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    if actor_a_mean_ao is not None and actor_b_mean_ao is not None:
        add_line_with_average(axes[1], episode_ids[: actor_a_mean_ao.size], actor_a_mean_ao, "Actor A 平均 AO", ACTOR_A_COLOR)
        add_line_with_average(axes[1], episode_ids[: actor_b_mean_ao.size], actor_b_mean_ao, "Actor B 平均 AO", ACTOR_B_COLOR)
    axes[1].set_title("整局平均 AO 曲线")
    axes[1].set_ylabel("AO(deg)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    if actor_a_mean_ta is not None and actor_b_mean_ta is not None:
        add_line_with_average(axes[2], episode_ids[: actor_a_mean_ta.size], actor_a_mean_ta, "Actor A 平均 TA", ACTOR_A_COLOR)
        add_line_with_average(axes[2], episode_ids[: actor_b_mean_ta.size], actor_b_mean_ta, "Actor B 平均 TA", ACTOR_B_COLOR)
    axes[2].set_title("整局平均 TA 曲线")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("TA(deg)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    return save_plot(fig, plot_dir, "tactical_metrics.png")


def plot_status_counts(rows: List[Dict[str, object]], plot_dir: Path) -> Path:
    statuses = ["alive", "crash", "shotdown", "unknown"]
    status_labels = ["存活", "坠毁", "被击落", "未知"]
    actor_a_counts = [sum(row["actor_a_status"] == status for row in rows) for status in statuses]
    actor_b_counts = [sum(row["actor_b_status"] == status for row in rows) for status in statuses]
    x = np.arange(len(statuses))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, actor_a_counts, width, label="Actor A", color=ACTOR_A_COLOR)
    ax.bar(x + width / 2, actor_b_counts, width, label="Actor B", color=ACTOR_B_COLOR)
    ax.set_title("终局状态统计")
    ax.set_xticks(x)
    ax.set_xticklabels(status_labels)
    ax.set_ylabel("回合数")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    return save_plot(fig, plot_dir, "status_counts.png")


def plot_missile_stats(rows: List[Dict[str, object]], plot_dir: Path) -> Optional[Path]:
    missile_keys = [
        "actor_a_missiles_launched",
        "actor_b_missiles_launched",
        "actor_a_missile_hits",
        "actor_b_missile_hits",
        "actor_a_missiles_remaining",
        "actor_b_missiles_remaining",
    ]
    if not any(float(row[key]) > 0 for row in rows for key in missile_keys):
        return None

    episode_ids = numeric_series(rows, "episode")
    actor_a_launched = numeric_series(rows, "actor_a_missiles_launched")
    actor_b_launched = numeric_series(rows, "actor_b_missiles_launched")
    actor_a_hits = numeric_series(rows, "actor_a_missile_hits")
    actor_b_hits = numeric_series(rows, "actor_b_missile_hits")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    width = 0.35
    axes[0].bar(episode_ids - width / 2, actor_a_launched, width, label="Actor A发射", color=ACTOR_A_COLOR)
    axes[0].bar(episode_ids + width / 2, actor_b_launched, width, label="Actor B发射", color=ACTOR_B_COLOR)
    axes[0].set_title("各回合导弹发射数")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("发射数")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].bar(episode_ids - width / 2, actor_a_hits, width, label="Actor A命中", color=ACTOR_A_COLOR)
    axes[1].bar(episode_ids + width / 2, actor_b_hits, width, label="Actor B命中", color=ACTOR_B_COLOR)
    axes[1].set_title("各回合导弹命中数")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("命中数")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()
    return save_plot(fig, plot_dir, "missile_stats.png")


def plot_results(rows: List[Dict[str, object]], output_dir: Path) -> List[Path]:
    if not SAVE_PLOTS:
        return []
    if plt is None:
        print("未安装 matplotlib，跳过图表绘制。可安装 matplotlib 后重新运行。")
        return []
    if not rows:
        return []

    configure_plot_style()
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_paths = [
        plot_reward_curve(rows, plot_dir),
        plot_reward_margin(rows, plot_dir),
        plot_outcome_curve(rows, plot_dir),
        plot_terminal_metrics(rows, plot_dir),
        plot_status_counts(rows, plot_dir),
    ]
    tactical_plot = plot_tactical_metrics(rows, plot_dir)
    if tactical_plot is not None:
        plot_paths.append(tactical_plot)

    missile_plot = plot_missile_stats(rows, plot_dir)
    if missile_plot is not None:
        plot_paths.append(missile_plot)

    with (plot_dir / "plot_index.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "plot_files": [str(path) for path in plot_paths],
                "moving_average_window": MOVING_AVERAGE_WINDOW,
                "note": "reward_margin 为 Actor A 累计奖励减 Actor B 累计奖励。",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"图表目录: {plot_dir}")
    return plot_paths


def run_episode(
    episode: int,
    env: SingleCombatEnv,
    actor_a: ActorController,
    actor_b: ActorController,
    output_dir: Path,
) -> Dict[str, object]:
    actor_a.reset()
    actor_b.reset()
    obs = reset_eval_env(env)

    # 偶数局按 A/B，奇数局交换阵营。
    if SWAP_ACTOR_ORDER and episode % 2 == 1:
        slot_controllers = [actor_b, actor_a]
    else:
        slot_controllers = [actor_a, actor_b]

    agent_ids = (env.ego_ids + env.enm_ids)[: env.num_agents]
    actor_to_slot = {controller.name: idx for idx, controller in enumerate(slot_controllers)}
    actor_to_agent_id = {
        controller.name: agent_ids[idx] for idx, controller in enumerate(slot_controllers)
    }

    actor_rewards = {"actor_a": 0.0, "actor_b": 0.0}
    relative_history: Dict[str, List[float]] = {}
    acmi_path = None
    if SAVE_ACMI and episode in ACMI_EPISODES:
        acmi_path = output_dir / "acmi" / f"episode_{episode:04d}.txt.acmi"
        env._create_records = False
        env.render(mode="txt", filepath=str(acmi_path))

    while True:
        actions = [
            controller.act(obs[idx], env=env, agent_id=actor_to_agent_id[controller.name])
            for idx, controller in enumerate(slot_controllers)
        ]
        obs, rewards, dones, info = env.step(np.stack(actions, axis=0))

        for controller in slot_controllers:
            slot = actor_to_slot[controller.name]
            actor_rewards[controller.name] += float(rewards[slot, 0])

        actor_a_sim = env.agents[actor_to_agent_id["actor_a"]]
        actor_b_sim = env.agents[actor_to_agent_id["actor_b"]]
        append_metric_history(
            relative_history,
            relative_state_metrics(actor_a_sim, actor_b_sim),
        )

        if acmi_path is not None:
            env.render(mode="txt", filepath=str(acmi_path))

        if bool(np.asarray(dones).all()):
            break

    actor_a_agent_id = actor_to_agent_id["actor_a"]
    actor_b_agent_id = actor_to_agent_id["actor_b"]
    actor_a_sim = env.agents[actor_a_agent_id]
    actor_b_sim = env.agents[actor_b_agent_id]

    actor_a_status = status_name(actor_a_sim)
    actor_b_status = status_name(actor_b_sim)
    winner = decide_winner(
        actor_rewards["actor_a"],
        actor_rewards["actor_b"],
        actor_a_status,
        actor_b_status,
    )

    actor_a_missiles = missile_stats(env, actor_a_agent_id)
    actor_b_missiles = missile_stats(env, actor_b_agent_id)
    relative_summary = summarize_relative_history(relative_history)
    initial_summary = initial_state_row(env, actor_a_agent_id, actor_b_agent_id)

    row = {
        "episode": episode,
        "winner": winner,
        "steps": int(env.current_step),
        "duration_sec": float(env.current_step * env.time_interval),
        "actor_a_agent_id": actor_a_agent_id,
        "actor_b_agent_id": actor_b_agent_id,
        "actor_a_reward": actor_rewards["actor_a"],
        "actor_b_reward": actor_rewards["actor_b"],
        "reward_margin": actor_rewards["actor_a"] - actor_rewards["actor_b"],
        **initial_summary,
        **relative_summary,
        "actor_a_bloods": float(actor_a_sim.bloods),
        "actor_b_bloods": float(actor_b_sim.bloods),
        "actor_a_status": actor_a_status,
        "actor_b_status": actor_b_status,
        "actor_a_missiles_launched": actor_a_missiles["missiles_launched"],
        "actor_b_missiles_launched": actor_b_missiles["missiles_launched"],
        "actor_a_missile_hits": actor_a_missiles["missile_hits"],
        "actor_b_missile_hits": actor_b_missiles["missile_hits"],
        "actor_a_missiles_remaining": actor_a_missiles["missiles_remaining"],
        "actor_b_missiles_remaining": actor_b_missiles["missiles_remaining"],
        "acmi_path": str(acmi_path) if acmi_path is not None else "",
    }
    print(
        f"[{episode + 1}/{NUM_EPISODES}] winner={winner} "
        f"A_reward={row['actor_a_reward']:.2f} B_reward={row['actor_b_reward']:.2f} "
        f"final_dist={row['final_distance_m'] / 1000.0:.2f}km steps={row['steps']}"
    )
    return row


def plot_existing_results(result_dir: Path) -> None:
    """只读取已有 episodes.csv 并补充生成图表。"""
    episodes_path = result_dir / "episodes.csv"
    summary_path = result_dir / "summary.json"
    if not episodes_path.exists():
        raise FileNotFoundError(f"未找到已有结果文件: {episodes_path}")

    rows = load_csv(episodes_path)
    summary = load_summary(summary_path) or build_summary(rows, result_dir)
    plot_files = plot_results(rows, result_dir)
    summary["plot_files"] = [str(path) for path in plot_files]
    save_summary(summary, summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    apply_cli_args(args)

    if PLOT_ONLY_RESULT_DIR is not None:
        plot_existing_results(Path(PLOT_ONLY_RESULT_DIR))
        return

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = resolve_device(DEVICE)
    output_dir = make_output_dir()
    print(f"输出目录: {output_dir}")
    print(f"使用设备: {device}")

    env = SingleCombatEnv(EVAL_SCENARIO_NAME)
    env.seed(SEED)

    try:
        lowlevel_policy = load_lowlevel_policy(device)
        actor_a = load_controller(
            "actor_a",
            Path(ACTOR_A_PATH),
            ACTOR_A_SCENARIO_NAME,
            lowlevel_policy,
            device,
        )
        actor_b = load_controller(
            "actor_b",
            Path(ACTOR_B_PATH),
            ACTOR_B_SCENARIO_NAME,
            lowlevel_policy,
            device,
        )
        validate_setup(env, [actor_a, actor_b])
        rows = [
            run_episode(episode, env, actor_a, actor_b, output_dir)
            for episode in range(NUM_EPISODES)
        ]
    finally:
        env.close()

    save_csv(rows, output_dir / "episodes.csv")
    summary = build_summary(rows, output_dir)
    plot_files = plot_results(rows, output_dir)
    summary["plot_files"] = [str(path) for path in plot_files]
    save_summary(summary, output_dir / "summary.json")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
