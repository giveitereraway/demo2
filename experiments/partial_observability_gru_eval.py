#!/usr/bin/env python
"""使用冻结策略评估 1v1 观测退化下的鲁棒性。

本脚本只加载已有 actor，不执行训练。Actor A 和 Actor B 可分别加载手动指定的模型：
普通无武器 Selfplay 默认以 E0 完整观测为基线，E1–E6 逐档增加 Actor A 本机状态的噪声、丢帧和时延。
Actor B 始终接收完整观测，双方每回合按各自的编码器正常维护时序状态。
使用 --ownship-degradation 可只评估 E0 和 E2。

运行方式：
    python experiments/partial_observability_gru_eval.py

所有常用超参数都集中在下方“可调实验参数”区域。
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import deque
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:
    from envs.JSBSim.envs.selfplay_observation import OwnshipDegradationSpec
    from envs.JSBSim.envs.singlecombat_env import RadarDegradationSpec


# =============================================================================
# 可调实验参数：通常只需要修改本区域
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

# Actor 模型绝对路径：请分别手动修改为要评估的模型。
# 可以填写具体 actor_latest.pt，也可以填写包含该文件的实验目录。
ACTOR_A_PATH = Path(
    "/root/autodl-tmp/demo2/scripts/results/SingleCombat/1v1/NoWeapon/"
    "Selfplay/ppo/1v1_follow_2e8"
)
ACTOR_B_PATH = Path(
    "/root/autodl-tmp/demo2/scripts/results/SingleCombat/1v1/NoWeapon/"
    "Selfplay/ppo/1v1_follow_2e8"
)

# Actor A/B 对应的训练场景以及本次评估场景。
ACTOR_A_SCENARIO_NAME = "1v1/NoWeapon/Selfplay"
ACTOR_B_SCENARIO_NAME = "1v1/NoWeapon/Selfplay"
EVAL_SCENARIO_NAME = "1v1/NoWeapon/Selfplay"

# 基本评估参数。
NUM_EPISODES_PER_CONDITION = 100
ENV_SEED = 20260716
SENSOR_SEED = 31001
BOOTSTRAP_SEED = 41001
DEVICE = "auto"                                  # auto / cpu / cuda:0
DETERMINISTIC = True
SWAP_ACTOR_ORDER = True                          # 奇偶回合交换双方物理阵营
STRICT_WINNER = True                             # 超时双方存活时记为平局，不用训练奖励判胜

# 随机初始态。fixed 使用 YAML 初始态；random 每局重新采样。
INITIAL_STATE_MODE = "random"                    # fixed / random
RANDOM_SIDE_SWAP = True                          # fixed 模式下随机交换初始阵位
RANDOM_INITIAL_STATE_CONFIG = {
    # 默认无武器 Selfplay 训练的初始水平距离约为 5576 m，评估在其上下各 1000 m 均匀采样。
    "distance_m": [4576.0, 6576.0],
    "center_north_m": [-1500.0, 1500.0],
    "center_east_m": [-1500.0, 1500.0],
    "altitude_ft": [16000.0, 26000.0],
    "altitude_difference_ft": [-3000.0, 3000.0],
    "speed_fps": [650.0, 950.0],
    "heading_mode": "mixed",                   # mixed/head_on/tail_chase/crossing/random
    "heading_noise_deg": 20.0,
}

# E1-E3：混合观测退化参数，依次对应低、中、高退化。
# 每一组都同时包含噪声、丢帧和观测延迟；当前环境每个决策步约为 0.2 秒。
# 角度噪声作用于 AO/TA，距离单位为米；丢帧时保持上一帧有效相对观测。
MIXED_DEGRADATION_PARAMETERS = {
    "E1": {
        "dropout_probability": 0.10,
        "delay_steps": 1,
        "angle_noise_std_deg": 1.0,
        "range_noise_std_m": 100.0,
        "relative_speed_noise_std_mps": 0.0,
        "relative_altitude_noise_std_m": 0.0,
    },
    "E2": {
        "dropout_probability": 0.20,
        "delay_steps": 2,
        "angle_noise_std_deg": 3.0,
        "range_noise_std_m": 200.0,
        "relative_speed_noise_std_mps": 0.0,
        "relative_altitude_noise_std_m": 0.0,
    },
    "E3": {
        "dropout_probability": 0.30,
        "delay_steps": 3,
        "angle_noise_std_deg": 5.0,
        "range_noise_std_m": 300.0,
        "relative_speed_noise_std_mps": 0.0,
        "relative_altitude_noise_std_m": 0.0,
    },
}

# 优势窗口和动作连续性判据。
ADVANTAGE_AO_MAX_DEG = 30.0
ADVANTAGE_TA_MAX_DEG = 30.0
ADVANTAGE_DISTANCE_MAX_M = 3000.0
LARGE_ACTION_JUMP_THRESHOLD = 0.20

# 结果、轨迹和绘图。
OUTPUT_ROOT = REPO_ROOT / "experiments/results/partial_observability_gru"
SAVE_ACMI = True
ACMI_EPISODES = {0}                              # 每个实验组保存哪些回合的轨迹
SAVE_PLOTS = True
PLOT_DPI = 180
REWARD_MOVING_AVERAGE_WINDOW = 10                # 单局奖励曲线的滑动平均窗口
BOOTSTRAP_SAMPLES = 2000
SUPPRESS_POLICY_DEBUG_PRINT = True


# =============================================================================
# 实验条件定义
# =============================================================================

RELATIVE_FEATURE_SLICE = slice(9, 15)
DELTA_SPEED_INDEX = 9
DELTA_ALTITUDE_INDEX = 10
AO_INDEX = 11
TA_INDEX = 12
DISTANCE_INDEX = 13


@dataclass(frozen=True)
class DegradationConfig:
    """敌机相对观测退化参数。"""

    dropout_probability: float = 0.0
    delay_steps: int = 0
    angle_noise_std_deg: float = 0.0
    range_noise_std_m: float = 0.0
    relative_speed_noise_std_mps: float = 0.0
    relative_altitude_noise_std_m: float = 0.0


@dataclass(frozen=True)
class JointDegradationSpec:
    """联合观测退化参数：同时记录本机状态退化与火控雷达航迹退化。"""

    ownship: "OwnshipDegradationSpec"
    radar: "RadarDegradationSpec"


@dataclass(frozen=True)
class ExperimentCondition:
    """一个可独立运行的实验组。"""

    condition_id: str
    label: str
    degradation: DegradationConfig | "OwnshipDegradationSpec" | "RadarDegradationSpec" | JointDegradationSpec
    sensor_seed_group: int = 0
    ownship_level: Optional[int] = None
    radar_level: Optional[int] = None


MIXED_DEGRADATION_CONFIGS = {
    condition_id: DegradationConfig(**parameters)
    for condition_id, parameters in MIXED_DEGRADATION_PARAMETERS.items()
}

RELATIVE_CONDITIONS: Tuple[ExperimentCondition, ...] = (
    ExperimentCondition(
        "E0",
        "完整观测",
        DegradationConfig(),
        sensor_seed_group=0,
    ),
    ExperimentCondition(
        "E1",
        "低度混合观测退化（噪声+丢帧+延迟）",
        MIXED_DEGRADATION_CONFIGS["E1"],
        sensor_seed_group=1,
    ),
    ExperimentCondition(
        "E2",
        "中度混合观测退化（噪声+丢帧+延迟）",
        MIXED_DEGRADATION_CONFIGS["E2"],
        sensor_seed_group=1,
    ),
    ExperimentCondition(
        "E3",
        "高度混合观测退化（噪声+丢帧+延迟）",
        MIXED_DEGRADATION_CONFIGS["E3"],
        sensor_seed_group=1,
    ),
)


def ownship_conditions(*, quick: bool) -> Tuple[ExperimentCondition, ...]:
    from envs.JSBSim.envs.selfplay_observation import (
        OWN_SHIP_DEGRADATION_SPECS,
        OWN_SHIP_EVAL_LEVEL,
    )

    levels = (0, OWN_SHIP_EVAL_LEVEL) if quick else range(len(OWN_SHIP_DEGRADATION_SPECS))
    return tuple(
        ExperimentCondition(
            f"E{level}",
            "完整观测" if level == 0 else f"本机状态混合退化 E{level}",
            OWN_SHIP_DEGRADATION_SPECS[level],
            sensor_seed_group=int(level > 0),
            ownship_level=level,
        )
        for level in levels
    )


def radar_conditions(*, quick: bool) -> Tuple[ExperimentCondition, ...]:
    from envs.JSBSim.envs.selfplay_observation import JOINT_EVAL_LEVEL
    from envs.JSBSim.envs.singlecombat_env import RADAR_DEGRADATION_SPECS, RadarDegradationSpec

    # 丢帧率以整数百分比设档，噪声标准差按各自物理单位取整数。原 E4 作为评估的 E6。
    eval_specs = (
        RADAR_DEGRADATION_SPECS[0],
        replace(RADAR_DEGRADATION_SPECS[1], azimuth_noise_std_deg=2),
        RADAR_DEGRADATION_SPECS[2],
        RadarDegradationSpec(0.30, 3, 300, 4, 80, 8),
        RadarDegradationSpec(0.35, 4, 350, 5, 100, 10),
        RadarDegradationSpec(0.45, 4, 450, 7, 130, 13),
        RADAR_DEGRADATION_SPECS[4],
    )
    levels = (0, JOINT_EVAL_LEVEL) if quick else range(len(eval_specs))
    return tuple(
        ExperimentCondition(
            f"E{level}",
            "完整观测" if level == 0 else f"火控雷达航迹退化 E{level}",
            eval_specs[level],
            sensor_seed_group=int(level > 0),
            radar_level=level,
        )
        for level in levels
    )


def joint_conditions(*, quick: bool) -> Tuple[ExperimentCondition, ...]:
    from envs.JSBSim.envs.selfplay_observation import (
        JOINT_DEGRADATION_MAX_LEVEL,
        JOINT_EVAL_LEVEL,
        OWN_SHIP_DEGRADATION_SPECS,
    )
    from envs.JSBSim.envs.singlecombat_env import RADAR_DEGRADATION_SPECS

    levels = (0, JOINT_EVAL_LEVEL) if quick else range(JOINT_DEGRADATION_MAX_LEVEL + 1)
    return tuple(
        ExperimentCondition(
            f"E{level}",
            "完整观测" if level == 0 else f"联合观测退化（本机+雷达） E{level}",
            JointDegradationSpec(
                ownship=OWN_SHIP_DEGRADATION_SPECS[level],
                radar=RADAR_DEGRADATION_SPECS[level],
            ),
            sensor_seed_group=int(level > 0),
            ownship_level=level,
            radar_level=level,
        )
        for level in levels
    )


def validate_experiment_design() -> None:
    """检查兼容模式的相对观测退化强度。"""
    degradation_fields = (
        "dropout_probability",
        "delay_steps",
        "angle_noise_std_deg",
        "range_noise_std_m",
        "relative_speed_noise_std_mps",
        "relative_altitude_noise_std_m",
    )
    degradation_ids = ("E1", "E2", "E3")
    configs = [MIXED_DEGRADATION_CONFIGS[item_id] for item_id in degradation_ids]
    for item_id, config in zip(degradation_ids, configs):
        if not 0.0 < config.dropout_probability <= 1.0:
            raise ValueError(f"{item_id} 的丢帧概率必须位于 (0, 1]。")
        if config.delay_steps <= 0:
            raise ValueError(f"{item_id} 的观测延迟必须为正整数。")
        noise_values = (
            config.angle_noise_std_deg,
            config.range_noise_std_m,
            config.relative_speed_noise_std_mps,
            config.relative_altitude_noise_std_m,
        )
        if any(value < 0 for value in noise_values) or not any(
            value > 0 for value in noise_values
        ):
            raise ValueError(f"{item_id} 必须包含至少一种非零且非负的观测噪声。")

    for field in degradation_fields:
        values = [getattr(config, field) for config in configs]
        if any(left > right for left, right in zip(values, values[1:])):
            raise ValueError(f"E1-E3 的 {field} 必须由低到高设置。")
    for lower, higher in zip(configs, configs[1:]):
        if not any(
            getattr(lower, field) < getattr(higher, field)
            for field in degradation_fields
        ):
            raise ValueError("E1、E2、E3 的退化程度必须逐级增强。")

# =============================================================================
# 相对观测退化
# =============================================================================


class ObservationDegrader:
    """对单个智能体的敌机相对观测执行噪声、丢帧和时延处理。"""

    def __init__(self, config: DegradationConfig, rng: np.random.Generator):
        if not 0.0 <= config.dropout_probability <= 1.0:
            raise ValueError("dropout_probability 必须位于 [0, 1]。")
        if config.delay_steps < 0:
            raise ValueError("delay_steps 不能为负数。")
        self.config = config
        self.rng = rng
        self._last_valid: Optional[np.ndarray] = None
        self._delay_buffer: deque[np.ndarray] = deque()
        self.transform_count = 0
        self.dropout_count = 0

    def reset(self, initial_obs: np.ndarray) -> None:
        initial_obs = self._validate_obs(initial_obs)
        initial_relative = initial_obs[RELATIVE_FEATURE_SLICE].copy()
        self._last_valid = initial_relative
        self._delay_buffer = deque(
            initial_relative.copy() for _ in range(self.config.delay_steps)
        )
        self.transform_count = 0
        self.dropout_count = 0

    def transform(self, obs: np.ndarray) -> np.ndarray:
        obs = self._validate_obs(obs)
        if self._last_valid is None:
            self.reset(obs)

        degraded = obs.copy()
        measured = obs[RELATIVE_FEATURE_SLICE].copy()
        measured = self._add_noise(measured)

        self.transform_count += 1
        if self.rng.random() < self.config.dropout_probability:
            self.dropout_count += 1
            measured = self._last_valid.copy()
        else:
            self._last_valid = measured.copy()

        if self.config.delay_steps > 0:
            self._delay_buffer.append(measured.copy())
            measured = self._delay_buffer.popleft()

        degraded[RELATIVE_FEATURE_SLICE] = measured
        return np.clip(degraded, -10.0, 10.0).astype(np.float32, copy=False)

    @property
    def actual_dropout_rate(self) -> float:
        if self.transform_count == 0:
            return 0.0
        return self.dropout_count / self.transform_count

    def _add_noise(self, relative: np.ndarray) -> np.ndarray:
        relative = relative.copy()
        config = self.config

        # 观测中的相对速度以 340 m/s 归一化，高度差以 1000 m 归一化。
        if config.relative_speed_noise_std_mps > 0:
            relative[DELTA_SPEED_INDEX - RELATIVE_FEATURE_SLICE.start] += self.rng.normal(
                0.0, config.relative_speed_noise_std_mps / 340.0
            )
        if config.relative_altitude_noise_std_m > 0:
            relative[DELTA_ALTITUDE_INDEX - RELATIVE_FEATURE_SLICE.start] += self.rng.normal(
                0.0, config.relative_altitude_noise_std_m / 1000.0
            )

        angle_std_rad = np.deg2rad(config.angle_noise_std_deg)
        if angle_std_rad > 0:
            for absolute_index in (AO_INDEX, TA_INDEX):
                local_index = absolute_index - RELATIVE_FEATURE_SLICE.start
                relative[local_index] += self.rng.normal(0.0, angle_std_rad)
                relative[local_index] = np.clip(relative[local_index], 0.0, np.pi)

        # 距离在观测中以 10 km 归一化。
        if config.range_noise_std_m > 0:
            local_index = DISTANCE_INDEX - RELATIVE_FEATURE_SLICE.start
            relative[local_index] += self.rng.normal(0.0, config.range_noise_std_m / 10000.0)
            relative[local_index] = max(relative[local_index], 0.0)
        return relative

    @staticmethod
    def _validate_obs(obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim != 1 or obs.shape[0] < RELATIVE_FEATURE_SLICE.stop:
            raise ValueError(
                f"观测必须是一维且至少包含 {RELATIVE_FEATURE_SLICE.stop} 个元素，当前形状={obs.shape}"
            )
        return obs


# =============================================================================
# 评估执行
# =============================================================================


def load_base_eval_module():
    """加载现有 experiments/1v1.py，复用模型与随机初始态逻辑。"""
    module_path = Path(__file__).with_name("1v1.py")
    if not module_path.is_file():
        raise FileNotFoundError(f"缺少基础评估脚本: {module_path}")
    module_name = "jsbsim_1v1_eval_base"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载基础评估脚本: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"当前 Python 环境缺少依赖 {exc.name!r}。请使用原训练环境运行本脚本。"
        ) from exc
    validate_base_module_api(module)
    return module


def resolve_repo_path(path: Path | str) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


REQUIRED_BASE_API = (
    "SingleCombatEnv",
    "load_controller",
    "validate_setup",
    "reset_eval_env",
    "relative_state_metrics",
    "append_metric_history",
    "status_name",
    "initial_state_row",
    "summarize_relative_history",
    "decide_winner",
)


def validate_base_module_api(base: Any) -> None:
    """在加载模型前一次性确认基础评估脚本提供了所需接口。"""
    missing = [name for name in REQUIRED_BASE_API if not hasattr(base, name)]
    if missing:
        raise RuntimeError(
            "experiments/1v1.py 缺少快速评估需要的接口: " + ", ".join(missing)
        )


def scenario_config_path(scenario_name: str) -> Path:
    """把 1v1/NoWeapon/Selfplay 转换为仓库内 YAML 绝对路径。"""
    normalized = str(scenario_name).replace("\\", "/").strip("/")
    return (REPO_ROOT / "envs/JSBSim/configs" / f"{normalized}.yaml").resolve()


def validate_scenario_configs(scenario_names: Iterable[str]) -> None:
    """启动仿真前检查所有场景，避免模型加载后才报告 YAML 缺失。"""
    missing = []
    for scenario_name in dict.fromkeys(scenario_names):
        path = scenario_config_path(scenario_name)
        if not path.is_file():
            missing.append(f"{scenario_name} -> {path}")
    if missing:
        raise FileNotFoundError("缺少以下场景配置:\n" + "\n".join(missing))


def resolve_actor_path(path: Path | str) -> Path:
    """支持传入 actor 文件或实验目录；目录中自动寻找最新 actor_latest.pt。"""
    resolved = resolve_repo_path(path)
    if resolved.is_file():
        return resolved
    if not resolved.exists():
        raise FileNotFoundError(
            f"Actor 路径不存在: {resolved}\n"
            "请修改脚本开头的 ACTOR_A_PATH / ACTOR_B_PATH。"
        )
    if not resolved.is_dir():
        raise FileNotFoundError(f"Actor 路径既不是文件也不是目录: {resolved}")

    candidates = sorted(
        resolved.rglob("actor_latest.pt"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"目录中没有找到 actor_latest.pt: {resolved}")
    selected = candidates[0].resolve()
    if len(candidates) > 1:
        print(f"Actor 目录包含 {len(candidates)} 个检查点，自动选择最新文件: {selected}")
    return selected


def configure_base_module(base: Any) -> None:
    """把本脚本开头的公共参数同步给基础评估模块。"""
    base.DETERMINISTIC = DETERMINISTIC
    base.SUPPRESS_POLICY_DEBUG_PRINT = SUPPRESS_POLICY_DEBUG_PRINT
    base.SWAP_ACTOR_ORDER = SWAP_ACTOR_ORDER
    base.RANDOM_SIDE_SWAP = RANDOM_SIDE_SWAP
    base.INITIAL_STATE_MODE = INITIAL_STATE_MODE
    base.RANDOM_INITIAL_STATE_CONFIG = dict(RANDOM_INITIAL_STATE_CONFIG)
    base.CUSTOM_INITIAL_STATES = None


def strict_winner(actor_a_status: str, actor_b_status: str) -> str:
    actor_a_alive = actor_a_status == "alive"
    actor_b_alive = actor_b_status == "alive"
    if actor_a_alive and not actor_b_alive:
        return "actor_a"
    if actor_b_alive and not actor_a_alive:
        return "actor_b"
    return "tie"


def action_continuity_metrics(actions: Sequence[np.ndarray]) -> Dict[str, float]:
    if len(actions) < 2:
        return {
            "actor_a_action_total_variation": 0.0,
            "actor_a_action_second_difference": 0.0,
            "actor_a_large_action_jump_rate": 0.0,
        }
    action_array = np.asarray(actions, dtype=np.float64)
    first_difference = np.diff(action_array, axis=0)
    total_variation = np.abs(first_difference).sum(axis=1)
    large_jump_rate = np.max(np.abs(first_difference), axis=1) >= LARGE_ACTION_JUMP_THRESHOLD
    second_difference = np.diff(action_array, n=2, axis=0)
    return {
        "actor_a_action_total_variation": float(np.mean(total_variation)),
        "actor_a_action_second_difference": (
            float(np.mean(np.abs(second_difference).sum(axis=1)))
            if len(second_difference)
            else 0.0
        ),
        "actor_a_large_action_jump_rate": float(np.mean(large_jump_rate)),
    }


def metric_mean(history: Dict[str, List[float]], key: str) -> float:
    values = history.get(key, [])
    return float(np.mean(values)) if values else 0.0


def run_episode(
    base: Any,
    condition: ExperimentCondition,
    episode: int,
    env: Any,
    actor_a: Any,
    actor_b: Any,
    condition_dir: Path,
    *,
    save_acmi: bool = True,
    degrade_both: bool = False,
) -> Dict[str, object]:
    actor_a.reset()
    actor_b.reset()
    obs = base.reset_eval_env(env)

    if SWAP_ACTOR_ORDER and episode % 2 == 1:
        slot_controllers = [actor_b, actor_a]
    else:
        slot_controllers = [actor_a, actor_b]

    agent_ids = (env.ego_ids + env.enm_ids)[: env.num_agents]
    actor_to_slot = {controller.name: idx for idx, controller in enumerate(slot_controllers)}
    actor_to_agent_id = {
        controller.name: agent_ids[idx] for idx, controller in enumerate(slot_controllers)
    }
    actor_a_slot = actor_to_slot["actor_a"]
    actor_a_agent_id = actor_to_agent_id["actor_a"]
    actor_b_agent_id = actor_to_agent_id["actor_b"]

    degraded_actors = ("actor_a", "actor_b") if degrade_both else ("actor_a",)
    base_seed = SENSOR_SEED + condition.sensor_seed_group * 100000 + episode
    ownship_degraders: Dict[str, Any] = {}
    radar_degraders: Dict[str, Any] = {}
    relative_degraders: Dict[str, ObservationDegrader] = {}

    for actor_index, actor_name in enumerate(degraded_actors):
        slot = actor_to_slot[actor_name]
        agent_id = actor_to_agent_id[actor_name]
        controller = slot_controllers[slot]
        # Actor A 保持原随机种子不变，保证既有单方退化评估结果完全可复现；Actor B 使用独立偏移种子。
        ownship_rng = np.random.default_rng(base_seed + actor_index * 1000003)
        radar_rng = np.random.default_rng(
            base_seed + (500009 if condition.ownship_level is not None else 0) + actor_index * 1000003
        )
        if condition.ownship_level is not None:
            from envs.JSBSim.envs.selfplay_observation import OwnshipObservationDegrader

            ownship_degraders[actor_name] = OwnshipObservationDegrader(
                condition.ownship_level, obs[slot], ownship_rng
            )
        if condition.radar_level is not None:
            from envs.JSBSim.envs.singlecombat_env import (
                RadarTrackDegrader,
                extract_true_enemy_track,
            )

            spec = condition.degradation
            radar_spec = spec.radar if isinstance(spec, JointDegradationSpec) else spec
            radar_degrader = RadarTrackDegrader(radar_spec, radar_rng)
            radar_degrader.reset(extract_true_enemy_track(env, agent_id))
            radar_degraders[actor_name] = radar_degrader
        elif condition.ownship_level is None and bool(getattr(controller, "is_tactical", False)):
            from envs.JSBSim.envs.singlecombat_env import (
                RadarDegradationSpec,
                RadarTrackDegrader,
                extract_true_enemy_track,
            )

            deg = condition.degradation
            radar_spec = RadarDegradationSpec(
                dropout_probability=deg.dropout_probability,
                delay_steps=deg.delay_steps,
                range_noise_std_m=deg.range_noise_std_m,
                azimuth_noise_std_deg=deg.angle_noise_std_deg,
                altitude_noise_std_m=deg.relative_altitude_noise_std_m,
                velocity_noise_std_mps=deg.relative_speed_noise_std_mps,
            )
            radar_degrader = RadarTrackDegrader(radar_spec, radar_rng)
            radar_degrader.reset(extract_true_enemy_track(env, agent_id))
            radar_degraders[actor_name] = radar_degrader
        elif condition.ownship_level is None:
            degrader = ObservationDegrader(condition.degradation, ownship_rng)
            degrader.reset(obs[slot])
            relative_degraders[actor_name] = degrader

    actor_rewards = {"actor_a": 0.0, "actor_b": 0.0}
    relative_history: Dict[str, List[float]] = {}
    normalized_actor_a_actions: List[np.ndarray] = []
    advantage_steps = 0

    acmi_path: Optional[Path] = None
    if save_acmi and SAVE_ACMI and episode in ACMI_EPISODES:
        acmi_dir = condition_dir / "acmi"
        acmi_dir.mkdir(parents=True, exist_ok=True)
        acmi_path = acmi_dir / f"episode_{episode:04d}.txt.acmi"
        env._create_records = False
        env.render(mode="txt", filepath=str(acmi_path))

    while True:
        actions = []
        for slot, controller in enumerate(slot_controllers):
            controller_obs = obs[slot]
            agent_id = actor_to_agent_id[controller.name]
            actor_name = controller.name
            radar_degrader = radar_degraders.get(actor_name)
            ownship_degrader = ownship_degraders.get(actor_name)
            relative_degrader = relative_degraders.get(actor_name)

            if radar_degrader is not None:
                from envs.JSBSim.envs.singlecombat_env import extract_true_enemy_track

                degraded_track = radar_degrader.transform(
                    env.agents[agent_id].get_position(),
                    extract_true_enemy_track(env, agent_id),
                )
                env._radar_track_overrides = {agent_id: degraded_track}
                try:
                    controller_obs = env.task.get_obs(env, agent_id).astype(np.float32)
                    if ownship_degrader is not None:
                        controller_obs = ownship_degrader.transform(controller_obs)
                    action_out = controller.act(controller_obs, env=env, agent_id=agent_id)
                finally:
                    env._radar_track_overrides = None
                actions.append(action_out)
                continue

            if ownship_degrader is not None:
                controller_obs = ownship_degrader.transform(controller_obs)
            elif relative_degrader is not None:
                controller_obs = relative_degrader.transform(controller_obs)
            actions.append(
                controller.act(
                    controller_obs,
                    env=env,
                    agent_id=agent_id,
                )
            )

        # 记录实际写入 JSBSim 前的归一化控制量，用于动作连续性分析。
        actor_a_action = np.asarray(actions[actor_a_slot], dtype=np.float32)
        normalized_action = env.task.normalize_action(
            env, actor_a_agent_id, actor_a_action.copy()
        )
        normalized_actor_a_actions.append(np.asarray(normalized_action, dtype=np.float32))

        obs, rewards, dones, _ = env.step(np.stack(actions, axis=0))
        for controller in slot_controllers:
            slot = actor_to_slot[controller.name]
            actor_rewards[controller.name] += float(rewards[slot, 0])

        actor_a_sim = env.agents[actor_a_agent_id]
        actor_b_sim = env.agents[actor_b_agent_id]
        relative_metrics = base.relative_state_metrics(actor_a_sim, actor_b_sim)
        base.append_metric_history(relative_history, relative_metrics)
        if (
            relative_metrics["actor_a_ao_deg"] <= ADVANTAGE_AO_MAX_DEG
            and relative_metrics["actor_a_ta_deg"] <= ADVANTAGE_TA_MAX_DEG
            and relative_metrics["distance_m"] <= ADVANTAGE_DISTANCE_MAX_M
        ):
            advantage_steps += 1

        if acmi_path is not None:
            env.render(mode="txt", filepath=str(acmi_path))

        if bool(np.asarray(dones).all()):
            break

    actor_a_sim = env.agents[actor_a_agent_id]
    actor_b_sim = env.agents[actor_b_agent_id]
    actor_a_status = base.status_name(actor_a_sim)
    actor_b_status = base.status_name(actor_b_sim)
    if STRICT_WINNER:
        winner = strict_winner(actor_a_status, actor_b_status)
    else:
        winner = base.decide_winner(
            actor_rewards["actor_a"],
            actor_rewards["actor_b"],
            actor_a_status,
            actor_b_status,
        )

    initial_summary = base.initial_state_row(env, actor_a_agent_id, actor_b_agent_id)
    terminal_summary = base.summarize_relative_history(relative_history)
    continuity_summary = action_continuity_metrics(normalized_actor_a_actions)
    total_metric_steps = max(len(relative_history.get("distance_m", [])), 1)

    actor_a_ownship_degrader = ownship_degraders.get("actor_a")
    actor_a_radar_degrader = radar_degraders.get("actor_a")
    actor_a_relative_degrader = relative_degraders.get("actor_a")
    dropout_rates = [
        item.actual_dropout_rate
        for item in (actor_a_ownship_degrader, actor_a_radar_degrader, actor_a_relative_degrader)
        if item is not None
    ]
    actual_dropout_rate = float(np.mean(dropout_rates)) if dropout_rates else 0.0

    row: Dict[str, object] = {
        "condition_id": condition.condition_id,
        "condition_label": condition.label,
        "episode": episode,
        "winner": winner,
        "win_score": 1.0 if winner == "actor_a" else 0.5 if winner == "tie" else 0.0,
        "steps": int(env.current_step),
        "duration_sec": float(env.current_step * env.time_interval),
        "actor_a_agent_id": actor_a_agent_id,
        "actor_b_agent_id": actor_b_agent_id,
        "actor_a_reward": actor_rewards["actor_a"],
        "actor_b_reward": actor_rewards["actor_b"],
        "reward_margin": actor_rewards["actor_a"] - actor_rewards["actor_b"],
        "actor_a_status": actor_a_status,
        "actor_b_status": actor_b_status,
        "actor_a_bloods": float(actor_a_sim.bloods),
        "actor_b_bloods": float(actor_b_sim.bloods),
        "advantage_window_occupancy": advantage_steps / total_metric_steps,
        "actor_a_mean_ao_deg": metric_mean(relative_history, "actor_a_ao_deg"),
        "actor_a_mean_ta_deg": metric_mean(relative_history, "actor_a_ta_deg"),
        "mean_distance_m": metric_mean(relative_history, "distance_m"),
        "actual_dropout_rate": actual_dropout_rate,
        "ownship_dropout_rate": (
            actor_a_ownship_degrader.actual_dropout_rate if actor_a_ownship_degrader is not None else 0.0
        ),
        "radar_dropout_rate": (
            actor_a_radar_degrader.actual_dropout_rate if actor_a_radar_degrader is not None else 0.0
        ),
        "acmi_path": str(acmi_path) if acmi_path is not None else "",
        **initial_summary,
        **terminal_summary,
        **continuity_summary,
    }
    print(
        f"[{condition.condition_id} {episode + 1:03d}/{NUM_EPISODES_PER_CONDITION}] "
        f"winner={winner} reward={row['actor_a_reward']:.2f} "
        f"adv={row['advantage_window_occupancy']:.3f} "
        f"AO={row['actor_a_mean_ao_deg']:.1f}° "
        f"R={row['mean_distance_m'] / 1000.0:.2f}km"
    )
    return row


def bootstrap_mean_ci(
    values: Iterable[float], seed: int
) -> Tuple[float, float, float]:
    values = np.asarray(list(values), dtype=np.float64)
    if values.size == 0:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(values))
    if values.size == 1 or BOOTSTRAP_SAMPLES <= 0:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(BOOTSTRAP_SAMPLES, values.size))
    samples = values[indices].mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return mean, float(low), float(high)


def summarize_condition(
    condition: ExperimentCondition,
    rows: List[Dict[str, object]],
    summary_seed: int,
) -> Dict[str, object]:
    def values(key: str) -> np.ndarray:
        return np.asarray([float(row[key]) for row in rows], dtype=np.float64)

    win_mean, win_low, win_high = bootstrap_mean_ci(values("win_score"), summary_seed)
    adv_mean, adv_low, adv_high = bootstrap_mean_ci(
        values("advantage_window_occupancy"), summary_seed + 1
    )
    return {
        "condition_id": condition.condition_id,
        "condition_label": condition.label,
        "num_episodes": len(rows),
        "actor_a_win_rate": float(np.mean([row["winner"] == "actor_a" for row in rows])),
        "actor_b_win_rate": float(np.mean([row["winner"] == "actor_b" for row in rows])),
        "tie_rate": float(np.mean([row["winner"] == "tie" for row in rows])),
        "win_score": win_mean,
        "win_score_ci_low": win_low,
        "win_score_ci_high": win_high,
        "actor_a_crash_rate": float(
            np.mean([row["actor_a_status"] == "crash" for row in rows])
        ),
        "actor_a_avg_reward": float(np.mean(values("actor_a_reward"))),
        "mean_reward_margin": float(np.mean(values("reward_margin"))),
        "advantage_window_occupancy": adv_mean,
        "advantage_window_occupancy_ci_low": adv_low,
        "advantage_window_occupancy_ci_high": adv_high,
        "actor_a_mean_ao_deg": float(np.mean(values("actor_a_mean_ao_deg"))),
        "actor_a_mean_ta_deg": float(np.mean(values("actor_a_mean_ta_deg"))),
        "mean_distance_m": float(np.mean(values("mean_distance_m"))),
        "actor_a_action_total_variation": float(
            np.mean(values("actor_a_action_total_variation"))
        ),
        "actor_a_action_second_difference": float(
            np.mean(values("actor_a_action_second_difference"))
        ),
        "actor_a_large_action_jump_rate": float(
            np.mean(values("actor_a_large_action_jump_rate"))
        ),
        "actual_dropout_rate": float(np.mean(values("actual_dropout_rate"))),
    }


def save_csv(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def load_pyplot() -> Optional[Any]:
    """加载无界面的绘图库，并统一中文字体设置。"""
    if not SAVE_PLOTS:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"跳过绘图，matplotlib 不可用: {exc}")
        return None

    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )
    return plt


def plot_summaries(summaries: List[Dict[str, object]], output_dir: Path) -> List[Path]:
    plt = load_pyplot()
    if plt is None:
        return []

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []

    def error(summary: Dict[str, object], key: str) -> Tuple[float, float]:
        value = float(summary[key])
        return (
            value - float(summary[f"{key}_ci_low"]),
            float(summary[f"{key}_ci_high"]) - value,
        )

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    ids = [str(item["condition_id"]) for item in summaries]
    win_errors = np.asarray([error(item, "win_score") for item in summaries]).T
    adv_errors = np.asarray(
        [error(item, "advantage_window_occupancy") for item in summaries]
    ).T
    axes[0, 0].bar(ids, [item["win_score"] for item in summaries], yerr=win_errors, capsize=4)
    axes[0, 0].set_title("部分可观测条件下的胜负得分")
    axes[0, 0].set_ylim(0.0, 1.0)
    axes[0, 0].set_ylabel("胜=1，平=0.5，负=0")
    axes[0, 1].bar(
        ids,
        [item["advantage_window_occupancy"] for item in summaries],
        yerr=adv_errors,
        capsize=4,
        color="#2ca02c",
    )
    axes[0, 1].set_title("优势窗口占用率")
    axes[0, 1].set_ylim(0.0, 1.0)
    axes[1, 0].bar(ids, [item["actor_a_mean_ao_deg"] for item in summaries], color="#ff7f0e")
    axes[1, 0].set_title("平均 AO")
    axes[1, 0].set_ylabel("deg")
    axes[1, 1].bar(
        ids,
        [float(item["mean_distance_m"]) / 1000.0 for item in summaries],
        color="#9467bd",
    )
    axes[1, 1].set_title("平均相对距离")
    axes[1, 1].set_ylabel("km")
    figure.suptitle("混合观测退化下的鲁棒性")
    figure.tight_layout()
    path = plot_dir / "experiment_1_partial_observability.png"
    figure.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(figure)
    paths.append(path)

    return paths


def trailing_moving_average(values: Sequence[float], window: int) -> np.ndarray:
    """计算包含当前回合的后向滑动平均，开头不足窗口时使用已有回合。"""
    if window <= 0:
        raise ValueError("滑动平均窗口必须为正整数。")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("滑动平均输入必须是一维序列。")
    if array.size == 0:
        return array.copy()

    result = np.empty_like(array)
    cumulative = np.cumsum(array, dtype=np.float64)
    for index in range(array.size):
        start = max(0, index - window + 1)
        total = cumulative[index]
        if start > 0:
            total -= cumulative[start - 1]
        result[index] = total / (index - start + 1)
    return result


def plot_episode_reward_curves(
    rows: List[Dict[str, object]], output_dir: Path
) -> List[Path]:
    """绘制各退化档位的 Actor A 单局奖励及其滑动平均。"""
    plt = load_pyplot()
    if plt is None:
        return []

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(12, 5))
    condition_ids = dict.fromkeys(str(row["condition_id"]) for row in rows)
    colors = plt.get_cmap("tab10").colors

    for color_index, condition_id in enumerate(condition_ids):
        condition_rows = sorted(
            (row for row in rows if str(row["condition_id"]) == condition_id),
            key=lambda row: int(row["episode"]),
        )
        episodes = np.asarray([int(row["episode"]) + 1 for row in condition_rows], dtype=np.int64)
        rewards = np.asarray([float(row["actor_a_reward"]) for row in condition_rows], dtype=np.float64)
        smoothed = trailing_moving_average(rewards, REWARD_MOVING_AVERAGE_WINDOW)
        color = colors[color_index % len(colors)]
        axis.plot(episodes, rewards, color=color, alpha=0.22, linewidth=0.9)
        axis.plot(episodes, smoothed, color=color, linewidth=2.0, label=condition_id)

    axis.set_xlabel("测试局数")
    axis.set_ylabel("Actor A 单局累计奖励")
    axis.grid(alpha=0.25)
    axis.legend()

    figure.suptitle(
        "Actor A 单局奖励曲线"
        f"（细线=单局奖励，粗线={REWARD_MOVING_AVERAGE_WINDOW}局滑动平均）"
    )
    figure.tight_layout()
    path = plot_dir / "episode_reward_curves.png"
    figure.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(figure)
    return [path]


def episode_reward_series(rows: Sequence[Dict[str, Any]]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """按档位和实际局号整理奖励，接受运行中的记录和回读的 CSV 记录。"""
    grouped: Dict[str, Dict[int, float]] = {}
    for row in rows:
        condition_id = str(row["condition_id"])
        episode = int(row["episode"]) + 1
        reward = float(row["actor_a_reward"])
        condition = grouped.setdefault(condition_id, {})
        if episode < 1 or episode in condition or not np.isfinite(reward):
            raise ValueError(f"{condition_id} 第 {episode} 局的局号或奖励无效")
        condition[episode] = reward
    series = {}
    for condition_id, condition in grouped.items():
        episodes, rewards = zip(*sorted(condition.items()))
        series[condition_id] = (np.asarray(episodes), np.asarray(rewards))
    return series


def plot_model_reward_comparison(
    rows_by_model: Dict[str, List[Dict[str, Any]]],
    baseline_rows: List[Dict[str, Any]] | None,
    output_dir: Path,
) -> Path | None:
    """每个档位一张子图，所有子图共用干净 A 的 E0 奖励作为基线。"""
    series_by_model = {name: episode_reward_series(rows) for name, rows in rows_by_model.items()}
    reference = next(iter(series_by_model.values()), {})
    if not reference:
        raise ValueError("被测模型没有可绘制的单局奖励")
    baseline = None if baseline_rows is None else episode_reward_series(baseline_rows).get("E0")
    if baseline_rows is not None and baseline is None:
        raise ValueError("干净 A 的结果缺少 E0 奖励")
    for name, series in series_by_model.items():
        if series.keys() != reference.keys():
            raise ValueError(f"{name} 的观测退化档位与其他模型不一致")
        for condition_id, (episodes, _) in series.items():
            if not np.array_equal(episodes, reference[condition_id][0]):
                raise ValueError(f"{name} 的 {condition_id} 局号与其他模型不一致")
            if baseline is not None and not np.array_equal(episodes, baseline[0]):
                raise ValueError(f"干净 A 的 E0 局号与 {condition_id} 不一致")

    plt = load_pyplot()
    if plt is None:
        return None
    ncols = min(len(reference), 3)
    nrows = (len(reference) + ncols - 1) // ncols
    figure, axes = plt.subplots(
        nrows, ncols, figsize=(4.8 * ncols, 2.9 * nrows + 1.2),
        sharex=True, sharey=True, squeeze=False,
    )
    colors = plt.get_cmap("tab10").colors
    for axis, condition_id in zip(axes.flat, reference):
        lines = []
        if baseline is not None:
            lines.append(("干净 A 基线（E0）", baseline, "#454545", "--"))
        for index, (name, series) in enumerate(series_by_model.items()):
            lines.append((name, series[condition_id], colors[index % len(colors)], "-"))
        for name, (episodes, rewards), color, linestyle in lines:
            axis.plot(episodes, rewards, color=color, alpha=0.22, linewidth=0.8)
            smoothed = trailing_moving_average(rewards, REWARD_MOVING_AVERAGE_WINDOW)
            axis.plot(episodes, smoothed, color=color, linestyle=linestyle, linewidth=1.8,
                      marker="o" if episodes.size == 1 else None, label=name)
        axis.set_title(condition_id)
        axis.grid(alpha=0.22)
        axis.tick_params(labelbottom=True)
    for axis in list(axes.flat)[len(reference):]:
        axis.set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    min_episode = min(episodes[0] for episodes, _ in reference.values())
    max_episode = max(episodes[-1] for episodes, _ in reference.values())
    if min_episode == max_episode:
        axes[0, 0].set_xticks([min_episode])
    else:
        axes[0, 0].set_xlim(min_episode, max_episode)
    figure.suptitle("各观测退化档位的 Actor A 奖励对比", y=0.98)
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=min(len(labels), 5))
    figure.supxlabel(f"测试局数（细线为每局奖励，粗线为 {REWARD_MOVING_AVERAGE_WINDOW} 局滑动平均）")
    figure.supylabel("单局累计奖励")
    figure.tight_layout(rect=(0.025, 0.035, 1.0, 0.87))
    path = output_dir / "plots" / "model_reward_curves_by_condition.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(figure)
    return path


def load_reward_result(directory: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    with (directory / "config.json").open(encoding="utf-8-sig") as file:
        config = json.load(file)
    # 归档结果可能从另一台机器复制，仓库内的权重用相对位置比较。
    checkpoint = str(config["actor_b_path"]).replace("\\", "/")
    _, separator, relative = checkpoint.partition("/scripts/results/")
    config["actor_b_path"] = f"scripts/results/{relative}" if separator else checkpoint
    with (directory / "all_episodes.csv").open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    return config, rows


def plot_saved_model_rewards(
    results: Dict[str, Path], baseline_result: Path | None, output_dir: Path,
) -> Path | None:
    loaded = {name: load_reward_result(path) for name, path in results.items()}
    reference = next(iter(loaded.values()))[0]
    pairing_keys = (
        "actor_b_path", "actor_b_scenario_name", "eval_scenario_name", "env_seed",
        "deterministic", "swap_actor_order", "initial_state_mode", "random_initial_state_config",
    )
    comparison_keys = (*pairing_keys, "observation_mode", "degrade_both", "sensor_seed", "conditions")
    for name, (config, _) in loaded.items():
        mismatches = [key for key in comparison_keys if config[key] != reference[key]]
        if mismatches:
            raise ValueError(f"{name} 的评估配置与其他模型不同：{', '.join(mismatches)}")
    baseline_rows = None
    if baseline_result is not None:
        baseline_config, baseline_rows = load_reward_result(baseline_result)
        mismatches = [key for key in pairing_keys if baseline_config[key] != reference[key]]
        if mismatches:
            raise ValueError(f"干净 A 基线的评估配置与被测模型不同：{', '.join(mismatches)}")
    return plot_model_reward_comparison(
        {name: rows for name, (_, rows) in loaded.items()}, baseline_rows, output_dir,
    )


def config_snapshot(
    actor_a_path: Path, actor_b_path: Path, conditions: Sequence[ExperimentCondition]
) -> Dict[str, object]:
    return {
        "actor_a_path": str(actor_a_path),
        "actor_b_path": str(actor_b_path),
        "actor_a_scenario_name": ACTOR_A_SCENARIO_NAME,
        "actor_b_scenario_name": ACTOR_B_SCENARIO_NAME,
        "eval_scenario_name": EVAL_SCENARIO_NAME,
        "num_episodes_per_condition": NUM_EPISODES_PER_CONDITION,
        "env_seed": ENV_SEED,
        "sensor_seed": SENSOR_SEED,
        "device": DEVICE,
        "deterministic": DETERMINISTIC,
        "swap_actor_order": SWAP_ACTOR_ORDER,
        "strict_winner": STRICT_WINNER,
        "reward_moving_average_window": REWARD_MOVING_AVERAGE_WINDOW,
        "initial_state_mode": INITIAL_STATE_MODE,
        "random_initial_state_config": RANDOM_INITIAL_STATE_CONFIG,
        "advantage_window": {
            "ao_max_deg": ADVANTAGE_AO_MAX_DEG,
            "ta_max_deg": ADVANTAGE_TA_MAX_DEG,
            "distance_max_m": ADVANTAGE_DISTANCE_MAX_M,
        },
        "conditions": [asdict(condition) for condition in conditions],
    }


def parse_cli_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """支持命令行覆盖路径、场景、局数和设备；无参数时沿用顶部配置。"""
    parser = argparse.ArgumentParser(description="普通无武器 Selfplay 的 E0–E6 观测退化评估")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--actor-a-path", type=Path, help="被测 Actor A 权重或目录")
    inputs.add_argument("--model", action="append", nargs=2, metavar=("LABEL", "PATH"),
                        help="增加一组被测 Actor A，可重复传入模型标签和权重路径")
    inputs.add_argument("--compare-result", action="append", nargs=2, metavar=("LABEL", "DIR"),
                        help="读取一组已有的评估结果出图，可重复传入标签和结果目录")
    baseline = parser.add_mutually_exclusive_group()
    baseline.add_argument("--baseline-actor-path", type=Path,
                          help="配合 --model，在 E0 评估干净训练的 Actor A")
    baseline.add_argument("--baseline-result", type=Path,
                          help="配合 --compare-result，读取已有干净 Actor A 的 E0 奖励")
    parser.add_argument("--actor-b-path", type=Path, default=None, help="对手 Actor B 权重或目录")
    parser.add_argument("--actor-a-scenario", type=str, default=None, help="Actor A 训练场景名")
    parser.add_argument("--actor-b-scenario", type=str, default=None, help="Actor B 训练场景名")
    parser.add_argument("--eval-scenario", type=str, default=None, help="对战评估环境场景名")
    parser.add_argument("--episodes", type=int, default=None, help="每个实验条件运行的对局数")
    parser.add_argument("--device", type=str, default=None, help="推理设备，如 auto / cpu / cuda:0")
    parser.add_argument("--output-dir", type=Path, default=None, help="自定义结果输出目录")
    parser.add_argument("--ownship-degradation", action="store_true",
                        help="只运行普通无武器 Selfplay 的 E0/E2 快验，默认运行 E0–E6")
    parser.add_argument("--degradation-mode", choices=["ownship", "radar", "joint"], default=None,
                        help="显式指定退化口径：ownship（仅前9维本机）、radar（仅后6维雷达）、joint（本机+雷达联合）")
    parser.add_argument("--quick", action="store_true",
                        help="仅运行 E0 与 E2 两档快验（适用于 ownship / radar / joint 模式）")
    parser.add_argument("--degrade-both", action="store_true",
                        help="红蓝双方同时施加同档观测退化，用于公平的跨实验 1v1 对抗评估")
    args = parser.parse_args(argv)
    if args.model and args.baseline_actor_path is None:
        parser.error("--model 需要搭配 --baseline-actor-path")
    if args.baseline_actor_path is not None and not args.model:
        parser.error("--baseline-actor-path 需要搭配 --model")
    if args.baseline_result is not None and not args.compare_result:
        parser.error("--baseline-result 需要搭配 --compare-result")
    named_inputs = args.model or args.compare_result or []
    labels = [label for label, _ in named_inputs]
    if len(set(labels)) != len(labels) or any(not label.strip() for label in labels):
        parser.error("模型标签须非空且不能重复")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    cli = parse_cli_args(argv)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (cli.output_dir.resolve() if cli.output_dir is not None else OUTPUT_ROOT / f"run_{timestamp}")
    if cli.compare_result:
        plot_path = plot_saved_model_rewards(
            {label: Path(directory) for label, directory in cli.compare_result},
            cli.baseline_result, output_dir,
        )
        if plot_path is not None:
            print(f"奖励对比图: {plot_path}")
        return

    num_episodes = cli.episodes if cli.episodes is not None else NUM_EPISODES_PER_CONDITION
    actor_a_scenario = cli.actor_a_scenario or ACTOR_A_SCENARIO_NAME
    actor_b_scenario = cli.actor_b_scenario or ACTOR_B_SCENARIO_NAME
    eval_scenario = cli.eval_scenario or EVAL_SCENARIO_NAME
    device_name = cli.device or DEVICE

    if num_episodes <= 0:
        raise ValueError("NUM_EPISODES_PER_CONDITION 必须为正整数。")
    if REWARD_MOVING_AVERAGE_WINDOW <= 0:
        raise ValueError("奖励曲线滑动平均窗口必须为正整数。")

    validate_scenario_configs(
        (actor_a_scenario, actor_b_scenario, eval_scenario)
    )
    if cli.model:
        candidate_paths = [(label, resolve_actor_path(path)) for label, path in cli.model]
        baseline_path = resolve_actor_path(cli.baseline_actor_path)
    else:
        candidate_paths = [("Actor A", resolve_actor_path(cli.actor_a_path or ACTOR_A_PATH))]
    actor_b_path = resolve_actor_path(cli.actor_b_path or ACTOR_B_PATH)

    base = load_base_eval_module()
    configure_base_module(base)
    from envs.JSBSim.envs.selfplay_observation import SELFPLAY_SCENARIO, validate_unarmed_selfplay

    is_unarmed_selfplay = (actor_a_scenario, actor_b_scenario, eval_scenario) == (SELFPLAY_SCENARIO,) * 3
    quick_mode = bool(cli.quick or cli.ownship_degradation)
    if cli.ownship_degradation and cli.degradation_mode not in (None, "ownship"):
        raise ValueError("--ownship-degradation 与非 ownship 的 --degradation-mode 冲突")
    if cli.ownship_degradation and not is_unarmed_selfplay:
        raise ValueError("本机状态退化快验的双方模型和评估环境都须使用普通无武器 Selfplay")

    selected_mode = cli.degradation_mode
    if selected_mode is None:
        selected_mode = "ownship" if is_unarmed_selfplay else "relative"

    if selected_mode in ("ownship", "joint"):
        if not is_unarmed_selfplay:
            raise ValueError(f"{selected_mode} 退化评估的双方模型和评估环境都须使用普通无武器 Selfplay")
        from envs.JSBSim.utils.utils import parse_config

        validate_unarmed_selfplay(parse_config(SELFPLAY_SCENARIO))
        conditions = (
            ownship_conditions(quick=quick_mode)
            if selected_mode == "ownship"
            else joint_conditions(quick=quick_mode)
        )
        observation_mode = f"{selected_mode}_degradation"
    elif selected_mode == "radar":
        conditions = radar_conditions(quick=quick_mode)
        observation_mode = "radar_degradation"
    else:
        validate_experiment_design()
        conditions = RELATIVE_CONDITIONS
        observation_mode = "relative_degradation"

    np.random.seed(ENV_SEED)
    base.torch.manual_seed(ENV_SEED)
    device = base.resolve_device(device_name)

    needs_lowlevel = any("Hierarchy" in name for name in (actor_a_scenario, actor_b_scenario))
    lowlevel_policy = base.load_lowlevel_policy(device) if needs_lowlevel else None
    actor_b = base.load_controller(
        "actor_b", actor_b_path, actor_b_scenario, lowlevel_policy, device
    )

    jobs = [
        (label, path, conditions, output_dir / f"model_{index:02d}" if cli.model else output_dir)
        for index, (label, path) in enumerate(candidate_paths, start=1)
    ]
    if cli.model:
        jobs.append((None, baseline_path, conditions[:1], output_dir / "clean_baseline"))

    rows_by_model = {}
    baseline_rows = None
    for label, actor_path, run_conditions, run_dir in jobs:
        actor_a = base.load_controller("actor_a", actor_path, actor_a_scenario, lowlevel_policy, device)
        if (actor_a.is_hierarchical or actor_b.is_hierarchical) and lowlevel_policy is None:
            lowlevel_policy = base.load_lowlevel_policy(device)
            actor_a.lowlevel_policy = lowlevel_policy
            actor_b.lowlevel_policy = lowlevel_policy

        snapshot = config_snapshot(actor_path, actor_b_path, run_conditions)
        snapshot.update({
            "actor_a_scenario_name": actor_a_scenario,
            "actor_b_scenario_name": actor_b_scenario,
            "eval_scenario_name": eval_scenario,
            "num_episodes_per_condition": num_episodes,
            "device": str(device),
            "observation_mode": observation_mode,
            "degrade_both": bool(cli.degrade_both),
        })
        if cli.model:
            snapshot["model_label"] = label or "干净 A 基线（E0）"
            print(f"\n===== {snapshot['model_label']} =====")
        rows = evaluate_pair(base, actor_a, actor_b, run_conditions, run_dir, snapshot, quick_mode=quick_mode)
        if label is None:
            baseline_rows = rows
        else:
            rows_by_model[label] = rows

    if cli.model:
        plot_path = plot_model_reward_comparison(rows_by_model, baseline_rows, output_dir)
        if plot_path is not None:
            print(f"奖励对比图: {plot_path}")


def evaluate_pair(
    base: Any,
    actor_a: Any,
    actor_b: Any,
    conditions: Sequence[ExperimentCondition],
    output_dir: Path,
    snapshot: Dict[str, Any],
    *,
    quick_mode: bool,
) -> List[Dict[str, Any]]:
    """评估一组 A/B，按原有目录结构保存逐局结果和汇总。"""
    save_json(snapshot, output_dir / "config.json")
    num_episodes = snapshot["num_episodes_per_condition"]

    print(f"输出目录: {output_dir}")
    print(f"设备: {snapshot['device']}")
    print(f"每组回合数: {num_episodes}")
    print(f"退化口径: {snapshot['observation_mode']} | 双方同退化: {snapshot['degrade_both']}")

    all_rows: List[Dict[str, object]] = []
    summaries: List[Dict[str, object]] = []
    for condition_index, condition in enumerate(conditions):
        print(f"\n===== {condition.condition_id}: {condition.label} =====")
        condition_dir = output_dir / condition.condition_id
        condition_dir.mkdir(parents=True, exist_ok=True)
        env = base.SingleCombatEnv(snapshot["eval_scenario_name"])
        env.seed(ENV_SEED)
        try:
            base.validate_setup(env, [actor_a, actor_b])
            rows = [
                run_episode(
                    base,
                    condition,
                    episode,
                    env,
                    actor_a,
                    actor_b,
                    condition_dir,
                    save_acmi=not quick_mode,
                    degrade_both=snapshot["degrade_both"],
                )
                for episode in range(num_episodes)
            ]
        finally:
            env.close()

        summary = summarize_condition(
            condition,
            rows,
            BOOTSTRAP_SEED + condition_index * 100,
        )
        save_csv(rows, condition_dir / "episodes.csv")
        save_json(summary, condition_dir / "summary.json")
        all_rows.extend(rows)
        summaries.append(summary)

    if summaries:
        baseline = summaries[0]
        for summary in summaries:
            summary["win_score_change_from_e0"] = (
                float(summary["win_score"]) - float(baseline["win_score"])
            )
            summary["advantage_change_from_e0"] = (
                float(summary["advantage_window_occupancy"])
                - float(baseline["advantage_window_occupancy"])
            )
            summary["actor_a_win_rate_change_from_e0"] = (
                float(summary["actor_a_win_rate"])
                - float(baseline["actor_a_win_rate"])
            )
            summary["actor_a_reward_change_from_e0"] = (
                float(summary["actor_a_avg_reward"])
                - float(baseline["actor_a_avg_reward"])
            )
            summary["reward_margin_drop_from_e0"] = (
                float(baseline["mean_reward_margin"]) - float(summary["mean_reward_margin"])
            )

    save_csv(all_rows, output_dir / "all_episodes.csv")
    save_csv(summaries, output_dir / "summary.csv")
    plot_paths = []
    if not quick_mode:
        plot_paths = plot_summaries(summaries, output_dir)
        plot_paths.extend(plot_episode_reward_curves(all_rows, output_dir))
    combined_summary = {
        "output_dir": str(output_dir),
        "conditions": summaries,
        "plot_files": [str(path) for path in plot_paths],
    }
    save_json(combined_summary, output_dir / "summary.json")
    print("\n===== 全部实验完成 =====")
    print(json.dumps(combined_summary, ensure_ascii=False, indent=2))
    return all_rows


if __name__ == "__main__":
    main()
