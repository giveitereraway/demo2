#!/usr/bin/env python
"""
独立评估 GRU 或时序注意力低层控制器的指令跟踪能力。

实验会在 SingleControl/heading 环境中手动注入航向、高度、速度目标，
记录真实跟踪误差、越界率和控制平滑性，结果用于支撑分层架构中
低层飞控策略可以稳定执行高层指令。

1、读取底层模型, 初始状态随机(20个seed)
2、指令集是高层模型映射下来的输入, 共45种指令
3、20✖45=900个 episode, 每个episode最多跑30秒, 如果 episode 内无物理越界且末端误差满足阈值，则认为成功。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from gymnasium import spaces


# =========================
# 可修改实验参数E:\clone\demo2\scripts\results\SingleControl\1\heading\ppo\demo2_heading\wandb\offline-run-20260601_175507-5x9h1hyu\files
# =========================

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "scripts/results/SingleControl/1/heading/ppo/v1/wandb/run-20260120_173422-hvmwmud7/files/actor_latest.pt"
SCENARIO_NAME = "1/heading"
OUTPUT_DIR = REPO_ROOT / "experiments" / "results" / "chapter_3"

# 主实验指令集：对应分层高层任务常用的底层指令范围。
COMMAND_ALTITUDE_M = [-100.0, 0.0, 100.0]
COMMAND_HEADING_DEG = [-30.0, -15.0, 0.0, 15.0, 30.0]
COMMAND_SPEED_MPS = [-17.0, 0.0, 17.0]

SEEDS = list(range(1, 21))
HORIZON_SEC = 30.0 # 每个 episode 最多跑 30 秒。
SETTLING_WINDOW_SEC = 5.0 # 要求连续 5 秒都满足误差阈值，才认为进入稳定跟踪状态

# 成功判据：末端误差同时满足阈值，且 episode 内无物理越界。
SUCCESS_HEADING_DEG = 10.0 # 航向误差 <= 10 度
SUCCESS_ALTITUDE_M = 100.0 # 高度误差 <= 100 米
SUCCESS_SPEED_MPS = 24.0 # 速度误差 <= 24 m/s

# 物理安全边界，与 heading.yaml 和终止条件保持一致或更保守。
LOW_ALTITUDE_LIMIT_M = 2500.0 # 高度低于 2500m 算低空越界
OVERLOAD_CHECK_AFTER_SEC = 10.0 # 仿真 10 秒之后才检查过载
OVERLOAD_LIMIT_X = 10.0 # X 轴加速度不超过 10 m/s^2
OVERLOAD_LIMIT_Y = 10.0 # Y 轴加速度不超过 10 m/s^2
OVERLOAD_LIMIT_Z = 10.0 # Z 轴加速度不超过 10 m/s^2

# 目标值安全裁剪，防止随机初始状态叠加指令后落入明显危险区。
CLAMP_TARGETS_TO_SAFE_RANGE = True
MIN_TARGET_ALTITUDE_M = 3500.0
MAX_TARGET_ALTITUDE_M = 11000.0
MIN_TARGET_SPEED_MPS = 120.0
MAX_TARGET_SPEED_MPS = 420.0

# 固定目标，不让 UnreachHeading 在 30 秒检查点自动切换新目标。
FIXED_HEADING_CHECK_TIME_SEC = 1_000_000.0

# 无模型配置的旧 GRU 权重使用与分层任务一致的结构参数。
ACTOR_GAIN = 0.01
ACTOR_HIDDEN_SIZE = "128 128"
ACTOR_ACT_HIDDEN_SIZE = "128 128"
ACTOR_ACTIVATION_ID = 1
ACTOR_USE_FEATURE_NORMALIZATION = False
ACTOR_USE_RECURRENT_POLICY = True
ACTOR_RECURRENT_HIDDEN_SIZE = 128
ACTOR_RECURRENT_HIDDEN_LAYERS = 1
ACTOR_USE_PRIOR = False

DEVICE = "cuda:0"
WRITE_TIMESERIES = True
MAKE_PLOTS = True
PRINT_EACH_TRIAL = True


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.ppo.ppo_actor import PPOActor  # noqa: E402
from envs.JSBSim.core.catalog import Catalog as c  # noqa: E402
from envs.JSBSim.envs.singlecontrol_env import SingleControlEnv  # noqa: E402


FT_TO_M = 0.3048
ACTION_NAMES = ("aileron", "elevator", "rudder", "throttle")
AGENT_ID = "A0100"


@dataclass(frozen=True)
class Command:
    delta_altitude_m: float
    delta_heading_deg: float
    delta_velocity_mps: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="评估 GRU 或时序注意力模型对航向、高度、速度指令的跟踪效果。"
    )
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH,
                        help="actor 权重文件，模型配置 model_config.json 与权重放在同一目录。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--scenario-name", type=str, default=SCENARIO_NAME)
    parser.add_argument("--horizon-sec", type=float, default=HORIZON_SEC)
    parser.add_argument("--num-seeds", type=int, default=len(SEEDS))
    parser.add_argument("--no-timeseries", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="只跑 1 个 seed 和少量指令，用于快速检查脚本是否能启动。",
    )
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_actor(model_path: Path, device: torch.device) -> PPOActor:
    args = SimpleNamespace(
        gain=ACTOR_GAIN,
        hidden_size=ACTOR_HIDDEN_SIZE,
        act_hidden_size=ACTOR_ACT_HIDDEN_SIZE,
        activation_id=ACTOR_ACTIVATION_ID,
        use_feature_normalization=ACTOR_USE_FEATURE_NORMALIZATION,
        use_recurrent_policy=ACTOR_USE_RECURRENT_POLICY,
        recurrent_hidden_size=ACTOR_RECURRENT_HIDDEN_SIZE,
        recurrent_hidden_layers=ACTOR_RECURRENT_HIDDEN_LAYERS,
        use_prior=ACTOR_USE_PRIOR,
    )
    obs_space = spaces.Box(low=-10, high=10.0, shape=(12,))
    act_space = spaces.MultiDiscrete([41, 41, 41, 30])
    config_path = model_path.with_name("model_config.json")
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else None
    if config is not None:
        if config["observation_space"] != str(obs_space) or config["action_space"] != str(act_space):
            raise ValueError(f"{config_path} 的观测或动作空间与 Heading 任务不匹配")
        args.memory_encoder = config["memory_encoder"]
        args.hidden_size = config["hidden_size"]
        args.act_hidden_size = config["act_hidden_size"]
        args.activation_id = config["activation_id"]
        args.use_feature_normalization = config["use_feature_normalization"]
        args.use_prior = config["use_prior"]
        if args.memory_encoder == "gru":
            args.recurrent_hidden_layers, args.recurrent_hidden_size = config["state_shape"]
        elif args.memory_encoder == "temporal_attention":
            args.history_length = config["history_length"]
            args.attention_hidden_size = config["attention_hidden_size"]
            args.attention_heads = config["attention_heads"]
        else:
            raise ValueError(f"{config_path} 的编码器不受支持：{args.memory_encoder}")
    actor = PPOActor(args, obs_space, act_space, device=device)
    if config is not None and tuple(config["state_shape"]) != actor.state_shape:
        raise ValueError(f"{config_path} 的状态尺寸与 actor 不匹配，期望 {actor.state_shape}")
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)
    try:
        actor.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(f"无法加载 {model_path}，请核对同目录的 model_config.json 与权重是否匹配") from exc
    actor.eval()
    actor.requires_grad_(False)
    return actor


def build_commands(quick: bool = False) -> list[Command]:
    commands = [
        Command(delta_altitude_m, delta_heading_deg, delta_velocity_mps)
        for delta_altitude_m in COMMAND_ALTITUDE_M
        for delta_heading_deg in COMMAND_HEADING_DEG
        for delta_velocity_mps in COMMAND_SPEED_MPS
    ]
    if quick:
        return [
            Command(0.0, 0.0, 0.0),
            Command(100.0, 30.0, 17.0),
            Command(-100.0, -30.0, -17.0),
        ]
    return commands


def _clip_target(value: float, lower: float, upper: float) -> float:
    return float(min(max(value, lower), upper))


def set_command_target(env: SingleControlEnv, command: Command) -> dict[str, float]:
    sim = env.agents[AGENT_ID]
    current_heading_deg = sim.get_property_value(c.attitude_psi_deg)
    current_altitude_ft = sim.get_property_value(c.position_h_sl_ft)
    current_altitude_m = sim.get_property_value(c.position_h_sl_m)
    current_speed_mps = sim.get_property_value(c.velocities_u_mps)

    target_heading_deg = (current_heading_deg + command.delta_heading_deg) % 360.0
    target_altitude_m = current_altitude_m + command.delta_altitude_m
    target_speed_mps = current_speed_mps + command.delta_velocity_mps

    if CLAMP_TARGETS_TO_SAFE_RANGE:
        target_altitude_m = _clip_target(
            target_altitude_m, MIN_TARGET_ALTITUDE_M, MAX_TARGET_ALTITUDE_M
        )
        target_speed_mps = _clip_target(
            target_speed_mps, MIN_TARGET_SPEED_MPS, MAX_TARGET_SPEED_MPS
        )

    target_altitude_ft = target_altitude_m / FT_TO_M
    sim.set_property_value(c.target_heading_deg, target_heading_deg)
    sim.set_property_value(c.target_altitude_ft, target_altitude_ft)
    sim.set_property_value(c.target_velocities_u_mps, target_speed_mps)
    sim.set_property_value(c.heading_check_time, FIXED_HEADING_CHECK_TIME_SEC)

    return {
        "initial_heading_deg": current_heading_deg,
        "initial_altitude_m": current_altitude_m,
        "initial_altitude_ft": current_altitude_ft,
        "initial_speed_mps": current_speed_mps,
        "target_heading_deg": target_heading_deg,
        "target_altitude_m": target_altitude_m,
        "target_speed_mps": target_speed_mps,
        "actual_delta_heading_deg": command.delta_heading_deg,
        "actual_delta_altitude_m": target_altitude_m - current_altitude_m,
        "actual_delta_velocity_mps": target_speed_mps - current_speed_mps,
    }


def get_state_record(env: SingleControlEnv) -> dict[str, Any]:
    sim = env.agents[AGENT_ID]
    altitude_m = sim.get_property_value(c.position_h_sl_m)
    sim_time_sec = sim.get_property_value(c.simulation_sim_time_sec)
    accel_x = sim.get_property_value(c.accelerations_n_pilot_x_norm)
    accel_y = sim.get_property_value(c.accelerations_n_pilot_y_norm)
    accel_z = sim.get_property_value(c.accelerations_n_pilot_z_norm)
    extreme_state = bool(sim.get_property_value(c.detect_extreme_state))

    low_altitude = altitude_m <= LOW_ALTITUDE_LIMIT_M
    overload = False
    if sim_time_sec > OVERLOAD_CHECK_AFTER_SEC:
        overload = (
            abs(accel_x) > OVERLOAD_LIMIT_X
            or abs(accel_y) > OVERLOAD_LIMIT_Y
            or abs(accel_z + 1.0) > OVERLOAD_LIMIT_Z
        )

    return {
        "time_sec": env.current_step * env.time_interval,
        "sim_time_sec": sim_time_sec,
        "heading_error_deg": sim.get_property_value(c.delta_heading),
        "altitude_error_m": sim.get_property_value(c.delta_altitude),
        "speed_error_mps": sim.get_property_value(c.delta_velocities_u),
        "altitude_m": altitude_m,
        "roll_deg": math.degrees(sim.get_property_value(c.attitude_roll_rad)),
        "pitch_deg": math.degrees(sim.get_property_value(c.attitude_pitch_rad)),
        "heading_deg": sim.get_property_value(c.attitude_psi_deg),
        "speed_u_mps": sim.get_property_value(c.velocities_u_mps),
        "vc_mps": sim.get_property_value(c.velocities_vc_mps),
        "accel_x_g": accel_x,
        "accel_y_g": accel_y,
        "accel_z_g": accel_z,
        "low_altitude": low_altitude,
        "overload": overload,
        "extreme_state": extreme_state,
        "violation": low_altitude or overload or extreme_state,
    }


def infer_termination_reason(last_record: dict[str, Any], done: bool) -> str:
    if not done:
        return "horizon"
    if last_record.get("low_altitude"):
        return "low_altitude"
    if last_record.get("overload"):
        return "overload"
    if last_record.get("extreme_state"):
        return "extreme_state"
    return "task_done"


def compute_settling_time(records: list[dict[str, Any]], window_steps: int) -> float | None:
    if not records:
        return None
    ok = np.array(
        [
            abs(row["heading_error_deg"]) <= SUCCESS_HEADING_DEG
            and abs(row["altitude_error_m"]) <= SUCCESS_ALTITUDE_M
            and abs(row["speed_error_mps"]) <= SUCCESS_SPEED_MPS
            for row in records
        ],
        dtype=bool,
    )
    if len(ok) < window_steps:
        return None
    for start in range(0, len(ok) - window_steps + 1):
        if bool(np.all(ok[start : start + window_steps])):
            return float(records[start]["time_sec"])
    return None


def summarize_episode(
    trial_id: int,
    seed: int,
    command: Command,
    target_info: dict[str, float],
    records: list[dict[str, Any]],
    controls: list[np.ndarray],
    done: bool,
    time_interval: float,
) -> dict[str, Any]:
    heading_abs = np.array([abs(row["heading_error_deg"]) for row in records], dtype=np.float64)
    altitude_abs = np.array([abs(row["altitude_error_m"]) for row in records], dtype=np.float64)
    speed_abs = np.array([abs(row["speed_error_mps"]) for row in records], dtype=np.float64)
    violations = np.array([row["violation"] for row in records], dtype=bool)
    last_record = records[-1]

    control_arr = np.vstack(controls) if controls else np.zeros((0, 4), dtype=np.float64)
    if len(control_arr) > 1:
        control_diff = np.diff(control_arr, axis=0)
        smoothness_l1 = float(np.mean(np.abs(control_diff)))
        smoothness_l2 = float(np.mean(np.linalg.norm(control_diff, axis=1)))
        per_channel = np.mean(np.abs(control_diff), axis=0)
    else:
        smoothness_l1 = 0.0
        smoothness_l2 = 0.0
        per_channel = np.zeros(4, dtype=np.float64)

    window_steps = max(1, int(math.ceil(SETTLING_WINDOW_SEC / time_interval)))
    settling_time = compute_settling_time(records, window_steps)
    success = (
        abs(last_record["heading_error_deg"]) <= SUCCESS_HEADING_DEG
        and abs(last_record["altitude_error_m"]) <= SUCCESS_ALTITUDE_M
        and abs(last_record["speed_error_mps"]) <= SUCCESS_SPEED_MPS
        and not bool(np.any(violations))
    )

    row: dict[str, Any] = {
        "trial_id": trial_id,
        "seed": seed,
        "requested_delta_altitude_m": command.delta_altitude_m,
        "requested_delta_heading_deg": command.delta_heading_deg,
        "requested_delta_velocity_mps": command.delta_velocity_mps,
        **target_info,
        "num_steps": len(records),
        "duration_sec": float(records[-1]["time_sec"]) if records else 0.0,
        "done": bool(done),
        "termination_reason": infer_termination_reason(last_record, done),
        "success": bool(success),
        "settling_time_sec": settling_time,
        "heading_mae_deg": float(np.mean(heading_abs)),
        "heading_rmse_deg": float(np.sqrt(np.mean(np.square(heading_abs)))),
        "heading_p95_deg": float(np.percentile(heading_abs, 95)),
        "heading_final_abs_deg": float(heading_abs[-1]),
        "heading_max_abs_deg": float(np.max(heading_abs)),
        "altitude_mae_m": float(np.mean(altitude_abs)),
        "altitude_rmse_m": float(np.sqrt(np.mean(np.square(altitude_abs)))),
        "altitude_p95_m": float(np.percentile(altitude_abs, 95)),
        "altitude_final_abs_m": float(altitude_abs[-1]),
        "altitude_max_abs_m": float(np.max(altitude_abs)),
        "speed_mae_mps": float(np.mean(speed_abs)),
        "speed_rmse_mps": float(np.sqrt(np.mean(np.square(speed_abs)))),
        "speed_p95_mps": float(np.percentile(speed_abs, 95)),
        "speed_final_abs_mps": float(speed_abs[-1]),
        "speed_max_abs_mps": float(np.max(speed_abs)),
        "violation_flag": bool(np.any(violations)),
        "violation_step_rate": float(np.mean(violations)),
        "low_altitude_flag": bool(any(row["low_altitude"] for row in records)),
        "overload_flag": bool(any(row["overload"] for row in records)),
        "extreme_state_flag": bool(any(row["extreme_state"] for row in records)),
        "smoothness_l1": smoothness_l1,
        "smoothness_l2": smoothness_l2,
        "control_rate_l1": smoothness_l1 / time_interval,
        "control_rate_l2": smoothness_l2 / time_interval,
    }
    for idx, name in enumerate(ACTION_NAMES):
        row[f"{name}_mean_abs_delta"] = float(per_channel[idx])
        row[f"{name}_rate_abs_delta"] = float(per_channel[idx] / time_interval)
    return row


def run_trial(
    env: SingleControlEnv,
    actor: PPOActor,
    trial_id: int,
    seed: int,
    command: Command,
    horizon_steps: int,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    set_random_seed(seed)
    env.seed(seed)
    env.reset()
    target_info = set_command_target(env, command)

    policy_state = np.zeros((1, *actor.state_shape), dtype=np.float32)
    masks = np.ones((1, 1), dtype=np.float32)
    records: list[dict[str, Any]] = []
    controls: list[np.ndarray] = []
    done = False

    for step_idx in range(horizon_steps):
        obs = env.task.get_obs(env, AGENT_ID).astype(np.float32)
        obs = np.expand_dims(obs, axis=0)
        with torch.no_grad():
            action_tensor, _, next_state = actor(
                obs,
                policy_state,
                masks,
                deterministic=True,
            )
        action = action_tensor.detach().cpu().numpy().reshape(1, -1).astype(np.int64)
        policy_state = next_state.detach().cpu().numpy()

        continuous_action = env.task.normalize_action(env, AGENT_ID, action[0])
        controls.append(np.asarray(continuous_action, dtype=np.float64))

        _, _, dones, _ = env.step(action)
        done = bool(np.asarray(dones).reshape(-1)[0])
        record = get_state_record(env)
        record.update(
            {
                "trial_id": trial_id,
                "seed": seed,
                "step": step_idx + 1,
                "requested_delta_altitude_m": command.delta_altitude_m,
                "requested_delta_heading_deg": command.delta_heading_deg,
                "requested_delta_velocity_mps": command.delta_velocity_mps,
                "actual_delta_altitude_m": target_info["actual_delta_altitude_m"],
                "actual_delta_heading_deg": target_info["actual_delta_heading_deg"],
                "actual_delta_velocity_mps": target_info["actual_delta_velocity_mps"],
                "action_aileron_idx": int(action[0, 0]),
                "action_elevator_idx": int(action[0, 1]),
                "action_rudder_idx": int(action[0, 2]),
                "action_throttle_idx": int(action[0, 3]),
                "aileron_cmd": float(continuous_action[0]),
                "elevator_cmd": float(continuous_action[1]),
                "rudder_cmd": float(continuous_action[2]),
                "throttle_cmd": float(continuous_action[3]),
            }
        )
        records.append(record)
        if done:
            break

    episode_row = summarize_episode(
        trial_id=trial_id,
        seed=seed,
        command=command,
        target_info=target_info,
        records=records,
        controls=controls,
        done=done,
        time_interval=env.time_interval,
    )
    _ = device
    return episode_row, records


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = [
        "heading_mae_deg",
        "heading_p95_deg",
        "heading_final_abs_deg",
        "altitude_mae_m",
        "altitude_p95_m",
        "altitude_final_abs_m",
        "speed_mae_mps",
        "speed_p95_mps",
        "speed_final_abs_mps",
        "violation_step_rate",
        "smoothness_l1",
        "smoothness_l2",
        "control_rate_l1",
        "control_rate_l2",
    ]
    summary: dict[str, Any] = {
        "num_episodes": len(rows),
        "success_rate": float(np.mean([row["success"] for row in rows])) if rows else 0.0,
        "violation_episode_rate": float(np.mean([row["violation_flag"] for row in rows])) if rows else 0.0,
        "early_done_rate": float(np.mean([row["done"] for row in rows])) if rows else 0.0,
    }
    for name in metric_names:
        values = np.array([row[name] for row in rows if row[name] is not None], dtype=np.float64)
        if len(values) == 0:
            continue
        summary[f"{name}_mean"] = float(np.mean(values))
        summary[f"{name}_std"] = float(np.std(values))
        summary[f"{name}_p50"] = float(np.percentile(values, 50))
        summary[f"{name}_p95"] = float(np.percentile(values, 95))
    return summary


def summarize_by_command(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[float, float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["requested_delta_altitude_m"],
            row["requested_delta_heading_deg"],
            row["requested_delta_velocity_mps"],
        )
        groups[key].append(row)

    output = []
    for key, group_rows in sorted(groups.items()):
        item = {
            "requested_delta_altitude_m": key[0],
            "requested_delta_heading_deg": key[1],
            "requested_delta_velocity_mps": key[2],
            **summarize_rows(group_rows),
        }
        output.append(item)
    return output


def write_summary(path: Path, rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    summary = {
        "config": config,
        "overall": summarize_rows(rows),
        "by_command": summarize_by_command(rows),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=_to_jsonable)


def configure_matplotlib_chinese_font() -> None:
    from matplotlib import font_manager, rcParams

    # 优先使用 Windows 常见中文字体，避免保存图片时中文标签显示成方块。
    font_files = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path("C:/Windows/Fonts/Deng.ttf"),
    ]
    for font_file in font_files:
        if font_file.exists():
            font_manager.fontManager.addfont(str(font_file))
            font_name = font_manager.FontProperties(fname=str(font_file)).get_name()
            rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            rcParams["axes.unicode_minus"] = False
            return

    font_names = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "DengXian",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
    ]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in font_names:
        if font_name in available_fonts:
            rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            rcParams["axes.unicode_minus"] = False
            return

    rcParams["axes.unicode_minus"] = False
    print("[WARN] 未找到可用中文字体，图片中的中文可能仍无法正确显示。")


def plot_timeseries(timeseries_rows: list[dict[str, Any]], output_dir: Path) -> None:
    if not timeseries_rows:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib 不可用，跳过曲线图生成。")
        return

    configure_matplotlib_chinese_font()

    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in timeseries_rows:
        by_step[int(row["step"])].append(row)
    steps = sorted(by_step.keys())
    times = np.array([np.median([item["time_sec"] for item in by_step[step]]) for step in steps])

    error_specs = [
        ("heading_error_deg", "航向误差 abs(deg)"),
        ("altitude_error_m", "高度误差 abs(m)"),
        ("speed_error_mps", "速度误差 abs(m/s)"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for ax, (metric, label) in zip(axes, error_specs):
        values = [
            np.array([abs(item[metric]) for item in by_step[step]], dtype=np.float64)
            for step in steps
        ]
        median = np.array([np.median(v) for v in values])
        q25 = np.array([np.percentile(v, 25) for v in values])
        q75 = np.array([np.percentile(v, 75) for v in values])
        ax.plot(times, median, label="median")
        ax.fill_between(times, q25, q75, alpha=0.25, label="25%-75%")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    axes[-1].set_xlabel("时间 (s)")
    fig.tight_layout()
    fig.savefig(output_dir / "tracking_errors.png", dpi=180)
    plt.close(fig)

    control_specs = [
        ("aileron_cmd", "副翼"),
        ("elevator_cmd", "升降舵"),
        ("rudder_cmd", "方向舵"),
        ("throttle_cmd", "油门"),
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    for metric, label in control_specs:
        median = np.array(
            [
                np.median([item[metric] for item in by_step[step]])
                for step in steps
            ]
        )
        ax.plot(times, median, label=label)
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("连续控制量")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "control_commands.png", dpi=180)
    plt.close(fig)


def run_evaluation(args: argparse.Namespace) -> None:
    model_path = args.model_path.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = SEEDS[: max(1, args.num_seeds)]
    if args.quick:
        seeds = seeds[:1]
    commands = build_commands(quick=args.quick)
    device = torch.device(DEVICE)
    actor = build_actor(model_path, device)
    env = SingleControlEnv(args.scenario_name)
    horizon_steps = max(1, int(math.ceil(args.horizon_sec / env.time_interval)))
    total_trials = len(seeds) * len(commands)

    config = {
        "model_path": str(model_path),
        "scenario_name": args.scenario_name,
        "output_dir": str(output_dir),
        "seeds": seeds,
        "horizon_sec": args.horizon_sec,
        "horizon_steps": horizon_steps,
        "time_interval_sec": env.time_interval,
        "commands": [command.__dict__ for command in commands],
        "success_thresholds": {
            "heading_deg": SUCCESS_HEADING_DEG,
            "altitude_m": SUCCESS_ALTITUDE_M,
            "speed_mps": SUCCESS_SPEED_MPS,
        },
        "clamp_targets_to_safe_range": CLAMP_TARGETS_TO_SAFE_RANGE,
        "write_timeseries": not args.no_timeseries and WRITE_TIMESERIES,
        "make_plots": not args.no_plots and MAKE_PLOTS,
    }

    print(f"模型: {model_path}")
    print(f"输出目录: {output_dir}")
    print(f"总实验数: {total_trials} = {len(seeds)} seeds x {len(commands)} commands")

    episode_rows: list[dict[str, Any]] = []
    timeseries_rows: list[dict[str, Any]] = []
    trial_id = 0
    try:
        for seed in seeds:
            for command in commands:
                trial_id += 1
                if PRINT_EACH_TRIAL:
                    print(
                        f"[{trial_id}/{total_trials}] seed={seed}, "
                        f"dh={command.delta_altitude_m:+.0f}m, "
                        f"dpsi={command.delta_heading_deg:+.0f}deg, "
                        f"dv={command.delta_velocity_mps:+.0f}m/s"
                    )
                episode_row, trial_records = run_trial(
                    env=env,
                    actor=actor,
                    trial_id=trial_id,
                    seed=seed,
                    command=command,
                    horizon_steps=horizon_steps,
                    device=device,
                )
                episode_rows.append(episode_row)
                if config["write_timeseries"]:
                    timeseries_rows.extend(trial_records)
    finally:
        env.close()

    write_csv(output_dir / "episodes.csv", episode_rows)
    if config["write_timeseries"]:
        write_csv(output_dir / "timeseries.csv", timeseries_rows)
    write_summary(output_dir / "summary.json", episode_rows, config)
    if config["make_plots"]:
        plot_timeseries(timeseries_rows, output_dir)

    overall = summarize_rows(episode_rows)
    print("\n完成。核心结果：")
    print(f"success_rate={overall.get('success_rate', 0.0):.3f}")
    print(f"violation_episode_rate={overall.get('violation_episode_rate', 0.0):.3f}")
    print(f"heading_final_abs_deg_mean={overall.get('heading_final_abs_deg_mean', 0.0):.3f}")
    print(f"altitude_final_abs_m_mean={overall.get('altitude_final_abs_m_mean', 0.0):.3f}")
    print(f"speed_final_abs_mps_mean={overall.get('speed_final_abs_mps_mean', 0.0):.3f}")


def main() -> None:
    args = parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
