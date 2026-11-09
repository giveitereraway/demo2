#!/usr/bin/env python
"""用固定阶跃指令评估 heading 策略的观测退化鲁棒性和历史依赖。

传入一个或多个 actor 权重文件，也可传入包含 actor_latest.pt 的单次训练目录。
注意力权重需要同目录的 model_config.json。PPO/SPO 标签由模型参数指定，
两者推理都使用 PPOActor。
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import evaluate_lowlevel_controller as heading
from experiments.partial_observability_gru_eval import (
    GRU_HISTORY_CONDITION_IDS as MEMORY_CONDITION_IDS,
    PLOT_DPI,
    REWARD_MOVING_AVERAGE_WINDOW,
    ROBUSTNESS_CONDITION_IDS,
    save_csv,
    save_json,
    trailing_moving_average,
)


MODEL_ENCODERS = {
    "ppo-gru": "gru",
    "ppo-attention": "temporal_attention",
    "spo-gru": "gru",
    "spo-attention": "temporal_attention",
}
OUTPUT_ROOT = REPO_ROOT / "experiments" / "results" / "chapter_3" /"observation_degradation" / "heading_observation_degradation"
SENSOR_SEED = 31001
HORIZON_SEC = 30.0
ERROR_SLICE = slice(0, 3)


@dataclass(frozen=True)
class Degradation:
    dropout_probability: float
    delay_steps: int
    altitude_noise_std_m: float
    heading_noise_std_deg: float
    speed_noise_std_mps: float


@dataclass(frozen=True)
class Condition:
    name: str
    degradation: Degradation | None = None
    reset_interval: int | None = None


# 噪声标准差按高度、航向、速度的物理单位给出。丢帧和延迟作用于整个误差向量。
# (丢帧率、时延、高度、航向、速度误差)
CONDITIONS = (
    Condition("E0"),
    Condition("E1", Degradation(0.10, 1, 10.0, 1.0, 1.0)),
    Condition("E2", Degradation(0.20, 2, 25.0, 3.0, 3.0)),
    Condition("E3", Degradation(0.30, 3, 50.0, 5.0, 5.0)),
    Condition("E4", reset_interval=20),
    Condition("E5", reset_interval=5),
    Condition("E6", reset_interval=1),
)


class TrackingErrorDegrader:
    """仅退化 heading 观测的前三维，保留实时本机状态。"""

    def __init__(self, config: Degradation, initial_obs: np.ndarray, rng: np.random.Generator):
        self.rng = rng
        self.dropout_probability = config.dropout_probability
        self.noise_std = np.array([
            config.altitude_noise_std_m / 1000.0,
            np.deg2rad(config.heading_noise_std_deg),
            config.speed_noise_std_mps / 340.0,
        ])
        # 和原实验一致，用回合首帧填充延迟队列。
        self.last_valid = initial_obs[ERROR_SLICE].copy()
        self.delay = deque([self.last_valid] * config.delay_steps)
        self.dropped_frames = 0

    def transform(self, obs: np.ndarray) -> np.ndarray:
        measured = obs[ERROR_SLICE] + self.rng.standard_normal(3) * self.noise_std
        measured[1] = (measured[1] + np.pi) % (2 * np.pi) - np.pi
        if self.rng.random() < self.dropout_probability:
            self.dropped_frames += 1
        else:
            self.last_valid = np.clip(measured, -10.0, 10.0)
        self.delay.append(self.last_valid)
        degraded = obs.copy()
        degraded[ERROR_SLICE] = self.delay.popleft()
        return degraded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in MODEL_ENCODERS:
        parser.add_argument(f"--{name}", type=Path, help="权重文件或单次训练目录")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT, help="结果根目录")
    parser.add_argument("--num-seeds", type=int, default=len(heading.SEEDS), help="初始状态种子数")
    parser.add_argument("--device", default="auto", help="auto、cpu 或 cuda:0 等")
    parser.add_argument("--quick", action="store_true", help="使用 1 个种子和 3 条指令跑完 E0–E6")
    parser.add_argument("--no-plots", action="store_true", help="仅输出表格和配置")
    args = parser.parse_args()
    args.models = {
        name: path for name in MODEL_ENCODERS
        if (path := getattr(args, name.replace("-", "_"))) is not None
    }
    if not args.models:
        parser.error("至少传入一种模型，如 --ppo-gru ACTOR_PATH")
    if not 1 <= args.num_seeds <= len(heading.SEEDS):
        parser.error(f"--num-seeds 必须在 1 到 {len(heading.SEEDS)} 之间")
    return args


def load_actor(path: Path, encoder: str, device: torch.device) -> tuple[Path, heading.PPOActor]:
    path = path.resolve()
    if path.is_dir():
        path = path / "actor_latest.pt"
    if not path.is_file():
        raise FileNotFoundError(f"权重文件不存在：{path}")
    if encoder == "temporal_attention" and not path.with_name("model_config.json").is_file():
        raise FileNotFoundError(f"注意力权重缺少同目录的 model_config.json：{path}")
    actor = heading.build_actor(path, device)
    if actor.memory_encoder != encoder:
        raise ValueError(f"模型 {path} 的编码器为 {actor.memory_encoder}，期望 {encoder}")
    if actor.use_prior:
        raise ValueError(f"heading 模型必须关闭导弹发射先验：{path}")
    return path, actor


@torch.inference_mode()
def run_trial(
    env: heading.SingleControlEnv,
    actor: heading.PPOActor,
    condition: Condition,
    trial_id: int,
    seed: int,
    command: heading.Command,
    horizon_steps: int,
    device: torch.device,
) -> dict:
    env.seed(seed)
    env.reset()
    target_info = heading.set_command_target(env, command)
    obs = env.task.get_obs(env, heading.AGENT_ID).astype(np.float32)
    degrader = None
    if condition.degradation is not None:
        # 同一试验跨模型和退化档位复用随机数流，并独立于上一回合长度。
        rng = np.random.default_rng([SENSOR_SEED, trial_id])
        degrader = TrackingErrorDegrader(condition.degradation, obs, rng)

    state = torch.zeros((1, *actor.state_shape), device=device)
    masks = torch.ones((1, 1), device=device)
    reset_count = 0
    episode_reward = 0.0
    records = []
    controls = []
    done = False
    interval = condition.reset_interval
    for step in range(horizon_steps):
        if interval is not None and step % interval == 0 and (step > 0 or interval == 1):
            # 注意力的历史状态包括有效位。清空旧帧后，actor 仍会纳入本步观测。
            state.zero_()
            reset_count += 1
        policy_obs = degrader.transform(obs) if degrader is not None else obs
        action, _, state = actor(policy_obs[None, :], state, masks, deterministic=True)
        action = action.cpu().numpy().astype(np.int64)
        controls.append(env.task.normalize_action(env, heading.AGENT_ID, action[0]))
        next_obs, rewards, dones, _ = env.step(action)
        obs = next_obs[0].astype(np.float32)
        episode_reward += float(rewards.item())
        records.append(heading.get_state_record(env))
        done = bool(dones.item())
        if done:
            break

    row = heading.summarize_episode(
        trial_id, seed, command, target_info, records, controls, done, env.time_interval
    )
    row["success"] = row["success"] and len(records) == horizon_steps
    row["episode_reward"] = episode_reward
    row["memory_reset_count"] = reset_count
    row["actual_dropout_rate"] = degrader.dropped_frames / len(records) if degrader else 0.0
    return row


def plot_summaries(summaries: list[dict], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = {
        "success_rate": "Success rate",
        "heading_mae_deg_mean": "Heading MAE (deg)",
        "altitude_mae_m_mean": "Altitude MAE (m)",
        "speed_mae_mps_mean": "Speed MAE (m/s)",
    }
    models = dict.fromkeys(row["model"] for row in summaries)
    for filename, conditions in (
        ("observation_degradation.png", ROBUSTNESS_CONDITION_IDS),
        ("memory_ablation.png", MEMORY_CONDITION_IDS),
    ):
        fig, axes = plt.subplots(2, 2, figsize=(9, 6), layout="constrained")
        for ax, (metric, label) in zip(axes.flat, metrics.items()):
            for model in models:
                rows = [
                    row for row in summaries if row["model"] == model and row["condition"] in conditions
                ]
                ax.plot(conditions, [row[metric] for row in rows], marker="o", label=model)
            ax.set_ylabel(label)
            ax.grid(alpha=0.25)
        axes[0, 0].set_ylim(-0.02, 1.02)
        axes[0, 0].legend()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)


def plot_mean_reward_comparison(summaries: list[dict], output_dir: Path) -> None:
    """比较不同模型在 E0–E3 的平均单局累计奖励。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {
        "ppo-gru": ("PPO+GRU", "o", "-", "tab:blue"),
        "ppo-attention": ("PPO+注意力", "s", "--", "tab:orange"),
        "spo-gru": ("SPO+GRU", "^", "-.", "tab:green"),
        "spo-attention": ("SPO+注意力", "D", ":", "tab:red"),
    }
    conditions = ROBUSTNESS_CONDITION_IDS
    rewards = {
        (row["model"], row["condition"]): float(row["episode_reward_mean"])
        for row in summaries if row["condition"] in conditions
    }
    models = {model for model, _ in rewards}
    with plt.rc_context():
        heading.configure_matplotlib_chinese_font()
        fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
        for model, (label, marker, linestyle, color) in styles.items():
            if model in models:
                values = [rewards[model, condition] for condition in conditions]
                ax.plot(conditions, values, label=label, marker=marker, linestyle=linestyle, color=color)

        ax.set_title("不同算法的平均奖励对比")
        ax.set_xlabel("观测退化条件")
        ax.set_ylabel("平均单局累计奖励")
        ax.set_xlim(-0.18, len(conditions) - 0.62)
        ax.margins(y=0.23)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="lower left", ncols=2)

        # 同一条件内按奖励从低到高排布标签，保留每个数据点的真实坐标。
        bottom, top = ax.get_ylim()
        label_gap = (top - bottom) * 0.045
        for index in range(len(conditions)):
            points = sorted((line.get_ydata()[index], line.get_color()) for line in ax.lines)
            label_y = bottom
            for reward, color in points:
                label_y = max(reward, label_y + label_gap)
                ax.annotate(
                    f"{reward:.2f}", xy=(index, reward), xytext=(index + 0.10, label_y),
                    color=color, fontsize=9, va="center",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 0.8},
                    arrowprops={"arrowstyle": "-", "color": color, "lw": 0.65},
                )
        fig.savefig(output_dir / "mean_reward_comparison.png", dpi=PLOT_DPI)
        plt.close(fig)


def plot_reward_curves(model_dir: Path, memory_encoder: str) -> None:
    """从单个模型的逐局 CSV 绘制两张奖励曲线图。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rewards_by_condition = {}
    for condition in CONDITIONS:
        path = model_dir / condition.name / "episodes.csv"
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = sorted(csv.DictReader(stream), key=lambda row: int(row["trial_id"]))
        if not rows:
            raise ValueError(f"缺少单局累计奖励数据：{path}")
        trials = [int(row["trial_id"]) for row in rows]
        rewards = [float(row["episode_reward"]) for row in rows]
        rewards_by_condition[condition.name] = (trials, rewards)

    memory_title = "GRU 隐状态清零" if memory_encoder == "gru" else "注意力历史帧清空"
    groups = (
        ("observation_degradation_reward_curves.png", "实验一：观测干扰", ROBUSTNESS_CONDITION_IDS),
        ("memory_ablation_reward_curves.png", f"实验二：{memory_title}", MEMORY_CONDITION_IDS),
    )
    with plt.rc_context():
        heading.configure_matplotlib_chinese_font()
        colors = plt.get_cmap("tab10").colors
        for filename, title, condition_ids in groups:
            fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
            for color, condition_id in zip(colors, condition_ids):
                trials, rewards = rewards_by_condition[condition_id]
                smoothed = trailing_moving_average(rewards, REWARD_MOVING_AVERAGE_WINDOW)
                ax.plot(trials, rewards, color=color, alpha=0.22, linewidth=0.9)
                ax.plot(trials, smoothed, color=color, linewidth=2.0, label=f"{condition_id} 滑动平均")
            ax.set_title(title)
            ax.set_xlabel("测试局数")
            ax.set_ylabel("单局累计奖励")
            ax.grid(alpha=0.25)
            ax.legend(loc="lower left", title=f"{REWARD_MOVING_AVERAGE_WINDOW} 局窗口")
            fig.savefig(model_dir / filename, dpi=PLOT_DPI)
            plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device(
        ("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    if device.type == "cpu":
        torch.set_num_threads(1)
    models = {
        name: load_actor(path, MODEL_ENCODERS[name], device) for name, path in args.models.items()
    }
    seeds = heading.SEEDS[:1 if args.quick else args.num_seeds]
    commands = heading.build_commands(quick=args.quick)
    trials = [(seed, command) for seed in seeds for command in commands]
    env = heading.SingleControlEnv(heading.SCENARIO_NAME)
    output_dir = args.output_dir.resolve() / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    summaries = []
    try:
        horizon_steps = math.ceil(HORIZON_SEC / env.time_interval)
        save_json({
            "scenario": heading.SCENARIO_NAME,
            "device": str(device),
            "models": {
                name: {
                    "path": str(path),
                    "memory_encoder": actor.memory_encoder,
                    "state_shape": list(actor.state_shape),
                }
                for name, (path, actor) in models.items()
            },
            "seeds": seeds,
            "sensor_seed": SENSOR_SEED,
            "commands": [asdict(command) for command in commands],
            "conditions": [asdict(condition) for condition in CONDITIONS],
            "horizon_sec": HORIZON_SEC,
            "step_sec": env.time_interval,
            "success_thresholds": {
                "heading_deg": heading.SUCCESS_HEADING_DEG,
                "altitude_m": heading.SUCCESS_ALTITUDE_M,
                "speed_mps": heading.SUCCESS_SPEED_MPS,
            },
        }, output_dir / "config.json")
        print(f"输出目录：{output_dir}", flush=True)
        for name, (_, actor) in models.items():
            baseline = {}
            for condition in CONDITIONS:
                print(f"{name} / {condition.name}：{len(trials)} 局", flush=True)
                rows = [
                    {
                        "model": name,
                        "condition": condition.name,
                        **run_trial(env, actor, condition, idx, seed, command, horizon_steps, device),
                    }
                    for idx, (seed, command) in enumerate(trials, start=1)
                ]
                save_csv(rows, output_dir / name / condition.name / "episodes.csv")
                summary = heading.summarize_rows(rows)
                for metric in ("episode_reward", "memory_reset_count", "actual_dropout_rate"):
                    summary[f"{metric}_mean"] = float(np.mean([row[metric] for row in rows]))
                if condition.name == "E0":
                    baseline = summary.copy()
                for metric in (
                    "success_rate", "violation_episode_rate", "episode_reward_mean",
                    "heading_mae_deg_mean", "altitude_mae_m_mean", "speed_mae_mps_mean",
                ):
                    summary[f"{metric}_change_from_e0"] = summary[metric] - baseline[metric]
                summaries.append({"model": name, "condition": condition.name, **summary})
                save_csv(summaries, output_dir / "summary.csv")
    finally:
        env.close()

    if not args.no_plots:
        plot_summaries(summaries, output_dir)
        plot_mean_reward_comparison(summaries, output_dir)
        for name, (_, actor) in models.items():
            plot_reward_curves(output_dir / name, actor.memory_encoder)
    print(f"完成：{output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
