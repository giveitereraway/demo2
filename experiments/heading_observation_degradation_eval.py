#!/usr/bin/env python
"""用单条阶跃指令或固定指令序列评估 heading 策略的观测退化鲁棒性。

传入一个或多个 actor 权重文件，也可传入包含 actor_latest.pt 的单次训练目录。
MLP 和注意力权重需要同目录的 model_config.json。PPO/SPO 标签由模型参数指定，
两者推理都使用 PPOActor。
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import deque
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import cycle
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import evaluate_lowlevel_controller as heading
from experiments.partial_observability_gru_eval import save_csv, save_json
from envs.JSBSim.envs.singlecontrol_env import HEADING_MEASUREMENT_NOISE_STD, HeadingMeasurementNoiseEnv


MODEL_ENCODERS = {
    "ppo-gru": "gru",
    "ppo-attention": "temporal_attention",
    "ppo-mlp": "mlp",
    "spo-gru": "gru",
    "spo-attention": "temporal_attention",
    "spo-mlp": "mlp",
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


# 噪声标准差按高度、航向、速度的物理单位给出。丢帧和延迟作用于整个误差向量。
# (丢帧率、时延、高度、航向、速度误差)
CONDITIONS = (
    Condition("E0"),
    Condition("E1", Degradation(0.10, 1, 10.0, 1.0, 1.0)),
    Condition("E2", Degradation(0.20, 2, 25.0, 3.0, 3.0)),
    Condition("E3", Degradation(0.30, 3, 50.0, 5.0, 5.0)),
    Condition("E4", Degradation(0.40, 4, 75.0, 7.0, 7.0)),
    Condition("E5", Degradation(0.50, 5, 100.0, 9.0, 9.0)),
    Condition("E6", Degradation(0.60, 6, 125.0, 11.0, 11.0)),
)
MEASUREMENT_NOISE_CONDITIONS = (
    Condition("E0"),
    *(Condition(f"N{level}") for level in range(1, len(HEADING_MEASUREMENT_NOISE_STD))),
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
    heading.add_fixed_sequence_argument(parser)
    parser.add_argument("--measurement-noise", action="store_true",
                        help="改为评估干净观测和三个测量噪声档位，默认评估 E0–E6 误差通道退化")
    parser.add_argument("--quick", action="store_true", help="用 1 个种子评估，单指令模式评估 3 条指令")
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
    if encoder != "gru" and not path.with_name("model_config.json").is_file():
        raise FileNotFoundError(f"{encoder} 权重缺少同目录的 model_config.json：{path}")
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
    command: heading.Command | None,
    horizon_steps: int,
    device: torch.device,
) -> tuple[dict, list[dict]]:
    env.seed(seed)
    env.reset()
    schedule = heading.iter_command_stages(env, command, horizon_steps)
    stage = next(schedule)
    stages = [stage]
    obs = env.get_obs()[heading.AGENT_ID].astype(np.float32)
    degrader = None
    if condition.degradation is not None:
        # 同一试验跨模型和退化档位复用随机数流，并独立于上一回合长度。
        rng = np.random.default_rng([SENSOR_SEED, trial_id])
        degrader = TrackingErrorDegrader(condition.degradation, obs, rng)

    state = torch.zeros((1, *actor.state_shape), device=device)
    masks = torch.ones((1, 1), device=device)
    episode_reward = 0.0
    records = []
    controls = []
    done = False
    for step in range(horizon_steps):
        if step == stage.end_step:
            stage = next(schedule)
            stages.append(stage)
            # 切换后立即计算新目标误差，沿用当前回合的退化队列。
            obs = env.get_obs()[heading.AGENT_ID].astype(np.float32)
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

    row, stage_rows = heading.summarize_trial(
        trial_id, seed, stages, records, controls, done, env.time_interval
    )
    row["success"] = row["success"] and len(records) == horizon_steps
    row["episode_reward"] = episode_reward
    row["actual_dropout_rate"] = degrader.dropped_frames / len(records) if degrader else 0.0
    return row, stage_rows


def plot_mean_rewards(conditions: tuple[str, ...], rows_by_model: dict[str, list[dict]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    styles = (("o", "-"), ("s", "--"), ("^", "-."), ("D", ":"), ("P", "--"), ("X", "-."), ("v", ":"))
    with plt.rc_context():
        heading.configure_matplotlib_chinese_font()
        fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
        positions = range(len(conditions))
        series = []
        for (model, rows), (marker, linestyle) in zip(rows_by_model.items(), cycle(styles)):
            rewards = [row["episode_reward_mean"] for row in rows]
            label = model.upper().replace("-", "+", 1).replace("ATTENTION", "注意力")
            line, = ax.plot(positions, rewards, marker=marker, linestyle=linestyle, label=label)
            series.append((rewards, line.get_color()))
        ax.set_xticks(positions, conditions)
        ax.set_xlim(-0.3, len(conditions) - 0.3)
        ax.margins(y=0.12)
        ax.set_title("不同算法的平均奖励对比")
        ax.set_xlabel("观测退化条件")
        ax.set_ylabel("平均单局累计奖励")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncols=min(3, len(series)))

        # 同一档位的均值接近时，按屏幕上的字号留出间距，并用引线指回实际数值。
        fig.canvas.draw()
        bottom, top = ax.get_ylim()
        gap = (top - bottom) * (15 * fig.dpi / 72) / ax.get_window_extent().height
        for x in positions:
            points = sorted(((rewards[x], color) for rewards, color in series), key=lambda point: point[0])
            label_y = [max(reward, bottom + gap / 2) for reward, _ in points]
            for i in range(1, len(label_y)):
                label_y[i] = max(label_y[i], label_y[i - 1] + gap)
            label_y[-1] = min(label_y[-1], top - gap / 2)
            for i in range(len(label_y) - 2, -1, -1):
                label_y[i] = min(label_y[i], label_y[i + 1] - gap)
            for (reward, color), y in zip(points, label_y):
                ax.annotate(
                    f"{reward:.2f}", (x, reward), xytext=(x + 0.18, y),
                    color=color, va="center", fontsize=9,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 0.3},
                    arrowprops={"arrowstyle": "-", "color": color, "lw": 0.6},
                ).set_in_layout(False)
        fig.savefig(output_dir / "mean_reward_comparison.png", dpi=180)
        plt.close(fig)


def plot_summaries(summaries: list[dict], output_dir: Path, *, measurement_noise: bool = False) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = {
        "success_rate": "Success rate",
        "heading_mae_deg_mean": "Heading MAE (deg)",
        "altitude_mae_m_mean": "Altitude MAE (m)",
        "speed_mae_mps_mean": "Speed MAE (m/s)",
    }
    suite = MEASUREMENT_NOISE_CONDITIONS if measurement_noise else CONDITIONS
    conditions = tuple(condition.name for condition in suite)
    rows_by_key = {(row["model"], row["condition"]): row for row in summaries}
    rows_by_model = {
        model: [rows_by_key[model, condition] for condition in conditions]
        for model in dict.fromkeys(row["model"] for row in summaries)
    }
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), layout="constrained")
    for ax, (metric, label) in zip(axes.flat, metrics.items()):
        for model, rows in rows_by_model.items():
            ax.plot(conditions, [row[metric] for row in rows], marker="o", label=model)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[0, 0].set_ylim(-0.02, 1.02)
    axes[0, 0].legend()
    filename = "measurement_noise.png" if measurement_noise else "observation_degradation.png"
    fig.savefig(output_dir / filename, dpi=180)
    plt.close(fig)
    if not measurement_noise:
        plot_mean_rewards(conditions, rows_by_model, output_dir)


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
    commands = [None] if args.fixed_sequence else heading.build_commands(quick=args.quick)
    trials = [(seed, command) for seed in seeds for command in commands]
    conditions = MEASUREMENT_NOISE_CONDITIONS if args.measurement_noise else CONDITIONS
    output_dir = args.output_dir.resolve() / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    summaries = []
    with ExitStack() as stack:
        env = heading.SingleControlEnv(heading.SCENARIO_NAME)
        stack.callback(env.close)
        evaluation_envs = {"E0": env}
        if args.measurement_noise:
            for level, condition in enumerate(conditions[1:], start=1):
                noisy_env = HeadingMeasurementNoiseEnv(heading.SCENARIO_NAME, noise_level=level, sensor_seed=SENSOR_SEED)
                stack.callback(noisy_env.close)
                evaluation_envs[condition.name] = noisy_env
        horizon_steps = (
            heading.fixed_sequence_horizon_steps(env) if args.fixed_sequence
            else math.ceil(HORIZON_SEC / env.time_interval)
        )
        config = {
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
            "conditions": [asdict(condition) for condition in conditions],
            "horizon_sec": horizon_steps * env.time_interval if args.fixed_sequence else HORIZON_SEC,
            "step_sec": env.time_interval,
            "success_thresholds": {
                "heading_deg": heading.SUCCESS_HEADING_DEG,
                "altitude_m": heading.SUCCESS_ALTITUDE_M,
                "speed_mps": heading.SUCCESS_SPEED_MPS,
            },
        }
        if args.measurement_noise:
            config["measurement_noise_std_m_deg_mps"] = {
                condition.name: std for condition, std in zip(conditions, HEADING_MEASUREMENT_NOISE_STD)
            }
        if args.fixed_sequence:
            config["fixed_sequence"] = heading.fixed_sequence_config(horizon_steps, env.time_interval)
        else:
            config["commands"] = [asdict(command) for command in commands]
        save_json(config, output_dir / "config.json")
        print(f"输出目录：{output_dir}", flush=True)
        for name, (_, actor) in models.items():
            baseline = {}
            for condition in conditions:
                print(f"{name} / {condition.name}：{len(trials)} 局", flush=True)
                rows = []
                stage_rows = []
                labels = {"model": name, "condition": condition.name}
                for idx, (seed, command) in enumerate(trials, start=1):
                    row, trial_stages = run_trial(
                        evaluation_envs.get(condition.name, env), actor, condition, idx, seed, command, horizon_steps, device,
                    )
                    rows.append({**labels, **row})
                    stage_rows.extend({**labels, **stage} for stage in trial_stages)
                condition_dir = output_dir / name / condition.name
                save_csv(rows, condition_dir / "episodes.csv")
                if args.fixed_sequence:
                    save_csv(stage_rows, condition_dir / "stages.csv")
                summary = heading.summarize_rows(rows)
                for metric in ("episode_reward", "actual_dropout_rate"):
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
    if not args.no_plots:
        plot_summaries(summaries, output_dir, measurement_noise=args.measurement_noise)
    print(f"完成：{output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
