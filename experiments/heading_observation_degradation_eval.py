#!/usr/bin/env python
"""用训练式随机指令序列或固定指令序列评估 heading 策略的观测退化鲁棒性。

默认每个种子评估一局，每 30 秒换一次目标，最多飞行 6 段共 180 秒。

传入一个或多个 actor 权重文件，也可传入包含 actor_latest.pt 的单次训练目录。
MLP 和注意力权重需要同目录的 model_config.json。PPO/SPO 标签由模型参数指定，
两者推理都使用 PPOActor。
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import asdict
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
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.envs.singlecontrol_env import (
    HEADING_MEASUREMENT_NOISE_STD,
    HeadingErrorChannelDegrader,
    HeadingMeasurementNoiseEnv,
)
from envs.JSBSim.termination_conditions import UnreachHeading


MODEL_ENCODERS = heading.MODEL_ENCODERS
load_actor = heading.load_named_actor
OUTPUT_ROOT = REPO_ROOT / "experiments" / "results" / "chapter_3" /"observation_degradation" / "heading_observation_degradation"
SENSOR_SEED = heading.SENSOR_SEED
TRAINING_STAGE_SEC = 30.0
TRAINING_STAGE_COUNT = 6
CONDITIONS = heading.OBSERVATION_DEGRADATION_CONDITIONS
MEASUREMENT_NOISE_CONDITIONS = (
    heading.ObservationCondition("E0"),
    *(heading.ObservationCondition(f"N{level}") for level in range(1, len(HEADING_MEASUREMENT_NOISE_STD))),
)


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
    parser.add_argument("--prefer-best", action="store_true",
                        help="当传入训练目录且存在 actor_best.pt 时优先加载 actor_best.pt")
    parser.add_argument("--quick", action="store_true", help="每个模型和退化档位只用 1 个种子评估")
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


def iter_training_stages(env: heading.SingleControlEnv, horizon_steps: int) -> Iterator[heading.CommandStage]:
    stage_steps = horizon_steps // TRAINING_STAGE_COUNT
    condition = next(cond for cond in env.task.termination_conditions if isinstance(cond, UnreachHeading))
    # 评估按控制步换目标并记录上一段的末步误差，暂停环境内部自行换目标。
    env.agents[heading.AGENT_ID].set_property_value(c.heading_check_time, heading.FIXED_HEADING_CHECK_TIME_SEC)
    for idx in range(TRAINING_STAGE_COUNT):
        delta_heading, delta_altitude_ft, delta_speed = condition.advance_target(env, heading.AGENT_ID)
        command = heading.Command(delta_altitude_ft * heading.FT_TO_M, delta_heading, delta_speed)
        yield heading.CommandStage(
            f"training_target_{idx + 1}", command, heading.read_command_target(env),
            idx * stage_steps, (idx + 1) * stage_steps,
        )


@torch.inference_mode()
def run_trial(
    env: heading.SingleControlEnv,
    actor: heading.PPOActor,
    condition: heading.ObservationCondition,
    trial_id: int,
    seed: int,
    command: heading.Command | None,
    horizon_steps: int,
    device: torch.device,
    *,
    training_sequence: bool = False,
) -> tuple[dict, list[dict]]:
    env.seed(seed)
    env.reset()
    schedule = (
        iter_training_stages(env, horizon_steps) if training_sequence
        else heading.iter_command_stages(env, command, horizon_steps)
    )
    stage = next(schedule)
    stages = [stage]
    obs = env.get_obs()[heading.AGENT_ID].astype(np.float32)
    degrader = None
    if condition.degradation is not None:
        # 同一试验跨模型和退化档位复用随机数流，并独立于上一回合长度。
        rng = np.random.default_rng([SENSOR_SEED, trial_id])
        degrader = HeadingErrorChannelDegrader(condition.degradation, obs, rng)

    state = torch.zeros((1, *actor.state_shape), device=device)
    masks = torch.ones((1, 1), device=device)
    episode_reward = 0.0
    gate_values = []
    records = []
    controls = []
    done = False
    heading_check_failed = False
    for step in range(horizon_steps):
        if step == stage.end_step:
            stage = next(schedule)
            stages.append(stage)
            # 切换后立即计算新目标误差，沿用当前回合的退化队列。
            obs = env.get_obs()[heading.AGENT_ID].astype(np.float32)
        policy_obs = degrader.transform(obs) if degrader is not None else obs
        action, _, state = actor(policy_obs[None, :], state, masks, deterministic=True)
        if actor.memory_encoder in {"gru_attention", "igru"} and hasattr(actor.base, "last_gate_mean"):
            gate_values.append(float(actor.base.last_gate_mean))
        action = action.cpu().numpy().astype(np.int64)
        controls.append(env.task.normalize_action(env, heading.AGENT_ID, action[0]))
        next_obs, rewards, dones, _ = env.step(action)
        obs = next_obs[0].astype(np.float32)
        episode_reward += float(rewards.item())
        records.append(heading.get_state_record(env))
        done = bool(dones.item())
        if done:
            break
        if (
            training_sequence and step + 1 == stage.end_step
            and abs(records[-1]["heading_error_deg"]) > heading.SUCCESS_HEADING_DEG
        ):
            heading_check_failed = True
            done = step + 1 < horizon_steps
            break

    row, stage_rows = heading.summarize_trial(
        trial_id, seed, stages, records, controls, done, env.time_interval,
        expected_stage_count=TRAINING_STAGE_COUNT if training_sequence else len(heading.FIXED_SEQUENCE),
    )
    row["success"] = row["success"] and len(records) == horizon_steps
    if heading_check_failed:
        row["termination_reason"] = stage_rows[-1]["termination_reason"] = "unreach_heading"
    row["episode_reward"] = episode_reward
    row["actual_dropout_rate"] = degrader.dropped_frames / len(records) if degrader else 0.0
    if gate_values:
        row["attention_gate_mean"] = float(np.mean(gate_values))
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
            label = (
                model.upper()
                .replace("-", "+", 1)
                .replace("-ATTENTION", "+注意力")
                .replace("ATTENTION", "注意力")
            )
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
        name: load_actor(path, MODEL_ENCODERS[name], device, prefer_best=args.prefer_best)
        for name, path in args.models.items()
    }
    seeds = heading.SEEDS[:1 if args.quick else args.num_seeds]
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
        stage_steps = math.ceil(TRAINING_STAGE_SEC / env.time_interval)
        if args.fixed_sequence:
            horizon_steps = heading.fixed_sequence_horizon_steps(env)
        else:
            horizon_steps = TRAINING_STAGE_COUNT * stage_steps
            if horizon_steps > env.config.max_steps:
                raise ValueError(f"场景最多允许 {env.config.max_steps} 步，训练式序列需要 {horizon_steps} 步")
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
            "horizon_sec": horizon_steps * env.time_interval,
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
            config["training_sequence"] = {
                "target_reference": "previous_target",
                "stage_count": TRAINING_STAGE_COUNT,
                "stage_duration_sec": stage_steps * env.time_interval,
                "stage_steps": stage_steps,
                "terminate_on_heading_miss": True,
            }
        save_json(config, output_dir / "config.json")
        print(f"输出目录：{output_dir}", flush=True)
        for name, (_, actor) in models.items():
            baseline = {}
            for condition in conditions:
                print(f"{name} / {condition.name}：{len(seeds)} 局", flush=True)
                rows = []
                stage_rows = []
                labels = {"model": name, "condition": condition.name}
                for idx, seed in enumerate(seeds, start=1):
                    row, trial_stages = run_trial(
                        evaluation_envs.get(condition.name, env), actor, condition, idx, seed, None, horizon_steps, device,
                        training_sequence=not args.fixed_sequence,
                    )
                    rows.append({**labels, **row})
                    stage_rows.extend({**labels, **stage} for stage in trial_stages)
                condition_dir = output_dir / name / condition.name
                save_csv(rows, condition_dir / "episodes.csv")
                save_csv(stage_rows, condition_dir / "stages.csv")
                summary = heading.summarize_rows(rows)
                for metric in ("episode_reward", "actual_dropout_rate"):
                    summary[f"{metric}_mean"] = float(np.mean([row[metric] for row in rows]))
                if rows and "attention_gate_mean" in rows[0]:
                    summary["attention_gate_mean"] = float(np.mean([row["attention_gate_mean"] for row in rows]))
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
