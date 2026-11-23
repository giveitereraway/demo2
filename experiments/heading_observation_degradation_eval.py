#!/usr/bin/env python
"""用训练式随机指令序列或固定指令序列评估 heading 策略的观测退化鲁棒性。

默认每个模型和档位使用 100 个种子，每个种子评估一局。
随机序列每 30 秒换一次目标，最多飞行 6 段共 180 秒。
使用 --fixed-sequence-suite 时，每个种子在四条固定轨迹上各评估一局。
每条固定轨迹飞行 4 段，每段 30 秒。每种序列都输出按退化档位分面的逐局奖励图。

传入一个或多个 actor 权重文件，也可传入包含 actor_latest.pt 的单次训练目录。
MLP 和注意力权重需要同目录的 model_config.json。PPO/SPO 标签由模型参数指定，
两者推理都使用 PPOActor。
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
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
DEFAULT_NUM_SEEDS = 100
TRAINING_STAGE_SEC = 30.0
TRAINING_STAGE_COUNT = 6
CONDITIONS = tuple(condition for condition in heading.OBSERVATION_DEGRADATION_CONDITIONS if condition.name != "E6")
MEASUREMENT_NOISE_CONDITIONS = (
    heading.ObservationCondition("E0"),
    *(heading.ObservationCondition(f"N{level}") for level in range(1, len(HEADING_MEASUREMENT_NOISE_STD))),
)
INITIAL_TARGET = heading.Command(0.0, 0.0, 0.0)
U_TURN_TARGET = heading.Command(100.0, 180.0, -10.0)
# 每条序列都覆盖航向、高度和速度变化，所有偏移相对本局初始状态。
FIXED_SEQUENCE_SUITE: dict[str, heading.FixedSequence] = {
    "original": heading.FIXED_SEQUENCE,
    "u_turn": (
        ("level_hold", INITIAL_TARGET),
        ("turn_to_cross_heading", heading.Command(100.0, 90.0, -10.0)),
        ("turn_to_reverse_heading", U_TURN_TARGET),
        ("hold_reverse_heading", U_TURN_TARGET),
    ),
    "step_climb_descent": (
        ("level_hold", INITIAL_TARGET),
        ("first_climb", heading.Command(100.0, 10.0, 8.0)),
        ("second_climb", heading.Command(200.0, 20.0, -8.0)),
        ("descend_to_initial_altitude", INITIAL_TARGET),
    ),
    "speed_change": (
        ("level_hold", INITIAL_TARGET),
        ("accelerate", heading.Command(50.0, -10.0, 30.0)),
        ("decelerate", heading.Command(-50.0, -20.0, -30.0)),
        ("recover_initial_targets", INITIAL_TARGET),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in MODEL_ENCODERS:
        parser.add_argument(f"--{name}", type=Path, help="权重文件或单次训练目录")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT, help="结果根目录")
    parser.add_argument(
        "--num-seeds", type=int, default=DEFAULT_NUM_SEEDS,
        help="每个模型、档位和序列的评估局数（默认 %(default)s）",
    )
    parser.add_argument("--device", default="auto", help="auto、cpu 或 cuda:0 等")
    parser.add_argument("--workers", type=int, default=1, help="并行评估进程数（仅支持 --device cpu）")
    parser.add_argument(
        "--no-timestamp-dir", action="store_true",
        help="直接输出到 --output-dir，不再额外嵌套 run_时间戳 子目录",
    )
    sequence_mode = parser.add_mutually_exclusive_group()
    heading.add_fixed_sequence_argument(sequence_mode)
    sequence_mode.add_argument(
        "--fixed-sequence-suite", action="store_true",
        help="评估原综合序列、调头飞行、阶梯爬降和加减速，输出每条序列及四条序列的汇总。",
    )
    parser.add_argument("--measurement-noise", action="store_true",
                        help="改为评估干净观测和三个测量噪声档位，默认评估 E0–E5 误差通道退化")
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
    if args.num_seeds < 1:
        parser.error("--num-seeds 必须是正整数")
    if args.workers < 1:
        parser.error("--workers 必须大于等于 1")
    if args.workers > 1 and args.device != "cpu":
        parser.error("多进程并行评估（--workers > 1）仅支持 --device cpu")
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
    fixed_sequence: heading.FixedSequence = heading.FIXED_SEQUENCE,
) -> tuple[dict, list[dict]]:
    env.seed(seed)
    env.reset()
    schedule = (
        iter_training_stages(env, horizon_steps) if training_sequence
        else heading.iter_command_stages(env, command, horizon_steps, fixed_sequence=fixed_sequence)
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
        action, state = heading.predict_action(actor, policy_obs[None, :], state, masks)
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
        expected_stage_count=TRAINING_STAGE_COUNT if training_sequence else len(fixed_sequence),
    )
    row["success"] = row["success"] and len(records) == horizon_steps
    if heading_check_failed:
        row["termination_reason"] = stage_rows[-1]["termination_reason"] = "unreach_heading"
    row["episode_reward"] = episode_reward
    row["actual_dropout_rate"] = degrader.dropped_frames / len(records) if degrader else 0.0
    if gate_values:
        row["attention_gate_mean"] = float(np.mean(gate_values))
    return row, stage_rows


def reward_plot_label(model: str) -> str:
    return model.upper().replace("-", "+", 1).replace("-ATTENTION", "+注意力").replace("ATTENTION", "注意力")


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
            line, = ax.plot(positions, rewards, marker=marker, linestyle=linestyle, label=reward_plot_label(model))
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


def plot_summaries(
    summaries: list[dict], output_dir: Path, *, measurement_noise: bool = False, across_sequences: bool = False,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    metrics = {
        "success_rate": "Success rate",
        "heading_mae_deg_mean": "Heading MAE (deg)",
        "altitude_mae_m_mean": "Altitude MAE (m)",
        "speed_mae_mps_mean": "Speed MAE (m/s)",
    }
    if across_sequences:
        metrics = {"episode_reward_mean": "Mean episode reward", **metrics}
    suite = MEASUREMENT_NOISE_CONDITIONS if measurement_noise else CONDITIONS
    conditions = tuple(condition.name for condition in suite)
    rows_by_key = {(row["model"], row["condition"]): row for row in summaries}
    rows_by_model = {
        model: [rows_by_key[model, condition] for condition in conditions]
        for model in dict.fromkeys(row["model"] for row in summaries)
    }
    ncols = 3 if across_sequences else 2
    fig, axes = plt.subplots(2, ncols, figsize=(4.5 * ncols, 6), layout="constrained")
    for ax, (metric, label) in zip(axes.flat, metrics.items()):
        for model, rows in rows_by_model.items():
            ax.plot(conditions, [row[metric] for row in rows], marker="o", label=model)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    if across_sequences:
        fig.suptitle("Average across fixed sequences")
        fig.supxlabel("Observation condition")
        axes[0, 1].set_ylim(-0.02, 1.02)
        axes[0, 1].yaxis.set_major_formatter(PercentFormatter(xmax=1))
        axes[1, 2].set_axis_off()
        axes[1, 2].legend(*axes[0, 0].get_legend_handles_labels(), loc="center")
        filename = "mean_metrics_across_sequences.png"
    else:
        axes[0, 0].set_ylim(-0.02, 1.02)
        axes[0, 0].legend()
        filename = "measurement_noise.png" if measurement_noise else "observation_degradation.png"
    fig.savefig(output_dir / filename, dpi=180)
    plt.close(fig)
    if not measurement_noise and not across_sequences:
        plot_mean_rewards(conditions, rows_by_model, output_dir)


def plot_episode_rewards(
    episodes: dict[tuple[str, str], list[dict]],
    conditions: tuple[heading.ObservationCondition, ...],
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    models = tuple(dict.fromkeys(model for model, _ in episodes))
    trial_count = len(next(iter(episodes.values())))
    ncols = 3 if len(conditions) >= 6 else min(2, len(conditions))
    nrows = math.ceil(len(conditions) / ncols)
    with plt.rc_context():
        heading.configure_matplotlib_chinese_font()
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(4.3 * ncols, 2.9 * nrows),
            sharex=True, sharey=True, squeeze=False, layout="constrained",
        )
        for ax, condition in zip(axes.flat, conditions):
            for model in models:
                rows = episodes[model, condition.name]
                ax.plot(
                    [row["trial_id"] for row in rows], [row["episode_reward"] for row in rows],
                    label=reward_plot_label(model), marker=".", markersize=3, linewidth=1,
                )
            ax.set_title(condition.name)
            ax.grid(alpha=0.25)
        for ax in axes.flat[len(conditions):]:
            ax.set_visible(False)
        axes[0, 0].set_xlim(0.5, trial_count + 0.5)
        if trial_count == 1:
            axes[0, 0].set_xticks([1])
        else:
            axes[0, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
        fig.supxlabel("评估局号")
        fig.supylabel("单局累计奖励")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncols=min(4, len(models)), frameon=False)
        fig.savefig(output_dir / "episode_rewards_by_condition.png", dpi=180)
        plt.close(fig)


def summarize_condition(rows: list[dict], baseline: dict | None) -> dict:
    summary = heading.summarize_rows(rows)
    for metric in ("episode_reward", "actual_dropout_rate"):
        summary[f"{metric}_mean"] = float(np.mean([row[metric] for row in rows]))
    if rows and "attention_gate_mean" in rows[0]:
        summary["attention_gate_mean"] = float(np.mean([row["attention_gate_mean"] for row in rows]))
    if baseline is None:
        baseline = summary
    for metric in (
        "success_rate", "violation_episode_rate", "episode_reward_mean",
        "heading_mae_deg_mean", "altitude_mae_m_mean", "speed_mae_mps_mean",
    ):
        summary[f"{metric}_change_from_e0"] = summary[metric] - baseline[metric]
    return summary


def evaluate_sequence(
    models: dict[str, tuple[Path, heading.PPOActor]],
    conditions: tuple[heading.ObservationCondition, ...],
    evaluation_envs: dict[str, heading.SingleControlEnv],
    seeds: list[int],
    horizon_steps: int,
    device: torch.device,
    output_dir: Path,
    *,
    sequence: heading.FixedSequence | None,
) -> tuple[list[dict], dict[tuple[str, str], list[dict]]]:
    summaries = []
    episodes = {}
    for name, (_, actor) in models.items():
        baseline = None
        for condition in conditions:
            print(f"{name} / {condition.name}：{len(seeds)} 局", flush=True)
            rows = []
            stage_rows = []
            labels = {"model": name, "condition": condition.name}
            for idx, seed in enumerate(seeds, start=1):
                row, trial_stages = run_trial(
                    evaluation_envs[condition.name], actor, condition, idx, seed, None, horizon_steps, device,
                    training_sequence=sequence is None,
                    fixed_sequence=sequence if sequence is not None else heading.FIXED_SEQUENCE,
                )
                rows.append({**labels, **row})
                stage_rows.extend({**labels, **stage} for stage in trial_stages)
            condition_dir = output_dir / name / condition.name
            save_csv(rows, condition_dir / "episodes.csv")
            save_csv(stage_rows, condition_dir / "stages.csv")
            episodes[name, condition.name] = rows
            summary = summarize_condition(rows, baseline)
            if condition.name == "E0":
                baseline = summary
            summaries.append({**labels, **summary})
            save_csv(summaries, output_dir / "summary.csv")
    return summaries, episodes


@dataclass(frozen=True)
class ConditionEvalJob:
    """单进程或多进程执行的单个 (序列, 模型, 退化档位) 评估任务。"""

    sequence_name: str
    model_name: str
    model_path: Path
    condition: heading.ObservationCondition
    noise_level: int | None
    seeds: tuple[int, ...]
    horizon_steps: int
    sequence: heading.FixedSequence | None
    sequence_dir: Path


def _run_condition_job(job: ConditionEvalJob) -> tuple[str, str, str, list[dict]]:
    """在独立工作进程内完成一个 (序列, 模型, 退化档位) 的全部种子评估。"""
    torch.set_num_threads(1)
    device = torch.device("cpu")
    actor = heading.build_actor(job.model_path, device)
    if job.noise_level is not None and job.noise_level > 0:
        env: heading.SingleControlEnv = HeadingMeasurementNoiseEnv(
            heading.SCENARIO_NAME, noise_level=job.noise_level, sensor_seed=SENSOR_SEED
        )
    else:
        env = heading.SingleControlEnv(heading.SCENARIO_NAME)

    tag = f"{job.sequence_name} / " if job.sequence_name else ""
    print(f"[开始] {tag}{job.model_name} / {job.condition.name}：{len(job.seeds)} 局", flush=True)
    rows: list[dict] = []
    stage_rows: list[dict] = []
    labels = {"model": job.model_name, "condition": job.condition.name}
    try:
        for idx, seed in enumerate(job.seeds, start=1):
            row, trial_stages = run_trial(
                env, actor, job.condition, idx, seed, None, job.horizon_steps, device,
                training_sequence=job.sequence is None,
                fixed_sequence=job.sequence if job.sequence is not None else heading.FIXED_SEQUENCE,
            )
            rows.append({**labels, **row})
            stage_rows.extend({**labels, **stage} for stage in trial_stages)
    finally:
        env.close()

    condition_dir = job.sequence_dir / job.model_name / job.condition.name
    save_csv(rows, condition_dir / "episodes.csv")
    save_csv(stage_rows, condition_dir / "stages.csv")
    mean_reward = float(np.mean([r["episode_reward"] for r in rows])) if rows else 0.0
    succ_rate = float(np.mean([r["success"] for r in rows])) if rows else 0.0
    print(
        f"[完成] {tag}{job.model_name} / {job.condition.name}："
        f"成功率 {succ_rate:.2%}，均值奖励 {mean_reward:.2f}",
        flush=True,
    )
    return job.sequence_name, job.model_name, job.condition.name, rows


def _expected_algorithm(model_name: str) -> str:
    prefix = model_name.split("-", 1)[0]
    return prefix if prefix in heading.HEADING_BASELINES else "ppo"


def main() -> None:
    args = parse_args()
    device = torch.device(
        ("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    if device.type == "cpu":
        torch.set_num_threads(1)
    models = {
        name: load_actor(
            path, MODEL_ENCODERS[name], device, prefer_best=args.prefer_best,
            algorithm=_expected_algorithm(name),
        )
        for name, path in args.models.items()
    }
    num_seeds = 1 if args.quick else args.num_seeds
    seeds = list(range(1, num_seeds + 1))
    conditions = MEASUREMENT_NOISE_CONDITIONS if args.measurement_noise else CONDITIONS
    output_dir = (
        args.output_dir.resolve()
        if args.no_timestamp_dir
        else args.output_dir.resolve() / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    )
    sequences = FIXED_SEQUENCE_SUITE if args.fixed_sequence_suite else {
        "": heading.FIXED_SEQUENCE if args.fixed_sequence else None,
    }
    pooled_episodes: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with ExitStack() as stack:
        env = heading.SingleControlEnv(heading.SCENARIO_NAME)
        stack.callback(env.close)
        evaluation_envs = {condition.name: env for condition in conditions}
        if args.measurement_noise and args.workers == 1:
            for level, condition in enumerate(conditions[1:], start=1):
                noisy_env = HeadingMeasurementNoiseEnv(heading.SCENARIO_NAME, noise_level=level, sensor_seed=SENSOR_SEED)
                stack.callback(noisy_env.close)
                evaluation_envs[condition.name] = noisy_env
        stage_steps = math.ceil(TRAINING_STAGE_SEC / env.time_interval)
        horizons = {
            name: heading.fixed_sequence_horizon_steps(env, sequence) if sequence is not None
            else TRAINING_STAGE_COUNT * stage_steps
            for name, sequence in sequences.items()
        }
        horizon_steps = max(horizons.values())
        if horizon_steps > env.config.max_steps:
            raise ValueError(f"场景最多允许 {env.config.max_steps} 步，训练式序列需要 {horizon_steps} 步")
        config = {
            "scenario": heading.SCENARIO_NAME,
            "device": str(device),
            "workers": args.workers,
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
        if args.fixed_sequence_suite:
            config["fixed_sequence_suite"] = {
                "aggregation": "pooled_episodes_per_model_and_condition",
                "sequences": {
                    name: heading.fixed_sequence_config(horizons[name], env.time_interval, sequence)
                    for name, sequence in FIXED_SEQUENCE_SUITE.items()
                },
            }
        elif args.fixed_sequence:
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

        if args.workers == 1:
            for sequence_name, sequence in sequences.items():
                sequence_dir = output_dir
                if args.fixed_sequence_suite:
                    sequence_dir = output_dir / "sequences" / sequence_name
                    print(f"固定序列：{sequence_name}", flush=True)
                summaries, episodes = evaluate_sequence(
                    models, conditions, evaluation_envs, seeds, horizons[sequence_name], device, sequence_dir,
                    sequence=sequence,
                )
                if not args.no_plots:
                    plot_summaries(summaries, sequence_dir, measurement_noise=args.measurement_noise)
                    plot_episode_rewards(episodes, conditions, sequence_dir)
                if args.fixed_sequence_suite:
                    for key, rows in episodes.items():
                        pooled_episodes[key].extend(rows)
        else:
            # 多进程路径先关闭主进程占位环境，再跨序列统一调度任务减少尾部空转。
            stack.close()
            jobs: list[ConditionEvalJob] = []
            for sequence_name, sequence in sequences.items():
                sequence_dir = (output_dir / "sequences" / sequence_name) if args.fixed_sequence_suite else output_dir
                for name, (model_path, _) in models.items():
                    for cond_idx, condition in enumerate(conditions):
                        noise_level = cond_idx if args.measurement_noise else None
                        jobs.append(ConditionEvalJob(
                            sequence_name=sequence_name,
                            model_name=name,
                            model_path=model_path,
                            condition=condition,
                            noise_level=noise_level,
                            seeds=tuple(seeds),
                            horizon_steps=horizons[sequence_name],
                            sequence=sequence,
                            sequence_dir=sequence_dir,
                        ))
            # 将较慢的注意力模型任务排在前面优先调度，缓解长尾等待。
            scheduled_jobs = sorted(
                jobs,
                key=lambda j: (0 if "attention" in j.model_name else 1),
            )
            results_map: dict[tuple[str, str, str], list[dict]] = {}
            with ProcessPoolExecutor(max_workers=min(args.workers, len(scheduled_jobs))) as pool:
                for seq_name, model_name, cond_name, rows in pool.map(_run_condition_job, scheduled_jobs):
                    results_map[seq_name, model_name, cond_name] = rows

            for sequence_name in sequences:
                sequence_dir = (output_dir / "sequences" / sequence_name) if args.fixed_sequence_suite else output_dir
                summaries = []
                episodes: dict[tuple[str, str], list[dict]] = {}
                for name in models:
                    baseline = None
                    for condition in conditions:
                        rows = results_map[sequence_name, name, condition.name]
                        episodes[name, condition.name] = rows
                        summary = summarize_condition(rows, baseline)
                        if condition.name == "E0":
                            baseline = summary
                        summaries.append({"model": name, "condition": condition.name, **summary})
                save_csv(summaries, sequence_dir / "summary.csv")
                if not args.no_plots:
                    plot_summaries(summaries, sequence_dir, measurement_noise=args.measurement_noise)
                    plot_episode_rewards(episodes, conditions, sequence_dir)
                if args.fixed_sequence_suite:
                    for key, rows in episodes.items():
                        pooled_episodes[key].extend(rows)
    if args.fixed_sequence_suite:
        # 每条序列使用相同数量的种子，合并单局样本后，均值和成功率对各序列等权。
        summaries = []
        baselines = {}
        for (name, condition), rows in pooled_episodes.items():
            summary = summarize_condition(rows, baselines.get(name))
            if condition == "E0":
                baselines[name] = summary
            summaries.append({"model": name, "condition": condition, **summary})
        save_csv(summaries, output_dir / "summary.csv")
        if not args.no_plots:
            plot_summaries(summaries, output_dir, measurement_noise=args.measurement_noise, across_sequences=True)
    print(f"完成：{output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
