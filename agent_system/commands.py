from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from runner.experiment_tracking import DEFAULT_TRAIN_SWANLAB_MODE

from .settings import AgentSettings, DEFAULT_TACTICAL_ACTOR_PATH, REPO_ROOT
from .tactical_actions import parse_action_reference
from .tactical_policy import resolve_actor_checkpoint_path


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _path_text(value: str | Path | None) -> str:
    """清理用户复制来的路径，兼容引号、空格以及两种斜杠。"""
    if value is None:
        return ""
    text = str(value).strip().strip('"').strip("'")
    if os.name == "nt":
        text = text.replace("/", "\\")
    return text


def _path(value: str | Path | None, *, base: Path = REPO_ROOT) -> Path | None:
    text = _path_text(value)
    if text == "":
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _required_existing_path(label: str, value: str | Path | None) -> tuple[Path | None, list[str]]:
    path = _path(value)
    if path is None:
        return None, [f"{label} 路径不能为空。"]
    if not path.exists():
        return path, [f"{label} 文件不存在：{path}"]
    return path, []


def _required_existing_actor_checkpoint(label: str, value: str | Path | None) -> tuple[Path | None, list[str]]:
    text = _path_text(value)
    if text == "":
        return None, [f"{label} 路径不能为空。"]
    path = resolve_actor_checkpoint_path(text)
    if not path.exists():
        return path, [f"{label} 文件不存在：{path}"]
    return path, []


def _python() -> str:
    return AgentSettings.load().runtime_python


@dataclass
class CommandSpec:
    name: str
    command: list[str]
    cwd: Path = REPO_ROOT
    env: dict[str, str] = field(default_factory=dict)
    expected_output_dir: Path | None = None
    validation_errors: list[str] = field(default_factory=list)

    def preview(self) -> str:
        return subprocess.list2cmdline(self.command)

    def merged_env(self) -> dict[str, str]:
        merged = os.environ.copy()
        merged.update(self.env)
        return merged


@dataclass
class TrainConfig:
    profile: str = "hierarchy_no_weapon_selfplay"
    env_name: str = "SingleCombat"
    scenario_name: str = "1v1/NoWeapon/HierarchySelfplay"
    algorithm_name: str = "ppo"
    experiment_name: str = "agent_hierarchy_no_weapon"
    seed: int = 1
    cuda: bool = True
    swanlab_mode: str = DEFAULT_TRAIN_SWANLAB_MODE
    use_selfplay: bool = True
    selfplay_algorithm: str = "fsp"
    n_choose_opponents: int = 1
    use_eval: bool = True
    n_rollout_threads: int = 48
    n_eval_rollout_threads: int = 5
    eval_interval: int = 10
    eval_episodes: int = 5
    num_env_steps: str = "1e8"
    buffer_size: int = 3000
    num_mini_batch: int = 6
    lr: str = "3e-4"
    gamma: str = "0.99"
    ppo_epoch: int = 4
    entropy_coef: str = "1e-3"
    clip_param: str = "0.2"
    use_prior: bool = False
    model_dir: str = ""


TRAIN_PROFILES: dict[str, TrainConfig] = {
    "hierarchy_no_weapon_selfplay": TrainConfig(),
    "hierarchy_shoot_selfplay": TrainConfig(
        profile="hierarchy_shoot_selfplay",
        scenario_name="1v1/ShootMissile/HierarchySelfplay",
        experiment_name="agent_hierarchy_shoot",
        use_prior=True,
    ),
    "heading_control": TrainConfig(
        profile="heading_control",
        env_name="SingleControl",
        scenario_name="1/heading",
        experiment_name="agent_heading_control",
        use_selfplay=False,
        use_eval=True,
        n_rollout_threads=16,
        n_eval_rollout_threads=1,
        eval_episodes=3,
        num_mini_batch=4,
    ),
}


def training_profile(name: str) -> TrainConfig:
    return TRAIN_PROFILES.get(name, TRAIN_PROFILES["hierarchy_no_weapon_selfplay"])


def build_train_command(config: TrainConfig) -> CommandSpec:
    script = REPO_ROOT / "scripts" / "train" / "train_jsbsim.py"
    command = [
        _python(),
        str(script),
        "--env-name",
        config.env_name,
        "--algorithm-name",
        config.algorithm_name,
        "--scenario-name",
        config.scenario_name,
        "--experiment-name",
        config.experiment_name,
        "--seed",
        str(config.seed),
        "--n-training-threads",
        "1",
        "--n-rollout-threads",
        str(config.n_rollout_threads),
        "--log-interval",
        "1",
        "--save-interval",
        "1",
        "--num-mini-batch",
        str(config.num_mini_batch),
        "--buffer-size",
        str(config.buffer_size),
        "--num-env-steps",
        str(config.num_env_steps),
        "--lr",
        str(config.lr),
        "--gamma",
        str(config.gamma),
        "--ppo-epoch",
        str(config.ppo_epoch),
        "--clip-param",
        str(config.clip_param),
        "--max-grad-norm",
        "2",
        "--entropy-coef",
        str(config.entropy_coef),
        "--hidden-size",
        "128 128",
        "--act-hidden-size",
        "128 128",
        "--recurrent-hidden-size",
        "128",
        "--recurrent-hidden-layers",
        "1",
        "--data-chunk-length",
        "8",
        "--user-name",
        "sf",
        "--swanlab-mode",
        config.swanlab_mode,
    ]
    if config.cuda:
        command.append("--cuda")
    if config.use_eval:
        command.extend(
            [
                "--use-eval",
                "--n-eval-rollout-threads",
                str(config.n_eval_rollout_threads),
                "--eval-interval",
                str(config.eval_interval),
                "--eval-episodes",
                str(config.eval_episodes),
            ]
        )
    if config.use_selfplay:
        command.extend(
            [
                "--use-selfplay",
                "--selfplay-algorithm",
                config.selfplay_algorithm,
                "--n-choose-opponents",
                str(config.n_choose_opponents),
            ]
        )
    if config.use_prior:
        command.append("--use-prior")
    if config.model_dir:
        command.extend(["--model-dir", str(_path(config.model_dir))])

    expected_dir = REPO_ROOT / "scripts" / "results" / config.env_name / config.scenario_name / config.algorithm_name / config.experiment_name
    return CommandSpec("训练 Agent", command, REPO_ROOT, env={"PYTHONIOENCODING": "utf-8"},
                       expected_output_dir=expected_dir)


@dataclass
class Eval1v1Config:
    eval_scenario_name: str = "1v1/NoWeapon/Selfplay"
    actor_a_path: str = "envs/JSBSim/model/actor_latest.pt"
    actor_a_scenario_name: str = "1v1/NoWeapon/Selfplay"
    actor_b_path: str = "envs/JSBSim/model/actor_latest.pt"
    actor_b_scenario_name: str = "1v1/NoWeapon/Selfplay"
    lowlevel_actor_path: str = "envs/JSBSim/model/actor_heading.pt"
    experiment_name: str = "agent_1v1_eval"
    num_episodes: int = 50
    seed: int = 1
    device: str = "auto"
    save_acmi: bool = True
    acmi_episodes: str = "0"
    save_plots: bool = True
    output_dir: str = ""


def build_eval_1v1_command(config: Eval1v1Config) -> CommandSpec:
    script = REPO_ROOT / "experiments" / "1v1.py"
    actor_a_path, actor_a_errors = _required_existing_path("Actor A", config.actor_a_path)
    actor_b_path, actor_b_errors = _required_existing_path("Actor B", config.actor_b_path)
    lowlevel_actor_path, lowlevel_errors = _required_existing_path("低层控制器", config.lowlevel_actor_path)
    validation_errors = actor_a_errors + actor_b_errors + lowlevel_errors
    command = [
        _python(),
        str(script),
        "--eval-scenario-name",
        config.eval_scenario_name,
        "--actor-a-path",
        str(actor_a_path or ""),
        "--actor-a-scenario-name",
        config.actor_a_scenario_name,
        "--actor-b-path",
        str(actor_b_path or ""),
        "--actor-b-scenario-name",
        config.actor_b_scenario_name,
        "--lowlevel-actor-path",
        str(lowlevel_actor_path or ""),
        "--experiment-name",
        config.experiment_name,
        "--num-episodes",
        str(config.num_episodes),
        "--seed",
        str(config.seed),
        "--device",
        config.device,
        "--save-acmi",
        _bool_text(config.save_acmi),
        "--acmi-episodes",
        config.acmi_episodes,
        "--save-plots",
        _bool_text(config.save_plots),
    ]
    output_dir = _path(config.output_dir)
    if output_dir is not None:
        command.extend(["--output-dir", str(output_dir)])
    return CommandSpec(
        "1v1 评估 Agent",
        command,
        REPO_ROOT,
        expected_output_dir=output_dir,
        validation_errors=validation_errors,
    )


@dataclass
class HumanLoopConfig:
    mode: str = "free_fly"
    seed: int = 5
    cuda: bool = True


HUMAN_MODES: Mapping[str, tuple[str, str, str, bool]] = {
    "free_fly": ("scripts/human_combat/human_free_fly.py", "SingleControl", "1/HumanFreeFly", False),
    "no_weapon_1v1": ("scripts/human_combat/human_1v1.py", "SingleCombat", "1v1/NoWeapon/HumanSingleCombat", False),
    "shoot_1v1": ("scripts/human_combat/human_shoot_1v1.py", "SingleCombat", "1v1/ShootMissile/HumanWithMissile", True),
}


def build_human_loop_command(config: HumanLoopConfig) -> CommandSpec:
    script_rel, env_name, scenario_name, use_prior = HUMAN_MODES.get(config.mode, HUMAN_MODES["free_fly"])
    command = [
        _python(),
        str(REPO_ROOT / script_rel),
        "--env-name",
        env_name,
        "--algorithm-name",
        "ppo",
        "--scenario-name",
        scenario_name,
        "--experiment-name",
        "agent_human_loop",
        "--seed",
        str(config.seed),
        "--n-training-threads",
        "1",
        "--n-rollout-threads",
        "1",
        "--render-mode",
        "real_time",
    ]
    if config.cuda:
        command.append("--cuda")
    if use_prior:
        command.append("--use-prior")
    return CommandSpec("人机交互 Agent", command, REPO_ROOT)


@dataclass
class TacticalDemoConfig:
    actor_path: str = str(DEFAULT_TACTICAL_ACTOR_PATH)
    enemy_path: str = ""
    scenario_name: str = "1v1/NoWeapon/TacticalHierarchySelfplay"
    agent_id: str = "A0100"
    enemy_action: str = ""
    hold_steps: int = 10
    max_steps: int = 1000
    seed: int = 1
    device: str = "auto"
    render_mode: str = "txt"
    log_path: str = "output/agent_tactical_1v1/demo_log.jsonl"
    acmi_path: str = "output/agent_tactical_1v1/demo.txt.acmi"
    step_sleep: float = 0.2
    status_interval: int = 25
    verbose_steps: bool = False
    disable_llm: bool = False


def build_tactical_demo_command(config: TacticalDemoConfig) -> CommandSpec:
    script = REPO_ROOT / "scripts" / "agent" / "agent_tactical_1v1_demo.py"
    actor_path, validation_errors = _required_existing_actor_checkpoint("Tactical actor", config.actor_path)
    enemy_path_value = config.enemy_path or config.actor_path
    enemy_path = resolve_actor_checkpoint_path(enemy_path_value)
    if config.enemy_action:
        if parse_action_reference(config.enemy_action) is None:
            validation_errors.append(f"enemy-action 无法识别：{config.enemy_action}")
    else:
        enemy_path, enemy_errors = _required_existing_actor_checkpoint("Enemy tactical actor", enemy_path_value)
        validation_errors.extend(enemy_errors)
    log_path = _path(config.log_path)
    acmi_path = _path(config.acmi_path)
    command = [
        _python(),
        str(script),
        "--actor-path",
        str(actor_path or ""),
        "--enemy-path",
        str(enemy_path or ""),
        "--scenario-name",
        config.scenario_name,
        "--agent-id",
        config.agent_id,
        "--hold-steps",
        str(config.hold_steps),
        "--max-steps",
        str(config.max_steps),
        "--seed",
        str(config.seed),
        "--device",
        config.device,
        "--render-mode",
        config.render_mode,
        "--log-path",
        str(log_path or ""),
        "--acmi-path",
        str(acmi_path or ""),
        "--step-sleep",
        str(config.step_sleep),
        "--status-interval",
        str(config.status_interval),
    ]
    if config.enemy_action:
        command.extend(["--enemy-action", config.enemy_action])
    if config.verbose_steps:
        command.append("--verbose-steps")
    if config.disable_llm:
        command.append("--disable-llm")
    return CommandSpec(
        "1v1 LLM-Agent 战术调度演示",
        command,
        REPO_ROOT,
        expected_output_dir=log_path.parent if log_path is not None else None,
        validation_errors=validation_errors,
    )


@dataclass
class VisualizeConfig:
    model_dir: str = ""
    env_name: str = "SingleCombat"
    scenario_name: str = "1v1/NoWeapon/Selfplay"
    algorithm_name: str = "ppo"
    experiment_name: str = "agent_visualize"
    num_agents: int = 1
    episode_length: int = 1000
    seed: int = 1
    cuda: bool = False
    use_selfplay: bool = False
    render_index: str = "latest"
    render_opponent_index: str = "latest"


def build_visualize_command(config: VisualizeConfig) -> CommandSpec:
    script = REPO_ROOT / "scripts" / "render" / "render_jsbsim.py"
    model_dir = _path(config.model_dir)
    if model_dir is None:
        raise ValueError("可视化流程需要指定模型目录。")
    command = [
        _python(),
        str(script),
        "--model-dir",
        str(model_dir),
        "--env-name",
        config.env_name,
        "--scenario-name",
        config.scenario_name,
        "--algorithm-name",
        config.algorithm_name,
        "--experiment-name",
        config.experiment_name,
        "--num-agents",
        str(config.num_agents),
        "--episode-length",
        str(config.episode_length),
        "--seed",
        str(config.seed),
        "--render-index",
        config.render_index,
        "--render-opponent-index",
        config.render_opponent_index,
    ]
    if config.cuda:
        command.append("--cuda")
    if config.use_selfplay:
        command.append("--use-selfplay")
    return CommandSpec("可视化 Agent", command, REPO_ROOT)
