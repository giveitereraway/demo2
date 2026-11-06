import logging
import os
import re
from contextlib import nullcontext
from itertools import count
from pathlib import Path

from project_env import REPO_ROOT, load_dotenv


SWANLAB_MODES = ("online", "offline", "local", "disabled")
DEFAULT_TRAIN_SWANLAB_MODE = "offline"


def create_run_dir(args):
    """为每次运行独立分配目录，避免并行实验覆盖模型。"""
    root = REPO_ROOT / "scripts" / "results" / args.env_name / args.scenario_name / args.algorithm_name / args.experiment_name
    root.mkdir(parents=True, exist_ok=True)
    existing = [int(match[1]) for entry in root.iterdir() if (match := re.fullmatch(r"run([0-9]+)", entry.name))]
    for index in count(max(existing, default=0) + 1):
        run_dir = root / f"run{index}"
        try:
            run_dir.mkdir()
        except FileExistsError:
            continue
        logging.info("输出目录: %s", run_dir)
        return run_dir


def track_swanlab(args, run_dir: Path, *, job_type="training"):
    """从环境读取凭据，并将实验日志放入当前运行目录。"""
    load_dotenv(prefix="SWANLAB_")
    default_mode = os.getenv("SWANLAB_MODE", DEFAULT_TRAIN_SWANLAB_MODE) if job_type == "training" else "disabled"
    mode = args.swanlab_mode or default_mode
    if mode == "disabled":
        return nullcontext()
    if mode not in SWANLAB_MODES:
        raise ValueError(f"SwanLab 模式须为 {', '.join(SWANLAB_MODES)} 之一。")

    try:
        import swanlab
    except ModuleNotFoundError as exc:
        if exc.name != "swanlab":
            raise
        raise RuntimeError('请在训练环境安装 "swanlab>=0.10.4"，或指定 --swanlab-mode disabled。') from exc

    logdir = run_dir / "swanlog"
    logdir.mkdir(exist_ok=True)
    return swanlab.init(
        project=args.env_name,
        experiment_name=f"{args.experiment_name}_seed{args.seed}",
        group=args.scenario_name,
        job_type=job_type,
        config={**vars(args), "swanlab_mode": mode},
        logdir=str(logdir),
        mode=mode,
        public=False,
        settings=swanlab.Settings(interactive=False),
    )
