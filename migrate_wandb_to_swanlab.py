"""将本地 W&B 训练记录迁移到 SwanLab。"""

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path
from time import sleep

from project_env import REPO_ROOT, load_dotenv


EVAL_REWARD = "eval_average_episode_rewards"
EPISODE_REWARD_PREFIX = "eval_episode_rewards/episode_"


class CloudVerificationError(RuntimeError):
    """同步后的云端核对未完成。"""


@dataclass
class RunHistory:
    run_id: str
    name: str
    project: str
    config: dict
    metrics_by_step: dict
    skipped_metrics: set


def find_log(path):
    """接受日志文件、单次运行目录，或只含一次运行的 wandb 目录。"""
    if path.is_file() and path.suffix.lower() == ".wandb":
        return path
    if path.is_dir():
        logs = sorted(path.glob("*.wandb"))
        if not logs:
            logs = sorted([*path.glob("offline-run-*/*.wandb"), *path.glob("run-*/*.wandb")])
        if len(logs) == 1:
            return logs[0]
        if logs:
            raise ValueError("目录中有多个 W&B 日志，请指定其中一个运行目录或 .wandb 文件。")
    raise ValueError(f"找不到 .wandb 日志文件：{path}")


def item_key(item):
    return item.key or "/".join(item.nested_key)


def update_config(config, record):
    for item in record.update:
        key = item_key(item)
        if key and not key.startswith("_") and key != "user_name":
            config[key] = json.loads(item.value_json)
    for item in record.remove:
        config.pop(item_key(item), None)


def is_scalar(value):
    return type(value) in (int, float) and math.isfinite(value)


def convert_metrics(values, step, skipped):
    metrics = {}
    for key, value in values.items():
        if is_scalar(value):
            metrics[key] = value
        elif key == EVAL_REWARD and isinstance(value, list) and value:
            # 旧版有一维回报数组和形状为 n×1 的数组。
            rewards = [item[0] if isinstance(item, list) and len(item) == 1 else item
                       for item in value]
            if not all(is_scalar(reward) for reward in rewards):
                raise ValueError(f"环境步 {step} 的评估回报数组包含无效数值。")
            metrics[key] = math.fsum(rewards) / len(rewards)
            for index, reward in enumerate(rewards, 1):
                metrics[f"{EPISODE_REWARD_PREFIX}{index:02d}"] = reward
        else:
            skipped.add(key)
    return metrics


def read_history(path):
    from swanlab.converter.wb.wbdatastore import WBDataStoreReader
    from wandb.proto.wandb_internal_pb2 import Record

    run_id = name = project = started_at = None
    config = {}
    raw_history = {}
    with WBDataStoreReader() as reader:
        reader.open(path)
        while (payload := reader.scan()) is not None:
            record = Record.FromString(payload)
            if record.HasField("run"):
                run = record.run
                run_id = run.run_id or run_id
                name = run.display_name or name
                project = run.project or project
                if run.HasField("start_time"):
                    started_at = run.start_time.ToDatetime(tzinfo=timezone.utc).isoformat()
                update_config(config, run.config)
            elif record.HasField("config"):
                update_config(config, record.config)
            elif record.HasField("history"):
                items = {item_key(item): item.value_json for item in record.history.item}
                if "_step" in items:
                    step = json.loads(items["_step"])
                elif record.history.HasField("step"):
                    step = record.history.step.num
                else:
                    raise ValueError("W&B 指标记录缺少步数，无法保留原始横轴。")
                if not is_scalar(step) or step < 0 or step != int(step):
                    raise ValueError(f"W&B 日志含有无效步数：{step!r}")
                values = {key: json.loads(value) for key, value in items.items()
                          if key and not key.startswith("_")}
                raw_history.setdefault(int(step), {}).update(values)

    if not run_id and path.stem.startswith("run-"):
        run_id = path.stem.removeprefix("run-")
    if not run_id:
        raise ValueError("日志中缺少 W&B 运行编号。")
    if isinstance(config.get("model_dir"), str):
        config.pop("model_dir")
    config["wandb_run_id"] = run_id
    if started_at:
        config["wandb_started_at"] = started_at
    skipped = set()
    metrics_by_step = {}
    for step, values in sorted(raw_history.items()):
        metrics = convert_metrics(values, step, skipped)
        if metrics:
            metrics_by_step[step] = metrics
    if not metrics_by_step:
        raise ValueError("日志中没有可上传的数值指标。")
    return RunHistory(run_id, name or run_id, project, config, metrics_by_step, skipped)


def stage_run(history, project, workspace):
    import swanlab

    settings = swanlab.Settings(
        interactive=False,
        terminal={"proxy_type": "none"},
        probe={"hardware": False, "runtime": False, "requirements": False,
               "git": False, "swanlab": False, "monitor": False},
    )
    with swanlab.init(
        project=project, workspace=workspace, public=False, name=history.name,
        id=history.run_id, mode="offline", log_dir=str(REPO_ROOT / "tmp" / "wandb-migration"),
        group=history.config.get("scenario_name"), job_type="training",
        config=history.config, tags=["wandb-import"], settings=settings,
    ) as run:
        for step, metrics in history.metrics_by_step.items():
            run.log(metrics, step=step)
        return run.dir


def upload_run(history, project, workspace, latest_steps):
    import swanlab
    from swanlab.exceptions import AuthenticationError
    from swanlab.sdk.internal.core_python import client
    from swanlab.sdk.internal.core_python.api.experiment import get_experiment_summary
    from swanlab.sdk.internal.core_python.api.project import get_or_create_project

    load_dotenv(prefix="SWANLAB_")
    api_key = os.getenv("SWANLAB_API_KEY")
    if not api_key:
        raise ValueError("请在项目根目录的 .env 中配置 SWANLAB_API_KEY。")
    run_dir = stage_run(history, project, workspace)
    swanlab.login(api_key=api_key, save=False)
    workspace = workspace or client.username()
    target = get_or_create_project(username=workspace, name=project, public=False)
    if target["visibility"] != "PRIVATE":
        raise ValueError(f"{workspace}/{project} 是公开项目，请用 --project 指定私有项目。")
    swanlab.sync(run_dir, settings=swanlab.Settings(
        interactive=False, project={"workspace": workspace, "name": project, "public": False},
        run={"id": history.run_id},
    ))

    try:
        # 同步结束后原会话可能失效，重新登录再回读实验。
        swanlab.login(api_key=api_key, relogin=True, save=False)
        for _ in range(3):
            info = client.get(f"/project/{workspace}/{project}/runs/{history.run_id}").data
            summary = get_experiment_summary(target["cuid"], info["cuid"], info["createdAt"])
            uploaded = {item["key"]: item["step"] for item in summary.get("scalar") or []}
            config = (info.get("profile") or {}).get("config") or info.get("config") or {}
            if isinstance(config, str):
                config = json.loads(config)
            if (info["state"] == "FINISHED" and config.get("wandb_run_id", {}).get("value") == history.run_id
                    and all(uploaded.get(key) == step for key, step in latest_steps.items())):
                print(f"上传已核对，共 {len(latest_steps)} 条曲线，运行编号 {history.run_id}。")
                return
            sleep(2)
    except (AuthenticationError, OSError, ValueError) as exc:
        raise CloudVerificationError(
            f"回读实验 {history.run_id} 时出错。转换日志位于 {run_dir}。\n{exc}"
        ) from exc
    raise CloudVerificationError(f"实验 {history.run_id} 的状态、配置或指标尚未核对一致。转换日志位于 {run_dir}。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="本地 W&B 运行目录或 .wandb 文件")
    parser.add_argument("--project", help="SwanLab 私有项目，默认沿用 W&B 项目名")
    parser.add_argument("--workspace", help="SwanLab 工作空间，默认当前账号")
    parser.add_argument("--dry-run", action="store_true", help="仅在本机解析并核对数据")
    args = parser.parse_args()
    try:
        raw_path = args.path or input("请输入本地 W&B 运行目录或 .wandb 文件路径：")
        path = raw_path.strip().strip('"').strip("'")
        if not path:
            raise ValueError("请输入 W&B 日志路径。")
        history = read_history(find_log(Path(path).expanduser()))
        project = args.project or history.project or history.config.get("env_name")
        if not project:
            raise ValueError("日志中缺少项目名，请用 --project 指定 SwanLab 项目。")
        latest_steps = {}
        for step, metrics in history.metrics_by_step.items():
            latest_steps.update(dict.fromkeys(metrics, step))
        print(f"{history.name}（{history.run_id}）将导入 {project}。")
        print(f"共 {len(history.metrics_by_step):,} 个日志步和 {len(latest_steps)} 条曲线，"
              f"最后一步为 {max(history.metrics_by_step):,}。")
        if history.skipped_metrics:
            print(f"跳过非标量或非有限值：{', '.join(sorted(history.skipped_metrics))}")
        if not args.dry_run:
            upload_run(history, project, args.workspace, latest_steps)
    except ModuleNotFoundError as exc:
        if exc.name == "wandb":
            parser.exit(1, '请先运行 python -m pip install wandb "swanlab>=0.10.4,<1.0"。\n')
        if exc.name == "swanlab":
            parser.exit(1, "请先在当前 Python 环境安装项目 requirements.txt 中的 SwanLab。\n")
        raise
    except CloudVerificationError as exc:
        parser.exit(2, f"同步已执行，云端核对未完成：{exc}\n")
    except (ValueError, OSError, RuntimeError, EOFError) as exc:
        parser.exit(1, f"迁移失败：{exc}\n")


if __name__ == "__main__":
    main()
