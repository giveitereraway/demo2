from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from .llm import LLMMessage, SiliconFlowClient
from .settings import REPO_ROOT


def infer_result_dir(output: str, fallback: Path | None = None) -> Path | None:
    """从脚本输出中识别结果目录。"""
    patterns = [
        r"输出目录:\s*(.+)",
        r'"output_dir"\s*:\s*"([^"]+)"',
    ]
    for pattern in patterns:
        match = re.search(pattern, output)
        if match:
            return Path(match.group(1).strip()).expanduser()
    return fallback


def load_result_brief(result_dir: Path) -> str:
    summary_path = result_dir / "summary.json"
    episodes_path = result_dir / "episodes.csv"
    parts = [f"结果目录: {result_dir}"]

    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        parts.append("summary.json:")
        parts.append(json.dumps(summary, ensure_ascii=False, indent=2))

    if episodes_path.exists():
        with episodes_path.open("r", newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        parts.append(f"episodes.csv 回合数: {len(rows)}")
        if rows:
            parts.append("前 3 局:")
            parts.append(json.dumps(rows[:3], ensure_ascii=False, indent=2))

    if result_dir.exists():
        training_summary = result_dir / "training-summary.json"
        if training_summary.is_file():
            parts.append("training-summary.json:")
            parts.append(training_summary.read_text(encoding="utf-8"))

        children = sorted(result_dir.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True)[:10]
        if children:
            parts.append("最近产物:")
            parts.extend(str(path) for path in children)

    if len(parts) == 1:
        parts.append("未找到 summary.json 或 episodes.csv。")
    return "\n".join(parts)


def fallback_analysis(result_dir: Path, brief: str) -> str:
    return (
        "# LLM 实验分析\n\n"
        "当前未配置硅基流动 API Key，已生成基于结果文件的占位分析。\n\n"
        "## 已读取信息\n\n"
        f"{brief}\n\n"
        "## 解读建议\n\n"
        "- 重点查看胜率、平均奖励、坠毁率、被击落率和平均距离等指标。\n"
        "- 如果 reward_margin 波动很大，应结合 Tacview 轨迹判断是否存在初始阵营偏置或策略不稳定。\n"
        "- 如果导弹场景命中率为 0，需要检查攻击距离、攻击角和发射间隔是否过严。\n"
    )


def analyze_result(
    result_dir: Path,
    *,
    task_name: str,
    client: SiliconFlowClient | None = None,
) -> Path:
    brief = load_result_brief(result_dir)
    if client is None:
        content = fallback_analysis(result_dir, brief)
    else:
        prompt = (
            "你是飞行器智能自主决策实验分析 Agent。请用中文分析以下训练或评估结果，"
            "围绕自主决策能力、分层强化学习表现、空战对抗指标、Tacview 复盘建议组织。"
            "不要夸大未验证结论。\n\n"
            f"任务类型: {task_name}\n\n{brief}"
        )
        try:
            content = client.chat(
                [
                    LLMMessage("system", "你负责给硕士毕设实验结果写严谨、可复述的中文分析。"),
                    LLMMessage("user", prompt),
                ],
                temperature=0.2,
                max_tokens=1800,
            )
        except Exception as exc:
            content = fallback_analysis(result_dir, brief) + f"\n\nLLM 调用失败: {exc}\n"

    analysis_path = result_dir / "analysis.md"
    analysis_path.write_text(content, encoding="utf-8")
    return analysis_path


def default_training_result_root() -> Path:
    return REPO_ROOT / "scripts" / "results"
