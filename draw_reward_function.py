"""
绘制一维奖励函数曲线。

使用方法：
1. 在“手动修改区”修改横轴范围、奖励函数公式和图像标题。
2. 运行：python draw_reward_function.py
3. 如果 matplotlib 可用，会保存 PNG 并弹出窗口；如果 matplotlib/Pillow 环境有问题，会自动保存 SVG。
"""

from html import escape
from pathlib import Path

import numpy as np


# =========================
# 手动修改区：从这里开始改
# =========================

# 横轴变量 x 的范围和采样点数量。
# 例如：如果画 range_reward，x 可以表示距离 R，单位 km。
X_MIN = 0.0
X_MAX = 15.0
NUM_POINTS = 1500

# 图像显示配置。
TITLE = "Range Reward Function"
X_LABEL = "R / km"
Y_LABEL = "reward"
SAVE_PATH = None
FALLBACK_SVG_PATH = "reward_function.svg"
SHOW_FIG = True

# 如果奖励函数里有参数，可以写在这里。
target_dist = 3.0

# 在这里写奖励函数公式。
# 约定：lambda R 中的 R 是 numpy 数组，因此应使用 np.exp、np.clip、np.minimum、np.maximum 等函数。
# 注意：不要对数组使用 Python 内置的 min(...) / max(...)，应改用 np.minimum(...) / np.maximum(...)。
REWARD_FUNCTIONS = [
    (
        "range_reward_v0",
        lambda R: np.exp(-(R - target_dist) ** 2 * 0.004)
        / (1.0 + np.exp(-(R - target_dist + 2.0) * 2.0)),
    ),
    (
        "range_reward_v1",
        lambda R: np.clip(
            1.2
            * np.minimum(np.exp(-(R - target_dist) * 0.21), 1.0)
            / (1.0 + np.exp(-(R - target_dist + 1.0) * 0.8)),
            0.3,
            1.0,
        ),
    ),
    (
        "range_reward_v2",
        lambda R: np.maximum(
            np.clip(
                1.2
                * np.minimum(np.exp(-(R - target_dist) * 0.21), 1.0)
                / (1.0 + np.exp(-(R - target_dist + 1.0) * 0.8)),
                0.3,
                1.0,
            ),
            np.sign(7.0 - R),
        ),
    ),
    (
        "range_reward_v3",
            lambda R: 1 * (R < target_dist) + (R >= target_dist) * np.clip(
                -0.032 * R**2 + (0.064 * target_dist - 0.036) * R
                + 1.0 + 0.036 * target_dist - 0.032 * target_dist**2,0,1) + np.clip(np.exp(-0.16 * (R + 5.0 - target_dist)), 0, 0.2)
    ),
]

# =========================
# 手动修改区：到这里结束
# =========================


def _evaluate_reward(name, fn, x):
    """计算奖励函数，并把标量结果扩展成和 x 一样长的数组。"""
    try:
        y = np.asarray(fn(x), dtype=float)
    except Exception as exc:
        raise RuntimeError(f"奖励函数 {name} 计算失败，请检查公式是否支持 numpy 数组输入") from exc

    if y.ndim == 0:
        y = np.full_like(x, float(y))

    if y.shape != x.shape:
        raise ValueError(f"奖励函数 {name} 输出形状为 {y.shape}，但期望形状为 {x.shape}")

    if not np.all(np.isfinite(y)):
        raise ValueError(f"奖励函数 {name} 输出包含 nan 或 inf，请检查除零、log、arctanh 等运算")

    return y


def _build_curves(x):
    return [(name, _evaluate_reward(name, fn, x)) for name, fn in REWARD_FUNCTIONS]


def _save_svg_fallback(x, curves, output_path):
    """当 matplotlib 不可用时，直接生成一个简单 SVG 曲线图。"""
    width = 1000
    height = 620
    margin_left = 80
    margin_right = 30
    margin_top = 55
    margin_bottom = 75
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    finite_parts = [y[np.isfinite(y)] for _, y in curves]
    all_y = np.concatenate(finite_parts)
    if all_y.size == 0:
        raise ValueError("所有奖励函数输出都不是有限数，无法绘图")

    y_min = float(np.min(all_y))
    y_max = float(np.max(all_y))
    if np.isclose(y_min, y_max):
        y_min -= 1.0
        y_max += 1.0
    else:
        padding = 0.08 * (y_max - y_min)
        y_min -= padding
        y_max += padding

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]

    def sx(value):
        return margin_left + (value - x_min) / (x_max - x_min) * plot_w

    def sy(value):
        return margin_top + (y_max - value) / (y_max - y_min) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-size="22" font-family="Arial">{escape(TITLE)}</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="#111827"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#111827"/>',
    ]

    for i in range(6):
        tx = x_min + (x_max - x_min) * i / 5.0
        px = sx(tx)
        lines.append(f'<line x1="{px:.2f}" y1="{margin_top}" x2="{px:.2f}" y2="{margin_top + plot_h}" stroke="#e5e7eb"/>')
        lines.append(f'<text x="{px:.2f}" y="{height - 45}" text-anchor="middle" font-size="13" font-family="Arial">{tx:.2f}</text>')

        ty = y_min + (y_max - y_min) * i / 5.0
        py = sy(ty)
        lines.append(f'<line x1="{margin_left}" y1="{py:.2f}" x2="{margin_left + plot_w}" y2="{py:.2f}" stroke="#e5e7eb"/>')
        lines.append(f'<text x="{margin_left - 12}" y="{py + 4:.2f}" text-anchor="end" font-size="13" font-family="Arial">{ty:.3f}</text>')

    for index, (name, y) in enumerate(curves):
        color = colors[index % len(colors)]
        points = " ".join(f"{sx(float(xv)):.2f},{sy(float(yv)):.2f}" for xv, yv in zip(x, y))
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.4" points="{points}"/>')
        legend_y = margin_top + 24 + index * 24
        lines.append(f'<line x1="{width - 250}" y1="{legend_y}" x2="{width - 215}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{width - 205}" y="{legend_y + 5}" font-size="14" font-family="Arial">{escape(name)}</text>')

    lines.append(f'<text x="{width / 2}" y="{height - 12}" text-anchor="middle" font-size="15" font-family="Arial">{escape(X_LABEL)}</text>')
    lines.append(f'<text x="20" y="{height / 2}" text-anchor="middle" font-size="15" font-family="Arial" transform="rotate(-90 20 {height / 2})">{escape(Y_LABEL)}</text>')
    lines.append("</svg>")

    output_path = Path(output_path)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path.resolve()


def main():
    x = np.linspace(X_MIN, X_MAX, NUM_POINTS)
    curves = _build_curves(x)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        output_path = Path(SAVE_PATH).with_suffix(".svg") if SAVE_PATH else Path(FALLBACK_SVG_PATH)
        resolved_path = _save_svg_fallback(x, curves, output_path)
        print("matplotlib/Pillow 导入失败，已使用 SVG 备用绘图。")
        print(f"SVG 图像已保存到：{resolved_path}")
        print(f"原始错误：{exc}")
        return

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)

    for name, y in curves:
        ax.plot(x, y, linewidth=2.0, label=name)

    ax.set_title(TITLE)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.45)
    ax.legend()

    fig.tight_layout()

    if SAVE_PATH:
        output_path = Path(SAVE_PATH)
        fig.savefig(output_path, bbox_inches="tight")
        print(f"图像已保存到：{output_path.resolve()}")

    if SHOW_FIG:
        plt.show()


if __name__ == "__main__":
    main()
