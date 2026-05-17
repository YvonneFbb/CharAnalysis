#!/usr/bin/env python3
"""
Export a compact delta figure showing how regional gaps change across periods.
Uses a zero-centered dumbbell layout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "src/paper/figures/results_space"

TEXT = "#111827"
GRID = "#E5E7EB"
MID = "#94A3B8"
COLORS = {
    "aspect_ratio": "#64748B",
    "face_ratio": "#C44536",
    "ink_coverage": "#E58E26",
}
LABELS = {
    "aspect_ratio": "长宽比",
    "face_ratio": "字面率",
    "ink_coverage": "灰度比（黑/白）",
}

FONT_DIR = Path.home() / "Library/Fonts"
FONT_REGULAR_PATH = FONT_DIR / "PingFangSC-Regular.ttf"
FONT_SEMIBOLD_PATH = FONT_DIR / "PingFangSC-Semibold.ttf"


def _font_prop(path: Path):
    if path.exists():
        return fm.FontProperties(fname=str(path))
    return None


FONT_REGULAR = _font_prop(FONT_REGULAR_PATH)
FONT_SEMIBOLD = _font_prop(FONT_SEMIBOLD_PATH)


def _setup_fonts() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Songti SC",
        "Heiti TC",
        "STHeiti",
        "Noto Sans CJK SC",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"


def main() -> None:
    ap = argparse.ArgumentParser(description="Export delta comparison figure.")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--name", default="spatial_delta_compare")
    ap.add_argument("--title", default="区域差异变化")
    args = ap.parse_args()

    _setup_fonts()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fujian relative to Zhejiang
    data = {
        "aspect_ratio": [0.0838, 0.0178],
        "face_ratio": [0.0184, 0.0300],
        "ink_coverage": [-0.0181, 0.0409],
    }
    phases = ["中期", "后期"]
    fig, ax = plt.subplots(1, 1, figsize=(8.8, 4.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#9CA3AF")
    ax.spines["bottom"].set_color("#9CA3AF")

    ax.axvline(0, color=MID, linewidth=1.2, linestyle="--", zorder=1)

    markers = {
        "aspect_ratio": "o",
        "face_ratio": "s",
        "ink_coverage": "^",
    }
    y_positions = np.arange(len(data))[::-1]
    phase_offsets = {"中期": 0.12, "后期": -0.12}
    phase_colors = {"中期": "#64748B", "后期": "#C44536"}

    for y, (metric, vals) in zip(y_positions, data.items()):
        x1, x2 = vals
        ax.plot([x1, x2], [y, y], color=COLORS[metric], linewidth=2.6, alpha=0.9, zorder=2)
        for phase, xval in zip(phases, vals):
            ax.text(
                xval,
                y + phase_offsets[phase],
                f"{xval:+.4f}",
                ha="center",
                va="center",
                fontsize=8.2,
                color=phase_colors[phase],
                fontproperties=FONT_REGULAR,
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.85),
            )
        ax.scatter(
            [x1],
            [y],
            s=62,
            marker=markers[metric],
            facecolors="white",
            edgecolors=phase_colors["中期"],
            linewidths=1.6,
            zorder=3,
        )
        ax.scatter(
            [x2],
            [y],
            s=62,
            marker=markers[metric],
            facecolors="white",
            edgecolors=phase_colors["后期"],
            linewidths=1.6,
            zorder=3,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([LABELS[k] for k in data.keys()], fontproperties=FONT_REGULAR)
    ax.set_xlabel("福建相对两浙的均值差值  Δμ", color=TEXT, fontproperties=FONT_REGULAR, fontsize=10)
    ax.set_title(args.title, color=TEXT, fontsize=15, fontproperties=FONT_SEMIBOLD, pad=10)
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=phase_colors["中期"], markeredgewidth=1.5, markersize=7, label="中期"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=phase_colors["后期"], markeredgewidth=1.5, markersize=7, label="后期"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9, prop=FONT_REGULAR)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(FONT_REGULAR)

    fig.tight_layout()
    png_path = out_dir / f"{args.name}.png"
    pdf_path = out_dir / f"{args.name}.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
