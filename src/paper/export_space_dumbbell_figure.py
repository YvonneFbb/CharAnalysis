#!/usr/bin/env python3
"""
Export a dumbbell chart showing absolute regional means and gap changes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D

from src.paper.figure_export import default_dpi, parse_formats, save_figure_formats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "src/paper/figures/results_space"

TEXT = "#111827"
GRID = "#E5E7EB"
ZJ = "#64748B"
FJ = "#C44536"
MID_LINE = "#CBD5E1"
LATE_LINE = "#94A3B8"
FONT_DIR = Path.home() / "Library/Fonts"
FONT_REGULAR_PATH = FONT_DIR / "PingFangSC-Regular.ttf"
FONT_SEMIBOLD_PATH = FONT_DIR / "PingFangSC-Semibold.ttf"


def _font_prop(path: Path):
    if path.exists():
        return fm.FontProperties(fname=str(path))
    return None


FONT_REGULAR = _font_prop(FONT_REGULAR_PATH)
FONT_SEMIBOLD = _font_prop(FONT_SEMIBOLD_PATH)

LABELS = {
    "zh": {
        "title": "福建与两浙的空间差异变化",
        "metrics": {
            "长宽比": "长宽比（高/宽）",
            "字面率": "字面率",
            "灰度比（黑/白）": "灰度比（黑/白）",
        },
        "liangzhe": "两浙",
        "fujian": "福建",
        "middle": "中期（空心）",
        "late": "后期（实心）",
        "period": "时期",
        "yticks": ["中期", "后期"],
    },
    "en": {
        "title": "Change in regional difference between Fujian and Liangzhe",
        "metrics": {
            "长宽比": "aspect ratio (height/width)",
            "字面率": "face ratio",
            "灰度比（黑/白）": "ink coverage (black/white)",
        },
        "liangzhe": "Liangzhe",
        "fujian": "Fujian",
        "middle": "Middle (hollow)",
        "late": "Late (solid)",
        "period": "Period",
        "yticks": ["Middle", "Late"],
    },
}


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


def _labels(lang: str) -> dict:
    return LABELS["en" if str(lang).lower() == "en" else "zh"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Export dumbbell chart for regional comparisons.")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--name", default="spatial_dumbbell_compare")
    ap.add_argument("--title", default="")
    ap.add_argument("--lang", choices=("zh", "en"), default="zh")
    ap.add_argument("--formats", default="pdf")
    ap.add_argument("--dpi-kind", choices=("lineart", "grayscale", "colour", "color"), default="colour")
    args = ap.parse_args()

    _setup_fonts()
    labels = _labels(args.lang)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Absolute mu values from current summaries
    data = {
        "长宽比": {"mid": {"zj": 0.9180, "fj": 1.0018}, "late": {"zj": 0.9310, "fj": 0.9488}},
        "字面率": {"mid": {"zj": 0.7538, "fj": 0.7722}, "late": {"zj": 0.7749, "fj": 0.8042}},
        "灰度比（黑/白）": {"mid": {"zj": 0.1765, "fj": 0.1584}, "late": {"zj": 0.1945, "fj": 0.2354}},
    }

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.8), gridspec_kw={"wspace": 0.36})
    fig.patch.set_facecolor("white")
    fig.suptitle(args.title.strip() or labels["title"], y=0.94, color=TEXT, fontsize=14, fontproperties=FONT_SEMIBOLD)

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ZJ, markeredgecolor="white", markeredgewidth=1.0, markersize=8, label=labels["liangzhe"]),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=FJ, markeredgecolor="white", markeredgewidth=1.0, markersize=8, label=labels["fujian"]),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#64748B", markeredgewidth=1.5, markersize=8, label=labels["middle"]),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#64748B", markeredgecolor="white", markeredgewidth=1.0, markersize=8, label=labels["late"]),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.885),
        ncol=4,
        frameon=False,
        handletextpad=0.6,
        columnspacing=1.2,
        prop=FONT_REGULAR,
    )

    for ax, (title, vals) in zip(axes, data.items()):
        ax.set_facecolor("white")
        ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.85)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#9CA3AF")

        y_mid = 1.0
        y_late = 0.0

        # middle-period dumbbell
        ax.plot([vals["mid"]["zj"], vals["mid"]["fj"]], [y_mid, y_mid], color=MID_LINE, linewidth=2.6, zorder=1)
        ax.scatter(vals["mid"]["zj"], y_mid, s=70, facecolors="white", edgecolors=ZJ, linewidths=1.6, zorder=3)
        ax.scatter(vals["mid"]["fj"], y_mid, s=70, facecolors="white", edgecolors=FJ, linewidths=1.6, zorder=3)

        # late-period dumbbell
        ax.plot([vals["late"]["zj"], vals["late"]["fj"]], [y_late, y_late], color=LATE_LINE, linewidth=2.6, zorder=1)
        ax.scatter(vals["late"]["zj"], y_late, s=70, facecolors=ZJ, edgecolors="white", linewidths=1.0, zorder=3)
        ax.scatter(vals["late"]["fj"], y_late, s=70, facecolors=FJ, edgecolors="white", linewidths=1.0, zorder=3)

        delta_mid = vals["mid"]["fj"] - vals["mid"]["zj"]
        delta_late = vals["late"]["fj"] - vals["late"]["zj"]
        value_min = min(vals["mid"]["zj"], vals["mid"]["fj"], vals["late"]["zj"], vals["late"]["fj"])
        value_max = max(vals["mid"]["zj"], vals["mid"]["fj"], vals["late"]["zj"], vals["late"]["fj"])
        span = value_max - value_min
        pad = max(span * 0.18, 0.012)
        ax.set_xlim(value_min - pad * 0.35, value_max + pad * 1.15)
        delta_x = value_max + pad * 0.18
        ax.text(delta_x, y_mid, f"Δ={delta_mid:+.4f}", ha="left", va="center", fontsize=8.2, color="#475569", fontproperties=FONT_REGULAR)
        ax.text(delta_x, y_late, f"Δ={delta_late:+.4f}", ha="left", va="center", fontsize=8.2, color="#475569", fontproperties=FONT_REGULAR)

        ax.set_yticks([y_mid, y_late])
        if ax is axes[0]:
            ax.set_yticklabels(labels["yticks"], fontproperties=FONT_REGULAR)
            ax.tick_params(axis="y", length=0, colors=TEXT)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=9, colors=TEXT)
        for label in ax.get_xticklabels():
            label.set_fontproperties(FONT_REGULAR)
        ax.set_title(labels["metrics"][title], fontsize=11, color=TEXT, fontproperties=FONT_SEMIBOLD, pad=8)

    axes[0].set_ylabel(labels["period"], color=TEXT, fontsize=10, fontproperties=FONT_REGULAR)
    fig.subplots_adjust(top=0.76, left=0.08, right=0.98, bottom=0.14)

    saved_paths = save_figure_formats(fig, out_dir, args.name, parse_formats(args.formats), default_dpi(args.dpi_kind))
    plt.close(fig)
    for path in saved_paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
