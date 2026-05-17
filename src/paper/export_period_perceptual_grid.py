#!/usr/bin/env python3
"""
Export a period-level perceptual summary.

Each period is represented by one block:
- width/height ratio encodes aspect_ratio
- overall block size encodes face_ratio
- grayscale encodes ink_coverage without cross-period contrast enhancement
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle

from src.paper.figure_export import default_dpi, parse_formats, save_figure_formats
from src.analysis.api import build_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "src/paper/figures/results_stats"
BUNDLE = PROJECT_ROOT / "data/analysis"
META = PROJECT_ROOT / "data/metadata/books_metadata.csv"

TEXT = "#111827"
INK = "#111111"
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


def _collect_period_means():
    rows, _, char_means_by_book = build_dataset(BUNDLE, META, threshold=160)
    q = 0.05
    periods = {
        "早期": (1127, 1162),
        "中期": (1162, 1208),
        "后期": (1208, 1273),
    }
    metrics = ["aspect_ratio", "face_ratio", "ink_coverage"]
    result = {p: {} for p in periods}
    for pname, (start, end) in periods.items():
        rs = [r for r in rows if start <= int(r["year"]) < end]
        for metric in metrics:
            mus = []
            for r in rs:
                vals = [
                    float(v.get(metric, float("nan")))
                    for _, v in char_means_by_book[r["book"]].items()
                    if math.isfinite(float(v.get(metric, float("nan"))))
                ]
                if len(vals) >= 4:
                    lo = float(np.quantile(vals, q))
                    hi = float(np.quantile(vals, 1 - q))
                    vals = [v for v in vals if lo <= v <= hi]
                if vals:
                    mus.append(float(np.mean(np.asarray(vals, dtype=float))))
            result[pname][metric] = float(np.mean(np.asarray(mus, dtype=float)))
    return result


LABELS = {
    "zh": {
        "title": "三项核心刻体指标的分期典型水平",
        "periods": ["早期", "中期", "后期"],
    },
    "en": {
        "title": "Period-specific typical levels of the three core indicators",
        "periods": ["Early", "Middle", "Late"],
    },
}


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Export period-level perceptual summary.")
    ap.add_argument("--lang", choices=("zh", "en"), default="zh")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--name", default="period_perceptual_grid")
    ap.add_argument("--formats", default="pdf")
    ap.add_argument("--dpi-kind", choices=("lineart", "grayscale", "colour", "color"), default="grayscale")
    args = ap.parse_args()

    _setup_fonts()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = _collect_period_means()
    labels = LABELS["en" if args.lang == "en" else "zh"]

    fig = plt.figure(figsize=(7.2, 3.0))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        labels["title"],
        y=0.965,
        color=TEXT,
        fontsize=14,
        fontproperties=FONT_SEMIBOLD,
    )

    ax = fig.add_axes([0.03, 0.07, 0.94, 0.84])
    ax.set_xlim(0.0, 9.6)
    ax.set_ylim(0.0, 2.75)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    centers = [1.5, 4.8, 8.1]
    periods = ["早期", "中期", "后期"]
    period_names = labels["periods"]

    # Use a hidden common reference side. Width/height are drawn directly from data.
    ref_side = 1.65
    block_center_y = 1.42

    for cx, period, period_name in zip(centers, periods, period_names):
        ax.text(
            cx,
            2.23,
            period_name,
            ha="center",
            va="center",
            fontsize=12,
            color=TEXT,
            fontproperties=FONT_SEMIBOLD,
        )

        ar = stats[period]["aspect_ratio"]
        fr = stats[period]["face_ratio"]
        ic = stats[period]["ink_coverage"]

        # ar = h / w
        # fr = ((w + h) / 2) / L
        # with L normalized to 1:
        #   w = 2fr / (1 + ar)
        #   h = 2fr*ar / (1 + ar)
        w = (2.0 * fr) / (1.0 + ar)
        h = w * ar

        w_disp = ref_side * w
        h_disp = ref_side * h
        gx = cx - w_disp / 2
        gy = block_center_y - h_disp / 2

        # Objective grayscale from actual black fraction in the box:
        # r = B/W  =>  B/(B+W) = r/(1+r)
        black_fraction = ic / (1.0 + ic)
        gray = 1.0 - black_fraction

        ax.add_patch(
            Rectangle(
                (gx, gy),
                w_disp,
                h_disp,
                facecolor=(gray, gray, gray),
                edgecolor=INK,
                linewidth=1.2,
            )
        )

        ax.text(
            cx,
            0.36,
            f"a={ar:.3f}\nf={fr:.3f}\ni={ic:.3f}",
            ha="center",
            va="center",
            linespacing=1.18,
            fontsize=9.2,
            color=TEXT,
            fontproperties=FONT_REGULAR,
        )

    saved_paths = save_figure_formats(fig, out_dir, args.name, parse_formats(args.formats), default_dpi(args.dpi_kind))
    plt.close(fig)
    for path in saved_paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
