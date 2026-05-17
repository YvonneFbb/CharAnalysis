#!/usr/bin/env python3
"""
Export a paper-ready control-metrics figure grouped by period.

This figure is intentionally different from the main time-trend chart:
it emphasizes period-level distribution rather than continuous line trends.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle

from src.paper.figure_export import default_dpi, parse_formats, save_figure_formats
from src.analysis.api import build_dataset
from src.analysis.stats import compute_trend_stats, rankdata


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "src/paper/figures/results_time"

TEXT = "#111827"
GRID = "#E5E7EB"
GRAY_FILL = "#D9DEE7"
GRAY_EDGE = "#7C8A9A"
RED = "#C44536"
ORANGE = "#E58E26"
POINT = "#9AA3AF"
SUBTEXT = "#6B7280"
FONT_DIR = Path.home() / "Library/Fonts"
FONT_REGULAR_PATH = FONT_DIR / "PingFangSC-Regular.ttf"
FONT_SEMIBOLD_PATH = FONT_DIR / "PingFangSC-Semibold.ttf"

PERIODS = [
    ("早期", 1127, 1162),
    ("中期", 1162, 1208),
    ("后期", 1208, 1274),
]

LABELS = {
    "zh": {
        "periods": [("早期", 1127, 1162), ("中期", 1162, 1208), ("后期", 1208, 1274)],
        "reference_size": "基准尺寸",
        "clipping_rate": "裁切率",
        "ylabel": "书级标量",
        "title": "{region} 控制性指标变化",
    },
    "en": {
        "periods": [("Early", 1127, 1162), ("Middle", 1162, 1208), ("Late", 1208, 1274)],
        "reference_size": "reference size",
        "clipping_rate": "clipping rate",
        "ylabel": "book-level scalar",
        "title": "Auxiliary measures in the {region} sample",
    },
}


@dataclass
class PeriodMetric:
    metric: str
    label: str
    values_by_period: Dict[str, List[float]]
    points_by_period: Dict[str, List[Tuple[int, float]]]
    stats: Dict[str, float]


def _font_prop(path: Path):
    if path.exists():
        return fm.FontProperties(fname=str(path))
    return None


FONT_REGULAR = _font_prop(FONT_REGULAR_PATH)
FONT_SEMIBOLD = _font_prop(FONT_SEMIBOLD_PATH)


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _parse_list_arg(value: Optional[str]) -> List[str]:
    if not value:
        return []
    raw = [v.strip() for v in str(value).replace("，", ",").split(",")]
    return [v for v in raw if v]


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


def _labels(lang: str) -> Dict:
    return LABELS["en" if str(lang).lower() == "en" else "zh"]


def _select_rows(
    rows: List[Dict],
    region: str,
    styles: Sequence[str],
    include_books: Sequence[str],
    exclude_books: Sequence[str],
) -> List[Dict]:
    style_set = set(styles)
    include_set = set(include_books)
    exclude_set = set(exclude_books)
    out: List[Dict] = []
    for r in rows:
        if region and (r.get("region") or "") != region:
            continue
        book = r.get("book") or ""
        if include_set and book not in include_set:
            continue
        if book in exclude_set:
            continue
        if style_set:
            tags = set(r.get("style_tags") or [])
            if tags.isdisjoint(style_set):
                continue
        out.append(r)
    out.sort(key=lambda r: (int(r.get("year") or 0), int(r.get("order") or 0), r.get("book") or ""))
    return out


def _period_label(year: int, periods=None) -> Optional[str]:
    periods = periods or PERIODS
    for label, start, end in periods:
        if start <= year < end:
            return label
    return None


def _short_book_label(book: str) -> str:
    parts = (book or "").split("_", 2)
    if len(parts) >= 2:
        return f"{parts[0]}\n{parts[1]}"
    return book


def _compute_period_metric(rows: List[Dict], metric: str, label: str, periods=None) -> PeriodMetric:
    periods = periods or PERIODS
    values_by_period: Dict[str, List[float]] = {p[0]: [] for p in periods}
    points_by_period: Dict[str, List[Tuple[int, float]]] = {p[0]: [] for p in periods}
    years: List[int] = []
    values: List[float] = []
    for r in rows:
        year = int(r.get("year") or 0)
        period = _period_label(year, periods)
        value = float(r.get(metric, float("nan")))
        if not period or not math.isfinite(value):
            continue
        values_by_period[period].append(value)
        points_by_period[period].append((year, value))
        years.append(year)
        values.append(value)

    if len(values) >= 2:
        xs = rankdata(np.asarray([float(y) for y in years], dtype=float)).tolist()
        stats = compute_trend_stats(xs, values)
    else:
        stats = compute_trend_stats([], [])

    return PeriodMetric(
        metric=metric,
        label=label,
        values_by_period=values_by_period,
        points_by_period=points_by_period,
        stats=stats,
    )


def _fmt_num(value: float, digits: int = 3) -> str:
    return "NA" if not math.isfinite(value) else f"{value:.{digits}f}"


def _jitter(n: int, width: float = 0.16) -> np.ndarray:
    if n <= 1:
        return np.zeros(n, dtype=float)
    return np.linspace(-width, width, n)


def _add_stats(ax, stats: Dict[str, float]) -> None:
    txt = (
        f"ρ={_fmt_num(float(stats.get('rho', float('nan'))), 2)}, "
        f"p={_fmt_num(float(stats.get('p_rho', float('nan'))), 3)}, "
        f"τ={_fmt_num(float(stats.get('tau', float('nan'))), 2)}"
    )
    ax.text(
        0.03,
        0.96,
        txt,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        color=SUBTEXT,
        fontproperties=FONT_REGULAR,
    )


def _plot_period_panel(ax, pm: PeriodMetric, accent: str) -> None:
    period_names = [p[0] for p in PERIODS]
    data = [pm.values_by_period[name] for name in period_names]
    positions = np.arange(1, len(period_names) + 1)

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.42,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color=accent, linewidth=1.8),
        whiskerprops=dict(color=GRAY_EDGE, linewidth=1.0),
        capprops=dict(color=GRAY_EDGE, linewidth=1.0),
        boxprops=dict(color=GRAY_EDGE, linewidth=1.0),
    )
    for box in bp["boxes"]:
        box.set_facecolor(GRAY_FILL)
        box.set_alpha(0.35)

    for pos, vals in zip(positions, data):
        vals = list(vals)
        jit = _jitter(len(vals))
        ax.scatter(
            np.full(len(vals), pos, dtype=float) + jit,
            vals,
            s=24,
            facecolor="white",
            edgecolor=POINT,
            linewidth=1.0,
            zorder=3,
        )
        if vals:
            med = float(np.median(np.asarray(vals, dtype=float)))
            ax.hlines(med, pos - 0.28, pos + 0.28, colors=accent, linewidth=2.2, zorder=4)

    ax.set_xticks(positions)
    ax.set_xticklabels(period_names, fontsize=9, color=TEXT)
    ax.set_title(pm.label, fontsize=11, color=TEXT, pad=8, fontproperties=FONT_SEMIBOLD)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, alpha=0.8)
    ax.tick_params(axis="y", labelsize=9, colors=TEXT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRAY_EDGE)
    ax.spines["bottom"].set_color(GRAY_EDGE)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(FONT_REGULAR)
    _add_stats(ax, pm.stats)


def _plot_range_band_panel(ax, pm: PeriodMetric, accent: str) -> None:
    ordered_points = sorted(
        [(year, value) for pts in pm.points_by_period.values() for year, value in pts],
        key=lambda item: item[0],
    )
    xs = list(range(len(ordered_points)))
    values = [value for _, value in ordered_points]

    if values:
        q1 = float(np.quantile(np.asarray(values, dtype=float), 0.25))
        q3 = float(np.quantile(np.asarray(values, dtype=float), 0.75))
        median = float(np.median(np.asarray(values, dtype=float)))
        x0 = -0.5
        x1 = max(len(xs) - 0.5, 0.5)
        ax.hlines(q1, x0, x1, colors=GRAY_EDGE, linewidth=1.2, linestyle="--", zorder=1)
        ax.hlines(q3, x0, x1, colors=GRAY_EDGE, linewidth=1.2, linestyle="--", zorder=1)
        ax.hlines(median, x0, x1, colors=accent, linewidth=2.0, zorder=2)
        label_x = x1 - 0.45
        label_box = dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.88)
        ax.text(label_x, q3, "Q3", ha="left", va="bottom", fontsize=8.0, color=GRAY_EDGE, fontproperties=FONT_REGULAR, bbox=label_box)
        ax.text(label_x, median, "Median", ha="left", va="bottom", fontsize=8.0, color=accent, fontproperties=FONT_REGULAR, bbox=label_box)
        ax.text(label_x, q1, "Q1", ha="left", va="top", fontsize=8.0, color=GRAY_EDGE, fontproperties=FONT_REGULAR, bbox=label_box)

    if xs:
        ax.scatter(
            xs,
            values,
            s=34,
            facecolor="white",
            edgecolor=POINT,
            linewidth=1.3,
            zorder=3,
        )

    ax.set_xlim(-0.5, max(len(xs) - 0.5, 0.5))
    tick_idx = list(range(len(xs)))
    tick_labels = [str(i + 1).zfill(2) for i in tick_idx]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_labels, fontsize=8.5, color=TEXT)
    ax.set_title(pm.label, fontsize=11, color=TEXT, pad=8, fontproperties=FONT_SEMIBOLD)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, alpha=0.8)
    ax.tick_params(axis="x", labelsize=8.5, colors=TEXT)
    ax.tick_params(axis="y", labelsize=9, colors=TEXT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRAY_EDGE)
    ax.spines["bottom"].set_color(GRAY_EDGE)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontproperties(FONT_REGULAR)
    _add_stats(ax, pm.stats)


def _plot_sequence_trend_panel(ax, pm: PeriodMetric, accent: str) -> None:
    ordered_points = sorted(
        [(year, value) for pts in pm.points_by_period.values() for year, value in pts],
        key=lambda item: item[0],
    )
    xs = np.arange(len(ordered_points), dtype=float)
    values = np.asarray([value for _, value in ordered_points], dtype=float)

    if values.size == 0:
        return

    ax.scatter(
        xs,
        values,
        s=34,
        facecolor="white",
        edgecolor=POINT,
        linewidth=1.3,
        zorder=3,
    )

    if values.size >= 2:
        coeff = np.polyfit(xs, values, deg=1)
        fit = np.polyval(coeff, xs)
        ax.plot(xs, fit, color=accent, linewidth=2.2, zorder=4)

    q1 = float(np.quantile(values, 0.25))
    q3 = float(np.quantile(values, 0.75))
    rect = Rectangle(
        (-0.5, q1),
        max(len(xs), 1),
        q3 - q1,
        facecolor=GRAY_FILL,
        edgecolor="none",
        alpha=0.18,
        zorder=0,
    )
    ax.add_patch(rect)

    tick_idx = list(range(len(xs)))
    tick_labels = [str(i + 1).zfill(2) for i in tick_idx]
    ax.set_xlim(-0.5, max(len(xs) - 0.5, 0.5))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_labels, fontsize=8.5, color=TEXT)
    ax.set_title(pm.label, fontsize=11, color=TEXT, pad=8, fontproperties=FONT_SEMIBOLD)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, alpha=0.8)
    ax.tick_params(axis="x", labelsize=8.5, colors=TEXT)
    ax.tick_params(axis="y", labelsize=9, colors=TEXT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRAY_EDGE)
    ax.spines["bottom"].set_color(GRAY_EDGE)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontproperties(FONT_REGULAR)
    _add_stats(ax, pm.stats)


def _write_summary(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a static control-metrics figure grouped by period.")
    parser.add_argument("--bundle", default="data/analysis", help="Analysis bundle root")
    parser.add_argument("--metadata", default="data/metadata/books_metadata.csv", help="Metadata CSV")
    parser.add_argument("--threshold", type=int, default=160, help="Black threshold")
    parser.add_argument("--region", required=True, help="Target region")
    parser.add_argument("--styles", default="", help="Comma-separated style tags")
    parser.add_argument("--include-books", default="", help="Explicit book IDs to include")
    parser.add_argument("--exclude-books", default="", help="Book IDs to exclude")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--name", default="time_control", help="Output stem")
    parser.add_argument("--title", default="", help="Figure title")
    parser.add_argument("--lang", choices=("zh", "en"), default="zh")
    parser.add_argument("--formats", default="pdf")
    parser.add_argument("--dpi-kind", choices=("lineart", "grayscale", "colour", "color"), default="grayscale")
    args = parser.parse_args()

    _setup_fonts()
    labels = _labels(args.lang)
    periods = labels["periods"]

    bundle_root = _resolve_path(args.bundle)
    metadata_path = _resolve_path(args.metadata)
    out_dir = _resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, _samples_by_book, _char_means_by_book = build_dataset(
        bundle_root=bundle_root,
        metadata_path=metadata_path,
        threshold=int(args.threshold),
    )
    selected_rows = _select_rows(
        rows=rows,
        region=args.region,
        styles=_parse_list_arg(args.styles),
        include_books=_parse_list_arg(args.include_books),
        exclude_books=_parse_list_arg(args.exclude_books),
    )
    if not selected_rows:
        raise SystemExit("No rows selected for plotting.")

    metrics = [
        _compute_period_metric(selected_rows, "Lb", labels["reference_size"], periods),
        _compute_period_metric(selected_rows, "clipping_rate", labels["clipping_rate"], periods),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.1), gridspec_kw={"wspace": 0.22})
    fig.patch.set_facecolor("white")

    _plot_range_band_panel(axes[0], metrics[0], RED)
    _plot_sequence_trend_panel(axes[1], metrics[1], ORANGE)
    axes[0].set_ylabel(labels["ylabel"], fontsize=10, color=TEXT, fontproperties=FONT_REGULAR)

    title = args.title.strip() if args.title else labels["title"].format(region=args.region)
    fig.suptitle(title, fontsize=14, color=TEXT, y=0.98, fontproperties=FONT_SEMIBOLD)
    fig.subplots_adjust(top=0.88, bottom=0.10, left=0.07, right=0.98)

    saved_paths = save_figure_formats(fig, out_dir, args.name, parse_formats(args.formats), default_dpi(args.dpi_kind))
    plt.close(fig)

    summary = {
        "title": title,
        "region": args.region,
        "styles": _parse_list_arg(args.styles),
        "metrics": [
            {
                "metric": pm.metric,
                "stats": pm.stats,
                "values_by_period": pm.values_by_period,
            }
            for pm in metrics
        ],
        "outputs": {p.suffix.lstrip("."): str(p) for p in saved_paths},
    }
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
