#!/usr/bin/env python3
"""
Export a paper-ready static time-trend figure.

The figure is designed for result sections. It uses a clean white background and
three aligned metric panels. For each metric:
  - top row: per-book distribution (boxplot) + overall-fit / sensitivity-fit mu
  - bottom row: overall-fit / sensitivity-fit sigma

A JSON/TXT summary is also written for drafting captions and result prose.
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.paper.figure_export import default_dpi, parse_formats, save_figure_formats
from src.analysis.api import (
    METRIC_SAMPLES,
    METRICS,
    build_dataset,
    compute_influence,
)
from src.analysis.stats import compute_trend_stats, rankdata


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "src/paper/figures/results_time"

MAIN_RED = "#C44536"
ACCENT_ORANGE = "#E58E26"
BOX_FILL = "#D9DEE7"
BOX_EDGE = "#7C8A9A"
GRID = "#E5E7EB"
TEXT = "#111827"
OVERALL_LINE = "#C7CDD7"
POINT_GRAY = "#9AA3AF"
FONT_DIR = Path.home() / "Library/Fonts"
FONT_REGULAR_PATH = FONT_DIR / "PingFangSC-Regular.ttf"
FONT_SEMIBOLD_PATH = FONT_DIR / "PingFangSC-Semibold.ttf"


@dataclass
class MetricSeries:
    metric: str
    rows: List[Dict]
    arrays: List[np.ndarray]
    mus: List[float]
    sigmas: List[float]
    common_chars_count: Optional[int]
    stats_u: Dict[str, float]
    stats_a: Dict[str, float]
    influence_u: List[Dict]
    influence_a: List[Dict]
    top2_u_books: List[str]
    top2_a_books: List[str]
    stats_u_top2: Dict[str, float]
    stats_a_top2: Dict[str, float]


def _font_prop(path: Path):
    if path.exists():
        return fm.FontProperties(fname=str(path))
    return None


FONT_REGULAR = _font_prop(FONT_REGULAR_PATH)
FONT_SEMIBOLD = _font_prop(FONT_SEMIBOLD_PATH)

METRIC_TITLES = {
    "zh": {
        "aspect_ratio": "长宽比（高/宽）",
        "face_ratio": "字面率",
        "ink_coverage": "灰度比（黑/白）",
        "ylabel_mu": "书级均值 μ",
        "ylabel_sigma": "书内离散度 σ",
        "legend_distribution": "分布",
        "legend_overall_trend": "总体趋势",
        "legend_overall_mu": "总体均值 μ",
        "legend_overall_sigma": "总体离散度 σ",
        "legend_potential_mu": "潜在均值 μ",
        "legend_potential_sigma": "潜在离散度 σ",
        "subtitle_common": "common chars",
        "prefix_overall": "总",
        "prefix_potential": "潜",
        "default_title": "{region} 时间演化",
    },
    "en": {
        "aspect_ratio": "aspect ratio (height/width)",
        "face_ratio": "face ratio",
        "ink_coverage": "ink coverage (black/white)",
        "ylabel_mu": "book-level mean μ",
        "ylabel_sigma": "within-book dispersion σ",
        "legend_distribution": "distribution",
        "legend_overall_trend": "overall trend",
        "legend_overall_mu": "book-level mean μ",
        "legend_overall_sigma": "within-book dispersion σ",
        "legend_potential_mu": "potential mean μ",
        "legend_potential_sigma": "potential dispersion σ",
        "subtitle_common": "common characters",
        "prefix_overall": "overall",
        "prefix_potential": "potential",
        "default_title": "Temporal change in the {region} sample",
    },
}


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


def _short_book_label(book: str) -> str:
    parts = (book or "").split("_", 2)
    if len(parts) >= 2:
        return f"{parts[0]}\n{parts[1]}"
    return book


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


def _labels(lang: str) -> Dict[str, str]:
    return METRIC_TITLES["en" if str(lang).lower() == "en" else "zh"]


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

    out.sort(
        key=lambda r: (
            int(r.get("year") or 0),
            int(r.get("order") or 0),
            r.get("book") or "",
        )
    )
    return out


def _compute_metric_series(
    rows: List[Dict],
    char_means_by_book: Dict[str, Dict[str, Dict[str, float]]],
    metric: str,
    q: float,
    common_chars_enabled: bool,
) -> MetricSeries:
    sample_key = METRIC_SAMPLES[metric]
    per_book_vals: Dict[str, List[Tuple[str, float]]] = {}
    for r in rows:
        book = r.get("book") or ""
        char_map = char_means_by_book.get(book, {})
        vals = [
            (c, float(v.get(sample_key, float("nan"))))
            for c, v in char_map.items()
            if math.isfinite(float(v.get(sample_key, float("nan"))))
        ]
        if vals:
            per_book_vals[book] = vals

    common_chars: List[str] = []
    if common_chars_enabled and per_book_vals:
        common_set: Optional[set[str]] = None
        for r in rows:
            book = r.get("book") or ""
            if book not in per_book_vals:
                continue
            keys = {c for c, _ in per_book_vals[book]}
            common_set = keys if common_set is None else (common_set & keys)
        common_chars = sorted(common_set) if common_set else []

    plot_rows: List[Dict] = []
    arrays: List[np.ndarray] = []
    for r in rows:
        book = r.get("book") or ""
        vals = per_book_vals.get(book, [])
        if common_chars_enabled:
            vals = [(c, v) for c, v in vals if c in common_chars]
        values_only = [v for _, v in vals]
        if q > 0.0 and len(values_only) >= 4:
            lo = float(np.quantile(values_only, q))
            hi = float(np.quantile(values_only, 1.0 - q))
            vals = [(c, v) for c, v in vals if lo <= v <= hi]
        arr = np.asarray([v for _, v in vals], dtype=float)
        if arr.size == 0:
            continue
        plot_rows.append(r)
        arrays.append(arr)

    mus = [float(np.mean(arr)) if arr.size else float("nan") for arr in arrays]
    sigmas = [float(np.std(arr)) if arr.size else float("nan") for arr in arrays]
    years = [int(r.get("year") or 0) for r in plot_rows]

    valid_pairs_u = [(y, v) for y, v in zip(years, mus) if math.isfinite(v)]
    if len(valid_pairs_u) >= 2:
        xs = rankdata(np.asarray([float(y) for y, _ in valid_pairs_u], dtype=float)).tolist()
        ys = [float(v) for _, v in valid_pairs_u]
        stats_u = compute_trend_stats(xs, ys)
    else:
        stats_u = compute_trend_stats([], [])

    valid_pairs_a = [(y, v) for y, v in zip(years, sigmas) if math.isfinite(v)]
    if len(valid_pairs_a) >= 2:
        xs = rankdata(np.asarray([float(y) for y, _ in valid_pairs_a], dtype=float)).tolist()
        ys = [float(v) for _, v in valid_pairs_a]
        stats_a = compute_trend_stats(xs, ys)
    else:
        stats_a = compute_trend_stats([], [])

    influence_u = compute_influence(plot_rows, years, mus, limit=12)
    influence_a = compute_influence(plot_rows, years, sigmas, limit=12)

    top2_u_books = [row.get("book") or "" for row in influence_u[:2] if row.get("book")]
    top2_a_books = [row.get("book") or "" for row in influence_a[:2] if row.get("book")]

    def _stats_after_excluding(excluded: Sequence[str], values: Sequence[float]) -> Dict[str, float]:
        pairs = [
            (plot_rows[i], years[i], float(values[i]))
            for i in range(len(plot_rows))
            if math.isfinite(float(values[i])) and (plot_rows[i].get("book") or "") not in set(excluded)
        ]
        if len(pairs) < 2:
            return compute_trend_stats([], [])
        xs = rankdata(np.asarray([float(y) for _, y, _ in pairs], dtype=float)).tolist()
        ys = [float(v) for _, _, v in pairs]
        return compute_trend_stats(xs, ys)

    stats_u_top2 = _stats_after_excluding(top2_u_books, mus)
    stats_a_top2 = _stats_after_excluding(top2_a_books, sigmas)

    return MetricSeries(
        metric=metric,
        rows=plot_rows,
        arrays=arrays,
        mus=mus,
        sigmas=sigmas,
        common_chars_count=len(common_chars) if common_chars_enabled else None,
        stats_u=stats_u,
        stats_a=stats_a,
        influence_u=influence_u,
        influence_a=influence_a,
        top2_u_books=top2_u_books,
        top2_a_books=top2_a_books,
        stats_u_top2=stats_u_top2,
        stats_a_top2=stats_a_top2,
    )


def _fmt_num(value: float, digits: int = 3) -> str:
    return "NA" if not math.isfinite(value) else f"{value:.{digits}f}"


def _smooth_trend(
    xs: Sequence[float],
    ys: Sequence[float],
    eval_xs: Optional[Sequence[float]] = None,
) -> Optional[np.ndarray]:
    xv = np.asarray(list(xs), dtype=float)
    yv = np.asarray(list(ys), dtype=float)
    mask = np.isfinite(xv) & np.isfinite(yv)
    xv = xv[mask]
    yv = yv[mask]
    if xv.size < 2:
        return None
    order = np.argsort(xv)
    xv = xv[order]
    yv = yv[order]

    if xv.size >= 10:
        window = 5
    elif xv.size >= 5:
        window = 3
    else:
        window = 1

    if window > 1:
        pad = window // 2
        kernel = np.ones(window, dtype=float) / float(window)
        y_pad = np.pad(yv, (pad, pad), mode="edge")
        y_smooth = np.convolve(y_pad, kernel, mode="valid")
    else:
        y_smooth = yv

    target = np.asarray(list(eval_xs if eval_xs is not None else xs), dtype=float)
    return np.interp(target, xv, y_smooth)


def _smooth_trend_after_excluding(
    series: MetricSeries,
    values: Sequence[float],
    excluded_books: Sequence[str],
) -> Optional[np.ndarray]:
    excluded = set(excluded_books)
    positions = list(range(len(series.rows)))
    xs: List[float] = []
    ys: List[float] = []
    for pos, row, value in zip(positions, series.rows, values):
        if (row.get("book") or "") in excluded:
            continue
        if math.isfinite(float(value)):
            xs.append(float(pos))
            ys.append(float(value))
    return _smooth_trend(xs, ys, eval_xs=positions)


def _add_panel_stats(
    ax,
    overall_stats: Dict[str, float],
    conditional_stats: Optional[Dict[str, float]],
    conditional_color: str,
    labels: Dict[str, str],
) -> None:
    overall_txt = (
        f"{labels['prefix_overall']} "
        f"$\\rho$={_fmt_num(float(overall_stats.get('rho', float('nan'))), 2)}, "
        f"$p$={_fmt_num(float(overall_stats.get('p_rho', float('nan'))), 3)}, "
        f"$\\tau$={_fmt_num(float(overall_stats.get('tau', float('nan'))), 2)}"
    )
    ax.text(
        0.02,
        0.97,
        overall_txt,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        color="#6B7280",
        fontproperties=FONT_REGULAR,
    )
    if conditional_stats is not None:
        conditional_txt = (
            f"{labels['prefix_potential']} "
            f"$\\rho$={_fmt_num(float(conditional_stats.get('rho', float('nan'))), 2)}, "
            f"$p$={_fmt_num(float(conditional_stats.get('p_rho', float('nan'))), 3)}, "
            f"$\\tau$={_fmt_num(float(conditional_stats.get('tau', float('nan'))), 2)}"
        )
        ax.text(
            0.02,
            0.915,
            conditional_txt,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.2,
            color=conditional_color,
            fontproperties=FONT_REGULAR,
        )


def _plot_top_panel(ax, series: MetricSeries, labels: Dict[str, str], overall_only: bool = False) -> None:
    positions = list(range(len(series.rows)))
    excluded_u = set(series.top2_u_books)
    whis = (5.0, 95.0)
    box = ax.boxplot(
        series.arrays,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showmeans=False,
        meanline=False,
        whis=whis,
        showfliers=True,
    )
    for patch in box["boxes"]:
        patch.set_facecolor(BOX_FILL)
        patch.set_edgecolor(BOX_EDGE)
        patch.set_linewidth(1.0)
        patch.set_alpha(0.35)
    for med in box["medians"]:
        med.set_color(BOX_EDGE)
        med.set_linewidth(1.2)
    for line in box["whiskers"] + box["caps"]:
        line.set_linewidth(0.9)
        line.set_color(BOX_EDGE)
        line.set_alpha(0.65)
    for flier in box["fliers"]:
        flier.set_marker("o")
        flier.set_markersize(2.5)
        flier.set_alpha(0.08)
        flier.set_markerfacecolor(BOX_EDGE)
        flier.set_markeredgecolor("none")

    ax.scatter(
        positions,
        series.mus,
        s=11,
        color=POINT_GRAY,
        alpha=0.55,
        zorder=4,
    )

    overall_fit = _smooth_trend(positions, series.mus)
    if overall_fit is not None:
        ax.plot(
            positions,
            overall_fit,
            color=MAIN_RED if overall_only else OVERALL_LINE,
            linewidth=2.0,
            linestyle="--",
            zorder=4,
        )

    if not overall_only:
        top2_fit = _smooth_trend_after_excluding(series, series.mus, series.top2_u_books)
        if top2_fit is not None:
            ax.plot(
                positions,
                top2_fit,
                color=MAIN_RED,
                linewidth=2.2,
                linestyle="-",
                zorder=5,
            )

    if overall_only:
        ax.plot(
            positions,
            series.mus,
            linestyle="none",
            marker="o",
            markersize=3.8,
            markerfacecolor="white",
            markeredgecolor=MAIN_RED,
            markeredgewidth=1.1,
            zorder=6,
        )
    else:
        keep_pos = [pos for pos, row in zip(positions, series.rows) if (row.get("book") or "") not in excluded_u]
        keep_vals = [val for val, row in zip(series.mus, series.rows) if (row.get("book") or "") not in excluded_u]
        drop_pos = [pos for pos, row in zip(positions, series.rows) if (row.get("book") or "") in excluded_u]
        drop_vals = [val for val, row in zip(series.mus, series.rows) if (row.get("book") or "") in excluded_u]

        if keep_pos:
            ax.plot(
                keep_pos,
                keep_vals,
                linestyle="none",
                marker="o",
                markersize=3.8,
                markerfacecolor="white",
                markeredgecolor=MAIN_RED,
                markeredgewidth=1.1,
                zorder=6,
            )
        if drop_pos:
            ax.plot(
                drop_pos,
                drop_vals,
                linestyle="none",
                marker="o",
                markersize=3.8,
                markerfacecolor="white",
                markeredgecolor=POINT_GRAY,
                markeredgewidth=1.1,
                zorder=6,
            )

    ax.set_title(labels[series.metric], fontsize=11, color=TEXT, pad=3, fontproperties=FONT_SEMIBOLD)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, alpha=0.8)
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax.tick_params(axis="y", labelsize=9, colors=TEXT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9CA3AF")
    ax.spines["bottom"].set_color("#9CA3AF")
    for label in ax.get_yticklabels():
        label.set_fontproperties(FONT_REGULAR)
    _add_panel_stats(ax, series.stats_u, None if overall_only else series.stats_u_top2, MAIN_RED, labels)


def _plot_bottom_panel(ax, series: MetricSeries, labels: Dict[str, str], overall_only: bool = False) -> None:
    positions = list(range(len(series.rows)))
    excluded_a = set(series.top2_a_books)
    ax.scatter(
        positions,
        series.sigmas,
        s=12,
        color=POINT_GRAY,
        alpha=0.55,
        zorder=3,
    )

    overall_fit = _smooth_trend(positions, series.sigmas)
    if overall_fit is not None:
        ax.plot(
            positions,
            overall_fit,
            color=ACCENT_ORANGE if overall_only else OVERALL_LINE,
            linewidth=2.0,
            linestyle="--",
            zorder=3,
        )

    if not overall_only:
        top2_fit = _smooth_trend_after_excluding(series, series.sigmas, series.top2_a_books)
        if top2_fit is not None:
            ax.plot(
                positions,
                top2_fit,
                color=ACCENT_ORANGE,
                linewidth=2.2,
                linestyle="-",
                zorder=4,
            )

    if overall_only:
        ax.plot(
            positions,
            series.sigmas,
            color=ACCENT_ORANGE,
            linewidth=0.0,
            marker="^",
            markersize=4.0,
            markerfacecolor="white",
            markeredgecolor=ACCENT_ORANGE,
            markeredgewidth=1.0,
            zorder=5,
        )
    else:
        keep_pos = [pos for pos, row in zip(positions, series.rows) if (row.get("book") or "") not in excluded_a]
        keep_vals = [val for val, row in zip(series.sigmas, series.rows) if (row.get("book") or "") not in excluded_a]
        drop_pos = [pos for pos, row in zip(positions, series.rows) if (row.get("book") or "") in excluded_a]
        drop_vals = [val for val, row in zip(series.sigmas, series.rows) if (row.get("book") or "") in excluded_a]

        if keep_pos:
            ax.plot(
                keep_pos,
                keep_vals,
                color=ACCENT_ORANGE,
                linewidth=0.0,
                marker="^",
                markersize=4.0,
                markerfacecolor="white",
                markeredgecolor=ACCENT_ORANGE,
                markeredgewidth=1.0,
                zorder=5,
            )
        if drop_pos:
            ax.plot(
                drop_pos,
                drop_vals,
                color=ACCENT_ORANGE,
                linewidth=0.0,
                marker="^",
                markersize=4.0,
                markerfacecolor="white",
                markeredgecolor=POINT_GRAY,
                markeredgewidth=1.0,
                zorder=5,
            )
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, alpha=0.8)
    ax.tick_params(axis="y", labelsize=8.5, colors=TEXT)
    ax.tick_params(axis="x", labelsize=7.5, colors=TEXT, rotation=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9CA3AF")
    ax.spines["bottom"].set_color("#9CA3AF")
    ax.set_xticks(positions)
    ax.set_xticklabels([_short_book_label(r.get("book") or "") for r in series.rows])
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(FONT_REGULAR)
    _add_panel_stats(ax, series.stats_a, None if overall_only else series.stats_a_top2, ACCENT_ORANGE, labels)


def _write_summary(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text_summary(path: Path, payload: Dict) -> None:
    lines: List[str] = []
    lines.append(payload["title"])
    lines.append("")
    lines.append("selected_books:")
    for row in payload["selected_books"]:
        lines.append(f"  {row['book']}  {row['year']}  {row['style']}")
    lines.append("")
    for item in payload["metrics"]:
        lines.append(f"[{item['metric']}] common_chars={item['common_chars_count']}")
        lines.append(
            "  mu overall: rho={rho} p={p} tau={tau}".format(
                rho=_fmt_num(float(item["stats_u"]["rho"]), 3),
                p=_fmt_num(float(item["stats_u"]["p_rho"]), 3),
                tau=_fmt_num(float(item["stats_u"]["tau"]), 3),
            )
        )
        lines.append(
            "  mu top2({books}): rho={rho} p={p} tau={tau}".format(
                books=",".join(item["top2_u_books"]) if item["top2_u_books"] else "NA",
                rho=_fmt_num(float(item["stats_u_top2"]["rho"]), 3),
                p=_fmt_num(float(item["stats_u_top2"]["p_rho"]), 3),
                tau=_fmt_num(float(item["stats_u_top2"]["tau"]), 3),
            )
        )
        lines.append(
            "  sigma overall: rho={rho} p={p} tau={tau}".format(
                rho=_fmt_num(float(item["stats_a"]["rho"]), 3),
                p=_fmt_num(float(item["stats_a"]["p_rho"]), 3),
                tau=_fmt_num(float(item["stats_a"]["tau"]), 3),
            )
        )
        lines.append(
            "  sigma top2({books}): rho={rho} p={p} tau={tau}".format(
                books=",".join(item["top2_a_books"]) if item["top2_a_books"] else "NA",
                rho=_fmt_num(float(item["stats_a_top2"]["rho"]), 3),
                p=_fmt_num(float(item["stats_a_top2"]["p_rho"]), 3),
                tau=_fmt_num(float(item["stats_a_top2"]["tau"]), 3),
            )
        )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a static time-trend figure for the paper.")
    parser.add_argument("--bundle", default="data/analysis", help="Analysis bundle root")
    parser.add_argument("--metadata", default="data/metadata/books_metadata.csv", help="Metadata CSV")
    parser.add_argument("--threshold", type=int, default=160, help="Black threshold")
    parser.add_argument("--region", required=True, help="Target region")
    parser.add_argument("--styles", default="", help="Comma-separated style tags")
    parser.add_argument("--include-books", default="", help="Explicit book IDs to include")
    parser.add_argument("--exclude-books", default="", help="Book IDs to exclude")
    parser.add_argument("--q", type=float, default=0.05, help="Trim quantile")
    parser.add_argument("--no-common-chars", action="store_true", help="Disable common-char filtering")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--name", default="time_trend", help="Output stem")
    parser.add_argument("--title", default="", help="Figure title")
    parser.add_argument("--show-subtitle", action="store_true", help="Show subtitle with n/common chars/q")
    parser.add_argument("--overall-only", action="store_true", help="Show only overall trends without conditional results")
    parser.add_argument("--lang", choices=("zh", "en"), default="zh")
    parser.add_argument("--formats", default="pdf", help="Comma-separated formats: pdf,tiff,jpeg")
    parser.add_argument("--dpi-kind", choices=("lineart", "grayscale", "colour", "color"), default="colour")
    args = parser.parse_args()

    _setup_fonts()
    labels = _labels(args.lang)

    bundle_root = _resolve_path(args.bundle)
    metadata_path = _resolve_path(args.metadata)
    out_dir = _resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, _samples_by_book, char_means_by_book = build_dataset(
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

    metrics = ["aspect_ratio", "face_ratio", "ink_coverage"]
    series_list = [
        _compute_metric_series(
            rows=selected_rows,
            char_means_by_book=char_means_by_book,
            metric=metric,
            q=max(0.0, min(0.25, float(args.q))),
            common_chars_enabled=not bool(args.no_common_chars),
        )
        for metric in metrics
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15.5, 8.8),
        sharex="col",
        gridspec_kw={"height_ratios": [3.2, 1.35], "wspace": 0.18, "hspace": 0.10},
    )
    fig.patch.set_facecolor("white")

    for col, series in enumerate(series_list):
        _plot_top_panel(axes[0, col], series, labels, overall_only=bool(args.overall_only))
        _plot_bottom_panel(axes[1, col], series, labels, overall_only=bool(args.overall_only))

    axes[0, 0].set_ylabel(labels["ylabel_mu"], fontsize=10, color=TEXT, fontproperties=FONT_REGULAR)
    axes[1, 0].set_ylabel(labels["ylabel_sigma"], fontsize=10, color=TEXT, fontproperties=FONT_REGULAR)

    title = args.title.strip() if args.title else labels["default_title"].format(region=args.region)
    common_count = series_list[0].common_chars_count
    subtitle = f"n={len(series_list[0].rows)}"
    if common_count is not None:
        subtitle += f", {labels['subtitle_common']}={common_count}"
    subtitle += f", q={float(args.q):.2f}"
    fig.suptitle(title, fontsize=14, color=TEXT, y=0.962, fontproperties=FONT_SEMIBOLD)
    if args.show_subtitle:
        fig.text(0.5, 0.936, subtitle, ha="center", va="top", fontsize=9.5, color="#4B5563", fontproperties=FONT_REGULAR)
    fig.subplots_adjust(top=0.86 if args.show_subtitle else 0.88, bottom=0.08, left=0.06, right=0.985)

    if args.overall_only:
        legend_handles = [
            Patch(facecolor=BOX_FILL, edgecolor=BOX_EDGE, alpha=0.35, label=labels["legend_distribution"]),
            Line2D([0], [0], color=MAIN_RED, lw=2.2, linestyle="--", label=labels["legend_overall_mu"]),
            Line2D([0], [0], color=ACCENT_ORANGE, lw=2.2, linestyle="--", label=labels["legend_overall_sigma"]),
        ]
        legend_ncol = 3
    else:
        legend_handles = [
            Patch(facecolor=BOX_FILL, edgecolor=BOX_EDGE, alpha=0.35, label=labels["legend_distribution"]),
            Line2D([0], [0], color=OVERALL_LINE, lw=2.0, linestyle="--", label=labels["legend_overall_trend"]),
            Line2D([0], [0], color=MAIN_RED, lw=2.2, linestyle="-", label=labels["legend_potential_mu"]),
            Line2D([0], [0], color=ACCENT_ORANGE, lw=2.2, linestyle="-", label=labels["legend_potential_sigma"]),
        ]
        legend_ncol = 4
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925 if args.show_subtitle else 0.945),
        ncol=legend_ncol,
        frameon=False,
        fontsize=8.5,
        handlelength=1.8,
        columnspacing=1.2,
        prop=FONT_REGULAR,
    )

    saved_paths = save_figure_formats(fig, out_dir, args.name, parse_formats(args.formats), default_dpi(args.dpi_kind))
    plt.close(fig)

    summary = {
        "title": title,
        "subtitle": subtitle,
        "region": args.region,
        "styles": _parse_list_arg(args.styles),
        "selected_books": [
            {
                "book": r.get("book"),
                "year": r.get("year"),
                "style": r.get("style"),
                "style_tags": r.get("style_tags"),
            }
            for r in selected_rows
        ],
        "metrics": [
            {
                "metric": series.metric,
                "common_chars_count": series.common_chars_count,
                "stats_u": series.stats_u,
                "stats_a": series.stats_a,
                "top2_u_books": series.top2_u_books,
                "top2_a_books": series.top2_a_books,
                "stats_u_top2": series.stats_u_top2,
                "stats_a_top2": series.stats_a_top2,
                "influence_u": series.influence_u,
                "influence_a": series.influence_a,
            }
            for series in series_list
        ],
        "outputs": {
            p.suffix.lstrip("."): str(p) for p in saved_paths
        },
    }

    for path in saved_paths:
        print(f"Saved: {path}")
    print(f"Selected books: {len(selected_rows)}")
    for series in series_list:
        print(
            f"{series.metric}: common={series.common_chars_count} "
            f"rho_mu={_fmt_num(float(series.stats_u.get('rho', float('nan'))), 3)} "
            f"rho_sigma={_fmt_num(float(series.stats_a.get('rho', float('nan'))), 3)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
