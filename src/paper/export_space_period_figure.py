#!/usr/bin/env python3
"""
Export a paper-ready spatial comparison figure for a single period.

Layout:
  - 2 x 3 panels
  - top row: book-level mu by region
  - bottom row: book-level sigma by region
  - each column corresponds to one metric
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm

from src.paper.figure_export import default_dpi, parse_formats, save_figure_formats
from src.analysis.api import METRIC_SAMPLES, METRICS, build_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "src/paper/figures/results_space"

REGION_COLORS = {
    "两浙地区": "#AEB7C2",
    "福建地区": "#C44536",
    "江淮湖广": "#E58E26",
}

BOX_EDGE = "#7C8A9A"
GRID = "#E5E7EB"
TEXT = "#111827"
POINT_EDGE = "#475569"
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
        "regions": {
            "两浙地区": "两浙地区",
            "福建地区": "福建地区",
            "江淮湖广": "江淮湖广",
        },
        "titles": {
            "aspect_ratio": "长宽比（高/宽）",
            "face_ratio": "字面率",
            "ink_coverage": "灰度比（黑/白）",
        },
        "ylabel_mu": "书级均值 μ",
        "ylabel_sigma": "书内离散度 σ",
        "n_fmt": "(n={n})",
        "default_title": "{start}--{end} 区域比较",
    },
    "en": {
        "regions": {
            "两浙地区": "Liangzhe",
            "福建地区": "Fujian",
            "江淮湖广": "Jianghuai–Huguang",
        },
        "titles": {
            "aspect_ratio": "aspect ratio (height/width)",
            "face_ratio": "face ratio",
            "ink_coverage": "ink coverage (black/white)",
        },
        "ylabel_mu": "book-level mean μ",
        "ylabel_sigma": "within-book dispersion σ",
        "n_fmt": "(n = {n})",
        "default_title": "Regional comparison in {period}",
    },
}


@dataclass
class BookMetric:
    book: str
    title: str
    year: int
    region: str
    mu: float
    sigma: float
    n_values: int


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


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


def _parse_list_arg(value: Optional[str]) -> List[str]:
    if not value:
        return []
    raw = [v.strip() for v in str(value).replace("，", ",").split(",")]
    return [v for v in raw if v]


def _parse_period(value: str) -> Tuple[int, int]:
    s = value.replace("—", "-").replace("–", "-")
    start_s, end_s = [x.strip() for x in s.split("-", 1)]
    return int(start_s), int(end_s)


def _filter_rows(rows: List[Dict], start: int, end: int, regions: Sequence[str]) -> List[Dict]:
    region_set = set(regions)
    out: List[Dict] = []
    for r in rows:
        year = int(r.get("year") or 0)
        region = (r.get("region") or "").strip()
        if year < start or year >= end:
            continue
        if region_set and region not in region_set:
            continue
        out.append(r)
    out.sort(key=lambda r: (regions.index((r.get("region") or "").strip()), int(r.get("year") or 0), int(r.get("order") or 0)))
    return out


def _compute_metric_books(
    rows: List[Dict],
    char_means_by_book: Dict[str, Dict[str, Dict[str, float]]],
    metric: str,
    q: float,
    common_chars_enabled: bool,
) -> Tuple[List[BookMetric], Optional[int]]:
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

    out: List[BookMetric] = []
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
        out.append(
            BookMetric(
                book=book,
                title=r.get("title") or "",
                year=int(r.get("year") or 0),
                region=(r.get("region") or "").strip(),
                mu=float(np.mean(arr)),
                sigma=float(np.std(arr)),
                n_values=int(arr.size),
            )
        )
    return out, (len(common_chars) if common_chars_enabled else None)


def _jitter(n: int, width: float = 0.10) -> np.ndarray:
    if n <= 1:
        return np.asarray([0.0])
    return np.linspace(-width, width, n)


def _style_axes(ax) -> None:
    ax.set_facecolor("white")
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#9CA3AF")
    ax.spines["bottom"].set_color("#9CA3AF")
    ax.tick_params(colors=TEXT, labelsize=9)
    if FONT_REGULAR:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(FONT_REGULAR)


def _draw_panel(
    ax,
    books: List[BookMetric],
    regions: Sequence[str],
    value_attr: str,
    ylabel: str,
    title: str,
    show_xlabels: bool,
    labels: Dict,
) -> None:
    _style_axes(ax)
    groups: List[List[float]] = []
    positions = np.arange(1, len(regions) + 1)
    for reg in regions:
        vals = [getattr(b, value_attr) for b in books if b.region == reg and math.isfinite(getattr(b, value_attr))]
        groups.append(vals)

    bp = ax.boxplot(
        groups,
        positions=positions,
        widths=0.56,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color=TEXT, linewidth=1.4),
        whiskerprops=dict(color=BOX_EDGE, linewidth=1.0),
        capprops=dict(color=BOX_EDGE, linewidth=1.0),
        boxprops=dict(edgecolor=BOX_EDGE, linewidth=1.0),
    )
    for patch, reg in zip(bp["boxes"], regions):
        patch.set_facecolor(REGION_COLORS.get(reg, "#D9DEE7"))
        patch.set_alpha(0.35)

    for idx, reg in enumerate(regions, start=1):
        reg_books = [b for b in books if b.region == reg and math.isfinite(getattr(b, value_attr))]
        reg_books.sort(key=lambda b: (b.year, b.book))
        jit = _jitter(len(reg_books))
        for j, b in enumerate(reg_books):
            ax.scatter(
                idx + jit[j],
                getattr(b, value_attr),
                s=34,
                facecolors=REGION_COLORS.get(reg, "#D9DEE7"),
                edgecolors=POINT_EDGE,
                linewidths=0.8,
                zorder=3,
            )

    ax.set_xticks(positions)
    if show_xlabels:
        xticklabels = []
        counts = []
        for reg in regions:
            n_books = sum(1 for b in books if b.region == reg and math.isfinite(getattr(b, value_attr)))
            xticklabels.append(labels["regions"].get(reg, reg))
            counts.append(n_books)
        ax.set_xticklabels(xticklabels)
        for label in ax.get_xticklabels():
            label.set_fontproperties(FONT_REGULAR)
        for x, n_books in zip(positions, counts):
            ax.text(
                x,
                -0.11,
                labels["n_fmt"].format(n=n_books),
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=7.5,
                color="#6B7280",
                fontproperties=FONT_REGULAR,
            )
    else:
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax.set_ylabel(ylabel, color=TEXT, fontproperties=FONT_REGULAR, fontsize=10)
    ax.tick_params(axis="x", labelsize=8.5, colors=TEXT)
    ax.set_title(title, color=TEXT, fontproperties=FONT_SEMIBOLD, fontsize=11, pad=8)


def _write_summary(path: Path, period_label: str, common_counts: Dict[str, Optional[int]], by_metric: Dict[str, List[BookMetric]], regions: Sequence[str]) -> None:
    lines = [period_label, ""]
    for metric, books in by_metric.items():
        lines.append(f"[{metric}] common_chars={common_counts.get(metric)}")
        for reg in regions:
            vals_mu = [b.mu for b in books if b.region == reg and math.isfinite(b.mu)]
            vals_sigma = [b.sigma for b in books if b.region == reg and math.isfinite(b.sigma)]
            if not vals_mu:
                continue
            lines.append(
                f"  {reg}: n={len(vals_mu)} mu_mean={np.mean(vals_mu):.4f} mu_median={np.median(vals_mu):.4f} "
                f"sigma_mean={np.mean(vals_sigma):.4f} sigma_median={np.median(vals_sigma):.4f}"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a spatial comparison figure for one period.")
    ap.add_argument("--period", required=True, help="Period range, e.g. 1162-1208")
    ap.add_argument("--regions", required=True, help="Comma-separated regions")
    ap.add_argument("--q", type=float, default=0.05)
    ap.add_argument("--title", default="")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--name", default="space_period")
    ap.add_argument("--bundle-root", default="data/analysis")
    ap.add_argument("--metadata-path", default="data/metadata/books_metadata.csv")
    ap.add_argument("--lang", choices=("zh", "en"), default="zh")
    ap.add_argument("--formats", default="pdf")
    ap.add_argument("--dpi-kind", choices=("lineart", "grayscale", "colour", "color"), default="colour")
    args = ap.parse_args()

    _setup_fonts()
    labels = _labels(args.lang)

    start, end = _parse_period(args.period)
    regions = _parse_list_arg(args.regions)
    out_dir = _resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, _samples_by_book, char_means_by_book = build_dataset(
        _resolve_path(args.bundle_root),
        _resolve_path(args.metadata_path),
        threshold=160,
    )
    filtered_rows = _filter_rows(rows, start, end, regions)
    metrics = ["aspect_ratio", "face_ratio", "ink_coverage"]
    by_metric: Dict[str, List[BookMetric]] = {}
    common_counts: Dict[str, Optional[int]] = {}
    for metric in metrics:
        books, common_count = _compute_metric_books(
            filtered_rows,
            char_means_by_book,
            metric=metric,
            q=float(args.q),
            common_chars_enabled=True,
        )
        by_metric[metric] = books
        common_counts[metric] = common_count

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15.5, 8.8),
        gridspec_kw={"hspace": 0.12, "wspace": 0.18},
    )
    fig.patch.set_facecolor("white")
    if args.title.strip():
        title = args.title.strip()
    elif args.lang == "en":
        period_name = "the selected period"
        if (start, end) == (1162, 1208):
            period_name = "the middle Southern Song period"
        elif (start, end) == (1208, 1273):
            period_name = "the late Southern Song period"
        title = labels["default_title"].format(period=period_name)
    else:
        title = labels["default_title"].format(start=start, end=end)
    fig.suptitle(title, y=0.98, color=TEXT, fontsize=14, fontproperties=FONT_SEMIBOLD)
    for idx, metric in enumerate(metrics):
        _draw_panel(
            axes[0, idx], by_metric[metric], regions, "mu", labels["ylabel_mu"], labels["titles"][metric], show_xlabels=False, labels=labels
        )
        _draw_panel(
            axes[1, idx], by_metric[metric], regions, "sigma", labels["ylabel_sigma"], "", show_xlabels=True, labels=labels
        )
        if idx > 0:
            axes[0, idx].set_ylabel("")
            axes[1, idx].set_ylabel("")

    fig.subplots_adjust(top=0.92, left=0.07, right=0.98, bottom=0.10)

    saved_paths = save_figure_formats(fig, out_dir, args.name, parse_formats(args.formats), default_dpi(args.dpi_kind))
    plt.close(fig)
    for path in saved_paths:
        print(f"Saved: {path}")
    for metric in metrics:
        print(metric, "common=", common_counts.get(metric))
        for reg in regions:
            vals = [b.mu for b in by_metric[metric] if b.region == reg]
            print(" ", reg, "n=", len(vals), "mu_mean=", (float(np.mean(vals)) if vals else float('nan')))


if __name__ == "__main__":
    main()
