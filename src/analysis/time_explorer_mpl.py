#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interactive time explorer (matplotlib).

Single entry: compute metrics from bundle and show boxplots.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, TextBox, Button, CheckButtons
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METRICS: Dict[str, str] = {
    "aspect_ratio_mean": "长宽比（高/宽，均值）",
    "face_ratio_mean": "字面率（均值）",
    "ink_coverage_mean": "灰度比（黑/白）",
    "aspect_ratio_std": "长宽比（高/宽，离散度）",
    "face_ratio_std": "字面率（离散度）",
    "ink_coverage_std": "灰度比（离散度）",
    "Lb": "固定框 Lb",
    "clipping_rate": "裁切率（w>Lb 或 h>Lb）",
}

COLOR_KEYS = {
    "region": "区域",
    "style": "刻体倾向",
    "none": "无",
}

METRIC_SAMPLES = {
    "aspect_ratio_mean": "aspect_ratio",
    "face_ratio_mean": "face_ratio",
    "ink_coverage_mean": "ink_coverage",
    "aspect_ratio_std": "aspect_ratio_std",
    "face_ratio_std": "face_ratio_std",
    "ink_coverage_std": "ink_coverage_std",
}


@dataclass(frozen=True)
class BookMeta:
    book_id: str
    title: str
    year: Optional[int]
    number: Optional[int]
    region: str
    province: str
    place: str
    style: str


def percentile(values: List[int], p: float) -> int:
    if not values:
        return 0
    values_sorted = sorted(values)
    k = max(0, math.ceil((p / 100.0) * len(values_sorted)) - 1)
    return int(values_sorted[k])


def center_crop_fixed_box(gray: Image.Image, Lb: int) -> Image.Image:
    if Lb <= 0:
        return gray.convert("L")
    img = gray.convert("L")
    w, h = img.size
    cx, cy = w // 2, h // 2
    half = Lb // 2
    x0 = cx - half
    y0 = cy - half
    x1 = x0 + Lb
    y1 = y0 + Lb

    out = Image.new("L", (Lb, Lb), 255)
    sx0 = max(0, x0)
    sy0 = max(0, y0)
    sx1 = min(w, x1)
    sy1 = min(h, y1)
    if sx0 < sx1 and sy0 < sy1:
        sub = img.crop((sx0, sy0, sx1, sy1))
        dx = sx0 - x0
        dy = sy0 - y0
        out.paste(sub, (dx, dy))
    return out


def parse_book_prefix(book: str) -> Tuple[Optional[int], Optional[int]]:
    book = book.strip()
    if not book:
        return (None, None)
    parts = book.split("_", 2)
    if len(parts) < 2:
        return (None, None)
    order = None
    year = None
    try:
        order = int(parts[0])
    except Exception:
        order = None
    try:
        year = int(parts[1])
    except Exception:
        year = None
    return (order, year)


def read_books_metadata(path: Path) -> Dict[str, BookMeta]:
    out: Dict[str, BookMeta] = {}
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            book_id = (row.get("BookID") or "").strip()
            if not book_id:
                continue
            title = (row.get("宋刻本") or "").strip()
            year_raw = (row.get("年份") or "").strip()
            num_raw = (row.get("Number") or "").strip()
            year: Optional[int]
            try:
                year = int(year_raw) if year_raw else None
            except Exception:
                year = None
            number: Optional[int]
            try:
                number = int(num_raw) if num_raw else None
            except Exception:
                number = None
            out[book_id] = BookMeta(
                book_id=book_id,
                title=title,
                year=year,
                number=number,
                region=(row.get("区域划分") or "").strip(),
                province=(row.get("省份") or "").strip(),
                place=(row.get("地点") or "").strip(),
                style=(row.get("刻体倾向") or "").strip(),
            )
    return out


def load_manifest_books(bundle_root: Path) -> List[str]:
    manifest_path = bundle_root / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"找不到 bundle：{manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    books = sorted((manifest.get("books") or {}).keys())
    if not books:
        raise SystemExit("bundle 中没有任何 books。")
    return books


def iter_book_entries(bundle_root: Path, book: str) -> List[Dict]:
    entries_path = bundle_root / "books" / book / "entries.json"
    with open(entries_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("entries") or []


def rankdata(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values)
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(v.size, dtype=float)
    i = 0
    while i < v.size:
        j = i
        while j + 1 < v.size and v[order[j + 1]] == v[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = r
        i = j + 1
    return ranks


def spearman_rho(x: List[float], y: List[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    rx = rankdata(xv)
    ry = rankdata(yv)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def compute_book_data(bundle_root: Path, book: str, threshold: int) -> Tuple[Dict, Dict[str, List[float]]]:
    entries = iter_book_entries(bundle_root, book)

    instance_sizes: List[Tuple[str, int, int, Path]] = []
    long_sides: List[int] = []
    for e in entries:
        rel = e.get("image")
        if not rel:
            continue
        img_path = bundle_root / rel
        if not img_path.exists():
            continue
        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception:
            continue
        char = e.get("char") or ""
        instance_sizes.append((char, int(w), int(h), img_path))
        long_sides.append(max(int(w), int(h)))

    if not instance_sizes or not long_sides:
        metrics = {
            "instances": 0,
            "chars": 0,
            "Lb": 0,
            "clipping_rate": float("nan"),
            "aspect_ratio_mean": float("nan"),
            "aspect_ratio_std": float("nan"),
            "face_ratio_mean": float("nan"),
            "face_ratio_std": float("nan"),
            "ink_coverage_mean": float("nan"),
            "ink_coverage_std": float("nan"),
        }
        samples = {
            "aspect_ratio": [],
            "face_ratio": [],
            "ink_coverage": [],
            "aspect_ratio_std": [],
            "face_ratio_std": [],
            "ink_coverage_std": [],
        }
        return metrics, samples

    Lb = int(math.ceil(percentile(long_sides, 90) * 1.05))

    per_char: Dict[str, List[Tuple[float, float, float]]] = {}
    clipped = 0
    for char, w, h, img_path in instance_sizes:
        if not char:
            continue
        ar = float(h) / float(max(1, w))
        face_ratio = float((w + h) / 2.0) / float(max(1, Lb))
        if w > Lb or h > Lb:
            clipped += 1

        ink = float("nan")
        try:
            with Image.open(img_path) as im:
                gray = im.convert("L")
                fixed = center_crop_fixed_box(gray, Lb)
                arr = np.asarray(fixed, dtype=np.uint8)
                black = int(np.sum(arr < np.uint8(threshold)))
                white = int(Lb * Lb - black)
                if white > 0:
                    ink = float(black) / float(white)
        except Exception:
            pass

        per_char.setdefault(char, []).append((ar, face_ratio, ink))

    char_means: List[Tuple[float, float, float]] = []
    char_stds: List[Tuple[float, float, float]] = []
    for rows in per_char.values():
        ars = [r[0] for r in rows if math.isfinite(r[0])]
        frs = [r[1] for r in rows if math.isfinite(r[1])]
        inks = [r[2] for r in rows if math.isfinite(r[2])]
        if not ars or not frs:
            continue
        char_means.append((
            float(np.mean(np.asarray(ars, dtype=float))),
            float(np.mean(np.asarray(frs, dtype=float))),
            float(np.mean(np.asarray(inks, dtype=float))) if inks else float("nan"),
        ))
        ar_std = float(np.std(np.asarray(ars, dtype=float), ddof=0)) if len(ars) >= 2 else float("nan")
        fr_std = float(np.std(np.asarray(frs, dtype=float), ddof=0)) if len(frs) >= 2 else float("nan")
        ic_std = float(np.std(np.asarray(inks, dtype=float), ddof=0)) if len(inks) >= 2 else float("nan")
        char_stds.append((ar_std, fr_std, ic_std))

    samples = {
        "aspect_ratio": [x[0] for x in char_means if math.isfinite(x[0])],
        "face_ratio": [x[1] for x in char_means if math.isfinite(x[1])],
        "ink_coverage": [x[2] for x in char_means if math.isfinite(x[2])],
        "aspect_ratio_std": [x[0] for x in char_stds if math.isfinite(x[0])],
        "face_ratio_std": [x[1] for x in char_stds if math.isfinite(x[1])],
        "ink_coverage_std": [x[2] for x in char_stds if math.isfinite(x[2])],
    }

    def mean_or_nan(values: List[float]) -> float:
        if not values:
            return float("nan")
        return float(np.mean(np.asarray(values, dtype=float)))

    metrics = {
        "instances": len(instance_sizes),
        "chars": len(char_means),
        "Lb": Lb,
        "clipping_rate": float(clipped) / float(max(1, len(instance_sizes))),
        "aspect_ratio_mean": mean_or_nan(samples["aspect_ratio"]),
        "aspect_ratio_std": mean_or_nan(samples["aspect_ratio_std"]),
        "face_ratio_mean": mean_or_nan(samples["face_ratio"]),
        "face_ratio_std": mean_or_nan(samples["face_ratio_std"]),
        "ink_coverage_mean": mean_or_nan(samples["ink_coverage"]),
        "ink_coverage_std": mean_or_nan(samples["ink_coverage_std"]),
    }
    return metrics, samples


def build_dataset(bundle_root: Path, metadata_path: Path, threshold: int) -> Tuple[List[Dict], Dict[str, Dict[str, List[float]]]]:
    books = load_manifest_books(bundle_root)
    meta_map = read_books_metadata(metadata_path)

    rows: List[Dict] = []
    samples_by_book: Dict[str, Dict[str, List[float]]] = {}

    for book in books:
        metrics, samples = compute_book_data(bundle_root, book, threshold=threshold)
        meta = meta_map.get(book)
        order = meta.number if meta and meta.number is not None else None
        year = meta.year if meta and meta.year is not None else None
        if order is None or year is None:
            pref_order, pref_year = parse_book_prefix(book)
            if order is None:
                order = pref_order
            if year is None:
                year = pref_year

        rows.append({
            "book": book,
            "title": meta.title if meta else "",
            "year": year,
            "order": order,
            "region": meta.region if meta else "",
            "province": meta.province if meta else "",
            "place": meta.place if meta else "",
            "style": meta.style if meta else "",
            **metrics,
        })
        samples_by_book[book] = samples

    return rows, samples_by_book


def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive time explorer (matplotlib).")
    ap.add_argument("--bundle", type=str, default="data/analysis", help="Bundle root folder (default: data/analysis)")
    ap.add_argument("--metadata", type=str, default="data/metadata/books_metadata.csv", help="Books metadata CSV")
    ap.add_argument("--threshold", type=int, default=160, help="Black threshold (default: 160)")
    args = ap.parse_args()

    bundle_root = PROJECT_ROOT / args.bundle
    metadata_path = PROJECT_ROOT / args.metadata
    rows, samples_by_book = build_dataset(bundle_root, metadata_path, threshold=int(args.threshold))
    if not rows:
        raise SystemExit("没有可用数据。")

    state = {
        "metric": "aspect_ratio_mean",
        "color_by": "region",
        "search": "",
    }

    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Hiragino Sans GB",
        "Heiti SC",
        "Arial Unicode MS",
        "Noto Sans CJK SC",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(18.0, 9.2))
    ax = fig.add_axes([0.36, 0.12, 0.62, 0.82])

    ax_metric = fig.add_axes([0.02, 0.62, 0.30, 0.32], facecolor="#f8fafc")
    ax_color = fig.add_axes([0.02, 0.52, 0.30, 0.08], facecolor="#f8fafc")
    ax_region = fig.add_axes([0.02, 0.32, 0.30, 0.18], facecolor="#f8fafc")
    ax_style = fig.add_axes([0.02, 0.16, 0.30, 0.14], facecolor="#f8fafc")
    ax_place_toggle = fig.add_axes([0.02, 0.10, 0.30, 0.06], facecolor="#f8fafc")
    ax_search = fig.add_axes([0.36, 0.02, 0.26, 0.05])
    ax_reset = fig.add_axes([0.65, 0.02, 0.08, 0.05])

    metric_radio = RadioButtons(ax_metric, list(METRICS.keys()), active=0)
    color_radio = RadioButtons(ax_color, list(COLOR_KEYS.keys()), active=0)
    search_box = TextBox(ax_search, "书名/ID", initial="")
    reset_btn = Button(ax_reset, "重置")

    regions = sorted({(r.get("region") or "（空）") for r in rows})
    styles = sorted({(r.get("style") or "（空）") for r in rows})
    region_check = CheckButtons(ax_region, regions, [True] * len(regions))
    style_check = CheckButtons(ax_style, styles, [True] * len(styles))
    place_toggle = CheckButtons(ax_place_toggle, ["区分书局"], [False])

    for cb in (region_check, style_check, place_toggle):
        for lbl in cb.labels:
            lbl.set_fontsize(8)

    def selected_labels(labels: List[str], actives: List[bool]) -> set:
        return {label for label, on in zip(labels, actives) if on}

    def filter_rows() -> List[Dict]:
        out: List[Dict] = []
        selected_regions = selected_labels(regions, region_check.get_status())
        selected_styles = selected_labels(styles, style_check.get_status())
        for r in rows:
            if r["year"] is None:
                continue
            v = r.get(state["metric"])
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            if (r.get("region") or "（空）") not in selected_regions:
                continue
            if (r.get("style") or "（空）") not in selected_styles:
                continue
            if state["search"]:
                s = (r.get("book", "") + " " + r.get("title", "")).lower()
                if state["search"].lower() not in s:
                    continue
            out.append(r)
        return out

    def update_plot(_=None) -> None:
        ax.clear()
        current = filter_rows()
        if not current:
            ax.set_title("无符合条件的数据")
            fig.canvas.draw_idle()
            return

        current = sorted(
            current,
            key=lambda r: (
                r["year"] if r["year"] is not None else 9999,
                r.get("order") if r.get("order") is not None else 9999,
                r.get("book") or "",
            ),
        )

        metric = state["metric"]
        sample_key = METRIC_SAMPLES.get(metric)

        plot_rows: List[Dict] = []
        plot_samples: List[np.ndarray] = []
        plot_values: List[float] = []
        for r in current:
            book = r.get("book")
            if sample_key:
                vals = samples_by_book.get(book, {}).get(sample_key, []) if book else []
                if not vals:
                    continue
                plot_rows.append(r)
                plot_samples.append(np.asarray(vals, dtype=float))
            else:
                y = r.get(metric)
                if y is None or (isinstance(y, float) and math.isnan(y)):
                    continue
                plot_rows.append(r)
                plot_values.append(float(y))

        if not plot_rows:
            ax.set_title("无可绘制数据")
            fig.canvas.draw_idle()
            return

        color_key = "place" if place_toggle.get_status()[0] else state["color_by"]
        if color_key == "none":
            colors = ["#111827"] * len(plot_rows)
            legend_items = []
        else:
            groups = sorted({r.get(color_key, "") or "（空）" for r in plot_rows})
            cmap = plt.get_cmap("tab20")
            color_map = {g: cmap(i % 20) for i, g in enumerate(groups)}
            colors = [color_map[(r.get(color_key, "") or "（空）")] for r in plot_rows]
            legend_items = [(g, color_map[g]) for g in groups]

        means: List[float] = []
        if sample_key:
            positions = list(range(len(plot_rows)))
            box = ax.boxplot(
                plot_samples,
                positions=positions,
                widths=0.55,
                patch_artist=True,
                showmeans=True,
                meanline=False,
                whis=1.5,
                showfliers=True,
            )
            for i, patch in enumerate(box["boxes"]):
                patch.set_facecolor(colors[i])
                patch.set_edgecolor(colors[i])
                patch.set_alpha(0.25)
                patch.set_linewidth(1.2)
            for i, med in enumerate(box["medians"]):
                med.set_color(colors[i])
                med.set_linewidth(1.6)
            for i, mean in enumerate(box["means"]):
                mean.set_marker("D")
                mean.set_markerfacecolor(colors[i])
                mean.set_markeredgecolor("white")
                mean.set_markersize(5.5)
            for line in box["whiskers"]:
                line.set_linewidth(1.0)
                line.set_color("#6b7280")
                line.set_alpha(0.7)
            for line in box["caps"]:
                line.set_linewidth(1.0)
                line.set_color("#6b7280")
                line.set_alpha(0.7)
            for flier in box["fliers"]:
                flier.set_marker("o")
                flier.set_markersize(2.6)
                flier.set_alpha(0.35)
                flier.set_markerfacecolor("#6b7280")
                flier.set_markeredgecolor("none")

            for vals in plot_samples:
                means.append(float(np.mean(vals)) if vals.size else float("nan"))
        else:
            for idx, y in enumerate(plot_values):
                ax.scatter([float(idx)], [y], c=[colors[idx]], s=36, alpha=0.85, edgecolors="none")
                means.append(y)

        ax.set_xlabel("书籍（按时间顺序排列）")
        ax.set_ylabel(METRICS[state["metric"]])
        ax.grid(True, alpha=0.25)

        valid_pairs = [(i, y) for i, y in enumerate(means) if math.isfinite(y)]
        if len(valid_pairs) >= 2:
            xs = [float(i) for i, _ in valid_pairs]
            ys = [float(y) for _, y in valid_pairs]
            rho = spearman_rho(xs, ys)
            rho_text = f"ρ={rho:.3f}"
        else:
            rho_text = "ρ=NA"
        ax.set_title(f"{METRICS[state['metric']]}（n={len(plot_rows)}，{rho_text}）")

        labels = [r.get("book") or r.get("title") or "" for r in plot_rows]
        step = max(1, len(labels) // 16)
        ticks = list(range(0, len(labels), step))
        ax.set_xticks(ticks)
        ax.set_xticklabels([labels[i] for i in ticks], rotation=45, ha="right", fontsize=8)

        if legend_items:
            handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=6) for _, c in legend_items]
            labels = [g for g, _ in legend_items]
            legend_title = "书局" if color_key == "place" else COLOR_KEYS.get(color_key, "")
            ax.legend(handles, labels, loc="upper left", fontsize=8, frameon=False, title=legend_title)

        fig.canvas.draw_idle()

    def on_metric(label: str) -> None:
        state["metric"] = label
        update_plot()

    def on_color(label: str) -> None:
        state["color_by"] = label
        update_plot()

    def on_search(text: str) -> None:
        state["search"] = text.strip()
        update_plot()

    def on_reset(_event) -> None:
        state["metric"] = "aspect_ratio_mean"
        state["color_by"] = "region"
        state["search"] = ""
        metric_radio.set_active(0)
        color_radio.set_active(0)
        for i in range(len(regions)):
            if not region_check.get_status()[i]:
                region_check.set_active(i)
        for i in range(len(styles)):
            if not style_check.get_status()[i]:
                style_check.set_active(i)
        if place_toggle.get_status()[0]:
            place_toggle.set_active(0)
        search_box.set_val("")
        update_plot()

    metric_radio.on_clicked(on_metric)
    color_radio.on_clicked(on_color)
    search_box.on_submit(on_search)
    reset_btn.on_clicked(on_reset)
    region_check.on_clicked(lambda _label: update_plot())
    style_check.on_clicked(lambda _label: update_plot())
    place_toggle.on_clicked(lambda _label: update_plot())

    update_plot()
    plt.show()


if __name__ == "__main__":
    main()
