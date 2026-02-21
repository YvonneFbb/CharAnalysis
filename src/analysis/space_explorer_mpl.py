#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Space explorer (matplotlib/CLI).

Goal: compare groups (e.g. regions) within time periods.

This is intentionally simpler than time_explorer_mpl.py:
- focus on period slicing + group comparisons
- keep output "AI-friendly" (JSON by default)
"""

from __future__ import annotations

import argparse
import csv
import json
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, CheckButtons, Slider
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]

METRICS: Dict[str, str] = {
    "aspect_ratio": "长宽比（高/宽）",
    "face_ratio": "字面率",
    "ink_coverage": "灰度比（黑/白）",
    "Lb": "固定框 Lb（每书标量）",
    "clipping_rate": "裁切率（每书标量）",
}

METRIC_SAMPLES = {
    "aspect_ratio": "aspect_ratio",
    "face_ratio": "face_ratio",
    "ink_coverage": "ink_coverage",
}


@dataclass(frozen=True)
class Period:
    label: str
    start: int
    end: int
    end_inclusive: bool

    def contains(self, year: int) -> bool:
        if year < self.start:
            return False
        if self.end_inclusive:
            return year <= self.end
        return year < self.end


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
    style_tags: List[str]


def split_style_tags(style_raw: str) -> List[str]:
    if not style_raw:
        return ["（空）"]
    cleaned = (
        style_raw.replace("，", ",")
        .replace("、", ",")
        .replace("/", ",")
        .replace(";", ",")
        .replace("；", ",")
    )
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    return parts or ["（空）"]


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
            style_raw = (row.get("刻体倾向") or "").strip()
            out[book_id] = BookMeta(
                book_id=book_id,
                title=title,
                year=year,
                number=number,
                region=(row.get("区域划分") or "").strip(),
                province=(row.get("省份") or "").strip(),
                place=(row.get("地点") or "").strip(),
                style=style_raw,
                style_tags=split_style_tags(style_raw),
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


def parse_list_arg(value: str) -> List[str]:
    if not value:
        return []
    parts = [v.strip() for v in value.replace("，", ",").split(",")]
    return [p for p in parts if p]


def parse_styles_by_region(value: str) -> Dict[str, set[str]]:
    """
    Format:
      "两浙地区:近欧型|扁方欧体;福建地区:近欧型"
    Separators:
      - region blocks: ';' or '；'
      - region/styles: ':'
      - styles inside region: '|' or ',' or '，'
    """
    out: Dict[str, set[str]] = {}
    if not value:
        return out
    blocks = [b.strip() for b in value.replace("；", ";").split(";") if b.strip()]
    for blk in blocks:
        if ":" not in blk:
            continue
        reg, raw = blk.split(":", 1)
        region = reg.strip()
        if not region:
            continue
        styles = [
            s.strip()
            for s in raw.replace("，", ",").replace("|", ",").split(",")
            if s.strip()
        ]
        out[region] = set(styles)
    return out


def _is_finite_number(value: object) -> bool:
    try:
        return math.isfinite(float(value))  # type: ignore[arg-type]
    except Exception:
        return False


def _json_safe(value: object):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _fmt_num(value: object, digits: int = 4, empty: str = "-") -> str:
    if not _is_finite_number(value):
        return empty
    return f"{float(value):.{digits}f}"


def _fmt_csv_num(value: object, digits: int = 6) -> str:
    if not _is_finite_number(value):
        return ""
    return f"{float(value):.{digits}f}"


def resolve_styles_by_region(
    regions: List[str],
    style_tags: List[str],
    selected_styles_default: set[str],
    styles_by_region_override: Dict[str, set[str]],
) -> Dict[str, set[str]]:
    base = set(selected_styles_default) if selected_styles_default else set(style_tags)
    out = {r: set(base) for r in regions}
    for region, styles in styles_by_region_override.items():
        out[region] = set(styles)
    return out


def resolve_focus_period(periods: List["Period"], token: str) -> int:
    if not periods:
        return 0
    raw = (token or "").strip()
    if not raw:
        return 0
    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(periods):
            return idx
    for i, p in enumerate(periods):
        if raw == p.label:
            return i
    raise SystemExit(f"无效 focus-period: {token}，可用 1..{len(periods)} 或标签 {[p.label for p in periods]}")


def parse_periods(value: str) -> List[Period]:
    """
    Format:
      "1127-1162,1162-1208,1208-1273"
    Semantics:
      - first N-1 periods are [start, end)
      - last period is [start, end] (end-inclusive)
    """
    raw = [p.strip() for p in (value or "").replace("，", ",").split(",") if p.strip()]
    out: List[Period] = []
    for i, part in enumerate(raw):
        if "-" not in part:
            raise SystemExit(f"无效 periods 段：{part}，应为 start-end")
        s, e = part.split("-", 1)
        try:
            start = int(s.strip())
            end = int(e.strip())
        except Exception:
            raise SystemExit(f"无效 periods 段：{part}，start/end 必须为整数")
        if end < start:
            raise SystemExit(f"无效 periods 段：{part}，end < start")
        end_inclusive = (i == len(raw) - 1)
        label = "早期" if i == 0 else ("中期" if i == 1 else ("后期" if i == 2 else f"P{i+1}"))
        out.append(Period(label=label, start=start, end=end, end_inclusive=end_inclusive))
    if not out:
        raise SystemExit("periods 不能为空，例如：1127-1162,1162-1208,1208-1273")
    return out


def assign_period(periods: List[Period], year: int) -> Optional[Period]:
    for p in periods:
        if p.contains(year):
            return p
    return None


def build_dataset(
    bundle_root: Path, metadata_path: Path, threshold: int
) -> Tuple[List[Dict], Dict[str, Dict[str, Dict[str, float]]], Dict[str, int]]:
    """
    Returns:
      rows: per-book rows with metadata + scalar metrics
      char_means_by_book: book -> char -> {aspect_ratio, face_ratio, ink_coverage}
      Lb_by_book: book -> Lb (for reference)
    """
    books = load_manifest_books(bundle_root)
    meta_map = read_books_metadata(metadata_path)

    rows: List[Dict] = []
    char_means_by_book: Dict[str, Dict[str, Dict[str, float]]] = {}
    Lb_by_book: Dict[str, int] = {}

    for book in books:
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
                "aspect_ratio": float("nan"),
                "face_ratio": float("nan"),
                "ink_coverage": float("nan"),
            }
            char_means_by_book[book] = {}
            Lb_by_book[book] = 0
        else:
            Lb = int(math.ceil(percentile(long_sides, 90) * 1.05))
            Lb_by_book[book] = Lb
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

            char_means_map: Dict[str, Dict[str, float]] = {}
            char_means: List[Tuple[float, float, float]] = []
            for char, items in per_char.items():
                ars = [v[0] for v in items if math.isfinite(v[0])]
                frs = [v[1] for v in items if math.isfinite(v[1])]
                inks = [v[2] for v in items if math.isfinite(v[2])]
                if not ars or not frs:
                    continue
                ar_mean = float(np.mean(np.asarray(ars, dtype=float)))
                fr_mean = float(np.mean(np.asarray(frs, dtype=float)))
                ic_mean = float(np.mean(np.asarray(inks, dtype=float))) if inks else float("nan")
                char_means.append((ar_mean, fr_mean, ic_mean))
                char_means_map[char] = {
                    "aspect_ratio": ar_mean,
                    "face_ratio": fr_mean,
                    "ink_coverage": ic_mean,
                }
            char_means_by_book[book] = char_means_map

            def mean_or_nan(values: List[float]) -> float:
                if not values:
                    return float("nan")
                return float(np.mean(np.asarray(values, dtype=float)))

            aspect_vals = [x[0] for x in char_means if math.isfinite(x[0])]
            face_vals = [x[1] for x in char_means if math.isfinite(x[1])]
            ink_vals = [x[2] for x in char_means if math.isfinite(x[2])]

            metrics = {
                "instances": len(instance_sizes),
                "chars": len(char_means),
                "Lb": Lb,
                "clipping_rate": float(clipped) / float(max(1, len(instance_sizes))),
                "aspect_ratio": mean_or_nan(aspect_vals),
                "face_ratio": mean_or_nan(face_vals),
                "ink_coverage": mean_or_nan([v for v in ink_vals if math.isfinite(v)]),
            }

        meta = meta_map.get(book)
        year = meta.year if meta else None
        number = meta.number if meta else None
        rows.append(
            {
                "book": book,
                "title": meta.title if meta else "",
                "year": year,
                "order": number,
                "region": meta.region if meta else "",
                "province": meta.province if meta else "",
                "place": meta.place if meta else "",
                "style": meta.style if meta else "",
                "style_tags": meta.style_tags if meta else ["（空）"],
                **metrics,
            }
        )

    return rows, char_means_by_book, Lb_by_book


def _stable_seed(tag: str, values: List[float]) -> int:
    h = hashlib.blake2b(digest_size=8)
    h.update(tag.encode("utf-8"))
    h.update(b"\0")
    h.update(np.asarray(values, dtype=float).tobytes())
    return int.from_bytes(h.digest(), "little", signed=False) & 0xFFFFFFFF


def apply_q_trim(values: List[float], q: float) -> List[float]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if q <= 0.0 or len(vals) < 4:
        return vals
    lo = float(np.quantile(vals, q))
    hi = float(np.quantile(vals, 1.0 - q))
    return [v for v in vals if lo <= v <= hi]


def compute_book_mu_a_from_chars(
    char_map: Dict[str, Dict[str, float]],
    sample_key: str,
    common_chars: Optional[set[str]],
    q: float,
) -> Tuple[float, float, List[str]]:
    vals = [(c, float(v.get(sample_key, float("nan")))) for c, v in char_map.items()]
    vals = [(c, v) for c, v in vals if math.isfinite(v)]

    used_chars: List[str] = []
    if common_chars is not None:
        vals = [(c, v) for c, v in vals if c in common_chars]

    values_only = [v for _, v in vals]
    values_only = apply_q_trim(values_only, q)
    values_set = set(values_only)
    # keep mapping consistent with trim
    vals = [(c, v) for c, v in vals if v in values_set]

    used_chars = [c for c, _ in vals]
    vv = [v for _, v in vals]
    if not vv:
        return (float("nan"), float("nan"), [])
    mu = float(np.mean(np.asarray(vv, dtype=float)))
    a = float(np.std(np.asarray(vv, dtype=float)))
    return (mu, a, used_chars)


def filter_rows_by_region_styles(
    rows: List[Dict],
    selected_regions: set[str],
    styles_by_region: Dict[str, set[str]],
) -> List[Dict]:
    out: List[Dict] = []
    for r in rows:
        year = r.get("year")
        if year is None:
            continue
        region_value = (r.get("region") or "（空）")
        if selected_regions and region_value not in selected_regions:
            continue
        tags = set(r.get("style_tags") or ["（空）"])
        style_sel = set(styles_by_region.get(region_value, set()))
        if (not style_sel) or tags.isdisjoint(style_sel):
            continue
        out.append(r)
    return out


def collect_books_for_period(
    rows: List[Dict],
    char_means_by_book: Dict[str, Dict[str, Dict[str, float]]],
    period: "Period",
    metric: str,
    q: float,
    common_chars_enabled: bool,
) -> Tuple[List[Dict], Optional[int]]:
    sample_key = METRIC_SAMPLES.get(metric)
    period_rows = [r for r in rows if period.contains(int(r.get("year") or 0))]
    if not period_rows:
        return [], None

    common_chars_set: Optional[set[str]] = None
    if sample_key and common_chars_enabled:
        for r in period_rows:
            book = r.get("book") or ""
            char_map = char_means_by_book.get(book, {})
            keys = {
                c for c, v in char_map.items()
                if math.isfinite(float(v.get(sample_key, float("nan"))))
            }
            if common_chars_set is None:
                common_chars_set = keys
            else:
                common_chars_set &= keys
        common_chars_set = common_chars_set or set()

    out: List[Dict] = []
    scalar_metric = sample_key is None
    for r in period_rows:
        book = r.get("book") or ""
        region_val = (r.get("region") or "（空）")
        vals: List[float] = []
        if sample_key:
            char_map = char_means_by_book.get(book, {})
            tmp = [
                (c, float(v.get(sample_key, float("nan"))))
                for c, v in char_map.items()
                if math.isfinite(float(v.get(sample_key, float("nan"))))
            ]
            if common_chars_set is not None:
                tmp = [(c, v) for c, v in tmp if c in common_chars_set]
            vv = [v for _, v in tmp]
            vals = apply_q_trim(vv, q)
        else:
            scalar = r.get(metric)
            if scalar is not None and math.isfinite(float(scalar)):
                vals = [float(scalar)]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        a_val: Optional[float]
        if scalar_metric:
            a_val = None
        else:
            a_val = float(np.std(arr))
        out.append(
            {
                "book": book,
                "title": r.get("title") or "",
                "year": r.get("year"),
                "order": r.get("order"),
                "region": region_val,
                "style_tags": r.get("style_tags") or [],
                "values": vals,
                "mu": float(np.mean(arr)),
                "a": a_val,
                "n_values": len(vals),
            }
        )
    common_count = len(common_chars_set) if (sample_key and common_chars_enabled and common_chars_set is not None) else None
    return out, common_count


def cluster_books_by_region(
    books_data: List[Dict],
    region_order: List[str],
) -> Tuple[List[Dict], List[float], Dict[str, List[float]]]:
    region_rank = {r: i for i, r in enumerate(region_order)}
    ordered = sorted(
        books_data,
        key=lambda b: (
            region_rank.get(b.get("region", "（空）"), 999),
            b.get("year") if b.get("year") is not None else 9999,
            b.get("order") if b.get("order") is not None else 9999,
            b.get("book") or "",
        ),
    )
    positions: List[float] = []
    spans: Dict[str, List[float]] = {}
    pos = 1.0
    prev_region: Optional[str] = None
    for b in ordered:
        region_val = b.get("region") or "（空）"
        if prev_region is not None and region_val != prev_region:
            pos += 1.0
        positions.append(pos)
        spans.setdefault(region_val, []).append(pos)
        prev_region = region_val
        pos += 1.0
    return ordered, positions, spans


def draw_clustered_books_plot(
    ax,
    books_data: List[Dict],
    region_order: List[str],
    metric_label: str,
    title: str,
    q: float = 0.05,
    common_count: Optional[int] = None,
    annotate_mu_sigma: bool = False,
    region_color_map: Optional[Dict[str, object]] = None,
    show_xlabels: bool = True,
) -> Tuple[List[Dict], List[float], Dict[str, List[float]]]:
    if not books_data:
        ax.text(0.5, 0.5, "无可用数据", transform=ax.transAxes, ha="center", va="center", color="#111827")
        ax.set_title(title)
        ax.set_ylabel(metric_label)
        return [], [], {}
    # Remove previous right-side a-axis if this panel is re-rendered.
    prev_a_axis = getattr(ax, "_a_axis", None)
    if prev_a_axis is not None:
        try:
            prev_a_axis.remove()
        except Exception:
            pass
    ax._a_axis = None
    ax.clear()

    ordered, positions, region_spans = cluster_books_by_region(books_data, region_order)
    labels = [b["book"] for b in ordered]
    data = [list(b["values"]) for b in ordered]
    book_regions = [b["region"] for b in ordered]
    mus = [float(b["mu"]) for b in ordered]
    sigmas = [float(b["a"]) if _is_finite_number(b.get("a")) else float("nan") for b in ordered]

    uniq_regions = [r for r in region_order if r in region_spans] + [r for r in sorted(region_spans.keys()) if r not in region_order]
    if region_color_map is None:
        cmap = plt.get_cmap("tab10")
        color_map = {reg: cmap(i % 10) for i, reg in enumerate(region_order)}
    else:
        color_map = dict(region_color_map)

    whis = (100.0 * q, 100.0 * (1.0 - q))
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.46,
        tick_labels=labels,
        showfliers=True,
        patch_artist=True,
        showmeans=True,
        meanline=False,
        whis=whis,
    )
    for patch, reg in zip(bp["boxes"], book_regions):
        color = color_map.get(reg, "#cbd5e1")
        patch.set_facecolor(color)
        patch.set_alpha(0.25)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.2)
    for med, reg in zip(bp["medians"], book_regions):
        med.set_color(color_map.get(reg, "#334155"))
        med.set_linewidth(1.6)
    for mean, reg in zip(bp["means"], book_regions):
        mean.set_marker("D")
        mean.set_markerfacecolor(color_map.get(reg, "#334155"))
        mean.set_markeredgecolor("white")
        mean.set_markersize(5.5)
        mean.set_alpha(0.95)
    for line in bp["whiskers"]:
        line.set_linewidth(1.0)
        line.set_color("#111827")
        line.set_alpha(0.45)
    for line in bp["caps"]:
        line.set_linewidth(1.0)
        line.set_color("#111827")
        line.set_alpha(0.45)
    for flier in bp["fliers"]:
        flier.set_marker("o")
        flier.set_markersize(2.6)
        flier.set_alpha(0.25)
        flier.set_markerfacecolor("#111827")
        flier.set_markeredgecolor("none")

    for idx, vals in enumerate(data):
        rng = np.random.default_rng(_stable_seed(f"space-book-{labels[idx]}", vals))
        jitter = rng.normal(0.0, 0.04, len(vals))
        xs = np.full(len(vals), float(positions[idx])) + jitter
        reg = book_regions[idx]
        ax.scatter(
            xs,
            vals,
            s=6,
            alpha=0.20,
            color=color_map.get(reg, "#64748b"),
            edgecolors="none",
        )

    if annotate_mu_sigma:
        for idx, mu in enumerate(mus):
            ax.annotate(
                f"μ={mu:.2f}",
                xy=(positions[idx], mu),
                xytext=(5, 7),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=7,
                color="#334155",
                bbox=dict(facecolor="white", alpha=0.70, edgecolor="none", boxstyle="round,pad=0.15"),
                zorder=5,
            )

        std_values = [sd for sd in sigmas if math.isfinite(sd)]
        if std_values:
            u_values = [v for vals in data for v in vals if math.isfinite(v)]
            u_min = min(u_values) if u_values else 0.0
            u_max = max(u_values) if u_values else 1.0
            u_span = max(0.2, u_max - u_min)
            a_min = min(std_values)
            a_max = max(std_values)
            a_span = max(0.02, a_max - a_min)
            a_low = max(0.0, a_min - a_span * 0.30)
            a_high = a_max + a_span * 0.35

            # Map a-values into a compact band just below u-values (same axis, separated heights).
            band_height = max(0.08, u_span * 0.16)
            band_gap = max(0.02, u_span * 0.04)
            band_top = u_min - band_gap
            band_bottom = band_top - band_height
            y_high = u_max + u_span * 0.10
            ax.set_ylim(band_bottom - max(0.01, u_span * 0.02), y_high)

            def _map_a_to_band(v: float) -> float:
                if not math.isfinite(v):
                    return float("nan")
                if a_high <= a_low:
                    return (band_bottom + band_top) * 0.5
                ratio = (v - a_low) / (a_high - a_low)
                ratio = max(0.0, min(1.0, ratio))
                return band_bottom + ratio * (band_top - band_bottom)

            for idx, sd in enumerate(sigmas):
                if not math.isfinite(sd):
                    continue
                reg = book_regions[idx]
                c = color_map.get(reg, "#64748b")
                y_a = _map_a_to_band(sd)
                ax.scatter(
                    [positions[idx]],
                    [y_a],
                    marker="^",
                    s=28,
                    color=c,
                    edgecolors="white",
                    linewidths=0.5,
                    alpha=0.90,
                    zorder=4,
                )
                ax.annotate(
                    f"a={sd:.2f}",
                    xy=(positions[idx], y_a),
                    xytext=(4, 4),
                    textcoords="offset points",
                    ha="left",
                    va="bottom",
                    fontsize=6.5,
                    color="#475569",
                    clip_on=False,
                    zorder=5,
                )

            # Right-side a axis (same y space, but ticks only for a band).
            ax_a = ax.twinx()
            ax_a.set_ylim(ax.get_ylim())
            ax_a.spines["left"].set_visible(False)
            ax_a.spines["top"].set_visible(False)
            ax_a.spines["bottom"].set_visible(False)
            ax_a.spines["right"].set_color("#111827")
            ax_a.tick_params(axis="y", labelsize=8, colors="#111827")
            ax_a.yaxis.set_label_position("right")
            ax_a.set_ylabel("a（字间差异）", fontsize=9, color="#111827")
            a_ticks = np.linspace(a_low, a_high, num=4)
            ax_a.set_yticks([_map_a_to_band(float(t)) for t in a_ticks])
            ax_a.set_yticklabels([f"{t:.2f}" for t in a_ticks])
            ax._a_axis = ax_a

    for reg, span in region_spans.items():
        x0, x1 = min(span), max(span)
        ax.text((x0 + x1) / 2.0, 0.99, reg, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9, color="#1f2937")
        ax.axvline(x=x1 + 0.5, color="#e5e7eb", linewidth=1.0, zorder=0)

    handles = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=color_map[r], markersize=8, alpha=0.6)
        for r in uniq_regions
    ]
    ax.legend(handles, uniq_regions, loc="upper left", frameon=False, title="区域", fontsize=9, title_fontsize=10)
    cc_text = f" | common={common_count}" if common_count is not None else ""
    ax.set_title(f"{title}{cc_text}")
    ax.set_ylabel(metric_label)
    ax.set_xticks(positions)
    if show_xlabels:
        ax.set_xticklabels(labels, rotation=90)
        ax.set_xlabel("书籍（按地区分组）")
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", which="both", length=0)
    ax.grid(True, axis="y", alpha=0.20)
    ax.set_xlim(min(positions) - 0.5, max(positions) + 0.5)
    return ordered, positions, region_spans


def compute_period_payloads(
    candidate_rows: List[Dict],
    char_means_by_book: Dict[str, Dict[str, Dict[str, float]]],
    periods: List[Period],
    metric: str,
    q: float,
    common_chars_enabled: bool,
) -> List[Dict]:
    sample_key = METRIC_SAMPLES.get(metric)
    out_payloads: List[Dict] = []

    for p in periods:
        books_in_period = [r for r in candidate_rows if p.contains(int(r.get("year") or 0))]

        common_chars_set: Optional[set[str]] = None
        if common_chars_enabled and sample_key:
            for r in books_in_period:
                book = r.get("book") or ""
                char_map = char_means_by_book.get(book, {})
                vals = [
                    (c, float(v.get(sample_key, float("nan"))))
                    for c, v in char_map.items()
                    if math.isfinite(float(v.get(sample_key, float("nan"))))
                ]
                keys = {c for c, _ in vals}
                if common_chars_set is None:
                    common_chars_set = keys
                else:
                    common_chars_set &= keys
            common_chars_set = common_chars_set or set()

        records: List[Dict] = []
        for r in books_in_period:
            book = r.get("book") or ""
            group_val = (r.get("region") or "（空）")

            if sample_key:
                mu, a, used_chars = compute_book_mu_a_from_chars(
                    char_means_by_book.get(book, {}),
                    sample_key=sample_key,
                    common_chars=common_chars_set,
                    q=q,
                )
                common_used = len(common_chars_set) if common_chars_set is not None else None
                used = len(used_chars)
            else:
                mu = float(r.get(metric, float("nan")))
                a = None
                common_used = None
                used = None
            if not _is_finite_number(mu):
                continue

            records.append(
                {
                    "book": book,
                    "title": r.get("title") or "",
                    "year": r.get("year"),
                    "region": r.get("region") or "",
                    "province": r.get("province") or "",
                    "place": r.get("place") or "",
                    "style": r.get("style") or "",
                    "style_tags": r.get("style_tags") or [],
                    "group": group_val,
                    "mu": mu,
                    "a": a,
                    "common_chars_count": common_used,
                    "used_chars_count": used,
                }
            )

        groups = sorted({rec["group"] for rec in records})
        group_summaries: List[Dict] = []
        for g in groups:
            subset = [rec for rec in records if rec["group"] == g]
            mus = [float(rec["mu"]) for rec in subset if math.isfinite(float(rec["mu"]))]
            aas = [float(rec["a"]) for rec in subset if _is_finite_number(rec.get("a"))]
            group_summaries.append(
                {
                    "group": g,
                    "n_books": len(subset),
                    "mu_mean": float(np.mean(mus)) if mus else None,
                    "mu_median": float(np.median(mus)) if mus else None,
                    "mu_std": float(np.std(mus)) if mus else None,
                    "a_mean": float(np.mean(aas)) if aas else None,
                    "a_median": float(np.median(aas)) if aas else None,
                    "a_std": float(np.std(aas)) if aas else None,
                }
            )

        out_payloads.append(
            {
                "label": p.label,
                "start": p.start,
                "end": p.end,
                "end_inclusive": p.end_inclusive,
                "n_books": len(records),
                "common_chars_count": len(common_chars_set) if (common_chars_enabled and sample_key) else None,
                "groups": group_summaries,
                "books": sorted(records, key=lambda r: (r.get("year") or 0, r.get("book") or "")),
            }
        )

    return out_payloads


def launch_gui(
    rows: List[Dict],
    char_means_by_book: Dict[str, Dict[str, Dict[str, float]]],
    periods: List[Period],
    metric_default: str,
    q_default: float,
    common_chars_default: bool,
    selected_regions_default: set[str],
    selected_styles_default: set[str],
    styles_by_region_default: Dict[str, set[str]],
) -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "STHeiti",
        "Heiti TC",
        "PingFang SC",
        "PingFang HK",
        "Hiragino Sans GB",
        "Hiragino Sans",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["text.color"] = "#111827"
    plt.rcParams["axes.labelcolor"] = "#111827"
    plt.rcParams["xtick.color"] = "#111827"
    plt.rcParams["ytick.color"] = "#111827"
    plt.rcParams["axes.titlecolor"] = "#111827"

    state = {
        "metric": metric_default if metric_default in METRICS else "aspect_ratio",
        "q": max(0.0, min(0.25, float(q_default))),
        "common_chars": bool(common_chars_default),
        "bulk_update": False,
    }

    regions = sorted({(r.get("region") or "（空）") for r in rows}) or ["（空）"]
    style_tags = sorted({tag for r in rows for tag in (r.get("style_tags") or ["（空）"])}) or ["（空）"]
    state["style_by_region"] = resolve_styles_by_region(
        regions=regions,
        style_tags=style_tags,
        selected_styles_default=set(selected_styles_default),
        styles_by_region_override=styles_by_region_default,
    )
    state["style_target_region"] = regions[0]

    fig = plt.figure(figsize=(16.8, 10.0))
    control_x, control_w = 0.02, 0.14
    # Keep a clearer gap from controls and slightly narrow the plot area.
    plot_x = control_x + control_w + 0.04
    plot_w = 0.96 - plot_x

    ax_metric = fig.add_axes([control_x, 0.73, control_w, 0.23], facecolor="#f8fafc")
    ax_toggles = fig.add_axes([control_x, 0.64, control_w, 0.07], facecolor="#f8fafc")
    ax_region = fig.add_axes([control_x, 0.42, control_w, 0.16], facecolor="#f8fafc")
    ax_style_scope = fig.add_axes([control_x, 0.35, control_w, 0.06], facecolor="#f8fafc")
    ax_style = fig.add_axes([control_x, 0.15, control_w, 0.19], facecolor="#f8fafc")
    ax_q = fig.add_axes([control_x, 0.11, control_w, 0.03], facecolor="#f8fafc")

    n_periods = max(1, len(periods))
    # Slightly compress each period panel so 3-row layout fits smaller screens.
    top, bottom, v_gap = 0.93, 0.17, 0.03
    plot_h = (top - bottom - v_gap * (n_periods - 1)) / n_periods
    plot_axes = []
    for i in range(n_periods):
        y = top - (i + 1) * plot_h - i * v_gap
        plot_axes.append(fig.add_axes([plot_x, y, plot_w, plot_h], facecolor="#ffffff"))

    metric_labels = list(METRICS.keys())
    metric_active = metric_labels.index(state["metric"]) if state["metric"] in metric_labels else 0
    metric_radio = RadioButtons(ax_metric, metric_labels, active=metric_active)
    toggle_check = CheckButtons(ax_toggles, ["仅共同字"], [state["common_chars"]])
    region_check = CheckButtons(ax_region, regions, [(r in selected_regions_default) if selected_regions_default else True for r in regions])
    style_scope_radio = RadioButtons(ax_style_scope, regions, active=0)
    style_check = CheckButtons(ax_style, style_tags, [True] * len(style_tags))
    q_slider = Slider(ax_q, "Q", 0.0, 0.25, valinit=state["q"], valstep=0.01)

    ax_style_scope.set_title("样式编辑地区", fontsize=9, color="#111827")

    def selected_labels(labels: List[str], status: List[bool]) -> set[str]:
        return {l for l, on in zip(labels, status) if on}

    def style_set_for_row(region_value: str) -> set[str]:
        return set(state["style_by_region"].get(region_value, set()))

    def sync_style_check_from_state() -> None:
        target = set(state["style_by_region"].get(state["style_target_region"], set()))
        current_status = list(style_check.get_status())
        state["bulk_update"] = True
        for idx, tag in enumerate(style_tags):
            want = tag in target
            if current_status[idx] != want:
                style_check.set_active(idx)
                current_status[idx] = want
        state["bulk_update"] = False

    def current_filtered_rows() -> List[Dict]:
        region_sel = selected_labels(regions, list(region_check.get_status()))
        out: List[Dict] = []
        for r in rows:
            if r.get("year") is None:
                continue
            region_value = (r.get("region") or "（空）")
            if region_sel and region_value not in region_sel:
                continue
            tags = set(r.get("style_tags") or ["（空）"])
            style_sel_local = style_set_for_row(region_value)
            if (not style_sel_local) or tags.isdisjoint(style_sel_local):
                continue
            out.append(r)
        return out

    def on_style_changed(_label: str) -> None:
        if state.get("bulk_update"):
            return
        selected = selected_labels(style_tags, list(style_check.get_status()))
        state["style_by_region"][state["style_target_region"]] = set(selected)
        update()

    def on_style_scope_change(label: str) -> None:
        state["style_target_region"] = label
        sync_style_check_from_state()
        fig.canvas.draw_idle()

    def update() -> None:
        state["metric"] = metric_radio.value_selected
        st = toggle_check.get_status()
        state["common_chars"] = bool(st[0])
        state["q"] = float(q_slider.val)
        q = max(0.0, min(0.25, float(state["q"])))

        for ax in plot_axes:
            ax.clear()
        filt = current_filtered_rows()
        if not periods:
            plot_axes[0].text(0.5, 0.5, "无可用数据", transform=plot_axes[0].transAxes, ha="center", va="center", color="#111827")
            fig.canvas.draw_idle()
            return

        active_region_order = [region for region, on in zip(regions, region_check.get_status()) if on]
        cmap = plt.get_cmap("tab10")
        region_color_map = {reg: cmap(i % 10) for i, reg in enumerate(active_region_order)}
        for idx, (ax, period) in enumerate(zip(plot_axes, periods)):
            books_data, common_count = collect_books_for_period(
                rows=filt,
                char_means_by_book=char_means_by_book,
                period=period,
                metric=state["metric"],
                q=q,
                common_chars_enabled=bool(state.get("common_chars")),
            )
            if not books_data:
                ax.text(0.5, 0.5, f"{period.label}: 无可用数据", transform=ax.transAxes, ha="center", va="center", color="#111827")
                ax.set_ylabel(METRICS.get(state["metric"], state["metric"]))
            else:
                draw_clustered_books_plot(
                    ax=ax,
                    books_data=books_data,
                    region_order=active_region_order,
                    metric_label=METRICS.get(state["metric"], state["metric"]),
                    title=f"{period.label} {period.start}-{period.end}{'含' if period.end_inclusive else ''} | n={len(books_data)}",
                    q=q,
                    common_count=common_count,
                    annotate_mu_sigma=True,
                    region_color_map=region_color_map,
                    show_xlabels=(idx == len(periods) - 1),
                )
        fig.canvas.draw_idle()

    metric_radio.on_clicked(lambda _label: update())
    toggle_check.on_clicked(lambda _label: update())
    region_check.on_clicked(lambda _label: update())
    style_scope_radio.on_clicked(on_style_scope_change)
    style_check.on_clicked(on_style_changed)
    q_slider.on_changed(lambda _val: update())

    sync_style_check_from_state()
    update()
    plt.show()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Space explorer（当前固定按 region 分组；无图环境请使用 --no-gui）."
    )
    ap.add_argument("--bundle", type=str, default="data/analysis", help="Bundle root folder (default: data/analysis)")
    ap.add_argument("--metadata", type=str, default="data/metadata/books_metadata.csv", help="Books metadata CSV")
    ap.add_argument("--threshold", type=int, default=160, help="Black threshold (default: 160)")
    ap.add_argument("--metric", type=str, default="aspect_ratio", choices=list(METRICS.keys()), help="Metric to analyze")
    ap.add_argument("--q", type=float, default=0.05, help="Quantile trimming per book (default: 0.05)")
    ap.add_argument("--periods", type=str, default="1127-1162,1162-1208,1208-1273", help="Year periods")
    ap.add_argument("--regions", type=str, default="", help="Filter regions (comma-separated)")
    ap.add_argument("--styles", type=str, default="", help="Filter style tags (comma-separated)")
    ap.add_argument("--styles-by-region", type=str, default="", help="Per-region styles, e.g. 两浙地区:近欧型|扁方欧体;福建地区:近欧型")
    ap.add_argument("--search", type=str, default="", help="Filter by book id/title substring")
    ap.add_argument("--exclude-books", type=str, default="", help="Exclude books (comma-separated)")
    ap.add_argument("--focus-period", type=str, default="", help="Focus period for CLI/plot, e.g. 1 or 早期")
    ap.add_argument("--all-periods", action="store_true", help="CLI: output all periods (not only focus period)")
    ap.add_argument("--common-chars", action="store_true", help="Use common chars intersection within each period")
    ap.add_argument("--format", type=str, default="json", choices=["json", "csv", "text"], help="Output format")
    ap.add_argument("--out", type=str, default="", help="Write output to file (default: stdout)")
    ap.add_argument("--plot", action="store_true", help="Show matplotlib plot (single focused period, clustered by region)")
    ap.add_argument("--save-fig", type=str, default="", help="Save figure to file (PNG)")
    ap.add_argument("--gui", action="store_true", help="Interactive GUI mode (compatibility alias; default behavior)")
    ap.add_argument("--no-gui", action="store_true", help="Run CLI mode only (recommended for headless/CI)")
    args = ap.parse_args()

    bundle_root = PROJECT_ROOT / args.bundle
    metadata_path = PROJECT_ROOT / args.metadata
    rows, char_means_by_book, _ = build_dataset(bundle_root, metadata_path, threshold=int(args.threshold))

    periods = parse_periods(args.periods)
    q = max(0.0, min(0.25, float(args.q)))
    selected_regions = set(parse_list_arg(args.regions))
    selected_styles = set(parse_list_arg(args.styles))
    exclude_books = set(parse_list_arg(args.exclude_books))
    styles_by_region_override = parse_styles_by_region(args.styles_by_region)

    # Candidate books (by meta filters only; period filter happens later)
    candidate: List[Dict] = []
    for r in rows:
        year = r.get("year")
        if year is None:
            continue
        if selected_regions and (r.get("region") or "（空）") not in selected_regions:
            continue
        if args.search:
            s = (r.get("book", "") + " " + r.get("title", "")).lower()
            if args.search.lower() not in s:
                continue
        if (r.get("book") or "") in exclude_books:
            continue
        candidate.append(r)

    all_regions = sorted({(r.get("region") or "（空）") for r in candidate}) or ["（空）"]
    all_style_tags = sorted({tag for r in candidate for tag in (r.get("style_tags") or ["（空）"])}) or ["（空）"]
    styles_by_region = resolve_styles_by_region(
        regions=all_regions,
        style_tags=all_style_tags,
        selected_styles_default=selected_styles,
        styles_by_region_override=styles_by_region_override,
    )
    filtered_rows = filter_rows_by_region_styles(
        rows=candidate,
        selected_regions=selected_regions,
        styles_by_region=styles_by_region,
    )

    if (not args.no_gui) or args.gui:
        launch_gui(
            rows=candidate,
            char_means_by_book=char_means_by_book,
            periods=periods,
            metric_default=args.metric,
            q_default=q,
            common_chars_default=bool(args.common_chars),
            selected_regions_default=selected_regions,
            selected_styles_default=selected_styles,
            styles_by_region_default=styles_by_region_override,
        )
        return

    if args.all_periods:
        results = compute_period_payloads(
            candidate_rows=filtered_rows,
            char_means_by_book=char_means_by_book,
            periods=periods,
            metric=args.metric,
            q=q,
            common_chars_enabled=bool(args.common_chars),
        )
        payload = {
            "metric": args.metric,
            "metric_label": METRICS.get(args.metric, args.metric),
            "q": q,
            "group_by": "region",
            "filters": {
                "regions": sorted(selected_regions),
                "styles": sorted(selected_styles),
                "styles_by_region": {k: sorted(v) for k, v in styles_by_region.items()},
                "exclude_books": sorted(exclude_books),
                "search": args.search,
                "common_chars": bool(args.common_chars),
                "common_order": "common-first" if args.common_chars else None,
            },
            "periods": [
                {"label": p.label, "start": p.start, "end": p.end, "end_inclusive": p.end_inclusive}
                for p in periods
            ],
            "results": results,
        }
        if args.plot or args.save_fig:
            fig, axes = plt.subplots(len(periods), 1, figsize=(14.5, 3.6 * max(1, len(periods))))
            if not isinstance(axes, np.ndarray):
                axes = np.asarray([axes])
            region_order = [r for r in all_regions if (not selected_regions or r in selected_regions)]
            cmap = plt.get_cmap("tab10")
            region_color_map = {reg: cmap(i % 10) for i, reg in enumerate(region_order)}
            for i, (ax, p) in enumerate(zip(axes, periods)):
                books_data, common_count = collect_books_for_period(
                    rows=filtered_rows,
                    char_means_by_book=char_means_by_book,
                    period=p,
                    metric=args.metric,
                    q=q,
                    common_chars_enabled=bool(args.common_chars),
                )
                draw_clustered_books_plot(
                    ax=ax,
                    books_data=books_data,
                    region_order=region_order,
                    metric_label=METRICS.get(args.metric, args.metric),
                    title=f"{p.label} {p.start}-{p.end}{'含' if p.end_inclusive else ''} | n={len(books_data)}",
                    q=q,
                    common_count=common_count,
                    annotate_mu_sigma=False,
                    region_color_map=region_color_map,
                    show_xlabels=(i == len(periods) - 1),
                )
            fig.tight_layout()
            if args.save_fig:
                Path(args.save_fig).parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(args.save_fig, dpi=160)
            if args.plot:
                plt.show()

        if args.format == "json":
            output = json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False)
        elif args.format == "csv":
            lines = ["period,group,n_books,mu_mean,mu_median,mu_std,a_mean,a_median,a_std"]
            for pres in results:
                for g in pres["groups"]:
                    lines.append(
                        "{period},{group},{n},{mu_mean},{mu_median},{mu_std},{a_mean},{a_median},{a_std}".format(
                            period=pres["label"],
                            group=g["group"],
                            n=g["n_books"],
                            mu_mean=_fmt_csv_num(g["mu_mean"]),
                            mu_median=_fmt_csv_num(g["mu_median"]),
                            mu_std=_fmt_csv_num(g["mu_std"]),
                            a_mean=_fmt_csv_num(g["a_mean"]),
                            a_median=_fmt_csv_num(g["a_median"]),
                            a_std=_fmt_csv_num(g["a_std"]),
                        )
                    )
            output = "\n".join(lines)
        else:
            lines = [
                f"metric={args.metric} q={q:.2f} group_by=region common_chars={bool(args.common_chars)} order=common-first",
                f"periods={args.periods}",
                "",
            ]
            for pres in results:
                cc = pres.get("common_chars_count")
                cc_hint = f" common_chars={cc}" if cc is not None else ""
                lines.append(f"[{pres['label']}] {pres['start']}~{pres['end']}{' (inclusive)' if pres['end_inclusive'] else ''} n_books={pres['n_books']}{cc_hint}")
                for g in pres["groups"]:
                    lines.append(
                        "  {group}: n={n} mu_med={mu_m} mu_mean={mu_mean}".format(
                            group=g["group"],
                            n=g["n_books"],
                            mu_m=_fmt_num(g["mu_median"], digits=4, empty="-"),
                            mu_mean=_fmt_num(g["mu_mean"], digits=4, empty="-"),
                        )
                    )
                lines.append("")
            output = "\n".join(lines)
        if args.out:
            Path(args.out).write_text(output, encoding="utf-8")
        else:
            print(output)
        return

    focus_idx = resolve_focus_period(periods, args.focus_period)
    focus_period = periods[focus_idx]
    books_data, common_count = collect_books_for_period(
        rows=filtered_rows,
        char_means_by_book=char_means_by_book,
        period=focus_period,
        metric=args.metric,
        q=q,
        common_chars_enabled=bool(args.common_chars),
    )
    region_order = [r for r in all_regions if (not selected_regions or r in selected_regions)]
    books_data, positions, region_spans = cluster_books_by_region(books_data, region_order)
    books_out: List[Dict] = []
    for b, x in zip(books_data, positions):
        row = dict(b)
        row["plot_x"] = float(x)
        books_out.append(row)

    payload = {
        "metric": args.metric,
        "metric_label": METRICS.get(args.metric, args.metric),
        "q": q,
        "focus_period": {
            "index": focus_idx + 1,
            "label": focus_period.label,
            "start": focus_period.start,
            "end": focus_period.end,
            "end_inclusive": focus_period.end_inclusive,
        },
        "group_by": "region",
        "filters": {
            "regions": sorted(selected_regions),
            "styles": sorted(selected_styles),
            "styles_by_region": {k: sorted(v) for k, v in styles_by_region.items()},
            "exclude_books": sorted(exclude_books),
            "search": args.search,
            "common_chars": bool(args.common_chars),
            "common_order": "common-first" if args.common_chars else None,
        },
        "summary": {
            "n_books": len(books_data),
            "common_chars_count": common_count,
            "regions": sorted(region_spans.keys()),
            "region_spans": {k: [float(vv) for vv in vals] for k, vals in region_spans.items()},
        },
        "books": books_out,
    }

    # Plot (optional)
    if args.plot or args.save_fig:
        fig, ax = plt.subplots(1, 1, figsize=(14.0, 7.8))
        if books_data:
            cmap = plt.get_cmap("tab10")
            region_color_map = {reg: cmap(i % 10) for i, reg in enumerate(region_order)}
            draw_clustered_books_plot(
                ax=ax,
                books_data=books_data,
                region_order=region_order,
                metric_label=METRICS.get(args.metric, args.metric),
                title=f"{focus_period.label} {focus_period.start}-{focus_period.end}{'含' if focus_period.end_inclusive else ''} | n={len(books_data)}",
                q=q,
                common_count=common_count,
                annotate_mu_sigma=False,
                region_color_map=region_color_map,
            )
        else:
            ax.text(0.5, 0.5, "无可用数据", transform=ax.transAxes, ha="center", va="center", color="#111827")
            ax.set_title(f"{focus_period.label} {focus_period.start}-{focus_period.end}{'含' if focus_period.end_inclusive else ''}")
        fig.tight_layout()
        if args.save_fig:
            Path(args.save_fig).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.save_fig, dpi=160)
        if args.plot:
            plt.show()

    output = ""
    if args.format == "json":
        output = json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False)
    elif args.format == "csv":
        lines = ["book,title,year,order,region,plot_x,mu,a,n_values"]
        for b in books_out:
            lines.append(
                "{book},{title},{year},{order},{region},{plot_x},{mu},{a},{n_values}".format(
                    book=b.get("book", ""),
                    title=b.get("title", ""),
                    year=b.get("year", ""),
                    order=b.get("order", ""),
                    region=b.get("region", ""),
                    plot_x=_fmt_csv_num(b.get("plot_x")),
                    mu=_fmt_csv_num(b.get("mu")),
                    a=_fmt_csv_num(b.get("a")),
                    n_values=b.get("n_values", ""),
                )
            )
        output = "\n".join(lines)
    else:
        lines = [
            f"metric={args.metric} q={q:.2f} group_by=region common_chars={bool(args.common_chars)} order=common-first",
            f"periods={args.periods}",
            f"focus={focus_period.label}({focus_idx+1})",
            "",
        ]
        cc_hint = f" common_chars={common_count}" if common_count is not None else ""
        lines.append(f"[{focus_period.label}] {focus_period.start}~{focus_period.end}{' (inclusive)' if focus_period.end_inclusive else ''} n_books={len(books_data)}{cc_hint}")
        for b in books_data:
            a_text = _fmt_num(b.get("a"), digits=4, empty="-")
            lines.append(
                "  {book} ({region}) mu={mu} a={a} n={n}".format(
                    book=b.get("book", ""),
                    region=b.get("region", ""),
                    mu=_fmt_num(b.get("mu"), digits=4, empty="-"),
                    a=a_text,
                    n=int(b.get("n_values", 0)),
                )
            )
        lines.append("")
        output = "\n".join(lines)

    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
