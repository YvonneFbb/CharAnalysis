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


def compute_influence(records: List[Dict], value_key: str, limit: int = 12) -> List[Dict]:
    valid = [
        (i, float(r.get(value_key, float("nan"))))
        for i, r in enumerate(records)
        if math.isfinite(float(r.get(value_key, float("nan"))))
        and r.get("year") is not None
        and not bool(r.get("excluded"))
    ]
    if len(valid) < 3:
        return []
    years = np.asarray([float(records[i].get("year")) for i, _ in valid], dtype=float)
    xs_full = rankdata(years).tolist()
    ys_full = [v for _, v in valid]
    rho_full = spearman_rho(xs_full, ys_full)
    out: List[Dict] = []
    for k, (i, _) in enumerate(valid):
        xs = [xs_full[j] for j in range(len(xs_full)) if j != k]
        ys = [ys_full[j] for j in range(len(ys_full)) if j != k]
        rho_loo = spearman_rho(xs, ys)
        out.append(
            {
                "book": records[i].get("book") or "",
                "title": records[i].get("title") or "",
                "delta": float(rho_loo - rho_full) if math.isfinite(rho_loo) and math.isfinite(rho_full) else float("nan"),
            }
        )
    out = [r for r in out if math.isfinite(float(r.get("delta", float("nan"))))]
    return sorted(out, key=lambda r: abs(float(r["delta"])), reverse=True)[:limit]


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


def compute_period_payloads(
    candidate_rows: List[Dict],
    char_means_by_book: Dict[str, Dict[str, Dict[str, float]]],
    periods: List[Period],
    metric: str,
    group_by: str,
    q: float,
    common_chars_enabled: bool,
    exclude_books: Optional[set[str]] = None,
) -> List[Dict]:
    sample_key = METRIC_SAMPLES.get(metric)
    exclude_books = exclude_books or set()
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
            group_val = (r.get(group_by) or "（空）") if group_by != "style" else (r.get("style") or "（空）")
            excluded = book in exclude_books

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
                a = float("nan")
                common_used = None
                used = None

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
                    "excluded": excluded,
                    "common_chars_count": common_used,
                    "used_chars_count": used,
                }
            )

        groups = sorted({rec["group"] for rec in records})
        group_summaries: List[Dict] = []
        for g in groups:
            subset = [rec for rec in records if rec["group"] == g and not bool(rec.get("excluded"))]
            mus = [float(rec["mu"]) for rec in subset if math.isfinite(float(rec["mu"]))]
            aas = [float(rec["a"]) for rec in subset if math.isfinite(float(rec["a"]))]
            group_summaries.append(
                {
                    "group": g,
                    "n_books": len(subset),
                    "mu_mean": float(np.mean(mus)) if mus else float("nan"),
                    "mu_median": float(np.median(mus)) if mus else float("nan"),
                    "mu_std": float(np.std(mus)) if mus else float("nan"),
                    "a_mean": float(np.mean(aas)) if aas else float("nan"),
                    "a_median": float(np.median(aas)) if aas else float("nan"),
                    "a_std": float(np.std(aas)) if aas else float("nan"),
                }
            )

        out_payloads.append(
            {
                "label": p.label,
                "start": p.start,
                "end": p.end,
                "end_inclusive": p.end_inclusive,
                "n_books": len(records),
                "n_books_used": len([r for r in records if not bool(r.get("excluded"))]),
                "common_chars_count": len(common_chars_set) if (common_chars_enabled and sample_key) else None,
                "groups": group_summaries,
                "influence": {
                    "u": compute_influence(records, "mu", limit=12),
                    "a": compute_influence(records, "a", limit=12) if sample_key else [],
                },
                "books": sorted(records, key=lambda r: (r.get("year") or 0, r.get("book") or "")),
            }
        )

    return out_payloads


def launch_gui(
    rows: List[Dict],
    char_means_by_book: Dict[str, Dict[str, Dict[str, float]]],
    periods: List[Period],
    metric_default: str,
    group_by_default: str,
    q_default: float,
    common_chars_default: bool,
    selected_regions_default: set[str],
    selected_styles_default: set[str],
    exclude_books_default: set[str],
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
        "metric": metric_default if metric_default in METRICS else "face_ratio",
        "group_by": group_by_default if group_by_default in {"region", "province", "place", "style"} else "region",
        "q": max(0.0, min(0.25, float(q_default))),
        "common_chars": bool(common_chars_default),
        "exclude_books": set(exclude_books_default),
        "focus_period_idx": 0,
    }

    regions = sorted({(r.get("region") or "（空）") for r in rows})
    style_tags = sorted({tag for r in rows for tag in (r.get("style_tags") or ["（空）"])})
    if not regions:
        regions = ["（空）"]
    if not style_tags:
        style_tags = ["（空）"]

    fig = plt.figure(figsize=(17.5, 10.0))
    control_x, control_w = 0.02, 0.14
    panel_x = control_x + control_w + 0.01
    panel_w = 0.13
    plot_x = panel_x + panel_w + 0.01
    plot_w = 0.98 - plot_x

    ax_metric = fig.add_axes([control_x, 0.73, control_w, 0.23], facecolor="#f8fafc")
    ax_group = fig.add_axes([control_x, 0.66, control_w, 0.06], facecolor="#f8fafc")
    ax_toggles = fig.add_axes([control_x, 0.60, control_w, 0.05], facecolor="#f8fafc")
    ax_region = fig.add_axes([control_x, 0.42, control_w, 0.16], facecolor="#f8fafc")
    ax_style = fig.add_axes([control_x, 0.18, control_w, 0.22], facecolor="#f8fafc")
    ax_q = fig.add_axes([control_x, 0.11, control_w, 0.03], facecolor="#f8fafc")

    ax_period = fig.add_axes([panel_x, 0.78, panel_w, 0.18], facecolor="#f8fafc")
    ax_influence = fig.add_axes([panel_x, 0.44, panel_w, 0.32], facecolor="#f8fafc")
    ax_exclude = fig.add_axes([panel_x, 0.10, panel_w, 0.30], facecolor="#f8fafc")

    n_p = max(1, len(periods))
    sub_h = 0.23
    sub_gap = 0.03
    top = 0.94
    plot_axes: List = []
    for i in range(n_p):
        y = top - (i + 1) * sub_h - i * sub_gap
        plot_axes.append(fig.add_axes([plot_x, y, plot_w, sub_h], facecolor="#ffffff"))

    metric_labels = list(METRICS.keys())
    metric_active = metric_labels.index(state["metric"]) if state["metric"] in metric_labels else 0
    metric_radio = RadioButtons(ax_metric, metric_labels, active=metric_active)
    group_labels = ["region", "province", "place", "style"]
    group_active = group_labels.index(state["group_by"])
    group_radio = RadioButtons(ax_group, group_labels, active=group_active)
    toggle_check = CheckButtons(ax_toggles, ["仅共同字"], [state["common_chars"]])
    region_check = CheckButtons(ax_region, regions, [(r in selected_regions_default) if selected_regions_default else True for r in regions])
    style_check = CheckButtons(ax_style, style_tags, [(s in selected_styles_default) if selected_styles_default else True for s in style_tags])
    q_slider = Slider(ax_q, "Q", 0.0, 0.25, valinit=state["q"], valstep=0.01)
    period_radio = RadioButtons(ax_period, [p.label for p in periods], active=0 if periods else None)

    exclude_cb = None
    exclude_keys: List[str] = []

    def selected_labels(labels: List[str], status: List[bool]) -> set[str]:
        return {l for l, on in zip(labels, status) if on}

    def current_filtered_rows() -> List[Dict]:
        region_sel = selected_labels(regions, list(region_check.get_status()))
        style_sel = selected_labels(style_tags, list(style_check.get_status()))
        out: List[Dict] = []
        for r in rows:
            if r.get("year") is None:
                continue
            if region_sel and (r.get("region") or "（空）") not in region_sel:
                continue
            tags = set(r.get("style_tags") or ["（空）"])
            if style_sel and tags.isdisjoint(style_sel):
                continue
            out.append(r)
        return out

    def on_toggle_exclude(book_id: str) -> None:
        if book_id in state["exclude_books"]:
            state["exclude_books"].remove(book_id)
        else:
            state["exclude_books"].add(book_id)
        update()

    def update() -> None:
        nonlocal exclude_cb, exclude_keys
        state["metric"] = metric_labels[metric_radio.value_selected]
        state["group_by"] = group_radio.value_selected
        st = toggle_check.get_status()
        state["common_chars"] = bool(st[0])
        state["q"] = float(q_slider.val)
        if periods:
            state["focus_period_idx"] = [p.label for p in periods].index(period_radio.value_selected)

        filt = current_filtered_rows()
        payloads = compute_period_payloads(
            candidate_rows=filt,
            char_means_by_book=char_means_by_book,
            periods=periods,
            metric=state["metric"],
            group_by=state["group_by"],
            q=float(state["q"]),
            common_chars_enabled=bool(state["common_chars"]),
            exclude_books=state["exclude_books"],
        )

        for ax in plot_axes:
            ax.clear()
        for i, pres in enumerate(payloads):
            ax = plot_axes[i]
            groups = [g["group"] for g in pres["groups"] if int(g.get("n_books", 0)) > 0]
            group_data = []
            for g in groups:
                vals = [
                    float(b["mu"])
                    for b in pres["books"]
                    if b.get("group") == g and not bool(b.get("excluded")) and math.isfinite(float(b.get("mu", float("nan"))))
                ]
                group_data.append(vals)
            if groups and any(len(v) > 0 for v in group_data):
                ax.boxplot(group_data, labels=groups, showfliers=True)
            else:
                ax.text(0.5, 0.5, "无可用数据", transform=ax.transAxes, ha="center", va="center", color="#111827")
            cc = pres.get("common_chars_count")
            cc_text = f" | common={cc}" if cc is not None else ""
            ax.set_title(f"{pres['label']} {pres['start']}-{pres['end']}{'含' if pres['end_inclusive'] else ''} | n={pres['n_books_used']}/{pres['n_books']}{cc_text}")
            ax.set_ylabel(METRICS.get(state["metric"], state["metric"]))
            ax.tick_params(axis="x", labelrotation=45)
            ax.grid(True, alpha=0.20)

        ax_influence.clear()
        ax_influence.axis("off")
        ax_influence.set_title("影响书籍（焦点时期）", fontsize=9)
        ax_exclude.clear()
        ax_exclude.axis("off")
        ax_exclude.set_title("剔除书籍（统计排除）", fontsize=9)
        if payloads:
            focus = payloads[int(state["focus_period_idx"])]
            lines: List[str] = []
            inf_u = focus.get("influence", {}).get("u", [])
            inf_a = focus.get("influence", {}).get("a", [])
            lines.append("u:")
            for r in inf_u[:8]:
                lines.append(f"{r.get('book')} {float(r.get('delta', float('nan'))):+.3f}")
            if inf_a:
                lines.append("")
                lines.append("a:")
                for r in inf_a[:8]:
                    lines.append(f"{r.get('book')} {float(r.get('delta', float('nan'))):+.3f}")
            ax_influence.text(0.0, 1.0, "\n".join(lines) if lines else "NA", ha="left", va="top", fontsize=8, color="#111827")

            keys = []
            for r in inf_u[:8]:
                b = r.get("book") or ""
                if b:
                    keys.append(b)
            for r in inf_a[:8]:
                b = r.get("book") or ""
                if b and b not in keys:
                    keys.append(b)
            exclude_keys = keys
            if exclude_keys:
                status = [k in state["exclude_books"] for k in exclude_keys]
                exclude_cb = CheckButtons(ax_exclude, exclude_keys, status)
                for lbl in exclude_cb.labels:
                    lbl.set_fontsize(8)
                    lbl.set_color("#111827")
                exclude_cb.on_clicked(on_toggle_exclude)
            else:
                ax_exclude.text(0.0, 1.0, "无可剔除候选", ha="left", va="top", fontsize=8, color="#111827")

        fig.canvas.draw_idle()

    metric_radio.on_clicked(lambda _label: update())
    group_radio.on_clicked(lambda _label: update())
    toggle_check.on_clicked(lambda _label: update())
    region_check.on_clicked(lambda _label: update())
    style_check.on_clicked(lambda _label: update())
    q_slider.on_changed(lambda _val: update())
    period_radio.on_clicked(lambda _label: update())

    update()
    plt.show()


def main() -> None:
    ap = argparse.ArgumentParser(description="Space explorer (periods + group comparisons).")
    ap.add_argument("--bundle", type=str, default="data/analysis", help="Bundle root folder (default: data/analysis)")
    ap.add_argument("--metadata", type=str, default="data/metadata/books_metadata.csv", help="Books metadata CSV")
    ap.add_argument("--threshold", type=int, default=160, help="Black threshold (default: 160)")
    ap.add_argument("--metric", type=str, default="aspect_ratio", choices=list(METRICS.keys()), help="Metric to analyze")
    ap.add_argument("--q", type=float, default=0.05, help="Quantile trimming per book (default: 0.05)")
    ap.add_argument("--periods", type=str, default="1127-1162,1162-1208,1208-1273", help="Year periods")
    ap.add_argument("--regions", type=str, default="", help="Filter regions (comma-separated)")
    ap.add_argument("--styles", type=str, default="", help="Filter style tags (comma-separated)")
    ap.add_argument("--search", type=str, default="", help="Filter by book id/title substring")
    ap.add_argument("--group-by", type=str, default="region", choices=["region", "province", "place", "style"], help="Group key")
    ap.add_argument("--common-chars", action="store_true", help="Use common chars intersection within each period")
    ap.add_argument("--format", type=str, default="json", choices=["json", "csv", "text"], help="Output format")
    ap.add_argument("--out", type=str, default="", help="Write output to file (default: stdout)")
    ap.add_argument("--plot", action="store_true", help="Show matplotlib plot (boxplot by group in each period)")
    ap.add_argument("--save-fig", type=str, default="", help="Save figure to file (PNG)")
    ap.add_argument("--gui", action="store_true", help="Interactive GUI mode (MVP, compatibility alias)")
    ap.add_argument("--no-gui", action="store_true", help="Run CLI mode only (default is GUI)")
    ap.add_argument("--exclude-books", type=str, default="", help="Exclude books from stats (comma-separated)")
    args = ap.parse_args()

    bundle_root = PROJECT_ROOT / args.bundle
    metadata_path = PROJECT_ROOT / args.metadata
    rows, char_means_by_book, _ = build_dataset(bundle_root, metadata_path, threshold=int(args.threshold))

    periods = parse_periods(args.periods)
    q = max(0.0, min(0.25, float(args.q)))
    selected_regions = set(parse_list_arg(args.regions))
    selected_styles = set(parse_list_arg(args.styles))
    exclude_books = set(parse_list_arg(args.exclude_books))

    # Candidate books (by meta filters only; period filter happens later)
    candidate: List[Dict] = []
    for r in rows:
        year = r.get("year")
        if year is None:
            continue
        if selected_regions and (r.get("region") or "（空）") not in selected_regions:
            continue
        if selected_styles:
            tags = set(r.get("style_tags") or ["（空）"])
            if tags.isdisjoint(selected_styles):
                continue
        if args.search:
            s = (r.get("book", "") + " " + r.get("title", "")).lower()
            if args.search.lower() not in s:
                continue
        candidate.append(r)

    if (not args.no_gui) or args.gui:
        launch_gui(
            rows=candidate,
            char_means_by_book=char_means_by_book,
            periods=periods,
            metric_default=args.metric,
            group_by_default=args.group_by,
            q_default=q,
            common_chars_default=bool(args.common_chars),
            selected_regions_default=selected_regions,
            selected_styles_default=selected_styles,
            exclude_books_default=exclude_books,
        )
        return

    period_payloads = compute_period_payloads(
        candidate_rows=candidate,
        char_means_by_book=char_means_by_book,
        periods=periods,
        metric=args.metric,
        group_by=args.group_by,
        q=q,
        common_chars_enabled=bool(args.common_chars),
        exclude_books=exclude_books,
    )

    payload = {
        "metric": args.metric,
        "metric_label": METRICS.get(args.metric, args.metric),
        "q": q,
        "group_by": args.group_by,
        "filters": {
            "regions": sorted(selected_regions),
            "styles": sorted(selected_styles),
            "search": args.search,
            "exclude_books": sorted(exclude_books),
            "common_chars": bool(args.common_chars),
            "common_order": "common-first" if args.common_chars else None,
        },
        "periods": [
            {"label": p.label, "start": p.start, "end": p.end, "end_inclusive": p.end_inclusive}
            for p in periods
        ],
        "results": period_payloads,
    }

    # Plot (optional)
    if args.plot or args.save_fig:
        fig, axes = plt.subplots(len(period_payloads), 1, figsize=(12.5, 3.2 * max(1, len(period_payloads))))
        if not isinstance(axes, np.ndarray):
            axes = np.asarray([axes])
        for ax, pres in zip(axes, period_payloads):
            groups = [g["group"] for g in pres["groups"]]
            data = []
            for g in groups:
                mus = [
                    float(b["mu"])
                    for b in pres["books"]
                    if b["group"] == g and not bool(b.get("excluded")) and math.isfinite(float(b["mu"]))
                ]
                data.append(mus)
            if groups and any(len(v) > 0 for v in data):
                ax.boxplot(data, labels=groups, showfliers=True)
            else:
                ax.text(0.5, 0.5, "无可用数据", transform=ax.transAxes, ha="center", va="center", color="#111827")
            ax.set_title(
                f"{pres['label']} {pres['start']}~{pres['end']}{'（含）' if pres['end_inclusive'] else ''} | n={pres['n_books_used']}/{pres['n_books']}"
            )
            ax.set_ylabel(METRICS.get(args.metric, args.metric))
            ax.tick_params(axis="x", labelrotation=45)
        fig.tight_layout()
        if args.save_fig:
            Path(args.save_fig).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.save_fig, dpi=160)
        if args.plot:
            plt.show()

    output = ""
    if args.format == "json":
        output = json.dumps(payload, ensure_ascii=False, indent=2)
    elif args.format == "csv":
        # period/group summaries only
        lines = ["period,group,n_books,mu_mean,mu_median,mu_std,a_mean,a_median,a_std"]
        for pres in period_payloads:
            for g in pres["groups"]:
                lines.append(
                    "{period},{group},{n},{mu_mean},{mu_median},{mu_std},{a_mean},{a_median},{a_std}".format(
                        period=pres["label"],
                        group=g["group"],
                        n=g["n_books"],
                        mu_mean=g["mu_mean"],
                        mu_median=g["mu_median"],
                        mu_std=g["mu_std"],
                        a_mean=g["a_mean"],
                        a_median=g["a_median"],
                        a_std=g["a_std"],
                    )
                )
        output = "\n".join(lines)
    else:
        lines = [
            f"metric={args.metric} q={q:.2f} group_by={args.group_by} common_chars={bool(args.common_chars)} order=common-first",
            f"periods={args.periods}",
            "",
        ]
        for pres in period_payloads:
            cc = pres.get("common_chars_count")
            cc_hint = f" common_chars={cc}" if cc is not None else ""
            lines.append(f"[{pres['label']}] {pres['start']}~{pres['end']}{' (inclusive)' if pres['end_inclusive'] else ''} n_books={pres['n_books']}{cc_hint}")
            for g in pres["groups"]:
                lines.append(
                    "  {group}: n={n} mu_med={mu_m:.4f} mu_mean={mu_mean:.4f}".format(
                        group=g["group"],
                        n=g["n_books"],
                        mu_m=g["mu_median"] if math.isfinite(float(g["mu_median"])) else float("nan"),
                        mu_mean=g["mu_mean"] if math.isfinite(float(g["mu_mean"])) else float("nan"),
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
