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
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button, CheckButtons, Slider
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]

METRICS: Dict[str, str] = {
    "aspect_ratio": "长宽比（高/宽）",
    "face_ratio": "字面率",
    "ink_coverage": "灰度比（黑/白）",
    "Lb": "固定框 Lb",
    "clipping_rate": "裁切率（w>Lb 或 h>Lb）",
}

COLOR_KEYS = {
    "region": "区域",
    "style": "刻体倾向",
    "none": "无",
}

METRIC_SAMPLES = {
    "aspect_ratio": "aspect_ratio",
    "face_ratio": "face_ratio",
    "ink_coverage": "ink_coverage",
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
    style_tags: List[str]


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
            style_raw = (row.get("刻体倾向") or "").strip()
            style_tags = split_style_tags(style_raw)
            out[book_id] = BookMeta(
                book_id=book_id,
                title=title,
                year=year,
                number=number,
                region=(row.get("区域划分") or "").strip(),
                province=(row.get("省份") or "").strip(),
                place=(row.get("地点") or "").strip(),
                style=style_raw,
                style_tags=style_tags,
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


def spearman_p_value(rho: float, n: int) -> float:
    if not math.isfinite(rho) or n < 3:
        return float("nan")
    if abs(rho) >= 1.0:
        return 0.0
    t = abs(rho) * math.sqrt((n - 2) / max(1e-12, 1.0 - rho * rho))
    try:
        from scipy import stats as scipy_stats  # type: ignore

        p = 2.0 * (1.0 - float(scipy_stats.t.cdf(t, df=n - 2)))
        return min(1.0, max(0.0, p))
    except Exception:
        z = t
        p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))
        return min(1.0, max(0.0, p))


def kendall_tau_b(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0 and dy == 0:
                ties_x += 1
                ties_y += 1
            elif dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    if denom == 0:
        return float("nan")
    return (concordant - discordant) / denom


BOOTSTRAP_N = 400
PERMUTATION_N = 400


def _stable_seed(tag: str, x: List[float], y: List[float]) -> int:
    h = hashlib.blake2b(digest_size=8)
    h.update(tag.encode("utf-8"))
    h.update(b"\0")
    h.update(np.asarray(x, dtype=float).tobytes())
    h.update(b"\0")
    h.update(np.asarray(y, dtype=float).tobytes())
    return int.from_bytes(h.digest(), "little", signed=False) & 0xFFFFFFFF


def bootstrap_ci(
    stat_fn,
    x: List[float],
    y: List[float],
    n_boot: int = BOOTSTRAP_N,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    n = len(x)
    if n < 3 or n_boot < 10:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    xs = list(map(float, x))
    ys = list(map(float, y))
    stats: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb = [xs[int(i)] for i in idx]
        yb = [ys[int(i)] for i in idx]
        v = float(stat_fn(xb, yb))
        if math.isfinite(v):
            stats.append(v)
    if len(stats) < 10:
        return (float("nan"), float("nan"))
    lo = float(np.quantile(np.asarray(stats, dtype=float), alpha / 2.0))
    hi = float(np.quantile(np.asarray(stats, dtype=float), 1.0 - alpha / 2.0))
    return (lo, hi)


def kendall_p_value_permutation(
    x: List[float],
    y: List[float],
    tau_obs: float,
    n_perm: int = PERMUTATION_N,
    seed: int = 0,
) -> float:
    n = len(x)
    if n < 3 or n_perm < 10 or not math.isfinite(tau_obs):
        return float("nan")
    rng = np.random.default_rng(seed)
    xs = list(map(float, x))
    ys = np.asarray(y, dtype=float)
    target = abs(float(tau_obs))
    hits = 1  # add-one smoothing
    for _ in range(n_perm):
        perm = rng.permutation(n)
        tau = kendall_tau_b(xs, [float(ys[int(i)]) for i in perm])
        if math.isfinite(tau) and abs(tau) >= target:
            hits += 1
    return hits / float(n_perm + 1)


def compute_trend_stats(x: List[float], y: List[float]) -> Dict[str, float]:
    n = len(x)
    rho = spearman_rho(x, y)
    p_rho = spearman_p_value(rho, n)
    rho_ci_lo, rho_ci_hi = bootstrap_ci(
        lambda a, b: spearman_rho(a, b),
        x,
        y,
        seed=_stable_seed("rho", x, y),
    )
    tau = kendall_tau_b(x, y)
    p_tau = kendall_p_value_permutation(
        x,
        y,
        tau,
        seed=_stable_seed("tau_p", x, y),
    )
    tau_ci_lo, tau_ci_hi = bootstrap_ci(
        lambda a, b: kendall_tau_b(a, b),
        x,
        y,
        seed=_stable_seed("tau_ci", x, y),
    )
    return {
        "n": float(n),
        "rho": float(rho),
        "p_rho": float(p_rho),
        "rho_ci_lo": float(rho_ci_lo),
        "rho_ci_hi": float(rho_ci_hi),
        "tau": float(tau),
        "p_tau": float(p_tau),
        "tau_ci_lo": float(tau_ci_lo),
        "tau_ci_hi": float(tau_ci_hi),
    }


def theil_sen_slope(x: List[float], y: List[float]) -> float:
    slopes: List[float] = []
    n = len(x)
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            if dx == 0:
                continue
            slopes.append((y[j] - y[i]) / dx)
    if not slopes:
        return float("nan")
    return float(np.median(np.asarray(slopes, dtype=float)))


def robust_slope_irls(x: List[float], y: List[float], method: str = "huber") -> float:
    if len(x) < 2:
        return float("nan")
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    w = np.ones_like(x_arr, dtype=float)
    c = 1.345 if method == "huber" else 4.685
    for _ in range(20):
        w_sum = np.sum(w)
        if w_sum == 0:
            return float("nan")
        x_bar = np.sum(w * x_arr) / w_sum
        y_bar = np.sum(w * y_arr) / w_sum
        num = np.sum(w * (x_arr - x_bar) * (y_arr - y_bar))
        den = np.sum(w * (x_arr - x_bar) ** 2)
        slope = num / den if den != 0 else 0.0
        intercept = y_bar - slope * x_bar
        resid = y_arr - (slope * x_arr + intercept)
        mad = np.median(np.abs(resid - np.median(resid)))
        scale = 1.4826 * mad
        if scale <= 1e-12:
            scale = float(np.std(resid)) if np.std(resid) > 0 else 1.0
        u = resid / scale
        if method == "huber":
            w = np.where(np.abs(u) <= c, 1.0, c / np.abs(u))
        else:
            w = np.where(np.abs(u) < c, (1 - (u / c) ** 2) ** 2, 0.0)
    return float(slope)


def parse_list_arg(value: Optional[str]) -> List[str]:
    if not value:
        return []
    parts = [v.strip() for v in value.replace("，", ",").split(",")]
    return [p for p in parts if p]


def compute_influence(rows: List[Dict], years: List[int], values: List[float], limit: int) -> List[Dict]:
    valid_pairs = [
        (i, v)
        for i, v in enumerate(values)
        if math.isfinite(v) and rows[i].get("year") is not None
    ]
    if len(valid_pairs) < 3:
        return []
    years_arr = np.asarray([float(years[i]) for i, _ in valid_pairs], dtype=float)
    xs_full = rankdata(years_arr).tolist()
    ys_full = [float(values[i]) for i, _ in valid_pairs]
    rho_full = spearman_rho(xs_full, ys_full)
    influence: List[Dict] = []
    for k, (i, _) in enumerate(valid_pairs):
        xs_loo = [xs_full[j] for j in range(len(xs_full)) if j != k]
        ys_loo = [ys_full[j] for j in range(len(ys_full)) if j != k]
        rho_loo = spearman_rho(xs_loo, ys_loo)
        influence.append({
            "book": rows[i].get("book"),
            "title": rows[i].get("title"),
            "delta": rho_loo - rho_full,
        })
    return sorted(influence, key=lambda r: abs(r["delta"]), reverse=True)[:limit]


def split_style_tags(style_raw: str) -> List[str]:
    if not style_raw:
        return ["（空）"]
    cleaned = style_raw.replace("，", ",").replace("、", ",").replace("/", ",").replace(";", ",").replace("；", ",")
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    return parts or ["（空）"]


def compute_book_data(
    bundle_root: Path, book: str, threshold: int
) -> Tuple[Dict, Dict[str, List[float]], Dict[str, Dict[str, float]]]:
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
        samples = {
            "aspect_ratio": [],
            "face_ratio": [],
            "ink_coverage": [],
        }
        return metrics, samples, {}

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
    char_means_map: Dict[str, Dict[str, float]] = {}
    for char, rows in per_char.items():
        ars = [r[0] for r in rows if math.isfinite(r[0])]
        frs = [r[1] for r in rows if math.isfinite(r[1])]
        inks = [r[2] for r in rows if math.isfinite(r[2])]
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

    samples = {
        "aspect_ratio": [x[0] for x in char_means if math.isfinite(x[0])],
        "face_ratio": [x[1] for x in char_means if math.isfinite(x[1])],
        "ink_coverage": [x[2] for x in char_means if math.isfinite(x[2])],
    }

    def mean_or_nan(values: List[float]) -> float:
        if not values:
            return float("nan")
        return float(np.mean(np.asarray(values, dtype=float)))

    aspect_vals = samples["aspect_ratio"]
    face_vals = samples["face_ratio"]
    ink_vals = [v for v in samples["ink_coverage"] if math.isfinite(v)]
    metrics = {
        "instances": len(instance_sizes),
        "chars": len(char_means),
        "Lb": Lb,
        "clipping_rate": float(clipped) / float(max(1, len(instance_sizes))),
        "aspect_ratio": mean_or_nan(aspect_vals),
        "face_ratio": mean_or_nan(face_vals),
        "ink_coverage": mean_or_nan(ink_vals),
    }
    return metrics, samples, char_means_map


def build_dataset(
    bundle_root: Path, metadata_path: Path, threshold: int
) -> Tuple[List[Dict], Dict[str, Dict[str, List[float]]], Dict[str, Dict[str, Dict[str, float]]]]:
    books = load_manifest_books(bundle_root)
    meta_map = read_books_metadata(metadata_path)

    rows: List[Dict] = []
    samples_by_book: Dict[str, Dict[str, List[float]]] = {}
    char_means_by_book: Dict[str, Dict[str, Dict[str, float]]] = {}

    for book in books:
        metrics, samples, char_means = compute_book_data(bundle_root, book, threshold=threshold)
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
            "style_tags": meta.style_tags if meta else ["（空）"],
            **metrics,
        })
        samples_by_book[book] = samples
        char_means_by_book[book] = char_means

    return rows, samples_by_book, char_means_by_book


def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive time explorer (matplotlib).")
    ap.add_argument("--bundle", type=str, default="data/analysis", help="Bundle root folder (default: data/analysis)")
    ap.add_argument("--metadata", type=str, default="data/metadata/books_metadata.csv", help="Books metadata CSV")
    ap.add_argument("--threshold", type=int, default=160, help="Black threshold (default: 160)")
    ap.add_argument("--metric", type=str, default="aspect_ratio", choices=list(METRICS.keys()), help="Metric to analyze")
    ap.add_argument("--regions", type=str, default="", help="Filter regions (comma-separated)")
    ap.add_argument("--styles", type=str, default="", help="Filter styles (comma-separated)")
    ap.add_argument("--search", type=str, default="", help="Filter by book id/title substring")
    ap.add_argument("--exclude-books", type=str, default="", help="Exclude books from correlation (comma-separated)")
    ap.add_argument("--common-chars", action="store_true", help="Only compare common chars across selected books")
    ap.add_argument("--q", type=float, default=0.05, help="Quantile trimming (default: 0.05)")
    ap.add_argument("--format", type=str, default="json", choices=["json", "csv", "text"], help="CLI output format")
    ap.add_argument("--out", type=str, default="", help="Write output to file (default: stdout)")
    ap.add_argument("--gui", action="store_true", help="Interactive GUI mode (compatibility alias; default behavior)")
    ap.add_argument("--no-gui", action="store_true", help="Run CLI output only")
    args = ap.parse_args()

    bundle_root = PROJECT_ROOT / args.bundle
    metadata_path = PROJECT_ROOT / args.metadata
    rows, samples_by_book, char_means_by_book = build_dataset(bundle_root, metadata_path, threshold=int(args.threshold))
    if not rows:
        raise SystemExit("没有可用数据。")

    if args.no_gui:
        metric = args.metric
        sample_key = METRIC_SAMPLES.get(metric)
        selected_regions = set(parse_list_arg(args.regions))
        selected_styles = set(parse_list_arg(args.styles))
        exclude_books = set(parse_list_arg(args.exclude_books))
        q = max(0.0, min(0.25, float(args.q)))

        candidate_rows: List[Dict] = []
        for r in sorted(
            rows,
            key=lambda row: (
                row["year"] if row["year"] is not None else 9999,
                row.get("order") if row.get("order") is not None else 9999,
                row.get("book") or "",
            ),
        ):
            if r["year"] is None:
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
            candidate_rows.append(r)

        common_chars: List[str] = []
        filtered: List[Dict] = []
        samples: List[np.ndarray] = []
        values: List[float] = []

        if sample_key:
            per_book_vals: Dict[str, List[Tuple[str, float]]] = {}
            for r in candidate_rows:
                book = r.get("book")
                char_map = char_means_by_book.get(book, {}) if book else {}
                vals = [
                    (c, float(v.get(sample_key, float("nan"))))
                    for c, v in char_map.items()
                    if math.isfinite(v.get(sample_key, float("nan")))
                ]
                if vals:
                    per_book_vals[book] = vals

            if args.common_chars and per_book_vals:
                common_set = None
                for r in candidate_rows:
                    book = r.get("book")
                    if book not in per_book_vals:
                        continue
                    keys = {c for c, _ in per_book_vals[book]}
                    if common_set is None:
                        common_set = keys
                    else:
                        common_set &= keys
                common_chars = sorted(common_set) if common_set else []

            per_book_filtered: Dict[str, List[Tuple[str, float]]] = {}
            for r in candidate_rows:
                book = r.get("book")
                vals = per_book_vals.get(book, [])
                if not vals:
                    continue
                if args.common_chars:
                    vals = [(c, v) for c, v in vals if c in common_chars]
                values_only = [v for _, v in vals]
                if q > 0.0 and len(values_only) >= 4:
                    lo = float(np.quantile(values_only, q))
                    hi = float(np.quantile(values_only, 1.0 - q))
                    vals = [(c, v) for c, v in vals if lo <= v <= hi]
                per_book_filtered[book] = vals

            for r in candidate_rows:
                book = r.get("book")
                vals = per_book_filtered.get(book, [])
                if args.common_chars:
                    vals = [(c, v) for c, v in vals if c in common_chars]
                val_list = [v for _, v in vals]
                if not val_list:
                    continue
                filtered.append(r)
                samples.append(np.asarray(val_list, dtype=float))
        else:
            for r in candidate_rows:
                v = r.get(metric)
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    continue
                filtered.append(r)
                values.append(float(v))

        if sample_key:
            mus = [float(np.mean(arr)) if arr.size else float("nan") for arr in samples]
            sigmas = [float(np.std(arr)) if arr.size else float("nan") for arr in samples]
        else:
            mus = values[:]
            sigmas = [float("nan")] * len(mus)

        years = [int(r.get("year") or 0) for r in filtered]
        valid_u = [(r, y) for r, y in zip(filtered, mus) if math.isfinite(y) and r.get("book") not in exclude_books]
        if len(valid_u) >= 2:
            xs_u = rankdata(np.asarray([float(r.get("year") or 0) for r, _ in valid_u], dtype=float)).tolist()
            ys_u = [float(y) for _, y in valid_u]
            stats_u = compute_trend_stats(xs_u, ys_u)
        else:
            stats_u = compute_trend_stats([], [])

        valid_a = [(r, y) for r, y in zip(filtered, sigmas) if math.isfinite(y) and r.get("book") not in exclude_books]
        if len(valid_a) >= 2:
            xs_a = rankdata(np.asarray([float(r.get("year") or 0) for r, _ in valid_a], dtype=float)).tolist()
            ys_a = [float(y) for _, y in valid_a]
            stats_a = compute_trend_stats(xs_a, ys_a)
        else:
            stats_a = compute_trend_stats([], [])

        influence_u = compute_influence(filtered, years, mus, limit=12)
        influence_a = compute_influence(filtered, years, sigmas, limit=12) if sample_key else []

        records = []
        for r, mu, a in zip(filtered, mus, sigmas):
            records.append({
                "book": r.get("book"),
                "title": r.get("title"),
                "year": r.get("year"),
                "order": r.get("order"),
                "region": r.get("region"),
                "province": r.get("province"),
                "place": r.get("place"),
                "style": r.get("style"),
                "style_tags": r.get("style_tags"),
                "mu": mu,
                "a": a,
                "excluded": (r.get("book") in exclude_books),
            })

        payload = {
            "metric": metric,
            "q": q,
            "filters": {
                "regions": sorted(selected_regions),
                "styles": sorted(selected_styles),
                "search": args.search,
                "exclude_books": sorted(exclude_books),
                "common_chars": args.common_chars,
                "common_order": "common-first" if args.common_chars else None,
            },
            "summary": {
                "n": len(records),
                "rho_u": stats_u.get("rho"),
                "p_u": stats_u.get("p_rho"),
                "rho_u_ci_lo": stats_u.get("rho_ci_lo"),
                "rho_u_ci_hi": stats_u.get("rho_ci_hi"),
                "tau_u": stats_u.get("tau"),
                "p_tau_u": stats_u.get("p_tau"),
                "tau_u_ci_lo": stats_u.get("tau_ci_lo"),
                "tau_u_ci_hi": stats_u.get("tau_ci_hi"),
                "rho_a": stats_a.get("rho"),
                "p_a": stats_a.get("p_rho"),
                "rho_a_ci_lo": stats_a.get("rho_ci_lo"),
                "rho_a_ci_hi": stats_a.get("rho_ci_hi"),
                "tau_a": stats_a.get("tau"),
                "p_tau_a": stats_a.get("p_tau"),
                "tau_a_ci_lo": stats_a.get("tau_ci_lo"),
                "tau_a_ci_hi": stats_a.get("tau_ci_hi"),
                "year_rank": True,
                "common_chars_count": len(common_chars) if args.common_chars else None,
            },
            "common_chars": common_chars if args.common_chars else [],
            "influence": {
                "u": influence_u,
                "a": influence_a,
            },
            "books": records,
        }

        output = ""
        if args.format == "json":
            output = json.dumps(payload, ensure_ascii=False, indent=2)
        elif args.format == "csv":
            header = [
                "book",
                "title",
                "year",
                "order",
                "region",
                "province",
                "place",
                "style",
                "style_tags",
                "mu",
                "a",
                "excluded",
            ]
            lines = [",".join(header)]
            for r in records:
                row = [str(r.get(k, "")) for k in header]
                lines.append(",".join(row))
            output = "\n".join(lines)
        else:
            common_hint = f" common_chars={len(common_chars)}" if args.common_chars else ""
            lines = [
                f"metric={metric} q={q:.2f} n={len(records)}{common_hint}",
                "u: rho={rho:.3f} p={p:.3f} CI95=[{lo:.3f},{hi:.3f}] | tau={tau:.3f} p={pt:.3f} CI95=[{tlo:.3f},{thi:.3f}]".format(
                    rho=float(stats_u.get("rho", float("nan"))),
                    p=float(stats_u.get("p_rho", float("nan"))),
                    lo=float(stats_u.get("rho_ci_lo", float("nan"))),
                    hi=float(stats_u.get("rho_ci_hi", float("nan"))),
                    tau=float(stats_u.get("tau", float("nan"))),
                    pt=float(stats_u.get("p_tau", float("nan"))),
                    tlo=float(stats_u.get("tau_ci_lo", float("nan"))),
                    thi=float(stats_u.get("tau_ci_hi", float("nan"))),
                ),
                "a: rho={rho:.3f} p={p:.3f} CI95=[{lo:.3f},{hi:.3f}] | tau={tau:.3f} p={pt:.3f} CI95=[{tlo:.3f},{thi:.3f}]".format(
                    rho=float(stats_a.get("rho", float("nan"))),
                    p=float(stats_a.get("p_rho", float("nan"))),
                    lo=float(stats_a.get("rho_ci_lo", float("nan"))),
                    hi=float(stats_a.get("rho_ci_hi", float("nan"))),
                    tau=float(stats_a.get("tau", float("nan"))),
                    pt=float(stats_a.get("p_tau", float("nan"))),
                    tlo=float(stats_a.get("tau_ci_lo", float("nan"))),
                    thi=float(stats_a.get("tau_ci_hi", float("nan"))),
                ),
                "",
                "influence_u:",
            ]
            for row in influence_u:
                lines.append(f"  {row.get('book')} {row.get('delta'):+.3f}")
            if influence_a:
                lines.append("influence_a:")
                for row in influence_a:
                    lines.append(f"  {row.get('book')} {row.get('delta'):+.3f}")
            lines.append("")
            lines.append("books:")
            for r in records:
                lines.append(
                    "  {book} {year} {region}/{place} {style} mu={mu:.3f} a={a:.3f}".format(
                        book=r.get("book") or "",
                        year=r.get("year") or "",
                        region=r.get("region") or "",
                        place=r.get("place") or "",
                        style=r.get("style") or "",
                        mu=r.get("mu") if math.isfinite(r.get("mu", float("nan"))) else float("nan"),
                        a=r.get("a") if math.isfinite(r.get("a", float("nan"))) else float("nan"),
                    )
                )
            output = "\n".join(lines)

        if args.out:
            Path(args.out).write_text(output, encoding="utf-8")
        else:
            print(output)
        return

    state = {
        "metric": "aspect_ratio",
        "color_by": "region",
        "search": "",
        "q": 0.05,
        "bulk_update": False,
        "common_chars": False,
        "exclude_books": set(),
        "influence_keys": [],
    }

    plt.rcParams["font.family"] = "sans-serif"
    # Use fonts that are confirmed on your machine, ordered by expected stroke thickness.
    plt.rcParams["font.sans-serif"] = [
        "STHeiti",
        "Heiti TC",
        "PingFang SC",
        "PingFang HK",
        "Hiragino Sans GB",
        "Hiragino Sans",
        "DejaVu Sans",
    ]
    plt.rcParams["font.weight"] = "medium"
    plt.rcParams["axes.labelweight"] = "medium"
    plt.rcParams["axes.titleweight"] = "medium"
    plt.rcParams["axes.unicode_minus"] = False
    # Make UI text readable: matplotlib defaults often render widget labels in gray.
    plt.rcParams["text.color"] = "#111827"
    plt.rcParams["axes.labelcolor"] = "#111827"
    plt.rcParams["xtick.color"] = "#111827"
    plt.rcParams["ytick.color"] = "#111827"
    plt.rcParams["axes.titlecolor"] = "#111827"

    # 确保交互式窗口不被提前回收
    plt.ioff()

    fig = plt.figure(figsize=(18.0, 9.4))
    plot_x = 0.29
    plot_w = 0.69
    ax = fig.add_axes([plot_x, 0.42, plot_w, 0.52])
    ax_std = None

    control_x = 0.02
    control_w = 0.13
    gap = 0.01
    influence_x = control_x + control_w + gap
    influence_w = 0.12

    # Left control column: keep strict vertical non-overlap (matplotlib widgets do not clip).
    ax_metric = fig.add_axes([control_x, 0.69, control_w, 0.27], facecolor="#f8fafc")
    ax_color = fig.add_axes([control_x, 0.63, control_w, 0.05], facecolor="#f8fafc")
    ax_place_toggle = fig.add_axes([control_x, 0.59, control_w, 0.035], facecolor="#f8fafc")
    ax_common_toggle = fig.add_axes([control_x, 0.545, control_w, 0.04], facecolor="#f8fafc")
    ax_region = fig.add_axes([control_x, 0.375, control_w, 0.16], facecolor="#f8fafc")
    ax_style_all = fig.add_axes([control_x, 0.335, control_w * 0.48, 0.032])
    ax_style_none = fig.add_axes([control_x + control_w * 0.52, 0.335, control_w * 0.48, 0.032])
    ax_style_left = fig.add_axes([control_x, 0.17, control_w * 0.48, 0.155], facecolor="#f8fafc")
    ax_style_right = fig.add_axes([control_x + control_w * 0.52, 0.17, control_w * 0.48, 0.155], facecolor="#f8fafc")
    ax_q = fig.add_axes([control_x, 0.08, control_w, 0.03])

    ax_influence_list = fig.add_axes([influence_x, 0.12, influence_w, 0.84], facecolor="#f8fafc")
    ax_labels = fig.add_axes([plot_x, 0.30, plot_w, 0.07], facecolor="#ffffff")
    ax_exclude = fig.add_axes([plot_x, 0.19, plot_w, 0.06], facecolor="#ffffff")
    # Bottom stats text: keep a bit further from the checkbox row to avoid visual clutter.
    ax_stats = fig.add_axes([plot_x, 0.045, plot_w, 0.11], facecolor="#ffffff")
    metric_radio = RadioButtons(ax_metric, list(METRICS.keys()), active=0)
    color_radio = RadioButtons(ax_color, list(COLOR_KEYS.keys()), active=0)

    regions = sorted({(r.get("region") or "（空）") for r in rows})
    style_tags = sorted({tag for r in rows for tag in (r.get("style_tags") or ["（空）"])})
    region_check = CheckButtons(ax_region, regions, [True] * len(regions))
    mid = max(1, math.ceil(len(style_tags) / 2))
    style_left = style_tags[:mid]
    style_right = style_tags[mid:]
    style_check_left = CheckButtons(ax_style_left, style_left, [True] * len(style_left))
    style_check_right = CheckButtons(ax_style_right, style_right, [True] * len(style_right))
    style_all_btn = Button(ax_style_all, "字体全选")
    style_none_btn = Button(ax_style_none, "字体全不选")
    place_toggle = CheckButtons(ax_place_toggle, ["区分书局"], [False])
    common_toggle = CheckButtons(ax_common_toggle, ["仅共同字"], [False])
    q_slider = Slider(ax_q, "Q", 0.0, 0.25, valinit=state["q"], valstep=0.01)
    influence_slots = 12
    ax_influence_list.set_title("影响书籍（排序）", fontsize=9, color="#111827")
    ax_influence_list.axis("off")
    ax_stats.axis("off")
    ax_exclude.axis("off")
    ax_labels.axis("off")

    for cb in (region_check, place_toggle, common_toggle):
        for lbl in cb.labels:
            lbl.set_fontsize(9)
            lbl.set_color("#111827")
    for cb in (style_check_left, style_check_right):
        for lbl in cb.labels:
            lbl.set_fontsize(8)
            lbl.set_color("#111827")
    for btn in (style_all_btn, style_none_btn):
        btn.label.set_fontsize(10)
        btn.label.set_color("#111827")

    for lbl in metric_radio.labels:
        lbl.set_fontsize(11)
        lbl.set_color("#111827")
    for lbl in color_radio.labels:
        lbl.set_fontsize(10)
        lbl.set_color("#111827")

    def selected_labels(labels: List[str], actives: List[bool]) -> set:
        return {label for label, on in zip(labels, actives) if on}

    def filter_rows() -> List[Dict]:
        out: List[Dict] = []
        selected_regions = selected_labels(regions, region_check.get_status())
        selected_styles = (
            selected_labels(style_left, style_check_left.get_status())
            | selected_labels(style_right, style_check_right.get_status())
        )
        for r in rows:
            if r["year"] is None:
                continue
            v = r.get(state["metric"])
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            if (r.get("region") or "（空）") not in selected_regions:
                continue
            tags = set(r.get("style_tags") or ["（空）"])
            if tags.isdisjoint(selected_styles):
                continue
            if state["search"]:
                s = (r.get("book", "") + " " + r.get("title", "")).lower()
                if state["search"].lower() not in s:
                    continue
            out.append(r)
        return out

    def set_checkbuttons(cb: CheckButtons, target: bool) -> None:
        status = cb.get_status()
        for i, on in enumerate(status):
            if on != target:
                cb.set_active(i)

    def on_style_all(_event) -> None:
        state["bulk_update"] = True
        set_checkbuttons(style_check_left, True)
        set_checkbuttons(style_check_right, True)
        state["bulk_update"] = False
        update_plot()

    def on_style_none(_event) -> None:
        state["bulk_update"] = True
        set_checkbuttons(style_check_left, False)
        set_checkbuttons(style_check_right, False)
        state["bulk_update"] = False
        update_plot()

    style_all_btn.on_clicked(on_style_all)
    style_none_btn.on_clicked(on_style_none)

    def update_plot(_=None) -> None:
        nonlocal ax_std
        if ax_std is not None:
            ax_std.remove()
            ax_std = None
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
        y_label = METRICS[metric]

        plot_rows: List[Dict] = []
        plot_samples: List[np.ndarray] = []
        plot_values: List[float] = []
        q = state.get("q", 0.10)
        q = max(0.0, min(0.25, float(q)))
        common_chars: List[str] = []
        per_book_vals: Dict[str, List[Tuple[str, float]]] = {}
        if sample_key:
            for r in current:
                book = r.get("book")
                char_map = char_means_by_book.get(book, {}) if book else {}
                vals = [
                    (c, float(v.get(sample_key, float("nan"))))
                    for c, v in char_map.items()
                    if math.isfinite(v.get(sample_key, float("nan")))
                ]
                if vals:
                    per_book_vals[book] = vals

            if state.get("common_chars") and per_book_vals:
                common_set = None
                for r in current:
                    book = r.get("book")
                    if book not in per_book_vals:
                        continue
                    keys = {c for c, _ in per_book_vals[book]}
                    if common_set is None:
                        common_set = keys
                    else:
                        common_set &= keys
                common_chars = sorted(common_set) if common_set else []

            per_book_filtered: Dict[str, List[Tuple[str, float]]] = {}
            for r in current:
                book = r.get("book")
                vals = per_book_vals.get(book, [])
                if not vals:
                    continue
                if state.get("common_chars"):
                    vals = [(c, v) for c, v in vals if c in common_chars]
                values_only = [v for _, v in vals]
                if q > 0.0 and len(values_only) >= 4:
                    lo = float(np.quantile(values_only, q))
                    hi = float(np.quantile(values_only, 1.0 - q))
                    vals = [(c, v) for c, v in vals if lo <= v <= hi]
                per_book_filtered[book] = vals

            if sample_key:
                common_toggle.labels[0].set_text(f"仅共同字 ({len(common_chars)})")
            else:
                common_toggle.labels[0].set_text("仅共同字 (—)")

            for r in current:
                book = r.get("book")
                vals = per_book_filtered.get(book, [])
                if state.get("common_chars"):
                    if not common_chars:
                        continue
                    vals = [(c, v) for c, v in vals if c in common_chars]
                val_list = [v for _, v in vals]
                if not val_list:
                    continue
                arr = np.asarray(val_list, dtype=float)
                plot_rows.append(r)
                plot_samples.append(arr)
        else:
            common_toggle.labels[0].set_text("仅共同字 (—)")
            for r in current:
                y = r.get(metric)
                if y is None or (isinstance(y, float) and math.isnan(y)):
                    continue
                plot_rows.append(r)
                plot_values.append(float(y))

        if not plot_rows:
            ax.set_title("无可绘制数据")
            fig.canvas.draw_idle()
            return

        means: List[float] = []
        stds: List[float] = []
        if sample_key:
            for arr in plot_samples:
                means.append(float(np.mean(arr)) if arr.size else float("nan"))
                stds.append(float(np.std(arr)) if arr.size else float("nan"))
        else:
            means = [float(v) for v in plot_values]

        # Influence list (computed before optional exclusion)
        base_years: List[int] = []
        for r in plot_rows:
            base_years.append(int(r.get("year") or 0))

        def compute_influence(values: List[float]) -> List[Dict]:
            valid_pairs = [
                (i, v)
                for i, v in enumerate(values)
                if math.isfinite(v) and plot_rows[i].get("year") is not None
            ]
            if len(valid_pairs) < 3:
                return []
            years = np.asarray([float(base_years[i]) for i, _ in valid_pairs], dtype=float)
            xs_full = rankdata(years).tolist()
            ys_full = [float(values[i]) for i, _ in valid_pairs]
            rho_full = spearman_rho(xs_full, ys_full)
            influence: List[Dict] = []
            for k, (i, _) in enumerate(valid_pairs):
                xs_loo = [xs_full[j] for j in range(len(xs_full)) if j != k]
                ys_loo = [ys_full[j] for j in range(len(ys_full)) if j != k]
                rho_loo = spearman_rho(xs_loo, ys_loo)
                influence.append({
                    "book": plot_rows[i].get("book"),
                    "title": plot_rows[i].get("title"),
                    "delta": rho_loo - rho_full,
                })
            return sorted(influence, key=lambda r: abs(r["delta"]), reverse=True)[:influence_slots]

        influence_books_u = compute_influence(means)
        influence_books_a = compute_influence(stds) if stds else []

        state["bulk_update"] = True
        keys: List[str] = []
        lines: List[str] = []
        if influence_books_u:
            lines.append("u 影响书籍（排序）")
            for row in influence_books_u:
                key = row.get("book") or ""
                delta = row.get("delta") or 0.0
                lines.append(f"{key} {delta:+.3f}" if key else "—")
                keys.append(key)
        else:
            lines.append("u 影响书籍：NA")

        if influence_books_a:
            lines.append("")
            lines.append("a 影响书籍（排序）")
            for row in influence_books_a:
                key = row.get("book") or ""
                delta = row.get("delta") or 0.0
                lines.append(f"{key} {delta:+.3f}" if key else "—")
                keys.append(key)
        elif stds:
            lines.append("")
            lines.append("a 影响书籍：NA")

        state["influence_keys"] = keys
        state["exclude_candidates"] = [k for k in keys if k]
        state["exclude_order"] = [r.get("book") or "" for r in plot_rows]

        ax_influence_list.clear()
        ax_influence_list.axis("off")
        ax_influence_list.set_title("影响书籍（排序）", fontsize=9, color="#111827")

        ax_stats.clear()
        ax_stats.axis("off")
        ax_labels.clear()
        ax_labels.axis("off")
        ax_exclude.clear()
        ax_exclude.axis("off")
        if plot_rows:
            ax_exclude.set_xlim(-0.5, len(plot_rows) - 0.5)
            ax_exclude.set_ylim(0.0, 1.0)
            for i, row in enumerate(plot_rows):
                book = row.get("book") or ""
                excluded = book in state["exclude_books"]
                # Excluded = "not participating in stats", but still visible. Use black stroke only (no gray).
                edge = "#111827"
                rect = plt.Rectangle((i - 0.18, 0.25), 0.36, 0.50, facecolor="none", edgecolor=edge, linewidth=1.0)
                ax_exclude.add_patch(rect)
                if excluded:
                    ax_exclude.text(i, 0.50, "×", ha="center", va="center", fontsize=8, color=edge, alpha=0.35)
        state["bulk_update"] = False

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

        excluded_flags = [r.get("book") in state["exclude_books"] for r in plot_rows]

        if sample_key:
            positions = list(range(len(plot_rows)))
            whis = (100.0 * q, 100.0 * (1.0 - q))
            box = ax.boxplot(
                plot_samples,
                positions=positions,
                widths=0.55,
                patch_artist=True,
                showmeans=True,
                meanline=False,
                whis=whis,
                showfliers=True,
            )
            for i, patch in enumerate(box["boxes"]):
                if excluded_flags[i]:
                    patch.set_facecolor("none")
                    patch.set_edgecolor("#111827")
                    patch.set_alpha(1.0)
                else:
                    patch.set_facecolor(colors[i])
                    patch.set_edgecolor(colors[i])
                    patch.set_alpha(0.25)
                patch.set_linewidth(1.2)
            for i, med in enumerate(box["medians"]):
                med.set_color(colors[i] if not excluded_flags[i] else "#111827")
                med.set_alpha(1.0 if not excluded_flags[i] else 0.35)
                med.set_linewidth(1.6)
            for i, mean in enumerate(box["means"]):
                mean.set_marker("D")
                mean.set_markerfacecolor(colors[i] if not excluded_flags[i] else "#111827")
                mean.set_markeredgecolor("white")
                mean.set_markersize(5.5)
                mean.set_alpha(1.0 if not excluded_flags[i] else 0.35)
            for line in box["whiskers"]:
                line.set_linewidth(1.0)
                line.set_color("#111827")
                line.set_alpha(0.45)
            for line in box["caps"]:
                line.set_linewidth(1.0)
                line.set_color("#111827")
                line.set_alpha(0.45)
            for flier in box["fliers"]:
                flier.set_marker("o")
                flier.set_markersize(2.6)
                flier.set_alpha(0.25)
                flier.set_markerfacecolor("#111827")
                flier.set_markeredgecolor("none")

            for i, (mu, _sd) in enumerate(zip(means, stds)):
                if not math.isfinite(mu):
                    continue
                label = f"μ={mu:.2f}"
                y_label_pos = mu
                ax.annotate(
                    label,
                    xy=(positions[i], y_label_pos),
                    xytext=(6, 8),
                    textcoords="offset points",
                    ha="left",
                    va="bottom",
                    fontsize=8,
                    color="#111827",
                    alpha=1.0 if not excluded_flags[i] else 0.35,
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", boxstyle="round,pad=0.2"),
                    zorder=5,
                )
            std_values = [sd for sd in stds if math.isfinite(sd)]
            if std_values:
                ax_std = ax.inset_axes([0.0, 0.02, 1.0, 0.34], transform=ax.transAxes)
                ax_std.set_zorder(ax.get_zorder() + 1)
                ax_std.patch.set_alpha(0.0)
                ax_std.set_xlim(-0.5, len(plot_rows) - 0.5)
                ax_std.set_xticks([])
                ax_std.yaxis.tick_right()
                ax_std.yaxis.set_label_position("right")
                ax_std.set_ylabel("a（字间差异）", fontsize=10, color="#111827")
                ax_std.tick_params(axis="y", labelsize=9, colors="#111827")
                ax_std.spines["right"].set_color("#111827")
                ax_std.spines["left"].set_visible(False)
                ax_std.spines["top"].set_visible(False)
                ax_std.spines["bottom"].set_visible(False)
                min_std = min(std_values)
                max_std = max(std_values)
                pad = max(0.005, (max_std - min_std) * 0.15)
                ax_std.set_ylim(min_std - pad, max_std + pad)
                for i, sd in enumerate(stds):
                    if not math.isfinite(sd):
                        continue
                    alpha = 0.9 if not excluded_flags[i] else 0.35
                    ax_std.scatter(
                        [positions[i]],
                        [sd],
                        marker="^",
                        s=38,
                        color=colors[i] if not excluded_flags[i] else "#111827",
                        edgecolors="white",
                        alpha=alpha,
                        zorder=6,
                    )
                    ax_std.annotate(
                        f"a={sd:.2f}",
                        xy=(positions[i], sd),
                        xytext=(6, -10),
                        textcoords="offset points",
                        ha="left",
                        va="top",
                        fontsize=8,
                        color="#111827",
                        alpha=1.0 if not excluded_flags[i] else 0.35,
                        bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", boxstyle="round,pad=0.2"),
                        zorder=6,
                    )
        else:
            for idx, y in enumerate(plot_values):
                alpha = 0.85 if not excluded_flags[idx] else 0.20
                ax.scatter([float(idx)], [y], c=[colors[idx]], s=36, alpha=alpha, edgecolors="none")

        ax.set_xlabel("")
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="y", labelsize=11, colors="#111827")
        if sample_key and plot_samples:
            max_val = max(float(np.max(vals)) for vals in plot_samples if vals.size)
            pad = max(0.05, max_val * 0.08)
            ax.set_ylim(0.0, max_val + pad)

        valid_pairs_u = [
            (r.get("year"), y)
            for r, y in zip(plot_rows, means)
            if math.isfinite(y) and r.get("year") is not None and r.get("book") not in state["exclude_books"]
        ]
        if len(valid_pairs_u) >= 2:
            years = np.asarray([float(y[0]) for y in valid_pairs_u], dtype=float)
            ys = [float(y[1]) for y in valid_pairs_u]
            xs = rankdata(years).tolist()
            stats_u = compute_trend_stats(xs, ys)
        else:
            stats_u = compute_trend_stats([], [])

        if stds:
            valid_pairs_a = [
                (r.get("year"), y)
                for r, y in zip(plot_rows, stds)
                if math.isfinite(y) and r.get("year") is not None and r.get("book") not in state["exclude_books"]
            ]
            if len(valid_pairs_a) >= 2:
                years_a = np.asarray([float(y[0]) for y in valid_pairs_a], dtype=float)
                ys_a = [float(y[1]) for y in valid_pairs_a]
                xs_a = rankdata(years_a).tolist()
                stats_a = compute_trend_stats(xs_a, ys_a)
            else:
                stats_a = compute_trend_stats([], [])
        else:
            stats_a = compute_trend_stats([], [])

        ax.set_title(f"{METRICS[state['metric']]}（n={len(plot_rows)}，Q={q:.2f}，year-rank）")

        def fmt_num(value: float, nd: int = 3) -> str:
            return "NA" if not math.isfinite(value) else f"{value:.{nd}f}"

        def fmt_ci(lo: float, hi: float, nd: int = 3) -> str:
            if not (math.isfinite(lo) and math.isfinite(hi)):
                return "CI95=[NA,NA]"
            return f"CI95=[{lo:.{nd}f},{hi:.{nd}f}]"

        stats_lines = [
            "u: rho={rho} p={p} {ci}".format(
                rho=fmt_num(float(stats_u.get("rho", float("nan")))),
                p=fmt_num(float(stats_u.get("p_rho", float("nan")))),
                ci=fmt_ci(float(stats_u.get("rho_ci_lo", float("nan"))), float(stats_u.get("rho_ci_hi", float("nan")))),
            ),
            "   tau={tau} p={pt} {tci}".format(
                tau=fmt_num(float(stats_u.get("tau", float("nan")))),
                pt=fmt_num(float(stats_u.get("p_tau", float("nan")))),
                tci=fmt_ci(float(stats_u.get("tau_ci_lo", float("nan"))), float(stats_u.get("tau_ci_hi", float("nan")))),
            ),
            "a: rho={rho} p={p} {ci}".format(
                rho=fmt_num(float(stats_a.get("rho", float("nan")))),
                p=fmt_num(float(stats_a.get("p_rho", float("nan")))),
                ci=fmt_ci(float(stats_a.get("rho_ci_lo", float("nan"))), float(stats_a.get("rho_ci_hi", float("nan")))),
            ),
            "   tau={tau} p={pt} {tci}".format(
                tau=fmt_num(float(stats_a.get("tau", float("nan")))),
                pt=fmt_num(float(stats_a.get("p_tau", float("nan")))),
                tci=fmt_ci(float(stats_a.get("tau_ci_lo", float("nan"))), float(stats_a.get("tau_ci_hi", float("nan")))),
            ),
        ]

        ax_stats.text(
            0.0,
            0.95,
            "\n".join(stats_lines),
            ha="left",
            va="top",
            fontsize=9,
            color="#111827",
            clip_on=True,
        )

        if lines:
            text = "\n".join(lines)
        else:
            text = "影响书籍：NA"
        ax_influence_list.text(
            0.0,
            1.0,
            text,
            va="top",
            ha="left",
            fontsize=9,
            color="#111827",
            clip_on=True,
        )

        labels = [r.get("book") or r.get("title") or "" for r in plot_rows]
        ticks = list(range(len(labels)))
        ax.set_xticks(ticks)
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.5, len(labels) - 0.5)
        for i, label in enumerate(labels):
            ax.text(
                i,
                0.0,
                label,
                transform=ax.get_xaxis_transform(),
                rotation=90,
                ha="right",
                va="top",
                fontsize=8,
                rotation_mode="anchor",
                clip_on=False,
            )

        if legend_items:
            handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=6) for _, c in legend_items]
            labels = [g for g, _ in legend_items]
            legend_title = "书局" if color_key == "place" else COLOR_KEYS.get(color_key, "")
            legend = ax.legend(handles, labels, loc="upper left", fontsize=9, frameon=False, title=legend_title)
            legend.get_title().set_fontsize(10)

        fig.canvas.draw_idle()

    def on_metric(label: str) -> None:
        state["metric"] = label
        update_plot()

    def on_color(label: str) -> None:
        state["color_by"] = label
        update_plot()

    metric_radio.on_clicked(on_metric)
    color_radio.on_clicked(on_color)
    # 搜索/重置控件已移除
    region_check.on_clicked(lambda _label: update_plot())
    style_check_left.on_clicked(lambda _label: update_plot() if not state.get("bulk_update") else None)
    style_check_right.on_clicked(lambda _label: update_plot() if not state.get("bulk_update") else None)
    place_toggle.on_clicked(lambda _label: update_plot())
    common_toggle.on_clicked(
        lambda _label: (
            state.update({
                "common_chars": common_toggle.get_status()[0],
            }),
            update_plot(),
        )
    )
    q_slider.on_changed(lambda val: (state.update({"q": float(val)}), update_plot()))

    def on_exclude_click(event) -> None:
        if state.get("bulk_update"):
            return
        if event.inaxes != ax_exclude:
            return
        if event.xdata is None:
            return
        idx = int(round(event.xdata))
        if idx < 0 or idx >= len(state.get("exclude_order", [])):
            return
        key = state["exclude_order"][idx]
        if not key:
            return
        if key in state["exclude_books"]:
            state["exclude_books"].discard(key)
        else:
            state["exclude_books"].add(key)
        update_plot()

    fig.canvas.mpl_connect("button_press_event", on_exclude_click)

    # 保持控件引用，避免被垃圾回收导致交互失效
    _widgets = {
        "metric": metric_radio,
        "color": color_radio,
        "region": region_check,
        "style_left": style_check_left,
        "style_right": style_check_right,
        "style_all": style_all_btn,
        "style_none": style_none_btn,
        "place": place_toggle,
        "common": common_toggle,
        "q": q_slider,
        "exclude_ax": ax_exclude,
        "stats_ax": ax_stats,
    }
    globals()["_TIME_EXPLORER_WIDGETS"] = _widgets

    update_plot()
    plt.show(block=True)


if __name__ == "__main__":
    main()
