#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time-focused analysis for CharAnalysis.

Goal (simple, restart-friendly):
  - Compute book-level metrics from the analysis bundle (data/analysis)
  - Use core distribution (trimmed) rather than full extremes

Inputs
  - data/analysis/manifest.json
  - data/analysis/books/{book}/entries.json
  - data/analysis/books/{book}/images/*.png
  - data/metadata/books_metadata.csv

Outputs (default: data/analysis/time)
  - book_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def iter_book_entries(bundle_root: Path, book: str) -> List[Dict]:
    entries_path = bundle_root / "books" / book / "entries.json"
    with open(entries_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("entries") or []


def trimmed_mean_std(vals: List[float], q: float) -> Tuple[float, float]:
    vv = [v for v in vals if v is not None and math.isfinite(v)]
    if not vv:
        return (float("nan"), float("nan"))
    arr = np.asarray(vv, dtype=float)
    if q <= 0 or arr.size < 4:
        return (float(np.mean(arr)), float(np.std(arr, ddof=0)))
    lo = float(np.quantile(arr, q))
    hi = float(np.quantile(arr, 1.0 - q))
    core = arr[(arr >= lo) & (arr <= hi)]
    if core.size == 0:
        core = arr
    return (float(np.mean(core)), float(np.std(core, ddof=0)))


def compute_book_metrics(bundle_root: Path, book: str, threshold: int, core_q: float) -> Tuple[Dict, Dict[str, List[float]]]:
    entries = iter_book_entries(bundle_root, book)

    # Pass 1: load sizes, compute Lb
    instance_sizes: List[Tuple[str, str, int, int, Path]] = []
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
        inst = e.get("instance_id") or ""
        instance_sizes.append((char, inst, int(w), int(h), img_path))
        long_sides.append(max(int(w), int(h)))

    if not instance_sizes or not long_sides:
        return {
            "book": book,
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
        }, {"aspect_ratio": [], "face_ratio": [], "ink_coverage": []}

    # fixed box: P90 * 1.05 (same idea as montage)
    Lb = int(math.ceil(percentile(long_sides, 90) * 1.05))

    # Pass 2: per-instance metrics
    per_char: Dict[str, List[Tuple[float, float, float]]] = {}
    clipped = 0
    for char, _inst, w, h, img_path in instance_sizes:
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
                elif black > 0:
                    ink = float("nan")
        except Exception:
            pass

        per_char.setdefault(char, []).append((ar, face_ratio, ink))

    # char-weighted aggregation: mean per char, then stats across chars
    char_means: List[Tuple[float, float, float]] = []
    char_stds: List[Tuple[float, float, float]] = []
    for rows in per_char.values():
        ars = [r[0] for r in rows if not math.isnan(r[0])]
        frs = [r[1] for r in rows if not math.isnan(r[1])]
        inks = [r[2] for r in rows if r[2] is not None and math.isfinite(r[2])]
        if not ars or not frs:
            continue
        char_means.append((
            float(np.mean(np.asarray(ars, dtype=float))),
            float(np.mean(np.asarray(frs, dtype=float))),
            float(np.mean(np.asarray(inks, dtype=float))) if inks else float("nan"),
        ))
        if len(ars) >= 2:
            ar_std = float(np.std(np.asarray(ars, dtype=float), ddof=0))
        else:
            ar_std = float("nan")
        if len(frs) >= 2:
            fr_std = float(np.std(np.asarray(frs, dtype=float), ddof=0))
        else:
            fr_std = float("nan")
        if len(inks) >= 2:
            ic_std = float(np.std(np.asarray(inks, dtype=float), ddof=0))
        else:
            ic_std = float("nan")
        char_stds.append((ar_std, fr_std, ic_std))

    ar_mean, ar_std = trimmed_mean_std([x[0] for x in char_means], core_q)
    fr_mean, fr_std = trimmed_mean_std([x[1] for x in char_means], core_q)
    ic_mean, ic_std = trimmed_mean_std([x[2] for x in char_means], core_q)

    metrics = {
        "book": book,
        "instances": len(instance_sizes),
        "chars": len(char_means),
        "Lb": Lb,
        "clipping_rate": float(clipped) / float(max(1, len(instance_sizes))),
        "aspect_ratio_mean": ar_mean,
        "aspect_ratio_std": ar_std,
        "face_ratio_mean": fr_mean,
        "face_ratio_std": fr_std,
        "ink_coverage_mean": ic_mean,
        "ink_coverage_std": ic_std,
    }
    samples = {
        "aspect_ratio": [x[0] for x in char_means if not math.isnan(x[0])],
        "face_ratio": [x[1] for x in char_means if not math.isnan(x[1])],
        "ink_coverage": [x[2] for x in char_means if not math.isnan(x[2])],
        "aspect_ratio_std": [x[0] for x in char_stds if x[0] is not None and math.isfinite(x[0])],
        "face_ratio_std": [x[1] for x in char_stds if x[1] is not None and math.isfinite(x[1])],
        "ink_coverage_std": [x[2] for x in char_stds if x[2] is not None and math.isfinite(x[2])],
    }
    return metrics, samples


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def build_summary(bundle_root: Path, metadata_path: Path, out_dir: Path, threshold: int, core_quantile: float) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = bundle_root / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"找不到 bundle：{manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    books = sorted((manifest.get("books") or {}).keys())
    if not books:
        raise SystemExit("bundle 中没有任何 books。")

    meta_map = read_books_metadata(metadata_path)
    summary_rows: List[Dict] = []
    samples_by_book: Dict[str, Dict] = {}

    for book in books:
        metrics, samples = compute_book_metrics(
            bundle_root,
            book,
            threshold=int(threshold),
            core_q=float(core_quantile),
        )
        m = meta_map.get(book)
        summary_rows.append({
            "book": book,
            "title": m.title if m else "",
            "year": m.year if (m and m.year is not None) else "",
            "order": m.number if (m and m.number is not None) else "",
            "region": m.region if m else "",
            "province": m.province if m else "",
            "place": m.place if m else "",
            "style": m.style if m else "",
            "instances": metrics["instances"],
            "chars": metrics["chars"],
            "Lb": metrics["Lb"],
            "clipping_rate": f"{metrics['clipping_rate']:.6f}" if not math.isnan(metrics["clipping_rate"]) else "",
            "aspect_ratio_mean": f"{metrics['aspect_ratio_mean']:.6f}" if not math.isnan(metrics["aspect_ratio_mean"]) else "",
            "aspect_ratio_std": f"{metrics['aspect_ratio_std']:.6f}" if not math.isnan(metrics["aspect_ratio_std"]) else "",
            "face_ratio_mean": f"{metrics['face_ratio_mean']:.6f}" if not math.isnan(metrics["face_ratio_mean"]) else "",
            "face_ratio_std": f"{metrics['face_ratio_std']:.6f}" if not math.isnan(metrics["face_ratio_std"]) else "",
            "ink_coverage_mean": f"{metrics['ink_coverage_mean']:.6f}" if not math.isnan(metrics["ink_coverage_mean"]) else "",
            "ink_coverage_std": f"{metrics['ink_coverage_std']:.6f}" if not math.isnan(metrics["ink_coverage_std"]) else "",
            "threshold": int(threshold),
            "core_quantile": f"{float(core_quantile):.3f}",
        })
        samples_by_book[book] = {
            "order": m.number if (m and m.number is not None) else None,
            "year": m.year if (m and m.year is not None) else None,
            "title": m.title if m else "",
            "region": m.region if m else "",
            "province": m.province if m else "",
            "place": m.place if m else "",
            "style": m.style if m else "",
            "samples": samples,
        }

    fields = [
        "book",
        "title",
        "year",
        "order",
        "region",
        "province",
        "place",
        "style",
        "instances",
        "chars",
        "Lb",
        "clipping_rate",
        "aspect_ratio_mean",
        "aspect_ratio_std",
        "face_ratio_mean",
        "face_ratio_std",
        "ink_coverage_mean",
        "ink_coverage_std",
        "threshold",
        "core_quantile",
    ]
    summary_path = out_dir / "book_summary.csv"
    write_csv(summary_path, summary_rows, fieldnames=fields)
    samples_payload = {
        "core_quantile": float(core_quantile),
        "threshold": int(threshold),
        "books": samples_by_book,
    }
    with open(out_dir / "book_samples.json", "w", encoding="utf-8") as f:
        json.dump(samples_payload, f, ensure_ascii=False, indent=2)
    return summary_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Time-focused analysis for analysis bundle.")
    ap.add_argument("--bundle", type=str, default="data/analysis", help="Bundle root folder (default: data/analysis)")
    ap.add_argument("--metadata", type=str, default="data/metadata/books_metadata.csv", help="Books metadata CSV")
    ap.add_argument("--out", type=str, default="data/analysis/time", help="Output folder (default: data/analysis/time)")
    ap.add_argument("--threshold", type=int, default=160, help="Black threshold (pixel < thr). Default 160")
    ap.add_argument("--core-quantile", type=float, default=0.10, help="Trim quantile for core distribution (default 0.10)")
    args = ap.parse_args()

    summary_path = build_summary(
        bundle_root=PROJECT_ROOT / args.bundle,
        metadata_path=PROJECT_ROOT / args.metadata,
        out_dir=PROJECT_ROOT / args.out,
        threshold=int(args.threshold),
        core_quantile=float(args.core_quantile),
    )
    print(f"✓ 输出：{summary_path}")


if __name__ == "__main__":
    main()
