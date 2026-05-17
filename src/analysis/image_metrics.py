#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared image metric calculations for analysis scripts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from src.analysis.dataset import iter_book_entries


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


def mean_or_nan(values: List[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=float)))


def compute_book_metrics(
    bundle_root: Path,
    book: str,
    threshold: int,
) -> Tuple[Dict, Dict[str, List[float]], Dict[str, Dict[str, float]]]:
    entries = iter_book_entries(bundle_root, book)

    instance_sizes: List[Tuple[str, int, int, Path]] = []
    long_sides: List[int] = []
    for entry in entries:
        rel = entry.get("image")
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
        char = entry.get("char") or ""
        instance_sizes.append((char, int(w), int(h), img_path))
        long_sides.append(max(int(w), int(h)))

    empty_metrics = {
        "instances": 0,
        "chars": 0,
        "Lb": 0,
        "clipping_rate": float("nan"),
        "aspect_ratio": float("nan"),
        "face_ratio": float("nan"),
        "ink_coverage": float("nan"),
    }
    empty_samples = {
        "aspect_ratio": [],
        "face_ratio": [],
        "ink_coverage": [],
    }
    if not instance_sizes or not long_sides:
        return empty_metrics, empty_samples, {}

    Lb = int(math.ceil(percentile(long_sides, 90) * 1.05))
    per_char: Dict[str, List[Tuple[float, float, float]]] = {}
    clipped = 0

    for char, w, h, img_path in instance_sizes:
        if not char:
            continue
        aspect_ratio = float(h) / float(max(1, w))
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

        per_char.setdefault(char, []).append((aspect_ratio, face_ratio, ink))

    char_means: List[Tuple[float, float, float]] = []
    char_means_map: Dict[str, Dict[str, float]] = {}
    for char, rows in per_char.items():
        aspect_values = [r[0] for r in rows if math.isfinite(r[0])]
        face_values = [r[1] for r in rows if math.isfinite(r[1])]
        ink_values = [r[2] for r in rows if math.isfinite(r[2])]
        if not aspect_values or not face_values:
            continue
        aspect_mean = mean_or_nan(aspect_values)
        face_mean = mean_or_nan(face_values)
        ink_mean = mean_or_nan(ink_values)
        char_means.append((aspect_mean, face_mean, ink_mean))
        char_means_map[char] = {
            "aspect_ratio": aspect_mean,
            "face_ratio": face_mean,
            "ink_coverage": ink_mean,
        }

    samples = {
        "aspect_ratio": [x[0] for x in char_means if math.isfinite(x[0])],
        "face_ratio": [x[1] for x in char_means if math.isfinite(x[1])],
        "ink_coverage": [x[2] for x in char_means if math.isfinite(x[2])],
    }
    metrics = {
        "instances": len(instance_sizes),
        "chars": len(char_means),
        "Lb": Lb,
        "clipping_rate": float(clipped) / float(max(1, len(instance_sizes))),
        "aspect_ratio": mean_or_nan(samples["aspect_ratio"]),
        "face_ratio": mean_or_nan(samples["face_ratio"]),
        "ink_coverage": mean_or_nan(samples["ink_coverage"]),
    }
    return metrics, samples, char_means_map
