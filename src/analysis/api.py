#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared analysis API for CLI explorers and paper export scripts.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.analysis.dataset import load_manifest_books, make_book_row, read_books_metadata
from src.analysis.image_metrics import compute_book_metrics
from src.analysis.stats import rankdata, spearman_rho


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


def compute_book_data(
    bundle_root: Path, book: str, threshold: int
) -> Tuple[Dict, Dict[str, List[float]], Dict[str, Dict[str, float]]]:
    return compute_book_metrics(bundle_root, book, threshold=threshold)


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
        rows.append(make_book_row(book, meta_map, metrics))
        samples_by_book[book] = samples
        char_means_by_book[book] = char_means

    return rows, samples_by_book, char_means_by_book
