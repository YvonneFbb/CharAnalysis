#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从第二轮审查结果（data/results/segmentation_review.json）读取已确认的实例，
为每本书计算基础特征并输出：

- JSON：data/analysis/features/{book}.json
  - characters -> 每个字符的样本与聚合统计
- CSV： data/analysis/summaries/{book}_summary.csv
  - 按字符的聚合概览

仅使用 status == 'confirmed' 的条目，并从 segmented_path 读取最终图片。

用法：
  python scripts/analyze_seg_review.py
  python scripts/analyze_seg_review.py --books 01_1127_尚书正义 02_XXXX --output data/analysis

"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from PIL import Image
import numpy as np

try:
    # Python 3.8+
    from statistics import mean, median, pstdev, quantiles
except Exception:  # pragma: no cover
    # 兜底：简单实现（不推荐），但尽量避免运行到这里
    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    def median(xs):
        xs2 = sorted(xs)
        n = len(xs2)
        if n == 0:
            return 0.0
        mid = n // 2
        if n % 2:
            return float(xs2[mid])
        return (xs2[mid - 1] + xs2[mid]) / 2.0

    def pstdev(xs):
        if not xs:
            return 0.0
        m = mean(xs)
        return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5

    def quantiles(xs, n=4):  # very rough quartiles
        xs2 = sorted(xs)
        if not xs2:
            return [0.0] * (n - 1)
        q1_idx = max(0, int(0.25 * (len(xs2) - 1)))
        q3_idx = max(0, int(0.75 * (len(xs2) - 1)))
        return [float(xs2[q1_idx]), float(xs2[q3_idx])]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEW_JSON = PROJECT_ROOT / 'data/results/segmentation_review.json'


@dataclass
class SampleFeature:
    instance_id: str
    path: str
    width: int
    height: int
    aspect_ratio: Optional[float]


def load_review() -> Dict[str, Any]:
    if not REVIEW_JSON.exists():
        raise FileNotFoundError(f'未找到审查结果文件: {REVIEW_JSON}')
    with open(REVIEW_JSON, 'r', encoding='utf-8') as f:
        return json.load(f)


def iter_confirmed(book: str, review: Dict[str, Any]) -> List[Tuple[str, str, Path]]:
    """返回该书所有（char, instance_id, abs_path）仅限 confirmed。"""
    out: List[Tuple[str, str, Path]] = []
    book_obj = (review.get('books') or {}).get(book, {})
    for ch, inst_map in book_obj.items():
        if not isinstance(inst_map, dict):
            continue
        for inst_id, entry in inst_map.items():
            if not isinstance(entry, dict):
                continue
            if entry.get('status') != 'confirmed':
                continue
            seg_rel = entry.get('segmented_path')
            if not seg_rel:
                continue
            abs_path = PROJECT_ROOT / seg_rel
            out.append((ch, inst_id, abs_path))
    # 排序稳定
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def open_image_size(p: Path) -> Optional[Tuple[int, int]]:
    try:
        if not p.exists():
            return None
        with Image.open(p) as im:
            w, h = im.size
            return int(w), int(h)
    except Exception:
        return None


def compute_statistics(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            'mean': 0.0, 'std': 0.0, 'median': 0.0,
            'min': 0.0, 'max': 0.0, 'q1': 0.0, 'q3': 0.0,
            'count': 0
        }
    vals = list(map(float, values))
    q1, q3 = 0.0, 0.0
    try:
        qs = quantiles(vals, n=4)
        # statistics.quantiles returns 3 cut points by default if n=4 in Python 3.11,
        # but for compatibility we handle both 2 or 3 outputs.
        if len(qs) >= 3:
            q1, q3 = float(qs[0]), float(qs[2])
        elif len(qs) == 2:
            q1, q3 = float(qs[0]), float(qs[1])
    except Exception:
        q1, q3 = 0.0, 0.0

    return {
        'mean': float(mean(vals)),
        'std': float(pstdev(vals)),
        'median': float(median(vals)),
        'min': float(min(vals)),
        'max': float(max(vals)),
        'q1': float(q1),
        'q3': float(q3),
        'count': int(len(vals)),
    }


def analyze_book(book: str, review: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    返回：
    - features_json（用于写入 JSON 文件）
    - summary_rows（用于写入 CSV，每行一个字符）
    """
    items = iter_confirmed(book, review)
    ts = datetime.now().isoformat(timespec='seconds')
    char_map: Dict[str, Dict[str, Any]] = {}
    missing = 0

    # ==================== per-book Fixed Box 标定（P95 of long side）====================
    long_sides: List[int] = []
    paths: List[Tuple[str, str, Path]] = []
    for ch, inst_id, abs_path in items:
        paths.append((ch, inst_id, abs_path))
        try:
            with Image.open(abs_path) as im:
                w, h = im.size
                long_sides.append(int(max(w, h)))
        except Exception:
            missing += 1
            continue

    if long_sides:
        p95 = float(np.percentile(np.array(long_sides, dtype=float), 95))
        L_b = int(np.ceil(p95 * 1.05))  # 加 5% 安全边界
    else:
        L_b = 0

    def _compute_gray_ratio_bw(im: Image.Image, Lb: int) -> Optional[float]:
        if Lb <= 0:
            return None
        gray = im.convert('L')
        w, h = gray.size
        # 以图像中心为锚点，截取 Lb×Lb 的居中窗口（超出边界的部分视为白）
        cx, cy = w // 2, h // 2
        half = Lb // 2
        x0 = max(0, cx - half)
        y0 = max(0, cy - half)
        x1 = min(w, cx - half + Lb)
        y1 = min(h, cy - half + Lb)
        if x0 >= x1 or y0 >= y1:
            # 盒子完全落在图外，视为全白
            black = 0
        else:
            sub = np.array(gray.crop((x0, y0, x1, y1)), dtype=np.uint8)
            # 简单阈值（多数切割图已接近黑白），避开外部依赖
            black_mask = sub < 128
            black = int(np.count_nonzero(black_mask))
        box_area = int(Lb * Lb)
        white = max(box_area - black, 0)
        if white == 0:
            return float('inf') if black > 0 else 0.0
        return float(black) / float(white)

    # 第二遍：计算特征
    for ch, inst_id, abs_path in paths:
        size = open_image_size(abs_path)
        if not size:
            missing += 1
            continue
        w, h = size
        ar = float(h) / float(w) if w > 0 else None

        # 几何字面率（参考定义）：(w+h)/2 相对固定框边长 L_b
        geom_face_ratio = None
        if L_b > 0:
            geom_face_ratio = float((w + h) / 2.0) / float(L_b)

        # 灰度比（黑/白面积比例，基于 per-book 固定框，不缩放字形）
        gray_ratio_bw = None
        try:
            with Image.open(abs_path) as im:
                gray_ratio_bw = _compute_gray_ratio_bw(im, L_b)
        except Exception:
            pass

        entry = char_map.setdefault(ch, {
            'count': 0,
            'samples': [],
            'stats': {}
        })
        entry['count'] += 1
        entry['samples'].append({
            'instance_id': inst_id,
            'path': str(abs_path),
            'image_size': {'width': w, 'height': h},
            'features': {
                'width': w,
                'height': h,
                'aspect_ratio': ar,
                'geom_face_ratio': geom_face_ratio,
                'gray_ratio_bw': gray_ratio_bw
            }
        })

    # 统计每字符的聚合
    summaries: List[Dict[str, Any]] = []
    for ch, data in char_map.items():
        widths = [s['features']['width'] for s in data['samples']]
        heights = [s['features']['height'] for s in data['samples']]
        ars = [s['features']['aspect_ratio'] for s in data['samples'] if s['features']['aspect_ratio'] is not None]
        gfr = [s['features'].get('geom_face_ratio') for s in data['samples'] if s['features'].get('geom_face_ratio') is not None]
        grb = [s['features'].get('gray_ratio_bw') for s in data['samples'] if s['features'].get('gray_ratio_bw') is not None and np.isfinite(s['features'].get('gray_ratio_bw'))]

        data['stats']['width'] = compute_statistics(list(map(float, widths)))
        data['stats']['height'] = compute_statistics(list(map(float, heights)))
        data['stats']['aspect_ratio'] = compute_statistics(list(map(float, ars))) if ars else compute_statistics([])
        data['stats']['geom_face_ratio'] = compute_statistics(list(map(float, gfr))) if gfr else compute_statistics([])
        data['stats']['gray_ratio_bw'] = compute_statistics(list(map(float, grb))) if grb else compute_statistics([])

        summaries.append({
            'char': ch,
            'samples': data['count'],
            'width_mean': data['stats']['width']['mean'],
            'width_median': data['stats']['width']['median'],
            'width_min': data['stats']['width']['min'],
            'width_max': data['stats']['width']['max'],
            'height_mean': data['stats']['height']['mean'],
            'height_median': data['stats']['height']['median'],
            'height_min': data['stats']['height']['min'],
            'height_max': data['stats']['height']['max'],
            'ar_mean': data['stats']['aspect_ratio']['mean'],
            'ar_median': data['stats']['aspect_ratio']['median'],
            'ar_min': data['stats']['aspect_ratio']['min'],
            'ar_max': data['stats']['aspect_ratio']['max'],
        })

    # 总体信息
    features_json = {
        'book_name': book,
        'metadata': {
            'generated_at': ts,
            'total_samples': sum(v['count'] for v in char_map.values()),
            'unique_chars': len(char_map),
            'missing_images': missing,
            'source': str(REVIEW_JSON.relative_to(PROJECT_ROOT)),
            'fixed_box': {
                'long_side_p95': L_b
            }
        },
        'characters': char_map
    }

    # 按字符排序
    summaries.sort(key=lambda r: r['char'])

    return features_json, summaries


def write_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'char', 'samples',
        'width_mean', 'width_median', 'width_min', 'width_max',
        'height_mean', 'height_median', 'height_min', 'height_max',
        'ar_mean', 'ar_median', 'ar_min', 'ar_max'
    ]
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    os.replace(tmp, path)


def list_books(review: Dict[str, Any]) -> List[str]:
    books = list((review.get('books') or {}).keys())
    books.sort()
    return books


def safe_name(name: str) -> str:
    return name.replace('/', '_').replace(' ', '_')


def main():
    ap = argparse.ArgumentParser(description='Analyze confirmed segmentation results per book.')
    ap.add_argument('--books', nargs='+', help='仅处理指定书名（默认全部）')
    ap.add_argument('--output', default=str(PROJECT_ROOT / 'data/analysis'), help='输出根目录（默认 data/analysis）')
    args = ap.parse_args()

    review = load_review()
    books = args.books if args.books else list_books(review)

    out_root = Path(args.output)
    out_features = out_root / 'features'
    out_summaries = out_root / 'summaries'

    if not books:
        print('未在 segmentation_review.json 中找到任何书籍。')
        return

    for book in books:
        print(f'▶ 处理书籍：{book}')
        features_json, summary_rows = analyze_book(book, review)

        json_path = out_features / f'{safe_name(book)}.json'
        csv_path = out_summaries / f'{safe_name(book)}_summary.csv'
        write_json(features_json, json_path)
        write_csv(summary_rows, csv_path)
        print(f'  ✓ JSON: {json_path}')
        print(f'  ✓ CSV : {csv_path}  （字符数={len(summary_rows)}，样本总数={features_json["metadata"]["total_samples"]}）')


if __name__ == '__main__':
    main()
