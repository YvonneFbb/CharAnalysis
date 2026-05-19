#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
为每本书生成一个 PNG 拼贴图，仅使用分书文件中的最终切割结果，
并为每个小图添加淡色边框。

数据来源（唯一）：data/results/manual/review_books/*.json
  - 仅采集 items[*].review 中 status == "confirmed" 且 decision != "drop" 的实例
  - 使用 review.confirmed_path 加载图片（相对项目根目录）

输出：data/exports/montage/{book}.png

可选参数（命令行）：
  --books BOOK [BOOK ...]     仅处理指定书名（默认处理所有书）
  --tile-size N               每个小图目标尺寸（正方形边长，默认 64）
  --cols N                    每行列数（默认 50）
  --border N                  小图淡色边框像素（默认 1）
  --bg #RRGGBB                背景色（默认 #FFFFFF）

用法示例：
  python src/analysis/generate_book_montage.py
  python src/analysis/generate_book_montage.py --books 01_1127_尚书正义 --tile-size 72 --cols 40
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXPORT_DIR = PROJECT_ROOT / 'data/exports/montage'

from src.analysis.image_metrics import center_crop_fixed_box, percentile
from src.analysis.montage import build_montage, hex_to_rgb, make_tile
from src.review.identity import get_confirmed_path
from src.review.storage.review_books import iter_confirmed_items, list_review_books, read_review_book


def iter_confirmed_instances(book: str) -> List[Tuple[str, str, Path]]:
    """返回该书所有已确认且非 drop 的 (char, instance_id, abs_image_path)。"""
    out: List[Tuple[str, str, Path]] = []
    book_obj = read_review_book(book) or {}
    for ch, inst_id, item in iter_confirmed_items(book_obj):
        seg_rel = get_confirmed_path(item.get('review') or {})
        if not seg_rel:
            continue
        abs_path = PROJECT_ROOT / seg_rel
        out.append((ch, inst_id, abs_path))
    # 排序：按字符、实例ID
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def open_confirmed_image(abs_path: Path) -> Optional[Image.Image]:
    """打开已确认的切割图（绝对路径），失败返回 None。"""
    try:
        if abs_path.exists():
            img = Image.open(abs_path)
            # 转成 L 以减小体积且统一背景
            if img.mode != 'L':
                img = img.convert('L')
            return img
    except Exception:
        return None


def list_books() -> List[str]:
    books = list_review_books()
    return books


def main():
    ap = argparse.ArgumentParser(description='Generate per-book character montage PNGs.')
    ap.add_argument('--books', nargs='+', help='Specify book names to process (default: all)')
    ap.add_argument('--tile-size', type=int, default=64, help='Tile size in pixels (square). Default 64')
    ap.add_argument('--cols', type=int, default=50, help='Columns per row. Default 50')
    ap.add_argument('--border', type=int, default=1, help='Light border width in pixels. Default 1')
    ap.add_argument('--bg', type=str, default='#FFFFFF', help='Background color hex. Default #FFFFFF')
    ap.add_argument('--use-fixed-box', action='store_true', help='Use per-book fixed box (P90 long side * 1.05) for montage display')
    args = ap.parse_args()

    tile_size: int = max(8, args.tile_size)
    cols: int = max(1, args.cols)
    border: int = max(0, args.border)
    try:
        bg_color = hex_to_rgb(args.bg)
    except Exception:
        bg_color = (255, 255, 255)

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    books = args.books if args.books else list_books()

    if not books:
        print('未找到任何书籍（review_books 为空）。')
        return

    for book in books:
        print(f'▶ 处理书籍：{book}')
        items = iter_confirmed_instances(book)
        # 如果用户限定了书名但该书没有已确认实例，仍生成一张空白提示？这里选择跳过并提示
        tiles: List[Image.Image] = []
        missing = 0
        L_b = None
        if args.use_fixed_box:
            long_sides = []
            for _, _, abs_path in items:
                try:
                    with Image.open(abs_path) as im:
                        w, h = im.size
                        long_sides.append(max(w, h))
                except Exception:
                    continue
            if long_sides:
                L_b = int(math.ceil(percentile(long_sides, 90) * 1.05))
                print(f'  • Fixed box L_b={L_b} (P90 * 1.05)')
            else:
                L_b = 0
        for ch, inst_id, abs_path in items:
            img = open_confirmed_image(abs_path)
            if img is None:
                missing += 1
                continue
            if args.use_fixed_box and L_b and L_b > 0:
                roi = center_crop_fixed_box(img, L_b)
                # 将固定框 ROI 缩放到 tile（保持统一固定框视角，仅显示窗口，不改变字形与框的相对关系）
                # 这里复用 make_tile 以统一边框与留白（roi 近似正方形，缩放效果一致）
                tile = make_tile(roi, tile_size=tile_size, border=border, bg_color=bg_color)
            else:
                tile = make_tile(img, tile_size=tile_size, border=border, bg_color=bg_color)
            tiles.append(tile)

        if not tiles:
            print(f'  ⚠️ 无确认实例或图片缺失，跳过：{book}')
            continue

        montage = build_montage(tiles, cols=cols, tile_size=tile_size, bg_color=bg_color)
        out_path = EXPORT_DIR / f'{book}.png'
        montage.save(out_path)
        print(f'  ✓ 已生成：{out_path}（{len(tiles)} 张，缺失 {missing}）')


if __name__ == '__main__':
    main()
