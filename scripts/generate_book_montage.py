#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
为每本书生成一个 PNG 拼贴图，严格只使用 segmentation_review.json 中“已确认”的最终切割图，
并为每个小图添加淡色边框。

数据来源（唯一）：data/results/segmentation_review.json
  - 仅采集 status == "confirmed" 的实例
  - 使用其中的 segmented_path 加载图片（相对项目根目录）

输出：data/exports/montage/{book}.png

可选参数（命令行）：
  --books BOOK [BOOK ...]     仅处理指定书名（默认处理所有书）
  --tile-size N               每个小图目标尺寸（正方形边长，默认 64）
  --cols N                    每行列数（默认 50）
  --border N                  小图淡色边框像素（默认 1）
  --bg #RRGGBB                背景色（默认 #FFFFFF）

用法示例：
  python scripts/generate_book_montage.py
  python scripts/generate_book_montage.py --books 01_1127_尚书正义 --tile-size 72 --cols 40
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEW_PATH = PROJECT_ROOT / 'data/results/segmentation_review.json'
EXPORT_DIR = PROJECT_ROOT / 'data/exports/montage'


def load_review() -> Dict:
    if REVIEW_PATH.exists():
        with open(REVIEW_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return { 'version': 1, 'books': {} }


def iter_confirmed_instances(review_data: Dict, book: str) -> List[Tuple[str, str, Path]]:
    """返回该书所有已确认的 (char, instance_id, abs_image_path)。"""
    out: List[Tuple[str, str, Path]] = []
    books = review_data.get('books', {})
    if book not in books:
        return out
    book_obj: Dict = books[book]
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


def make_tile(img: Image.Image, tile_size: int, border: int, bg_color: Tuple[int, int, int]=(255, 255, 255)) -> Image.Image:
    """将任意尺寸的 img 等比缩放放入 tile_size 方块，四周留白并绘制淡色边框。"""
    canvas = Image.new('RGB', (tile_size, tile_size), bg_color)
    if img is None:
        # 空图：直接画边框
        draw = ImageDraw.Draw(canvas)
        border_color = (210, 210, 210)  # 淡灰
        for b in range(border):
            draw.rectangle([b, b, tile_size - 1 - b, tile_size - 1 - b], outline=border_color)
        return canvas

    # 等比缩放，留2*border的安全边
    max_w = tile_size - 2*border - 2
    max_h = tile_size - 2*border - 2
    if max_w <= 0 or max_h <= 0:
        max_w = max_h = max(1, tile_size - 2)

    w, h = img.size
    scale = min(max_w / max(1, w), max_h / max(1, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img_resized = img.resize((new_w, new_h), Image.BICUBIC)

    # 居中粘贴
    ox = (tile_size - new_w) // 2
    oy = (tile_size - new_h) // 2
    canvas.paste(img_resized, (ox, oy))

    # 画淡色边框
    draw = ImageDraw.Draw(canvas)
    border_color = (210, 210, 210)
    for b in range(border):
        draw.rectangle([b, b, tile_size - 1 - b, tile_size - 1 - b], outline=border_color)

    return canvas


def build_montage(tiles: List[Image.Image], cols: int, tile_size: int, bg_color: Tuple[int, int, int]=(255, 255, 255)) -> Image.Image:
    if not tiles:
        return Image.new('RGB', (tile_size, tile_size), bg_color)
    cols = max(1, cols)
    rows = math.ceil(len(tiles) / cols)
    W = cols * tile_size
    H = rows * tile_size
    out = Image.new('RGB', (W, H), bg_color)
    for idx, tile in enumerate(tiles):
        r = idx // cols
        c = idx % cols
        out.paste(tile, (c * tile_size, r * tile_size))
    return out


def list_books(review_data: Dict) -> List[str]:
    books = list((review_data.get('books') or {}).keys())
    books.sort()
    return books


def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    s = hex_str.strip().lstrip('#')
    if len(s) == 6:
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
        return (r, g, b)
    raise ValueError('Invalid hex color, expect #RRGGBB')


def main():
    ap = argparse.ArgumentParser(description='Generate per-book character montage PNGs.')
    ap.add_argument('--books', nargs='+', help='Specify book names to process (default: all)')
    ap.add_argument('--tile-size', type=int, default=64, help='Tile size in pixels (square). Default 64')
    ap.add_argument('--cols', type=int, default=50, help='Columns per row. Default 50')
    ap.add_argument('--border', type=int, default=1, help='Light border width in pixels. Default 1')
    ap.add_argument('--bg', type=str, default='#FFFFFF', help='Background color hex. Default #FFFFFF')
    args = ap.parse_args()

    tile_size: int = max(8, args.tile_size)
    cols: int = max(1, args.cols)
    border: int = max(0, args.border)
    try:
        bg_color = hex_to_rgb(args.bg)
    except Exception:
        bg_color = (255, 255, 255)

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    review = load_review()
    books = args.books if args.books else list_books(review)

    if not books:
        print('未找到任何书籍（segmented 或 lookup 都为空）。')
        return

    for book in books:
        print(f'▶ 处理书籍：{book}')
        items = iter_confirmed_instances(review, book)
        # 如果用户限定了书名但该书没有已确认实例，仍生成一张空白提示？这里选择跳过并提示
        tiles: List[Image.Image] = []
        missing = 0
        for ch, inst_id, abs_path in items:
            img = open_confirmed_image(abs_path)
            if img is None:
                missing += 1
                continue
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
