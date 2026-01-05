#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
为每本书生成一个 PNG 拼贴图，仅使用分书文件中的最终切割结果，
并为每个小图添加淡色边框。

数据来源（唯一）：data/results/manual/review_books/*.json
  - 仅采集 segments 中 status == "confirmed" 且 decision != "drop" 的实例
  - 使用其中的 segmented_path 加载图片（相对项目根目录）

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
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from PIL import Image, ImageDraw
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REVIEW_BOOKS_DIR = PROJECT_ROOT / 'data/results/manual/review_books'
EXPORT_DIR = PROJECT_ROOT / 'data/exports/montage'


def load_review_segments() -> Dict:
    """返回切割状态视图（books->char->instance_id->entry），来自分片文件。"""
    out = { 'version': 2, 'books': {} }
    if not REVIEW_BOOKS_DIR.exists():
        return out
    for path in REVIEW_BOOKS_DIR.glob('*.json'):
        try:
            payload = json.load(open(path, 'r', encoding='utf-8'))
        except Exception:
            continue
        book = payload.get('book') or path.stem
        chars = payload.get('chars') or {}
        book_out = {}
        for char, char_obj in chars.items():
            if not isinstance(char_obj, dict):
                continue
            seg_map = char_obj.get('segments', {})
            if seg_map:
                book_out[char] = seg_map
        if book_out:
            out['books'][book] = book_out
    return out


def iter_confirmed_instances(review_data: Dict, book: str) -> List[Tuple[str, str, Path]]:
    """返回该书所有已确认且非 drop 的 (char, instance_id, abs_image_path)。"""
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
            if entry.get('decision') == 'drop':
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


def center_crop_fixed_box(img: Image.Image, Lb: int) -> Image.Image:
    """以图像中心为锚点，截取 Lb×Lb 的固定框，越界用白色填充。"""
    if Lb <= 0:
        return img
    gray = img.convert('L')
    w, h = gray.size
    cx, cy = w // 2, h // 2
    half = Lb // 2
    x0 = cx - half
    y0 = cy - half
    x1 = x0 + Lb
    y1 = y0 + Lb
    # 目标画布（白底）
    out = Image.new('L', (Lb, Lb), 255)
    # 源与目标重叠区域
    sx0 = max(0, x0)
    sy0 = max(0, y0)
    sx1 = min(w, x1)
    sy1 = min(h, y1)
    if sx0 < sx1 and sy0 < sy1:
        sub = gray.crop((sx0, sy0, sx1, sy1))
        dx = sx0 - x0
        dy = sy0 - y0
        out.paste(sub, (dx, dy))
    return out


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

    review = load_review_segments()
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
                p90 = float(np.percentile(np.array(long_sides, dtype=float), 90))
                L_b = int(np.ceil(p90 * 1.05))
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
