#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Collect final review results into a single analysis folder.

Output structure (default: data/analysis):
  analysis/
    manifest.json
    montage/{book}.png
    books/{book}/entries.json
    books/{book}/images/{char}_{instance_id}.png
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEW_BOOKS_DIR = PROJECT_ROOT / 'data/results/review_books'
SEGMENTED_DIR = PROJECT_ROOT / 'data/results/segmented'


def list_review_books() -> List[str]:
    if not REVIEW_BOOKS_DIR.exists():
        return []
    return sorted([p.stem for p in REVIEW_BOOKS_DIR.glob('*.json')])


def read_review_book(book_name: str) -> Optional[Dict]:
    path = REVIEW_BOOKS_DIR / f'{book_name}.json'
    if not path.exists():
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
    except Exception:
        return None
    if isinstance(payload, dict):
        if 'chars' in payload and isinstance(payload['chars'], dict):
            return payload['chars']
        if 'books' in payload and isinstance(payload['books'], dict):
            return payload['books'].get(book_name)
    return None


def percentile(values: List[int], p: float) -> int:
    if not values:
        return 0
    values_sorted = sorted(values)
    k = max(0, math.ceil((p / 100.0) * len(values_sorted)) - 1)
    return int(values_sorted[k])


def center_crop_fixed_box(img: Image.Image, Lb: int) -> Image.Image:
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
    out = Image.new('L', (Lb, Lb), 255)
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


def make_tile(
    img: Image.Image,
    tile_size: int,
    border: int,
    bg_color: Tuple[int, int, int],
    scale: bool = True
) -> Image.Image:
    canvas = Image.new('RGB', (tile_size, tile_size), bg_color)
    if img is None:
        draw = ImageDraw.Draw(canvas)
        border_color = (210, 210, 210)
        for b in range(border):
            draw.rectangle([b, b, tile_size - 1 - b, tile_size - 1 - b], outline=border_color)
        return canvas

    w, h = img.size
    if scale:
        max_w = tile_size - 2 * border - 2
        max_h = tile_size - 2 * border - 2
        if max_w <= 0 or max_h <= 0:
            max_w = max_h = max(1, tile_size - 2)
        ratio = min(max_w / max(1, w), max_h / max(1, h))
        new_w = max(1, int(round(w * ratio)))
        new_h = max(1, int(round(h * ratio)))
        img_resized = img.resize((new_w, new_h), Image.BICUBIC)
    else:
        img_resized = img
        new_w, new_h = img.size

    ox = (tile_size - new_w) // 2
    oy = (tile_size - new_h) // 2
    canvas.paste(img_resized.convert('RGB'), (ox, oy))

    draw = ImageDraw.Draw(canvas)
    border_color = (210, 210, 210)
    for b in range(border):
        draw.rectangle([b, b, tile_size - 1 - b, tile_size - 1 - b], outline=border_color)
    return canvas


def build_montage(tiles: List[Image.Image], cols: int, tile_size: int, bg_color: Tuple[int, int, int]) -> Image.Image:
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


def collect_book_entries(book_name: str, use_fixed_box: bool) -> Tuple[List[Dict], Dict]:
    book_obj = read_review_book(book_name) or {}
    entries: List[Dict] = []
    missing_images = 0
    skipped_unmatched = 0
    missing_lookup = 0

    for char, char_obj in book_obj.items():
        if not isinstance(char_obj, dict):
            continue
        segments = char_obj.get('segments') or {}
        lookup = char_obj.get('lookup') or {}
        for inst_id, seg in segments.items():
            if not isinstance(seg, dict):
                continue
            if seg.get('status') != 'confirmed':
                continue
            if seg.get('decision') == 'drop':
                continue
            seg_rel = seg.get('segmented_path')
            if not seg_rel:
                missing_images += 1
                continue
            abs_path = PROJECT_ROOT / seg_rel
            if not abs_path.exists():
                missing_images += 1
                continue
            info = lookup.get(inst_id) or {}
            if not info:
                missing_lookup += 1
            entries.append({
                'char': char,
                'instance_id': inst_id,
                'segmented_path': seg_rel,
                'image_name': abs_path.name,
                'status': seg.get('status'),
                'decision': seg.get('decision'),
                'method': seg.get('method'),
                'timestamp': seg.get('timestamp'),
                'bbox': info.get('bbox'),
                'source_image': info.get('source_image'),
                'confidence': info.get('confidence'),
                'volume': info.get('volume'),
                'page': info.get('page'),
                'char_index': info.get('char_index'),
                'index': info.get('index'),
                'lookup_missing': not bool(info)
            })

    meta = {
        'total': len(entries),
        'missing_images': missing_images,
        'skipped_unmatched': skipped_unmatched,
        'missing_lookup': missing_lookup
    }
    return entries, meta


def main():
    ap = argparse.ArgumentParser(description='Collect final review results into analysis folder.')
    ap.add_argument('--out', type=str, default='data/analysis', help='Output folder (default: data/analysis)')
    ap.add_argument('--books', nargs='+', help='Specify book names to process (default: all)')
    ap.add_argument('--clean', action='store_true', help='Clean output folder before writing')
    ap.add_argument('--tile-size', type=int, default=64, help='Montage tile size (default 64)')
    ap.add_argument('--cols', type=int, default=50, help='Montage columns (default 50)')
    ap.add_argument('--border', type=int, default=1, help='Tile border width (default 1)')
    ap.add_argument('--use-fixed-box', action='store_true', help='Use fixed box (P90 long side * 1.05)')
    args = ap.parse_args()

    out_root = PROJECT_ROOT / args.out
    montage_dir = out_root / 'montage'
    books_dir = out_root / 'books'

    if args.clean and out_root.exists():
        shutil.rmtree(out_root)

    montage_dir.mkdir(parents=True, exist_ok=True)
    books_dir.mkdir(parents=True, exist_ok=True)

    books = args.books if args.books else list_review_books()
    if not books:
        print('未找到任何分书文件。')
        return

    from datetime import datetime, timezone
    manifest = {
        'version': 1,
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'source': {
            'review_books_dir': str(REVIEW_BOOKS_DIR.relative_to(PROJECT_ROOT)),
            'segmented_dir': str(SEGMENTED_DIR.relative_to(PROJECT_ROOT))
        },
        'books': {}
    }

    for book in books:
        entries, meta = collect_book_entries(book, args.use_fixed_box)
        book_dir = books_dir / book
        img_dir = book_dir / 'images'
        img_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        long_sides: List[int] = []
        for entry in entries:
            src = PROJECT_ROOT / entry['segmented_path']
            dst = img_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
            entry['image'] = str(Path('books') / book / 'images' / src.name)
            copied += 1
            try:
                with Image.open(dst) as im:
                    w, h = im.size
                    long_sides.append(max(w, h))
            except Exception:
                pass

        # Save entries
        book_dir.mkdir(parents=True, exist_ok=True)
        with open(book_dir / 'entries.json', 'w', encoding='utf-8') as f:
            json.dump({'book': book, 'entries': entries}, f, ensure_ascii=False, indent=2)

        # Build montage
        tiles: List[Image.Image] = []
        Lb = 0
        if args.use_fixed_box and long_sides:
            Lb = int(math.ceil(percentile(long_sides, 90) * 1.05))
        content_size = Lb if (args.use_fixed_box and Lb > 0) else (max(long_sides) if long_sides else 0)
        tile_size = max(8, content_size + max(0, args.border) * 2 + 2) if content_size else max(8, args.tile_size)
        for entry in entries:
            img_path = out_root / entry['image']
            try:
                with Image.open(img_path) as im:
                    img = im.convert('L')
            except Exception:
                img = None
            if img is None:
                continue
            if args.use_fixed_box and Lb > 0:
                img = center_crop_fixed_box(img, Lb)
            tile = make_tile(
                img,
                tile_size=tile_size,
                border=max(0, args.border),
                bg_color=(255, 255, 255),
                scale=False
            )
            tiles.append(tile)
        montage = build_montage(tiles, cols=max(1, args.cols), tile_size=tile_size, bg_color=(255, 255, 255))
        montage.save(montage_dir / f'{book}.png')

        manifest['books'][book] = {
            'count': len(entries),
            'copied': copied,
            'missing_images': meta['missing_images'],
            'skipped_unmatched': meta['skipped_unmatched'],
            'fixed_box_Lb': Lb
        }

        print(f'✓ {book}: {len(entries)} items, montage={montage_dir / f"{book}.png"}')

    with open(out_root / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f'完成：{out_root}')


if __name__ == '__main__':
    main()
