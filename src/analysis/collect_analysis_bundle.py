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
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.image_metrics import center_crop_fixed_box, percentile
from src.analysis.montage import build_montage, make_tile
from src.review import config as review_config
from src.review.identity import get_confirmed_path
from src.review.storage.review_books import REVIEW_BOOKS_DIR, list_review_books, read_review_book

CONFIRMED_DIR = review_config.CONFIRMED_DIR


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
            seg_rel = get_confirmed_path(seg)
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
                'confirmed_path': seg_rel,
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
        'version': 2,
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'parameters': {
            'use_fixed_box': bool(args.use_fixed_box),
            'tile_size': int(args.tile_size),
            'cols': int(args.cols),
            'border': int(args.border),
            'books': list(books),
        },
        'source': {
            'review_books_dir': str(REVIEW_BOOKS_DIR.relative_to(PROJECT_ROOT)),
            'confirmed_dir': str(CONFIRMED_DIR.relative_to(PROJECT_ROOT)),
            'review_books_mtime': max(
                (p.stat().st_mtime for p in REVIEW_BOOKS_DIR.glob('*.json')),
                default=0,
            ),
        },
        'summary': {
            'total_books': 0,
            'total_entries': 0,
            'total_copied': 0,
            'total_missing_images': 0,
            'total_missing_lookup': 0,
            'total_skipped_unmatched': 0,
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
        if long_sides:
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
            'missing_lookup': meta['missing_lookup'],
            'skipped_unmatched': meta['skipped_unmatched'],
            'fixed_box_Lb': Lb
        }
        manifest['summary']['total_books'] += 1
        manifest['summary']['total_entries'] += len(entries)
        manifest['summary']['total_copied'] += copied
        manifest['summary']['total_missing_images'] += meta['missing_images']
        manifest['summary']['total_missing_lookup'] += meta['missing_lookup']
        manifest['summary']['total_skipped_unmatched'] += meta['skipped_unmatched']

        print(f'✓ {book}: {len(entries)} items, montage={montage_dir / f"{book}.png"}')

    with open(out_root / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f'完成：{out_root}')


if __name__ == '__main__':
    main()
