#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
修复 review_books 中 OCR/SEG 不一致的问题（不依赖 app.py）。

- 以 instances 重建 lookup（来自 matched_by_book）
- 删除不在 lookup 中的 segments
- 清理 _lookup_dirty 标记
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Tuple, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEW_BOOKS_DIR = PROJECT_ROOT / 'data/results/review_books'
MATCHED_JSON_PATH = PROJECT_ROOT / 'data/results/matched_by_book.json'
MATCHED_CACHE_DIR = PROJECT_ROOT / 'data/results/_cache'
MATCHED_SHARDS_DIR = MATCHED_CACHE_DIR / 'matched_by_book_shards'


def _review_book_path(book_name: str) -> Path:
    safe_name = book_name.replace('/', '_')
    return REVIEW_BOOKS_DIR / f'{safe_name}.json'


def _backup_book(book_name: str):
    src = _review_book_path(book_name)
    if not src.exists():
        return
    backup_dir = REVIEW_BOOKS_DIR / '_backups' / book_name.replace('/', '_')
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = f'{int(os.path.getmtime(src)*1000)}'
    dst = backup_dir / f'pre_repair_{ts}.json'
    try:
        shutil.copy2(src, dst)
    except Exception:
        pass


def read_review_book(book_name: str) -> Optional[Dict]:
    path = _review_book_path(book_name)
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


def write_review_book(book_name: str, book_data: Dict):
    REVIEW_BOOKS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        'version': 2,
        'book': book_name,
        'chars': book_data
    }
    tmp = _review_book_path(book_name).with_suffix(f'.json.tmp.{os.getpid()}')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, _review_book_path(book_name))


def _load_matched_book(book_name: str) -> Optional[Dict]:
    shard_path = MATCHED_SHARDS_DIR / f"{book_name}.json"
    if shard_path.exists():
        try:
            with open(shard_path, 'r', encoding='utf-8') as f:
                shard = json.load(f)
            data = shard.get('data')
            if data:
                return data
        except Exception:
            pass

    if not MATCHED_JSON_PATH.exists():
        return None
    try:
        with open(MATCHED_JSON_PATH, 'r', encoding='utf-8') as f:
            full = json.load(f)
        return (full.get('books') or {}).get(book_name)
    except Exception:
        return None


def normalize_to_preprocessed_path(raw_or_mixed_path: str) -> str:
    if '/preprocessed/' in raw_or_mixed_path and '_preprocessed.png' in raw_or_mixed_path:
        return raw_or_mixed_path
    match = re.search(r'data/raw/([^/]+)/(册\\d+_pages)/(page_\\d+)\\.png', raw_or_mixed_path)
    if match:
        book, volume_dir, page_name = match.groups()
        return f'data/results/preprocessed/{book}/{volume_dir}/{page_name}_preprocessed.png'
    return raw_or_mixed_path


def _make_instance_id(inst: Dict) -> str:
    try:
        vol = int(inst.get('volume', 0))
    except Exception:
        vol = 0
    page = inst.get('page', '')
    page_suffix = page.split('_')[-1] if page else ''
    char_index = inst.get('char_index', 0)
    return f"册{vol:02d}_page{page_suffix}_idx{char_index}"


def _sync_lookup_for_char(book_name: str, char: str, char_obj: Dict, matched_book: Dict) -> set:
    matched_list = (matched_book.get('chars') or {}).get(char)
    old_lookup = char_obj.get('lookup') or {}
    if not matched_list:
        char_obj['lookup'] = {}
        return set(old_lookup.keys())

    instances = char_obj.get('instances', {}) or {}
    new_lookup = {}
    selected_ids = set()

    for idx_str, selected in instances.items():
        if not selected:
            continue
        try:
            idx = int(idx_str)
        except Exception:
            continue
        if idx < 0 or idx >= len(matched_list):
            continue
        inst = matched_list[idx]
        inst_id = _make_instance_id(inst)
        selected_ids.add(inst_id)
        src = normalize_to_preprocessed_path(inst.get('source_image', ''))
        new_lookup[inst_id] = {
            'bbox': inst.get('bbox', {}),
            'source_image': src,
            'confidence': inst.get('confidence', 0.0),
            'volume': inst.get('volume'),
            'page': inst.get('page'),
            'char_index': inst.get('char_index'),
            'index': idx
        }

    removed_ids = set(old_lookup.keys()) - selected_ids
    char_obj['lookup'] = new_lookup
    return removed_ids


def repair_book(book_name: str, dry_run: bool = False) -> Tuple[int, int, int]:
    book_obj = read_review_book(book_name) or {}
    if not isinstance(book_obj, dict):
        return (0, 0, 0)

    matched_book = _load_matched_book(book_name)
    if not matched_book:
        return (0, 0, 0)

    changed = False
    chars_fixed = 0
    removed_segments = 0

    for char, char_obj in book_obj.items():
        if not isinstance(char_obj, dict):
            continue

        removed_ids = _sync_lookup_for_char(book_name, char, char_obj, matched_book)

        lookup_ids = set((char_obj.get('lookup') or {}).keys())
        seg_map = char_obj.get('segments') or {}
        if seg_map:
            stale = [k for k in list(seg_map.keys()) if k not in lookup_ids]
            for inst_id in stale:
                seg_map.pop(inst_id, None)
            if stale:
                removed_segments += len(stale)
                changed = True
                chars_fixed += 1
            if seg_map:
                char_obj['segments'] = seg_map
            elif 'segments' in char_obj:
                char_obj.pop('segments', None)
                changed = True

        if '_lookup_dirty' in char_obj:
            char_obj.pop('_lookup_dirty', None)
            changed = True

        if removed_ids:
            removed_segments += len(removed_ids)

        book_obj[char] = char_obj

    if changed and not dry_run:
        _backup_book(book_name)
        write_review_book(book_name, book_obj)

    return (chars_fixed, removed_segments, 1 if changed else 0)


def list_books() -> list:
    if not REVIEW_BOOKS_DIR.exists():
        return []
    return sorted([p.stem for p in REVIEW_BOOKS_DIR.glob('*.json')])


def main():
    ap = argparse.ArgumentParser(description='Repair review_books lookup/segments mismatch.')
    ap.add_argument('--books', nargs='+', help='指定书名（默认处理全部）')
    ap.add_argument('--dry-run', action='store_true', help='只扫描不写入')
    args = ap.parse_args()

    books = args.books if args.books else list_books()
    if not books:
        print('未找到任何分书文件。')
        return

    total_fixed = 0
    total_removed = 0
    total_changed = 0
    for book in books:
        chars_fixed, removed_segments, changed = repair_book(book, dry_run=args.dry_run)
        if chars_fixed or removed_segments:
            print(f'✓ {book}: 修复字符 {chars_fixed}，移除 segments {removed_segments}')
        total_fixed += chars_fixed
        total_removed += removed_segments
        total_changed += changed

    if args.dry_run:
        print(f'[dry-run] 共扫描 {len(books)} 本，需修复字符 {total_fixed}，需移除 segments {total_removed}')
    else:
        print(f'完成：处理 {len(books)} 本，修复字符 {total_fixed}，移除 segments {total_removed}')


if __name__ == '__main__':
    main()
