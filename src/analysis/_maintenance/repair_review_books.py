#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将 review_books 分书数据规范化到当前 items/source/filter/review 结构。

- 兼容读取历史的 instances/lookup/segments 结构
- 写回为当前 version 3 的 chars[*].items 结构
- 尽量补全缺失的 source 元信息
- 不删除 confirmed 图片文件
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.review import config as review_config
from src.review.identity import make_instance_id, normalize_to_preprocessed_path
from src.review.storage.review_books import (
    REVIEW_BOOK_VERSION,
    normalize_book_data,
    list_review_books,
    read_review_payload,
    review_book_backup_dir,
    review_book_path,
    write_review_book,
)

MATCHED_JSON_PATH = review_config.MATCHED_JSON_PATH
MATCHED_SHARDS_DIR = review_config.MATCHED_SHARDS_DIR


def _backup_book(book_name: str):
    src = review_book_path(book_name)
    if not src.exists():
        return
    backup_dir = review_book_backup_dir(book_name)
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = f'{int(os.path.getmtime(src)*1000)}'
    dst = backup_dir / f'pre_repair_{ts}.json'
    try:
        shutil.copy2(src, dst)
    except Exception:
        pass


def _load_matched_book(book_name: str) -> Optional[Dict]:
    shard_path = MATCHED_SHARDS_DIR / f"{book_name}.json"
    if shard_path.exists():
        try:
            payload = read_review_payload(shard_path)
            shard = payload if isinstance(payload, dict) else {}
            data = shard.get('data')
            if data:
                return data
        except Exception:
            pass

    if not MATCHED_JSON_PATH.exists():
        return None
    try:
        full = read_review_payload(MATCHED_JSON_PATH) or {}
        return (full.get('books') or {}).get(book_name)
    except Exception:
        return None


def _extract_raw_chars(payload: Optional[Dict], book_name: str) -> Dict:
    if not isinstance(payload, dict):
        return {}
    if isinstance(payload.get('chars'), dict):
        return payload.get('chars') or {}
    books = payload.get('books')
    if isinstance(books, dict) and isinstance(books.get(book_name), dict):
        return books.get(book_name) or {}
    return {}


def _source_from_matched_instance(inst: Dict, index: Optional[int] = None) -> Dict:
    bbox = inst.get('bbox') or {}
    return {
        'instance_id': make_instance_id(inst),
        'index': index,
        'bbox': bbox,
        'source_image': normalize_to_preprocessed_path(inst.get('source_image', '')),
        'confidence': inst.get('confidence', 0.0),
        'volume': inst.get('volume'),
        'page': inst.get('page'),
        'char_index': inst.get('char_index'),
        'width': int(bbox.get('width') or 0),
        'height': int(bbox.get('height') or 0),
    }


def _matched_sources_for_book(book_name: str) -> Dict[str, Dict[str, Dict]]:
    matched_book = _load_matched_book(book_name)
    if not matched_book:
        return {}
    out: Dict[str, Dict[str, Dict]] = {}
    for char, instances in (matched_book.get('chars') or {}).items():
        if not isinstance(instances, list):
            continue
        char_map: Dict[str, Dict] = {}
        for idx, inst in enumerate(instances):
            if not isinstance(inst, dict):
                continue
            src = _source_from_matched_instance(inst, index=idx)
            char_map[src['instance_id']] = src
        if char_map:
            out[char] = char_map
    return out


def _merge_source(item: Dict, fallback: Optional[Dict]) -> bool:
    if not isinstance(item, dict) or not isinstance(fallback, dict):
        return False
    source = item.setdefault('source', {})
    changed = False
    normalized_image = normalize_to_preprocessed_path(source.get('source_image', ''))
    if source.get('source_image') != normalized_image:
        source['source_image'] = normalized_image
        changed = True
    for key, value in fallback.items():
        current = source.get(key)
        if current in (None, ''):
            source[key] = value
            changed = True
    bbox = source.get('bbox') or {}
    if not source.get('width') and bbox.get('width') not in (None, ''):
        source['width'] = int(bbox.get('width') or 0)
        changed = True
    if not source.get('height') and bbox.get('height') not in (None, ''):
        source['height'] = int(bbox.get('height') or 0)
        changed = True
    return changed


def repair_book(book_name: str, dry_run: bool = False) -> Tuple[int, int, int, int]:
    book_path = review_book_path(book_name)
    payload = read_review_payload(book_path)
    raw_chars = _extract_raw_chars(payload, book_name)
    if not raw_chars:
        return (0, 0, 0, 0)

    normalized = normalize_book_data(raw_chars)
    matched_sources = _matched_sources_for_book(book_name)

    changed = (payload or {}).get('version') != REVIEW_BOOK_VERSION
    legacy_chars = 0
    enriched_items = 0
    stale_items = 0

    for char, raw_char in raw_chars.items():
        if not isinstance(raw_char, dict):
            continue
        if any(key in raw_char for key in ('instances', 'lookup', 'segments', 'timestamp', '_lookup_dirty')):
            legacy_chars += 1
            changed = True

        char_entry = normalized.get(char)
        if not isinstance(char_entry, dict):
            continue
        items = char_entry.get('items') or {}
        source_map = matched_sources.get(char) or {}
        for instance_id, item in items.items():
            fallback = source_map.get(instance_id)
            if fallback is None:
                stale_items += 1
                continue
            if _merge_source(item, fallback):
                enriched_items += 1
                changed = True
        char_entry['items'] = items
        normalized[char] = char_entry

    if changed and not dry_run:
        _backup_book(book_name)
        write_review_book(book_name, normalized, skip_backup=True)

    return (legacy_chars, enriched_items, stale_items, 1 if changed else 0)


def main():
    ap = argparse.ArgumentParser(description='Normalize review_books shards to version 3 items schema.')
    ap.add_argument('--books', nargs='+', help='指定书名（默认处理全部）')
    ap.add_argument('--dry-run', action='store_true', help='只扫描不写入')
    args = ap.parse_args()

    books = args.books if args.books else list_review_books()
    if not books:
        print('未找到任何分书文件。')
        return

    total_legacy_chars = 0
    total_enriched = 0
    total_stale = 0
    total_changed = 0
    for book in books:
        legacy_chars, enriched_items, stale_items, changed = repair_book(book, dry_run=args.dry_run)
        if legacy_chars or enriched_items or stale_items:
            print(
                f'✓ {book}: 迁移 legacy 字符 {legacy_chars}，补全 source {enriched_items}，'
                f'未匹配实例 {stale_items}'
            )
        total_legacy_chars += legacy_chars
        total_enriched += enriched_items
        total_stale += stale_items
        total_changed += changed

    if args.dry_run:
        print(
            f'[dry-run] 共扫描 {len(books)} 本，需迁移 legacy 字符 {total_legacy_chars}，'
            f'可补全 source {total_enriched}，未匹配实例 {total_stale}，需改写 {total_changed} 本'
        )
    else:
        print(
            f'完成：处理 {len(books)} 本，迁移 legacy 字符 {total_legacy_chars}，'
            f'补全 source {total_enriched}，未匹配实例 {total_stale}，改写 {total_changed} 本'
        )


if __name__ == '__main__':
    main()
