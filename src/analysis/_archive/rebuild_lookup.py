#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""同步 review_results.json 中的 lookup 字段，并清理二轮残留记录。

运行后会：
1. 重新根据 matched_by_book.json 里实例索引构建 lookup；
2. 删除 segmentation_review.json 中与 OCR 结果不一致的条目；
3. 直接覆盖原文件（会生成 .bak 备份）。

用法：
    python src/analysis/_archive/rebuild_lookup.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEW_PATH = PROJECT_ROOT / 'data/results/review_results.json'
MATCHED_PATH = PROJECT_ROOT / 'data/results/matched_by_book.json'
SEG_REVIEW_PATH = PROJECT_ROOT / 'data/results/segmentation_review.json'

# 复用 normalize_to_preprocessed_path
import re

def normalize_to_preprocessed_path(raw_or_mixed_path: str) -> str:
    """复制自后端：将路径统一转换为 preprocessed 目录。"""
    if not raw_or_mixed_path:
        return raw_or_mixed_path
    if '/preprocessed/' in raw_or_mixed_path and '_preprocessed.png' in raw_or_mixed_path:
        return raw_or_mixed_path
    match = re.search(r'data/raw/([^/]+)/(册\d+_pages)/(page_\d+)\.png', raw_or_mixed_path)
    if match:
        book, volume_dir, page_name = match.groups()
        return f'data/results/preprocessed/{book}/{volume_dir}/{page_name}_preprocessed.png'
    return raw_or_mixed_path

def load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_json(path: Path, data: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = path.with_suffix(path.suffix + '.bak')
    if path.exists():
        path.replace(backup)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f'  ✓ 写入 {path} (备份: {backup.name})')

def make_instance_id(inst: Dict) -> str:
    vol = int(inst.get('volume', 0) or 0)
    page = inst.get('page', '')
    page_suffix = page.split('_')[-1] if page else ''
    char_index = inst.get('char_index', 0)
    return f"册{vol:02d}_page{page_suffix}_idx{char_index}"


def parse_instance_id(inst_id: str):
    """反解析实例 id -> (volume, page_suffix, char_index)，不匹配返回 None。"""
    m = re.match(r'^册(\d+)_page(\d+)_idx(\d+)$', inst_id)
    if not m:
        return None
    vol, page_suffix, char_index = m.groups()
    return int(vol), page_suffix, int(char_index)

def rebuild_lookup():
    review = load_json(REVIEW_PATH)
    matched = load_json(MATCHED_PATH)
    seg_review = load_json(SEG_REVIEW_PATH) or {'version': 1, 'books': {}}

    if not review:
        raise SystemExit('review_results.json 不存在或为空，无法同步。')
    if not matched:
        raise SystemExit('matched_by_book.json 不存在，无法同步。')

    removed_total = 0
    books = review.setdefault('books', {})
    for book_name, char_map in books.items():
        matched_book = matched.get('books', {}).get(book_name)
        if not matched_book:
            continue
        for char, char_obj in char_map.items():
            if not isinstance(char_obj, dict):
                continue
            instances = char_obj.get('instances', {}) or {}
            matched_chars = matched_book.get('chars', {}).get(char)
            if not matched_chars:
                char_obj['lookup'] = {}
                continue
            lookup = {}
            selected_ids: Set[str] = set()
            # 1) 根据 OCR 选中实例构建 lookup
            for idx_str, selected in instances.items():
                if not selected:
                    continue
                try:
                    idx = int(idx_str)
                    inst = matched_chars[idx]
                except Exception:
                    continue
                inst_id = make_instance_id(inst)
                selected_ids.add(inst_id)
                lookup[inst_id] = {
                    'bbox': inst.get('bbox', {}),
                    'source_image': normalize_to_preprocessed_path(inst.get('source_image', '')),
                    'confidence': inst.get('confidence', 0.0),
                    'volume': inst.get('volume'),
                    'page': inst.get('page'),
                    'char_index': inst.get('char_index'),
                    'index': idx
                }

            # 2) 补齐 decision 但不反向新增实例
            seg_book = seg_review.get('books', {}).get(book_name, {})
            seg_char = seg_book.get(char, {})
            for inst_id, seg_entry in list(seg_char.items()):
                status = seg_entry.get('status')
                if status == 'dropped':
                    if 'decision' not in seg_entry:
                        seg_entry['decision'] = 'drop'
                elif status == 'confirmed' and inst_id in selected_ids:
                    if 'decision' not in seg_entry:
                        seg_entry['decision'] = 'need'

            char_obj['lookup'] = lookup
            char_obj['instances'] = instances

            # 3) 同步二轮记录：移除 OCR 已取消的实例
            seg_char = seg_book.get(char, {})
            for inst_id in list(seg_char.keys()):
                if inst_id not in selected_ids:
                    seg_char.pop(inst_id, None)
                    removed_total += 1
            if seg_char:
                seg_book[char] = seg_char
            elif char in seg_book:
                seg_book.pop(char)
        # 若某本书已无条目，清掉
        seg_book = seg_review.get('books', {}).get(book_name)
        if seg_book is not None and not seg_book:
            seg_review['books'].pop(book_name, None)

    write_json(REVIEW_PATH, review)
    write_json(SEG_REVIEW_PATH, seg_review)
    print(f'同步完成（移除残留切割记录 {removed_total} 条）。')

if __name__ == '__main__':
    rebuild_lookup()
