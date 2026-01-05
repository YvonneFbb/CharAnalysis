#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
清理 data/results/manual/segmented 下不再需要的切割图片。

规则（默认）：
  - 仅保留 review_results.json 中 segments 里 status == "confirmed" 且有 segmented_path 的图片
  - 其他图片移动到 data/results/manual/segmented/_orphaned 下（不直接删除）
  - manual/ 目录默认跳过

用法：
  python src/analysis/_maintenance/cleanup_segmented_images.py
  python src/analysis/_maintenance/cleanup_segmented_images.py --delete   # 直接删除，不移动
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Set

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REVIEW_PATH = PROJECT_ROOT / 'data/results/manual/review_results.json'
SEG_DIR = PROJECT_ROOT / 'data/results/manual/segmented'
ORPHAN_DIR = SEG_DIR / '_orphaned'


def load_keep_paths() -> Set[Path]:
    keep: Set[Path] = set()
    if not REVIEW_PATH.exists():
        return keep
    data = json.load(open(REVIEW_PATH, 'r', encoding='utf-8'))
    for book, chars in (data.get('books') or {}).items():
        if not isinstance(chars, dict):
            continue
        for char, char_obj in chars.items():
            if not isinstance(char_obj, dict):
                continue
            seg_map = char_obj.get('segments', {}) or {}
            for inst_id, entry in seg_map.items():
                if not isinstance(entry, dict):
                    continue
                if entry.get('status') != 'confirmed':
                    continue
                seg_rel = entry.get('segmented_path')
                if not seg_rel:
                    continue
                keep.add((PROJECT_ROOT / seg_rel).resolve())
    return keep


def main() -> None:
    ap = argparse.ArgumentParser(description='Cleanup unused segmented images.')
    ap.add_argument('--delete', action='store_true', help='Delete unused files instead of moving to _orphaned')
    args = ap.parse_args()

    keep = load_keep_paths()
    if not SEG_DIR.exists():
        print('segmented 目录不存在')
        return

    moved = 0
    deleted = 0
    skipped = 0

    if not args.delete:
        ORPHAN_DIR.mkdir(parents=True, exist_ok=True)

    for path in SEG_DIR.rglob('*.png'):
        rel = path.relative_to(SEG_DIR)
        if rel.parts and rel.parts[0] in ('manual', '_orphaned'):
            skipped += 1
            continue
        if path.resolve() in keep:
            continue
        if args.delete:
            path.unlink(missing_ok=True)
            deleted += 1
        else:
            target = ORPHAN_DIR / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            path.replace(target)
            moved += 1

    print(f'保留 {len(keep)} 个已确认切割图')
    if args.delete:
        print(f'已删除 {deleted} 个未引用图片（跳过 {skipped}）')
    else:
        print(f'已移动 {moved} 个未引用图片到 {ORPHAN_DIR}（跳过 {skipped}）')


if __name__ == '__main__':
    main()
