#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将 review_results.json 拆分为按书籍存储的分片文件。

输出目录：data/results/review_books/{book}.json
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEW_PATH = PROJECT_ROOT / 'data/results/review_results.json'
OUT_DIR = PROJECT_ROOT / 'data/results/review_books'


def main() -> None:
    if not REVIEW_PATH.exists():
        print('review_results.json 不存在')
        return
    data = json.load(open(REVIEW_PATH, 'r', encoding='utf-8'))
    books = data.get('books') or {}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    for book, chars in books.items():
        payload = {
            'version': data.get('version', 2),
            'book': book,
            'chars': chars
        }
        out_path = OUT_DIR / f'{book}.json'
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        count += 1
    print(f'已写入 {count} 本书到 {OUT_DIR}')


if __name__ == '__main__':
    main()
