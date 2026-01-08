#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Split data/results/matched_by_book.json into per-book files.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.review.filter.match_standard_chars import save_matched_books


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MATCHED_JSON = PROJECT_ROOT / "data/results/matched_by_book.json"


def main():
    if not MATCHED_JSON.exists():
        print(f"找不到文件: {MATCHED_JSON}")
        return
    with open(MATCHED_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    save_matched_books(data)
    print("✓ 拆分完成: data/results/matched_books/")


if __name__ == "__main__":
    main()
