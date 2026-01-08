#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inspect per-char width ordering and Paddle results for a single book.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MATCHED_BOOKS_DIR = PROJECT_ROOT / "data/results/matched_books"
MATCHED_JSON = PROJECT_ROOT / "data/results/matched_by_book.json"
PADDLE_BOOKS_DIR = PROJECT_ROOT / "data/results/paddle/review_books"


def load_matched_book(book: str) -> Optional[Dict]:
    book_path = MATCHED_BOOKS_DIR / f"{book}.json"
    if book_path.exists():
        try:
            payload = json.loads(book_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
                return payload["data"]
            if isinstance(payload.get("chars"), dict):
                return payload
        except Exception:
            pass
    if MATCHED_JSON.exists():
        with open(MATCHED_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        return (data.get("books") or {}).get(book)
    return None


def load_paddle_book(book: str) -> Optional[Dict]:
    path = PADDLE_BOOKS_DIR / f"{book}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def make_instance_id(inst: Dict) -> str:
    try:
        vol = int(inst.get("volume", 0))
    except Exception:
        vol = 0
    page = inst.get("page", "")
    page_suffix = page.split("_")[-1] if page else ""
    char_index = inst.get("char_index", 0)
    return f"册{vol:02d}_page{page_suffix}_idx{char_index}"


def main():
    ap = argparse.ArgumentParser(description="Inspect width-sorted instances vs Paddle results.")
    ap.add_argument("--book", required=True, help="书名，如 03_1127_周易注疏")
    ap.add_argument("--char", required=True, help="要检查的字")
    ap.add_argument("--limit", type=int, default=50, help="最多显示多少条（默认 50）")
    args = ap.parse_args()

    matched_book = load_matched_book(args.book)
    if not matched_book or not isinstance(matched_book.get("chars"), dict):
        print("找不到该书的 matched 数据。")
        return
    instances = matched_book["chars"].get(args.char) or []
    if not instances:
        print("该字在 matched 中没有实例。")
        return

    paddle_book = load_paddle_book(args.book) or {}
    paddle_chars = paddle_book.get("chars") if isinstance(paddle_book, dict) else None
    paddle_entry = paddle_chars.get(args.char) if isinstance(paddle_chars, dict) else None
    paddle_order = paddle_entry.get("order") if isinstance(paddle_entry, dict) else []
    paddle_scores = paddle_entry.get("scores") if isinstance(paddle_entry, dict) else {}
    order_index = {inst_id: idx for idx, inst_id in enumerate(paddle_order)} if isinstance(paddle_order, list) else {}

    instances_sorted = sorted(
        instances,
        key=lambda inst: -int((inst.get("bbox") or {}).get("width") or 0)
    )

    print(f"书籍: {args.book}  字: {args.char}")
    print(f"matched 实例数: {len(instances_sorted)}  paddle 记录数: {len(paddle_order) if isinstance(paddle_order, list) else 0}")
    print("-" * 80)
    print(" idx  width  inst_id                        vol  page     paddle_rank  conf  text  match")
    print("-" * 80)
    for i, inst in enumerate(instances_sorted[: max(1, args.limit)]):
        bbox = inst.get("bbox") or {}
        width = int(bbox.get("width") or 0)
        inst_id = make_instance_id(inst)
        vol = inst.get("volume")
        page = inst.get("page")
        rank = order_index.get(inst_id)
        score = paddle_scores.get(inst_id) if isinstance(paddle_scores, dict) else None
        conf = None
        text = None
        match = None
        if isinstance(score, dict):
            conf = score.get("paddle_conf")
            text = score.get("paddle_text")
            match = score.get("match")
        rank_str = "-" if rank is None else str(rank)
        conf_str = "-" if conf is None else f"{conf:.3f}"
        text_str = "-" if not text else text
        match_str = "-" if match is None else ("Y" if match else "N")
        print(f"{i:4d} {width:6d}  {inst_id:28s} {str(vol):>3}  {str(page):>7}  {rank_str:>10}  {conf_str:>4}  {text_str:>4}  {match_str:>5}")


if __name__ == "__main__":
    main()
