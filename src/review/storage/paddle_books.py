"""Read/write helpers for Paddle review book shards."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

from src.review import config as review_config
from src.review.storage.common import acquire_file_lock, fsync_dir
from src.review.storage.review_books import safe_book_name


PADDLE_REVIEW_BOOKS_DIR = review_config.PADDLE_REVIEW_BOOKS_DIR


def paddle_book_path(book_name: str) -> Path:
    return PADDLE_REVIEW_BOOKS_DIR / f"{safe_book_name(book_name)}.json"


def paddle_book_lock_path(book_name: str) -> Path:
    return PADDLE_REVIEW_BOOKS_DIR / f"{safe_book_name(book_name)}.json.lock"


def list_paddle_books() -> list[str]:
    if not PADDLE_REVIEW_BOOKS_DIR.exists():
        return []
    return sorted([p.stem for p in PADDLE_REVIEW_BOOKS_DIR.glob("*.json")])


def read_paddle_book(book_name: str) -> Optional[Dict]:
    path = paddle_book_path(book_name)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def write_paddle_book(book_name: str, payload: Dict) -> bool:
    PADDLE_REVIEW_BOOKS_DIR.mkdir(parents=True, exist_ok=True)
    book_path = paddle_book_path(book_name)
    lock_path = paddle_book_lock_path(book_name)
    with open(lock_path, "a+", encoding="utf-8") as lock_fp:
        if not acquire_file_lock(lock_fp, timeout_sec=2.0):
            return False
        tmp = book_path.with_suffix(book_path.suffix + f".tmp.{os.getpid()}-{int(time.time() * 1000)}")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, book_path)
        fsync_dir(book_path.parent)
    return True
