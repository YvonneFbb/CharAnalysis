"""Read/write helpers for manual review book shards."""

from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from src.review import config as review_config
from src.review.storage.common import fsync_dir


REVIEW_BOOKS_DIR = review_config.REVIEW_BOOKS_DIR
REVIEW_BOOK_BACKUP_KEEP = 5
REVIEW_BOOK_BACKUP_COOLDOWN = 60


def safe_book_name(book_name: str) -> str:
    return book_name.replace("/", "_")


def review_book_path(book_name: str) -> Path:
    return REVIEW_BOOKS_DIR / f"{safe_book_name(book_name)}.json"


def review_book_lock_path(book_name: str) -> Path:
    return REVIEW_BOOKS_DIR / f"{safe_book_name(book_name)}.json.lock"


def review_book_backup_dir(book_name: str) -> Path:
    return REVIEW_BOOKS_DIR / "_backups" / safe_book_name(book_name)


def read_review_payload(path: Path) -> Optional[Dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def extract_review_book_chars(payload: Dict, book_name: str) -> Optional[Dict]:
    if not isinstance(payload, dict):
        return None
    if "chars" in payload and isinstance(payload["chars"], dict):
        return payload["chars"]
    if "books" in payload and isinstance(payload["books"], dict):
        return payload["books"].get(book_name)
    return None


def read_latest_review_backup(book_name: str) -> Optional[Dict]:
    backup_dir = review_book_backup_dir(book_name)
    if not backup_dir.exists():
        return None
    candidates = sorted(backup_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        payload = read_review_payload(path)
        if payload is not None:
            print(f"⚠️  {book_name}: 主文件损坏，回退到备份 {path.name}")
            return payload
    return None


def rotate_review_backups(backup_dir: Path, keep: int = REVIEW_BOOK_BACKUP_KEEP) -> None:
    try:
        backups = sorted(backup_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in backups[keep:]:
            old.unlink(missing_ok=True)
    except Exception:
        pass


def maybe_backup_review_book(book_name: str, book_path: Path) -> None:
    if not book_path.exists():
        return
    backup_dir = review_book_backup_dir(book_name)
    backup_dir.mkdir(parents=True, exist_ok=True)
    backups = sorted(backup_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if backups:
        latest_mtime = backups[-1].stat().st_mtime
        if time.time() - latest_mtime < REVIEW_BOOK_BACKUP_COOLDOWN:
            return
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup_path = backup_dir / f"{ts}.json"
    try:
        shutil.copy2(book_path, backup_path)
    except Exception:
        return
    rotate_review_backups(backup_dir)


def read_review_book(book_name: str) -> Optional[Dict]:
    """Read one review book shard and return its chars mapping."""
    path = review_book_path(book_name)
    if not path.exists():
        return None
    payload = read_review_payload(path)
    if payload is None:
        payload = read_latest_review_backup(book_name)
    if payload is None:
        return None
    return extract_review_book_chars(payload, book_name)


def write_review_book(book_name: str, book_data: Dict, skip_backup: bool = False) -> None:
    """Atomically write one review book shard."""
    REVIEW_BOOKS_DIR.mkdir(parents=True, exist_ok=True)
    book_path = review_book_path(book_name)
    if not skip_backup:
        maybe_backup_review_book(book_name, book_path)
    payload = {
        "version": 2,
        "book": book_name,
        "chars": book_data,
    }
    tmp = book_path.with_suffix(f".json.tmp.{os.getpid()}-{int(time.time() * 1000)}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, book_path)
    fsync_dir(book_path.parent)


def list_review_books() -> list[str]:
    if not REVIEW_BOOKS_DIR.exists():
        return []
    return sorted([p.stem for p in REVIEW_BOOKS_DIR.glob("*.json")])


def read_all_review_books() -> Dict:
    out = {"version": 2, "books": {}}
    for book_name in list_review_books():
        chars = read_review_book(book_name)
        if isinstance(chars, dict):
            out["books"][book_name] = chars
    return out

