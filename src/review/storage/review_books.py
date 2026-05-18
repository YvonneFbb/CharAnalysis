"""Read/write helpers for manual review book shards."""

from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

from src.review import config as review_config
from src.review.identity import (
    get_confirmed_path,
    make_instance_id,
    normalize_to_preprocessed_path,
    set_confirmed_path,
)
from src.review.storage.common import fsync_dir


REVIEW_BOOKS_DIR = review_config.REVIEW_BOOKS_DIR
REVIEW_BOOK_BACKUP_KEEP = 5
REVIEW_BOOK_BACKUP_COOLDOWN = 60
REVIEW_BOOK_VERSION = 3
FILTER_REOCR_PAD_DEFAULT = 4


def safe_book_name(book_name: str) -> str:
    return book_name.replace("/", "_")


def review_book_path(book_name: str) -> Path:
    return REVIEW_BOOKS_DIR / f"{safe_book_name(book_name)}.json"


def review_book_lock_path(book_name: str) -> Path:
    return REVIEW_BOOKS_DIR / f"{safe_book_name(book_name)}.json.lock"


def review_book_backup_dir(book_name: str) -> Path:
    return REVIEW_BOOKS_DIR / "_backups" / safe_book_name(book_name)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def make_empty_char_entry() -> Dict:
    return {
        "updated_at": None,
        "items": {},
    }


def read_review_payload(path: Path) -> Optional[Dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _normalize_bbox(bbox: Optional[Dict]) -> Dict:
    bbox = bbox or {}
    return {
        "x": int(bbox.get("x", 0) or 0),
        "y": int(bbox.get("y", 0) or 0),
        "width": int(bbox.get("width", 0) or 0),
        "height": int(bbox.get("height", 0) or 0),
    }


def _normalize_source(source: Optional[Dict], instance_id: str) -> Dict:
    source = source or {}
    bbox = _normalize_bbox(source.get("bbox"))
    return {
        "instance_id": instance_id,
        "index": source.get("index"),
        "bbox": bbox,
        "source_image": source.get("source_image"),
        "confidence": source.get("confidence"),
        "volume": source.get("volume"),
        "page": source.get("page"),
        "char_index": source.get("char_index"),
        "width": int(source.get("width") or bbox.get("width") or 0),
        "height": int(source.get("height") or bbox.get("height") or 0),
    }


def _normalize_filter(filter_state: Optional[Dict], review_state: Optional[Dict]) -> Dict:
    filter_state = dict(filter_state or {})
    review_state = dict(review_state or {})

    status = filter_state.get("status")
    if status not in {"pending", "accepted", "rejected"}:
        review_status = review_state.get("status")
        status = "accepted" if review_status in {"confirmed", "dropped"} else "pending"

    return {
        "status": status,
        "timestamp": filter_state.get("timestamp"),
        "segmented_preview_path": filter_state.get("segmented_preview_path"),
        "segmented_width": int(filter_state.get("segmented_width") or 0),
        "segmented_height": int(filter_state.get("segmented_height") or 0),
        "reocr_text": filter_state.get("reocr_text"),
        "reocr_confidence": filter_state.get("reocr_confidence"),
        "reocr_matches": filter_state.get("reocr_matches"),
        "reocr_pad": int(filter_state.get("reocr_pad") or FILTER_REOCR_PAD_DEFAULT),
        "reocr_state": filter_state.get("reocr_state"),
        "reocr_error": filter_state.get("reocr_error"),
        "reocr_context_checked": bool(filter_state.get("reocr_context_checked") or False),
    }


def _normalize_review(review_state: Optional[Dict]) -> Dict:
    review_state = dict(review_state or {})
    status = review_state.get("status")
    if status not in {"pending", "confirmed", "dropped"}:
        if review_state.get("decision") == "drop":
            status = "dropped"
        else:
            status = "pending"

    normalized = {
        "status": status,
        "timestamp": review_state.get("timestamp"),
        "method": review_state.get("method"),
        "decision": review_state.get("decision") or ("drop" if status == "dropped" else "need"),
        "confirmed_path": None,
        "segmented_path": None,
    }
    return set_confirmed_path(normalized, get_confirmed_path(review_state))


def _normalize_item(instance_id: str, item: Optional[Dict]) -> Dict:
    item = item or {}
    review_state = _normalize_review(item.get("review"))
    filter_state = _normalize_filter(item.get("filter"), review_state)
    return {
        "source": _normalize_source(item.get("source"), instance_id),
        "filter": filter_state,
        "review": review_state,
    }


def _legacy_source_from_lookup(instance_id: str, lookup_entry: Optional[Dict]) -> Dict:
    return _normalize_source(lookup_entry, instance_id)


def _legacy_filter_from_segment(segment_entry: Optional[Dict], updated_at: Optional[str]) -> Dict:
    segment_entry = dict(segment_entry or {})
    return {
        "status": "accepted",
        "timestamp": segment_entry.get("timestamp") or updated_at,
        "segmented_preview_path": get_confirmed_path(segment_entry),
        "segmented_width": int(segment_entry.get("segmented_width") or 0),
        "segmented_height": int(segment_entry.get("segmented_height") or 0),
        "reocr_text": None,
        "reocr_confidence": None,
        "reocr_matches": None,
        "reocr_pad": FILTER_REOCR_PAD_DEFAULT,
        "reocr_state": None,
        "reocr_error": None,
        "reocr_context_checked": False,
    }


def source_from_matched_instance(inst: Optional[Dict], index: Optional[int] = None) -> Dict:
    inst = inst or {}
    bbox = _normalize_bbox(inst.get("bbox"))
    return {
        "instance_id": make_instance_id(inst),
        "index": index,
        "bbox": bbox,
        "source_image": normalize_to_preprocessed_path(inst.get("source_image", "")),
        "confidence": inst.get("confidence", 0.0),
        "volume": inst.get("volume"),
        "page": inst.get("page"),
        "char_index": inst.get("char_index"),
        "width": int(inst.get("width") or bbox.get("width") or 0),
        "height": int(inst.get("height") or bbox.get("height") or 0),
    }


def ensure_char_item(
    book_obj: Dict,
    char: str,
    instance_id: str,
    source: Optional[Dict] = None,
) -> Dict:
    char_obj = book_obj.setdefault(char, make_empty_char_entry())
    if not isinstance(char_obj, dict):
        char_obj = make_empty_char_entry()
        book_obj[char] = char_obj
    items = char_obj.setdefault("items", {})
    item = items.get(instance_id)
    if not isinstance(item, dict):
        item = {
            "source": _normalize_source(source or {"instance_id": instance_id}, instance_id),
            "filter": {
                "status": "pending",
                "timestamp": None,
                "segmented_preview_path": None,
                "segmented_width": 0,
                "segmented_height": 0,
                "reocr_text": None,
                "reocr_confidence": None,
                "reocr_matches": None,
                "reocr_pad": FILTER_REOCR_PAD_DEFAULT,
                "reocr_state": None,
                "reocr_error": None,
                "reocr_context_checked": False,
            },
            "review": {
                "status": "pending",
                "timestamp": None,
                "method": None,
                "decision": "need",
                "confirmed_path": None,
                "segmented_path": None,
            },
        }
        items[instance_id] = item
    elif source:
        current_source = item.setdefault("source", {})
        normalized_source = _normalize_source(source, instance_id)
        for key, value in normalized_source.items():
            if current_source.get(key) in (None, "", 0):
                current_source[key] = value

    char_obj["items"] = items
    char_obj["updated_at"] = utc_now_iso()
    book_obj[char] = char_obj
    return item


def _legacy_review_from_segment(segment_entry: Optional[Dict]) -> Dict:
    segment_entry = dict(segment_entry or {})
    decision = segment_entry.get("decision")
    status = segment_entry.get("status")
    if status == "confirmed":
        review_status = "confirmed"
    elif decision == "drop" or status == "dropped":
        review_status = "dropped"
    else:
        review_status = "pending"

    normalized = {
        "status": review_status,
        "timestamp": segment_entry.get("timestamp"),
        "method": segment_entry.get("method"),
        "decision": decision or ("drop" if review_status == "dropped" else "need"),
        "confirmed_path": None,
        "segmented_path": None,
    }
    return set_confirmed_path(normalized, get_confirmed_path(segment_entry))


def normalize_char_entry(char_entry: Optional[Dict]) -> Dict:
    char_entry = char_entry or {}
    if "items" in char_entry and isinstance(char_entry["items"], dict):
        items = {
            instance_id: _normalize_item(instance_id, item)
            for instance_id, item in char_entry["items"].items()
            if isinstance(item, dict)
        }
        return {
            "updated_at": char_entry.get("updated_at") or char_entry.get("timestamp"),
            "items": items,
        }

    lookup = char_entry.get("lookup") or {}
    segments = char_entry.get("segments") or {}
    updated_at = char_entry.get("timestamp")
    items: Dict[str, Dict] = {}

    for instance_id, lookup_entry in lookup.items():
        seg = segments.get(instance_id) or {}
        items[instance_id] = _normalize_item(instance_id, {
            "source": _legacy_source_from_lookup(instance_id, lookup_entry),
            "filter": _legacy_filter_from_segment(seg, updated_at),
            "review": _legacy_review_from_segment(seg),
        })

    return {
        "updated_at": updated_at,
        "items": items,
    }


def normalize_book_data(book_data: Optional[Dict]) -> Dict:
    if not isinstance(book_data, dict):
        return {}
    out = {}
    for char, char_entry in book_data.items():
        if not isinstance(char_entry, dict):
            continue
        out[char] = normalize_char_entry(char_entry)
    return out


def iter_char_items(char_entry: Optional[Dict]) -> Dict[str, Dict]:
    char_entry = normalize_char_entry(char_entry)
    items = char_entry.get("items")
    return items if isinstance(items, dict) else {}


def iter_book_items(book_data: Optional[Dict]) -> Iterator[Tuple[str, str, Dict]]:
    for char, char_entry in normalize_book_data(book_data).items():
        for instance_id, item in iter_char_items(char_entry).items():
            yield char, instance_id, item


def iter_accepted_items(book_data: Optional[Dict]) -> Iterator[Tuple[str, str, Dict]]:
    for char, instance_id, item in iter_book_items(book_data):
        if (item.get("filter") or {}).get("status") == "accepted":
            yield char, instance_id, item


def iter_confirmed_items(book_data: Optional[Dict]) -> Iterator[Tuple[str, str, Dict]]:
    for char, instance_id, item in iter_accepted_items(book_data):
        review = item.get("review") or {}
        if review.get("status") == "confirmed" and review.get("decision") != "drop":
            yield char, instance_id, item


def extract_review_book_chars(payload: Dict, book_name: str) -> Optional[Dict]:
    if not isinstance(payload, dict):
        return None
    if "chars" in payload and isinstance(payload["chars"], dict):
        return normalize_book_data(payload["chars"])
    if "books" in payload and isinstance(payload["books"], dict):
        return normalize_book_data(payload["books"].get(book_name))
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
        "version": REVIEW_BOOK_VERSION,
        "book": book_name,
        "chars": normalize_book_data(book_data),
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
    out = {"version": REVIEW_BOOK_VERSION, "books": {}}
    for book_name in list_review_books():
        chars = read_review_book(book_name)
        if isinstance(chars, dict):
            out["books"][book_name] = chars
    return out
