"""Read/write helpers for segmented atlas shards."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

from src.review import config as review_config
from src.review.storage.common import atomic_write_json, read_json_file
from src.review.storage.review_books import safe_book_name, utc_now_iso


PROJECT_ROOT = review_config.PROJECT_ROOT
SEGMENT_BOOKS_DIR = review_config.SEGMENT_BOOKS_DIR
SEGMENT_ATLAS_DIR = review_config.SEGMENT_ATLAS_DIR
SEGMENT_BOOK_VERSION = 1


def segment_book_path(book_name: str) -> Path:
    return SEGMENT_BOOKS_DIR / f"{safe_book_name(book_name)}.json"


def segment_book_atlas_dir(book_name: str) -> Path:
    return SEGMENT_ATLAS_DIR / safe_book_name(book_name)


def segment_atlas_relpath(book_name: str, atlas_filename: str) -> str:
    return str((segment_book_atlas_dir(book_name) / atlas_filename).relative_to(PROJECT_ROOT))


def list_segment_books() -> list[str]:
    if not SEGMENT_BOOKS_DIR.exists():
        return []
    return sorted([p.stem for p in SEGMENT_BOOKS_DIR.glob("*.json")])


def _normalize_bbox(bbox: Optional[Dict]) -> Dict:
    bbox = bbox or {}
    return {
        "x": int(bbox.get("x", 0) or 0),
        "y": int(bbox.get("y", 0) or 0),
        "width": int(bbox.get("width", 0) or 0),
        "height": int(bbox.get("height", 0) or 0),
    }


def _normalize_item(item: Optional[Dict]) -> Dict:
    item = dict(item or {})
    state = str(item.get("state") or "pending")
    if state not in {"pending", "ready", "error"}:
        state = "pending"
    return {
        "state": state,
        "timestamp": item.get("timestamp"),
        "error": item.get("error"),
        "atlas_relpath": item.get("atlas_relpath"),
        "atlas_bbox": _normalize_bbox(item.get("atlas_bbox")),
        "source_bbox": _normalize_bbox(item.get("source_bbox")),
        "roi_bbox": _normalize_bbox(item.get("roi_bbox")),
        "segmented_bbox": _normalize_bbox(item.get("segmented_bbox")),
        "segmented_bbox_absolute": _normalize_bbox(item.get("segmented_bbox_absolute")),
        "segmented_width": int(item.get("segmented_width") or 0),
        "segmented_height": int(item.get("segmented_height") or 0),
        "roi_width": int(item.get("roi_width") or 0),
        "roi_height": int(item.get("roi_height") or 0),
        "metrics": {
            "width_ratio": float(((item.get("metrics") or {}).get("width_ratio")) or 0.0),
            "height_ratio": float(((item.get("metrics") or {}).get("height_ratio")) or 0.0),
        },
    }


def make_empty_char_entry() -> Dict:
    return {
        "updated_at": None,
        "items": {},
    }


def normalize_segment_book_data(book_data: Optional[Dict]) -> Dict:
    out: Dict[str, Dict] = {}
    if not isinstance(book_data, dict):
        return out
    for char, char_entry in book_data.items():
        if not isinstance(char_entry, dict):
            continue
        items = {}
        raw_items = char_entry.get("items") or {}
        if not isinstance(raw_items, dict):
            raw_items = {}
        for instance_id, item in raw_items.items():
            if not isinstance(item, dict):
                continue
            items[str(instance_id)] = _normalize_item(item)
        out[str(char)] = {
            "updated_at": char_entry.get("updated_at"),
            "items": items,
        }
    return out


def read_segment_book(book_name: str) -> Optional[Dict]:
    payload = read_json_file(segment_book_path(book_name))
    if not isinstance(payload, dict):
        return None
    chars = payload.get("chars")
    if not isinstance(chars, dict):
        return None
    return normalize_segment_book_data(chars)


def write_segment_book(book_name: str, book_data: Dict) -> None:
    payload = {
        "version": SEGMENT_BOOK_VERSION,
        "book": book_name,
        "generated_at": utc_now_iso(),
        "chars": normalize_segment_book_data(book_data),
    }
    atomic_write_json(segment_book_path(book_name), payload)


def ensure_segment_item(book_obj: Dict, char: str, instance_id: str) -> Dict:
    char_obj = book_obj.setdefault(char, make_empty_char_entry())
    if not isinstance(char_obj, dict):
        char_obj = make_empty_char_entry()
        book_obj[char] = char_obj
    items = char_obj.setdefault("items", {})
    item = items.get(instance_id)
    if not isinstance(item, dict):
        item = _normalize_item(None)
        items[instance_id] = item
    char_obj["updated_at"] = utc_now_iso()
    book_obj[char] = char_obj
    return item


def iter_segment_items(book_data: Optional[Dict]) -> Iterator[Tuple[str, str, Dict]]:
    for char, char_entry in normalize_segment_book_data(book_data).items():
        items = char_entry.get("items") or {}
        for instance_id, item in items.items():
            yield char, instance_id, item

