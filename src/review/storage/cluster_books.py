"""Read/write helpers for per-book cluster shards used by filter reOCR selection."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

from src.review import config as review_config
from src.review.storage.common import atomic_write_json, read_json_file
from src.review.storage.review_books import safe_book_name, utc_now_iso


CLUSTER_BOOKS_DIR = review_config.CLUSTER_BOOKS_DIR
CLUSTER_BOOK_VERSION = 1


def cluster_book_path(book_name: str) -> Path:
    return CLUSTER_BOOKS_DIR / f"{safe_book_name(book_name)}.json"


def list_cluster_books() -> list[str]:
    if not CLUSTER_BOOKS_DIR.exists():
        return []
    return sorted([p.stem for p in CLUSTER_BOOKS_DIR.glob("*.json")])


def _normalize_item(item: Optional[Dict]) -> Dict:
    item = dict(item or {})
    return {
        "selected": bool(item.get("selected", False)),
        "size_group": str(item.get("size_group") or "single"),
        "size_rank": int(item.get("size_rank") or 0),
        "width": int(item.get("width") or 0),
        "height": int(item.get("height") or 0),
        "confirmed_distance": float(item.get("confirmed_distance") or 0.0),
        "reocr_matches": bool(item.get("reocr_matches", False)),
        "reocr_state_rank": int(item.get("reocr_state_rank") or 0),
    }


def _normalize_char_entry(char_entry: Optional[Dict]) -> Dict:
    char_entry = dict(char_entry or {})
    raw_items = char_entry.get("items") or {}
    items: Dict[str, Dict] = {}
    if isinstance(raw_items, dict):
        for instance_id, item in raw_items.items():
            if isinstance(item, dict):
                items[str(instance_id)] = _normalize_item(item)

    clusters = char_entry.get("clusters") or {}
    if not isinstance(clusters, dict):
        clusters = {}

    anchors = char_entry.get("anchors") or {}
    if not isinstance(anchors, dict):
        anchors = {}

    return {
        "updated_at": char_entry.get("updated_at"),
        "mode": str(char_entry.get("mode") or "single"),
        "target_group": str(char_entry.get("target_group") or "single"),
        "target_limit": int(char_entry.get("target_limit") or 0),
        "selected_count": int(char_entry.get("selected_count") or 0),
        "anchors": {
            "confirmed_widths": [int(v or 0) for v in list(anchors.get("confirmed_widths") or []) if int(v or 0) > 0],
            "confirmed_width": int(anchors.get("confirmed_width") or 0),
            "confirmed_tolerance": int(anchors.get("confirmed_tolerance") or 0),
            "confirmed_group": str(anchors.get("confirmed_group") or ""),
            "confirmed_filter_applied": bool(anchors.get("confirmed_filter_applied", False)),
        },
        "clusters": {
            name: {
                "count": int((meta or {}).get("count") or 0),
                "min_width": int((meta or {}).get("min_width") or 0),
                "max_width": int((meta or {}).get("max_width") or 0),
                "median_width": int((meta or {}).get("median_width") or 0),
            }
            for name, meta in clusters.items()
            if isinstance(meta, dict)
        },
        "items": items,
    }


def normalize_cluster_book_data(book_data: Optional[Dict]) -> Dict:
    out: Dict[str, Dict] = {}
    if not isinstance(book_data, dict):
        return out
    for char, char_entry in book_data.items():
        if isinstance(char_entry, dict):
            out[str(char)] = _normalize_char_entry(char_entry)
    return out


def read_cluster_book(book_name: str) -> Optional[Dict]:
    payload = read_json_file(cluster_book_path(book_name))
    if not isinstance(payload, dict):
        return None
    chars = payload.get("chars")
    if not isinstance(chars, dict):
        return None
    return normalize_cluster_book_data(chars)


def write_cluster_book(book_name: str, book_data: Dict) -> None:
    payload = {
        "version": CLUSTER_BOOK_VERSION,
        "book": book_name,
        "generated_at": utc_now_iso(),
        "chars": normalize_cluster_book_data(book_data),
    }
    atomic_write_json(cluster_book_path(book_name), payload)


def iter_selected_cluster_items(book_data: Optional[Dict]) -> Iterator[Tuple[str, str, Dict]]:
    for char, char_entry in normalize_cluster_book_data(book_data).items():
        items = char_entry.get("items") or {}
        for instance_id, item in items.items():
            if bool(item.get("selected")):
                yield char, instance_id, item
