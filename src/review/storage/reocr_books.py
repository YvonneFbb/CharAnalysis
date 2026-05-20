"""Read/write helpers for engine-specific reOCR shards."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from src.review import config as review_config
from src.review.storage.common import atomic_write_json, read_json_file
from src.review.storage.review_books import safe_book_name, utc_now_iso


REOCR_BOOKS_DIR = review_config.REOCR_BOOKS_DIR
REOCR_BOOK_VERSION = 2
DEFAULT_REOCR_ENGINE = "livetext"
VALID_REOCR_ENGINES = ("livetext", "paddle")


def _normalize_bbox(bbox: Optional[Dict]) -> Dict:
    bbox = bbox or {}
    return {
        "x": int(bbox.get("x", 0) or 0),
        "y": int(bbox.get("y", 0) or 0),
        "width": int(bbox.get("width", 0) or 0),
        "height": int(bbox.get("height", 0) or 0),
    }


def _normalize_polygon(points: Optional[List]) -> List[Dict]:
    out: List[Dict] = []
    if not isinstance(points, list):
        return out
    for point in points:
        if isinstance(point, dict):
            out.append({
                "x": int(point.get("x", 0) or 0),
                "y": int(point.get("y", 0) or 0),
            })
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            out.append({
                "x": int(point[0] or 0),
                "y": int(point[1] or 0),
            })
    return out


def _normalize_detection(det: Optional[Dict]) -> Dict:
    det = det or {}
    bbox_cell = det.get("bbox_cell")
    if bbox_cell is None:
        bbox_cell = det.get("bbox_input")
    if bbox_cell is None:
        bbox_cell = det.get("bbox_padded")
    return {
        "text": det.get("text"),
        "confidence": det.get("confidence"),
        "bbox_cell": _normalize_bbox(bbox_cell),
    }


def _normalize_primary_detection(det: Optional[Dict]):
    if not isinstance(det, dict):
        return None
    has_signal = (
        det.get("text") not in (None, "")
        or det.get("confidence") not in (None, 0, 0.0)
        or any(int((_normalize_bbox(det.get("bbox_cell") or det.get("bbox_input") or det.get("bbox_padded"))).get(k) or 0) != 0 for k in ("x", "y", "width", "height"))
    )
    return _normalize_detection(det) if has_signal else None


def _normalize_detections(detections: Optional[List]) -> List[Dict]:
    out: List[Dict] = []
    if not isinstance(detections, list):
        return out
    for det in detections:
        if isinstance(det, dict):
            out.append(_normalize_detection(det))
    return out


def normalize_engine_name(engine: Optional[str]) -> str:
    engine_name = str(engine or DEFAULT_REOCR_ENGINE).strip().lower()
    if engine_name not in VALID_REOCR_ENGINES:
        raise ValueError(f"不支持的 reOCR engine: {engine}")
    return engine_name


def reocr_engine_dir(engine: str) -> Path:
    return REOCR_BOOKS_DIR / normalize_engine_name(engine)


def reocr_book_path(book_name: str, engine: str) -> Path:
    return reocr_engine_dir(engine) / f"{safe_book_name(book_name)}.json"


def list_reocr_books(engine: str = DEFAULT_REOCR_ENGINE) -> list[str]:
    engine_dir = reocr_engine_dir(engine)
    if not engine_dir.exists():
        return []
    return sorted([p.stem for p in engine_dir.glob("*.json")])


def make_empty_char_entry() -> Dict:
    return {
        "updated_at": None,
        "items": {},
    }


def _normalize_item(item: Optional[Dict]) -> Dict:
    item = dict(item or {})
    state = str(item.get("state") or "pending")
    if state not in {"pending", "ready", "error"}:
        state = "pending"
    return {
        "state": state,
        "timestamp": item.get("timestamp"),
        "text": item.get("text"),
        "confidence": item.get("confidence"),
        "matches": item.get("matches"),
        "error": item.get("error"),
        "pad": int(item.get("pad") or 0),
        "duration_ms": int(item.get("duration_ms") or 0),
        "ocr_mode": item.get("ocr_mode"),
        "cell_size": {
            "width": int((((item.get("cell_size") or item.get("input_image") or {})).get("width")) or 0),
            "height": int((((item.get("cell_size") or item.get("input_image") or {})).get("height")) or 0),
        },
        "detection_count": int(item.get("detection_count") or len(item.get("detections") or [])),
        "primary_detection": _normalize_primary_detection(item.get("primary_detection")),
    }


def normalize_reocr_book_data(book_data: Optional[Dict]) -> Dict:
    out: Dict[str, Dict] = {}
    if not isinstance(book_data, dict):
        return out
    for char, char_entry in book_data.items():
        if not isinstance(char_entry, dict):
            continue
        raw_items = char_entry.get("items") or {}
        if not isinstance(raw_items, dict):
            raw_items = {}
        items = {}
        for instance_id, item in raw_items.items():
            if not isinstance(item, dict):
                continue
            items[str(instance_id)] = _normalize_item(item)
        out[str(char)] = {
            "updated_at": char_entry.get("updated_at"),
            "items": items,
        }
    return out


def read_reocr_book(book_name: str, engine: str = DEFAULT_REOCR_ENGINE) -> Optional[Dict]:
    payload = read_json_file(reocr_book_path(book_name, engine))
    if not isinstance(payload, dict):
        return None
    chars = payload.get("chars")
    if not isinstance(chars, dict):
        return None
    return normalize_reocr_book_data(chars)


def write_reocr_book(book_name: str, book_data: Dict, engine: str = DEFAULT_REOCR_ENGINE) -> None:
    engine_name = normalize_engine_name(engine)
    payload = {
        "version": REOCR_BOOK_VERSION,
        "engine": engine_name,
        "book": book_name,
        "generated_at": utc_now_iso(),
        "chars": normalize_reocr_book_data(book_data),
    }
    atomic_write_json(reocr_book_path(book_name, engine_name), payload)


def ensure_reocr_item(book_obj: Dict, char: str, instance_id: str) -> Dict:
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


def iter_reocr_items(book_data: Optional[Dict]) -> Iterator[Tuple[str, str, Dict]]:
    for char, char_entry in normalize_reocr_book_data(book_data).items():
        items = char_entry.get("items") or {}
        for instance_id, item in items.items():
            yield char, instance_id, item
