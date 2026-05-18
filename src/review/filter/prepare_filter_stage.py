"""Offline filter-stage preparation helpers.

This stage materializes:
- segmented preview images for filter browsing
- isolated LiveText reOCR results with context fallback when needed
- filter metadata stored in review_books shards

The web app should only read these prepared artifacts.
"""

from __future__ import annotations

import concurrent.futures
from collections import OrderedDict
import json
import multiprocessing
import os
import queue as queue_module
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from src.review import config as review_config
from src.review.identity import get_confirmed_path, make_instance_id
from src.review.ocr.livetext import ocr_image
from src.review.segment import segment_character_from_image
from src.review.storage.review_books import (
    FILTER_REOCR_PAD_DEFAULT,
    ensure_char_item,
    read_review_book,
    source_from_matched_instance,
    utc_now_iso,
    write_review_book,
)


PROJECT_ROOT = review_config.PROJECT_ROOT
MATCHED_BOOKS_DIR = review_config.MATCHED_BOOKS_DIR
MATCHED_JSON_PATH = review_config.MATCHED_JSON_PATH
FILTER_CACHE_DIR = review_config.FILTER_CACHE_DIR

FILTER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_IMAGE_CACHE_SIZE = 8
_SOURCE_IMAGE_CACHE: "OrderedDict[str, np.ndarray]" = OrderedDict()


def _extract_book_payload(payload: Optional[Dict], book_name: Optional[str] = None) -> Optional[Dict]:
    if not isinstance(payload, dict):
        return None
    if isinstance(payload.get("data"), dict):
        return payload.get("data")
    if isinstance(payload.get("chars"), dict):
        return payload
    if isinstance(payload.get("books"), dict) and book_name:
        return payload["books"].get(book_name)
    return None


def list_matched_books() -> List[str]:
    if MATCHED_BOOKS_DIR.exists():
        books = sorted(p.stem for p in MATCHED_BOOKS_DIR.glob("*.json"))
        if books:
            return books
    if MATCHED_JSON_PATH.exists():
        try:
            payload = json.loads(MATCHED_JSON_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
        books = payload.get("books")
        if isinstance(books, dict):
            return sorted(books.keys())
    return []


def read_matched_book(book_name: str) -> Optional[Dict]:
    if not book_name:
        return None
    shard_path = MATCHED_BOOKS_DIR / f"{book_name}.json"
    if shard_path.exists():
        try:
            payload = json.loads(shard_path.read_text(encoding="utf-8"))
        except Exception:
            payload = None
        data = _extract_book_payload(payload, book_name)
        if isinstance(data, dict):
            return data
    if MATCHED_JSON_PATH.exists():
        try:
            payload = json.loads(MATCHED_JSON_PATH.read_text(encoding="utf-8"))
        except Exception:
            return None
        return _extract_book_payload(payload, book_name)
    return None


def filter_preview_relpath(book_name: str, char: str, instance_id: str) -> str:
    safe_char = str(char or "_").replace("/", "_")
    rel = Path(FILTER_CACHE_DIR.relative_to(PROJECT_ROOT)) / book_name / f"{safe_char}_{instance_id}.png"
    return str(rel)


def normalize_ocr_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    return "".join(str(text).split())


def pad_segmented_image(img: np.ndarray, pad: int) -> np.ndarray:
    if img is None:
        raise ValueError("segmented image is empty")
    pad = max(0, int(pad or 0))
    if pad <= 0:
        return img
    return cv2.copyMakeBorder(
        img,
        pad,
        pad,
        pad,
        pad,
        borderType=cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )


def _read_cached_source_image(image_path: Path) -> np.ndarray:
    key = str(image_path.resolve())
    cached = _SOURCE_IMAGE_CACHE.get(key)
    if cached is not None:
        _SOURCE_IMAGE_CACHE.move_to_end(key)
        return cached

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法加载图片: {image_path}")

    _SOURCE_IMAGE_CACHE[key] = image
    _SOURCE_IMAGE_CACHE.move_to_end(key)
    while len(_SOURCE_IMAGE_CACHE) > SOURCE_IMAGE_CACHE_SIZE:
        _SOURCE_IMAGE_CACHE.popitem(last=False)
    return image


def _segment_from_source_image(
    preprocessed_abs: Path,
    source_bbox: Dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict, np.ndarray]:
    image = _read_cached_source_image(preprocessed_abs)
    return segment_character_from_image(image, source_bbox)


def _run_livetext_ocr(image_bgr: np.ndarray) -> Dict:
    fd, temp_path = tempfile.mkstemp(suffix=".png", dir=str(FILTER_CACHE_DIR))
    os.close(fd)
    fd_out, output_path = tempfile.mkstemp(suffix=".json", dir=str(FILTER_CACHE_DIR))
    os.close(fd_out)
    try:
        if not cv2.imwrite(temp_path, image_bgr):
            raise RuntimeError("无法写入 LiveText 临时图片")

        result = ocr_image(temp_path, output_path=output_path, verbose=False)
        if not result.get("success"):
            raise RuntimeError(result.get("error") or "LiveText reOCR 失败")
        return result
    except Exception as exc:
        raise RuntimeError(f"LiveText reOCR 失败: {exc}") from exc
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        try:
            os.unlink(output_path)
        except OSError:
            pass


def call_livetext_reocr(image_bgr: np.ndarray) -> Tuple[str, float]:
    result = _run_livetext_ocr(image_bgr)
    characters = result.get("characters") or []
    text = "".join(str(ch.get("text") or "") for ch in characters)
    confidence = max((float(ch.get("confidence") or 0.0) for ch in characters), default=0.0)
    return text, confidence


def _bbox_rect(bbox: Optional[Dict]) -> Tuple[int, int, int, int]:
    bbox = bbox or {}
    x = int(bbox.get("x") or 0)
    y = int(bbox.get("y") or 0)
    w = max(1, int(bbox.get("width") or 0))
    h = max(1, int(bbox.get("height") or 0))
    return x, y, w, h


def _intersection_area(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> int:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0
    return int(ix2 - ix1) * int(iy2 - iy1)


def _center_distance_sq(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    acx = ax + aw / 2.0
    acy = ay + ah / 2.0
    bcx = bx + bw / 2.0
    bcy = by + bh / 2.0
    return (acx - bcx) ** 2 + (acy - bcy) ** 2


def _contains_cjk(text: str) -> bool:
    for ch in text:
        code = ord(ch)
        if 0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF or 0xF900 <= code <= 0xFAFF:
            return True
    return False


def should_fallback_to_context(text: Optional[str], matches: bool) -> bool:
    if matches:
        return False
    normalized = normalize_ocr_text(text)
    if not normalized:
        return True
    if len(normalized) != 1:
        return True
    if not _contains_cjk(normalized):
        return True
    return False


def _context_pad_from_source_bbox(source_bbox: Dict, pad: int) -> Tuple[int, int]:
    _, _, width, height = _bbox_rect(source_bbox)
    pad_x = max(int(pad or 0), int(round(max(width * 0.3, 8.0))))
    pad_y = max(int(pad or 0), int(round(max(height * 1.2, 16.0))))
    return min(pad_x, 96), min(pad_y, 128)


def crop_context_reocr_image(
    full_image: np.ndarray,
    source_bbox: Dict,
    pad: int,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    if full_image is None or full_image.size == 0:
        raise ValueError("原图为空，无法构造 reOCR 上下文")
    img_h, img_w = full_image.shape[:2]
    x, y, w, h = _bbox_rect(source_bbox)
    pad_x, pad_y = _context_pad_from_source_bbox(source_bbox, pad)
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(img_w, x + w + pad_x)
    y1 = min(img_h, y + h + pad_y)
    crop = full_image[y0:y1, x0:x1]
    if crop.size == 0:
        raise ValueError("reOCR 上下文 crop 为空")
    return crop, (x0, y0)


def call_livetext_reocr_with_context(
    image_bgr: np.ndarray,
    crop_origin: Tuple[int, int],
    target_bbox_abs: Dict,
) -> Tuple[str, float]:
    result = _run_livetext_ocr(image_bgr)
    characters = result.get("characters") or []
    if not characters:
        raise RuntimeError("LiveText 无检测结果")

    target_rect = _bbox_rect(target_bbox_abs)
    origin_x, origin_y = crop_origin
    best = None
    for ch in characters:
        bbox = ch.get("bbox") or {}
        candidate_rect = (
            origin_x + int(bbox.get("x") or 0),
            origin_y + int(bbox.get("y") or 0),
            max(1, int(bbox.get("width") or 0)),
            max(1, int(bbox.get("height") or 0)),
        )
        overlap = _intersection_area(target_rect, candidate_rect)
        distance_sq = _center_distance_sq(target_rect, candidate_rect)
        candidate = (
            overlap,
            -distance_sq,
            float(ch.get("confidence") or 0.0),
            str(ch.get("text") or ""),
        )
        if best is None or candidate > best:
            best = candidate

    if best is None:
        raise RuntimeError("LiveText 无法定位目标字符")
    return best[3], best[2]

def _preview_abs_from_states(review_state: Dict, filter_state: Dict) -> Tuple[Optional[str], Optional[Path]]:
    preview_rel = get_confirmed_path(review_state) or filter_state.get("segmented_preview_path")
    if not preview_rel:
        return None, None
    preview_abs = PROJECT_ROOT / preview_rel
    if not preview_abs.exists():
        return preview_rel, None
    return preview_rel, preview_abs


def _normalize_reocr_pad(filter_state: Dict) -> Tuple[int, bool]:
    raw_pad = filter_state.get("reocr_pad")
    if raw_pad in (None, "", 0, 6):
        pad = FILTER_REOCR_PAD_DEFAULT
    else:
        pad = int(raw_pad)
    changed = int(filter_state.get("reocr_pad") or 0) != pad
    if changed:
        filter_state["reocr_pad"] = pad
    return pad, changed


def set_filter_error_state(filter_state: Dict, error_message: str) -> bool:
    changed = False
    if filter_state.get("reocr_text") is not None:
        filter_state["reocr_text"] = None
        changed = True
    if filter_state.get("reocr_confidence") is not None:
        filter_state["reocr_confidence"] = None
        changed = True
    if filter_state.get("reocr_matches") is not None:
        filter_state["reocr_matches"] = None
        changed = True
    if filter_state.get("reocr_state") != "error":
        filter_state["reocr_state"] = "error"
        changed = True
    if filter_state.get("reocr_error") != error_message:
        filter_state["reocr_error"] = error_message
        changed = True
    if filter_state.get("reocr_context_checked") is not False:
        filter_state["reocr_context_checked"] = False
        changed = True
    return changed


def set_filter_ready_state(
    filter_state: Dict,
    text: str,
    confidence: float,
    matches: bool,
) -> bool:
    changed = False
    if filter_state.get("reocr_text") != text:
        filter_state["reocr_text"] = text
        changed = True
    current_confidence = filter_state.get("reocr_confidence")
    if current_confidence is None or float(current_confidence) != float(confidence):
        filter_state["reocr_confidence"] = confidence
        changed = True
    if filter_state.get("reocr_matches") is not matches:
        filter_state["reocr_matches"] = matches
        changed = True
    if filter_state.get("reocr_state") != "ready":
        filter_state["reocr_state"] = "ready"
        changed = True
    if filter_state.get("reocr_error") is not None:
        filter_state["reocr_error"] = None
        changed = True
    if filter_state.get("reocr_context_checked") is not True:
        filter_state["reocr_context_checked"] = True
        changed = True
    return changed


def item_is_qualified_sample(item: Optional[Dict]) -> bool:
    item = item or {}
    filter_state = item.get("filter") or {}
    if filter_state.get("status") == "accepted":
        return True
    return filter_state.get("reocr_matches") is True


def count_qualified_samples(char_entry: Optional[Dict]) -> int:
    char_entry = char_entry or {}
    items = char_entry.get("items") or {}
    return sum(1 for item in items.values() if item_is_qualified_sample(item))


def item_has_manual_state(item: Optional[Dict]) -> bool:
    item = item or {}
    filter_state = item.get("filter") or {}
    review_state = item.get("review") or {}
    if filter_state.get("status") in {"accepted", "rejected"}:
        return True
    if review_state.get("status") in {"confirmed", "dropped"}:
        return True
    return bool(get_confirmed_path(review_state))


def keep_filter_item(instance_id: str, item: Optional[Dict], visited_ids: set[str]) -> bool:
    if instance_id in visited_ids:
        return True
    return item_has_manual_state(item)


def _target_reached(
    force: bool,
    target_samples_per_char: int,
    qualified_total: int,
    qualified_seen_this_run: int,
) -> bool:
    if target_samples_per_char <= 0:
        return False
    if force:
        return qualified_seen_this_run >= target_samples_per_char
    return qualified_total >= target_samples_per_char


def matched_instance_width(inst: Optional[Dict]) -> int:
    inst = inst or {}
    bbox = inst.get("bbox") or {}
    return int(inst.get("width") or bbox.get("width") or 0)


def ranked_instances(
    instances: List[Dict],
    limit_instances: Optional[int] = None,
) -> List[Tuple[int, Dict]]:
    enumerated_instances = [(orig_idx, inst) for orig_idx, inst in enumerate(instances) if isinstance(inst, dict)]
    enumerated_instances.sort(
        key=lambda pair: (
            -matched_instance_width(pair[1]),
            pair[0],
        )
    )
    if limit_instances is not None:
        enumerated_instances = enumerated_instances[: max(0, int(limit_instances))]
    return enumerated_instances


def prepare_filter_item(
    book_name: str,
    char: str,
    item: Dict,
    force: bool = False,
) -> Dict[str, bool]:
    source = item.setdefault("source", {})
    filter_state = item.setdefault("filter", {})
    review_state = item.setdefault("review", {})
    instance_id = source.get("instance_id")
    if not instance_id:
        raise ValueError("缺少 instance_id")

    changed = False
    preview_prepared = False
    reocr_attempted = False
    if (review_state.get("status") in {"confirmed", "dropped"} or get_confirmed_path(review_state)) and filter_state.get("status") != "accepted":
        filter_state["status"] = "accepted"
        filter_state["timestamp"] = filter_state.get("timestamp") or utc_now_iso()
        changed = True

    pad, pad_changed = _normalize_reocr_pad(filter_state)
    changed = changed or pad_changed

    preview_rel, preview_abs = _preview_abs_from_states(review_state, filter_state)
    source_bbox = source.get("bbox") or {}
    preprocessed_abs = None
    preprocessed_image = source.get("source_image")
    if preprocessed_image:
        preprocessed_abs = Path(preprocessed_image)
        if not preprocessed_abs.is_absolute():
            preprocessed_abs = PROJECT_ROOT / preprocessed_abs
        if not preprocessed_abs.exists():
            raise ValueError(f"源图片不存在: {preprocessed_abs}")

    segmented_img = None
    segmented_meta = None
    if preview_abs is None:
        if preprocessed_abs is None:
            raise ValueError(f"{book_name}/{char}/{instance_id} 缺少 source_image")
        _, segmented_img, _, segmented_meta, _ = _segment_from_source_image(preprocessed_abs, source_bbox)
        preview_rel = filter_preview_relpath(book_name, char, instance_id)
        preview_abs = PROJECT_ROOT / preview_rel
        preview_abs.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(preview_abs), segmented_img):
            raise ValueError(f"无法写入 filter 预览图: {preview_abs}")
        if filter_state.get("segmented_preview_path") != preview_rel:
            filter_state["segmented_preview_path"] = preview_rel
            changed = True
        preview_prepared = True
    if segmented_img is None and preview_abs is not None:
        missing_segmented_size = not filter_state.get("segmented_width") or not filter_state.get("segmented_height")
        if missing_segmented_size:
            segmented_img = cv2.imread(str(preview_abs), cv2.IMREAD_COLOR)

    if segmented_img is not None:
        seg_h, seg_w = segmented_img.shape[:2]
        if int(filter_state.get("segmented_width") or 0) != int(seg_w):
            filter_state["segmented_width"] = int(seg_w)
            changed = True
        if int(filter_state.get("segmented_height") or 0) != int(seg_h):
            filter_state["segmented_height"] = int(seg_h)
            changed = True

    if segmented_img is None and preview_abs is not None:
        segmented_img = cv2.imread(str(preview_abs), cv2.IMREAD_COLOR)

    if segmented_img is None and get_confirmed_path(review_state):
        confirmed_abs = PROJECT_ROOT / get_confirmed_path(review_state)
        if confirmed_abs.exists():
            segmented_img = cv2.imread(str(confirmed_abs), cv2.IMREAD_COLOR)

    reocr_state = filter_state.get("reocr_state")
    # `reocr_state` tracks whether the latest prepared preview has a usable result.
    # `reocr_context_checked` records whether the fallback decision has already been
    # resolved for that prepared preview, so `ready` items can be resumed cheaply.
    context_checked = bool(filter_state.get("reocr_context_checked") or False)
    needs_reocr = force or pad_changed or reocr_state not in {"ready", "error"}
    if not needs_reocr and reocr_state == "ready":
        needs_reocr = (
            filter_state.get("reocr_text") is None
            or filter_state.get("reocr_matches") is None
            or filter_state.get("reocr_confidence") is None
        )
    if not needs_reocr and reocr_state == "ready" and not context_checked:
        needs_reocr = should_fallback_to_context(
            filter_state.get("reocr_text"),
            filter_state.get("reocr_matches") is True,
        )

    if not needs_reocr and reocr_state == "ready":
        if preview_abs is None and get_confirmed_path(review_state):
            needs_reocr = True
        elif preview_rel and filter_state.get("segmented_preview_path") != preview_rel:
            needs_reocr = True

    if needs_reocr and segmented_img is not None:
        reocr_attempted = True
        try:
            if segmented_meta is None and preprocessed_abs is not None and source_bbox:
                _, _, _, segmented_meta, _ = _segment_from_source_image(preprocessed_abs, source_bbox)

            isolated_text = None
            isolated_confidence = 0.0
            isolated_error = None
            try:
                isolated_img = pad_segmented_image(segmented_img, pad)
                isolated_text, isolated_confidence = call_livetext_reocr(isolated_img)
            except Exception as exc:
                isolated_error = exc

            text = isolated_text
            confidence = isolated_confidence
            matches = normalize_ocr_text(text) == char if text is not None else False

            use_context_fallback = should_fallback_to_context(text, matches)
            if use_context_fallback and segmented_meta is not None and preprocessed_abs is not None:
                full_image = _read_cached_source_image(preprocessed_abs)
                target_bbox_abs = segmented_meta.get("segmented_bbox_absolute")
                if full_image is not None and isinstance(target_bbox_abs, dict):
                    context_img, crop_origin = crop_context_reocr_image(full_image, source_bbox, pad)
                    context_text, context_confidence = call_livetext_reocr_with_context(
                        context_img,
                        crop_origin,
                        target_bbox_abs,
                    )
                    text = context_text
                    confidence = context_confidence
                    matches = normalize_ocr_text(text) == char
                    isolated_error = None

            if text is None:
                if isolated_error is not None:
                    raise isolated_error
                raise RuntimeError("LiveText reOCR 失败")

            changed = set_filter_ready_state(filter_state, text, confidence, matches) or changed
        except Exception as exc:
            changed = set_filter_error_state(filter_state, str(exc)) or changed

    return {
        "changed": changed,
        "preview_prepared": preview_prepared,
        "reocr_attempted": reocr_attempted,
    }


def _iter_selected_books(all_books: List[str], selected_books: Optional[Iterable[str]]) -> List[str]:
    if not selected_books:
        return all_books
    selected = {str(name) for name in selected_books if str(name)}
    return [name for name in all_books if name in selected]


def _selected_char_names(
    chars_map: Dict,
    selected_chars: Optional[set[str]] = None,
    limit_chars: Optional[int] = None,
) -> List[str]:
    char_names = sorted(chars_map.keys())
    if selected_chars:
        char_names = [ch for ch in char_names if ch in selected_chars]
    if limit_chars is not None:
        char_names = char_names[: max(0, int(limit_chars))]
    return char_names


def _count_selected_instances(
    chars_map: Dict,
    char_names: List[str],
    limit_instances: Optional[int] = None,
) -> int:
    total = 0
    for char in char_names:
        instances = chars_map.get(char) or []
        if not isinstance(instances, list):
            continue
        count = sum(1 for inst in instances if isinstance(inst, dict))
        if limit_instances is not None:
            count = min(count, max(0, int(limit_instances)))
        total += count
    return total


def _build_prepare_filter_plan(
    selected_books: List[str],
    chars: Optional[Iterable[str]] = None,
    limit_chars: Optional[int] = None,
    limit_instances: Optional[int] = None,
) -> Dict[str, int]:
    selected_chars = {str(ch) for ch in chars or [] if str(ch)}
    total_chars = 0
    total_preview_instances = 0
    for book_name in selected_books:
        matched_book = read_matched_book(book_name)
        if not matched_book:
            continue
        chars_map = matched_book.get("chars")
        if not isinstance(chars_map, dict) or not chars_map:
            continue
        char_names = _selected_char_names(chars_map, selected_chars if selected_chars else None, limit_chars)
        total_chars += len(char_names)
        total_preview_instances += _count_selected_instances(chars_map, char_names, limit_instances)
    return {
        "total_books": len(selected_books),
        "total_chars": total_chars,
        "total_preview_instances": total_preview_instances,
    }


def _process_prepare_filter_book(
    book_idx: int,
    total_books: int,
    book_name: str,
    chars: Optional[Iterable[str]] = None,
    limit_chars: Optional[int] = None,
    limit_instances: Optional[int] = None,
    force: bool = False,
    checkpoint_items: int = 1000,
    target_samples_per_char: int = 10,
    progress_queue=None,
) -> Dict:
    matched_book = read_matched_book(book_name)
    logs: List[str] = []
    if not matched_book:
        logs.append(f"[{book_idx}/{total_books}] 跳过 {book_name}: matched_books 缺失")
        return {
            "book_name": book_name,
            "logs": logs,
            "changed": False,
            "written": False,
            "preview_processed": 0,
            "reocr_processed": 0,
            "errors": 0,
        }

    chars_map = matched_book.get("chars")
    if not isinstance(chars_map, dict) or not chars_map:
        logs.append(f"[{book_idx}/{total_books}] 跳过 {book_name}: 无 chars 数据")
        return {
            "book_name": book_name,
            "logs": logs,
            "changed": False,
            "written": False,
            "preview_processed": 0,
            "reocr_processed": 0,
            "errors": 0,
        }

    selected_chars = {str(ch) for ch in chars or [] if str(ch)}
    review_book = read_review_book(book_name) or {}
    book_changed = False
    book_written = False
    book_dirty = False
    book_errors = 0
    book_preview_processed = 0
    book_reocr_processed = 0
    processed_since_write = 0
    reported_preview = 0
    reported_reocr = 0
    progress_interval = max(100, min(max(1, int(checkpoint_items or 1000)), 500))

    def emit_progress(force_emit: bool = False) -> None:
        nonlocal reported_preview, reported_reocr
        if progress_queue is None:
            return
        preview_delta = book_preview_processed - reported_preview
        reocr_delta = book_reocr_processed - reported_reocr
        if not force_emit and preview_delta < progress_interval and reocr_delta < max(10, progress_interval // 10):
            return
        if preview_delta <= 0 and reocr_delta <= 0 and not force_emit:
            return
        progress_queue.put(
            {
                "type": "progress",
                "book_name": book_name,
                "preview_delta": max(0, int(preview_delta)),
                "reocr_delta": max(0, int(reocr_delta)),
            }
        )
        reported_preview = book_preview_processed
        reported_reocr = book_reocr_processed

    def flush_book(reason: str = "checkpoint") -> None:
        nonlocal book_written, book_dirty, processed_since_write
        if not book_dirty:
            return
        write_review_book(book_name, review_book)
        book_written = True
        emit_progress(force_emit=True)
        if reason == "checkpoint":
            logs.append(
                f"    checkpoint: preview={book_preview_processed} reocr={book_reocr_processed} errors={book_errors} "
                f"pending_write_reset={processed_since_write}"
            )
        book_dirty = False
        processed_since_write = 0

    char_names = _selected_char_names(chars_map, selected_chars if selected_chars else None, limit_chars)

    logs.append(f"[{book_idx}/{total_books}] {book_name}: {len(char_names)} chars")

    for char in char_names:
        instances = chars_map.get(char) or []
        if not isinstance(instances, list) or not instances:
            continue
        enumerated_instances = ranked_instances(instances, limit_instances=limit_instances)
        if not enumerated_instances:
            continue

        char_preview_processed = 0
        char_reocr_processed = 0
        char_errors = 0
        qualified_seen_this_run = 0
        visited_ids: set[str] = set()
        processed_ids: set[str] = set()

        def run_item_prepare(
            item: Dict,
            source: Dict,
            inst: Optional[Dict] = None,
        ) -> None:
            nonlocal book_changed, book_dirty, processed_since_write
            nonlocal book_errors, book_preview_processed, book_reocr_processed
            nonlocal char_errors, char_preview_processed, char_reocr_processed
            nonlocal qualified_seen_this_run

            instance_id = source.get("instance_id") or make_instance_id(inst or {})
            visited_ids.add(instance_id)
            processed_ids.add(instance_id)
            existing_item = ((((review_book.get(char) or {}).get("items")) or {}).get(instance_id))
            existing_source = dict((existing_item or {}).get("source") or {}) if isinstance(existing_item, dict) else None
            item_changed = False
            if existing_item is None:
                book_changed = True
                item_changed = True
            elif existing_source is not None and existing_source != dict(item.get("source") or {}):
                book_changed = True
                item_changed = True

            try:
                prepare_result = prepare_filter_item(
                    book_name,
                    char,
                    item,
                    force=force,
                )
                if prepare_result["changed"]:
                    book_changed = True
                    item_changed = True
                if prepare_result["preview_prepared"]:
                    book_preview_processed += 1
                    char_preview_processed += 1
                if prepare_result["reocr_attempted"]:
                    book_reocr_processed += 1
                    char_reocr_processed += 1
            except Exception as exc:
                filter_state = item.setdefault("filter", {})
                if set_filter_error_state(filter_state, str(exc)):
                    book_changed = True
                    item_changed = True
                book_errors += 1
                char_errors += 1

            now_qualified = item_is_qualified_sample(item)
            if force and now_qualified:
                qualified_seen_this_run += 1

            if item_changed:
                book_dirty = True
            emit_progress()
            if book_dirty:
                processed_since_write += 1
                if processed_since_write >= checkpoint_items:
                    flush_book("checkpoint")

        existing_items = dict((((review_book.get(char) or {}).get("items")) or {}))
        preserved_existing = []
        for instance_id, item in existing_items.items():
            if not isinstance(item, dict):
                continue
            if not item_has_manual_state(item):
                continue
            source = dict(item.get("source") or {})
            source["instance_id"] = instance_id
            preserved_existing.append((instance_id, source, item))

        preserved_existing.sort(
            key=lambda triple: (
                -int((triple[1] or {}).get("width") or 0),
                triple[0],
            )
        )

        for instance_id, source, item in preserved_existing:
            run_item_prepare(item, source)

        qualified_instance_ids = {
            instance_id
            for instance_id in visited_ids
            if item_is_qualified_sample(((((review_book.get(char) or {}).get("items")) or {}).get(instance_id)))
        }
        qualified_total = len(qualified_instance_ids)

        for inst_idx, inst in enumerated_instances:
            if _target_reached(force, target_samples_per_char, qualified_total, qualified_seen_this_run):
                break
            if not isinstance(inst, dict):
                continue
            source = source_from_matched_instance(inst, index=inst_idx)
            instance_id = source.get("instance_id") or make_instance_id(inst)
            visited_ids.add(instance_id)
            if instance_id in processed_ids:
                continue
            existing_item = ((((review_book.get(char) or {}).get("items")) or {}).get(instance_id))
            existing_source = dict((existing_item or {}).get("source") or {}) if isinstance(existing_item, dict) else None
            item = ensure_char_item(review_book, char, instance_id, source=source)
            if existing_item is None:
                book_changed = True
                book_dirty = True
            elif existing_source is not None and existing_source != dict(item.get("source") or {}):
                book_changed = True
                book_dirty = True
            run_item_prepare(item, source, inst=inst)
            after_qualified = item_is_qualified_sample(item)
            if after_qualified and instance_id not in qualified_instance_ids:
                qualified_instance_ids.add(instance_id)
                qualified_total += 1

        char_obj = review_book.get(char)
        char_pruned = 0
        if isinstance(char_obj, dict):
            items = dict(char_obj.get("items") or {})
            compacted_items = {}
            for instance_id, item in items.items():
                if keep_filter_item(instance_id, item, visited_ids):
                    compacted_items[instance_id] = item
                else:
                    char_pruned += 1
            if compacted_items != items:
                char_obj["items"] = compacted_items
                char_obj["updated_at"] = utc_now_iso()
                review_book[char] = char_obj
                book_changed = True
                book_dirty = True

        char_final_qualified = count_qualified_samples(review_book.get(char))
        emit_progress(force_emit=True)
        logs.append(
            f"    char={char} qualified={char_final_qualified}/{target_samples_per_char} "
            f"prepared={len(visited_ids)} preview_new={char_preview_processed} "
            f"reocr={char_reocr_processed} pruned={char_pruned} errors={char_errors}"
        )

    if book_changed:
        flush_book("final")
    emit_progress(force_emit=True)

    logs.append(
        f"    preview={book_preview_processed} reocr={book_reocr_processed} "
        f"errors={book_errors} changed={'yes' if book_changed else 'no'} written={'yes' if book_written else 'no'}"
    )
    return {
        "book_name": book_name,
        "logs": logs,
        "changed": book_changed,
        "written": book_written,
        "preview_processed": book_preview_processed,
        "reocr_processed": book_reocr_processed,
        "errors": book_errors,
    }


def _make_tqdm_writer():
    try:
        from tqdm.auto import tqdm
    except Exception:
        return None, print
    if not (sys.stdout.isatty() or sys.stderr.isatty()):
        return None, print
    return tqdm, tqdm.write


def _create_progress_bars(plan: Dict[str, int]):
    tqdm_mod, writer = _make_tqdm_writer()
    if tqdm_mod is None:
        return None, None, None, writer
    books_bar = tqdm_mod(total=max(0, int(plan.get("total_books") or 0)), desc="books", position=0, leave=True)
    preview_bar = tqdm_mod(total=None, desc="segment", unit="inst", position=1, leave=True)
    reocr_bar = tqdm_mod(total=None, desc="reOCR", unit="inst", position=2, leave=True)
    return books_bar, preview_bar, reocr_bar, writer


def _drain_progress_queue(progress_queue, preview_bar, reocr_bar) -> None:
    if progress_queue is None:
        return
    while True:
        try:
            event = progress_queue.get_nowait()
        except queue_module.Empty:
            break
        if not isinstance(event, dict):
            continue
        if event.get("type") != "progress":
            continue
        if preview_bar is not None:
            preview_bar.update(int(event.get("preview_delta") or 0))
        if reocr_bar is not None:
            reocr_bar.update(int(event.get("reocr_delta") or 0))


def run_prepare_filter(
    books: Optional[Iterable[str]] = None,
    chars: Optional[Iterable[str]] = None,
    limit_chars: Optional[int] = None,
    limit_instances: Optional[int] = None,
    workers: int = 8,
    force: bool = False,
    checkpoint_items: int = 1000,
    target_samples_per_char: int = 10,
) -> int:
    all_books = list_matched_books()
    selected_books = _iter_selected_books(all_books, books)
    if books:
        requested_books = {str(name) for name in books if str(name)}
        missing_books = sorted(requested_books - set(selected_books))
        if missing_books:
            print(f"警告：以下书籍未在 matched_books 中找到，将跳过：{', '.join(missing_books)}")
    if not selected_books:
        print("未找到可处理的 matched_books 数据")
        return 1

    selected_chars = {str(ch) for ch in chars or [] if str(ch)}
    checkpoint_items = max(1, int(checkpoint_items or 1000))
    total_books = len(selected_books)
    plan = _build_prepare_filter_plan(
        selected_books,
        chars=sorted(selected_chars) if selected_chars else None,
        limit_chars=limit_chars,
        limit_instances=limit_instances,
    )
    print("\n" + "=" * 60)
    print("准备 Filter 阶段数据")
    print("=" * 60)
    print("流程: OCR 宽度排序 -> 按需 segment -> isolated LiveText reOCR (+ context fallback) -> review_books")
    print("说明: 该阶段为离线预计算；到达每字目标样本数后即停止，不再全量 segment")
    print(f"书籍数量: {total_books}")
    print(f"目标字符数: {plan['total_chars']}")
    print(f"OCR 候选上界: {plan['total_preview_instances']}")
    print(f"每字目标样本数: {target_samples_per_char}")
    print(f"reOCR pad: {FILTER_REOCR_PAD_DEFAULT}px")
    print(f"checkpoint: 每 {checkpoint_items} 个已处理实例落盘一次")
    if force:
        print("模式: force 重新计算")
    print("-" * 60)

    changed_books = 0
    preview_processed_items = 0
    reocr_processed_items = 0
    error_items = 0

    worker_count = max(1, min(int(workers or 1), total_books))
    print(f"workers: {worker_count}")
    books_bar, preview_bar, reocr_bar, writer = _create_progress_bars(plan)
    progress_queue = None
    worker_kwargs = {
        "chars": sorted(selected_chars) if selected_chars else None,
        "limit_chars": limit_chars,
        "limit_instances": limit_instances,
        "force": force,
        "checkpoint_items": checkpoint_items,
        "target_samples_per_char": target_samples_per_char,
    }

    if worker_count <= 1:
        progress_queue: queue_module.Queue = queue_module.Queue()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future_map = {
                executor.submit(
                    _process_prepare_filter_book,
                    book_idx,
                    total_books,
                    book_name,
                    progress_queue=progress_queue,
                    **worker_kwargs,
                ): (book_idx, book_name)
                for book_idx, book_name in enumerate(selected_books, start=1)
            }
            pending = set(future_map)
            while pending:
                done, pending = concurrent.futures.wait(
                    pending,
                    timeout=0.2,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                _drain_progress_queue(progress_queue, preview_bar, reocr_bar)
                for future in done:
                    book_idx, book_name = future_map[future]
                    result = future.result()
                    if books_bar is not None:
                        books_bar.update(1)
                    writer(f"[done {book_idx}/{total_books}] {book_name}")
                    for line in result["logs"]:
                        writer(line)
                    if result["changed"]:
                        changed_books += 1
                    preview_processed_items += int(result["preview_processed"] or 0)
                    reocr_processed_items += int(result["reocr_processed"] or 0)
                    error_items += int(result["errors"] or 0)
            _drain_progress_queue(progress_queue, preview_bar, reocr_bar)
    else:
        ctx = multiprocessing.get_context("spawn")
        with multiprocessing.Manager() as manager:
            progress_queue = manager.Queue()
            with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count, mp_context=ctx) as executor:
                future_map = {
                    executor.submit(
                        _process_prepare_filter_book,
                        book_idx,
                        total_books,
                        book_name,
                        progress_queue=progress_queue,
                        **worker_kwargs,
                    ): (book_idx, book_name)
                    for book_idx, book_name in enumerate(selected_books, start=1)
                }
                pending = set(future_map)
                completed = 0
                while pending:
                    done, pending = concurrent.futures.wait(
                        pending,
                        timeout=0.2,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    _drain_progress_queue(progress_queue, preview_bar, reocr_bar)
                    for future in done:
                        book_idx, book_name = future_map[future]
                        completed += 1
                        try:
                            result = future.result()
                        except Exception as exc:
                            if books_bar is not None:
                                books_bar.update(1)
                            writer(f"[{book_idx}/{total_books}] {book_name}: worker 失败: {exc}")
                            error_items += 1
                            continue
                        if books_bar is not None:
                            books_bar.update(1)
                        writer(f"[done {completed}/{total_books}] {book_name}")
                        for line in result["logs"]:
                            writer(line)
                        if result["changed"]:
                            changed_books += 1
                        preview_processed_items += int(result["preview_processed"] or 0)
                        reocr_processed_items += int(result["reocr_processed"] or 0)
                        error_items += int(result["errors"] or 0)
                _drain_progress_queue(progress_queue, preview_bar, reocr_bar)

    if books_bar is not None:
        books_bar.total = max(int(books_bar.n), int(books_bar.total or 0))
        books_bar.refresh()
        books_bar.close()
    if preview_bar is not None:
        preview_bar.close()
    if reocr_bar is not None:
        reocr_bar.close()

    print("-" * 60)
    print("Filter 准备完成")
    print(f"changed_books: {changed_books}")
    print(f"preview_processed_items: {preview_processed_items}")
    print(f"reocr_processed_items: {reocr_processed_items}")
    print(f"error_items: {error_items}")
    return 0 if error_items == 0 else 1
