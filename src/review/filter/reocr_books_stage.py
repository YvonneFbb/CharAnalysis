"""Run engine-specific reOCR over segmented atlas crops."""

from __future__ import annotations

import concurrent.futures
import os
import tempfile
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from src.review import config as review_config
from src.review.ocr.livetext import ocr_image
from src.review.paddle.core import call_paddle_batch, encode_png_bytes
from src.review.storage.reocr_books import (
    DEFAULT_REOCR_ENGINE,
    VALID_REOCR_ENGINES,
    ensure_reocr_item,
    normalize_engine_name,
    read_reocr_book,
    write_reocr_book,
)
from src.review.storage.review_books import utc_now_iso
from src.review.storage.segment_books import read_segment_book


PROJECT_ROOT = review_config.PROJECT_ROOT
PADDLE_CONFIG = review_config.PADDLE_CONFIG
DEFAULT_REOCR_PAD = int(PADDLE_CONFIG.get("reocr_pad", 12) or 12)
FILTER_CONFIG = review_config.FILTER_CONFIG
TARGET_MATCHES_PER_CHAR = int(FILTER_CONFIG.get("reocr_target_matches_per_char", 0) or 0)
REOCR_SHEET_MAX_SLOTS = int(FILTER_CONFIG.get("reocr_sheet_max_slots", 24) or 24)
REOCR_SHEET_GUTTER_PX = int(FILTER_CONFIG.get("reocr_sheet_gutter_px", 40) or 40)
REOCR_SHEET_MARGIN_PX = int(FILTER_CONFIG.get("reocr_sheet_margin_px", 32) or 32)
REOCR_SHEET_MAX_EDGE = int(FILTER_CONFIG.get("reocr_sheet_max_edge", 3072) or 3072)
_ATLAS_IMAGE_CACHE: "OrderedDict[str, np.ndarray]" = OrderedDict()
ATLAS_IMAGE_CACHE_SIZE = 8
_LIVETEXT_TMP_PATH: Optional[str] = None
_LIVETEXT_TMP_DIR: Optional[str] = None


def normalize_ocr_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    return "".join(str(text).split())


def _zero_bbox() -> Dict:
    return {"x": 0, "y": 0, "width": 0, "height": 0}


def _bbox_from_points(points: Optional[List]) -> Dict:
    if not isinstance(points, list) or not points:
        return _zero_bbox()
    xs: List[int] = []
    ys: List[int] = []
    for point in points:
        if isinstance(point, dict):
            xs.append(int(point.get("x") or 0))
            ys.append(int(point.get("y") or 0))
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            xs.append(int(point[0] or 0))
            ys.append(int(point[1] or 0))
    if not xs or not ys:
        return _zero_bbox()
    x0 = min(xs)
    y0 = min(ys)
    x1 = max(xs)
    y1 = max(ys)
    return {
        "x": x0,
        "y": y0,
        "width": max(0, x1 - x0),
        "height": max(0, y1 - y0),
    }


def _coerce_bbox(bbox: Optional[object]) -> Dict:
    if isinstance(bbox, dict):
        return {
            "x": int(bbox.get("x") or 0),
            "y": int(bbox.get("y") or 0),
            "width": int(bbox.get("width") or 0),
            "height": int(bbox.get("height") or 0),
        }
    if isinstance(bbox, (list, tuple)):
        return _bbox_from_points(list(bbox))
    return _zero_bbox()


def _unpad_bbox(bbox: Optional[Dict], pad: int, image_shape: Tuple[int, int]) -> Dict:
    bbox = bbox or {}
    img_h, img_w = image_shape
    x = max(0, int(bbox.get("x") or 0) - pad)
    y = max(0, int(bbox.get("y") or 0) - pad)
    width = int(bbox.get("width") or 0)
    height = int(bbox.get("height") or 0)
    max_w = max(0, img_w - x)
    max_h = max(0, img_h - y)
    return {
        "x": x,
        "y": y,
        "width": max(0, min(width, max_w)),
        "height": max(0, min(height, max_h)),
    }


def _normalize_polygon(points: Optional[List], pad: int, image_shape: Tuple[int, int]) -> List[Dict]:
    out: List[Dict] = []
    if not isinstance(points, list):
        return out
    img_h, img_w = image_shape
    for point in points:
        if isinstance(point, dict):
            px = int(point.get("x") or 0) - pad
            py = int(point.get("y") or 0) - pad
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            px = int(point[0] or 0) - pad
            py = int(point[1] or 0) - pad
        else:
            continue
        out.append({
            "x": max(0, min(img_w, px)),
            "y": max(0, min(img_h, py)),
        })
    return out


def _build_detection_record(
    *,
    text: Optional[str],
    confidence: Optional[float],
    bbox_cell: Dict,
) -> Dict:
    return {
        "text": str(text or ""),
        "confidence": float(confidence or 0.0),
        "bbox_cell": _coerce_bbox(bbox_cell),
    }


def _sort_detection_priority(det: Dict) -> Tuple[float, int, int]:
    bbox = det.get("bbox_cell") or det.get("bbox_segment") or det.get("bbox_segmented") or {}
    area = int(bbox.get("width") or 0) * int(bbox.get("height") or 0)
    text = normalize_ocr_text(det.get("text"))
    return (
        -float(det.get("confidence") or 0.0),
        -area,
        0 if len(text) == 1 else 1,
    )


def _pick_primary_detection(detections: List[Dict], expected_char: Optional[str] = None) -> Optional[Dict]:
    if not detections:
        return None
    if expected_char:
        expected = normalize_ocr_text(expected_char)
        matched = [det for det in detections if normalize_ocr_text(det.get("text")) == expected]
        if matched:
            return sorted(matched, key=_sort_detection_priority)[0]
    return sorted(detections, key=_sort_detection_priority)[0]


def _compose_text_from_detections(detections: List[Dict], primary: Optional[Dict]) -> str:
    if primary is not None:
        return str(primary.get("text") or "")
    return "".join(str(det.get("text") or "") for det in detections)


def _read_cached_atlas_image(relpath: str) -> np.ndarray:
    atlas_path = PROJECT_ROOT / relpath
    key = str(atlas_path.resolve())
    cached = _ATLAS_IMAGE_CACHE.get(key)
    if cached is not None:
        _ATLAS_IMAGE_CACHE.move_to_end(key)
        return cached
    image = cv2.imread(str(atlas_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法加载 atlas: {atlas_path}")
    _ATLAS_IMAGE_CACHE[key] = image
    _ATLAS_IMAGE_CACHE.move_to_end(key)
    while len(_ATLAS_IMAGE_CACHE) > ATLAS_IMAGE_CACHE_SIZE:
        _ATLAS_IMAGE_CACHE.popitem(last=False)
    return image


def _crop_from_atlas(relpath: str, bbox: Dict) -> np.ndarray:
    atlas = _read_cached_atlas_image(relpath)
    x = int(bbox.get("x") or 0)
    y = int(bbox.get("y") or 0)
    width = int(bbox.get("width") or 0)
    height = int(bbox.get("height") or 0)
    if width <= 0 or height <= 0:
        raise ValueError("atlas bbox 无效")
    cropped = atlas[y:y + height, x:x + width]
    if cropped.size == 0:
        raise ValueError("atlas crop 为空")
    return cropped


def _pad_segmented_image(img: np.ndarray, pad: int) -> np.ndarray:
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


def _set_livetext_tmp_dir(tmp_dir: Optional[str]) -> None:
    global _LIVETEXT_TMP_DIR, _LIVETEXT_TMP_PATH
    resolved = str(Path(tmp_dir).resolve()) if tmp_dir else None
    if resolved == _LIVETEXT_TMP_DIR:
        return
    if _LIVETEXT_TMP_PATH and os.path.exists(_LIVETEXT_TMP_PATH):
        try:
            os.unlink(_LIVETEXT_TMP_PATH)
        except OSError:
            pass
    _LIVETEXT_TMP_DIR = resolved
    _LIVETEXT_TMP_PATH = None


def _run_livetext_ocr(image_bgr: np.ndarray) -> Dict:
    global _LIVETEXT_TMP_PATH
    try:
        if _LIVETEXT_TMP_PATH is None:
            fd, temp_path = tempfile.mkstemp(
                suffix=".png",
                prefix="reocr_livetext_",
                dir=_LIVETEXT_TMP_DIR,
            )
            os.close(fd)
            _LIVETEXT_TMP_PATH = temp_path
        temp_image = _LIVETEXT_TMP_PATH
        if not cv2.imwrite(temp_image, image_bgr):
            raise RuntimeError("无法写入 LiveText 临时图片")
        result = ocr_image(temp_image, verbose=False, save_output=False)
        if not result.get("success"):
            if str(result.get("error") or "") == "LiveText 无检测结果":
                img_h, img_w = image_bgr.shape[:2]
                return {
                    "text": "",
                    "confidence": 0.0,
                    "primary_detection": None,
                    "detections": [],
                    "image_shape": {"width": img_w, "height": img_h},
                }
            raise RuntimeError(result.get("error") or "LiveText reOCR 失败")
        characters = result.get("characters") or []
        detections: List[Dict] = []
        img_h, img_w = image_bgr.shape[:2]
        for ch in characters:
            if not isinstance(ch, dict):
                continue
            bbox_padded = ch.get("bbox") or _zero_bbox()
            detections.append(_build_detection_record(
                text=ch.get("text"),
                confidence=ch.get("confidence"),
                bbox_cell={
                    "x": int(bbox_padded.get("x") or 0),
                    "y": int(bbox_padded.get("y") or 0),
                    "width": int(bbox_padded.get("width") or 0),
                    "height": int(bbox_padded.get("height") or 0),
                },
            ))
        primary = _pick_primary_detection(detections)
        return {
            "text": _compose_text_from_detections(detections, primary),
            "confidence": float(primary.get("confidence") or 0.0) if primary else 0.0,
            "primary_detection": primary,
            "detections": detections,
            "image_shape": {"width": img_w, "height": img_h},
        }
    except Exception:
        raise


def _count_existing_ready_matches(
    reocr_book: Dict,
    chars: Optional[Iterable[str]],
    limit_chars: Optional[int],
    pad: int,
) -> Dict[str, int]:
    selected_chars = set(str(ch) for ch in (chars or []))
    char_items = list((reocr_book or {}).items())
    if selected_chars:
        char_items = [(char, entry) for char, entry in char_items if char in selected_chars]
    if limit_chars is not None:
        char_items = char_items[: max(0, int(limit_chars))]
    counts: Dict[str, int] = {}
    for char, char_entry in char_items:
        if not isinstance(char_entry, dict):
            continue
        total = 0
        for _instance_id, item in ((char_entry.get("items") or {}).items()):
            if not isinstance(item, dict):
                continue
            if str(item.get("state") or "") != "ready":
                continue
            if item.get("matches") is not True:
                continue
            if int(item.get("pad") or 0) != int(pad):
                continue
            total += 1
        if total > 0:
            counts[str(char)] = total
    return counts


def _target_reached(done_matches: Dict[str, int], char_targets: Dict[str, int], char: str) -> bool:
    target = int(char_targets.get(char, 0) or 0)
    if target <= 0:
        return False
    return int(done_matches.get(char, 0)) >= target


def _iter_segment_char_items(
    segment_book: Dict,
    chars: Optional[Iterable[str]],
    limit_chars: Optional[int],
) -> List[Tuple[str, Dict]]:
    selected_chars = set(str(ch) for ch in (chars or []))
    char_items = list((segment_book or {}).items())
    if selected_chars:
        char_items = [(char, entry) for char, entry in char_items if char in selected_chars]
    if limit_chars is not None:
        char_items = char_items[: max(0, int(limit_chars))]
    return char_items


def _sheet_layout_for(
    count: int,
    cell_w: int,
    cell_h: int,
    gutter: int = REOCR_SHEET_GUTTER_PX,
    margin: int = REOCR_SHEET_MARGIN_PX,
    max_edge: int = REOCR_SHEET_MAX_EDGE,
) -> Optional[Dict]:
    if count <= 0 or cell_w <= 0 or cell_h <= 0:
        return None
    usable = max_edge - margin * 2
    if usable <= 0:
        return None
    max_cols = max(1, (usable + gutter) // max(1, cell_w + gutter))
    max_rows = max(1, (usable + gutter) // max(1, cell_h + gutter))
    best: Optional[Dict] = None
    for cols in range(1, min(count, max_cols) + 1):
        rows = (count + cols - 1) // cols
        if rows > max_rows:
            continue
        width = margin * 2 + cols * cell_w + max(0, cols - 1) * gutter
        height = margin * 2 + rows * cell_h + max(0, rows - 1) * gutter
        if width > max_edge or height > max_edge:
            continue
        area = width * height
        aspect = abs(width - height)
        candidate = {
            "cols": cols,
            "rows": rows,
            "width": width,
            "height": height,
            "cell_w": cell_w,
            "cell_h": cell_h,
            "gutter": gutter,
            "margin": margin,
            "area": area,
            "aspect": aspect,
        }
        if best is None or (area, aspect, rows) < (best["area"], best["aspect"], best["rows"]):
            best = candidate
    return best


def _load_segment_task(task: Tuple[str, str, Dict], pad: int) -> Dict:
    char, instance_id, segment_item = task
    cropped = _crop_from_atlas(segment_item.get("atlas_relpath") or "", segment_item.get("atlas_bbox") or {})
    padded = _pad_segmented_image(cropped, pad)
    return {
        "task": task,
        "char": char,
        "instance_id": instance_id,
        "cropped_shape": cropped.shape[:2],
        "padded_shape": padded.shape[:2],
        "padded_image": padded,
    }


def _build_livetext_batches(
    tasks: List[Tuple[str, str, Dict]],
    pad: int,
    sheet_max_slots: int,
) -> Tuple[List[Dict], List[Tuple[Tuple[str, str, Dict], Dict]]]:
    batches: List[Dict] = []
    errors: List[Tuple[Tuple[str, str, Dict], Dict]] = []
    current: List[Dict] = []
    current_w = 0
    current_h = 0
    current_layout: Optional[Dict] = None

    def flush() -> None:
        nonlocal current, current_w, current_h, current_layout
        if current and current_layout is not None:
            batches.append({"items": current, "layout": current_layout})
        current = []
        current_w = 0
        current_h = 0
        current_layout = None

    for task in tasks:
        char, _instance_id, segment_item = task
        if segment_item.get("state") != "ready":
            errors.append((task, {
                "state": "error",
                "text": None,
                "confidence": None,
                "matches": None,
                "error": segment_item.get("error") or "segment 未完成",
                "pad": pad,
                "duration_ms": 0,
                "primary_detection": None,
                "detections": [],
            }))
            continue
        try:
            prepared = _load_segment_task(task, pad)
        except Exception as exc:
            errors.append((task, {
                "state": "error",
                "text": None,
                "confidence": None,
                "matches": None,
                "error": str(exc),
                "pad": pad,
                "duration_ms": 0,
                "primary_detection": None,
                "detections": [],
            }))
            continue
        next_count = len(current) + 1
        next_w = max(current_w, int(prepared["padded_shape"][1]))
        next_h = max(current_h, int(prepared["padded_shape"][0]))
        next_layout = _sheet_layout_for(next_count, next_w, next_h)
        if current and (len(current) >= max(1, int(sheet_max_slots or REOCR_SHEET_MAX_SLOTS)) or next_layout is None):
            flush()
            next_count = 1
            next_w = int(prepared["padded_shape"][1])
            next_h = int(prepared["padded_shape"][0])
            next_layout = _sheet_layout_for(next_count, next_w, next_h)
        if next_layout is None:
            errors.append((task, {
                "state": "error",
                "text": None,
                "confidence": None,
                "matches": None,
                "error": (
                    f"拼图尺寸超限: padded="
                    f"{int(prepared['padded_shape'][1])}x{int(prepared['padded_shape'][0])}"
                ),
                "pad": pad,
                "duration_ms": 0,
                "primary_detection": None,
                "detections": [],
            }))
            continue
        current.append(prepared)
        current_w = next_w
        current_h = next_h
        current_layout = next_layout
    flush()
    return batches, errors


def _compose_livetext_sheet(batch: Dict) -> Tuple[np.ndarray, List[Dict]]:
    items = list(batch.get("items") or [])
    layout = batch.get("layout") or {}
    sheet = np.full(
        (int(layout.get("height") or 0), int(layout.get("width") or 0), 3),
        255,
        dtype=np.uint8,
    )
    slots: List[Dict] = []
    cols = int(layout.get("cols") or 1)
    cell_w = int(layout.get("cell_w") or 0)
    cell_h = int(layout.get("cell_h") or 0)
    margin = int(layout.get("margin") or 0)
    gutter = int(layout.get("gutter") or 0)
    for idx, prepared in enumerate(items):
        row = idx // cols
        col = idx % cols
        cell_x = margin + col * (cell_w + gutter)
        cell_y = margin + row * (cell_h + gutter)
        img = prepared["padded_image"]
        img_h, img_w = img.shape[:2]
        offset_x = cell_x + max(0, (cell_w - img_w) // 2)
        offset_y = cell_y + max(0, (cell_h - img_h) // 2)
        sheet[offset_y:offset_y + img_h, offset_x:offset_x + img_w] = img
        slots.append({
            **prepared,
            "cell_bbox": {"x": cell_x, "y": cell_y, "width": cell_w, "height": cell_h},
            "placed_bbox": {"x": offset_x, "y": offset_y, "width": img_w, "height": img_h},
        })
    return sheet, slots


def _point_in_bbox(x: float, y: float, bbox: Dict) -> bool:
    bx = int(bbox.get("x") or 0)
    by = int(bbox.get("y") or 0)
    bw = int(bbox.get("width") or 0)
    bh = int(bbox.get("height") or 0)
    return bx <= x <= bx + bw and by <= y <= by + bh


def _bbox_intersection_area(a: Dict, b: Dict) -> int:
    ax0 = int(a.get("x") or 0)
    ay0 = int(a.get("y") or 0)
    ax1 = ax0 + int(a.get("width") or 0)
    ay1 = ay0 + int(a.get("height") or 0)
    bx0 = int(b.get("x") or 0)
    by0 = int(b.get("y") or 0)
    bx1 = bx0 + int(b.get("width") or 0)
    by1 = by0 + int(b.get("height") or 0)
    iw = max(0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0, min(ay1, by1) - max(ay0, by0))
    return iw * ih


def _translate_polygon(points: Optional[List], dx: int, dy: int) -> List[Dict]:
    out: List[Dict] = []
    for point in list(points or []):
        if not isinstance(point, dict):
            continue
        out.append({
            "x": int(point.get("x") or 0) - int(dx),
            "y": int(point.get("y") or 0) - int(dy),
        })
    return out


def _match_detection_slot(det: Dict, slots: List[Dict]) -> Optional[int]:
    bbox = det.get("bbox_cell") or det.get("bbox_input") or det.get("bbox_padded") or _zero_bbox()
    center_x = float(int(bbox.get("x") or 0) + int(bbox.get("width") or 0) / 2.0)
    center_y = float(int(bbox.get("y") or 0) + int(bbox.get("height") or 0) / 2.0)
    candidates = [
        idx for idx, slot in enumerate(slots)
        if _point_in_bbox(center_x, center_y, slot.get("cell_bbox") or {})
    ]
    if not candidates:
        best_idx = None
        best_area = 0
        for idx, slot in enumerate(slots):
            area = _bbox_intersection_area(bbox, slot.get("cell_bbox") or {})
            if area > best_area:
                best_area = area
                best_idx = idx
        return best_idx if best_area > 0 else None
    if len(candidates) == 1:
        return candidates[0]
    ranked = sorted(
        candidates,
        key=lambda idx: (
            -_bbox_intersection_area(bbox, slots[idx].get("placed_bbox") or {}),
            -_bbox_intersection_area(bbox, slots[idx].get("cell_bbox") or {}),
            idx,
        ),
    )
    return ranked[0] if ranked else None


def _relativize_detection_to_slot(det: Dict, slot: Dict, pad: int) -> Dict:
    placed = slot.get("placed_bbox") or {}
    bbox_sheet = det.get("bbox_cell") or det.get("bbox_input") or det.get("bbox_padded") or _zero_bbox()
    rel_bbox = {
        "x": max(0, int(bbox_sheet.get("x") or 0) - int(placed.get("x") or 0)),
        "y": max(0, int(bbox_sheet.get("y") or 0) - int(placed.get("y") or 0)),
        "width": int(bbox_sheet.get("width") or 0),
        "height": int(bbox_sheet.get("height") or 0),
    }
    return _build_detection_record(
        text=det.get("text"),
        confidence=det.get("confidence"),
        bbox_cell=rel_bbox,
    )


def _run_livetext_sheet_batch(batch: Dict, pad: int) -> List[Tuple[Tuple[str, str, Dict], Dict]]:
    sheet, slots = _compose_livetext_sheet(batch)
    started = time.time()
    parsed = _run_livetext_ocr(sheet)
    duration_ms = int((time.time() - started) * 1000 / max(1, len(slots)))
    assigned: List[List[Dict]] = [[] for _ in slots]
    for det in list(parsed.get("detections") or []):
        if not isinstance(det, dict):
            continue
        slot_idx = _match_detection_slot(det, slots)
        if slot_idx is None:
            continue
        assigned[slot_idx].append(_relativize_detection_to_slot(det, slots[slot_idx], pad))

    rows: List[Tuple[Tuple[str, str, Dict], Dict]] = []
    for slot, detections in zip(slots, assigned):
        task = slot["task"]
        char = slot["char"]
        primary = _pick_primary_detection(detections, expected_char=char)
        text = _compose_text_from_detections(detections, primary)
        rows.append((task, {
            "state": "ready",
            "text": text,
            "confidence": float(primary.get("confidence") or 0.0) if primary else 0.0,
            "matches": _match_reocr_text(char, text),
            "error": None,
            "pad": pad,
            "duration_ms": duration_ms,
            "ocr_mode": "sheet",
            "cell_size": {
                "width": int(slot["padded_shape"][1]),
                "height": int(slot["padded_shape"][0]),
            },
            "detection_count": len(detections),
            "primary_detection": primary,
        }))
    return rows


def _match_reocr_text(char: str, text: Optional[str]) -> bool:
    return normalize_ocr_text(char) == normalize_ocr_text(text)


def _extract_paddle_detections(payload: object, pad: int, image_shape: Tuple[int, int], expected_char: str) -> Dict:
    detections: List[Dict] = []
    img_h, img_w = image_shape

    def append_detection(
        text: Optional[str],
        confidence: Optional[float],
        bbox_padded: Optional[Dict] = None,
        polygon: Optional[List] = None,
    ) -> None:
        bbox_padded_local = _coerce_bbox(bbox_padded)
        if not any(int(bbox_padded_local.get(k) or 0) for k in ("x", "y", "width", "height")):
            bbox_padded_local = _bbox_from_points(polygon)
        detections.append(_build_detection_record(
            text=text,
            confidence=confidence,
            bbox_cell=bbox_padded_local,
        ))

    if isinstance(payload, dict):
        regions = payload.get("text_regions")
        if isinstance(regions, list):
            for item in regions:
                if not isinstance(item, dict):
                    continue
                append_detection(
                    text=item.get("text") or item.get("transcription"),
                    confidence=item.get("confidence") if item.get("confidence") is not None else item.get("score"),
                    bbox_padded=item.get("bbox") or item.get("box"),
                    polygon=item.get("points") or item.get("polygon"),
                )
        result_list = payload.get("result")
        if isinstance(result_list, list):
            for item in result_list:
                if not isinstance(item, dict):
                    continue
                append_detection(
                    text=item.get("text") or item.get("transcription"),
                    confidence=item.get("confidence") if item.get("confidence") is not None else item.get("score"),
                    bbox_padded=item.get("bbox") or item.get("box"),
                    polygon=item.get("points") or item.get("polygon"),
                )
        data_list = payload.get("data")
        if isinstance(data_list, list):
            for item in data_list:
                if not isinstance(item, dict):
                    continue
                append_detection(
                    text=item.get("text") or item.get("transcription"),
                    confidence=item.get("confidence") if item.get("confidence") is not None else item.get("score"),
                    bbox_padded=item.get("bbox") or item.get("box"),
                    polygon=item.get("points") or item.get("polygon"),
                )
        if not detections and ("text" in payload or "transcription" in payload):
            append_detection(
                text=payload.get("text") or payload.get("transcription"),
                confidence=payload.get("confidence") if payload.get("confidence") is not None else payload.get("score"),
                bbox_padded=payload.get("bbox") or payload.get("box") or {
                    "x": pad,
                    "y": pad,
                    "width": max(0, img_w - pad * 2),
                    "height": max(0, img_h - pad * 2),
                },
                polygon=payload.get("points") or payload.get("polygon"),
            )
    elif isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            append_detection(
                text=item.get("text") or item.get("transcription"),
                confidence=item.get("confidence") if item.get("confidence") is not None else item.get("score"),
                bbox_padded=item.get("bbox") or item.get("box"),
                polygon=item.get("points") or item.get("polygon"),
            )

    primary = _pick_primary_detection(detections, expected_char=expected_char)
    return {
        "text": _compose_text_from_detections(detections, primary),
        "confidence": float(primary.get("confidence") or 0.0) if primary else 0.0,
        "primary_detection": primary,
        "detections": detections,
    }


def _collect_segment_tasks(
    segment_book: Dict,
    reocr_book: Dict,
    chars: Optional[Iterable[str]],
    limit_chars: Optional[int],
    limit_instances: Optional[int],
    force: bool,
    pad: int,
) -> List[Tuple[str, str, Dict]]:
    tasks: List[Tuple[str, str, Dict]] = []
    for char, char_entry in _iter_segment_char_items(segment_book, chars, limit_chars):
        if not isinstance(char_entry, dict):
            continue
        segment_items = list(((char_entry.get("items") or {}).items()))
        if limit_instances is not None:
            segment_items = segment_items[: max(0, int(limit_instances))]
        for instance_id, segment_item in segment_items:
            if not isinstance(segment_item, dict):
                continue
            reocr_item = ((((reocr_book.get(char) or {}).get("items")) or {}).get(instance_id)) or {}
            if not force and reocr_item.get("state") in {"ready", "error"} and int(reocr_item.get("pad") or 0) == int(pad):
                continue
            tasks.append((char, instance_id, segment_item))
    return tasks


def _build_char_match_targets(
    segment_book: Dict,
    chars: Optional[Iterable[str]],
    limit_chars: Optional[int],
    target_matches: int,
) -> Dict[str, int]:
    targets: Dict[str, int] = {}
    resolved_target = max(0, int(target_matches or 0))
    if resolved_target <= 0:
        return targets
    for char, char_entry in _iter_segment_char_items(segment_book, chars, limit_chars):
        if not isinstance(char_entry, dict):
            continue
        ready_count = sum(
            1
            for _instance_id, item in ((char_entry.get("items") or {}).items())
            if isinstance(item, dict) and str(item.get("state") or "pending") == "ready"
        )
        if ready_count > 0:
            targets[str(char)] = min(resolved_target, int(ready_count))
    return targets


def _count_segment_candidates(
    segment_book: Dict,
    chars: Optional[Iterable[str]],
    limit_chars: Optional[int],
    limit_instances: Optional[int],
) -> int:
    total = 0
    for _char, char_entry in _iter_segment_char_items(segment_book, chars, limit_chars):
        if not isinstance(char_entry, dict):
            continue
        items = list(((char_entry.get("items") or {}).items()))
        if limit_instances is not None:
            items = items[: max(0, int(limit_instances))]
        total += sum(1 for _, item in items if isinstance(item, dict))
    return total


def _run_paddle_sheet_batch(
    batch: Dict,
    paddle_url: str,
    timeout: int,
    pad: int,
) -> List[Tuple[Tuple[str, str, Dict], Dict]]:
    sheet, slots = _compose_livetext_sheet(batch)
    started = time.time()
    parsed = _extract_paddle_detections(
        call_paddle_batch([encode_png_bytes(sheet)], paddle_url=paddle_url, timeout=timeout, return_payload=True)[0],
        pad=pad,
        image_shape=sheet.shape[:2],
        expected_char="",
    )
    duration_ms = int((time.time() - started) * 1000 / max(1, len(slots)))
    assigned: List[List[Dict]] = [[] for _ in slots]
    for det in list(parsed.get("detections") or []):
        if not isinstance(det, dict):
            continue
        slot_idx = _match_detection_slot(det, slots)
        if slot_idx is None:
            continue
        assigned[slot_idx].append(_relativize_detection_to_slot(det, slots[slot_idx], pad))

    rows: List[Tuple[Tuple[str, str, Dict], Dict]] = []
    for slot, detections in zip(slots, assigned):
        task = slot["task"]
        char = slot["char"]
        primary = _pick_primary_detection(detections, expected_char=char)
        text = _compose_text_from_detections(detections, primary)
        rows.append((task, {
            "state": "ready",
            "text": text,
            "confidence": float(primary.get("confidence") or 0.0) if primary else 0.0,
            "matches": _match_reocr_text(char, text),
            "error": None,
            "pad": pad,
            "duration_ms": duration_ms,
            "ocr_mode": "sheet",
            "cell_size": {
                "width": int(slot["padded_shape"][1]),
                "height": int(slot["padded_shape"][0]),
            },
            "detection_count": len(detections),
            "primary_detection": primary,
        }))
    return rows


def _process_reocr_tasks(
    *,
    engine_name: str,
    tasks: List[Tuple[str, str, Dict]],
    existing_matches: Dict[str, int],
    char_targets: Dict[str, int],
    pad: int,
    paddle_url: Optional[str],
    timeout: int,
    batch_size: int,
    sheet_max_slots: int,
) -> Tuple[List[Tuple[Tuple[str, str, Dict], Dict]], int]:
    skipped = 0
    if engine_name == "paddle":
        resolved_paddle_url = paddle_url or PADDLE_CONFIG.get("url")
        if not resolved_paddle_url:
            raise ValueError("缺少 Paddle 服务地址")
        result_rows = []
        done_matches: Dict[str, int] = dict(existing_matches)
        pending: List[Tuple[str, str, Dict]] = []
        for task in tasks:
            char = task[0]
            if _target_reached(done_matches, char_targets, char):
                skipped += 1
                continue
            pending.append(task)
        if pending:
            pending_by_char: Dict[str, List[Tuple[str, str, Dict]]] = defaultdict(list)
            for task in pending:
                pending_by_char[str(task[0])].append(task)
            for char, char_tasks in pending_by_char.items():
                batches, preload_errors = _build_livetext_batches(
                    char_tasks,
                    pad=pad,
                    sheet_max_slots=sheet_max_slots,
                )
                result_rows.extend(preload_errors)
                active_batches: List[Dict] = []
                for batch in batches:
                    if _target_reached(done_matches, char_targets, char):
                        skipped += len(list(batch.get("items") or []))
                        continue
                    active_tasks = []
                    for prepared in list(batch.get("items") or []):
                        if _target_reached(done_matches, char_targets, char):
                            skipped += 1
                            continue
                        active_tasks.append(prepared)
                    if not active_tasks:
                        continue
                    batch = {**batch, "items": active_tasks}
                    batch["layout"] = _sheet_layout_for(
                        len(active_tasks),
                        max(int(item["padded_shape"][1]) for item in active_tasks),
                        max(int(item["padded_shape"][0]) for item in active_tasks),
                    )
                    if batch["layout"] is None:
                        for prepared in active_tasks:
                            result_rows.append((prepared["task"], {
                                "state": "error",
                                "text": None,
                                "confidence": None,
                                "matches": None,
                                "error": "拼图布局失败",
                                "pad": pad,
                                "duration_ms": 0,
                                "ocr_mode": "sheet",
                                "cell_size": {"width": 0, "height": 0},
                                "detection_count": 0,
                                "primary_detection": None,
                            }))
                        continue
                    active_batches.append(batch)

                step = max(1, int(batch_size or 1))
                for start in range(0, len(active_batches), step):
                    request_batches = active_batches[start:start + step]
                    if not request_batches:
                        continue
                    sheet_payloads: List[bytes] = []
                    sheet_meta: List[Tuple[Dict, np.ndarray, List[Dict]]] = []
                    for batch in request_batches:
                        sheet, slots = _compose_livetext_sheet(batch)
                        sheet_payloads.append(encode_png_bytes(sheet))
                        sheet_meta.append((batch, sheet, slots))
                    started = time.time()
                    try:
                        raw_results = call_paddle_batch(
                            sheet_payloads,
                            paddle_url=resolved_paddle_url,
                            timeout=timeout,
                            return_payload=True,
                        )
                    except Exception as exc:
                        raw_results = []
                        for _batch, sheet, _slots in sheet_meta:
                            raw_results.append({
                                "text": "",
                                "confidence": 0.0,
                                "text_regions": [],
                                "error": str(exc),
                            })
                    total_slots = sum(len(slots) for _batch, _sheet, slots in sheet_meta)
                    duration_ms = int((time.time() - started) * 1000 / max(1, total_slots))
                    for (batch, sheet, slots), raw_result in zip(sheet_meta, raw_results):
                        parsed = _extract_paddle_detections(
                            raw_result,
                            pad=pad,
                            image_shape=sheet.shape[:2],
                            expected_char="",
                        )
                        assigned: List[List[Dict]] = [[] for _ in slots]
                        for det in list(parsed.get("detections") or []):
                            if not isinstance(det, dict):
                                continue
                            slot_idx = _match_detection_slot(det, slots)
                            if slot_idx is None:
                                continue
                            assigned[slot_idx].append(_relativize_detection_to_slot(det, slots[slot_idx], pad))
                        for slot, detections in zip(slots, assigned):
                            task = slot["task"]
                            char = slot["char"]
                            primary = _pick_primary_detection(detections, expected_char=char)
                            text = _compose_text_from_detections(detections, primary)
                            result = {
                                "state": "ready",
                                "text": text,
                                "confidence": float(primary.get("confidence") or 0.0) if primary else 0.0,
                                "matches": _match_reocr_text(char, text),
                                "error": None,
                                "pad": pad,
                                "duration_ms": duration_ms,
                                "ocr_mode": "sheet",
                                "cell_size": {
                                    "width": int(slot["padded_shape"][1]),
                                    "height": int(slot["padded_shape"][0]),
                                },
                                "detection_count": len(detections),
                                "primary_detection": primary,
                            }
                            result_rows.append((task, result))
                            if result.get("matches") is True:
                                done_matches[char] = int(done_matches.get(char, 0)) + 1
    else:
        result_rows = []
        done_matches: Dict[str, int] = dict(existing_matches)
        pending: List[Tuple[str, str, Dict]] = []
        for task in tasks:
            char = task[0]
            if _target_reached(done_matches, char_targets, char):
                skipped += 1
                continue
            pending.append(task)
        pending_by_char: Dict[str, List[Tuple[str, str, Dict]]] = defaultdict(list)
        for task in pending:
            pending_by_char[str(task[0])].append(task)
        for char, char_tasks in pending_by_char.items():
            batches, preload_errors = _build_livetext_batches(
                char_tasks,
                pad=pad,
                sheet_max_slots=sheet_max_slots,
            )
            result_rows.extend(preload_errors)
            for batch in batches:
                if _target_reached(done_matches, char_targets, char):
                    skipped += len(list(batch.get("items") or []))
                    continue
                active_tasks = []
                for prepared in list(batch.get("items") or []):
                    if _target_reached(done_matches, char_targets, char):
                        skipped += 1
                        continue
                    active_tasks.append(prepared)
                if not active_tasks:
                    continue
                batch = {**batch, "items": active_tasks}
                batch["layout"] = _sheet_layout_for(
                    len(active_tasks),
                    max(int(item["padded_shape"][1]) for item in active_tasks),
                    max(int(item["padded_shape"][0]) for item in active_tasks),
                )
                if batch["layout"] is None:
                    for prepared in active_tasks:
                        result_rows.append((prepared["task"], {
                            "state": "error",
                            "text": None,
                            "confidence": None,
                            "matches": None,
                            "error": "拼图布局失败",
                            "pad": pad,
                            "duration_ms": 0,
                            "ocr_mode": "sheet",
                            "cell_size": {"width": 0, "height": 0},
                            "detection_count": 0,
                            "primary_detection": None,
                        }))
                    continue
                try:
                    batch_rows = _run_livetext_sheet_batch(batch, pad=pad)
                except Exception as exc:
                    batch_rows = [
                        (prepared["task"], {
                            "state": "error",
                            "text": None,
                            "confidence": None,
                            "matches": None,
                            "error": str(exc),
                            "pad": pad,
                            "duration_ms": 0,
                            "ocr_mode": "sheet",
                            "cell_size": {"width": 0, "height": 0},
                            "detection_count": 0,
                            "primary_detection": None,
                        })
                        for prepared in active_tasks
                    ]
                for task, result in batch_rows:
                    result_rows.append((task, result))
                    if result.get("matches") is True:
                        done_matches[char] = int(done_matches.get(char, 0)) + 1
    return result_rows, skipped


def _apply_result_rows(reocr_book: Dict, result_rows: List[Tuple[Tuple[str, str, Dict], Dict]], pad: int) -> Tuple[int, int]:
    processed = 0
    errors = 0
    for (char, instance_id, _segment_item), result in result_rows:
        item = ensure_reocr_item(reocr_book, char, instance_id)
        item.update({
            "state": result.get("state"),
            "timestamp": utc_now_iso(),
            "text": result.get("text"),
            "confidence": result.get("confidence"),
            "matches": result.get("matches"),
            "error": result.get("error"),
            "pad": int(result.get("pad") or pad),
            "duration_ms": int(result.get("duration_ms") or 0),
            "ocr_mode": result.get("ocr_mode"),
            "cell_size": result.get("cell_size"),
            "detection_count": int(result.get("detection_count") or 0),
            "primary_detection": result.get("primary_detection"),
        })
        processed += 1
        if result.get("state") == "error":
            errors += 1
    return processed, errors


def _reocr_book(
    book_name: str,
    engine: str,
    chars: Optional[Iterable[str]] = None,
    limit_chars: Optional[int] = None,
    limit_instances: Optional[int] = None,
    force: bool = False,
    pad: int = DEFAULT_REOCR_PAD,
    paddle_url: Optional[str] = None,
    timeout: int = 20,
    batch_size: int = 32,
    sheet_max_slots: int = REOCR_SHEET_MAX_SLOTS,
    tmp_dir: Optional[str] = None,
    target_matches: int = TARGET_MATCHES_PER_CHAR,
) -> Dict:
    engine_name = normalize_engine_name(engine)
    _set_livetext_tmp_dir(tmp_dir)
    segment_book = read_segment_book(book_name)
    if not segment_book:
        return {"book": book_name, "processed": 0, "skipped": 0, "errors": 1, "error": "segment book 不存在"}

    reocr_book = {} if force else (read_reocr_book(book_name, engine_name) or {})
    total_candidates = _count_segment_candidates(
        segment_book=segment_book,
        chars=chars,
        limit_chars=limit_chars,
        limit_instances=limit_instances,
    )
    tasks = _collect_segment_tasks(
        segment_book=segment_book,
        reocr_book=reocr_book,
        chars=chars,
        limit_chars=limit_chars,
        limit_instances=limit_instances,
        force=force,
        pad=pad,
    )
    char_targets = _build_char_match_targets(
        segment_book=segment_book,
        chars=chars,
        limit_chars=limit_chars,
        target_matches=int(target_matches or TARGET_MATCHES_PER_CHAR),
    )
    existing_matches = _count_existing_ready_matches(
        reocr_book=reocr_book,
        chars=chars,
        limit_chars=limit_chars,
        pad=pad,
    )
    skipped = max(0, total_candidates - len(tasks))
    result_rows, extra_skipped = _process_reocr_tasks(
        engine_name=engine_name,
        tasks=tasks,
        existing_matches=existing_matches,
        char_targets=char_targets,
        pad=pad,
        paddle_url=paddle_url,
        timeout=timeout,
        batch_size=batch_size,
        sheet_max_slots=sheet_max_slots,
    )
    processed, errors = _apply_result_rows(reocr_book, result_rows, pad)
    skipped += extra_skipped

    write_reocr_book(book_name, reocr_book, engine_name)
    return {"book": book_name, "processed": processed, "skipped": skipped, "errors": errors}


def _reocr_book_worker(args: Tuple[str, str, Optional[List[str]], Optional[int], Optional[int], bool, int, Optional[str], int, int, int, Optional[str], int]) -> Dict:
    return _reocr_book(
        book_name=args[0],
        engine=args[1],
        chars=args[2],
        limit_chars=args[3],
        limit_instances=args[4],
        force=args[5],
        pad=args[6],
        paddle_url=args[7],
        timeout=args[8],
        batch_size=args[9],
        target_matches=args[10],
        tmp_dir=args[11],
        sheet_max_slots=args[12],
    )


def run_reocr_books(
    books: Optional[Iterable[str]] = None,
    engine: str = DEFAULT_REOCR_ENGINE,
    chars: Optional[Iterable[str]] = None,
    limit_chars: Optional[int] = None,
    limit_instances: Optional[int] = None,
    workers: int = 1,
    force: bool = False,
    pad: int = DEFAULT_REOCR_PAD,
    paddle_url: Optional[str] = None,
    timeout: int = 20,
    batch_size: int = 32,
    tmp_dir: Optional[str] = None,
    sheet_max_slots: Optional[int] = None,
    target_matches: int = TARGET_MATCHES_PER_CHAR,
) -> int:
    engine_raw = str(engine or DEFAULT_REOCR_ENGINE).strip().lower()
    engine_names = list(VALID_REOCR_ENGINES) if engine_raw == "both" else [normalize_engine_name(engine_raw)]
    resolved_sheet_max_slots = max(1, int(sheet_max_slots or REOCR_SHEET_MAX_SLOTS))
    selected_books = list(books or [])
    if not selected_books:
        selected_books = sorted((review_config.SEGMENT_BOOKS_DIR.glob("*.json")), key=lambda p: p.name)
        selected_books = [p.stem for p in selected_books]
    if not selected_books:
        print("未找到 segment_books 数据")
        return 1
    overall_rc = 0
    for engine_name in engine_names:
        total_processed = 0
        total_skipped = 0
        total_errors = 0
        print(f"流程: segment_books -> {engine_name} reOCR -> reocr_books")
        print(
            f"books={len(selected_books)} workers={workers} force={int(bool(force))} pad={pad} "
            f"sheet_max_slots={resolved_sheet_max_slots}"
        )
        tasks = [
            (
                book_name,
                engine_name,
                list(chars) if chars else None,
                limit_chars,
                limit_instances,
                force,
                int(pad or 0),
                paddle_url,
                int(timeout or 20),
                int(batch_size or 32),
                int(target_matches or TARGET_MATCHES_PER_CHAR),
                tmp_dir,
                resolved_sheet_max_slots,
            )
            for book_name in selected_books
        ]
        progress_desc = (
            f"reocr进度({engine_name})"
            if int(workers or 1) <= 1
            else f"reocr进度({engine_name},{workers}进程)"
        )

        with tqdm(total=len(tasks), desc=progress_desc, unit="book", dynamic_ncols=True) as pbar:
            pbar.set_postfix({
                "processed": total_processed,
                "skipped": total_skipped,
                "errors": total_errors,
            }, refresh=False)

            if int(workers or 1) <= 1:
                for task in tasks:
                    result = _reocr_book_worker(task)
                    total_processed += int(result.get("processed") or 0)
                    total_skipped += int(result.get("skipped") or 0)
                    total_errors += int(result.get("errors") or 0)
                    pbar.update(1)
                    pbar.set_postfix({
                        "processed": total_processed,
                        "skipped": total_skipped,
                        "errors": total_errors,
                    }, refresh=False)
                    tqdm.write(
                        f"[reocr:{engine_name}] {result['book']}: processed={result.get('processed', 0)} "
                        f"skipped={result.get('skipped', 0)} errors={result.get('errors', 0)}"
                    )
            else:
                with concurrent.futures.ProcessPoolExecutor(max_workers=int(workers or 1)) as executor:
                    future_to_book = {
                        executor.submit(_reocr_book_worker, task): task[0]
                        for task in tasks
                    }
                    for future in concurrent.futures.as_completed(future_to_book):
                        book_name = future_to_book[future]
                        try:
                            result = future.result()
                        except Exception as exc:
                            result = {
                                "book": book_name,
                                "processed": 0,
                                "skipped": 0,
                                "errors": 1,
                                "error": str(exc),
                            }
                        total_processed += int(result.get("processed") or 0)
                        total_skipped += int(result.get("skipped") or 0)
                        total_errors += int(result.get("errors") or 0)
                        pbar.update(1)
                        pbar.set_postfix({
                            "processed": total_processed,
                            "skipped": total_skipped,
                            "errors": total_errors,
                        }, refresh=False)
                        tqdm.write(
                            f"[reocr:{engine_name}] {result['book']}: processed={result.get('processed', 0)} "
                            f"skipped={result.get('skipped', 0)} errors={result.get('errors', 0)}"
                        )

        print(
            f"[reocr:{engine_name}] done: books={len(tasks)} processed={total_processed} "
            f"skipped={total_skipped} errors={total_errors}"
        )
        if total_errors != 0:
            overall_rc = 1
    return overall_rc
