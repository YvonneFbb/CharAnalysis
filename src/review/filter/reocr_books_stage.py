"""Run engine-specific reOCR over segmented atlas crops."""

from __future__ import annotations

import concurrent.futures
import json
import os
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from src.review import config as review_config
from src.review.ocr.livetext import ocr_image
from src.review.paddle.core import call_paddle_batch, encode_png_bytes
from src.review.storage.reocr_books import (
    DEFAULT_REOCR_ENGINE,
    ensure_reocr_item,
    normalize_engine_name,
    read_reocr_book,
    write_reocr_book,
)
from src.review.storage.review_books import utc_now_iso
from src.review.storage.segment_books import read_segment_book


PROJECT_ROOT = review_config.PROJECT_ROOT
PADDLE_CONFIG = review_config.PADDLE_CONFIG
_ATLAS_IMAGE_CACHE: "OrderedDict[str, np.ndarray]" = OrderedDict()
ATLAS_IMAGE_CACHE_SIZE = 8


def normalize_ocr_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    return "".join(str(text).split())


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


def _run_livetext_ocr(image_bgr: np.ndarray) -> Tuple[str, float]:
    temp_image = None
    temp_json = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as image_fp:
            temp_image = image_fp.name
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as json_fp:
            temp_json = json_fp.name
        if not cv2.imwrite(temp_image, image_bgr):
            raise RuntimeError("无法写入 LiveText 临时图片")
        result = ocr_image(temp_image, output_path=temp_json, verbose=False)
        if not result.get("success"):
            raise RuntimeError(result.get("error") or "LiveText reOCR 失败")
        characters = result.get("characters") or []
        text = "".join(str(ch.get("text") or "") for ch in characters)
        confidence = max((float(ch.get("confidence") or 0.0) for ch in characters), default=0.0)
        return text, confidence
    finally:
        for path in (temp_image, temp_json):
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:
                    pass


def _match_reocr_text(char: str, text: Optional[str]) -> bool:
    return normalize_ocr_text(char) == normalize_ocr_text(text)


def _collect_segment_tasks(
    book_name: str,
    segment_book: Dict,
    reocr_book: Dict,
    chars: Optional[Iterable[str]],
    limit_chars: Optional[int],
    limit_instances: Optional[int],
    force: bool,
    pad: int,
) -> List[Tuple[str, str, Dict]]:
    selected_chars = set(str(ch) for ch in (chars or []))
    tasks: List[Tuple[str, str, Dict]] = []
    char_items = list((segment_book or {}).items())
    if selected_chars:
        char_items = [(char, char_entry) for char, char_entry in char_items if char in selected_chars]
    if limit_chars is not None:
        char_items = char_items[: max(0, int(limit_chars))]

    for char, char_entry in char_items:
        if not isinstance(char_entry, dict):
            continue
        items = list(((char_entry.get("items") or {}).items()))
        if limit_instances is not None:
            items = items[: max(0, int(limit_instances))]
        for instance_id, segment_item in items:
            if not isinstance(segment_item, dict):
                continue
            reocr_item = ((((reocr_book.get(char) or {}).get("items")) or {}).get(instance_id)) or {}
            if not force and reocr_item.get("state") in {"ready", "error"} and int(reocr_item.get("pad") or 0) == int(pad):
                continue
            tasks.append((char, instance_id, segment_item))
    return tasks


def _count_segment_candidates(
    segment_book: Dict,
    chars: Optional[Iterable[str]],
    limit_chars: Optional[int],
    limit_instances: Optional[int],
) -> int:
    selected_chars = set(str(ch) for ch in (chars or []))
    char_items = list((segment_book or {}).items())
    if selected_chars:
        char_items = [(char, char_entry) for char, char_entry in char_items if char in selected_chars]
    if limit_chars is not None:
        char_items = char_items[: max(0, int(limit_chars))]
    total = 0
    for _char, char_entry in char_items:
        if not isinstance(char_entry, dict):
            continue
        items = list(((char_entry.get("items") or {}).items()))
        if limit_instances is not None:
            items = items[: max(0, int(limit_instances))]
        total += len(items)
    return total


def _run_paddle_batch_for_tasks(tasks: List[Tuple[str, str, Dict]], paddle_url: str, timeout: int, batch_size: int, pad: int) -> List[Tuple[Tuple[str, str, Dict], Dict]]:
    results: List[Tuple[Tuple[str, str, Dict], Dict]] = []
    batch_size = max(1, int(batch_size or 1))
    for start in range(0, len(tasks), batch_size):
        batch = tasks[start:start + batch_size]
        payloads: List[bytes] = []
        valid_batch: List[Tuple[str, str, Dict]] = []
        for task in batch:
            _, _, segment_item = task
            if segment_item.get("state") != "ready":
                results.append((task, {
                    "state": "error",
                    "text": None,
                    "confidence": None,
                    "matches": None,
                    "error": segment_item.get("error") or "segment 未完成",
                    "pad": pad,
                    "duration_ms": 0,
                }))
                continue
            try:
                cropped = _crop_from_atlas(segment_item.get("atlas_relpath") or "", segment_item.get("atlas_bbox") or {})
                padded = _pad_segmented_image(cropped, pad)
                payloads.append(encode_png_bytes(padded))
                valid_batch.append(task)
            except Exception as exc:
                results.append((task, {
                    "state": "error",
                    "text": None,
                    "confidence": None,
                    "matches": None,
                    "error": str(exc),
                    "pad": pad,
                    "duration_ms": 0,
                }))
        if not valid_batch:
            continue
        started = time.time()
        try:
            ocr_results = call_paddle_batch(payloads, paddle_url=paddle_url, timeout=timeout)
            duration_ms = int((time.time() - started) * 1000 / max(1, len(valid_batch)))
            for task, (text, confidence) in zip(valid_batch, ocr_results):
                char, _, _ = task
                results.append((task, {
                    "state": "ready",
                    "text": text,
                    "confidence": confidence,
                    "matches": _match_reocr_text(char, text),
                    "error": None,
                    "pad": pad,
                    "duration_ms": duration_ms,
                }))
        except Exception as exc:
            duration_ms = int((time.time() - started) * 1000)
            for task in valid_batch:
                results.append((task, {
                    "state": "error",
                    "text": None,
                    "confidence": None,
                    "matches": None,
                    "error": str(exc),
                    "pad": pad,
                    "duration_ms": duration_ms,
                }))
    return results


def _reocr_book(
    book_name: str,
    engine: str,
    chars: Optional[Iterable[str]] = None,
    limit_chars: Optional[int] = None,
    limit_instances: Optional[int] = None,
    force: bool = False,
    pad: int = 4,
    paddle_url: Optional[str] = None,
    timeout: int = 20,
    batch_size: int = 32,
) -> Dict:
    engine_name = normalize_engine_name(engine)
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
        book_name=book_name,
        segment_book=segment_book,
        reocr_book=reocr_book,
        chars=chars,
        limit_chars=limit_chars,
        limit_instances=limit_instances,
        force=force,
        pad=pad,
    )

    processed = 0
    skipped = max(0, total_candidates - len(tasks))
    errors = 0

    if engine_name == "paddle":
        resolved_paddle_url = paddle_url or PADDLE_CONFIG.get("url")
        if not resolved_paddle_url:
            raise ValueError("缺少 Paddle 服务地址")
        result_rows = _run_paddle_batch_for_tasks(
            tasks,
            paddle_url=resolved_paddle_url,
            timeout=timeout,
            batch_size=batch_size,
            pad=pad,
        )
    else:
        result_rows = []
        for task in tasks:
            char, _, segment_item = task
            if segment_item.get("state") != "ready":
                result_rows.append((task, {
                    "state": "error",
                    "text": None,
                    "confidence": None,
                    "matches": None,
                    "error": segment_item.get("error") or "segment 未完成",
                    "pad": pad,
                    "duration_ms": 0,
                }))
                continue
            started = time.time()
            try:
                cropped = _crop_from_atlas(segment_item.get("atlas_relpath") or "", segment_item.get("atlas_bbox") or {})
                padded = _pad_segmented_image(cropped, pad)
                text, confidence = _run_livetext_ocr(padded)
                result_rows.append((task, {
                    "state": "ready",
                    "text": text,
                    "confidence": confidence,
                    "matches": _match_reocr_text(char, text),
                    "error": None,
                    "pad": pad,
                    "duration_ms": int((time.time() - started) * 1000),
                }))
            except Exception as exc:
                result_rows.append((task, {
                    "state": "error",
                    "text": None,
                    "confidence": None,
                    "matches": None,
                    "error": str(exc),
                    "pad": pad,
                    "duration_ms": int((time.time() - started) * 1000),
                }))

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
        })
        processed += 1
        if result.get("state") == "error":
            errors += 1

    write_reocr_book(book_name, reocr_book, engine_name)
    return {"book": book_name, "processed": processed, "skipped": skipped, "errors": errors}


def _reocr_book_worker(args: Tuple[str, str, Optional[List[str]], Optional[int], Optional[int], bool, int, Optional[str], int, int]) -> Dict:
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
    )


def run_reocr_books(
    books: Optional[Iterable[str]] = None,
    engine: str = DEFAULT_REOCR_ENGINE,
    chars: Optional[Iterable[str]] = None,
    limit_chars: Optional[int] = None,
    limit_instances: Optional[int] = None,
    workers: int = 1,
    force: bool = False,
    pad: int = 4,
    paddle_url: Optional[str] = None,
    timeout: int = 20,
    batch_size: int = 32,
) -> int:
    engine_name = normalize_engine_name(engine)
    selected_books = list(books or [])
    if not selected_books:
        selected_books = sorted((review_config.SEGMENT_BOOKS_DIR.glob("*.json")), key=lambda p: p.name)
        selected_books = [p.stem for p in selected_books]
    if not selected_books:
        print("未找到 segment_books 数据")
        return 1

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
        )
        for book_name in selected_books
    ]

    total_processed = 0
    total_skipped = 0
    total_errors = 0

    print(f"流程: segment_books -> {engine_name} reOCR -> reocr_books")
    print(f"books={len(tasks)} workers={workers} force={int(bool(force))} pad={pad}")

    if int(workers or 1) <= 1:
        for task in tasks:
            result = _reocr_book_worker(task)
            total_processed += int(result.get("processed") or 0)
            total_skipped += int(result.get("skipped") or 0)
            total_errors += int(result.get("errors") or 0)
            print(
                f"[reocr:{engine_name}] {result['book']}: processed={result.get('processed', 0)} "
                f"skipped={result.get('skipped', 0)} errors={result.get('errors', 0)}"
            )
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=int(workers or 1)) as executor:
            futures = [executor.submit(_reocr_book_worker, task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                total_processed += int(result.get("processed") or 0)
                total_skipped += int(result.get("skipped") or 0)
                total_errors += int(result.get("errors") or 0)
                print(
                    f"[reocr:{engine_name}] {result['book']}: processed={result.get('processed', 0)} "
                    f"skipped={result.get('skipped', 0)} errors={result.get('errors', 0)}"
                )

    print(
        f"[reocr:{engine_name}] done: books={len(tasks)} processed={total_processed} "
        f"skipped={total_skipped} errors={total_errors}"
    )
    return 0 if total_errors == 0 else 1
