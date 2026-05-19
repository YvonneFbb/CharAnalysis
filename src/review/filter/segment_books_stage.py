"""Build segmented atlas shards from matched OCR instances."""

from __future__ import annotations

import concurrent.futures
import json
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from src.review import config as review_config
from src.review.identity import make_instance_id
from src.review.matched_dedupe import dedupe_matched_book_data
from src.review.segment import segment_character_from_image
from src.review.storage.review_books import source_from_matched_instance, utc_now_iso
from src.review.storage.segment_books import (
    ensure_segment_item,
    read_segment_book,
    segment_atlas_relpath,
    segment_book_atlas_dir,
    write_segment_book,
)


PROJECT_ROOT = review_config.PROJECT_ROOT
MATCHED_BOOKS_DIR = review_config.MATCHED_BOOKS_DIR
MATCHED_JSON_PATH = review_config.MATCHED_JSON_PATH
ATLAS_MAX_WIDTH = 4096
ATLAS_MAX_HEIGHT = 4096
ATLAS_PADDING = 4
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
        payload = json.loads(MATCHED_JSON_PATH.read_text(encoding="utf-8"))
        books = payload.get("books")
        if isinstance(books, dict):
            return sorted(books.keys())
    return []


def read_matched_book(book_name: str) -> Optional[Dict]:
    shard_path = MATCHED_BOOKS_DIR / f"{book_name}.json"
    if shard_path.exists():
        payload = json.loads(shard_path.read_text(encoding="utf-8"))
        return dedupe_matched_book_data(_extract_book_payload(payload, book_name))
    if MATCHED_JSON_PATH.exists():
        payload = json.loads(MATCHED_JSON_PATH.read_text(encoding="utf-8"))
        return dedupe_matched_book_data(_extract_book_payload(payload, book_name))
    return None


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


class AtlasWriter:
    def __init__(self, book_name: str) -> None:
        self.book_name = book_name
        self.atlas_dir = segment_book_atlas_dir(book_name)
        self.atlas_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(self.atlas_dir.glob("atlas_*.png"))
        if existing:
            existing_indices = [
                int(path.stem.split("_")[-1])
                for path in existing
                if path.stem.split("_")[-1].isdigit()
            ]
            self.page_index = (max(existing_indices) + 1) if existing_indices else len(existing)
        else:
            self.page_index = 0
        self.canvas: Optional[np.ndarray] = None
        self.row_x = 0
        self.row_y = 0
        self.row_height = 0
        self.used_width = 0
        self.used_height = 0
        self.current_filename: Optional[str] = None
        self.dirty = False
        self._new_page()

    def _new_page(self) -> None:
        self.current_filename = f"atlas_{self.page_index:04d}.png"
        self.canvas = np.full((ATLAS_MAX_HEIGHT, ATLAS_MAX_WIDTH, 3), 255, dtype=np.uint8)
        self.row_x = ATLAS_PADDING
        self.row_y = ATLAS_PADDING
        self.row_height = 0
        self.used_width = 0
        self.used_height = 0
        self.dirty = False

    def _flush_page(self) -> None:
        if not self.dirty or self.canvas is None or self.current_filename is None:
            return
        out_path = self.atlas_dir / self.current_filename
        used_w = max(1, min(ATLAS_MAX_WIDTH, self.used_width + ATLAS_PADDING))
        used_h = max(1, min(ATLAS_MAX_HEIGHT, self.used_height + ATLAS_PADDING))
        cropped = self.canvas[:used_h, :used_w]
        if not cv2.imwrite(str(out_path), cropped):
            raise RuntimeError(f"无法写入 atlas: {out_path}")
        self.page_index += 1
        self._new_page()

    def add(self, image_bgr: np.ndarray) -> Tuple[str, Dict]:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("segment 结果为空")
        if self.canvas is None or self.current_filename is None:
            self._new_page()

        height, width = image_bgr.shape[:2]
        if width + ATLAS_PADDING * 2 > ATLAS_MAX_WIDTH or height + ATLAS_PADDING * 2 > ATLAS_MAX_HEIGHT:
            raise ValueError(f"segment 结果过大，无法写入 atlas: {width}x{height}")

        if self.row_x + width + ATLAS_PADDING > ATLAS_MAX_WIDTH:
            self.row_x = ATLAS_PADDING
            self.row_y += self.row_height + ATLAS_PADDING
            self.row_height = 0

        if self.row_y + height + ATLAS_PADDING > ATLAS_MAX_HEIGHT:
            self._flush_page()

        assert self.canvas is not None
        assert self.current_filename is not None

        x = self.row_x
        y = self.row_y
        self.canvas[y:y + height, x:x + width] = image_bgr
        self.row_x += width + ATLAS_PADDING
        self.row_height = max(self.row_height, height)
        self.used_width = max(self.used_width, x + width)
        self.used_height = max(self.used_height, y + height)
        self.dirty = True
        return (
            segment_atlas_relpath(self.book_name, self.current_filename),
            {"x": x, "y": y, "width": width, "height": height},
        )

    def close(self) -> None:
        self._flush_page()


def _segment_one_instance(source: Dict) -> Tuple[np.ndarray, Dict]:
    preprocessed_abs = PROJECT_ROOT / str(source.get("source_image") or "")
    image = _read_cached_source_image(preprocessed_abs)
    _, segmented_img, _, metadata, _ = segment_character_from_image(
        image,
        source.get("bbox") or {},
    )
    return segmented_img, metadata


def _segment_book(
    book_name: str,
    chars: Optional[Iterable[str]] = None,
    limit_chars: Optional[int] = None,
    limit_instances: Optional[int] = None,
    force: bool = False,
) -> Dict:
    matched_book = read_matched_book(book_name)
    if not matched_book:
        return {"book": book_name, "processed": 0, "skipped": 0, "errors": 1, "error": "matched book 不存在"}

    if force:
        shutil.rmtree(segment_book_atlas_dir(book_name), ignore_errors=True)
        segment_book: Dict = {}
    else:
        segment_book = read_segment_book(book_name) or {}

    selected_chars = set(str(ch) for ch in (chars or []))
    matched_chars = matched_book.get("chars") or {}
    writer = AtlasWriter(book_name)
    processed = 0
    skipped = 0
    errors = 0

    char_items = list(matched_chars.items())
    if selected_chars:
        char_items = [(char, instances) for char, instances in char_items if char in selected_chars]
    if limit_chars is not None:
        char_items = char_items[: max(0, int(limit_chars))]

    for char, instances in char_items:
        if not isinstance(instances, list):
            continue
        iter_instances = list(enumerate(instances))
        if limit_instances is not None:
            iter_instances = iter_instances[: max(0, int(limit_instances))]

        for inst_idx, inst in iter_instances:
            if not isinstance(inst, dict):
                continue
            source = source_from_matched_instance(inst, index=inst_idx)
            instance_id = source.get("instance_id") or make_instance_id(inst)
            item = ensure_segment_item(segment_book, char, instance_id)
            if not force and item.get("state") == "ready":
                atlas_relpath = str(item.get("atlas_relpath") or "")
                atlas_bbox = item.get("atlas_bbox") or {}
                if (
                    atlas_relpath
                    and int(atlas_bbox.get("width") or 0) > 0
                    and int(atlas_bbox.get("height") or 0) > 0
                    and (PROJECT_ROOT / atlas_relpath).exists()
                ):
                    skipped += 1
                    continue
            try:
                segmented_img, metadata = _segment_one_instance(source)
                atlas_relpath, atlas_bbox = writer.add(segmented_img)
                item.update({
                    "state": "ready",
                    "timestamp": utc_now_iso(),
                    "error": None,
                    "atlas_relpath": atlas_relpath,
                    "atlas_bbox": atlas_bbox,
                    "source_bbox": source.get("bbox") or {},
                    "roi_bbox": metadata.get("roi_bbox") or {},
                    "segmented_bbox": metadata.get("segmented_bbox") or {},
                    "segmented_bbox_absolute": metadata.get("segmented_bbox_absolute") or {},
                    "segmented_width": int(segmented_img.shape[1]),
                    "segmented_height": int(segmented_img.shape[0]),
                    "roi_width": int((metadata.get("roi_bbox") or {}).get("width") or 0),
                    "roi_height": int((metadata.get("roi_bbox") or {}).get("height") or 0),
                    "metrics": {
                        "width_ratio": float(segmented_img.shape[1]) / float(max(1, int(source.get("width") or 0))),
                        "height_ratio": float(segmented_img.shape[0]) / float(max(1, int(source.get("height") or 0))),
                    },
                })
                processed += 1
            except Exception as exc:
                item.update({
                    "state": "error",
                    "timestamp": utc_now_iso(),
                    "error": str(exc),
                    "atlas_relpath": None,
                    "atlas_bbox": {"x": 0, "y": 0, "width": 0, "height": 0},
                    "source_bbox": source.get("bbox") or {},
                    "roi_bbox": {"x": 0, "y": 0, "width": 0, "height": 0},
                    "segmented_bbox": {"x": 0, "y": 0, "width": 0, "height": 0},
                    "segmented_bbox_absolute": {"x": 0, "y": 0, "width": 0, "height": 0},
                    "segmented_width": 0,
                    "segmented_height": 0,
                    "roi_width": 0,
                    "roi_height": 0,
                    "metrics": {"width_ratio": 0.0, "height_ratio": 0.0},
                })
                errors += 1

    writer.close()
    write_segment_book(book_name, segment_book)
    return {"book": book_name, "processed": processed, "skipped": skipped, "errors": errors}


def _segment_book_worker(args: Tuple[str, Optional[List[str]], Optional[int], Optional[int], bool]) -> Dict:
    book_name, chars, limit_chars, limit_instances, force = args
    return _segment_book(
        book_name=book_name,
        chars=chars,
        limit_chars=limit_chars,
        limit_instances=limit_instances,
        force=force,
    )


def run_segment_books(
    books: Optional[Iterable[str]] = None,
    chars: Optional[Iterable[str]] = None,
    limit_chars: Optional[int] = None,
    limit_instances: Optional[int] = None,
    workers: int = 1,
    force: bool = False,
) -> int:
    selected_books = list(books or list_matched_books())
    if not selected_books:
        print("未找到 matched_books 数据")
        return 1

    tasks = [
        (book_name, list(chars) if chars else None, limit_chars, limit_instances, force)
        for book_name in selected_books
    ]

    total_processed = 0
    total_skipped = 0
    total_errors = 0

    print("流程: matched -> segment -> segment_books + atlas")
    print(f"books={len(tasks)} workers={workers} force={int(bool(force))}")

    if int(workers or 1) <= 1:
        for task in tasks:
            result = _segment_book_worker(task)
            total_processed += int(result.get("processed") or 0)
            total_skipped += int(result.get("skipped") or 0)
            total_errors += int(result.get("errors") or 0)
            print(
                f"[segment] {result['book']}: processed={result.get('processed', 0)} "
                f"skipped={result.get('skipped', 0)} errors={result.get('errors', 0)}"
            )
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=int(workers or 1)) as executor:
            futures = [executor.submit(_segment_book_worker, task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                total_processed += int(result.get("processed") or 0)
                total_skipped += int(result.get("skipped") or 0)
                total_errors += int(result.get("errors") or 0)
                print(
                    f"[segment] {result['book']}: processed={result.get('processed', 0)} "
                    f"skipped={result.get('skipped', 0)} errors={result.get('errors', 0)}"
                )

    print(
        f"[segment] done: books={len(tasks)} processed={total_processed} "
        f"skipped={total_skipped} errors={total_errors}"
    )
    return 0 if total_errors == 0 else 1
