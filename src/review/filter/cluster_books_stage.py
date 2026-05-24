"""Cluster segmented instances into large/small groups and preselect filter candidates."""

from __future__ import annotations

import concurrent.futures
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm

from src.review import config as review_config
from src.review.storage.reocr_books import DEFAULT_REOCR_ENGINE, read_reocr_book
from src.review.storage.cluster_books import read_cluster_book, write_cluster_book
from src.review.storage.review_books import iter_confirmed_items, read_review_book
from src.review.storage.segment_books import list_segment_books, read_segment_book


FILTER_CONFIG = review_config.FILTER_CONFIG
TARGET_MATCHES_PER_CHAR = int(FILTER_CONFIG.get("target_matches_per_char", 15) or 15)
MIN_SPLIT_GAP_PX = int(FILTER_CONFIG.get("cluster_min_split_gap_px", 6) or 6)
MIN_SPLIT_RATIO = float(FILTER_CONFIG.get("cluster_min_split_ratio", 1.3) or 1.3)
MIN_GROUP_COUNT = int(FILTER_CONFIG.get("cluster_min_group_count", 2) or 2)
CONFIRMED_TOLERANCE_RATIO = float(FILTER_CONFIG.get("confirmed_width_tolerance_ratio", 0.25) or 0.25)
CONFIRMED_TOLERANCE_MIN_PX = int(FILTER_CONFIG.get("confirmed_width_tolerance_min_px", 6) or 6)


def _median_int(values: Sequence[int]) -> int:
    values = [int(v) for v in values if int(v) > 0]
    if not values:
        return 0
    return int(round(float(median(values))))


def _split_width_clusters(rows: Sequence[Tuple[str, int, int, int, int, int, int]]) -> Tuple[int, List[Tuple[str, int, int, int, int, int, int]], List[Tuple[str, int, int, int, int, int, int]]]:
    if len(rows) < 2:
        return 0, list(rows), []
    widths = [int(row[1]) for row in rows if int(row[1]) > 0]
    if len(widths) < 2:
        return 0, list(rows), []

    center_low = float(min(widths))
    center_high = float(max(widths))
    if abs(center_high - center_low) < 1e-6:
        return 0, list(rows), []

    lower_rows: List[Tuple[str, int, int, int, int, int, int]] = []
    upper_rows: List[Tuple[str, int, int, int, int, int, int]] = []
    for _ in range(32):
        lower_rows = []
        upper_rows = []
        for row in rows:
            width = float(row[1])
            if abs(width - center_low) <= abs(width - center_high):
                lower_rows.append(row)
            else:
                upper_rows.append(row)
        if not lower_rows or not upper_rows:
            return 0, list(rows), []
        next_low = sum(float(row[1]) for row in lower_rows) / len(lower_rows)
        next_high = sum(float(row[1]) for row in upper_rows) / len(upper_rows)
        if abs(next_low - center_low) < 1e-6 and abs(next_high - center_high) < 1e-6:
            center_low = next_low
            center_high = next_high
            break
        center_low = next_low
        center_high = next_high

    if center_low > center_high:
        center_low, center_high = center_high, center_low
        lower_rows, upper_rows = upper_rows, lower_rows

    threshold = int(round((center_low + center_high) / 2.0))
    lower_rows = [row for row in rows if row[1] <= threshold]
    upper_rows = [row for row in rows if row[1] > threshold]
    return threshold, lower_rows, upper_rows


def _sort_rows_for_target(
    rows: Sequence[Tuple[str, int, int, int, int, int, int]],
    *,
    confirmed_width: int,
) -> List[Tuple[str, int, int, int, int, int, int]]:
    if confirmed_width > 0:
        return sorted(
            rows,
            key=lambda row: (
                -int(row[5]),
                int(row[6]),
                abs(int(row[1]) - confirmed_width),
                -int(row[1]),
                -int(row[2]),
                -int(row[3]),
                -int(row[4]),
                row[0],
            ),
        )
    return sorted(
        rows,
        key=lambda row: (
            -int(row[5]),
            int(row[6]),
            -int(row[1]),
            -int(row[2]),
            -int(row[3]),
            -int(row[4]),
            row[0],
        ),
    )


def _cluster_char(
    char: str,
    segment_items: Dict[str, Dict],
    reocr_items: Dict[str, Dict],
    review_book: Dict,
    target_limit: int,
) -> Dict:
    confirmed_widths: List[int] = []
    review_char_entry = (review_book.get(char) or {}) if isinstance(review_book, dict) else {}
    confirmed_map = {
        instance_id: item
        for _char, instance_id, item in iter_confirmed_items({char: review_char_entry})
    }

    candidates: List[Tuple[str, int, int, int, int, int, int]] = []
    for instance_id, segment_item in (segment_items or {}).items():
        if not isinstance(segment_item, dict):
            continue
        if str(segment_item.get("state") or "pending") != "ready":
            continue
        ocr_width = int(((segment_item.get("source_bbox") or {}).get("width")) or 0)
        ocr_height = int(((segment_item.get("source_bbox") or {}).get("height")) or 0)
        segmented_width = int(segment_item.get("segmented_width") or 0)
        segmented_height = int(segment_item.get("segmented_height") or 0)
        if ocr_width <= 0 or ocr_height <= 0:
            continue
        reocr_item = (reocr_items or {}).get(instance_id) or {}
        if reocr_item.get("matches") is not True:
            continue
        match_rank = 1 if reocr_item.get("matches") is True else 0
        state_rank = 0
        reocr_state = str(reocr_item.get("state") or "pending")
        if reocr_state == "ready":
            state_rank = 1
        elif reocr_state == "error":
            state_rank = 2
        candidates.append((str(instance_id), ocr_width, ocr_height, segmented_width, segmented_height, match_rank, state_rank))
        if instance_id in confirmed_map:
            confirmed_widths.append(ocr_width)

    if not candidates:
        return {
            "updated_at": None,
            "mode": "single",
            "target_group": "single",
            "target_limit": target_limit,
            "selected_count": 0,
            "anchors": {
                "confirmed_widths": [],
                "confirmed_width": 0,
                "confirmed_tolerance": 0,
                "confirmed_group": "",
                "confirmed_filter_applied": False,
            },
            "clusters": {},
            "items": {},
        }

    candidates.sort(key=lambda row: (-row[5], int(row[6]), -row[1], -row[2], -row[3], -row[4], row[0]))
    confirmed_width = _median_int(confirmed_widths)

    mode = "single"
    target_group = "single"
    threshold = 0
    upper: List[Tuple[str, int, int, int, int, int, int]] = []
    lower: List[Tuple[str, int, int, int, int, int, int]] = []
    threshold, lower, upper = _split_width_clusters(candidates)
    if threshold > 0 and lower and upper:
        upper_med = _median_int([row[1] for row in upper])
        lower_med = _median_int([row[1] for row in lower])
        split_ratio = float(upper_med) / float(max(1, lower_med))
        split_gap = min(row[1] for row in upper) - max(row[1] for row in lower)
        if (
            len(upper) >= MIN_GROUP_COUNT
            and len(lower) >= MIN_GROUP_COUNT
            and upper_med > 0
            and lower_med > 0
            and split_ratio >= MIN_SPLIT_RATIO
        ):
            mode = "split"
            target_group = "large"
        else:
            threshold = 0
            lower = []
            upper = []

    if target_group == "large":
        target_rows = [row for row in candidates if row[1] > threshold]
        selected_rows = _sort_rows_for_target(
            target_rows,
            confirmed_width=confirmed_width if confirmed_width > threshold else 0,
        )[:target_limit]
        if not selected_rows:
            selected_rows = _sort_rows_for_target(
                candidates,
                confirmed_width=confirmed_width if confirmed_width > threshold else 0,
            )[:target_limit]
    else:
        selected_rows = _sort_rows_for_target(candidates, confirmed_width=confirmed_width)[:target_limit]

    confirmed_tolerance = (
        max(CONFIRMED_TOLERANCE_MIN_PX, int(round(confirmed_width * CONFIRMED_TOLERANCE_RATIO)))
        if confirmed_width > 0 else 0
    )
    confirmed_group = ""
    confirmed_filter_applied = False
    if confirmed_width > 0 and mode == "split":
        confirmed_group = "large" if confirmed_width > threshold else "small"
    elif confirmed_width > 0:
        confirmed_group = "single"

    if confirmed_width > 0 and (
        mode != "split"
        or target_group == "single"
        or confirmed_group == "large"
        or confirmed_group == "single"
    ):
        constrained_rows = [
            row for row in selected_rows
            if abs(int(row[1]) - confirmed_width) <= confirmed_tolerance
        ]
        selected_rows = constrained_rows
        confirmed_filter_applied = True

    selected_ids = {row[0] for row in selected_rows}
    items: Dict[str, Dict] = {}
    for rank, (instance_id, ocr_width, ocr_height, segmented_width, segmented_height, match_rank, state_rank) in enumerate(candidates, start=1):
        if mode == "split":
            size_group = "large" if ocr_width > threshold else "small"
        else:
            size_group = "single"
        items[instance_id] = {
            "selected": instance_id in selected_ids,
            "size_group": size_group,
            "size_rank": rank,
            "width": ocr_width,
            "height": ocr_height,
            "ocr_width": ocr_width,
            "ocr_height": ocr_height,
            "segmented_width": segmented_width,
            "segmented_height": segmented_height,
            "reocr_matches": bool(match_rank),
            "reocr_state_rank": int(state_rank),
            "confirmed_distance": float(abs(ocr_width - confirmed_width)) if confirmed_width > 0 else 0.0,
        }

    def summarize(group_name: str, rows: Sequence[Tuple[str, int, int, int, int, int, int]]) -> Dict:
        widths_local = [row[1] for row in rows]
        return {
            "count": len(rows),
            "min_width": min(widths_local) if widths_local else 0,
            "max_width": max(widths_local) if widths_local else 0,
            "median_width": _median_int(widths_local),
        }

    clusters: Dict[str, Dict] = {}
    if mode == "split":
        upper = [row for row in candidates if row[1] > threshold]
        lower = [row for row in candidates if row[1] <= threshold]
        clusters["large"] = summarize("large", upper)
        clusters["small"] = summarize("small", lower)
    else:
        clusters["single"] = summarize("single", candidates)

    return {
        "updated_at": None,
        "mode": mode,
        "target_group": target_group,
        "target_limit": target_limit,
        "selected_count": len(selected_ids),
        "anchors": {
            "confirmed_widths": confirmed_widths,
            "confirmed_width": confirmed_width,
            "confirmed_tolerance": confirmed_tolerance,
            "confirmed_group": confirmed_group,
            "confirmed_filter_applied": confirmed_filter_applied,
        },
        "clusters": clusters,
        "items": items,
    }


def _cluster_book(
    book_name: str,
    chars: Optional[Iterable[str]] = None,
    limit_chars: Optional[int] = None,
    limit_instances: Optional[int] = None,
    force: bool = False,
    target_limit: int = TARGET_MATCHES_PER_CHAR,
) -> Dict:
    segment_book = read_segment_book(book_name)
    if not segment_book:
        return {"book": book_name, "processed": 0, "skipped": 0, "errors": 1, "error": "segment book 不存在"}
    review_book = read_review_book(book_name) or {}
    reocr_book = read_reocr_book(book_name, DEFAULT_REOCR_ENGINE) or {}
    cluster_book = {} if force else (read_cluster_book(book_name) or {})

    selected_chars = set(str(ch) for ch in (chars or []))
    char_items = list((segment_book or {}).items())
    if selected_chars:
        char_items = [(char, entry) for char, entry in char_items if char in selected_chars]
    if limit_chars is not None:
        char_items = char_items[: max(0, int(limit_chars))]

    processed = 0
    skipped = 0
    errors = 0
    for char, char_entry in char_items:
        if not isinstance(char_entry, dict):
            continue
        if (not force) and isinstance(cluster_book.get(char), dict):
            skipped += 1
            continue
        try:
            segment_items = dict((char_entry.get("items") or {}))
            if limit_instances is not None:
                limited_items = list(segment_items.items())[: max(0, int(limit_instances))]
                segment_items = dict(limited_items)
            cluster_book[char] = _cluster_char(
                char=char,
                segment_items=segment_items,
                reocr_items=((reocr_book.get(char) or {}).get("items") or {}),
                review_book=review_book,
                target_limit=target_limit,
            )
            processed += 1
        except Exception as exc:
            cluster_book[char] = {
                "updated_at": None,
                "mode": "error",
                "target_group": "single",
                "target_limit": target_limit,
                "selected_count": 0,
                "anchors": {
                    "confirmed_widths": [],
                    "confirmed_width": 0,
                    "confirmed_tolerance": 0,
                    "confirmed_group": "",
                    "confirmed_filter_applied": False,
                },
                "clusters": {},
                "items": {},
                "error": str(exc),
            }
            errors += 1

    write_cluster_book(book_name, cluster_book)
    return {"book": book_name, "processed": processed, "skipped": skipped, "errors": errors}


def _cluster_book_worker(args: Tuple[str, Optional[List[str]], Optional[int], Optional[int], bool, int]) -> Dict:
    return _cluster_book(
        book_name=args[0],
        chars=args[1],
        limit_chars=args[2],
        limit_instances=args[3],
        force=args[4],
        target_limit=args[5],
    )


def run_cluster_books(
    books: Optional[Iterable[str]] = None,
    chars: Optional[Iterable[str]] = None,
    limit_chars: Optional[int] = None,
    limit_instances: Optional[int] = None,
    workers: int = 1,
    force: bool = False,
    target_limit: int = TARGET_MATCHES_PER_CHAR,
) -> int:
    selected_books = list(books or list_segment_books())
    if not selected_books:
        print("未找到 segment_books 数据")
        return 1

    tasks = [
        (
            book_name,
            list(chars) if chars else None,
            limit_chars,
            limit_instances,
            force,
            int(target_limit or TARGET_MATCHES_PER_CHAR),
        )
        for book_name in selected_books
    ]

    total_processed = 0
    total_skipped = 0
    total_errors = 0
    progress_desc = "cluster进度" if int(workers or 1) <= 1 else f"cluster进度 ({workers}进程)"

    print("流程: segment_books -> cluster_books")
    print(f"books={len(tasks)} workers={workers} force={int(bool(force))} target_limit={target_limit}")

    with tqdm(total=len(tasks), desc=progress_desc, unit="book", dynamic_ncols=True) as pbar:
        pbar.set_postfix({"processed": 0, "skipped": 0, "errors": 0}, refresh=False)
        if int(workers or 1) <= 1:
            for task in tasks:
                result = _cluster_book_worker(task)
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
                    f"[cluster] {result['book']}: processed={result.get('processed', 0)} "
                    f"skipped={result.get('skipped', 0)} errors={result.get('errors', 0)}"
                )
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=int(workers or 1)) as executor:
                future_to_book = {executor.submit(_cluster_book_worker, task): task[0] for task in tasks}
                for future in concurrent.futures.as_completed(future_to_book):
                    book_name = future_to_book[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {"book": book_name, "processed": 0, "skipped": 0, "errors": 1, "error": str(exc)}
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
                        f"[cluster] {result['book']}: processed={result.get('processed', 0)} "
                        f"skipped={result.get('skipped', 0)} errors={result.get('errors', 0)}"
                    )

    print(
        f"[cluster] done: books={len(tasks)} processed={total_processed} "
        f"skipped={total_skipped} errors={total_errors}"
    )
    return 0 if total_errors == 0 else 1
