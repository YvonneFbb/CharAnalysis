#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将 review_books 中挂在旧 instance_id 上的 confirmed 记录迁移到当前 canonical instance。

只处理安全场景：
- 同 book
- 同 char
- 同 source_image
- 同 bbox（x, y, width, height 全等）

迁移时：
- 保留 confirmed_path / review.method / review.timestamp / decision
- 使用当前 canonical source 覆盖 source 元信息
- 不改最终 confirmed 图片文件
- 若目标 instance 已存在，仅在无 confirmed 冲突时合并
"""

from __future__ import annotations

import argparse
import cv2
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.review.identity import get_confirmed_path, normalize_to_preprocessed_path, set_confirmed_path
from src.review.matched_dedupe import matched_bboxes_nearly_overlap
from src.review.storage.review_books import (
    list_review_books,
    maybe_backup_review_book,
    read_review_book,
    review_book_path,
    source_from_matched_instance,
    write_review_book,
)


MATCHED_BOOKS_DIR = PROJECT_ROOT / "data/results/matched_books"
SEGMENT_BOOKS_DIR = PROJECT_ROOT / "data/results/segment_books"


def _load_matched_book(book_name: str) -> Optional[Dict]:
    path = MATCHED_BOOKS_DIR / f"{book_name}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload.get("data"), dict):
        return payload.get("data")
    if isinstance(payload.get("chars"), dict):
        return payload
    return None


def _load_segment_book(book_name: str) -> Optional[Dict]:
    path = SEGMENT_BOOKS_DIR / f"{book_name}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload.get("chars"), dict):
        return payload
    return None


def _bbox_key(source: Optional[Dict]) -> Tuple[str, int, int, int, int]:
    source = dict(source or {})
    bbox = dict(source.get("bbox") or {})
    image = normalize_to_preprocessed_path(source.get("source_image", ""))
    return (
        image,
        int(bbox.get("x") or 0),
        int(bbox.get("y") or 0),
        int(bbox.get("width") or 0),
        int(bbox.get("height") or 0),
    )


def _matched_source_map(book_name: str) -> Dict[str, Dict[Tuple[str, int, int, int, int], Tuple[str, Dict]]]:
    matched_book = _load_matched_book(book_name) or {}
    out: Dict[str, Dict[Tuple[str, int, int, int, int], Tuple[str, Dict]]] = {}
    for char, instances in (matched_book.get("chars") or {}).items():
        if not isinstance(instances, list):
            continue
        char_map: Dict[Tuple[str, int, int, int, int], Tuple[str, Dict]] = {}
        for idx, inst in enumerate(instances):
            if not isinstance(inst, dict):
                continue
            src = source_from_matched_instance(inst, index=idx)
            key = _bbox_key(src)
            char_map[key] = (str(src.get("instance_id") or ""), src)
        if char_map:
            out[str(char)] = char_map
    return out


def _page_key(source: Optional[Dict]) -> Tuple[str, int, str]:
    source = dict(source or {})
    return (
        normalize_to_preprocessed_path(source.get("source_image", "")),
        int(source.get("volume") or 0),
        str(source.get("page") or ""),
    )


def _matched_page_map(book_name: str) -> Dict[str, Dict[Tuple[str, int, str], List[Tuple[str, Dict]]]]:
    matched_book = _load_matched_book(book_name) or {}
    out: Dict[str, Dict[Tuple[str, int, str], List[Tuple[str, Dict]]]] = {}
    for char, instances in (matched_book.get("chars") or {}).items():
        if not isinstance(instances, list):
            continue
        char_map: Dict[Tuple[str, int, str], List[Tuple[str, Dict]]] = {}
        for idx, inst in enumerate(instances):
            if not isinstance(inst, dict):
                continue
            src = source_from_matched_instance(inst, index=idx)
            page_key = _page_key(src)
            char_map.setdefault(page_key, []).append((str(src.get("instance_id") or ""), src))
        if char_map:
            out[str(char)] = char_map
    return out


def _review_status(item: Optional[Dict]) -> str:
    review = dict((item or {}).get("review") or {})
    return str(review.get("status") or "pending")


def _filter_status(item: Optional[Dict]) -> str:
    filt = dict((item or {}).get("filter") or {})
    return str(filt.get("status") or "pending")


def _clone_item_with_source(item: Dict, target_source: Dict) -> Dict:
    cloned = json.loads(json.dumps(item, ensure_ascii=False))
    cloned["source"] = dict(target_source or {})
    return cloned


def _expected_confirmed_relpath(book_name: str, char: str, instance_id: str) -> str:
    return f"data/results/manual/confirmed/{book_name}/{char}_{instance_id}.png"


def _restore_confirmed_from_segment(book_name: str, char: str, instance_id: str, dst_abs: Path) -> bool:
    segment_book = _load_segment_book(book_name) or {}
    entry = (((segment_book.get("chars") or {}).get(char) or {}).get("items") or {}).get(instance_id)
    if not isinstance(entry, dict):
        return False
    atlas_relpath = str(entry.get("atlas_relpath") or "")
    atlas_bbox = dict(entry.get("atlas_bbox") or {})
    if not atlas_relpath or not atlas_bbox:
        return False
    atlas_abs = PROJECT_ROOT / atlas_relpath
    if not atlas_abs.exists():
        return False
    x = int(atlas_bbox.get("x") or 0)
    y = int(atlas_bbox.get("y") or 0)
    width = int(atlas_bbox.get("width") or 0)
    height = int(atlas_bbox.get("height") or 0)
    if width <= 0 or height <= 0:
        return False

    atlas_img = cv2.imread(str(atlas_abs), cv2.IMREAD_UNCHANGED)
    if atlas_img is None:
        return False
    crop = atlas_img[y:y + height, x:x + width]
    if crop.size == 0:
        return False
    dst_abs.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(dst_abs), crop))


def _normalize_confirmed_file(
    book_name: str,
    char: str,
    instance_id: str,
    item: Dict,
    dry_run: bool,
) -> Tuple[bool, Counter]:
    stats = Counter()
    review = dict((item or {}).get("review") or {})
    confirmed_rel = get_confirmed_path(review)
    if not confirmed_rel:
        return False, stats

    expected_rel = _expected_confirmed_relpath(book_name, char, instance_id)
    expected_abs = PROJECT_ROOT / expected_rel
    current_abs = PROJECT_ROOT / confirmed_rel

    if confirmed_rel == expected_rel and expected_abs.exists():
        return False, stats

    changed = False
    can_update_path = False

    if current_abs.exists() and current_abs.resolve() != expected_abs.resolve():
        if expected_abs.exists():
            if not dry_run:
                expected_abs.unlink()
            stats["confirmed_path_overwrite_existing"] += 1
        if not dry_run:
            expected_abs.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(current_abs), str(expected_abs))
        stats["confirmed_path_renamed"] += 1
        changed = True
        can_update_path = True
    elif expected_abs.exists():
        stats["confirmed_path_relinked_existing"] += 1
        changed = True
        can_update_path = True
    elif str(review.get("method") or "") == "auto":
        if not dry_run:
            restored = _restore_confirmed_from_segment(book_name, char, instance_id, expected_abs)
        else:
            restored = True
        if restored:
            stats["confirmed_path_restored_from_segment"] += 1
            changed = True
            can_update_path = True
        else:
            stats["confirmed_path_missing_unresolved"] += 1
            return changed, stats
    else:
        stats["confirmed_path_missing_unresolved"] += 1
        return changed, stats

    if can_update_path:
        item["review"] = set_confirmed_path(review, expected_rel)
    return changed, stats


def _resolve_target(
    instance_id: str,
    source: Optional[Dict],
    char_match_map: Dict[Tuple[str, int, int, int, int], Tuple[str, Dict]],
    char_page_map: Dict[Tuple[str, int, str], List[Tuple[str, Dict]]],
) -> Tuple[str, Optional[str], Optional[Dict]]:
    source = dict(source or {})
    key = _bbox_key(source)
    target = char_match_map.get(key)
    if target:
        target_instance_id, target_source = target
        if target_instance_id == instance_id:
            return ("already_current", target_instance_id, target_source)
        return ("exact_match", target_instance_id, target_source)

    page_candidates = char_page_map.get(_page_key(source)) or []
    probe = {
        "source_image": source.get("source_image"),
        "volume": source.get("volume"),
        "page": source.get("page"),
        "bbox": dict(source.get("bbox") or {}),
    }
    overlap_candidates = [
        (candidate_id, candidate_source)
        for candidate_id, candidate_source in page_candidates
        if matched_bboxes_nearly_overlap(probe, candidate_source)
    ]
    if len(overlap_candidates) == 1:
        target_instance_id, target_source = overlap_candidates[0]
        if target_instance_id == instance_id:
            return ("refresh_current_source", target_instance_id, target_source)
        return ("near_overlap_unique", target_instance_id, target_source)
    if len(overlap_candidates) > 1:
        return ("unresolved_overlap_ambiguous", None, None)
    if page_candidates:
        return ("unresolved_same_page_no_overlap", None, None)
    return ("unresolved_page_missing", None, None)


def migrate_book(book_name: str, dry_run: bool = False) -> Dict[str, int]:
    book_data = read_review_book(book_name) or {}
    matched_map = _matched_source_map(book_name)
    matched_page_map = _matched_page_map(book_name)
    stats = Counter()
    changed = False

    for char, char_entry in book_data.items():
        if not isinstance(char_entry, dict):
            continue
        items = char_entry.get("items") or {}
        if not isinstance(items, dict) or not items:
            continue
        char_match_map = matched_map.get(char) or {}
        char_page_map = matched_page_map.get(char) or {}
        if not char_match_map and not char_page_map:
            continue

        migrations = []
        for instance_id, item in list(items.items()):
            if not isinstance(item, dict):
                continue
            review = dict(item.get("review") or {})
            if review.get("status") != "confirmed" or review.get("decision") == "drop":
                continue
            confirmed_path = get_confirmed_path(review)
            if not confirmed_path:
                continue
            source = dict(item.get("source") or {})
            resolution, target_instance_id, target_source = _resolve_target(
                instance_id=instance_id,
                source=source,
                char_match_map=char_match_map,
                char_page_map=char_page_map,
            )
            if resolution.startswith("unresolved_"):
                stats[resolution] += 1
                continue
            if not target_instance_id:
                stats["unresolved_missing_target"] += 1
                continue
            if resolution == "already_current":
                stats["already_current"] += 1
                continue
            if resolution == "refresh_current_source":
                replacement = _clone_item_with_source(item, target_source or {})
                replacement_filter = dict(replacement.get("filter") or {})
                replacement_filter["status"] = "accepted"
                replacement["filter"] = replacement_filter
                items[instance_id] = replacement
                stats["refreshed_current_source"] += 1
                changed = True
                continue
            migrations.append((instance_id, target_instance_id, target_source, resolution))

        for old_instance_id, new_instance_id, target_source, resolution in migrations:
            old_item = items.get(old_instance_id)
            if not isinstance(old_item, dict):
                continue
            target_item = items.get(new_instance_id)
            if isinstance(target_item, dict):
                old_confirmed = get_confirmed_path((old_item.get("review") or {}))
                target_confirmed = get_confirmed_path((target_item.get("review") or {}))
                if target_confirmed and old_confirmed and target_confirmed != old_confirmed:
                    stats["conflict_both_confirmed"] += 1
                    continue
                replacement = _clone_item_with_source(old_item, target_source)
                if target_confirmed and not old_confirmed:
                    replacement["review"] = dict(target_item.get("review") or {})
                replacement_filter = dict(replacement.get("filter") or {})
                replacement_filter["status"] = "accepted"
                replacement["filter"] = replacement_filter
                items[new_instance_id] = replacement
                items.pop(old_instance_id, None)
                stats["migrated_merge_existing"] += 1
                stats[f"migrated_{resolution}"] += 1
                changed = True
                continue

            items[new_instance_id] = _clone_item_with_source(old_item, target_source)
            items.pop(old_instance_id, None)
            stats["migrated_new_instance"] += 1
            stats[f"migrated_{resolution}"] += 1
            changed = True

        char_entry["items"] = items

        for current_instance_id, current_item in list(items.items()):
            if not isinstance(current_item, dict):
                continue
            review = dict(current_item.get("review") or {})
            if review.get("status") != "confirmed" or review.get("decision") == "drop":
                continue
            if not get_confirmed_path(review):
                continue
            normalized_changed, normalize_stats = _normalize_confirmed_file(
                book_name=book_name,
                char=char,
                instance_id=current_instance_id,
                item=current_item,
                dry_run=dry_run,
            )
            if normalize_stats:
                stats.update(normalize_stats)
            if normalized_changed:
                items[current_instance_id] = current_item
                changed = True

        book_data[char] = char_entry

    if changed and not dry_run:
        book_path = review_book_path(book_name)
        maybe_backup_review_book(book_name, book_path)
        write_review_book(book_name, book_data, skip_backup=True)
        stats["changed_books"] += 1
    elif changed:
        stats["changed_books"] += 1
    return dict(stats)


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate confirmed review records to current canonical instance ids.")
    parser.add_argument("--books", nargs="+", help="Only process selected books")
    parser.add_argument("--dry-run", action="store_true", help="Scan only, do not write files")
    args = parser.parse_args()

    books = args.books or list_review_books()
    if not books:
        print("未找到 review_books 数据。")
        return

    totals = Counter()
    for book_name in books:
        stats = Counter(migrate_book(book_name, dry_run=args.dry_run))
        if sum(stats.values()) == 0:
            continue
        totals.update(stats)
        print(
            f"{book_name}: "
            f"already_current={stats.get('already_current', 0)} "
            f"refreshed={stats.get('refreshed_current_source', 0)} "
            f"migrated_new={stats.get('migrated_new_instance', 0)} "
            f"migrated_merge={stats.get('migrated_merge_existing', 0)} "
            f"migrated_overlap={stats.get('migrated_near_overlap_unique', 0)} "
            f"conflict={stats.get('conflict_both_confirmed', 0)} "
            f"unresolved={stats.get('unresolved_missing_target', 0) + stats.get('unresolved_overlap_ambiguous', 0) + stats.get('unresolved_same_page_no_overlap', 0) + stats.get('unresolved_page_missing', 0)}"
        )

    mode = "dry-run" if args.dry_run else "done"
    print(
        f"[{mode}] books={len(books)} "
        f"changed_books={totals.get('changed_books', 0)} "
        f"already_current={totals.get('already_current', 0)} "
        f"refreshed={totals.get('refreshed_current_source', 0)} "
        f"migrated_new={totals.get('migrated_new_instance', 0)} "
        f"migrated_merge={totals.get('migrated_merge_existing', 0)} "
        f"migrated_overlap={totals.get('migrated_near_overlap_unique', 0)} "
        f"conflict={totals.get('conflict_both_confirmed', 0)} "
        f"unresolved={totals.get('unresolved_missing_target', 0) + totals.get('unresolved_overlap_ambiguous', 0) + totals.get('unresolved_same_page_no_overlap', 0) + totals.get('unresolved_page_missing', 0)}"
    )


if __name__ == "__main__":
    main()
