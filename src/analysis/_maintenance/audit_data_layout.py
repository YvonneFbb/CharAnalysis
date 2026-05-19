#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Audit current data layout and integrity across matched/manual/analysis layers.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.review import config as review_config
from src.review.identity import get_confirmed_path
from src.review.storage.paddle_books import list_paddle_books, read_paddle_book
from src.review.storage.review_books import list_review_books, read_review_book


@dataclass
class AuditIssue:
    level: str
    code: str
    message: str


def _count_files(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for path in root.rglob("*") if path.is_file())


def _analysis_manifest() -> Dict | None:
    path = review_config.DATA_DIR / "analysis" / "manifest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def build_report() -> Dict:
    issues: List[AuditIssue] = []

    matched_books = sorted(review_config.MATCHED_BOOKS_DIR.glob("*.json"))
    matched_shards = sorted(review_config.MATCHED_SHARDS_DIR.glob("*.json"))
    review_books = list_review_books()
    paddle_books = list_paddle_books()
    manifest = _analysis_manifest()

    if review_config.MATCHED_JSON_PATH.exists() and not matched_books:
        issues.append(AuditIssue("warn", "matched_books_missing", "matched_by_book.json exists but matched_books/*.json is empty"))

    if matched_books and matched_shards:
        issues.append(AuditIssue("info", "matched_shards_redundant", "matched_books and matched_by_book_shards both exist; shards are cache-only"))

    review_chars = 0
    selected_instances = 0
    lookup_entries = 0
    confirmed_segments = 0
    confirmed_non_drop = 0
    missing_segment_images = 0
    chars_without_segments = 0

    for book_name in review_books:
        book = read_review_book(book_name) or {}
        for char_obj in book.values():
            if not isinstance(char_obj, dict):
                continue
            review_chars += 1
            items = char_obj.get("items") or {}
            selected_for_char = 0
            sourced_for_char = 0
            confirmed_for_char = 0
            for item in items.values():
                if not isinstance(item, dict):
                    continue
                filter_state = item.get("filter") or {}
                review_state = item.get("review") or {}
                if filter_state.get("status") == "accepted":
                    selected_instances += 1
                    selected_for_char += 1
                if item.get("source"):
                    lookup_entries += 1
                    sourced_for_char += 1
                if review_state.get("status") != "confirmed":
                    continue
                confirmed_segments += 1
                confirmed_for_char += 1
                if review_state.get("decision") != "drop":
                    confirmed_non_drop += 1
                seg_rel = get_confirmed_path(review_state)
                if seg_rel and not (PROJECT_ROOT / seg_rel).exists():
                    missing_segment_images += 1
            if selected_for_char == 0 and confirmed_for_char == 0:
                chars_without_segments += 1

    paddle_chars = 0
    paddle_items = 0
    paddle_pending = 0
    for book_name in paddle_books:
        book = read_paddle_book(book_name) or {}
        for payload in (book.get("chars") or {}).values():
            if not isinstance(payload, dict):
                continue
            paddle_chars += 1
            item_map = payload.get("items") or {}
            paddle_items += len(item_map)
            for item in item_map.values():
                if not isinstance(item, dict):
                    continue
                if item.get("decision", "pending") == "pending":
                    paddle_pending += 1

    analysis_entries = 0
    analysis_books = 0
    analysis_missing_images = None
    analysis_missing_lookup = None
    if manifest:
        analysis_books = len(manifest.get("books") or {})
        summary = manifest.get("summary") or {}
        if summary:
            analysis_entries = int(summary.get("total_entries") or 0)
            analysis_missing_images = summary.get("total_missing_images")
            analysis_missing_lookup = summary.get("total_missing_lookup")
        else:
            analysis_entries = sum((payload or {}).get("count", 0) for payload in (manifest.get("books") or {}).values())
            analysis_missing_images = sum((payload or {}).get("missing_images", 0) for payload in (manifest.get("books") or {}).values())
            analysis_missing_lookup = sum((payload or {}).get("missing_lookup", 0) for payload in (manifest.get("books") or {}).values())
        if analysis_books and analysis_books != len(review_books):
            issues.append(AuditIssue("warn", "analysis_book_count_mismatch", f"analysis has {analysis_books} books, review_books has {len(review_books)}"))
        if analysis_entries and analysis_entries != confirmed_non_drop - missing_segment_images:
            issues.append(
                AuditIssue(
                    "warn",
                    "analysis_entry_count_mismatch",
                    f"analysis has {analysis_entries} entries, expected {confirmed_non_drop - missing_segment_images} from review_books",
                )
            )
    else:
        issues.append(AuditIssue("warn", "analysis_manifest_missing", "analysis/manifest.json is missing or unreadable"))

    if lookup_entries < selected_instances:
        issues.append(AuditIssue("warn", "lookup_gap", f"source-backed items ({lookup_entries}) < accepted filter items ({selected_instances})"))

    if missing_segment_images:
        issues.append(AuditIssue("warn", "missing_segment_images", f"{missing_segment_images} confirmed segment images are missing on disk"))

    report = {
        "paths": {
            "matched_json": str(review_config.MATCHED_JSON_PATH.relative_to(PROJECT_ROOT)),
            "matched_books_dir": str(review_config.MATCHED_BOOKS_DIR.relative_to(PROJECT_ROOT)),
            "matched_shards_dir": str(review_config.MATCHED_SHARDS_DIR.relative_to(PROJECT_ROOT)),
            "review_books_dir": str(review_config.REVIEW_BOOKS_DIR.relative_to(PROJECT_ROOT)),
            "analysis_dir": "data/analysis",
        },
        "inventory": {
            "raw_files": _count_files(review_config.RAW_DIR),
            "preprocessed_files": _count_files(review_config.PREPROCESSED_DIR),
            "ocr_files": _count_files(review_config.OCR_DIR),
            "matched_books_files": len(matched_books),
            "matched_shards_files": len(matched_shards),
            "review_books_files": len(review_books),
            "paddle_books_files": len(paddle_books),
            "analysis_entry_files": _count_files(review_config.DATA_DIR / "analysis" / "books"),
        },
        "review": {
            "book_count": len(review_books),
            "char_count": review_chars,
            "selected_instances": selected_instances,
            "lookup_entries": lookup_entries,
            "confirmed_segments": confirmed_segments,
            "confirmed_non_drop_segments": confirmed_non_drop,
            "chars_without_segments": chars_without_segments,
            "missing_segment_images": missing_segment_images,
        },
        "paddle": {
            "book_count": len(paddle_books),
            "char_count": paddle_chars,
            "item_count": paddle_items,
            "pending_count": paddle_pending,
        },
        "analysis": {
            "manifest_present": bool(manifest),
            "manifest_version": (manifest or {}).get("version"),
            "generated_at": (manifest or {}).get("generated_at"),
            "book_count": analysis_books,
            "entry_count": analysis_entries,
            "missing_images": analysis_missing_images,
            "missing_lookup": analysis_missing_lookup,
        },
        "issues": [asdict(issue) for issue in issues],
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit current data layout and integrity.")
    ap.add_argument("--json", action="store_true", help="Emit JSON report")
    ap.add_argument("--strict", action="store_true", help="Exit non-zero if any warn-level issue exists")
    args = ap.parse_args()

    report = build_report()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("=== Data Layout Audit ===")
        print(f"review_books: {report['review']['book_count']} books, {report['review']['char_count']} chars")
        print(f"selected_instances: {report['review']['selected_instances']}, lookup_entries: {report['review']['lookup_entries']}")
        print(f"confirmed_non_drop_segments: {report['review']['confirmed_non_drop_segments']}")
        print(f"matched_books_files: {report['inventory']['matched_books_files']}, matched_shards_files: {report['inventory']['matched_shards_files']}")
        print(f"analysis: version={report['analysis']['manifest_version']} books={report['analysis']['book_count']} entries={report['analysis']['entry_count']}")
        if report["issues"]:
            print("issues:")
            for issue in report["issues"]:
                print(f"  [{issue['level']}] {issue['code']}: {issue['message']}")
        else:
            print("issues: none")

    if args.strict and any(issue["level"] == "warn" for issue in report["issues"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
