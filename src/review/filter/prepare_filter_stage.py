"""High-level filter preparation workflow.

Current pipeline:
matched -> segment_books/atlas -> reocr_books(engine) -> cluster_books
"""

from __future__ import annotations

from typing import Iterable, Optional

from src.review.filter.cluster_books_stage import run_cluster_books
from src.review.filter.reocr_books_stage import run_reocr_books
from src.review.filter.segment_books_stage import run_segment_books
from src.review.storage.reocr_books import DEFAULT_REOCR_ENGINE
from src.review import config as review_config


DEFAULT_REOCR_PAD = int(review_config.PADDLE_CONFIG.get("reocr_pad", 12) or 12)
DEFAULT_TARGET_LIMIT = int(review_config.FILTER_CONFIG.get("target_matches_per_char", 15) or 15)


def run_prepare_filter(
    books: Optional[Iterable[str]] = None,
    chars: Optional[Iterable[str]] = None,
    limit_chars: Optional[int] = None,
    limit_instances: Optional[int] = None,
    workers: int = 1,
    force: bool = False,
    reocr_engine: str = DEFAULT_REOCR_ENGINE,
    reocr_pad: int = DEFAULT_REOCR_PAD,
    paddle_url: Optional[str] = None,
    timeout: int = 20,
    batch_size: int = 32,
    tmp_dir: Optional[str] = None,
    sheet_max_slots: Optional[int] = None,
    target_limit: int = DEFAULT_TARGET_LIMIT,
    target_matches: int = 0,
) -> int:
    print("prepare-filter: matched -> segment -> reocr -> cluster")

    rc = run_segment_books(
        books=books,
        chars=chars,
        limit_chars=limit_chars,
        limit_instances=limit_instances,
        workers=workers,
        force=force,
    )
    if rc != 0:
        return rc

    rc = run_reocr_books(
        books=books,
        engine=reocr_engine,
        chars=chars,
        limit_chars=limit_chars,
        limit_instances=limit_instances,
        workers=workers,
        force=force,
        pad=reocr_pad,
        paddle_url=paddle_url,
        timeout=timeout,
        batch_size=batch_size,
        tmp_dir=tmp_dir,
        sheet_max_slots=sheet_max_slots,
        target_matches=target_matches,
    )
    if rc != 0:
        return rc

    return run_cluster_books(
        books=books,
        chars=chars,
        limit_chars=limit_chars,
        limit_instances=limit_instances,
        workers=workers,
        force=force,
        target_limit=target_limit,
    )
