"""High-level filter preparation workflow.

Current pipeline:
matched -> segment_books/atlas -> reocr_books(engine)
"""

from __future__ import annotations

from typing import Iterable, Optional

from src.review.filter.reocr_books_stage import run_reocr_books
from src.review.filter.segment_books_stage import run_segment_books
from src.review.storage.reocr_books import DEFAULT_REOCR_ENGINE


def run_prepare_filter(
    books: Optional[Iterable[str]] = None,
    chars: Optional[Iterable[str]] = None,
    limit_chars: Optional[int] = None,
    limit_instances: Optional[int] = None,
    workers: int = 1,
    force: bool = False,
    reocr_engine: str = DEFAULT_REOCR_ENGINE,
    reocr_pad: int = 4,
    paddle_url: Optional[str] = None,
    timeout: int = 20,
    batch_size: int = 32,
) -> int:
    print("prepare-filter: matched -> segment -> reocr")

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

    return run_reocr_books(
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
    )

