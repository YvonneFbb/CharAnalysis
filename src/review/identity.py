"""Shared instance identity and path normalization helpers."""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple


def normalize_to_preprocessed_path(raw_or_mixed_path: str) -> str:
    """
    Convert raw page image paths to their preprocessed counterparts.

    data/raw/{book}/册XX_pages/page_XXXX.png
    -> data/results/preprocessed/{book}/册XX_pages/page_XXXX_preprocessed.png
    """
    if not raw_or_mixed_path:
        return raw_or_mixed_path
    if "/preprocessed/" in raw_or_mixed_path and "_preprocessed.png" in raw_or_mixed_path:
        return raw_or_mixed_path

    match = re.search(r"data/raw/([^/]+)/(册\d+_pages)/(page_\d+)\.png", raw_or_mixed_path)
    if match:
        book, volume_dir, page_name = match.groups()
        return f"data/results/preprocessed/{book}/{volume_dir}/{page_name}_preprocessed.png"
    return raw_or_mixed_path


def make_instance_id(inst: Dict) -> str:
    try:
        vol = int(inst.get("volume", 0))
    except Exception:
        vol = 0
    page = inst.get("page", "")
    page_suffix = page.split("_")[-1] if page else ""
    char_index = inst.get("char_index", 0)
    return f"册{vol:02d}_page{page_suffix}_idx{char_index}"


def parse_instance_id(instance_id: str) -> Optional[Tuple[int, str, int]]:
    match = re.match(r"^册(\d+)_page(\d+)_idx(\d+)$", instance_id or "")
    if not match:
        return None
    vol = int(match.group(1))
    page = match.group(2)
    idx = int(match.group(3))
    return vol, page, idx
