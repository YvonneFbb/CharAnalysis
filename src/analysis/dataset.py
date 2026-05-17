#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared dataset helpers for analysis and paper export scripts."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class BookMeta:
    book_id: str
    title: str
    year: Optional[int]
    number: Optional[int]
    region: str
    province: str
    place: str
    style: str
    style_tags: List[str]


def split_style_tags(style_raw: str) -> List[str]:
    if not style_raw:
        return ["（空）"]
    cleaned = (
        style_raw.replace("，", ",")
        .replace("、", ",")
        .replace("/", ",")
        .replace(";", ",")
        .replace("；", ",")
    )
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    return parts or ["（空）"]


def parse_book_prefix(book: str) -> Tuple[Optional[int], Optional[int]]:
    book = book.strip()
    if not book:
        return (None, None)
    parts = book.split("_", 2)
    if len(parts) < 2:
        return (None, None)
    order = None
    year = None
    try:
        order = int(parts[0])
    except Exception:
        order = None
    try:
        year = int(parts[1])
    except Exception:
        year = None
    return (order, year)


def read_books_metadata(path: Path) -> Dict[str, BookMeta]:
    out: Dict[str, BookMeta] = {}
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            book_id = (row.get("BookID") or "").strip()
            if not book_id:
                continue
            title = (row.get("宋刻本") or "").strip()
            year_raw = (row.get("年份") or "").strip()
            num_raw = (row.get("Number") or "").strip()
            try:
                year: Optional[int] = int(year_raw) if year_raw else None
            except Exception:
                year = None
            try:
                number: Optional[int] = int(num_raw) if num_raw else None
            except Exception:
                number = None
            style_raw = (row.get("刻体倾向") or "").strip()
            out[book_id] = BookMeta(
                book_id=book_id,
                title=title,
                year=year,
                number=number,
                region=(row.get("区域划分") or "").strip(),
                province=(row.get("省份") or "").strip(),
                place=(row.get("地点") or "").strip(),
                style=style_raw,
                style_tags=split_style_tags(style_raw),
            )
    return out


def load_manifest(bundle_root: Path) -> Dict:
    manifest_path = bundle_root / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"找不到 bundle：{manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_manifest_books(bundle_root: Path) -> List[str]:
    manifest = load_manifest(bundle_root)
    books = sorted((manifest.get("books") or {}).keys())
    if not books:
        raise SystemExit("bundle 中没有任何 books。")
    return books


def iter_book_entries(bundle_root: Path, book: str) -> List[Dict]:
    entries_path = bundle_root / "books" / book / "entries.json"
    with open(entries_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("entries") or []


def metadata_for_book(book: str, meta_map: Dict[str, BookMeta]) -> Tuple[Optional[BookMeta], Optional[int], Optional[int]]:
    meta = meta_map.get(book)
    order = meta.number if meta and meta.number is not None else None
    year = meta.year if meta and meta.year is not None else None
    if order is None or year is None:
        pref_order, pref_year = parse_book_prefix(book)
        if order is None:
            order = pref_order
        if year is None:
            year = pref_year
    return meta, order, year


def make_book_row(book: str, meta_map: Dict[str, BookMeta], metrics: Dict) -> Dict:
    meta, order, year = metadata_for_book(book, meta_map)
    return {
        "book": book,
        "title": meta.title if meta else "",
        "year": year,
        "order": order,
        "region": meta.region if meta else "",
        "province": meta.province if meta else "",
        "place": meta.place if meta else "",
        "style": meta.style if meta else "",
        "style_tags": meta.style_tags if meta else ["（空）"],
        **metrics,
    }
