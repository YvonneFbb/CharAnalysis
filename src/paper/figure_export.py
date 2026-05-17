#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Iterable


DPI_BY_KIND = {
    "lineart": 1200,
    "grayscale": 600,
    "colour": 300,
    "color": 300,
}


def parse_formats(value: str | None) -> list[str]:
    if not value:
        return ["pdf"]
    parts = [p.strip().lower() for p in str(value).replace("，", ",").split(",")]
    out = [p for p in parts if p]
    return out or ["pdf"]


def default_dpi(kind: str) -> int:
    return int(DPI_BY_KIND.get(str(kind).lower(), 300))


def save_figure_formats(fig, out_dir: Path, stem: str, formats: Iterable[str], dpi: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for fmt in formats:
        ext = fmt.lower()
        if ext == "jpg":
            ext = "jpeg"
        path = out_dir / f"{stem}.{ext}"
        if ext == "pdf":
            fig.savefig(path, bbox_inches="tight", facecolor="white")
        else:
            fig.savefig(path, format=ext, dpi=int(dpi), bbox_inches="tight", facecolor="white")
        saved.append(path)
    return saved
