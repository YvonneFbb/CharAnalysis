#!/usr/bin/env python3
"""
Export paper figures for the character extraction pipeline.

This script writes three images for a selected page when possible:
1. original page image
2. preprocessed page image
3. OCR bounding-box overlay

Typical usage:
  python src/paper/export_extraction_figure.py \
    --preprocessed data/results/preprocessed/03_1127_周易注疏/册01_pages/page_0020_preprocessed.png

If the original page image is available, pass it explicitly:
  python src/paper/export_extraction_figure.py \
    --original /path/to/page_0020.png \
    --preprocessed data/results/preprocessed/03_1127_周易注疏/册01_pages/page_0020_preprocessed.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "src/paper/figures/extraction"


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def infer_original_from_preprocessed(preprocessed_path: Path) -> Optional[Path]:
    name = preprocessed_path.name
    if "_preprocessed" not in name:
        return None
    original_name = name.replace("_preprocessed", "", 1)
    candidate = preprocessed_path.with_name(original_name)
    if candidate.exists():
        return candidate
    return None


def infer_ocr_json_from_preprocessed(preprocessed_path: Path) -> Path:
    rel = preprocessed_path.relative_to(PROJECT_ROOT / "data/results/preprocessed")
    return (PROJECT_ROOT / "data/results/ocr" / rel).with_name(
        preprocessed_path.stem + "_ocr.json"
    )


def load_characters(ocr_json_path: Path) -> list[dict]:
    with open(ocr_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("characters") or []


def draw_ocr_boxes(
    image_path: Path,
    characters: Iterable[dict],
    output_path: Path,
    line_width: int = 2,
    outline_color: tuple[int, int, int] = (220, 50, 47),
    fill_color: tuple[int, int, int, int] = (220, 50, 47, 48),
) -> None:
    base = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    for char in characters:
        bbox = char.get("bbox") or {}
        x = int(bbox.get("x") or 0)
        y = int(bbox.get("y") or 0)
        w = int(bbox.get("width") or 0)
        h = int(bbox.get("height") or 0)
        if w <= 0 or h <= 0:
            continue
        draw.rectangle(
            [x, y, x + w, y + h],
            outline=outline_color + (255,),
            fill=fill_color,
            width=line_width,
        )

    out = Image.alpha_composite(base, overlay).convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path)


def copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img.save(dst)


def build_output_stem(
    preprocessed_path: Optional[Path],
    original_path: Optional[Path],
    stem_override: Optional[str],
) -> str:
    if stem_override:
        return stem_override
    source = preprocessed_path or original_path
    if source is None:
        return "page"
    stem = source.stem
    if stem.endswith("_preprocessed"):
        stem = stem[: -len("_preprocessed")]
    return stem


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export original/preprocessed/OCR-box figures for paper writing."
    )
    parser.add_argument("--original", help="Original page image path")
    parser.add_argument("--preprocessed", help="Preprocessed page image path")
    parser.add_argument("--ocr-json", help="OCR JSON path")
    parser.add_argument(
        "--overlay-base",
        choices=("preprocessed", "original"),
        default="preprocessed",
        help="Which image to use for the OCR overlay",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory (default: src/paper/figures/extraction)",
    )
    parser.add_argument("--name", help="Output file stem override")
    args = parser.parse_args()

    original_path = _resolve_path(args.original) if args.original else None
    preprocessed_path = _resolve_path(args.preprocessed) if args.preprocessed else None

    if preprocessed_path is None and original_path is None:
        raise SystemExit("At least one of --original or --preprocessed is required.")

    if preprocessed_path is None and original_path is not None:
        raise SystemExit(
            "Current script expects an existing preprocessed image. "
            "Pass --preprocessed explicitly."
        )

    assert preprocessed_path is not None
    if not preprocessed_path.exists():
        raise SystemExit(f"Preprocessed image not found: {preprocessed_path}")

    if original_path is None:
        original_path = infer_original_from_preprocessed(preprocessed_path)

    ocr_json_path = _resolve_path(args.ocr_json) if args.ocr_json else infer_ocr_json_from_preprocessed(preprocessed_path)
    if not ocr_json_path.exists():
        raise SystemExit(f"OCR JSON not found: {ocr_json_path}")

    out_dir = _resolve_path(args.out_dir)
    stem = build_output_stem(preprocessed_path, original_path, args.name)
    page_dir = out_dir / stem
    page_dir.mkdir(parents=True, exist_ok=True)

    preprocessed_out = page_dir / f"{stem}_preprocessed.png"
    copy_image(preprocessed_path, preprocessed_out)

    original_out = None
    if original_path is not None and original_path.exists():
        original_out = page_dir / f"{stem}_original{original_path.suffix.lower()}"
        copy_image(original_path, original_out)

    characters = load_characters(ocr_json_path)
    overlay_source = preprocessed_path if args.overlay_base == "preprocessed" else original_path
    if overlay_source is None or not overlay_source.exists():
        overlay_source = preprocessed_path
    overlay_out = page_dir / f"{stem}_ocr_boxes.png"
    draw_ocr_boxes(overlay_source, characters, overlay_out)

    manifest = {
        "original": str(original_out) if original_out else None,
        "preprocessed": str(preprocessed_out),
        "ocr_json": str(ocr_json_path),
        "ocr_boxes": str(overlay_out),
        "overlay_base": args.overlay_base if original_out else "preprocessed",
        "character_count": len(characters),
    }
    with open(page_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Output directory: {page_dir}")
    if original_out is None:
        print("Original page image not found; exported preprocessed and OCR-box figures only.")
    print(f"Characters drawn: {len(characters)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
