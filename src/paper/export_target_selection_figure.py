#!/usr/bin/env python3
"""
Export paper figures for target-character selection and candidate segmentation.

Outputs:
1. target_highlight.png   - highlight OCR boxes whose text is in the standard-char pool
2. candidate_crops.png    - segmented candidates pasted back at original positions
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.review.segment.core import segment_character

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "src/paper/figures/selection"


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_standard_chars(path: Path) -> set[str]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    chars = payload.get("all_chars") or []
    return {c for c in chars if isinstance(c, str) and len(c) == 1}


def load_ocr_characters(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    chars = payload.get("characters") or []
    chars = [c for c in chars if isinstance(c.get("text"), str) and len(c["text"]) == 1]
    chars.sort(key=lambda c: (int((c.get("bbox") or {}).get("y") or 0), int((c.get("bbox") or {}).get("x") or 0)))
    return chars


def draw_target_highlight(
    base_image: Path,
    ocr_characters: list[dict],
    standard_chars: set[str],
    output_path: Path,
    big_topk: int,
) -> tuple[int, int, int]:
    img = Image.open(base_image).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    total = 0
    matched = 0
    for char in ocr_characters:
        bbox = char.get("bbox") or {}
        x = int(bbox.get("x") or 0)
        y = int(bbox.get("y") or 0)
        w = int(bbox.get("width") or 0)
        h = int(bbox.get("height") or 0)
        if w <= 0 or h <= 0:
            continue
        total += 1
        if char["text"] in standard_chars:
            matched += 1
            draw.rectangle(
                [x, y, x + w, y + h],
                outline=(211, 47, 47, 255),
                fill=(211, 47, 47, 60),
                width=2,
            )
        else:
            draw.rectangle(
                [x, y, x + w, y + h],
                outline=(120, 120, 120, 110),
                width=1,
            )

    out = Image.alpha_composite(img, overlay).convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path)
    return total, matched, 0


def segment_candidates(
    preprocessed_image: Path,
    ocr_characters: list[dict],
    standard_chars: set[str],
    big_topk: int,
    padding: int,
) -> tuple[list[dict], int]:
    records: list[dict] = []
    failures = 0
    for char in ocr_characters:
        if char["text"] not in standard_chars:
            continue
        bbox = char.get("bbox") or {}
        try:
            _, segmented_image, _, metadata, _ = segment_character(
                str(preprocessed_image),
                {
                    "x": int(bbox.get("x") or 0),
                    "y": int(bbox.get("y") or 0),
                    "width": int(bbox.get("width") or 0),
                    "height": int(bbox.get("height") or 0),
                },
                padding=padding,
            )
        except Exception:
            failures += 1
            continue

        pil = Image.fromarray(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)).convert("L")
        abs_bbox = metadata.get("segmented_bbox_absolute") or {}
        records.append(
            {
                "text": char["text"],
                "image": pil,
                "seg_width": pil.size[0],
                "seg_height": pil.size[1],
                "x": int(abs_bbox.get("x") or 0),
                "y": int(abs_bbox.get("y") or 0),
                "width": int(abs_bbox.get("width") or pil.size[0]),
                "height": int(abs_bbox.get("height") or pil.size[1]),
                "kind": "small",
            }
        )

    records.sort(key=lambda r: (-r["seg_width"], -r["seg_height"], r["y"], r["x"]))
    for idx, rec in enumerate(records):
        if idx < max(0, big_topk):
            rec["kind"] = "big"
    return records, failures


def _faded_base(base_image: Path) -> Image.Image:
    base = Image.open(base_image).convert("L").convert("RGB")
    white = Image.new("RGB", base.size, (255, 255, 255))
    return Image.blend(white, base, 0.18)


def _tint_segment(seg: Image.Image, color: tuple[int, int, int]) -> Image.Image:
    gray = seg.convert("L")
    rgba = Image.new("RGBA", gray.size, (255, 255, 255, 0))
    px = gray.load()
    out = rgba.load()
    w, h = gray.size
    for y in range(h):
        for x in range(w):
            v = px[x, y]
            if v < 245:
                alpha = max(40, 255 - v)
                out[x, y] = (color[0], color[1], color[2], alpha)
    return rgba


def make_segment_overlay(
    base_image: Path,
    records: list[dict],
    output_path: Path,
) -> None:
    if not records:
        raise SystemExit("No matched candidate segments to render.")

    canvas = _faded_base(base_image).convert("RGBA")
    for rec in records:
        color = (211, 47, 47) if rec["kind"] == "big" else (235, 125, 35)
        tinted = _tint_segment(rec["image"], color)
        x = rec["x"]
        y = rec["y"]
        canvas.alpha_composite(tinted, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export target-selection figures for paper writing.")
    parser.add_argument("--base-image", required=True, help="Base image used for OCR/cropping, usually the preprocessed crop")
    parser.add_argument("--ocr-json", required=True, help="OCR JSON for the base image")
    parser.add_argument("--standard-chars", default="src/standard_chars.json", help="Standard-char JSON")
    parser.add_argument("--name", default="selection", help="Output folder name")
    parser.add_argument("--padding", type=int, default=10, help="Padding around OCR bbox for segmentation")
    parser.add_argument("--big-topk", type=int, default=2, help="Number of widest matched OCR targets treated as large glyphs")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output root directory")
    args = parser.parse_args()

    base_image = _resolve_path(args.base_image)
    ocr_json = _resolve_path(args.ocr_json)
    standard_chars_path = _resolve_path(args.standard_chars)
    out_root = _resolve_path(args.out_dir) / args.name
    out_root.mkdir(parents=True, exist_ok=True)

    standard_chars = load_standard_chars(standard_chars_path)
    ocr_characters = load_ocr_characters(ocr_json)

    total, matched, _ = draw_target_highlight(
        base_image,
        ocr_characters,
        standard_chars,
        out_root / "target_highlight.png",
        big_topk=args.big_topk,
    )

    records, segmentation_failures = segment_candidates(
        base_image,
        ocr_characters,
        standard_chars,
        big_topk=args.big_topk,
        padding=args.padding,
    )
    make_segment_overlay(
        base_image,
        records,
        out_root / "candidate_crops.png",
    )

    big_count = sum(1 for rec in records if rec["kind"] == "big")
    small_count = sum(1 for rec in records if rec["kind"] == "small")

    manifest = {
        "base_image": str(base_image),
        "ocr_json": str(ocr_json),
        "standard_chars": str(standard_chars_path),
        "ocr_total": total,
        "target_matched": matched,
        "target_matched_small": small_count,
        "target_matched_big": big_count,
        "crop_count": len(records),
        "segmentation_failures": segmentation_failures,
        "target_highlight": str(out_root / "target_highlight.png"),
        "candidate_crops": str(out_root / "candidate_crops.png"),
    }
    with open(out_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Output directory: {out_root}")
    print(f"OCR total: {total}")
    print(f"Matched targets: {matched} (large={big_count}, small={small_count})")
    print(f"Segmentation failures: {segmentation_failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
