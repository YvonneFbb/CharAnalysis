#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.paper.figure_export import default_dpi, parse_formats


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = PROJECT_ROOT / "src/paper/figures"
FONT_DIR = Path.home() / "Library/Fonts"
FONT_REGULAR = FONT_DIR / "PingFangSC-Regular.ttf"


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _load_font(size: int):
    if FONT_REGULAR.exists():
        return ImageFont.truetype(str(FONT_REGULAR), size=size)
    return ImageFont.load_default()


def _fit_width(img: Image.Image, target_w: int) -> Image.Image:
    w, h = img.size
    scale = target_w / float(w)
    target_h = int(round(h * scale))
    return img.resize((target_w, target_h), Image.Resampling.LANCZOS)


def _draw_centered_text(draw: ImageDraw.ImageDraw, box, text: str, font, fill=(0, 0, 0)):
    left, top, right, bottom = box
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = left + (right - left - tw) / 2
    y = top + (bottom - top - th) / 2
    draw.text((x, y), text, font=font, fill=fill)


def main() -> int:
    ap = argparse.ArgumentParser(description="Compose five extraction/selection panels into one figure.")
    ap.add_argument("--original", default="src/paper/figures/extraction/orig_crop/orig_crop_original.png")
    ap.add_argument("--preprocessed", default="src/paper/figures/extraction/orig_crop/orig_crop_preprocessed.png")
    ap.add_argument("--ocr", default="src/paper/figures/extraction/orig_crop/orig_crop_ocr_boxes.png")
    ap.add_argument("--target", default="src/paper/figures/selection/orig_crop/target_highlight.png")
    ap.add_argument("--candidate", default="src/paper/figures/selection/orig_crop/candidate_crops.png")
    ap.add_argument("--name", default="extraction_pipeline_montage")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--formats", default="pdf,tiff")
    ap.add_argument("--dpi-kind", choices=("lineart", "grayscale", "colour", "color"), default="colour")
    args = ap.parse_args()

    paths = [
        _resolve(args.original),
        _resolve(args.preprocessed),
        _resolve(args.ocr),
        _resolve(args.target),
        _resolve(args.candidate),
    ]
    captions = [
        "(a) Original page crop",
        "(b) Pre-processing result",
        "(c) OCR initial localization",
        "(d) Target-character filtering",
        "(e) Single-character extraction",
    ]

    images = [Image.open(p).convert("RGB") for p in paths]
    cell_w = 1020
    imgs = [_fit_width(im, cell_w) for im in images]
    cell_h = imgs[0].size[1]

    side = 48
    gap_x = 28
    gap_y = 48
    caption_h = 78
    top_y = 36

    total_w = side * 2 + cell_w * 3 + gap_x * 2
    row1_h = cell_h + caption_h
    row2_h = cell_h + caption_h
    total_h = top_y + row1_h + gap_y + row2_h + 24

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = _load_font(34)

    x_top = [side, side + cell_w + gap_x, side + 2 * (cell_w + gap_x)]
    y_img_top = top_y
    y_cap_top = y_img_top + cell_h + 8

    # top row 3
    for i in range(3):
        canvas.paste(imgs[i], (x_top[i], y_img_top))
        _draw_centered_text(
            draw,
            (x_top[i], y_cap_top, x_top[i] + cell_w, y_cap_top + caption_h),
            captions[i],
            font,
        )

    # bottom row 2, centered
    total_two = cell_w * 2 + gap_x
    x0 = (total_w - total_two) // 2
    x_bottom = [x0, x0 + cell_w + gap_x]
    y_img_bottom = top_y + row1_h + gap_y
    y_cap_bottom = y_img_bottom + cell_h + 8
    for i in range(2):
        canvas.paste(imgs[3 + i], (x_bottom[i], y_img_bottom))
        _draw_centered_text(
            draw,
            (x_bottom[i], y_cap_bottom, x_bottom[i] + cell_w, y_cap_bottom + caption_h),
            captions[3 + i],
            font,
        )

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name
    dpi = default_dpi(args.dpi_kind)
    for fmt in parse_formats(args.formats):
        ext = "jpeg" if fmt.lower() == "jpg" else fmt.lower()
        out = out_dir / f"{stem}.{ext}"
        if ext == "pdf":
            canvas.save(out, resolution=float(dpi))
        elif ext == "tiff":
            canvas.save(out, compression="tiff_lzw", dpi=(dpi, dpi))
        elif ext == "jpeg":
            canvas.save(out, quality=95, subsampling=0, dpi=(dpi, dpi))
        else:
            canvas.save(out, dpi=(dpi, dpi))
        print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
