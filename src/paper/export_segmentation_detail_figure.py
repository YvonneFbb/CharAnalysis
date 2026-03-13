#!/usr/bin/env python3
"""
Export step-by-step segmentation figures for a single character example.

The script selects one OCR candidate for the requested character, runs the
current segmentation pipeline step by step, and saves intermediate results for
paper figures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.review.segment.config import merge_params
from src.review.segment.noise_removal import remove_noise_patches
from src.review.segment.projection_trim import binarize, trim_projection_from_bin
from src.review.segment.cc_filter import refine_binary_components
from src.review.segment.border_removal import trim_border_from_bin, remove_border_frames

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "src/paper/figures/segmentation"


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_ocr_characters(path: Path, target_text: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    chars = payload.get("characters") or []
    chars = [c for c in chars if c.get("text") == target_text]
    chars.sort(key=lambda c: (int((c.get("bbox") or {}).get("y") or 0), int((c.get("bbox") or {}).get("x") or 0)))
    return chars


def to_rgb(gray_or_bgr: np.ndarray) -> np.ndarray:
    if gray_or_bgr.ndim == 2:
        return cv2.cvtColor(gray_or_bgr, cv2.COLOR_GRAY2BGR)
    return gray_or_bgr.copy()


def to_white_bg_binary(bin_img: np.ndarray) -> np.ndarray:
    vis = 255 - bin_img
    return cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)


def save_image(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def crop_roi(img: np.ndarray, bbox: dict, padding: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = img.shape[:2]
    x0 = max(0, int(bbox["x"]) - padding)
    y0 = max(0, int(bbox["y"]) - padding)
    x1 = min(w, int(bbox["x"]) + int(bbox["width"]) + padding)
    y1 = min(h, int(bbox["y"]) + int(bbox["height"]) + padding)
    return img[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)


def render_bbox_context(img: np.ndarray, bbox: dict, padding: int = 30) -> np.ndarray:
    h, w = img.shape[:2]
    x = int(bbox["x"])
    y = int(bbox["y"])
    bw = int(bbox["width"])
    bh = int(bbox["height"])
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(w, x + bw + padding)
    y1 = min(h, y + bh + padding)
    crop = img[y0:y1, x0:x1].copy()
    cv2.rectangle(crop, (x - x0, y - y0), (x - x0 + bw, y - y0 + bh), (0, 0, 255), 2)
    return crop


def render_box_overlay(base_bgr: np.ndarray, box: tuple[int, int, int, int], color: tuple[int, int, int]) -> np.ndarray:
    out = base_bgr.copy()
    x0, y0, x1, y1 = box
    cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
    return out


def render_noise_overlay(gray_before: np.ndarray, gray_after: np.ndarray) -> np.ndarray:
    base = cv2.cvtColor(gray_after, cv2.COLOR_GRAY2BGR)
    removed = (gray_after.astype(np.int16) - gray_before.astype(np.int16)) > 8
    out = base.copy()
    out[removed] = (70, 70, 230)
    return out


def render_cc_overlay(bin_before: np.ndarray, bin_after: np.ndarray) -> np.ndarray:
    white_bg = 255 - bin_after
    out = cv2.cvtColor(white_bg, cv2.COLOR_GRAY2BGR)
    removed = (bin_before > 0) & (bin_after == 0)
    out[removed] = (70, 70, 230)
    return out


def render_projection_panel(
    gray_img: np.ndarray,
    bin_img: np.ndarray,
    xl: int,
    xr: int,
    yt: int,
    yb: int,
    final_box: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    roi = to_rgb(gray_img)
    h, w = gray_img.shape[:2]

    top_h = 80
    left_w = 80
    pad = 12
    canvas = np.full((top_h + h + pad * 2, left_w + w + pad * 2, 3), 255, dtype=np.uint8)

    ox = left_w + pad
    oy = top_h + pad
    canvas[oy:oy + h, ox:ox + w] = roi
    cv2.rectangle(canvas, (ox + xl, oy + yt), (ox + xr, oy + yb), (255, 0, 0), 2)
    if final_box is not None:
        fx0, fy0, fx1, fy1 = final_box
        cv2.rectangle(canvas, (ox + fx0, oy + fy0), (ox + fx1, oy + fy1), (0, 128, 255), 2)

    mask = (bin_img > 0).astype(np.uint8)
    hproj = mask.sum(axis=0).astype(np.float32)
    vproj = mask.sum(axis=1).astype(np.float32)
    max_h = float(hproj.max()) if hproj.size else 1.0
    max_v = float(vproj.max()) if vproj.size else 1.0

    # horizontal projection on top
    proj_base_y = top_h
    for x in range(w):
        val = 0 if max_h <= 0 else hproj[x] / max_h
        bar_h = int(round(val * (top_h - 18)))
        if bar_h > 0:
            cv2.line(
                canvas,
                (ox + x, proj_base_y - 6),
                (ox + x, proj_base_y - 6 - bar_h),
                (40, 40, 40),
                1,
            )
    cv2.line(canvas, (ox + xl, pad), (ox + xl, top_h), (255, 0, 0), 2)
    cv2.line(canvas, (ox + xr, pad), (ox + xr, top_h), (255, 0, 0), 2)
    cv2.putText(canvas, "H-Proj", (ox, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)

    # vertical projection on left
    proj_base_x = left_w
    for y in range(h):
        val = 0 if max_v <= 0 else vproj[y] / max_v
        bar_w = int(round(val * (left_w - 18)))
        if bar_w > 0:
            cv2.line(
                canvas,
                (proj_base_x - 6, oy + y),
                (proj_base_x - 6 - bar_w, oy + y),
                (40, 40, 40),
                1,
            )
    cv2.line(canvas, (pad, oy + yt), (left_w, oy + yt), (255, 0, 0), 2)
    cv2.line(canvas, (pad, oy + yb), (left_w, oy + yb), (255, 0, 0), 2)
    cv2.putText(canvas, "V", (18, oy + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)

    return canvas


def _resize_to_box(img: np.ndarray, box_w: int, box_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(box_w / max(1, w), box_h / max(1, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def make_composite(output_path: Path, panels: list[tuple[str, np.ndarray]], cols: int = 4) -> None:
    if not panels:
        raise ValueError("No panels for composite.")

    tile_w = 300
    tile_h = 220
    title_h = 34
    pad = 16
    rows = (len(panels) + cols - 1) // cols
    canvas_w = cols * tile_w + (cols + 1) * pad
    canvas_h = rows * (tile_h + title_h) + (rows + 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for idx, (title, panel) in enumerate(panels):
        r = idx // cols
        c = idx % cols
        x = pad + c * (tile_w + pad)
        y = pad + r * (tile_h + title_h + pad)

        draw.rectangle([x, y, x + tile_w, y + title_h + tile_h], outline=(190, 190, 190), width=1)
        draw.text((x + 8, y + 8), title, fill=(20, 20, 20))

        resized = _resize_to_box(panel, tile_w - 12, tile_h - 12)
        panel_img = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        px = x + (tile_w - resized.shape[1]) // 2
        py = y + title_h + (tile_h - resized.shape[0]) // 2
        canvas.paste(panel_img, (px, py))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def choose_candidate(preprocessed_image: Path, candidates: list[dict], padding: int) -> tuple[dict, dict]:
    img = cv2.imread(str(preprocessed_image))
    if img is None:
        raise ValueError(f"Cannot load image: {preprocessed_image}")

    scored: list[tuple[int, int, dict]] = []
    for char in candidates:
        bbox = char.get("bbox") or {}
        roi, _ = crop_roi(img, bbox, padding)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        merged = merge_params({})
        gray_cleaned = remove_noise_patches(gray.copy(), merged["noise_removal"])
        if isinstance(gray_cleaned, tuple):
            gray_cleaned = gray_cleaned[0]
        bin_before = binarize(
            gray_cleaned,
            mode=str(merged["projection_trim"].get("binarize", "otsu")).lower(),
            adaptive_block=int(merged["projection_trim"].get("adaptive_block", 31)),
            adaptive_C=int(merged["projection_trim"].get("adaptive_C", 3)),
        )
        bin_after, _ = refine_binary_components(bin_before.copy(), merged["cc_filter"], gray_cleaned)
        if merged["border_removal"].get("frame_removal", {}).get("enabled", False):
            bin_after = remove_border_frames(bin_after, merged["border_removal"])
        xl, xr, yt, yb = trim_projection_from_bin(bin_after, merged["projection_trim"])
        border_bin = bin_after[yt:yb, xl:xr]
        xl_b, xr_b, yt_b, yb_b = trim_border_from_bin(border_bin, merged["border_removal"])
        final_w = (xr_b - xl_b)
        final_h = (yb_b - yt_b)
        scored.append((final_w, final_h, char))

    scored.sort(key=lambda item: (-item[0], -item[1], int((item[2].get("bbox") or {}).get("y") or 0)))
    best = scored[0][2]
    return best, {"seg_width": scored[0][0], "seg_height": scored[0][1], "candidate_count": len(scored)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Export detailed segmentation figures for one character example.")
    parser.add_argument("--base-image", required=True, help="Preprocessed image used for segmentation")
    parser.add_argument("--ocr-json", required=True, help="OCR JSON for the base image")
    parser.add_argument("--char", default="武", help="Target character to visualize")
    parser.add_argument("--padding", type=int, default=10, help="Padding around OCR bbox")
    parser.add_argument("--name", default="orig_crop_wu", help="Output folder name")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output root directory")
    args = parser.parse_args()

    base_image = _resolve_path(args.base_image)
    ocr_json = _resolve_path(args.ocr_json)
    out_root = _resolve_path(args.out_dir) / args.name
    out_root.mkdir(parents=True, exist_ok=True)

    candidates = load_ocr_characters(ocr_json, args.char)
    if not candidates:
        raise SystemExit(f"No OCR candidates found for character: {args.char}")

    chosen, chosen_meta = choose_candidate(base_image, candidates, args.padding)
    bbox = chosen["bbox"]

    img = cv2.imread(str(base_image))
    if img is None:
        raise ValueError(f"Cannot load image: {base_image}")

    merged = merge_params({})
    noise_cfg = merged["noise_removal"]
    cc_cfg = merged["cc_filter"]
    proj_cfg = merged["projection_trim"]
    border_cfg = merged["border_removal"]

    roi, (x0, y0, x1, y1) = crop_roi(img, bbox, args.padding)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_cleaned = remove_noise_patches(gray.copy(), noise_cfg)
    if isinstance(gray_cleaned, tuple):
        gray_cleaned = gray_cleaned[0]

    bin_before = binarize(
        gray_cleaned,
        mode=str(proj_cfg.get("binarize", "otsu")).lower(),
        adaptive_block=int(proj_cfg.get("adaptive_block", 31)),
        adaptive_C=int(proj_cfg.get("adaptive_C", 3)),
    )
    bin_after, _ = refine_binary_components(bin_before.copy(), cc_cfg, gray_cleaned)
    if border_cfg.get("frame_removal", {}).get("enabled", False):
        bin_after = remove_border_frames(bin_after, border_cfg)

    xl, xr, yt, yb = trim_projection_from_bin(bin_after, proj_cfg)
    proj_overlay = render_projection_panel(gray_cleaned, bin_after, xl, xr, yt, yb)

    border_bin = bin_after[yt:yb, xl:xr]
    xl_b, xr_b, yt_b, yb_b = trim_border_from_bin(border_bin, border_cfg)
    xl_final = xl + xl_b
    xr_final = xl + xr_b
    yt_final = yt + yt_b
    yb_final = yt + yb_b
    final_overlay = render_projection_panel(
        gray_cleaned,
        bin_after,
        xl,
        xr,
        yt,
        yb,
        final_box=(xl_final, yt_final, xr_final, yb_final),
    )

    processed_gray = gray_cleaned.copy()
    processed_gray[bin_after == 0] = 255
    final_segmented = processed_gray[yt_final:yb_final, xl_final:xr_final]

    noise_overlay = render_noise_overlay(gray, gray_cleaned)
    cc_overlay = render_cc_overlay(bin_before, bin_after)

    context_img = render_bbox_context(img, bbox, padding=30)
    save_image(out_root / "01_context_bbox.png", context_img)
    save_image(out_root / "02_roi_gray.png", to_rgb(gray))
    save_image(out_root / "03_noise_cleaned.png", noise_overlay)
    save_image(out_root / "04_binarized.png", to_white_bg_binary(bin_before))
    save_image(out_root / "05_cc_filtered.png", cc_overlay)
    save_image(out_root / "06_projection_trim.png", proj_overlay)
    save_image(out_root / "07_final_trim.png", final_overlay)
    save_image(out_root / "08_final_segmented.png", to_rgb(final_segmented))

    panels = [
        ("(a) OCR bbox", context_img),
        ("(b) ROI", to_rgb(gray)),
        ("(c) Noise removal", noise_overlay),
        ("(d) Binarization", to_white_bg_binary(bin_before)),
        ("(e) CC filter", cc_overlay),
        ("(f) Projection trim", proj_overlay),
        ("(g) Final trim", final_overlay),
        ("(h) Final glyph", to_rgb(final_segmented)),
    ]
    make_composite(out_root / "00_segmentation_composite.png", panels)

    manifest = {
        "base_image": str(base_image),
        "ocr_json": str(ocr_json),
        "char": args.char,
        "chosen_candidate": chosen,
        "chosen_meta": chosen_meta,
        "roi_bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
        "projection_box": {"xl": xl, "xr": xr, "yt": yt, "yb": yb},
        "final_box": {"xl": xl_final, "xr": xr_final, "yt": yt_final, "yb": yb_final},
    }
    with open(out_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Output directory: {out_root}")
    print(f"Chosen char: {args.char}")
    print(f"Chosen bbox: {chosen['bbox']}")
    print(f"Chosen segmented size: {chosen_meta['seg_width']}x{chosen_meta['seg_height']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
