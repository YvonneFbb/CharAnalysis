#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OCRMAC 链路验证脚本

流程：
1) ocrmac 对整页进行一次 OCR（stage1）
2) 基于 stage1 的 bbox 进行裁剪，对每个裁剪图再跑一次 ocrmac（stage2）
3) 基于 stage1 的 bbox 做切割（segment），产出切割图（stage3）
4) 对切割图再跑一次 ocrmac（stage4）

输出写入 test/ocrmac_chain 下，便于对比各阶段结果。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import cv2
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.review.ocr.livetext import ocr_image
from src.review.segment.core import segment_character
from src.review.utils.file_filter import find_images_recursive
from src.review.utils.path import ensure_dir


def safe_stem(name: str) -> str:
    out = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "image"


def crop_bbox(image_path: Path, bbox: Dict[str, int], output_path: Path) -> None:
    with Image.open(image_path) as im:
        w, h = im.size
        x0 = max(0, min(w - 1, int(bbox["x"])))
        y0 = max(0, min(h - 1, int(bbox["y"])))
        x1 = max(x0 + 1, min(w, x0 + int(bbox["width"])))
        y1 = max(y0 + 1, min(h, y0 + int(bbox["height"])))
        crop = im.crop((x0, y0, x1, y1))
        ensure_dir(output_path.parent)
        crop.save(output_path)


def pick_primary_text(ocr_result: Dict[str, Any]) -> Dict[str, Any]:
    if not ocr_result.get("success"):
        return {
            "success": False,
            "text": "",
            "confidence": None,
            "count": 0,
        }
    chars = ocr_result.get("characters") or []
    if not chars:
        return {
            "success": True,
            "text": "",
            "confidence": None,
            "count": 0,
        }
    first = chars[0]
    return {
        "success": True,
        "text": str(first.get("text", "")),
        "confidence": first.get("confidence"),
        "count": len(chars),
    }


def extract_bboxes(ocr_result: Dict[str, Any]) -> List[Dict[str, int]]:
    if not ocr_result.get("success"):
        return []
    chars = ocr_result.get("characters") or []
    out: List[Dict[str, int]] = []
    for ch in chars:
        bbox = ch.get("bbox")
        if not bbox:
            continue
        try:
            out.append({
                "x": int(bbox["x"]),
                "y": int(bbox["y"]),
                "width": int(bbox["width"]),
                "height": int(bbox["height"]),
            })
        except Exception:
            continue
    return out


def merge_bboxes(bboxes: List[Dict[str, int]]) -> Optional[Dict[str, int]]:
    if not bboxes:
        return None
    x0 = min(b["x"] for b in bboxes)
    y0 = min(b["y"] for b in bboxes)
    x1 = max(b["x"] + b["width"] for b in bboxes)
    y1 = max(b["y"] + b["height"] for b in bboxes)
    return {
        "x": int(x0),
        "y": int(y0),
        "width": int(max(1, x1 - x0)),
        "height": int(max(1, y1 - y0)),
    }


def _resize_to_tile(img: Image.Image, tile: int) -> Image.Image:
    w, h = img.size
    if w <= 0 or h <= 0:
        return Image.new("RGB", (tile, tile), (255, 255, 255))
    scale = min(tile / float(w), tile / float(h), 1.0)
    if scale < 1.0:
        w = max(1, int(w * scale))
        h = max(1, int(h * scale))
        return img.resize((w, h), Image.Resampling.LANCZOS)
    return img


def _load_rgb(path: Path) -> Optional[Image.Image]:
    if not path or not path.exists():
        return None
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except Exception:
        return None


def build_pair_montage(items: List[Dict[str, Any]],
                       output_path: Path,
                       tile: int = 96,
                       cols: int = 8,
                       gap: int = 8) -> None:
    if not items:
        return
    cell_w = tile * 2 + gap
    cell_h = tile
    rows = int(math.ceil(len(items) / float(cols)))
    canvas_w = cols * cell_w + (cols - 1) * gap
    canvas_h = rows * cell_h + (rows - 1) * gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for i, item in enumerate(items):
        row = i // cols
        col = i % cols
        x0 = col * (cell_w + gap)
        y0 = row * (cell_h + gap)

        crop_img = _load_rgb(Path(item.get("crop_path", "")))
        seg_img = _load_rgb(Path(item.get("segment_path", "")))
        if crop_img:
            crop_img = _resize_to_tile(crop_img, tile)
            cx = x0 + (tile - crop_img.width) // 2
            cy = y0 + (tile - crop_img.height) // 2
            canvas.paste(crop_img, (cx, cy))
        if seg_img:
            seg_img = _resize_to_tile(seg_img, tile)
            sx = x0 + tile + gap + (tile - seg_img.width) // 2
            sy = y0 + (tile - seg_img.height) // 2
            canvas.paste(seg_img, (sx, sy))

        label = f"{item.get('index', i)}"
        draw.text((x0 + 2, y0 + 2), label, fill=(0, 0, 0))

    ensure_dir(output_path.parent)
    canvas.save(output_path)


def process_image(image_path: Path,
                  output_root: Path,
                  limit_instances: int,
                  min_confidence: float,
                  padding: int,
                  montage_tile: int,
                  montage_cols: int) -> Path:
    stem = safe_stem(image_path.stem)
    base_dir = output_root / stem

    stage1_dir = base_dir / "stage1"
    stage2_crops = base_dir / "stage2" / "crops"
    stage2_ocr_dir = base_dir / "stage2" / "ocr"
    stage3_dir = base_dir / "stage3" / "segments"
    stage4_ocr_dir = base_dir / "stage4" / "ocr"

    ensure_dir(stage1_dir)
    ensure_dir(stage2_crops)
    ensure_dir(stage2_ocr_dir)
    ensure_dir(stage3_dir)
    ensure_dir(stage4_ocr_dir)

    stage1_ocr_path = stage1_dir / f"{stem}_ocr.json"
    stage1 = ocr_image(str(image_path), output_path=str(stage1_ocr_path), verbose=True)
    if not stage1.get("success"):
        raise RuntimeError(f"OCR 失败: {stage1.get('error')}")

    characters = stage1.get("characters") or []
    summary: Dict[str, Any] = {
        "image": str(image_path),
        "stage1_ocr": str(stage1_ocr_path),
        "character_count": len(characters),
        "instances": [],
    }

    kept = 0
    for idx, ch in enumerate(characters):
        if kept >= limit_instances:
            break
        conf = float(ch.get("confidence", 0.0))
        if conf < min_confidence:
            continue

        bbox = ch.get("bbox") or {}
        if not bbox:
            continue

        item_id = f"idx_{idx:04d}"
        crop_path = stage2_crops / f"{item_id}.png"
        crop_bbox(image_path, bbox, crop_path)

        stage2_ocr_path = stage2_ocr_dir / f"{item_id}_ocr.json"
        stage2 = ocr_image(str(crop_path), output_path=str(stage2_ocr_path), verbose=False)
        stage2_pick = pick_primary_text(stage2)
        stage2_bbox_rel = merge_bboxes(extract_bboxes(stage2))
        stage2_bbox_abs = None
        if stage2_bbox_rel:
            stage2_bbox_abs = {
                "x": int(bbox["x"] + stage2_bbox_rel["x"]),
                "y": int(bbox["y"] + stage2_bbox_rel["y"]),
                "width": int(stage2_bbox_rel["width"]),
                "height": int(stage2_bbox_rel["height"]),
            }

        seg_path = stage3_dir / f"{item_id}.png"
        seg_meta = None
        try:
            _, segmented, _, metadata, _ = segment_character(
                preprocessed_image_path=str(image_path),
                bbox=bbox,
                custom_params=None,
                padding=padding,
            )
            cv2.imwrite(str(seg_path), segmented)
            seg_meta = metadata
        except Exception as e:
            seg_meta = {"error": str(e)}

        stage4_ocr_path = stage4_ocr_dir / f"{item_id}_seg_ocr.json"
        stage4 = ocr_image(str(seg_path), output_path=str(stage4_ocr_path), verbose=False)
        stage4_pick = pick_primary_text(stage4)
        stage4_bbox_rel = merge_bboxes(extract_bboxes(stage4))
        stage4_bbox_abs = None
        seg_abs = None
        if isinstance(seg_meta, dict):
            seg_abs = seg_meta.get("segmented_bbox_absolute")
        if stage4_bbox_rel and seg_abs:
            stage4_bbox_abs = {
                "x": int(seg_abs["x"] + stage4_bbox_rel["x"]),
                "y": int(seg_abs["y"] + stage4_bbox_rel["y"]),
                "width": int(stage4_bbox_rel["width"]),
                "height": int(stage4_bbox_rel["height"]),
            }

        summary["instances"].append({
            "index": idx,
            "text": str(ch.get("text", "")),
            "confidence": conf,
            "bbox": bbox,
            "stage2_bbox_rel": stage2_bbox_rel,
            "stage2_bbox_abs": stage2_bbox_abs,
            "segment_bbox_abs": seg_abs,
            "stage4_bbox_rel": stage4_bbox_rel,
            "stage4_bbox_abs": stage4_bbox_abs,
            "final_bbox_abs": stage4_bbox_abs or stage2_bbox_abs or bbox,
            "crop_path": str(crop_path),
            "stage2_ocr": str(stage2_ocr_path),
            "stage2_pick": stage2_pick,
            "segment_path": str(seg_path),
            "segment_meta": seg_meta,
            "stage4_ocr": str(stage4_ocr_path),
            "stage4_pick": stage4_pick,
        })
        kept += 1

    summary_path = base_dir / "summary.json"
    preview_path = base_dir / "preview.png"
    build_pair_montage(summary["instances"], preview_path, tile=montage_tile, cols=montage_cols)
    summary["preview"] = str(preview_path)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary_path


def main() -> None:
    ap = argparse.ArgumentParser(description="OCRMAC 链路验证脚本")
    ap.add_argument("input", help="输入图片或目录（建议使用 preprocessed 图）")
    ap.add_argument("--output", default="test/ocrmac_chain", help="输出目录（默认 test/ocrmac_chain）")
    ap.add_argument("--limit-images", type=int, default=1, help="最多处理多少张图片（默认 1）")
    ap.add_argument("--limit-instances", type=int, default=50, help="每张图最多处理多少个字符（默认 50）")
    ap.add_argument("--min-confidence", type=float, default=0.0, help="最低置信度过滤（默认 0）")
    ap.add_argument("--padding", type=int, default=10, help="切割 padding（默认 10）")
    ap.add_argument("--montage-tile", type=int, default=96, help="预览拼贴单元尺寸（默认 96）")
    ap.add_argument("--montage-cols", type=int, default=8, help="预览拼贴列数（默认 8）")
    args = ap.parse_args()

    input_path = Path(args.input)
    output_root = PROJECT_ROOT / args.output
    ensure_dir(output_root)

    if input_path.is_dir():
        rels = find_images_recursive(str(input_path))
        rels_sorted = sorted(rels)
        targets = [input_path / rel for rel in rels_sorted[: max(0, args.limit_images)]]
    else:
        targets = [input_path]

    summaries: List[str] = []
    for img_path in targets:
        if not img_path.exists():
            print(f"跳过不存在文件: {img_path}")
            continue
        print(f"\n=== 处理: {img_path} ===")
        try:
            summary_path = process_image(
                img_path,
                output_root,
                limit_instances=int(args.limit_instances),
                min_confidence=float(args.min_confidence),
                padding=int(args.padding),
                montage_tile=int(args.montage_tile),
                montage_cols=int(args.montage_cols),
            )
            summaries.append(str(summary_path))
            print(f"✓ 完成: {summary_path}")
        except Exception as e:
            print(f"✗ 失败: {img_path} - {e}")

    if summaries:
        print("\n完成汇总：")
        for s in summaries:
            print(f"  - {s}")


if __name__ == "__main__":
    main()
