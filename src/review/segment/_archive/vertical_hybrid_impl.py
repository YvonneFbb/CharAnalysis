"""
ocrmac-driven segmentation (CC + projection trimming, simplified).

LiveText detection → connected-component filtering → projection trimming
(src/segmentation/projection_trim.py) → save crops and debug. Legacy
projection/grabcut/morph code moved to vertical_hybrid_legacy.py and is not
used by default.
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional
import os
import json
import cv2
import numpy as np

# Lazy import ocrmac (Apple LiveText)
try:
    from ocrmac import ocrmac as _ocrmac
except ImportError as _e:
    _ocrmac = None
    _OCRMAC_IMPORT_ERROR = str(_e)
else:
    _OCRMAC_IMPORT_ERROR = None

# 使用本地 config.py
from ..config import (
    PROJECTION_TRIM_CONFIG, CC_FILTER_CONFIG,
    BORDER_REMOVAL_CONFIG, NOISE_REMOVAL_CONFIG,
)

# 兼容旧代码：SEGMENT_REFINE_CONFIG 在本地配置中不存在，创建默认值
SEGMENT_REFINE_CONFIG = {
    'enabled': True,
    'mode': 'ccprojection',
    'expand_px': {'left': 4, 'right': 4, 'top': 10, 'bottom': 4},
    'final_pad': 0,
    'debug_visualize': True,
    'debug_dirname': 'debug',
}

# 这些目录变量不在我们的使用场景中，设置为None
PREOCR_DIR = None
SEGMENTS_DIR = None

from ..projection_trim import trim_projection_from_bin, binarize
from ..cc_filter import refine_binary_components
from ..noise_removal import remove_noise_patches
from ..border_removal import trim_border_from_bin
from ..debug_viz import _render_combined_debug, _create_border_debug_image


def _ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def run_on_image(image_path: str, output_dir: str, expected_text: str | None = None,
                 framework: str = 'livetext', recognition_level: str = 'accurate',
                 language_preference: Optional[List[str]] = None) -> Dict[str, Any]:
    if _ocrmac is None:
        return {'success': False, 'error': f'ocrmac 未安装: {_OCRMAC_IMPORT_ERROR}'}
    img = cv2.imread(image_path)
    if img is None:
        return {'success': False, 'error': f'cannot read: {image_path}'}
    os.makedirs(output_dir, exist_ok=True)
    langs = language_preference or ['zh-Hans']
    try:
        o = _ocrmac.OCR(image_path, framework='livetext', recognition_level=recognition_level,
                        language_preference=langs, detail=True)
        res = o.recognize()
    except Exception as e:
        return {'success': False, 'error': f'LiveText 调用失败: {e}'}
    if not res:
        return {'success': False, 'error': 'LiveText 无检测结果'}
    W, H = int(o.image.width), int(o.image.height)
    boxes = []
    for i, tup in enumerate(res):
        if not isinstance(tup, (list, tuple)) or len(tup) != 3:
            continue
        text, conf, nb = tup
        x = int(round(nb[0] * W))
        y_top = int(round((1.0 - nb[1] - nb[3]) * H))
        w = int(round(nb[2] * W))
        h = int(round(nb[3] * H))
        x = max(0, min(W - 1, x))
        y_top = max(0, min(H - 1, y_top))
        w = max(1, min(W - x, w))
        h = max(1, min(H - y_top, h))
        boxes.append({
            'order': i+1,
            'text': str(text),
            'confidence': float(conf),
            'x': x,
            'y': y_top,
            'w': w,
            'h': h,
            'normalized_bbox': [float(nb[0]), float(nb[1]), float(nb[2]), float(nb[3])]
        })
    boxes.sort(key=lambda b: (b['y'], b['x']))

    chars_meta: List[Dict[str, Any]] = []
    refined_boxes: List[Tuple[int,int,int,int]] = []
    dbg_enabled = bool(SEGMENT_REFINE_CONFIG.get('debug_visualize', False))
    dbg_dirname = str(SEGMENT_REFINE_CONFIG.get('debug_dirname', 'debug'))

    raw_mode = str(SEGMENT_REFINE_CONFIG.get('mode', 'ccprojection')).lower()
    mode_alias = {
        'ccprojection': 'ccprojection',
        'cc_projection': 'ccprojection',
        'projection': 'ccprojection',
        'projection_only': 'projection_only',
        'cc_debug': 'cc_debug',
    }
    mode = mode_alias.get(raw_mode, raw_mode)
    if mode not in {'ccprojection', 'projection_only', 'cc_debug'}:
        mode = 'ccprojection'
    expand_cfg = SEGMENT_REFINE_CONFIG.get('expand_px', 0)
    if isinstance(expand_cfg, dict):
        ex_left = int(max(0, expand_cfg.get('left', 0)))
        ex_right = int(max(0, expand_cfg.get('right', 0)))
        ex_top = int(max(0, expand_cfg.get('top', 0)))
        ex_bottom = int(max(0, expand_cfg.get('bottom', 0)))
    else:
        ex_left = ex_right = ex_top = ex_bottom = int(max(0, expand_cfg))

    for b in boxes:
        x0 = max(0, b['x'] - ex_left); y0 = max(0, b['y'] - ex_top)
        x1 = min(W, b['x'] + b['w'] + ex_right); y1 = min(H, b['y'] + b['h'] + ex_bottom)
        roi = img[y0:y1, x0:x1]
        if roi.size == 0:
            continue

        # Skip tiny images (likely noise or artifacts)
        roi_h, roi_w = roi.shape[:2]
        if roi_w < 10 or roi_h < 10:
            continue

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Noise patch removal (before binarization)
        gray_cleaned = gray.copy()
        noise_debug_img = None
        if NOISE_REMOVAL_CONFIG.get('enabled', True):
            result = remove_noise_patches(gray_cleaned, NOISE_REMOVAL_CONFIG)
            # Handle potential debug output
            if isinstance(result, tuple):
                gray_cleaned, noise_debug_img = result
            else:
                gray_cleaned = result

        bin_before = binarize(
            gray_cleaned,
            mode=str(PROJECTION_TRIM_CONFIG.get('binarize', 'otsu')).lower(),
            adaptive_block=int(PROJECTION_TRIM_CONFIG.get('adaptive_block', 31)),
            adaptive_C=int(PROJECTION_TRIM_CONFIG.get('adaptive_C', 3)),
        )

        # choose CC filtering according to mode
        cc_debug_img = None
        if mode == 'projection_only':
            bin_after = bin_before.copy()  # skip CC filtering
        else:
            bin_after = bin_before.copy()
            bin_after, cc_debug_img = refine_binary_components(bin_after, CC_FILTER_CONFIG, gray_cleaned)
        if mode == 'cc_debug':
            # no projection trimming; use full ROI as crop
            xl, xr, yt, yb = 0, roi.shape[1], 0, roi.shape[0]
        else:
            xl, xr, yt, yb = trim_projection_from_bin(bin_after, PROJECTION_TRIM_CONFIG)

        # Store projection coordinates for debug
        xl_proj, xr_proj, yt_proj, yb_proj = xl, xr, yt, yb

        # Border removal (after projection trimming)
        xl_border, xr_border, yt_border, yb_border = xl, xr, yt, yb
        if BORDER_REMOVAL_CONFIG.get('enabled', True):
            # Apply border trimming to the projection-trimmed region
            border_bin = bin_after[yt:yb, xl:xr]
            xl_b, xr_b, yt_b, yb_b = trim_border_from_bin(border_bin, BORDER_REMOVAL_CONFIG)
            # Adjust coordinates back to full ROI space
            xl_border = xl + xl_b
            xr_border = xl + xr_b
            yt_border = yt + yt_b
            yb_border = yt + yb_b

        fp = int(SEGMENT_REFINE_CONFIG.get('final_pad', 0))
        xl_final = max(0, xl_border - fp)
        xr_final = min(roi.shape[1], xr_border + fp)
        yt_final = max(0, yt_border - fp)
        yb_final = min(roi.shape[0], yb_border + fp)
        rx, ry, rw, rh = x0 + int(xl_final), y0 + int(yt_final), max(1, int(xr_final - xl_final)), max(1, int(yb_final - yt_final))
        refined_boxes.append((rx, ry, rw, rh))

        # Create final processed image: apply NOISE removal + CC filtering effects
        # Start with noise-cleaned grayscale image
        processed_gray = gray_cleaned.copy()

        # Apply CC filtering effects by masking with bin_after
        # Where bin_after is 0 (removed by CC), set to white background
        cc_mask = (bin_after > 0)
        processed_gray[~cc_mask] = 255

        # Convert to BGR for final output
        processed_roi = cv2.cvtColor(processed_gray, cv2.COLOR_GRAY2BGR)

        # Create crops from processed image
        crop_before_border = processed_roi[yt_proj:yb_proj, xl_proj:xr_proj]
        crop_after_border = processed_roi[yt_final:yb_final, xl_final:xr_final]
        crop = crop_after_border

        fname = f"char_{b['order']:04d}.png"
        cv2.imwrite(os.path.join(output_dir, fname), crop)

        # Prepare debug output directory
        out_dbg = os.path.join(output_dir, dbg_dirname)

        # Save noise removal debug image (independent of main debug)
        if noise_debug_img is not None:
            try:
                _ensure_dir(out_dbg)
                noise_dbg_name = f"{os.path.splitext(fname)[0]}_noise_debug.png"
                cv2.imwrite(os.path.join(out_dbg, noise_dbg_name), noise_debug_img)
            except Exception as e:
                print(f"[NOISE DEBUG ERROR] Failed to save noise debug for {fname}: {e}")

        # Save CC filter debug image (independent of main debug)
        if cc_debug_img is not None:
            try:
                _ensure_dir(out_dbg)
                cc_dbg_name = f"{os.path.splitext(fname)[0]}_cc_debug.png"
                cv2.imwrite(os.path.join(out_dbg, cc_dbg_name), cc_debug_img)
            except Exception as e:
                print(f"[CC DEBUG ERROR] Failed to save CC debug for {fname}: {e}")

        # Main debug visualization (all stages)
        if dbg_enabled:
            try:
                _ensure_dir(out_dbg)

                # Pass all stages to show the complete pipeline (4 stages: noise, cc, proj, border)
                dbg_img = _render_combined_debug(roi, gray, gray_cleaned, bin_before, bin_after,
                                                crop_before_border, crop_after_border,
                                                int(xl_proj), int(xr_proj), int(yt_proj), int(yb_proj),
                                                int(xl_border), int(xr_border), int(yt_border), int(yb_border))
                dbg_name = f"{os.path.splitext(fname)[0]}_debug.png"
                cv2.imwrite(os.path.join(out_dbg, dbg_name), dbg_img)

                # Generate border detection debug image (verbose version)
                if BORDER_REMOVAL_CONFIG.get('debug_verbose', False):
                    # Calculate coordinates relative to border_region (Proj output)
                    xl_border_rel = xl_border - xl_proj  # Border左切割相对于Proj输出的位置
                    xr_border_rel = xr_border - xl_proj  # Border右切割相对于Proj输出的位置
                    yt_border_rel = yt_border - yt_proj  # Border上切割相对于Proj输出的位置
                    yb_border_rel = yb_border - yt_proj  # Border下切割相对于Proj输出的位置

                    border_debug_img = _create_border_debug_image(
                        bin_after[yt_proj:yb_proj, xl_proj:xr_proj],
                        int(xl_border_rel), int(xr_border_rel),
                        0, int(xr_proj - xl_proj),  # Proj在border_region中的范围就是整个区域
                        BORDER_REMOVAL_CONFIG,
                        int(yt_border_rel), int(yb_border_rel)
                    )
                    border_dbg_name = f"{os.path.splitext(fname)[0]}_border_debug.png"
                    cv2.imwrite(os.path.join(out_dbg, border_dbg_name), border_debug_img)

            except Exception as e:
                # Log debug failure with character information
                error_msg = f"DEBUG FAILURE for {fname}: {type(e).__name__}: {str(e)}"
                print(f"[DEBUG ERROR] {error_msg}")
                # Also write error to a log file
                try:
                    error_log_path = os.path.join(output_dir, dbg_dirname, "debug_errors.log")
                    _ensure_dir(os.path.join(output_dir, dbg_dirname))
                    with open(error_log_path, "a", encoding="utf-8") as f:
                        f.write(f"{error_msg}\n")
                except:
                    pass
        chars_meta.append({
            'filename': fname,
            'bbox': (b['x'], b['y'], b['w'], b['h']),
            'refined_bbox': (rx, ry, rw, rh),
            'text_hint': b['text'],
            'confidence': b['confidence'],
            'normalized_bbox': b.get('normalized_bbox')
        })

    # Create comparison overlay: original clean image vs assembled processed characters
    overlay_original = img.copy()  # Clean original image without any annotations

    # Create processed image assembly: white background with processed characters placed back
    overlay_processed = np.full_like(img, 255, dtype=np.uint8)  # Start with white background

    # Place processed characters back in their original positions with overlap avoidance
    if chars_meta:
        # Track occupied regions to avoid overlap
        occupied_regions = []  # List of (x, y, w, h) tuples

        def check_overlap(x1, y1, w1, h1, x2, y2, w2, h2):
            """Check if two rectangles overlap"""
            return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)

        def find_non_overlapping_position(rx, ry, rw, rh, occupied):
            """Find a non-overlapping position by shifting vertically (unlimited, can extend beyond boundaries)"""
            # Try original position first
            has_overlap = False
            for occupied_rect in occupied:
                ox, oy, ow, oh = occupied_rect
                if check_overlap(rx, ry, rw, rh, ox, oy, ow, oh):
                    has_overlap = True
                    break

            if not has_overlap:
                return rx, ry  # Original position is fine

            # Try shifting down until no overlap (no boundary limit)
            new_y = ry + 1
            max_iterations = 10000  # Safety limit to prevent infinite loop
            iteration = 0
            while iteration < max_iterations:
                overlap_found = False
                for occupied_rect in occupied:
                    ox, oy, ow, oh = occupied_rect
                    if check_overlap(rx, new_y, rw, rh, ox, oy, ow, oh):
                        overlap_found = True
                        break
                if not overlap_found:
                    return rx, new_y
                new_y += 1
                iteration += 1

            # Try shifting up until no overlap (can go negative)
            new_y = ry - 1
            iteration = 0
            while iteration < max_iterations:
                overlap_found = False
                for occupied_rect in occupied:
                    ox, oy, ow, oh = occupied_rect
                    if check_overlap(rx, new_y, rw, rh, ox, oy, ow, oh):
                        overlap_found = True
                        break
                if not overlap_found:
                    return rx, new_y
                new_y -= 1
                iteration += 1

            # If still no space found (extremely unlikely), return original position
            return rx, ry

        # First pass: determine all final positions and required canvas size
        placements = []  # List of (crop_img, final_x, final_y, rw, rh)
        min_y = 0
        max_y = overlay_processed.shape[0]

        for idx, meta in enumerate(chars_meta):
            crop_path = os.path.join(output_dir, meta['filename'])
            if os.path.exists(crop_path):
                crop_img = cv2.imread(crop_path)
                if crop_img is not None:
                    # Use refined bbox for placement, but adjust for overlap
                    rx, ry, rw, rh = meta['refined_bbox']

                    # Find non-overlapping position
                    final_x, final_y = find_non_overlapping_position(rx, ry, rw, rh, occupied_regions)

                    # Resize crop to fit the refined bbox
                    if crop_img.shape[:2] != (rh, rw):
                        crop_img_resized = cv2.resize(crop_img, (rw, rh))
                    else:
                        crop_img_resized = crop_img

                    placements.append((crop_img_resized, final_x, final_y, rw, rh))

                    # Mark this region as occupied
                    occupied_regions.append((final_x, final_y, rw, rh))

                    # Track vertical extent
                    min_y = min(min_y, final_y)
                    max_y = max(max_y, final_y + rh)

        # Expand canvas if needed
        if min_y < 0 or max_y > overlay_processed.shape[0]:
            original_height = overlay_processed.shape[0]
            new_height = max_y - min_y
            y_offset = -min_y  # Offset to shift all positions to positive coordinates

            # Create expanded canvas
            overlay_processed_expanded = np.full((new_height, overlay_processed.shape[1], 3), 255, dtype=np.uint8)

            # Expand overlay_original to match new height
            overlay_original_expanded = np.full((new_height, overlay_original.shape[1], 3), 255, dtype=np.uint8)
            overlay_original_expanded[y_offset:y_offset + original_height, :] = overlay_original

            # Update references
            overlay_processed = overlay_processed_expanded
            overlay_original = overlay_original_expanded

            # Second pass: place characters with adjusted coordinates
            for crop_img_resized, final_x, final_y, rw, rh in placements:
                adjusted_y = final_y + y_offset
                overlay_processed[adjusted_y:adjusted_y + rh, final_x:final_x + rw] = crop_img_resized

                # Draw bounding box (darker green)
                cv2.rectangle(overlay_processed, (final_x, adjusted_y),
                            (final_x + rw - 1, adjusted_y + rh - 1), (0, 180, 0), 2)
        else:
            # No expansion needed, place directly
            for crop_img_resized, final_x, final_y, rw, rh in placements:
                overlay_processed[final_y:final_y + rh, final_x:final_x + rw] = crop_img_resized

                # Draw bounding box (darker green)
                cv2.rectangle(overlay_processed, (final_x, final_y),
                            (final_x + rw - 1, final_y + rh - 1), (0, 180, 0), 2)

    # Create side-by-side comparison
    gap_width = 20
    gap = np.full((overlay_original.shape[0], gap_width, 3), 255, dtype=np.uint8)
    comparison = np.hstack([overlay_original, gap, overlay_processed])

    overlay_path = os.path.join(output_dir, 'overlay.png')
    cv2.imwrite(overlay_path, comparison)

    stats = {
        'framework': 'livetext',
        'recognition_level': recognition_level,
        'char_candidates': len(boxes),
        'expected_text_len': len(expected_text) if expected_text else None,
        'image_size': [W, H],
        'refine_config': {
            'mode': mode,
            'expand_px': {
                'left': ex_left,
                'right': ex_right,
                'top': ex_top,
                'bottom': ex_bottom,
            },
            'final_pad': int(SEGMENT_REFINE_CONFIG.get('final_pad', 0)),
        },
        'projection_trim': PROJECTION_TRIM_CONFIG,
    }
    return {
        'success': True,
        'character_count': len(chars_meta),
        'characters': chars_meta,
        'stats': stats,
        'overlay': overlay_path,
    }


def run_on_ocr_regions(dataset: str | None = None,
                       expected_texts: Dict[str, str] | None = None,
                       framework: str = 'livetext', recognition_level: str = 'accurate',
                       language_preference: Optional[List[str]] = None) -> Dict[str, Any]:
    def _find_region_images(base_ocr_dir: str, dataset: str | None = None) -> List[Tuple[str, str, str]]:
        results: List[Tuple[str, str, str]] = []
        if dataset:
            datasets = [dataset]
        else:
            try:
                datasets = [d for d in os.listdir(base_ocr_dir) if os.path.isdir(os.path.join(base_ocr_dir, d))]
            except FileNotFoundError:
                datasets = []
        for ds in datasets:
            region_dir = os.path.join(base_ocr_dir, ds, 'region_images')
            if not os.path.isdir(region_dir):
                continue
            try:
                for name in os.listdir(region_dir):
                    if not (name.lower().endswith('.jpg') or name.lower().endswith('.png')):
                        continue
                    if not name.startswith('region_'):
                        continue
                    img_path = os.path.join(region_dir, name)
                    region_name = os.path.splitext(name)[0]
                    results.append((ds, region_name, img_path))
            except FileNotFoundError:
                continue
        results.sort(key=lambda t: (t[0], t[1]))
        return results

    items = _find_region_images(PREOCR_DIR, dataset=dataset)
    processed = []
    errors = []
    for ds, region_name, img_path in items:
        out_dir = os.path.join(SEGMENTS_DIR, ds, region_name)
        os.makedirs(out_dir, exist_ok=True)
        exp_text = None
        if expected_texts:
            exp_text = expected_texts.get(region_name) or expected_texts.get(f"{ds}:{region_name}")
        try:
            res = run_on_image(img_path, out_dir, expected_text=exp_text, framework=framework,
                               recognition_level=recognition_level, language_preference=language_preference)
            with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
            processed.append({'dataset': ds, 'region': region_name, 'out_dir': out_dir,
                              'count': res.get('character_count', 0)})
        except Exception as e:
            errors.append({'dataset': ds, 'region': region_name, 'error': str(e)})
    return {'processed': processed, 'errors': errors}
