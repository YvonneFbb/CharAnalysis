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
from .config import (
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

from .projection_trim import trim_projection_from_bin, binarize
from .cc_filter import refine_binary_components
from .noise_removal import remove_noise_patches
from .border_removal import trim_border_from_bin, _debug_border_detection


def _ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def _create_border_projection_viz(border_region: np.ndarray, params: Dict, max_height: int = 120,
                                   xl_cut: int = 0, xr_cut: int = 0, target_width: int = 400) -> np.ndarray:
    """
    Create border projection visualization that can be embedded in main debug image.

    Args:
        border_region: Binary region after projection trimming
        params: Border removal configuration
        max_height: Maximum height for the visualization
        xl_cut: Left border cut position (relative to border_region)
        xr_cut: Right border cut position (relative to border_region)
        target_width: Target width for the visualization

    Returns:
        Visualization image showing projection, detection zones, and cut positions
    """
    if border_region.size == 0:
        return np.full((max_height, target_width, 3), 255, dtype=np.uint8)

    # Calculate projections
    border_hproj = border_region.sum(axis=0).astype(np.float32)
    if border_hproj.size == 0 or border_hproj.max() == 0:
        return np.full((max_height, target_width, 3), 255, dtype=np.uint8)

    bhp_coverage = border_hproj / (border_region.shape[0] * 255.0)
    max_coverage = bhp_coverage.max()

    # Get parameters
    max_width = int(border_region.shape[1] * params.get('border_max_width_ratio', 0.2))
    border_threshold = max_coverage * params.get('border_threshold_ratio', 0.5)

    # Create visualization
    viz_width = target_width  # Use the specified target width
    viz_height = max_height
    viz_img = np.full((viz_height, viz_width, 3), 255, dtype=np.uint8)

    # Scale projection to fit (optimized to use more space)
    if len(bhp_coverage) > 0:
        scale_x = viz_width / len(bhp_coverage)
        proj_height = int(viz_height * 0.95)  # Further increased to use 95% of space

        # Draw projection bars
        for i, val in enumerate(bhp_coverage):
            x = int(i * scale_x)
            bar_height = int(val / max_coverage * proj_height) if max_coverage > 0 else 0
            y_start = viz_height - 3 - bar_height  # Minimal bottom margin
            y_end = viz_height - 3

            if bar_height > 0:
                cv2.rectangle(viz_img, (x, y_start), (min(x+int(scale_x)+1, viz_width-1), y_end), (128, 128, 128), -1)

        # Draw detection zones (optimized positioning)
        left_zone_end = int(max_width * scale_x)
        right_zone_start = viz_width - int(max_width * scale_x)

        # Left zone
        cv2.rectangle(viz_img, (0, 2), (left_zone_end, 18), (255, 0, 0), 2)
        cv2.putText(viz_img, "LEFT ZONE", (2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)

        # Right zone
        if right_zone_start > left_zone_end:
            cv2.rectangle(viz_img, (right_zone_start, 2), (viz_width-1, 18), (255, 0, 0), 2)
            cv2.putText(viz_img, "RIGHT ZONE", (right_zone_start+2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)

        # Draw threshold line
        threshold_y = viz_height - 8 - int(border_threshold / max_coverage * proj_height) if max_coverage > 0 else viz_height - 8
        cv2.line(viz_img, (0, threshold_y), (viz_width-1, threshold_y), (0, 255, 0), 2)
        cv2.putText(viz_img, f"Threshold: {border_threshold:.3f}", (2, threshold_y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

        # Draw actual border cut positions
        if xl_cut > 0:
            xl_viz = int(xl_cut * scale_x)
            cv2.line(viz_img, (xl_viz, 22), (xl_viz, viz_height-5), (0, 0, 255), 2)
            cv2.putText(viz_img, f"L:{xl_cut}", (xl_viz+1, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 0, 255), 1)

        if xr_cut < border_region.shape[1]:
            xr_viz = int(xr_cut * scale_x)
            cv2.line(viz_img, (xr_viz, 22), (xr_viz, viz_height-5), (0, 0, 255), 2)
            cv2.putText(viz_img, f"R:{xr_cut}", (max(0, xr_viz-20), 32), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 0, 255), 1)

    return viz_img


def _create_border_debug_image(border_region: np.ndarray,
                              xl_border: int, xr_border: int,
                              xl_proj: int, xr_proj: int,
                              params: Dict,
                              yt_border: int = 0, yb_border: int = 0) -> np.ndarray:
    """
    Create a comprehensive border detection debug image with both horizontal and vertical analysis.
    """
    if border_region.size == 0:
        return np.full((800, 1200, 3), 255, dtype=np.uint8)

    debug_width = 1200
    debug_height = 800
    debug_img = np.full((debug_height, debug_width, 3), 255, dtype=np.uint8)

    h, w = border_region.shape

    # === Title ===
    cv2.putText(debug_img, "BORDER DETECTION ANALYSIS", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    # Draw separator line under title
    cv2.line(debug_img, (10, 50), (debug_width - 10, 50), (200, 200, 200), 2)

    # === Left Half: Horizontal Analysis ===
    left_w = debug_width // 2 - 40
    _draw_horizontal_border_analysis(debug_img, border_region, xl_border, xr_border, params,
                                   start_x=20, start_y=80, width=left_w, height=320)

    # === Right Half: Vertical Analysis ===
    right_start_x = debug_width // 2 + 20
    _draw_vertical_border_analysis(debug_img, border_region, yt_border, yb_border, params,
                                 start_x=right_start_x, start_y=80, width=left_w, height=320)

    return debug_img


def _draw_horizontal_border_analysis(debug_img: np.ndarray, border_region: np.ndarray,
                                   xl_border: int, xr_border: int, params: Dict,
                                   start_x: int, start_y: int, width: int, height: int):
    """Draw horizontal border detection analysis with clear layout."""
    h, w = border_region.shape

    # Title
    cv2.putText(debug_img, "HORIZONTAL ANALYSIS", (start_x, start_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Calculate projection
    hproj = border_region.sum(axis=0).astype(np.float32) / (h * 255.0)
    max_cov = hproj.max() if hproj.size > 0 else 0.0

    # Projection visualization area (optimized height to use more space)
    proj_height = 200  # Increased from 150 to 200
    proj_area_y = start_y + 10  # Moved up to use more space
    scale_x = (width - 30) / max(1, len(hproj))  # Reduced margin

    # Draw background for projection area
    cv2.rectangle(debug_img, (start_x + 15, proj_area_y),
                  (start_x + width - 15, proj_area_y + proj_height), (245, 245, 245), -1)

    # Draw projection bars
    for i, cov in enumerate(hproj):
        x = start_x + 15 + int(i * scale_x)
        bar_h = int(cov / max(max_cov, 1e-6) * (proj_height - 15))  # More space for bars
        y_start = proj_area_y + proj_height - 10 - bar_h
        cv2.rectangle(debug_img, (x, y_start),
                     (x + max(1, int(scale_x)), proj_area_y + proj_height - 10), (80, 80, 80), -1)

    # Detection zones with clear visualization
    max_width = int(w * params.get('border_max_width_ratio', 0.2))
    left_zone_end = start_x + 15 + int(max_width * scale_x)
    right_zone_start = start_x + 15 + int((w - max_width) * scale_x)

    # Draw detection zones with transparency effect
    cv2.rectangle(debug_img, (start_x + 15, proj_area_y),
                  (left_zone_end, proj_area_y + proj_height), (0, 0, 255), 2)
    cv2.rectangle(debug_img, (right_zone_start, proj_area_y),
                  (start_x + width - 15, proj_area_y + proj_height), (255, 0, 0), 2)

    # Zone labels without background
    cv2.putText(debug_img, "LEFT", (start_x + 25, proj_area_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.putText(debug_img, "RIGHT", (right_zone_start + 5, proj_area_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    # Threshold line with clear label
    threshold = max_cov * params.get('border_threshold_ratio', 0.5)
    threshold_y = proj_area_y + proj_height - 10 - int(threshold / max(max_cov, 1e-6) * (proj_height - 15))
    cv2.line(debug_img, (start_x + 15, threshold_y), (start_x + width - 15, threshold_y), (0, 255, 0), 2)

    # Threshold label without background
    thresh_text = f"Threshold: {threshold:.3f}"
    cv2.putText(debug_img, thresh_text, (start_x + 5, threshold_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Cut markers without background
    if xl_border > 0:
        cut_x = start_x + 15 + int(xl_border * scale_x)
        cv2.line(debug_img, (cut_x, proj_area_y), (cut_x, proj_area_y + proj_height), (255, 0, 255), 3)
        cv2.putText(debug_img, f"L:{xl_border}", (cut_x + 2, proj_area_y + proj_height + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

    if xr_border < w:
        cut_x = start_x + 15 + int(xr_border * scale_x)
        cv2.line(debug_img, (cut_x, proj_area_y), (cut_x, proj_area_y + proj_height), (255, 0, 255), 3)
        cv2.putText(debug_img, f"R:{w-xr_border}", (cut_x - 25, proj_area_y + proj_height + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

    # Parameters section with clear separation (adjusted for smaller space)
    info_y = proj_area_y + proj_height + 30
    cv2.line(debug_img, (start_x, info_y - 5), (start_x + width, info_y - 5), (200, 200, 200), 1)

    cv2.putText(debug_img, "PARAMETERS:", (start_x, info_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
    cv2.putText(debug_img, f"Width: {w}px | Coverage: {max_cov:.3f}",
                (start_x, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    cv2.putText(debug_img, f"Zone: {max_width}px ({params.get('border_max_width_ratio', 0.2)*100:.0f}%) | Thresh: {params.get('border_threshold_ratio', 0.5)*100:.0f}%",
                (start_x, info_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)


def _draw_vertical_border_analysis(debug_img: np.ndarray, border_region: np.ndarray,
                                 yt_border: int, yb_border: int, params: Dict,
                                 start_x: int, start_y: int, width: int, height: int):
    """Draw vertical border detection analysis with clear layout."""
    h, w = border_region.shape

    # Title
    cv2.putText(debug_img, "VERTICAL ANALYSIS", (start_x, start_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Calculate vertical projection
    vproj = border_region.sum(axis=1).astype(np.float32) / (w * 255.0)
    max_cov = vproj.max() if vproj.size > 0 else 0.0

    # Projection visualization area (optimized width to use more space)
    proj_width = 220  # Increased from 150 to 220
    proj_area_x = start_x + 10  # Moved left to use more space
    scale_y = (height - 60) / max(1, len(vproj))  # Reduced bottom margin

    # Draw background for projection area
    cv2.rectangle(debug_img, (proj_area_x, start_y + 10),
                  (proj_area_x + proj_width, start_y + height - 30), (245, 245, 245), -1)

    # Draw projection bars (horizontal bars representing vertical projection)
    for i, cov in enumerate(vproj):
        y = start_y + 10 + int(i * scale_y)
        bar_w = int(cov / max(max_cov, 1e-6) * (proj_width - 15))  # More space for bars
        cv2.rectangle(debug_img, (proj_area_x + 10, y),
                     (proj_area_x + 10 + bar_w, y + max(1, int(scale_y))), (80, 80, 80), -1)

    # Get vertical detection parameters
    v_detection = params.get('vertical_detection_range', {})
    v_cuts = params.get('vertical_cut_limits', {})
    top_detection_ratio = v_detection.get('top_ratio', 0.3)
    bottom_detection_ratio = v_detection.get('bottom_ratio', 0.3)

    # Detection zones with correct positioning
    # Top zone: from top of the image, height = top_detection_ratio * h
    top_zone_height = int(h * top_detection_ratio * scale_y)
    top_zone_start = start_y + 10
    top_zone_end = top_zone_start + top_zone_height

    # Bottom zone: from bottom of the image upward, height = bottom_detection_ratio * h
    bottom_zone_height = int(h * bottom_detection_ratio * scale_y)
    bottom_zone_end = start_y + 10 + int(h * scale_y)  # Bottom of the projection area
    bottom_zone_start = bottom_zone_end - bottom_zone_height

    # Draw detection zones
    cv2.rectangle(debug_img, (proj_area_x, top_zone_start),
                  (proj_area_x + proj_width, top_zone_end), (0, 255, 255), 2)
    cv2.rectangle(debug_img, (proj_area_x, bottom_zone_start),
                  (proj_area_x + proj_width, bottom_zone_end), (255, 200, 0), 2)

    # Zone labels without background
    cv2.putText(debug_img, "TOP", (proj_area_x + 5, start_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.putText(debug_img, "BOTTOM", (proj_area_x + 5, bottom_zone_start + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)

    # Cut markers without background
    if yt_border > 0:
        cut_y = start_y + 10 + int(yt_border * scale_y)
        cv2.line(debug_img, (proj_area_x, cut_y), (proj_area_x + proj_width, cut_y), (255, 0, 255), 3)
        cv2.putText(debug_img, f"T:{yt_border}", (proj_area_x + proj_width + 5, cut_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

    if yb_border < h:
        cut_y = start_y + 10 + int(yb_border * scale_y)
        cv2.line(debug_img, (proj_area_x, cut_y), (proj_area_x + proj_width, cut_y), (255, 0, 255), 3)
        cv2.putText(debug_img, f"B:{h-yb_border}", (proj_area_x + proj_width + 5, cut_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

    # Parameters section with clear separation (adjusted for smaller space)
    info_y = start_y + height - 30
    cv2.line(debug_img, (start_x, info_y - 5), (start_x + width, info_y - 5), (200, 200, 200), 1)

    cv2.putText(debug_img, "PARAMETERS:", (start_x, info_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
    cv2.putText(debug_img, f"Height: {h}px | Coverage: {max_cov:.3f}",
                (start_x, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    cv2.putText(debug_img, f"Top: {top_detection_ratio*100:.0f}% | Bottom: {bottom_detection_ratio*100:.0f}%",
                (start_x, info_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)


def _render_combined_debug(roi_bgr: np.ndarray,
                           gray_original: np.ndarray,
                           gray_cleaned: np.ndarray,
                           bin_original: np.ndarray,
                           bin_after: np.ndarray,
                           crop_before_border: np.ndarray,
                           crop_after_border: np.ndarray,
                           xl_proj: int, xr_proj: int, yt_proj: int, yb_proj: int,
                           xl_border: int, xr_border: int, yt_border: int, yb_border: int) -> np.ndarray:
    """Render 1x4 panel: NOISE + CC + PROJ + BORDER rows."""
    h, w = roi_bgr.shape[:2]
    if h == 0 or w == 0:
        return roi_bgr.copy()

    mask_original = (bin_original > 0).astype(np.uint8)
    mask_after = (bin_after > 0).astype(np.uint8)

    panel_h = int(max(120, min(260, round(h * 0.8))))
    scale = panel_h / float(max(1, h))
    base_w = max(1, int(round(w * scale)))

    def resize_panel(img: np.ndarray, interp: int = cv2.INTER_LINEAR, target_w: int | None = None) -> np.ndarray:
        if target_w is None:
            target_w = base_w
        return cv2.resize(img, (target_w, panel_h), interpolation=interp)

    def to_bgr(mask: np.ndarray) -> np.ndarray:
        vis = 255 - (mask.astype(np.uint8) * 255)
        return cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    def add_border(img: np.ndarray, pad: int = 2) -> np.ndarray:
        return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    # === Original 2x4 layout (CC vs Projection) ===
    # CC overlay and masks
    masked_roi = np.full_like(roi_bgr, 255)
    masked_roi[mask_after.astype(bool)] = roi_bgr[mask_after.astype(bool)]
    cc_overlay = masked_roi.copy()
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_original, connectivity=8)
    if num > 1:
        for i in range(1, num):
            x, y, ww, hh, _area = stats[i]
            kept = bool(mask_after[labels == i].any())
            color = (0, 200, 0) if kept else (0, 0, 255)
            cv2.rectangle(cc_overlay, (int(x), int(y)), (int(x + ww - 1), int(y + hh - 1)), color, 1)

    cc_overlay_panel = resize_panel(cc_overlay)
    mask_before_panel = resize_panel(to_bgr(mask_original), interp=cv2.INTER_NEAREST)
    mask_after_panel = resize_panel(to_bgr(mask_after), interp=cv2.INTER_NEAREST)

    # Projection overlay with boundary lines
    projection_overlay = masked_roi.copy()
    cv2.line(projection_overlay, (max(0, int(xl_proj)), 0), (max(0, int(xl_proj)), h - 1), (0, 0, 255), 1)
    cv2.line(projection_overlay, (max(0, int(xr_proj - 1)), 0), (max(0, int(xr_proj - 1)), h - 1), (0, 0, 255), 1)
    cv2.line(projection_overlay, (0, max(0, int(yt_proj))), (w - 1, max(0, int(yt_proj))), (0, 0, 255), 1)
    cv2.line(projection_overlay, (0, max(0, int(yb_proj - 1))), (w - 1, max(0, int(yb_proj - 1))), (0, 0, 255), 1)
    projection_panel = resize_panel(projection_overlay)

    # Projection histograms with improved visibility
    hist_w = base_w

    # Vertical projection panel with increased bar height
    v_panel = np.full((panel_h, hist_w), 255, dtype=np.uint8)
    vproj = mask_after.sum(axis=0).astype(np.float32)
    if vproj.size:
        vp = vproj / (vproj.max() + 1e-6)
        # Use 80% of panel height for better visibility
        max_bar_height = int(panel_h * 0.8)
        for col in range(hist_w):
            sx = int(round(col / max(1, hist_w - 1) * max(0, w - 1)))
            bar = int(round(vp[sx] * max_bar_height))
            if bar > 0:
                v_panel[panel_h - bar:panel_h, col] = 0
        xl_bar = int(round(max(0, xl_proj) / max(1, w - 1) * max(0, hist_w - 1)))
        xr_bar = int(round(max(0, xr_proj - 1) / max(1, w - 1) * max(0, hist_w - 1)))
        cv2.line(v_panel, (xl_bar, 0), (xl_bar, panel_h - 1), 128, 2)
        cv2.line(v_panel, (xr_bar, 0), (xr_bar, panel_h - 1), 128, 2)
    vertical_panel = cv2.cvtColor(v_panel, cv2.COLOR_GRAY2BGR)

    # Horizontal projection panel with increased bar width
    h_panel = np.full((panel_h, hist_w), 255, dtype=np.uint8)
    hproj = mask_after.sum(axis=1).astype(np.float32)
    if hproj.size:
        hp = hproj / (hproj.max() + 1e-6)
        # Use 80% of panel width for better visibility
        max_bar_width = int(hist_w * 0.8)
        for row in range(panel_h):
            sy = int(round(row / max(1, panel_h - 1) * max(0, h - 1)))
            bar = int(round(hp[sy] * max_bar_width))
            if bar > 0:
                h_panel[row, :bar] = 0
        yt_bar = int(round(max(0, yt_proj) / max(1, h - 1) * max(0, panel_h - 1)))
        yb_bar = int(round(max(0, yb_proj - 1) / max(1, h - 1) * max(0, panel_h - 1)))
        cv2.line(h_panel, (0, yt_bar), (max_bar_width, yt_bar), 128, 2)
        cv2.line(h_panel, (0, yb_bar), (max_bar_width, yb_bar), 128, 2)
    horizontal_panel = cv2.cvtColor(h_panel, cv2.COLOR_GRAY2BGR)

    # === Border removal row ===
    # Show Border input (Proj output) and Border output overlaid
    # Create a visualization showing the progression: Proj output -> Border input -> Border output
    border_overlay = np.full_like(roi_bgr, 255)  # Start with white background

    # Show Border's input region (Proj output) in gray
    proj_region_mask = np.zeros_like(bin_after)
    if xl_proj < xr_proj and yt_proj < yb_proj:
        proj_region_mask[yt_proj:yb_proj, xl_proj:xr_proj] = bin_after[yt_proj:yb_proj, xl_proj:xr_proj]
        border_overlay[proj_region_mask.astype(bool)] = roi_bgr[proj_region_mask.astype(bool)]

    # Draw Border boundaries only (no Proj boundaries in Border stage) - blue to distinguish from Proj
    cv2.line(border_overlay, (max(0, int(xl_border)), 0), (max(0, int(xl_border)), h - 1), (255, 0, 0), 1)
    cv2.line(border_overlay, (max(0, int(xr_border - 1)), 0), (max(0, int(xr_border - 1)), h - 1), (255, 0, 0), 1)
    cv2.line(border_overlay, (0, max(0, int(yt_border))), (w - 1, max(0, int(yt_border))), (255, 0, 0), 1)
    cv2.line(border_overlay, (0, max(0, int(yb_border - 1))), (w - 1, max(0, int(yb_border - 1))), (255, 0, 0), 1)

    border_panel = resize_panel(border_overlay)

    # Border horizontal projection analysis with detailed visualization
    # Use the actual Border input (Proj output region)
    border_region = bin_after[yt_proj:yb_proj, xl_proj:xr_proj] if xl_proj < xr_proj and yt_proj < yb_proj else np.zeros((1, 1), dtype=np.uint8)

    # Create comprehensive border projection visualization
    border_h_panel_bgr = _create_border_projection_viz(
        border_region,
        BORDER_REMOVAL_CONFIG,
        max_height=panel_h,
        xl_cut=xl_border - xl_proj,
        xr_cut=xr_border - xl_proj,
        target_width=hist_w
    )

    # Show final crop result with aspect ratio preserved
    if crop_after_border is not None and crop_after_border.size > 0:
        # Calculate aspect ratio preserving scale
        crop_h, crop_w = crop_after_border.shape[:2]
        if crop_h > 0 and crop_w > 0:
            scale_factor = min(panel_h / crop_h, base_w / crop_w)
            new_h = int(crop_h * scale_factor)
            new_w = int(crop_w * scale_factor)
            crop_resized = cv2.resize(crop_after_border, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Center the image in a panel_h x base_w canvas
            crop_after_panel = np.full((panel_h, base_w, 3), 255, dtype=np.uint8)
            y_offset = (panel_h - new_h) // 2
            x_offset = (base_w - new_w) // 2
            crop_after_panel[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = crop_resized
        else:
            crop_after_panel = np.full((panel_h, base_w, 3), 250, dtype=np.uint8)
    else:
        crop_after_panel = np.full((panel_h, base_w, 3), 250, dtype=np.uint8)

    # === Noise removal row ===
    # Show noise cleaning effect by highlighting differences
    noise_diff = cv2.absdiff(gray_original, gray_cleaned)
    noise_overlay = cv2.cvtColor(gray_cleaned, cv2.COLOR_GRAY2BGR)

    # Highlight removed noise in red
    noise_mask = (noise_diff > 5).astype(np.uint8)  # Threshold for visible differences
    noise_overlay[noise_mask > 0] = [0, 0, 255]  # Red for removed noise

    noise_overlay_panel = resize_panel(noise_overlay)
    gray_original_panel = resize_panel(cv2.cvtColor(gray_original, cv2.COLOR_GRAY2BGR))
    gray_cleaned_panel = resize_panel(cv2.cvtColor(gray_cleaned, cv2.COLOR_GRAY2BGR))


    # Labels
    label_w = max(60, hist_w // 4)
    label_color = (245, 245, 245)
    cc_label = np.full((panel_h, label_w, 3), label_color, dtype=np.uint8)
    proj_label = np.full((panel_h, label_w, 3), label_color, dtype=np.uint8)
    noise_label = np.full((panel_h, label_w, 3), label_color, dtype=np.uint8)
    border_label = np.full((panel_h, label_w, 3), label_color, dtype=np.uint8)
    cv2.putText(cc_label, 'CC', (10, panel_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(proj_label, 'PROJ', (10, panel_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(noise_label, 'NOISE', (5, panel_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(border_label, 'BORDER', (2, panel_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 2, cv2.LINE_AA)

    def pad_to_width(img: np.ndarray, target_w: int) -> np.ndarray:
        h_, w_ = img.shape[:2]
        if w_ >= target_w:
            return img
        pad_left = (target_w - w_) // 2
        pad_right = target_w - w_ - pad_left
        return cv2.copyMakeBorder(img, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    # Build 4 rows: NOISE, CC, PROJ, BORDER
    try:
        # Try to create each border panel component individually for better error reporting
        try:
            border_panel_bordered = add_border(border_panel)
        except Exception as e:
            raise RuntimeError(f"border_panel creation failed: {e}")

        try:
            border_h_panel_bordered = add_border(border_h_panel_bgr)
        except Exception as e:
            raise RuntimeError(f"border_h_panel_bgr creation failed: {e}")

        try:
            crop_after_panel_bordered = add_border(crop_after_panel)
        except Exception as e:
            raise RuntimeError(f"crop_after_panel creation failed: {e}")

        try:
            border_label_bordered = add_border(border_label)
        except Exception as e:
            raise RuntimeError(f"border_label creation failed: {e}")

        border_row = [border_panel_bordered, border_h_panel_bordered, crop_after_panel_bordered, border_label_bordered]

        rows = [
            [add_border(noise_overlay_panel), add_border(gray_original_panel), add_border(gray_cleaned_panel), add_border(noise_label)],
            [add_border(cc_overlay_panel), add_border(mask_before_panel), add_border(mask_after_panel), add_border(cc_label)],
            [add_border(projection_panel), add_border(vertical_panel), add_border(horizontal_panel), add_border(proj_label)],
            border_row
        ]
    except Exception as e:
        # Detailed error reporting for border row creation failure
        print(f"[BORDER ROW ERROR] {type(e).__name__}: {str(e)}")
        # Fallback to 3 rows if border row fails
        rows = [
            [add_border(noise_overlay_panel), add_border(gray_original_panel), add_border(gray_cleaned_panel), add_border(noise_label)],
            [add_border(cc_overlay_panel), add_border(mask_before_panel), add_border(mask_after_panel), add_border(cc_label)],
            [add_border(projection_panel), add_border(vertical_panel), add_border(horizontal_panel), add_border(proj_label)]
        ]

    grid_rows = []
    for row_panels in rows:
        # Pad all panels in this row to same width
        max_w = max(panel.shape[1] for panel in row_panels)
        padded_panels = [pad_to_width(panel, max_w) for panel in row_panels]

        # Combine with gaps
        gap = 6
        gap_col = np.full((padded_panels[0].shape[0], gap, 3), 255, dtype=np.uint8)

        row_img = padded_panels[0]
        for panel in padded_panels[1:]:
            row_img = np.hstack([row_img, gap_col, panel])
        grid_rows.append(row_img)

    try:
        result = np.vstack(grid_rows)
        return result
    except Exception as e:
        # Detailed error reporting for vstack failure
        print(f"[VSTACK ERROR] Failed to combine {len(grid_rows)} rows: {type(e).__name__}: {str(e)}")
        if len(grid_rows) >= 4:
            print(f"[VSTACK ERROR] Row shapes: {[row.shape for row in grid_rows]}")
        # Return first 3 rows if vstack fails
        try:
            return np.vstack(grid_rows[:3])
        except Exception as e2:
            print(f"[VSTACK ERROR] Even 3-row fallback failed: {type(e2).__name__}: {str(e2)}")
            # Last resort: return a blank image
            return np.full((400, 800, 3), 255, dtype=np.uint8)

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
