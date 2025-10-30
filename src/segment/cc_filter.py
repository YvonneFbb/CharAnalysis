"""
Connected-component pre-filter for segmentation.

Goal: remove clearly abnormal regions before projection trimming, with
minimal parameters and conservative rules:
  - tiny components relative to ROI area and touching the image border directly
  - tiny components relative to ROI area and within an edge margin band
  - edge-touching, extremely slender components (very high aspect ratio)
  - edge-touching components with min dimension below a small pixel threshold

Operates in-place on a binary (uint8) image with foreground=255.
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2


def refine_binary_components(bin_img: np.ndarray, params: Dict,
                             gray_img: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Remove abnormal connected components from a binary image based on 3-tier classification:
    1. Border touching (actual contact): components touching the exact image border
    2. Edge zone (margin area): components within edge margin but not touching border
    3. Interior (other): components not in edge zone

    Each tier has its own area ratio threshold for removal.

    Parameters:
      - border_touch_margin: pixel range for border touching detection (default: 0)
      - edge_zone_margin: pixel range for edge zone detection (default: 6)
      - border_touch_min_area_ratio: area ratio threshold for border touching components
      - edge_zone_min_area_ratio: area ratio threshold for edge zone components
      - interior_min_area_ratio: area ratio threshold for interior components
      - max_aspect_for_edge: max aspect ratio allowed for edge/border components
      - min_dim_px: min dimension allowed for edge/border components
      - debug_visualize: whether to generate detailed debug visualization
      - gray_img: optional grayscale image for debug visualization

    Returns:
      - bin_img: filtered binary image
      - debug_img: debug visualization image (if debug_visualize=True), otherwise None
    """
    if bin_img is None or bin_img.size == 0:
        return bin_img, None
    h, w = bin_img.shape[:2]
    if h == 0 or w == 0:
        return bin_img, None

    # Parameters for 3-tier classification
    border_touch_margin = int(params.get('border_touch_margin', 0))           # 实际触边范围
    edge_zone_margin = int(params.get('edge_zone_margin', 6))                 # 边缘区域范围
    border_touch_min_area_ratio = float(params.get('border_touch_min_area_ratio', 0.001))  # 触边组件面积阈值
    edge_zone_min_area_ratio = float(params.get('edge_zone_min_area_ratio', 0.002))        # 边缘组件面积阈值
    interior_min_area_ratio = float(params.get('interior_min_area_ratio', 0.0005))         # 内部组件面积阈值

    # Shape constraints for edge/border components
    max_aspect_for_edge = float(params.get('max_aspect_for_edge', 10.0))
    min_dim_px = int(params.get('min_dim_px', 2))
    interior_min_dim_px = int(params.get('interior_min_dim_px', 2))

    # Debug mode
    debug_enabled = bool(params.get('debug_visualize', False))

    # Calculate area thresholds
    total_area = h * w
    border_area_thr = max(1, int(border_touch_min_area_ratio * total_area))
    edge_area_thr = max(1, int(edge_zone_min_area_ratio * total_area))
    interior_area_thr = max(1, int(interior_min_area_ratio * total_area))

    m = (bin_img > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return bin_img, None

    # Collect debug info for all components
    debug_info = []

    for i in range(1, num):
        x, y, ww, hh, area = stats[i]

        # Calculate minimum area rectangle for rotated bounding box
        component_mask = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Get minimum area rectangle (rotated)
            rect = cv2.minAreaRect(contours[0])
            # rect = ((center_x, center_y), (width, height), angle)
            # width and height are already ordered, but we need min/max
            rect_width, rect_height = rect[1]
            min_rect_dim = min(rect_width, rect_height)
            max_rect_dim = max(rect_width, rect_height)
        else:
            # Fallback to axis-aligned box
            min_rect_dim = min(ww, hh)
            max_rect_dim = max(ww, hh)

        # Classify component based on position
        component_type, should_remove, reason = _classify_and_filter_component(
            x, y, ww, hh, area, w, h,
            border_touch_margin, edge_zone_margin,
            border_area_thr, edge_area_thr, interior_area_thr,
            max_aspect_for_edge, min_dim_px, interior_min_dim_px,
            min_rect_dim, max_rect_dim
        )

        if should_remove:
            bin_img[labels == i] = 0

        # Collect debug info
        if debug_enabled:
            aspect_rect = max_rect_dim / max(1.0, min_rect_dim)
            debug_info.append({
                'index': i,
                'bbox': (x, y, ww, hh),
                'area': area,
                'area_ratio': area / total_area,
                'min_rect_dim': min_rect_dim,
                'max_rect_dim': max_rect_dim,
                'aspect_ratio': aspect_rect,
                'component_type': component_type,
                'removed': should_remove,
                'reason': reason,
            })

    # Generate debug visualization
    debug_img = None
    if debug_enabled and debug_info and gray_img is not None:
        debug_img = _create_cc_debug_visualization(
            gray_img, bin_img, labels, debug_info, params,
            border_area_thr, edge_area_thr, interior_area_thr,
            total_area
        )

    return bin_img, debug_img


def _classify_and_filter_component(x: int, y: int, ww: int, hh: int, area: int,
                                   img_w: int, img_h: int,
                                   border_margin: int, edge_margin: int,
                                   border_thr: int, edge_thr: int, interior_thr: int,
                                   max_aspect: float, min_dim: int, interior_min_dim: int,
                                   min_rect_dim: float, max_rect_dim: float) -> tuple[str, bool, str]:
    """
    Classify component into border-touching, edge-zone, or interior, and decide if it should be removed.

    Args:
        min_rect_dim: Minimum dimension of the minimum area rectangle (rotated box)
        max_rect_dim: Maximum dimension of the minimum area rectangle (rotated box)

    Returns:
        (component_type, should_remove, reason): component classification, removal decision, and reason
    """

    # Check if component touches actual border (within border_margin)
    touches_border = (
        x <= border_margin or y <= border_margin or
        (x + ww) >= (img_w - border_margin) or (y + hh) >= (img_h - border_margin)
    )

    if touches_border:
        component_type = "border_touch"
        # Apply border touching rules
        if area < border_thr:
            return component_type, True, f"area={area} < thr={border_thr}"

        # Shape constraints for border components using minimum area rectangle
        aspect_rect = max_rect_dim / max(1.0, min_rect_dim)
        if aspect_rect >= max_aspect:
            return component_type, True, f"aspect={aspect_rect:.1f} >= max={max_aspect}"
        if min_rect_dim <= min_dim:
            return component_type, True, f"min_dim={min_rect_dim:.1f} <= thr={min_dim}"

        return component_type, False, "kept"

    # Check if component is in edge zone (within edge_margin but not touching border)
    in_edge_zone = (
        x <= edge_margin or y <= edge_margin or
        (x + ww) >= (img_w - edge_margin) or (y + hh) >= (img_h - edge_margin)
    )

    if in_edge_zone:
        component_type = "edge_zone"
        # Apply edge zone rules
        if area < edge_thr:
            return component_type, True, f"area={area} < thr={edge_thr}"

        # Shape constraints for edge components using minimum area rectangle
        aspect_rect = max_rect_dim / max(1.0, min_rect_dim)
        if aspect_rect >= max_aspect:
            return component_type, True, f"aspect={aspect_rect:.1f} >= max={max_aspect}"
        if min_rect_dim <= min_dim:
            return component_type, True, f"min_dim={min_rect_dim:.1f} <= thr={min_dim}"

        return component_type, False, "kept"

    # Interior component
    component_type = "interior"
    # Apply interior rules (usually more lenient)
    if area < interior_thr:
        return component_type, True, f"area={area} < thr={interior_thr}"

    # Shape constraints for interior components using minimum area rectangle
    if min_rect_dim <= interior_min_dim:
        return component_type, True, f"min_dim={min_rect_dim:.1f} <= thr={interior_min_dim}"

    return component_type, False, "kept"


def _create_cc_debug_visualization(gray_img: np.ndarray, bin_filtered: np.ndarray,
                                   labels: np.ndarray, debug_info: List[Dict],
                                   params: Dict, border_thr: int, edge_thr: int,
                                   interior_thr: int, total_area: int) -> np.ndarray:
    """
    Create detailed CC filter debug visualization showing top candidates by area.
    """
    # Sort by area (largest first) and take top 10
    sorted_info = sorted(debug_info, key=lambda x: x['area'], reverse=True)
    top_n = min(10, len(sorted_info))
    top_info = sorted_info[:top_n]

    # Count statistics
    total_removed = sum(1 for info in debug_info if info['removed'])
    total_kept = len(debug_info) - total_removed

    # Create main annotated image
    h, w = gray_img.shape[:2]
    annotated_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

    # Draw all components with color coding
    for info in debug_info:
        x, y, ww, hh = info['bbox']
        if info['removed']:
            color = (0, 0, 255)  # Red for removed
        else:
            color = (0, 255, 0)  # Green for kept
        cv2.rectangle(annotated_img, (x, y), (x + ww - 1, y + hh - 1), color, 1)

    # Create output image with title and main view
    title_h = 80
    panel_h = 200
    panel_w = 1200
    detail_panel_h = panel_h * top_n + 40
    output_h = title_h + panel_h + detail_panel_h
    output_img = np.full((output_h, panel_w, 3), 255, dtype=np.uint8)

    # Title section
    cv2.putText(output_img, "CC FILTER DEBUG - TOP COMPONENTS BY AREA",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    cv2.putText(output_img, f"Total: {len(debug_info)} | Removed: {total_removed} (red) | Kept: {total_kept} (green)",
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Main annotated view (scaled to fit)
    scale = min(panel_w / w, panel_h / h)
    scaled_w = int(w * scale)
    scaled_h = int(h * scale)
    annotated_resized = cv2.resize(annotated_img, (scaled_w, scaled_h))
    y_offset = title_h + (panel_h - scaled_h) // 2
    x_offset = (panel_w - scaled_w) // 2
    output_img[y_offset:y_offset+scaled_h, x_offset:x_offset+scaled_w] = annotated_resized

    # Detail panels for top N components
    detail_y = title_h + panel_h + 20
    cv2.line(output_img, (0, detail_y - 10), (panel_w, detail_y - 10), (200, 200, 200), 2)
    cv2.putText(output_img, f"TOP {top_n} COMPONENTS (sorted by area):",
                (10, detail_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    detail_y += 30

    for idx, info in enumerate(top_info):
        x, y, ww, hh = info['bbox']
        removed = info['removed']
        comp_type = info['component_type']
        reason = info['reason']

        # Extract ROI
        roi = gray_img[y:y+hh, x:x+ww]
        roi_bgr = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

        # Scale ROI to fit panel (max 150px height)
        max_roi_h = 150
        if roi.shape[0] > max_roi_h:
            roi_scale = max_roi_h / roi.shape[0]
            new_w = int(roi.shape[1] * roi_scale)
            new_h = max_roi_h
            roi_bgr = cv2.resize(roi_bgr, (new_w, new_h))

        # Place ROI in panel
        roi_h, roi_w = roi_bgr.shape[:2]
        roi_x = 10
        roi_y_offset = (panel_h - roi_h) // 2
        if detail_y + roi_y_offset + roi_h < output_h:
            output_img[detail_y + roi_y_offset:detail_y + roi_y_offset + roi_h,
                      roi_x:roi_x + roi_w] = roi_bgr

        # Component info text
        text_x = roi_x + roi_w + 20
        text_y = detail_y + 20

        # Rank and decision
        status = "REMOVED" if removed else "KEPT"
        status_color = (0, 0, 255) if removed else (0, 150, 0)
        cv2.putText(output_img, f"#{idx+1}: {status}",
                    (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        # Type badge
        type_colors = {
            'border_touch': (255, 100, 0),  # Orange
            'edge_zone': (255, 150, 0),     # Light orange
            'interior': (100, 100, 255)     # Blue
        }
        type_color = type_colors.get(comp_type, (128, 128, 128))
        cv2.putText(output_img, f"Type: {comp_type}",
                    (text_x, text_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, type_color, 1)

        # Metrics
        metrics_y = text_y + 50
        line_h = 18
        cv2.putText(output_img, f"Area: {info['area']} px ({info['area_ratio']*100:.2f}%)",
                    (text_x, metrics_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(output_img, f"Aspect ratio: {info['aspect_ratio']:.2f}",
                    (text_x, metrics_y + line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(output_img, f"Min rect dim: {info['min_rect_dim']:.1f} px",
                    (text_x, metrics_y + line_h * 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(output_img, f"BBox: {ww}x{hh} px",
                    (text_x, metrics_y + line_h * 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # Reason
        reason_y = metrics_y + line_h * 4 + 10
        cv2.putText(output_img, f"Reason: {reason}",
                    (text_x, reason_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 180), 1)

        # Thresholds reference
        if comp_type == 'border_touch':
            thr_text = f"Thresholds: area>={border_thr}, aspect<{params['max_aspect_for_edge']}, min_dim>{params['min_dim_px']}"
        elif comp_type == 'edge_zone':
            thr_text = f"Thresholds: area>={edge_thr}, aspect<{params['max_aspect_for_edge']}, min_dim>{params['min_dim_px']}"
        else:
            thr_text = f"Thresholds: area>={interior_thr}, min_dim>{params['interior_min_dim_px']}"

        cv2.putText(output_img, thr_text,
                    (text_x, reason_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

        # Separator line
        detail_y += panel_h
        if detail_y < output_h:
            cv2.line(output_img, (0, detail_y - 10), (panel_w, detail_y - 10), (220, 220, 220), 1)

    return output_img
