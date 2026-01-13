"""
Border removal for ancient text images using horizontal projection analysis.

Remove adhesion borders and frames that may still remain after projection trimming.
Uses horizontal projection to detect short, high-intensity regions at the edges
that typically indicate border structures.
"""
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import cv2


def remove_border_frames(binary_img: np.ndarray, params: Dict) -> np.ndarray:
    """
    Remove L-shaped border frames by detecting edge-near orthogonal line pairs.

    Only removes lines that form a corner near the ROI edges, to avoid deleting
    internal strokes.
    """
    if binary_img is None or binary_img.size == 0:
        return binary_img
    h, w = binary_img.shape[:2]
    if h == 0 or w == 0:
        return binary_img

    frame_cfg = params.get('frame_removal', {}) if isinstance(params, dict) else {}
    if not frame_cfg.get('enabled', False):
        return binary_img

    edge_ratio = float(frame_cfg.get('edge_margin_ratio', 0.08))
    min_len_ratio = float(frame_cfg.get('min_length_ratio', 0.7))
    min_len_px = int(frame_cfg.get('min_length_px', 12))
    max_thickness = int(frame_cfg.get('max_thickness_px', 3))
    min_corners = int(frame_cfg.get('min_corner_count', 1))
    max_remove_ratio = float(frame_cfg.get('max_removal_ratio', 0.2))

    margin_y = max(1, int(h * edge_ratio))
    margin_x = max(1, int(w * edge_ratio))

    k_len_h = max(min_len_px, int(w * min_len_ratio))
    k_len_h = max(3, min(k_len_h, w))
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (k_len_h, 1))
    h_mask = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel_h)

    k_len_v = max(min_len_px, int(h * min_len_ratio))
    k_len_v = max(3, min(k_len_v, h))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_len_v))
    v_mask = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel_v)

    inter = cv2.bitwise_and(h_mask, v_mask)
    if cv2.countNonZero(inter) == 0:
        return binary_img

    num_h, h_labels, h_stats, _ = cv2.connectedComponentsWithStats((h_mask > 0).astype(np.uint8), connectivity=8)
    num_v, v_labels, v_stats, _ = cv2.connectedComponentsWithStats((v_mask > 0).astype(np.uint8), connectivity=8)
    num_i, i_labels, i_stats, _ = cv2.connectedComponentsWithStats((inter > 0).astype(np.uint8), connectivity=8)

    def _is_h_valid(stats) -> bool:
        x, y, ww, hh, _ = stats
        if hh > max_thickness:
            return False
        if ww < min_len_px and ww < int(w * min_len_ratio):
            return False
        return (y <= margin_y) or (y + hh >= h - margin_y)

    def _is_v_valid(stats) -> bool:
        x, y, ww, hh, _ = stats
        if ww > max_thickness:
            return False
        if hh < min_len_px and hh < int(h * min_len_ratio):
            return False
        return (x <= margin_x) or (x + ww >= w - margin_x)

    to_remove_h = set()
    to_remove_v = set()
    corner_hits = 0

    for i in range(1, num_i):
        x, y, ww, hh, _ = i_stats[i]
        near_edge = (
            x <= margin_x or y <= margin_y or
            (x + ww) >= (w - margin_x) or
            (y + hh) >= (h - margin_y)
        )
        if not near_edge:
            continue
        # take one pixel from this intersection component
        ys, xs = np.where(i_labels == i)
        if ys.size == 0:
            continue
        y0, x0 = int(ys[0]), int(xs[0])
        hl = int(h_labels[y0, x0])
        vl = int(v_labels[y0, x0])
        if hl <= 0 or vl <= 0:
            continue
        if hl >= num_h or vl >= num_v:
            continue
        if not _is_h_valid(h_stats[hl]):
            continue
        if not _is_v_valid(v_stats[vl]):
            continue
        to_remove_h.add(hl)
        to_remove_v.add(vl)
        corner_hits += 1

    if corner_hits < min_corners:
        return binary_img

    removal_mask = np.zeros_like(binary_img, dtype=np.uint8)
    for hl in to_remove_h:
        removal_mask[h_labels == hl] = 255
    for vl in to_remove_v:
        removal_mask[v_labels == vl] = 255

    fg_total = int(np.count_nonzero(binary_img))
    fg_removed = int(np.count_nonzero(removal_mask))
    if fg_total > 0 and (fg_removed / float(fg_total)) > max_remove_ratio:
        return binary_img

    cleaned = binary_img.copy()
    cleaned[removal_mask > 0] = 0
    return cleaned


def trim_border_from_bin(binary_img: np.ndarray, params: Dict) -> Tuple[int, int, int, int]:
    """
    Find border trim coordinates using horizontal and vertical projection analysis.

    This function first removes horizontal borders, then performs vertical
    projection analysis to tighten the top/bottom boundaries.
    Supports multiple iterations to completely remove thick borders.

    Args:
        binary_img: Binary image (foreground=255, background=0)
        params: Configuration parameters

    Returns:
        Tuple of (xl, xr, yt, yb) - trim coordinates
    """
    h, w = binary_img.shape[:2]
    if h == 0 or w == 0:
        return 0, w, 0, h

    # Get iteration settings
    max_iterations = int(params.get('max_iterations', 1))

    # Initialize coordinates
    total_xl, total_xr = 0, w
    current_img = binary_img.copy()

    # Iterate border removal
    for iteration in range(max_iterations):
        # Step 1: Find border regions using horizontal projection
        left_cut, right_cut = _find_border_cuts_by_projection(current_img, params)

        # If no border found in this iteration, break
        if left_cut == 0 and right_cut == 0:
            break

        # Convert cuts to trim coordinates for this iteration
        xl = left_cut
        xr = current_img.shape[1] - right_cut

        # Ensure valid horizontal coordinates
        xl = max(0, min(current_img.shape[1]-1, xl))
        xr = max(xl+1, min(current_img.shape[1], xr))

        # Update total coordinates (accumulate cuts)
        total_xl += xl
        total_xr = total_xl + (xr - xl)

        # Prepare for next iteration - crop the current image
        if xl < xr and iteration < max_iterations - 1:
            current_img = current_img[:, xl:xr]
        else:
            break

    # Step 2: Perform vertical cutting only if horizontal border cutting actually occurred
    horizontal_cutting_occurred = (total_xl > 0) or (total_xr < w)

    if horizontal_cutting_occurred and total_xl < total_xr:
        final_region = binary_img[:, total_xl:total_xr]
        yt, yb = _find_vertical_trim_bounds(final_region, params)
    else:
        yt, yb = 0, h

    # Ensure valid coordinates
    total_xl = max(0, min(w-1, total_xl))
    total_xr = max(total_xl+1, min(w, total_xr))
    yt = max(0, min(h-1, yt))
    yb = max(yt+1, min(h, yb))

    return total_xl, total_xr, yt, yb


def _find_border_cuts_by_projection(binary_img: np.ndarray, params: Dict) -> Tuple[int, int]:
    """
    Find border cut positions using horizontal projection analysis.

    Args:
        binary_img: Binary image (foreground=255)
        params: Configuration parameters

    Returns:
        Tuple of (left_cut_pixels, right_cut_pixels)
    """
    h, w = binary_img.shape[:2]
    if h == 0 or w == 0:
        return 0, 0

    # Parameters
    max_border_width_ratio = float(params.get('border_max_width_ratio', 0.2))    # 最大边框宽度占比

    # Calculate horizontal projection (column-wise coverage)
    horizontal_projection = binary_img.sum(axis=0).astype(np.float32) / (h * 255.0)

    max_coverage = horizontal_projection.max() if horizontal_projection.size > 0 else 0.0
    if max_coverage <= 0.0:
        return 0, 0

    max_border_width = int(w * max_border_width_ratio)

    left_cut = 0
    right_cut = 0

    # Check left edge for border using spike detection
    left_projection = horizontal_projection[:max_border_width]
    left_debug = _debug_border_detection(left_projection, params, is_left=True, total_width=w, global_max_coverage=max_coverage)
    left_cut = left_debug.get('final_cut_pos', 0)

    # Check right edge for border using spike detection
    right_projection = horizontal_projection[-max_border_width:]
    right_debug = _debug_border_detection(right_projection, params, is_left=False, total_width=w, global_max_coverage=max_coverage)
    right_cut = right_debug.get('final_cut_pos', 0)

    return left_cut, right_cut


def _debug_border_detection(projection: np.ndarray, params: Dict, is_left: bool, total_width: int = None, global_max_coverage: float = None) -> Dict:
    """
    Debug function to analyze border detection process.

    Returns detailed information about detection process for debugging.
    """
    if projection.size == 0:
        return {'error': 'Empty projection'}

    max_coverage = projection.max() if projection.size > 0 else 0.0
    if max_coverage <= 0.0:
        return {'error': 'No coverage data'}

    # Use global max coverage for threshold calculation if provided (to match main algorithm)
    threshold_base = global_max_coverage if global_max_coverage is not None else max_coverage

    # Calculate spike lengths based on total width if provided, otherwise use projection length
    if total_width is not None:
        spike_min_length = max(1, int(total_width * float(params.get('spike_min_length_ratio', 0.02))))
        spike_max_length = max(spike_min_length, int(total_width * float(params.get('spike_max_length_ratio', 0.1))))
    else:
        spike_min_length = max(1, int(len(projection) * float(params.get('spike_min_length_ratio', 0.02))))
        spike_max_length = max(spike_min_length, int(len(projection) * float(params.get('spike_max_length_ratio', 0.1))))

    gradient_threshold = threshold_base * float(params.get('spike_gradient_threshold', 0.4))
    prominence_ratio = float(params.get('spike_prominence_ratio', 0.6))
    border_threshold = threshold_base * float(params.get('border_threshold_ratio', 0.5))
    edge_tolerance = int(params.get('edge_tolerance', 8))

    debug_info = {
        'projection_length': len(projection),
        'max_coverage': max_coverage,
        'border_threshold': border_threshold,
        'spike_min_length': spike_min_length,
        'spike_max_length': spike_max_length,
        'gradient_threshold': gradient_threshold,
        'prominence_ratio': prominence_ratio,
        'edge_tolerance': edge_tolerance,
        'is_left': is_left,
        'detection_attempts': [],
        'final_cut_pos': 0,
        'max_border_length': 0
    }

    best_cut_pos = 0
    max_border_length = 0

    if is_left:
        border_start = -1
        for i in range(len(projection)):
            if projection[i] >= border_threshold:
                if border_start == -1:
                    border_start = i
            else:
                if border_start != -1:
                    border_length = i - border_start
                    border_avg = projection[border_start:i].mean()

                    # Check if border is close enough to edge
                    is_near_edge = border_start <= edge_tolerance

                    lookahead_end = min(len(projection), i + border_length)
                    content_avg = projection[i:lookahead_end].mean() if i < len(projection) else 0
                    drop_magnitude = border_avg - content_avg

                    gradient_ok = drop_magnitude >= gradient_threshold
                    prominence_ok = drop_magnitude >= border_avg * prominence_ratio
                    length_ok = spike_min_length <= border_length <= spike_max_length

                    attempt = {
                        'position': i,
                        'border_start': border_start,
                        'border_length': border_length,
                        'border_avg': border_avg,
                        'content_avg': content_avg,
                        'drop_magnitude': drop_magnitude,
                        'is_near_edge': is_near_edge,
                        'length_ok': length_ok,
                        'gradient_ok': gradient_ok,
                        'prominence_ok': prominence_ok,
                        'passed': is_near_edge and length_ok and gradient_ok and prominence_ok
                    }
                    # Add detailed projection values for debugging
                    attempt['projection_values'] = projection[border_start:i].tolist() if border_length > 0 else []
                    attempt['gradient_threshold_check'] = gradient_threshold
                    attempt['prominence_threshold_check'] = border_avg * prominence_ratio if border_length > 0 else 0

                    debug_info['detection_attempts'].append(attempt)

                    if attempt['passed'] and border_length > max_border_length:
                        max_border_length = border_length
                        best_cut_pos = i

                    border_start = -1
    else:
        border_start = -1
        for i in range(len(projection) - 1, -1, -1):
            if projection[i] >= border_threshold:
                if border_start == -1:
                    border_start = i
            else:
                if border_start != -1:
                    border_length = border_start - i
                    border_avg = projection[i+1:border_start+1].mean()

                    # Check if border is close enough to edge
                    distance_from_edge = len(projection) - 1 - border_start
                    is_near_edge = distance_from_edge <= edge_tolerance

                    lookahead_start = max(0, i - border_length + 1)
                    content_avg = projection[lookahead_start:i+1].mean() if i >= 0 else 0
                    drop_magnitude = border_avg - content_avg

                    gradient_ok = drop_magnitude >= gradient_threshold
                    prominence_ok = drop_magnitude >= border_avg * prominence_ratio
                    length_ok = spike_min_length <= border_length <= spike_max_length

                    attempt = {
                        'position': i,
                        'border_start': border_start,
                        'border_length': border_length,
                        'border_avg': border_avg,
                        'content_avg': content_avg,
                        'drop_magnitude': drop_magnitude,
                        'is_near_edge': is_near_edge,
                        'length_ok': length_ok,
                        'gradient_ok': gradient_ok,
                        'prominence_ok': prominence_ok,
                        'passed': is_near_edge and length_ok and gradient_ok and prominence_ok
                    }
                    # Add detailed projection values for debugging
                    attempt['projection_values'] = projection[i+1:border_start+1].tolist() if border_length > 0 else []
                    attempt['gradient_threshold_check'] = gradient_threshold
                    attempt['prominence_threshold_check'] = border_avg * prominence_ratio if border_length > 0 else 0

                    debug_info['detection_attempts'].append(attempt)

                    if attempt['passed'] and border_length > max_border_length:
                        max_border_length = border_length
                        best_cut_pos = len(projection) - border_start

                    border_start = -1

    debug_info['final_cut_pos'] = best_cut_pos
    debug_info['max_border_length'] = max_border_length
    return debug_info


def _find_vertical_trim_bounds(binary_img: np.ndarray, params: Dict) -> Tuple[int, int]:
    """
    Find vertical trim bounds using structured detection range and cut limits.

    This function uses a two-stage approach similar to projection trimming:
    1. Detection stage: Find content boundaries within specified detection ranges
    2. Cut limit stage: Apply maximum cut constraints to protect content

    Args:
        binary_img: Binary image after horizontal border removal
        params: Configuration parameters

    Returns:
        Tuple of (yt, yb) - vertical trim coordinates
    """
    h, w = binary_img.shape[:2]
    if h == 0 or w == 0:
        return 0, h

    # Calculate vertical projection (row-wise coverage)
    coverage = binary_img.sum(axis=1).astype(np.float32) / (w * 255.0)

    if coverage.size == 0:
        return 0, h

    # Get parameters with fallback to legacy behavior
    vertical_detection = params.get('vertical_detection_range', {})
    vertical_cuts = params.get('vertical_cut_limits', {})

    # Detection range parameters
    top_detection_ratio = float(vertical_detection.get('top_ratio', 0.3))
    bottom_detection_ratio = float(vertical_detection.get('bottom_ratio', 0.3))

    # Cut limit parameters
    top_cut_ratio = float(vertical_cuts.get('top_max_ratio', 0.1))
    bottom_cut_ratio = float(vertical_cuts.get('bottom_max_ratio', 0.1))


    # New structured approach: Find content runs
    max_cov = coverage.max()
    if max_cov <= 0.0:
        return 0, h

    # Use same content detection threshold as Proj trimming
    min_cov = max(0.001, 0.01 * max_cov)  # Same as Proj: 1% of max coverage

    # Find content runs
    runs = []
    i = 0
    while i < h:
        if coverage[i] <= min_cov:
            i += 1
            continue
        j = i
        while j < h and coverage[j] > min_cov:
            j += 1
        runs.append((i, j))
        i = j

    if not runs:
        return 0, h

    # Calculate detection and cut limits (same as Proj)
    top_detection_limit = int(h * top_detection_ratio)
    bottom_detection_limit = int(h * bottom_detection_ratio)
    top_trim_limit = int(h * top_cut_ratio)
    bottom_trim_limit = int(h * bottom_cut_ratio)

    # 阶段1：在检测范围内寻找内容边界 (same logic as Proj)
    # Find top boundary within detection range
    top_idx = 0
    while top_idx < len(runs) and runs[top_idx][1] <= top_detection_limit:
        top_idx += 1
    if top_idx >= len(runs):
        top_idx = len(runs) - 1

    top_detected = runs[top_idx][0] if top_idx < len(runs) else 0

    # Find bottom boundary within detection range
    bottom_idx = len(runs) - 1
    while bottom_idx >= 0 and runs[bottom_idx][0] >= max(0, h - bottom_detection_limit):
        bottom_idx -= 1
    if bottom_idx < 0:
        bottom_idx = len(runs) - 1

    bottom_detected = runs[bottom_idx][1] if bottom_idx >= 0 else h

    # 阶段2：应用切割限制，确保不过度切割实际内容 (same logic as Proj)
    # Find overall content boundaries
    content_top = runs[0][0] if runs else 0
    content_bottom = runs[-1][1] if runs else h

    # Calculate maximum allowed cutting of actual content
    content_height = content_bottom - content_top
    max_top_cut = int(content_height * top_cut_ratio) if content_height > 0 else 0
    max_bottom_cut = int(content_height * bottom_cut_ratio) if content_height > 0 else 0

    # Calculate final cuts using Proj logic
    # Top: can cut whitespace freely, but limit content cutting
    if top_detected <= content_top:
        # Cutting only whitespace above content
        final_top = top_detected
    else:
        # Would cut into content, apply limit
        final_top = max(content_top, content_top + max(0, min(max_top_cut, top_detected - content_top)))

    # Bottom: similar logic
    if bottom_detected >= content_bottom:
        # Cutting only whitespace below content
        final_bottom = bottom_detected
    else:
        # Would cut into content, apply limit
        final_bottom = min(content_bottom, content_bottom - max(0, min(max_bottom_cut, content_bottom - bottom_detected)))

    yt = final_top
    yb = final_bottom

    # Fine adjustment: remove very low coverage pixels at edges
    tighten_threshold = max(0.005, 0.005 * max_cov)
    while yt < h and coverage[yt] <= tighten_threshold:
        yt += 1
    while yb > 0 and coverage[yb - 1] <= tighten_threshold:
        yb -= 1

    # Ensure valid bounds
    yt = max(0, min(h-1, yt))
    yb = max(yt+1, min(h, yb))

    return yt, yb

