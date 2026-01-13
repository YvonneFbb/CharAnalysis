"""
Legacy border removal helpers that are not used in the current pipeline.

These were kept for reference while simplifying the main segmentation code.
"""
from __future__ import annotations
from typing import Dict
import numpy as np
import cv2


def remove_border_lines(binary_img: np.ndarray, params: Dict) -> np.ndarray:
    """
    Remove long, thin border lines near image edges.

    Works on a binary image (foreground=255). Only lines close to the edges
    are removed to avoid deleting internal strokes.
    """
    if binary_img is None or binary_img.size == 0:
        return binary_img
    h, w = binary_img.shape[:2]
    if h == 0 or w == 0:
        return binary_img

    line_cfg = params.get('line_removal', {}) if isinstance(params, dict) else {}
    if not line_cfg.get('enabled', False):
        return binary_img

    edge_ratio = float(line_cfg.get('edge_margin_ratio', 0.08))
    min_len_ratio = float(line_cfg.get('min_length_ratio', 0.7))
    min_len_px = int(line_cfg.get('min_length_px', 12))
    max_thickness = int(line_cfg.get('max_thickness_px', 3))

    margin_y = max(1, int(h * edge_ratio))
    margin_x = max(1, int(w * edge_ratio))

    # Horizontal lines
    k_len_h = max(min_len_px, int(w * min_len_ratio))
    k_len_h = max(3, min(k_len_h, w))
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (k_len_h, 1))
    h_mask = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel_h)
    num, labels, stats, _ = cv2.connectedComponentsWithStats((h_mask > 0).astype(np.uint8), connectivity=8)
    for i in range(1, num):
        x, y, ww, hh, _area = stats[i]
        if hh > max_thickness:
            continue
        if ww < min_len_px and ww < int(w * min_len_ratio):
            continue
        if y <= margin_y or (y + hh) >= (h - margin_y):
            binary_img[labels == i] = 0

    # Vertical lines
    k_len_v = max(min_len_px, int(h * min_len_ratio))
    k_len_v = max(3, min(k_len_v, h))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_len_v))
    v_mask = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel_v)
    num, labels, stats, _ = cv2.connectedComponentsWithStats((v_mask > 0).astype(np.uint8), connectivity=8)
    for i in range(1, num):
        x, y, ww, hh, _area = stats[i]
        if ww > max_thickness:
            continue
        if hh < min_len_px and hh < int(h * min_len_ratio):
            continue
        if x <= margin_x or (x + ww) >= (w - margin_x):
            binary_img[labels == i] = 0

    return binary_img


def analyze_border_removal_effect(original: np.ndarray, cleaned: np.ndarray) -> Dict:
    """
    Analyze the effect of border removal for debugging.

    Returns:
        Dictionary with analysis results
    """
    if original is None or cleaned is None:
        return {'error': 'Invalid input images'}

    if len(original.shape) == 3:
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    else:
        orig_gray = original.copy()

    if len(cleaned.shape) == 3:
        clean_gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    else:
        clean_gray = cleaned.copy()

    _, orig_bin = cv2.threshold(orig_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, clean_bin = cv2.threshold(clean_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    removed_pixels = cv2.bitwise_and(orig_bin, cv2.bitwise_not(clean_bin))
    removed_area = np.sum(removed_pixels > 0)
    total_area = np.sum(orig_bin > 0)

    return {
        'removed_area': int(removed_area),
        'total_area': int(total_area),
        'removal_ratio': float(removed_area / max(1, total_area)),
        'has_effect': removed_area > 0
    }
