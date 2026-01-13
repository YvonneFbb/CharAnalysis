"""
Legacy noise removal strategies kept for reference.

These are not used in the current segmentation pipeline.
"""
from __future__ import annotations
from typing import Dict
import numpy as np
import cv2


def dual_threshold_noise_removal(gray_img: np.ndarray, params: Dict) -> np.ndarray:
    """Dual-threshold based noise removal."""
    h, w = gray_img.shape[:2]
    if h == 0 or w == 0:
        return gray_img

    # Parameters
    strict_ratio = float(params.get('strict_threshold_ratio', 0.7))
    loose_ratio = float(params.get('loose_threshold_ratio', 1.2))
    min_area = int(params.get('min_noise_area', 5))
    max_area_ratio = float(params.get('max_noise_area_ratio', 0.1))
    shape_verify = bool(params.get('shape_verification', True))

    # Calculate Otsu threshold
    otsu_thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[0]

    # Dual thresholds
    strict_thresh = int(otsu_thresh * strict_ratio)
    loose_thresh = int(otsu_thresh * loose_ratio)

    # Create masks
    strict_mask = (gray_img <= strict_thresh).astype(np.uint8)
    loose_mask = (gray_img <= loose_thresh).astype(np.uint8)

    # Noise candidates: appear in loose but not in strict
    noise_candidates = loose_mask & ~strict_mask

    if noise_candidates.sum() == 0:
        return gray_img

    # Connected component analysis on noise candidates
    num, labels, stats, _ = cv2.connectedComponentsWithStats(noise_candidates, connectivity=8)

    result = gray_img.copy()
    max_area = int(h * w * max_area_ratio)

    for i in range(1, num):
        x, y, ww, hh, area = stats[i]

        # Size filtering
        if area < min_area or area > max_area:
            continue

        # Shape verification
        if shape_verify:
            if not _is_likely_noise(gray_img, labels == i, x, y, ww, hh, area, params):
                continue

        # Mark as noise - set to local background color
        component_mask = (labels == i)
        background_color = _estimate_local_background(gray_img, component_mask)
        result[component_mask] = background_color

    return result


def statistical_noise_removal(gray_img: np.ndarray, params: Dict) -> np.ndarray:
    """Statistical analysis based noise removal."""
    hist, _ = np.histogram(gray_img.flatten(), bins=256, range=(0, 256))

    # Find dominant dark range (main text)
    dark_peak = np.argmax(hist[:128])  # Peak in dark range
    _text_upper_bound = min(dark_peak + 50, 180)  # Conservative upper bound

    # Noise range parameters
    noise_range = params.get('noise_gray_range', [150, 220])
    min_area = int(params.get('min_noise_area', 5))

    # Create noise mask
    noise_mask = ((gray_img >= noise_range[0]) & (gray_img <= noise_range[1])).astype(np.uint8)

    # Connected component filtering
    num, labels, stats, _ = cv2.connectedComponentsWithStats(noise_mask, connectivity=8)

    result = gray_img.copy()
    for i in range(1, num):
        x, y, ww, hh, area = stats[i]
        if area >= min_area:
            component_mask = (labels == i)
            background_color = _estimate_local_background(gray_img, component_mask)
            result[component_mask] = background_color

    return result


def hybrid_noise_removal(gray_img: np.ndarray, params: Dict) -> np.ndarray:
    """Combine dual-threshold and statistical methods."""
    result1 = dual_threshold_noise_removal(gray_img, params)
    result2 = statistical_noise_removal(result1, params)
    return result2


def _is_likely_noise(
    gray_img: np.ndarray,
    component_mask: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    area: int,
    params: Dict
) -> bool:
    """Analyze component characteristics to determine if it's likely noise."""
    aspect_ratio = max(w, h) / max(min(w, h), 1)
    perimeter = cv2.countNonZero(cv2.Canny(component_mask.astype(np.uint8) * 255, 50, 150))
    compactness = 4 * np.pi * area / max(perimeter * perimeter, 1) if perimeter > 0 else 0

    min_compactness = float(params.get('min_compactness', 0.1))

    if aspect_ratio > 8.0:
        return True

    if compactness < min_compactness:
        return True

    component_pixels = gray_img[component_mask]
    if len(component_pixels) > 0:
        mean_gray = np.mean(component_pixels)
        std_gray = np.std(component_pixels)
        if mean_gray > 180 and std_gray < 20:
            return True

    return False


def _estimate_local_background(gray_img: np.ndarray, component_mask: np.ndarray, radius: int = 8) -> int:
    # Keep logic identical to the main implementation for reference.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    dilated = cv2.dilate(component_mask.astype(np.uint8), kernel)
    background_mask = dilated & ~component_mask.astype(np.uint8)

    if background_mask.sum() > 0:
        background_pixels = gray_img[background_mask.astype(bool)]
        light_pixels = background_pixels[background_pixels >= 150]
        if len(light_pixels) > 0:
            return int(np.median(light_pixels))
        return int(np.median(background_pixels))

    y_coords, x_coords = np.where(component_mask)
    if len(y_coords) > 0:
        y_min, y_max = max(0, y_coords.min() - radius), min(gray_img.shape[0], y_coords.max() + radius + 1)
        x_min, x_max = max(0, x_coords.min() - radius), min(gray_img.shape[1], x_coords.max() + radius + 1)
        surrounding = gray_img[y_min:y_max, x_min:x_max]
        light_surrounding = surrounding[surrounding >= 150]
        if len(light_surrounding) > 0:
            return int(np.median(light_surrounding))
        return int(np.median(surrounding))

    return 245
