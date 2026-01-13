"""
Debug visualization helpers for segmentation.

Extracted from vertical_hybrid to reduce core module size.
"""
from __future__ import annotations
from typing import Dict
import cv2
import numpy as np

from .config import BORDER_REMOVAL_CONFIG


def _create_border_projection_viz(border_region: np.ndarray, params: Dict, max_height: int = 120,
                                  xl_cut: int = 0, xr_cut: int = 0, target_width: int = 400) -> np.ndarray:
    """
    Create border projection visualization that can be embedded in main debug image.
    """
    if border_region.size == 0:
        return np.full((max_height, target_width, 3), 255, dtype=np.uint8)

    border_hproj = border_region.sum(axis=0).astype(np.float32)
    if border_hproj.size == 0 or border_hproj.max() == 0:
        return np.full((max_height, target_width, 3), 255, dtype=np.uint8)

    bhp_coverage = border_hproj / (border_region.shape[0] * 255.0)
    max_coverage = bhp_coverage.max()

    max_width = int(border_region.shape[1] * params.get('border_max_width_ratio', 0.2))
    border_threshold = max_coverage * params.get('border_threshold_ratio', 0.5)

    viz_width = target_width
    viz_height = max_height
    viz_img = np.full((viz_height, viz_width, 3), 255, dtype=np.uint8)

    if len(bhp_coverage) > 0:
        scale_x = viz_width / len(bhp_coverage)
        proj_height = int(viz_height * 0.95)

        for i, val in enumerate(bhp_coverage):
            x = int(i * scale_x)
            bar_height = int(val / max_coverage * proj_height) if max_coverage > 0 else 0
            y_start = viz_height - 3 - bar_height
            y_end = viz_height - 3
            if bar_height > 0:
                cv2.rectangle(viz_img, (x, y_start), (min(x + int(scale_x) + 1, viz_width - 1), y_end), (128, 128, 128), -1)

        left_zone_end = int(max_width * scale_x)
        right_zone_start = viz_width - int(max_width * scale_x)

        cv2.rectangle(viz_img, (0, 2), (left_zone_end, 18), (255, 0, 0), 2)
        cv2.putText(viz_img, "LEFT ZONE", (2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)

        if right_zone_start > left_zone_end:
            cv2.rectangle(viz_img, (right_zone_start, 2), (viz_width - 1, 18), (255, 0, 0), 2)
            cv2.putText(viz_img, "RIGHT ZONE", (right_zone_start + 2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)

        threshold_y = viz_height - 8 - int(border_threshold / max_coverage * proj_height) if max_coverage > 0 else viz_height - 8
        cv2.line(viz_img, (0, threshold_y), (viz_width - 1, threshold_y), (0, 255, 0), 2)
        cv2.putText(viz_img, f"Threshold: {border_threshold:.3f}", (2, threshold_y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

        if xl_cut > 0:
            xl_viz = int(xl_cut * scale_x)
            cv2.line(viz_img, (xl_viz, 22), (xl_viz, viz_height - 5), (0, 0, 255), 2)
            cv2.putText(viz_img, f"L:{xl_cut}", (xl_viz + 1, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 0, 255), 1)

        if xr_cut < border_region.shape[1]:
            xr_viz = int(xr_cut * scale_x)
            cv2.line(viz_img, (xr_viz, 22), (xr_viz, viz_height - 5), (0, 0, 255), 2)
            cv2.putText(viz_img, f"R:{xr_cut}", (max(0, xr_viz - 20), 32), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 0, 255), 1)

    return viz_img


def _create_border_debug_image(border_region: np.ndarray,
                               xl_border: int, xr_border: int,
                               xl_proj: int, xr_proj: int,
                               params: Dict,
                               yt_border: int = 0, yb_border: int = 0) -> np.ndarray:
    if border_region.size == 0:
        return np.full((800, 1200, 3), 255, dtype=np.uint8)

    debug_width = 1200
    debug_height = 800
    debug_img = np.full((debug_height, debug_width, 3), 255, dtype=np.uint8)

    cv2.putText(debug_img, "BORDER DETECTION ANALYSIS", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    cv2.line(debug_img, (10, 50), (debug_width - 10, 50), (200, 200, 200), 2)

    left_w = debug_width // 2 - 40
    _draw_horizontal_border_analysis(debug_img, border_region, xl_border, xr_border, params,
                                     start_x=20, start_y=80, width=left_w, height=320)

    right_start_x = debug_width // 2 + 20
    _draw_vertical_border_analysis(debug_img, border_region, yt_border, yb_border, params,
                                   start_x=right_start_x, start_y=80, width=left_w, height=320)

    return debug_img


def _draw_horizontal_border_analysis(debug_img: np.ndarray, border_region: np.ndarray,
                                     xl_border: int, xr_border: int, params: Dict,
                                     start_x: int, start_y: int, width: int, height: int) -> None:
    h, w = border_region.shape
    cv2.putText(debug_img, "HORIZONTAL ANALYSIS", (start_x, start_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    hproj = border_region.sum(axis=0).astype(np.float32) / (h * 255.0)
    max_cov = hproj.max() if hproj.size > 0 else 0.0

    proj_height = 200
    proj_area_y = start_y + 10
    scale_x = (width - 30) / max(1, len(hproj))

    cv2.rectangle(debug_img, (start_x + 15, proj_area_y),
                  (start_x + width - 15, proj_area_y + proj_height), (245, 245, 245), -1)

    for i, cov in enumerate(hproj):
        x = start_x + 15 + int(i * scale_x)
        bar_h = int(cov / max(max_cov, 1e-6) * (proj_height - 15))
        y_start = proj_area_y + proj_height - 10 - bar_h
        cv2.rectangle(debug_img, (x, y_start),
                      (x + max(1, int(scale_x)), proj_area_y + proj_height - 10), (80, 80, 80), -1)

    max_width = int(w * params.get('border_max_width_ratio', 0.2))
    left_zone_end = start_x + 15 + int(max_width * scale_x)
    right_zone_start = start_x + 15 + int((w - max_width) * scale_x)

    cv2.rectangle(debug_img, (start_x + 15, proj_area_y),
                  (left_zone_end, proj_area_y + proj_height), (0, 0, 255), 2)
    cv2.rectangle(debug_img, (right_zone_start, proj_area_y),
                  (start_x + width - 15, proj_area_y + proj_height), (255, 0, 0), 2)

    cv2.putText(debug_img, "LEFT", (start_x + 25, proj_area_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.putText(debug_img, "RIGHT", (right_zone_start + 5, proj_area_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    border_threshold = max_cov * params.get('border_threshold_ratio', 0.5)
    threshold_y = proj_area_y + proj_height - 10 - int(border_threshold / max(max_cov, 1e-6) * (proj_height - 15))
    cv2.line(debug_img, (start_x + 15, threshold_y), (start_x + width - 15, threshold_y), (0, 255, 0), 2)
    thresh_text = f"Thresh: {border_threshold:.3f}"
    cv2.putText(debug_img, thresh_text, (start_x + 5, threshold_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 150, 0), 1)

    if xl_border > 0:
        cut_x = start_x + 15 + int(xl_border * scale_x)
        cv2.line(debug_img, (cut_x, proj_area_y), (cut_x, proj_area_y + proj_height), (255, 0, 255), 3)
        cv2.putText(debug_img, f"L:{xl_border}", (cut_x + 2, proj_area_y + proj_height + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)
    if xr_border < w:
        cut_x = start_x + 15 + int(xr_border * scale_x)
        cv2.line(debug_img, (cut_x, proj_area_y), (cut_x, proj_area_y + proj_height), (255, 0, 255), 3)
        cv2.putText(debug_img, f"R:{w - xr_border}", (cut_x - 25, proj_area_y + proj_height + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)

    info_y = proj_area_y + proj_height + 40
    cv2.line(debug_img, (start_x, info_y - 5), (start_x + width, info_y - 5), (200, 200, 200), 1)
    cv2.putText(debug_img, "PARAMETERS:", (start_x, info_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(debug_img, f"Width: {w}px | Coverage: {max_cov:.3f}",
                (start_x, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (50, 50, 50), 1)
    cv2.putText(debug_img,
                f"Zone: {max_width}px ({params.get('border_max_width_ratio', 0.2)*100:.0f}%) | Thresh: {params.get('border_threshold_ratio', 0.5)*100:.0f}%",
                (start_x, info_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (50, 50, 50), 1)


def _draw_vertical_border_analysis(debug_img: np.ndarray, border_region: np.ndarray,
                                   yt_border: int, yb_border: int, params: Dict,
                                   start_x: int, start_y: int, width: int, height: int) -> None:
    h, w = border_region.shape
    cv2.putText(debug_img, "VERTICAL ANALYSIS", (start_x, start_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    vproj = border_region.sum(axis=1).astype(np.float32) / (w * 255.0)
    max_cov = vproj.max() if vproj.size > 0 else 0.0

    proj_width = 200
    proj_area_x = start_x + 10
    scale_y = (height - 30) / max(1, len(vproj))

    cv2.rectangle(debug_img, (proj_area_x, start_y + 10),
                  (proj_area_x + proj_width, start_y + height - 20), (245, 245, 245), -1)

    for i, cov in enumerate(vproj):
        y = start_y + 10 + int(i * scale_y)
        bar_w = int(cov / max(max_cov, 1e-6) * (proj_width - 15))
        cv2.rectangle(debug_img, (proj_area_x + 10, y),
                      (proj_area_x + 10 + bar_w, y + max(1, int(scale_y))), (80, 80, 80), -1)

    vertical_detection = params.get('vertical_detection_range', {})
    top_detection_ratio = float(vertical_detection.get('top_ratio', 0.3))
    bottom_detection_ratio = float(vertical_detection.get('bottom_ratio', 0.3))
    top_zone_start = start_y + 10
    top_zone_end = start_y + 10 + int(h * top_detection_ratio * scale_y)
    bottom_zone_start = start_y + 10 + int((h - h * bottom_detection_ratio) * scale_y)

    cv2.rectangle(debug_img, (proj_area_x, top_zone_start),
                  (proj_area_x + proj_width, top_zone_end), (0, 0, 255), 2)
    cv2.rectangle(debug_img, (proj_area_x, bottom_zone_start),
                  (proj_area_x + proj_width, start_y + height - 20), (255, 0, 0), 2)

    cv2.putText(debug_img, "TOP", (proj_area_x + 5, start_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.putText(debug_img, "BOTTOM", (proj_area_x + 5, bottom_zone_start + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    if yt_border > 0:
        cut_y = start_y + 10 + int(yt_border * scale_y)
        cv2.line(debug_img, (proj_area_x, cut_y), (proj_area_x + proj_width, cut_y), (255, 0, 255), 3)
        cv2.putText(debug_img, f"T:{yt_border}", (proj_area_x + proj_width + 5, cut_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)
    if yb_border < h:
        cut_y = start_y + 10 + int(yb_border * scale_y)
        cv2.line(debug_img, (proj_area_x, cut_y), (proj_area_x + proj_width, cut_y), (255, 0, 255), 3)
        cv2.putText(debug_img, f"B:{h - yb_border}", (proj_area_x + proj_width + 5, cut_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)

    info_y = start_y + height - 10
    cv2.line(debug_img, (start_x, info_y - 5), (start_x + width, info_y - 5), (200, 200, 200), 1)
    cv2.putText(debug_img, "PARAMETERS:", (start_x, info_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(debug_img, f"Height: {h}px | Coverage: {max_cov:.3f}",
                (start_x, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (50, 50, 50), 1)
    cv2.putText(debug_img,
                f"Top: {top_detection_ratio*100:.0f}% | Bottom: {bottom_detection_ratio*100:.0f}%",
                (start_x, info_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (50, 50, 50), 1)


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

    projection_overlay = masked_roi.copy()
    cv2.line(projection_overlay, (max(0, int(xl_proj)), 0), (max(0, int(xl_proj)), h - 1), (0, 0, 255), 1)
    cv2.line(projection_overlay, (max(0, int(xr_proj - 1)), 0), (max(0, int(xr_proj - 1)), h - 1), (0, 0, 255), 1)
    cv2.line(projection_overlay, (0, max(0, int(yt_proj))), (w - 1, max(0, int(yt_proj))), (0, 0, 255), 1)
    cv2.line(projection_overlay, (0, max(0, int(yb_proj - 1))), (w - 1, max(0, int(yb_proj - 1))), (0, 0, 255), 1)
    projection_panel = resize_panel(projection_overlay)

    hist_w = base_w
    v_panel = np.full((panel_h, hist_w), 255, dtype=np.uint8)
    vproj = mask_after.sum(axis=0).astype(np.float32)
    if vproj.size:
        vp = vproj / (vproj.max() + 1e-6)
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

    h_panel = np.full((panel_h, hist_w), 255, dtype=np.uint8)
    hproj = mask_after.sum(axis=1).astype(np.float32)
    if hproj.size:
        hp = hproj / (hproj.max() + 1e-6)
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

    border_overlay = np.full_like(roi_bgr, 255)
    proj_region_mask = np.zeros_like(bin_after)
    if xl_proj < xr_proj and yt_proj < yb_proj:
        proj_region_mask[yt_proj:yb_proj, xl_proj:xr_proj] = bin_after[yt_proj:yb_proj, xl_proj:xr_proj]
        border_overlay[proj_region_mask.astype(bool)] = roi_bgr[proj_region_mask.astype(bool)]

    cv2.line(border_overlay, (max(0, int(xl_border)), 0), (max(0, int(xl_border)), h - 1), (255, 0, 0), 1)
    cv2.line(border_overlay, (max(0, int(xr_border - 1)), 0), (max(0, int(xr_border - 1)), h - 1), (255, 0, 0), 1)
    cv2.line(border_overlay, (0, max(0, int(yt_border))), (w - 1, max(0, int(yt_border))), (255, 0, 0), 1)
    cv2.line(border_overlay, (0, max(0, int(yb_border - 1))), (w - 1, max(0, int(yb_border - 1))), (255, 0, 0), 1)

    border_panel = resize_panel(border_overlay)

    border_region = bin_after[yt_proj:yb_proj, xl_proj:xr_proj] if xl_proj < xr_proj and yt_proj < yb_proj else np.zeros((1, 1), dtype=np.uint8)
    border_h_panel_bgr = _create_border_projection_viz(
        border_region,
        BORDER_REMOVAL_CONFIG,
        max_height=panel_h,
        xl_cut=xl_border - xl_proj,
        xr_cut=xr_border - xl_proj,
        target_width=hist_w
    )

    if crop_after_border is not None and crop_after_border.size > 0:
        crop_h, crop_w = crop_after_border.shape[:2]
        if crop_h > 0 and crop_w > 0:
            scale_factor = min(panel_h / crop_h, base_w / crop_w)
            new_h = int(crop_h * scale_factor)
            new_w = int(crop_w * scale_factor)
            crop_resized = cv2.resize(crop_after_border, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            crop_after_panel = np.full((panel_h, base_w, 3), 255, dtype=np.uint8)
            y_offset = (panel_h - new_h) // 2
            x_offset = (base_w - new_w) // 2
            crop_after_panel[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = crop_resized
        else:
            crop_after_panel = np.full((panel_h, base_w, 3), 250, dtype=np.uint8)
    else:
        crop_after_panel = np.full((panel_h, base_w, 3), 250, dtype=np.uint8)

    noise_diff = cv2.absdiff(gray_original, gray_cleaned)
    noise_overlay = cv2.cvtColor(gray_cleaned, cv2.COLOR_GRAY2BGR)
    noise_mask = (noise_diff > 5).astype(np.uint8)
    noise_overlay[noise_mask > 0] = [0, 0, 255]

    noise_overlay_panel = resize_panel(noise_overlay)
    gray_original_panel = resize_panel(cv2.cvtColor(gray_original, cv2.COLOR_GRAY2BGR))
    gray_cleaned_panel = resize_panel(cv2.cvtColor(gray_cleaned, cv2.COLOR_GRAY2BGR))

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

    try:
        border_panel_bordered = add_border(border_panel)
        border_h_panel_bordered = add_border(border_h_panel_bgr)
        crop_after_panel_bordered = add_border(crop_after_panel)
        border_label_bordered = add_border(border_label)
        border_row = [border_panel_bordered, border_h_panel_bordered, crop_after_panel_bordered, border_label_bordered]

        rows = [
            [add_border(noise_overlay_panel), add_border(gray_original_panel), add_border(gray_cleaned_panel), add_border(noise_label)],
            [add_border(cc_overlay_panel), add_border(mask_before_panel), add_border(mask_after_panel), add_border(cc_label)],
            [add_border(projection_panel), add_border(vertical_panel), add_border(horizontal_panel), add_border(proj_label)],
            border_row
        ]
    except Exception:
        rows = [
            [add_border(noise_overlay_panel), add_border(gray_original_panel), add_border(gray_cleaned_panel), add_border(noise_label)],
            [add_border(cc_overlay_panel), add_border(mask_before_panel), add_border(mask_after_panel), add_border(cc_label)],
            [add_border(projection_panel), add_border(vertical_panel), add_border(horizontal_panel), add_border(proj_label)]
        ]

    grid_rows = []
    for row_panels in rows:
        max_w = max(panel.shape[1] for panel in row_panels)
        padded_panels = [pad_to_width(panel, max_w) for panel in row_panels]
        gap = 6
        gap_col = np.full((padded_panels[0].shape[0], gap, 3), 255, dtype=np.uint8)
        row_img = padded_panels[0]
        for panel in padded_panels[1:]:
            row_img = np.hstack([row_img, gap_col, panel])
        grid_rows.append(row_img)

    try:
        result = np.vstack(grid_rows)
        return result
    except Exception:
        try:
            return np.vstack(grid_rows[:3])
        except Exception:
            return np.full((400, 800, 3), 255, dtype=np.uint8)
