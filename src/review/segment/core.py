"""
字符切割核心模块 - 包装器

参考 ref/charsis/src/segmentation 的实现进行整理
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional

from .config import merge_params
from .projection_trim import trim_projection_from_bin, binarize
from .cc_filter import refine_binary_components
from .border_removal import trim_border_from_bin, remove_border_frames
from .noise_removal import remove_noise_patches
from .debug_viz import _render_combined_debug


def segment_character(
    preprocessed_image_path: str,
    bbox: Dict[str, int],
    custom_params: Optional[Dict] = None,
    padding: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict, np.ndarray]:
    """
    对单个字符进行精确切割（参考 ref/charsis/src/segmentation）

    Args:
        preprocessed_image_path: 预处理图片路径
        bbox: OCR bbox (keys: x, y, width, height)
        custom_params: 自定义参数（可选）
        padding: bbox 扩展的边距（像素，默认10）

    Returns:
        roi_image: 原始 ROI 图像（彩色）
        segmented_image: 切割后的字符图像（彩色，应用了CC效果）
        debug_image: 调试可视化图像（4行布局）
        metadata: 元数据（bbox、参数等）
        processed_roi: 经过noise+CC处理后的ROI图像（用于bbox预览）
    """
    # 加载图片
    img = cv2.imread(preprocessed_image_path)
    if img is None:
        raise ValueError(f"无法加载图片: {preprocessed_image_path}")

    H, W = img.shape[:2]

    # 合并自定义参数（保持默认配置不被修改）
    merged = merge_params(custom_params or {})
    noise_cfg = merged['noise_removal']
    cc_cfg = merged['cc_filter']
    proj_cfg = merged['projection_trim']
    border_cfg = merged['border_removal']
    final_pad = int(custom_params.get('final_pad', 0)) if custom_params else 0

    # 提取 ROI（带 padding）
    x0 = max(0, bbox['x'] - padding)
    y0 = max(0, bbox['y'] - padding)
    x1 = min(W, bbox['x'] + bbox['width'] + padding)
    y1 = min(H, bbox['y'] + bbox['height'] + padding)
    
    roi = img[y0:y1, x0:x1]
    if roi.size == 0:
        raise ValueError(f"ROI 区域为空: bbox={bbox}, padding={padding}")

    roi_h, roi_w = roi.shape[:2]
    if roi_w < 10 or roi_h < 10:
        raise ValueError(f"ROI 区域太小: {roi_w}x{roi_h}")

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # ===== 完全按照参考代码的流程 =====

    # 1. Noise patch removal (before binarization)
    gray_cleaned = gray.copy()
    if noise_cfg.get('enabled', True):
        result = remove_noise_patches(gray_cleaned, noise_cfg)
        if isinstance(result, tuple):
            gray_cleaned = result[0]
        else:
            gray_cleaned = result

    # 2. Binarization
    bin_before = binarize(
        gray_cleaned,
        mode=str(proj_cfg.get('binarize', 'otsu')).lower(),
        adaptive_block=int(proj_cfg.get('adaptive_block', 31)),
        adaptive_C=int(proj_cfg.get('adaptive_C', 3)),
    )

    # 3. CC filtering
    bin_after = bin_before.copy()
    if cc_cfg.get('enabled', True):
        bin_after, _ = refine_binary_components(bin_after, cc_cfg, gray_cleaned)
    else:
        # Skip CC filtering, use binarized image directly
        pass

    # 3.5. Remove L-shaped border frames near edges (prevents projection from sticking to frames)
    if border_cfg.get('frame_removal', {}).get('enabled', False):
        bin_after = remove_border_frames(bin_after, border_cfg)

    # 4. Projection trimming
    if proj_cfg.get('enabled', True):
        xl, xr, yt, yb = trim_projection_from_bin(bin_after, proj_cfg)
    else:
        # Skip projection trimming, use full ROI
        xl, xr, yt, yb = 0, roi_w, 0, roi_h

    # Store projection coordinates for debug
    xl_proj, xr_proj, yt_proj, yb_proj = xl, xr, yt, yb

    # 5. Border removal (after projection trimming)
    xl_border, xr_border, yt_border, yb_border = xl, xr, yt, yb
    if border_cfg.get('enabled', True):
        # Apply border trimming to the projection-trimmed region
        border_bin = bin_after[yt:yb, xl:xr]
        xl_b, xr_b, yt_b, yb_b = trim_border_from_bin(border_bin, border_cfg)
        # Adjust coordinates back to full ROI space
        xl_border = xl + xl_b
        xr_border = xl + xr_b
        yt_border = yt + yt_b
        yb_border = yt + yb_b

    # 6. Apply final padding
    fp = int(final_pad)
    xl_final = max(0, xl_border - fp)
    xr_final = min(roi.shape[1], xr_border + fp)
    yt_final = max(0, yt_border - fp)
    yb_final = min(roi.shape[0], yb_border + fp)

    # 7. Create final processed image: apply NOISE removal + CC filtering effects
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
    segmented_image = crop_after_border

    # 8. Generate debug image (4-row layout)
    debug_image = _render_combined_debug(
        roi, gray, gray_cleaned, bin_before, bin_after,
        crop_before_border, crop_after_border,
        int(xl_proj), int(xr_proj), int(yt_proj), int(yb_proj),
        int(xl_border), int(xr_border), int(yt_border), int(yb_border)
    )

    # 9. Metadata
    metadata = {
        'original_bbox': bbox,
        'roi_bbox': {'x': x0, 'y': y0, 'width': x1-x0, 'height': y1-y0},
        'roi_shape': roi.shape,  # ROI图片的实际尺寸 (height, width, channels)
        'segmented_bbox': {
            'x': int(xl_final),
            'y': int(yt_final),
            'width': int(xr_final - xl_final),
            'height': int(yb_final - yt_final)
        },
        'segmented_bbox_absolute': {
            'x': x0 + int(xl_final),
            'y': y0 + int(yt_final),
            'width': int(xr_final - xl_final),
            'height': int(yb_final - yt_final)
        },
        'params_used': {
            'noise_removal': noise_cfg,
            'cc_filter': cc_cfg,
            'projection_trim': proj_cfg,
            'border_removal': border_cfg,
            'final_pad': final_pad
        }
    }

    return roi, segmented_image, debug_image, metadata, processed_roi


def adjust_bbox(
    preprocessed_image_path: str,
    original_bbox: Dict[str, int],
    adjusted_bbox: Dict[str, int]
) -> Tuple[np.ndarray, Dict]:
    """
    根据调整后的 bbox 重新裁切

    Args:
        preprocessed_image_path: 预处理图片路径
        original_bbox: 原始 OCR bbox
        adjusted_bbox: 调整后的 bbox（绝对坐标）

    Returns:
        segmented_image: 裁切后的图像
        metadata: 元数据
    """
    img = cv2.imread(preprocessed_image_path)
    if img is None:
        raise ValueError(f"无法加载图片: {preprocessed_image_path}")

    h, w = img.shape[:2]

    # 确保 bbox 在图像范围内
    x = max(0, min(w-1, adjusted_bbox['x']))
    y = max(0, min(h-1, adjusted_bbox['y']))
    x2 = max(x+1, min(w, adjusted_bbox['x'] + adjusted_bbox['width']))
    y2 = max(y+1, min(h, adjusted_bbox['y'] + adjusted_bbox['height']))

    segmented_image = img[y:y2, x:x2]

    metadata = {
        'original_bbox': original_bbox,
        'adjusted_bbox': adjusted_bbox,
        'method': 'manual_bbox'
    }

    return segmented_image, metadata
