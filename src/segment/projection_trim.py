"""Projection trimming based on dominant run (horizontal + vertical)."""
from __future__ import annotations
from typing import Tuple, Dict, List
import numpy as np
import cv2


def binarize(gray: np.ndarray, mode: str = 'otsu', adaptive_block: int = 31, adaptive_C: int = 3) -> np.ndarray:
    if mode == 'adaptive':
        if adaptive_block % 2 == 0:
            adaptive_block += 1
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV, adaptive_block, adaptive_C)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bin_img


def _find_runs(coverage: np.ndarray, min_cov: float) -> List[Tuple[int, int, float]]:
    runs: List[Tuple[int, int, float]] = []
    length = coverage.shape[0]
    i = 0
    while i < length:
        if coverage[i] <= min_cov:
            i += 1
            continue
        j = i
        total = 0.0
        while j < length and coverage[j] > min_cov:
            total += coverage[j]
            j += 1
        runs.append((i, j, total))
        i = j
    return runs


def _select_primary_run(runs: List[Tuple[int, int, float]], coverage: np.ndarray, total_len: int,
                         params: Dict) -> Tuple[int, int]:
    if not runs:
        return 0, total_len
    masses = [mass for _, _, mass in runs]
    lengths = [(end - start) for start, end, _ in runs]
    total_mass = sum(masses)

    idx = int(np.argmax(masses))
    primary_start, primary_end, _ = runs[idx]

    min_mass_ratio = float(params.get('primary_run_min_mass_ratio', 0.5))
    min_len_ratio = float(params.get('primary_run_min_length_ratio', 0.3))
    if masses[idx] < total_mass * min_mass_ratio or (primary_end - primary_start) < total_len * min_len_ratio:
        idx = int(np.argmax(lengths))
        primary_start, primary_end, _ = runs[idx]

    tighten_cov = float(params.get('tighten_min_coverage', 0.01))
    while primary_start < primary_end and coverage[primary_start] <= tighten_cov:
        primary_start += 1
    while primary_end > primary_start and coverage[primary_end - 1] <= tighten_cov:
        primary_end -= 1

    primary_start = int(max(0, min(total_len - 1, primary_start)))
    primary_end = int(max(primary_start + 1, min(total_len, primary_end)))
    return primary_start, primary_end


def _trim_axis(mask: np.ndarray, axis: int, params: Dict) -> Tuple[int, int]:
    if axis == 0:  # columns
        coverage = mask.sum(axis=0).astype(np.float32) / float(max(1, mask.shape[0]) * 255.0)
        total_len = mask.shape[1]
        # Parameter structure
        detection_left_key = ('detection_range', 'left_ratio')
        detection_right_key = ('detection_range', 'right_ratio')
        cut_left_key = ('cut_limits', 'left_max_ratio')
        cut_right_key = ('cut_limits', 'right_max_ratio')
    else:  # rows
        coverage = mask.sum(axis=1).astype(np.float32) / float(max(1, mask.shape[1]) * 255.0)
        total_len = mask.shape[0]
        # Parameter structure
        detection_left_key = ('detection_range', 'top_ratio')
        detection_right_key = ('detection_range', 'bottom_ratio')
        cut_left_key = ('cut_limits', 'top_max_ratio')
        cut_right_key = ('cut_limits', 'bottom_max_ratio')

    max_cov = coverage.max()
    if max_cov <= 0.0:
        return 0, total_len
    min_cov = max(float(params.get('run_min_coverage_abs', 0.005)),
                   float(params.get('run_min_coverage_ratio', 0.01)) * max_cov)

    # Get nested parameters
    def get_nested_param(keys, default):
        if len(keys) == 2 and keys[0] in params and isinstance(params[keys[0]], dict):
            return float(params[keys[0]].get(keys[1], default))
        return default

    # Detection range parameters
    detection_left_ratio = get_nested_param(detection_left_key, 0.3)
    detection_right_ratio = get_nested_param(detection_right_key, 0.3)

    # Cut limit parameters
    cut_left_ratio = get_nested_param(cut_left_key, 0.2)
    cut_right_ratio = get_nested_param(cut_right_key, 0.2)

    # Calculate actual limits
    detection_left_limit = int(round(total_len * min(detection_left_ratio, 1.0)))
    detection_right_limit = int(round(total_len * min(detection_right_ratio, 1.0)))
    left_trim_limit = total_len if cut_left_ratio <= 0 else int(round(total_len * min(cut_left_ratio, 1.0)))
    right_trim_limit = total_len if cut_right_ratio <= 0 else int(round(total_len * min(cut_right_ratio, 1.0)))

    runs = _find_runs(coverage, min_cov)

    if not runs:
        # 空列，允许裁掉到限制位置
        left = min(left_trim_limit, total_len)
        right = max(left + 1, total_len - right_trim_limit)
        return max(0, min(total_len - 1, left)), min(total_len, max(left + 1, right))

    tighten_cov = float(params.get('tighten_min_coverage', 0.01))

    # 新逻辑：在detection_range内查找离散块，若块的宽度≤cut_limits则切掉

    # 左侧处理：查找detection_range内的runs，判断是否应该切掉
    left = 0
    for i, (run_start, run_end, run_mass) in enumerate(runs):
        # 如果run完全在detection_range内
        if run_end <= detection_left_limit:
            # 计算这个run的宽度
            run_width = run_end - run_start
            # 如果run宽度 <= 总长度 * cut_limit，则切掉（从下一个run开始）
            if run_width <= total_len * cut_left_ratio:
                # 继续检查下一个run
                continue
            else:
                # 这个run太大，保留它，从它开始
                left = run_start
                break
        else:
            # run超出了detection_range，从它开始
            left = run_start
            break
    else:
        # 所有runs都在detection_range内且都被切掉了
        left = runs[-1][1] if runs else 0

    # 右侧处理：从右往左查找
    right = total_len
    for i in range(len(runs) - 1, -1, -1):
        run_start, run_end, run_mass = runs[i]
        # 如果run完全在detection_range内（从右侧算）
        if run_start >= total_len - detection_right_limit:
            # 计算这个run的宽度
            run_width = run_end - run_start
            # 如果run宽度 <= 总长度 * cut_limit，则切掉
            if run_width <= total_len * cut_right_ratio:
                # 继续检查前一个run
                continue
            else:
                # 这个run太大，保留它，到它结束
                right = run_end
                break
        else:
            # run超出了detection_range，到它结束
            right = run_end
            break
    else:
        # 所有runs都在detection_range内且都被切掉了
        right = runs[0][0] if runs else total_len

    # 精细调整：去除边缘的低覆盖像素
    while left < total_len and coverage[left] <= tighten_cov:
        left += 1
    while right > 0 and coverage[right - 1] <= tighten_cov:
        right -= 1

    left = int(max(0, min(total_len - 1, left)))
    right = int(max(left + 1, min(total_len, right)))
    return left, right


def _trim_edges_by_projection(mask: np.ndarray, params: Dict) -> Tuple[int, int, int, int]:
    xl, xr = _trim_axis(mask, axis=0, params=params)
    cropped = mask[:, xl:xr]
    yt, yb = _trim_axis(cropped, axis=1, params=params)
    return xl, xr, yt, yb


def trim_projection(gray: np.ndarray, params: Dict, debug: bool = False) -> Tuple[int, int, int, int]:
    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        return 0, w, 0, h
    bin_img = binarize(gray, mode=str(params.get('binarize', 'otsu')).lower(),
                       adaptive_block=int(params.get('adaptive_block', 31)),
                       adaptive_C=int(params.get('adaptive_C', 3)))
    mask = (bin_img > 0).astype(np.uint8) * 255
    xl, xr, yt, yb = _trim_edges_by_projection(mask, params)
    return xl, xr, yt, yb


def trim_projection_from_bin(bin_img: np.ndarray, params: Dict) -> Tuple[int, int, int, int]:
    h, w = bin_img.shape[:2]
    if h == 0 or w == 0:
        return 0, w, 0, h
    mask = (bin_img > 0).astype(np.uint8) * 255
    xl, xr, yt, yb = _trim_edges_by_projection(mask, params)
    return xl, xr, yt, yb

