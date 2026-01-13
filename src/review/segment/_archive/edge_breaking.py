"""
Edge adhesion breaking for ancient text segmentation.

Intelligently detect and break thin connections at image edges that typically
represent adhesion to borders or neighboring characters, while preserving
legitimate character strokes.
"""
from __future__ import annotations
from typing import Dict
import numpy as np
import cv2




def weak_connection_breaking(bin_img: np.ndarray, params: Dict) -> np.ndarray:
    """
    Advanced approach: detect and break weak connections between strokes.

    Strategy:
    1. Skeleton analysis to find connection points
    2. Local thickness analysis to identify thin connections
    3. Selective breaking of weak connections while preserving stroke structure
    """
    if bin_img is None or bin_img.size == 0:
        return bin_img

    h, w = bin_img.shape[:2]
    if h == 0 or w == 0:
        return bin_img

    # Parameters
    min_connection_width = int(params.get('min_connection_width', 2))  # 最小连接宽度
    thickness_ratio_threshold = float(params.get('thickness_ratio_threshold', 0.3))  # 厚度比阈值
    min_component_area = int(params.get('min_remaining_area', 10))

    # Step 1: Get binary mask
    mask = (bin_img > 0).astype(np.uint8)

    # Step 2: Distance transform to analyze thickness
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 3)

    # Step 3: Find skeleton
    skeleton = _compute_skeleton(mask)

    # Step 4: Analyze connection points on skeleton
    result = bin_img.copy()
    weak_connections = _find_weak_connections(
        skeleton, dist_transform,
        min_connection_width, thickness_ratio_threshold
    )

    # Step 5: Break weak connections
    for connection_point in weak_connections:
        result = _break_at_point(result, connection_point, min_component_area)

    return result


def _compute_skeleton(binary_mask: np.ndarray) -> np.ndarray:
    """计算二值图像的骨架"""
    skeleton = np.zeros_like(binary_mask)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.erode(binary_mask, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(binary_mask, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        binary_mask = eroded.copy()

        if cv2.countNonZero(binary_mask) == 0:
            break

    return skeleton


def _find_weak_connections(skeleton: np.ndarray, dist_transform: np.ndarray,
                          min_width: int, thickness_ratio: float) -> list:
    """在骨架上找到脆弱的连接点"""
    weak_points = []

    # 在骨架上寻找连接点（有3个或更多邻居的点）
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)

    # 计算每个骨架点的邻居数
    neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)

    # 找到连接点（邻居数>=3的骨架点）
    y_coords, x_coords = np.where((skeleton > 0) & (neighbor_count >= 3))

    for y, x in zip(y_coords, x_coords):
        # 分析该点周围的厚度
        local_thickness = dist_transform[y, x]

        # 分析周围区域的平均厚度
        region_size = 5
        y_min, y_max = max(0, y-region_size), min(dist_transform.shape[0], y+region_size+1)
        x_min, x_max = max(0, x-region_size), min(dist_transform.shape[1], x+region_size+1)

        region_thickness = dist_transform[y_min:y_max, x_min:x_max]
        avg_thickness = np.mean(region_thickness[region_thickness > 0])

        # 如果当前点厚度明显小于周围平均厚度，可能是脆弱连接
        if (local_thickness < min_width and
            local_thickness < avg_thickness * thickness_ratio):
            weak_points.append((y, x, local_thickness))

    return weak_points


def _break_at_point(bin_img: np.ndarray, connection_point: tuple,
                   min_area: int) -> np.ndarray:
    """在指定连接点进行断开"""
    y, x, thickness = connection_point

    # 创建一个小的断开区域
    break_radius = max(1, int(thickness + 1))

    # 临时断开
    temp_result = bin_img.copy()
    cv2.circle(temp_result, (x, y), break_radius, 0, -1)

    # 检查断开后的连通性
    mask = (temp_result > 0).astype(np.uint8)
    num_components, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # 只有当断开后产生的组件都足够大时才应用断开
    valid_break = True
    for i in range(1, num_components):
        if stats[i][4] < min_area:  # 面积太小
            valid_break = False
            break

    if valid_break and num_components > 1:
        return temp_result
    else:
        return bin_img


def break_edge_adhesions(bin_img: np.ndarray, params: Dict) -> np.ndarray:
    """
    Main entry point for weak connection breaking.

    Uses advanced skeleton-based analysis to detect and break weak connections
    between strokes while preserving legitimate character structure.
    """
    return weak_connection_breaking(bin_img, params)