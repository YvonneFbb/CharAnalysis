"""
Noise removal for ancient text images.

Remove light-colored impurities and stains that interfere with text recognition
while preserving legitimate character strokes.
"""
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import cv2


def remove_noise_patches(gray_img: np.ndarray, params: Dict) -> np.ndarray:
    """
    Remove light-colored noise patches from grayscale image.

    Simple strategy for ancient texts:
    1. Find medium-gray pixels (potential noise)
    2. Check if they connect to dark strokes (text edges vs isolated noise)
    3. Remove isolated light patches that don't support text

    Args:
        gray_img: Grayscale image (uint8)
        params: Configuration parameters

    Returns:
        Cleaned grayscale image
    """
    if gray_img is None or gray_img.size == 0:
        return gray_img

    return _connectivity_based_noise_removal(gray_img, params)


def _estimate_local_background(gray_img: np.ndarray, component_mask: np.ndarray,
                              radius: int = 8) -> int:
    """
    Estimate local background color around a component.

    策略：
    1. 膨胀杂质区域，取周围像素
    2. 过滤掉深色像素（可能是笔画或其他杂质）
    3. 使用剩余像素的中位数作为背景色
    4. 如果无法找到合适的背景，返回较亮的默认值
    """
    # Dilate the mask to get surrounding area (增大半径以获取更准确的背景)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1))
    dilated = cv2.dilate(component_mask.astype(np.uint8), kernel)

    # Background is dilated area minus original component
    background_mask = dilated & ~component_mask.astype(np.uint8)

    if background_mask.sum() > 0:
        background_pixels = gray_img[background_mask.astype(bool)]

        # 过滤掉深色像素（可能是笔画或其他杂质），只保留亮色背景像素
        # 阈值设为 150，过滤掉笔画和深色杂质
        light_pixels = background_pixels[background_pixels >= 150]

        if len(light_pixels) > 0:
            # Use median for robustness
            return int(np.median(light_pixels))
        else:
            # 周围都是深色，使用全局背景估算
            return int(np.median(background_pixels))
    else:
        # Fallback: use surrounding region
        y_coords, x_coords = np.where(component_mask)
        if len(y_coords) > 0:
            y_min, y_max = max(0, y_coords.min()-radius), min(gray_img.shape[0], y_coords.max()+radius+1)
            x_min, x_max = max(0, x_coords.min()-radius), min(gray_img.shape[1], x_coords.max()+radius+1)

            surrounding = gray_img[y_min:y_max, x_min:x_max]
            light_surrounding = surrounding[surrounding >= 150]

            if len(light_surrounding) > 0:
                return int(np.median(light_surrounding))
            else:
                return int(np.median(surrounding))

    return 245  # Default light background (更亮的默认值)


def _smart_noise_removal(result: np.ndarray, gray_img: np.ndarray,
                        component_mask: np.ndarray, dark_strokes: np.ndarray,
                        params: Dict) -> np.ndarray:
    """
    智能噪声去除：选择性保留笔画邻近像素

    策略：
    1. 计算杂质区域每个像素到深色笔画的距离
    2. 只移除距离较远的像素（可能是真正的杂质）
    3. 保留紧邻笔画的像素（可能是笔画边缘的自然减淡）

    这样可以避免删除笔画边缘的自然渐变，保持笔画的完整性。
    """
    if dark_strokes.sum() == 0:
        # 没有笔画参考，使用传统方法
        # 使用result而非gray_img估算背景色，避免已处理的杂质影响背景色估算
        background_color = _estimate_local_background(result, component_mask)
        result[component_mask] = background_color
        return result

    # 计算距离变换
    dist_transform = cv2.distanceTransform(1 - dark_strokes, cv2.DIST_L2, 3)

    # 提取杂质区域的距离值
    component_distances = dist_transform[component_mask > 0]

    # 距离阈值：保留距离<threshold的像素（紧邻笔画）
    # 移除距离>=threshold的像素（远离笔画）
    preserve_distance = float(params.get('smart_removal_preserve_distance', 3.0))

    # 创建要移除的像素mask
    y_coords, x_coords = np.where(component_mask)
    removal_mask = np.zeros_like(component_mask, dtype=bool)

    for i, (y, x) in enumerate(zip(y_coords, x_coords)):
        if component_distances[i] >= preserve_distance:
            removal_mask[y, x] = True

    # 如果所有像素都要保留，直接返回
    if not removal_mask.any():
        return result

    # 估算背景色
    # 使用result而非gray_img估算背景色，避免已处理的杂质影响背景色估算
    background_color = _estimate_local_background(result, component_mask)

    # 只移除远离笔画的部分
    result[removal_mask] = background_color

    return result


def _connectivity_based_noise_removal(gray_img: np.ndarray, params: Dict) -> np.ndarray:
    """
    Distance and structure-based noise removal for ancient texts.

    Advanced strategy:
    - Text edges are close to and structurally support dark strokes
    - Noise patches are distant from or structurally disconnected from main strokes
    - Use distance transform and structural analysis
    """
    h, w = gray_img.shape[:2]
    if h == 0 or w == 0:
        return gray_img

    # Parameters
    dark_threshold = int(params.get('dark_stroke_threshold', 100))    # Deep strokes
    light_threshold = int(params.get('light_noise_threshold', 180))   # Light noise candidates
    min_noise_area = int(params.get('min_noise_area', 5))
    max_noise_area = int(params.get('max_noise_area', 200))

    # Create masks
    dark_strokes_raw = (gray_img <= dark_threshold).astype(np.uint8)    # Deep text strokes (raw)

    # 过滤小的深色连通域（可能是噪点而非笔画）
    min_stroke_area = int(params.get('min_stroke_area', 10))
    if min_stroke_area > 0:
        num_strokes, stroke_labels, stroke_stats, _ = cv2.connectedComponentsWithStats(
            dark_strokes_raw, connectivity=8
        )
        dark_strokes = np.zeros_like(dark_strokes_raw)
        for i in range(1, num_strokes):
            area = stroke_stats[i, cv2.CC_STAT_AREA]
            if area >= min_stroke_area:
                dark_strokes[stroke_labels == i] = 1
    else:
        dark_strokes = dark_strokes_raw

    light_candidates = ((gray_img > dark_threshold) &
                       (gray_img < light_threshold)).astype(np.uint8)  # Medium gray candidates

    if light_candidates.sum() == 0:
        return gray_img

    # Find connected components in light candidates
    num, labels, stats, _ = cv2.connectedComponentsWithStats(light_candidates, connectivity=8)

    result = gray_img.copy()

    # For debug visualization
    debug_info = []
    debug_enabled = params.get('debug_visualize', False)

    for i in range(1, num):
        x, y, ww, hh, area = stats[i]

        # Size filtering
        if area < min_noise_area or area > max_noise_area:
            continue

        # Get this component's mask
        component_mask = (labels == i)

        # Advanced analysis: distance + structural relationship
        is_noise, feature_scores = _analyze_component_relationship(
            component_mask, dark_strokes, gray_img, params
        )

        if debug_enabled:
            debug_info.append({
                'bbox': (x, y, ww, hh),
                'area': area,
                'is_noise': is_noise,
                'scores': feature_scores,
                'component_mask': component_mask.copy(),  # 保存实际的component mask
            })

        if is_noise:
            # Smart removal: preserve stroke-adjacent pixels, only remove distant parts
            result = _smart_noise_removal(result, gray_img, component_mask, dark_strokes, params)

    # Generate debug visualization if enabled
    if debug_enabled and debug_info:
        debug_img = _create_noise_debug_visualization(
            gray_img, result, dark_strokes, light_candidates, debug_info, params
        )
        return result, debug_img

    return result


def _analyze_component_relationship(component_mask: np.ndarray, dark_strokes: np.ndarray,
                                   gray_img: np.ndarray, params: Dict) -> Tuple[bool, Dict]:
    """
    基于距离+形态的综合噪声判断

    区分笔画边缘和紧挨笔画的杂质：
    - 笔画边缘：狭长带状，沿笔画轮廓，距离一致，灰度渐变
    - 杂质：块状随机，局部附着，距离不一致，灰度均匀

    返回: (is_noise, feature_scores)
        is_noise: True 表示噪声，False 表示笔画边缘
        feature_scores: 各特征得分字典
    """
    area = np.sum(component_mask)
    if area == 0:
        return True, {}

    # 快速过滤：极大面积直接判定为噪声
    max_area = int(params.get('max_noise_area', 200))
    if area > max_area:
        return True, {'reason': 'too_large'}

    # 快速过滤：极小面积保留（可能是笔画细节）
    min_area = int(params.get('min_noise_area', 5))
    if area < min_area:
        return False, {'reason': 'too_small'}

    # === 特征1: 形态特征（细长度 + 紧凑度） ===
    morphology_score, morphology_details = _compute_morphology_features(component_mask, params)

    # === 特征2: 距离特征（到深色笔画的距离分析） ===
    distance_score, distance_details = _compute_distance_features(component_mask, dark_strokes, params)

    # === 综合判断（只使用形态和距离两个特征） ===
    # 每个特征返回 [0, 1]，0=笔画边缘，1=噪声
    weights = params.get('feature_weights', {
        'morphology': 0.3,
        'distance': 0.7,
    })

    final_score = (
        morphology_score * weights.get('morphology', 0.3) +
        distance_score * weights.get('distance', 0.7)
    )

    # 阈值判断
    threshold = float(params.get('noise_threshold', 0.5))
    is_noise = final_score >= threshold

    # 特征得分
    feature_scores = {
        'morphology': morphology_score,
        'morphology_details': morphology_details,
        'distance': distance_score,
        'distance_details': distance_details,
        'final': final_score,
        'threshold': threshold,
    }

    # Debug输出（可选）
    if params.get('debug_features', False):
        print(f"  Area={area:3d} | Morph={morphology_score:.2f} Dist={distance_score:.2f} | "
              f"Final={final_score:.2f} -> {'NOISE' if is_noise else 'EDGE'}")

    return is_noise, feature_scores


def _compute_morphology_features(component_mask: np.ndarray, params: Dict) -> Tuple[float, Dict]:
    """
    计算形态特征分数 [0, 1]
    0 = 狭长规则（笔画边缘特征）
    1 = 块状不规则（噪声特征）

    使用详细的可配置参数计算三个子特征：
    1. 长宽比 (aspect ratio)
    2. 凸包率 (solidity)
    3. 周长面积比 (perimeter-area ratio)

    返回: (总分, 详细信息字典)
    """
    area = np.sum(component_mask)
    if area == 0:
        return 1.0, {}

    # 获取形态特征配置
    morph_cfg = params.get('morphology', {})

    # 获取边界框
    y_coords, x_coords = np.where(component_mask)
    h_span = y_coords.max() - y_coords.min() + 1
    w_span = x_coords.max() - x_coords.min() + 1

    # === 1. 长宽比 (aspect ratio) ===
    aspect_ratio = max(h_span, w_span) / max(min(h_span, w_span), 1)
    ar_cfg = morph_cfg.get('aspect_ratio', {})
    ar_edge = float(ar_cfg.get('edge_threshold', 5.0))     # >5 为狭长边缘
    ar_noise = float(ar_cfg.get('noise_threshold', 2.0))   # <2 为块状噪声
    ar_weight = float(ar_cfg.get('weight', 0.4))
    # 归一化：>edge_threshold → 0 (edge), <noise_threshold → 1 (noise)
    aspect_score = np.clip((ar_edge - aspect_ratio) / (ar_edge - ar_noise), 0, 1)

    # === 2. 凸包率 (solidity) = 面积 / 凸包面积 ===
    contours, _ = cv2.findContours(component_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solidity = 1.0
    if contours:
        hull = cv2.convexHull(contours[0])
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area

    sol_cfg = morph_cfg.get('solidity', {})
    sol_edge = float(sol_cfg.get('edge_threshold', 0.8))    # >0.8 为规则边缘
    sol_noise = float(sol_cfg.get('noise_threshold', 0.5))  # <0.5 为不规则噪声
    sol_weight = float(sol_cfg.get('weight', 0.3))
    # 归一化：>edge_threshold → 0 (edge), <noise_threshold → 1 (noise)
    solidity_score = np.clip((sol_edge - solidity) / (sol_edge - sol_noise), 0, 1)

    # === 3. 周长面积比 ===
    perimeter = cv2.arcLength(contours[0], True) if contours else 0
    perimeter_area_ratio = (perimeter * perimeter) / max(area, 1)

    pa_cfg = morph_cfg.get('perimeter_area', {})
    pa_edge = float(pa_cfg.get('edge_threshold', 50.0))    # >50 为狭长边缘
    pa_noise = float(pa_cfg.get('noise_threshold', 20.0))  # <20 为紧凑噪声
    pa_weight = float(pa_cfg.get('weight', 0.3))
    # 归一化：>edge_threshold → 0 (edge), <noise_threshold → 1 (noise)
    pa_score = np.clip((pa_edge - perimeter_area_ratio) / (pa_edge - pa_noise), 0, 1)

    # === 加权组合 ===
    total_weight = ar_weight + sol_weight + pa_weight
    if total_weight == 0:
        return 0.5, {}

    morphology_score = (
        aspect_score * ar_weight +
        solidity_score * sol_weight +
        pa_score * pa_weight
    ) / total_weight

    # 返回详细信息
    details = {
        'aspect_ratio': {
            'value': aspect_ratio,
            'score': aspect_score,
            'weight': ar_weight,
        },
        'solidity': {
            'value': solidity,
            'score': solidity_score,
            'weight': sol_weight,
        },
        'perimeter_area': {
            'value': perimeter_area_ratio,
            'score': pa_score,
            'weight': pa_weight,
        },
    }

    return morphology_score, details


def _compute_distance_features(component_mask: np.ndarray, dark_strokes: np.ndarray, params: Dict) -> Tuple[float, Dict]:
    """
    计算距离特征分数 [0, 1]
    0 = 距离一致且接近（笔画边缘）
    1 = 距离不一致或远离（噪声）

    返回: (总分, 详细信息字典)
    """
    if dark_strokes.sum() == 0:
        return 1.0, {}  # 无笔画参考，保守判定为噪声

    # 计算距离变换：每个像素到最近深色笔画的距离
    dist_transform = cv2.distanceTransform(1 - dark_strokes, cv2.DIST_L2, 3)

    # 提取候选区域内的距离值
    distances = dist_transform[component_mask > 0]
    if len(distances) == 0:
        return 1.0, {}

    mean_dist = np.mean(distances)
    std_dist = np.std(distances)

    # 获取距离特征配置
    dist_cfg = params.get('distance', {})

    # === 1. 平均距离特征 ===
    md_cfg = dist_cfg.get('mean_distance', {})
    md_edge = float(md_cfg.get('edge_threshold', 2.0))     # <2 为紧贴笔画
    md_noise = float(md_cfg.get('noise_threshold', 5.0))   # >5 为远离笔画
    md_weight = float(md_cfg.get('weight', 0.5))
    # 归一化：<edge_threshold → 0 (edge), >noise_threshold → 1 (noise)
    # 避免除以零
    if abs(md_noise - md_edge) < 1e-6:
        mean_score = 1.0 if mean_dist >= md_noise else 0.0
    else:
        mean_score = np.clip((mean_dist - md_edge) / (md_noise - md_edge), 0, 1)

    # === 2. 距离变异系数 (CV = std / (mean + 1)) ===
    cv = std_dist / (mean_dist + 1.0)
    cv_cfg = dist_cfg.get('distance_cv', {})
    cv_edge = float(cv_cfg.get('edge_threshold', 0.5))     # <0.5 为一致分布
    cv_noise = float(cv_cfg.get('noise_threshold', 1.0))   # >1.0 为不一致
    cv_weight = float(cv_cfg.get('weight', 0.5))
    # 归一化：<edge_threshold → 0 (edge), >noise_threshold → 1 (noise)
    # 避免除以零
    if abs(cv_noise - cv_edge) < 1e-6:
        cv_score = 1.0 if cv >= cv_noise else 0.0
    else:
        cv_score = np.clip((cv - cv_edge) / (cv_noise - cv_edge), 0, 1)

    # === 加权组合 ===
    total_weight = md_weight + cv_weight
    if total_weight == 0:
        return 0.5, {}

    distance_score = (
        mean_score * md_weight +
        cv_score * cv_weight
    ) / total_weight

    # 返回详细信息
    details = {
        'mean_distance': {
            'value': mean_dist,
            'score': mean_score,
            'weight': md_weight,
        },
        'distance_cv': {
            'value': cv,
            'score': cv_score,
            'weight': cv_weight,
        },
    }

    return distance_score, details


def _compute_gray_features(component_mask: np.ndarray, gray_img: np.ndarray, params: Dict) -> float:
    """
    计算灰度特征分数 [0, 1]
    0 = 有渐变（笔画边缘）
    1 = 均匀（噪声）
    """
    pixels = gray_img[component_mask > 0]
    if len(pixels) == 0:
        return 1.0

    mean_gray = np.mean(pixels)
    std_gray = np.std(pixels)

    # === 灰度特征 ===
    # 笔画边缘：灰度有变化（std > 15），从深到浅的渐变
    # 噪声：灰度均匀（std < 10），且偏浅（mean > 180）

    # 1. 标准差：<10 认为均匀，>20 认为有渐变
    std_score = np.clip((15 - std_gray) / 15, 0, 1)

    # 2. 平均灰度：>180 偏浅（噪声倾向），<150 偏深（笔画倾向）
    mean_score = np.clip((mean_gray - 150) / 60, 0, 1)

    # 综合灰度得分
    gray_score = (std_score * 0.6 + mean_score * 0.4)

    return gray_score


def _compute_angular_coverage(component_mask: np.ndarray, dark_strokes: np.ndarray, params: Dict) -> float:
    """
    计算笔画包围度分数 [0, 1]
    0 = 沿笔画大范围分布（笔画边缘）
    1 = 局部附着（噪声）
    """
    if dark_strokes.sum() == 0:
        return 1.0

    # 找到候选区域的质心
    y_coords, x_coords = np.where(component_mask)
    if len(y_coords) == 0:
        return 1.0

    cy = np.mean(y_coords)
    cx = np.mean(x_coords)

    # 找到附近深色笔画的质心
    # 使用膨胀来扩大笔画区域，找到邻近的笔画
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    dilated_strokes = cv2.dilate(dark_strokes, kernel)

    # 检查候选区域与膨胀笔画的重叠
    overlap = np.sum((component_mask > 0) & (dilated_strokes > 0))
    overlap_ratio = overlap / max(np.sum(component_mask), 1)

    # 计算候选区域在笔画周围的角度分布
    # 找到邻近的笔画像素
    nearby_strokes = dilated_strokes & (cv2.dilate(component_mask.astype(np.uint8), kernel) > 0)
    stroke_y, stroke_x = np.where(nearby_strokes > 0)

    if len(stroke_y) == 0:
        # 没有附近的笔画，很可能是孤立噪声
        return 1.0

    # 计算从候选区域质心到笔画像素的角度分布
    angles = np.arctan2(stroke_y - cy, stroke_x - cx) * 180 / np.pi  # [-180, 180]
    angles = angles + 180  # [0, 360]

    # 计算角度覆盖范围：将360度分成36个bin，统计有多少个bin有分布
    bins = 36
    hist, _ = np.histogram(angles, bins=bins, range=(0, 360))
    occupied_bins = np.sum(hist > 0)
    angular_coverage = occupied_bins / bins

    # === 包围度特征 ===
    # 笔画边缘：重叠率高（>0.5），角度覆盖广（>0.5）
    # 噪声：重叠率低或角度覆盖窄

    # 1. 重叠率分数
    overlap_score = np.clip((0.3 - overlap_ratio) / 0.3, 0, 1)

    # 2. 角度覆盖分数
    coverage_score = np.clip((0.3 - angular_coverage) / 0.3, 0, 1)

    # 综合包围度得分
    final_coverage_score = (overlap_score + coverage_score) / 2.0

    return final_coverage_score


def _create_noise_debug_visualization(gray_img: np.ndarray, result_img: np.ndarray,
                                      dark_strokes: np.ndarray, light_candidates: np.ndarray,
                                      debug_info: list, params: Dict) -> np.ndarray:
    """
    创建噪声去除的调试可视化图像

    重点关注面积最大的几个候选区域，为每个创建详细分析面板
    """
    h, w = gray_img.shape[:2]

    # 统计信息
    total_candidates = len(debug_info)
    noise_count = sum(1 for info in debug_info if info['is_noise'])
    edge_count = total_candidates - noise_count

    # 按面积从大到小排序，只关注前5个最大的
    sorted_info = sorted(debug_info, key=lambda x: x['area'], reverse=True)
    top_candidates = sorted_info[:min(5, len(sorted_info))]

    # === 创建概览面板 ===
    overview_h = 400
    overview_w = 1400
    overview = np.full((overview_h, overview_w, 3), 245, dtype=np.uint8)

    # 标题
    cv2.putText(overview, "NOISE REMOVAL DEBUG - TOP CANDIDATES BY AREA",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(overview, f"Total: {total_candidates} | NOISE: {noise_count} (red) | EDGE: {edge_count} (green) | Threshold: {params.get('noise_threshold', 0.5):.2f}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # 在原图上标注TOP候选区域
    img_annotated = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

    for idx, info in enumerate(top_candidates):
        x, y, ww, hh = info['bbox']
        is_noise = info['is_noise']

        # 颜色
        color = (0, 0, 255) if is_noise else (0, 255, 0)

        # 绘制边界框（细线，避免遮挡）
        cv2.rectangle(img_annotated, (x, y), (x + ww, y + hh), color, 1)

        # 标注ID（缩小字体以减少遮挡）
        label = f"#{idx+1}"
        cv2.putText(img_annotated, label, (x + 2, y + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # 缩放原图到合适大小
    # Overview可用区域：从y=80开始，高度最多320（400-80），宽度最多500
    max_h = 320  # 确保不超出overview边界
    max_w = 500
    scale = min(max_h / h, max_w / w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    img_annotated_resized = cv2.resize(img_annotated, (new_w, new_h))

    # 放置到概览中（确保不会超出边界）
    end_y = min(80 + new_h, overview_h)
    end_x = min(20 + new_w, overview_w)
    actual_h = end_y - 80
    actual_w = end_x - 20
    overview[80:end_y, 20:end_x] = img_annotated_resized[:actual_h, :actual_w]

    # 添加图例说明
    legend_x = new_w + 50
    legend_y = 100
    cv2.putText(overview, "Annotated Image", (legend_x, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    legend_y += 40
    cv2.rectangle(overview, (legend_x, legend_y-15), (legend_x+20, legend_y), (0, 0, 255), -1)
    cv2.putText(overview, "= NOISE (removed)", (legend_x+30, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    legend_y += 30
    cv2.rectangle(overview, (legend_x, legend_y-15), (legend_x+20, legend_y), (0, 255, 0), -1)
    cv2.putText(overview, "= EDGE (kept)", (legend_x+30, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    panels = [overview]

    # === 为每个TOP候选区域创建详细分析面板 ===
    for idx, info in enumerate(top_candidates):
        panel = _create_candidate_detail_panel(idx + 1, info, gray_img, dark_strokes, params)
        panels.append(panel)

    # 垂直拼接所有面板
    final_vis = np.vstack([np.vstack([panel, np.full((10, panel.shape[1], 3), 255, dtype=np.uint8)])
                           for panel in panels[:-1]] + [panels[-1]])

    return final_vis


def _create_candidate_detail_panel(rank: int, info: dict, gray_img: np.ndarray,
                                   dark_strokes: np.ndarray, params: Dict) -> np.ndarray:
    """为单个候选区域创建详细分析面板"""

    x, y, ww, hh = info['bbox']
    area = info['area']
    is_noise = info['is_noise']
    scores = info.get('scores', {})
    component_mask_full = info.get('component_mask', None)  # 实际的component mask

    # 面板尺寸（增加高度以容纳详细子特征）
    panel_w = 1400
    panel_h = 400
    panel = np.full((panel_h, panel_w, 3), 250, dtype=np.uint8)

    # 标题颜色
    title_color = (0, 0, 200) if is_noise else (0, 150, 0)
    result_text = "NOISE (REMOVED)" if is_noise else "EDGE (KEPT)"

    # 标题
    cv2.putText(panel, f"#{rank} - Area: {area} px - {result_text}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, title_color, 2)

    # === 左侧：裁剪的候选区域图像（原图 + 彩色标注图） ===
    crop_size = 180
    crop_x = 20
    crop_y = 50

    # 提取候选区域
    roi = gray_img[y:y+hh, x:x+ww]
    roi_dark_strokes = dark_strokes[y:y+hh, x:x+ww] if dark_strokes is not None else None

    # 提取实际的component mask在ROI中的部分
    roi_component_mask = None
    if component_mask_full is not None:
        roi_component_mask = component_mask_full[y:y+hh, x:x+ww]

    if roi.size > 0:
        # === 原始灰度图 ===
        roi_bgr = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

        # 保持宽高比缩放
        scale = min(crop_size / hh, crop_size / ww)
        new_h = int(hh * scale)
        new_w = int(ww * scale)

        if new_h > 0 and new_w > 0:
            roi_resized = cv2.resize(roi_bgr, (new_w, new_h))

            # 居中放置原图
            y_offset = crop_y + (crop_size - new_h) // 2
            x_offset = crop_x + (crop_size - new_w) // 2

            # 绘制白色背景
            cv2.rectangle(panel, (crop_x, crop_y), (crop_x+crop_size, crop_y+crop_size), (255, 255, 255), -1)
            cv2.rectangle(panel, (crop_x, crop_y), (crop_x+crop_size, crop_y+crop_size), (200, 200, 200), 2)

            # 放置原图
            panel[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = roi_resized

            # === 彩色标注图：显示笔画和候选区域的关系 ===
            colored_x = crop_x + crop_size + 20

            # 创建彩色标注图
            roi_colored = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

            # 在ROI范围内，标注深色笔画（蓝色半透明）
            if roi_dark_strokes is not None and roi_dark_strokes.sum() > 0:
                stroke_mask = roi_dark_strokes > 0
                if stroke_mask.any():
                    stroke_pixels = roi_colored[stroke_mask]
                    if stroke_pixels.size > 0:
                        roi_colored[stroke_mask] = cv2.addWeighted(
                            stroke_pixels, 0.5,
                            np.full_like(stroke_pixels, (255, 100, 0)),  # 蓝色
                            0.5, 0
                        )

            # 标注候选区域（杂质区域）
            # 如果是噪声，用红色/绿色区分保留/删除的像素
            # 如果是边缘，用绿色标注

            # 使用实际的component mask，如果没有则回退到简化计算
            if roi_component_mask is not None and roi_component_mask.any():
                candidate_mask = roi_component_mask.astype(bool)
            else:
                # 回退：基于灰度阈值的简化计算（但不准确）
                dark_threshold = int(params.get('dark_stroke_threshold', 100))
                light_threshold = int(params.get('light_noise_threshold', 180))
                candidate_mask = (roi > dark_threshold) & (roi < light_threshold)

            if is_noise:
                # 噪声区域：计算smart removal的保留/删除决策
                if roi_dark_strokes is not None and roi_dark_strokes.sum() > 0:
                    # 计算距离
                    dist_transform_roi = cv2.distanceTransform(1 - roi_dark_strokes, cv2.DIST_L2, 3)
                    preserve_distance = float(params.get('smart_removal_preserve_distance', 3.0))

                    # 保留区域（绿色）：距离 < preserve_distance
                    preserve_mask = candidate_mask & (dist_transform_roi < preserve_distance)
                    if preserve_mask.any():
                        preserve_pixels = roi_colored[preserve_mask]
                        if preserve_pixels.size > 0:
                            roi_colored[preserve_mask] = cv2.addWeighted(
                                preserve_pixels, 0.3,
                                np.full_like(preserve_pixels, (0, 255, 0)),  # 绿色=保留
                                0.7, 0
                            )

                    # 删除区域（红色）：距离 >= preserve_distance
                    remove_mask = candidate_mask & (dist_transform_roi >= preserve_distance)
                    if remove_mask.any():
                        remove_pixels = roi_colored[remove_mask]
                        if remove_pixels.size > 0:
                            roi_colored[remove_mask] = cv2.addWeighted(
                                remove_pixels, 0.3,
                                np.full_like(remove_pixels, (0, 0, 255)),  # 红色=删除
                                0.7, 0
                            )
                else:
                    # 没有笔画参考，整个候选区域标为红色
                    if candidate_mask.any():
                        cand_pixels = roi_colored[candidate_mask]
                        if cand_pixels.size > 0:
                            roi_colored[candidate_mask] = cv2.addWeighted(
                                cand_pixels, 0.3,
                                np.full_like(cand_pixels, (0, 0, 255)),  # 红色
                                0.7, 0
                            )
            else:
                # 边缘区域：标注为绿色
                if candidate_mask.any():
                    edge_pixels = roi_colored[candidate_mask]
                    if edge_pixels.size > 0:
                        roi_colored[candidate_mask] = cv2.addWeighted(
                            edge_pixels, 0.3,
                            np.full_like(edge_pixels, (0, 255, 0)),  # 绿色=保留的边缘
                            0.7, 0
                        )

            # 缩放彩色标注图
            roi_colored_resized = cv2.resize(roi_colored, (new_w, new_h))

            # 放置彩色标注图
            cv2.rectangle(panel, (colored_x, crop_y), (colored_x+crop_size, crop_y+crop_size), (255, 255, 255), -1)
            cv2.rectangle(panel, (colored_x, crop_y), (colored_x+crop_size, crop_y+crop_size), (200, 200, 200), 2)
            panel[y_offset:y_offset+new_h, colored_x+x_offset-crop_x:colored_x+x_offset-crop_x+new_w] = roi_colored_resized

    # 标签
    cv2.putText(panel, "Original", (crop_x, crop_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(panel, "Color-coded", (crop_x + crop_size + 20, crop_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # 添加颜色图例
    legend_y = crop_y + crop_size + 15
    legend_x = crop_x
    cv2.putText(panel, "Legend:", (legend_x, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    legend_y += 18
    # 蓝色 = 笔画
    cv2.rectangle(panel, (legend_x, legend_y-10), (legend_x+15, legend_y), (255, 100, 0), -1)
    cv2.putText(panel, "= Dark strokes", (legend_x+20, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    legend_y += 18
    # 绿色 = 保留
    cv2.rectangle(panel, (legend_x, legend_y-10), (legend_x+15, legend_y), (0, 255, 0), -1)
    cv2.putText(panel, "= Preserved (close)", (legend_x+20, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    legend_y += 18
    # 红色 = 删除
    cv2.rectangle(panel, (legend_x, legend_y-10), (legend_x+15, legend_y), (0, 0, 255), -1)
    cv2.putText(panel, "= Removed (far)", (legend_x+20, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

    # === 右侧：特征得分详情 ===
    info_x = 440  # 调整到两个图像之后
    info_y = 70
    line_h = 35

    # 获取各项得分（只保留morphology和distance）
    morph = scores.get('morphology', 0)
    morph_details = scores.get('morphology_details', {})
    dist = scores.get('distance', 0)
    dist_details = scores.get('distance_details', {})
    final = scores.get('final', 0)
    threshold = scores.get('threshold', params.get('noise_threshold', 0.5))

    # 特征得分说明（只保留两个核心特征）
    feature_info = [
        ("Morphology Score:", morph, "0=elongated/regular, 1=blocky/irregular"),
        ("Distance Score:", dist, "0=close to strokes, 1=far/inconsistent"),
    ]

    for i, (label, score, desc) in enumerate(feature_info):
        y_pos = info_y + i * line_h

        # 标签
        cv2.putText(panel, label, (info_x, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # 得分（加粗）
        cv2.putText(panel, f"{score:.3f}", (info_x + 200, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, title_color, 2)

        # 描述
        cv2.putText(panel, desc, (info_x + 300, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

    # === 显示子特征详细数值 ===
    detail_y = info_y + len(feature_info) * line_h + 15
    detail_line_h = 25

    # Morphology 子特征
    if morph_details:
        cv2.putText(panel, "  - Morphology Details:", (info_x + 20, detail_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)
        detail_y += detail_line_h

        for sub_name, sub_data in morph_details.items():
            value = sub_data.get('value', 0)
            score = sub_data.get('score', 0)
            weight = sub_data.get('weight', 0)

            # 格式化显示名称
            display_name = sub_name.replace('_', ' ').title()
            cv2.putText(panel, f"    {display_name}:", (info_x + 40, detail_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
            cv2.putText(panel, f"val={value:.2f}, score={score:.3f}, wt={weight:.1f}",
                       (info_x + 250, detail_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
            detail_y += detail_line_h

    # Distance 子特征
    if dist_details:
        cv2.putText(panel, "  - Distance Details:", (info_x + 20, detail_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)
        detail_y += detail_line_h

        for sub_name, sub_data in dist_details.items():
            value = sub_data.get('value', 0)
            score = sub_data.get('score', 0)
            weight = sub_data.get('weight', 0)

            # 格式化显示名称
            display_name = sub_name.replace('_', ' ').title()
            cv2.putText(panel, f"    {display_name}:", (info_x + 40, detail_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
            cv2.putText(panel, f"val={value:.2f}, score={score:.3f}, wt={weight:.1f}",
                       (info_x + 250, detail_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
            detail_y += detail_line_h

    # 分隔线
    sep_y = detail_y + 10
    cv2.line(panel, (info_x, sep_y), (panel_w - 20, sep_y), (150, 150, 150), 2)

    # 最终得分
    final_y = sep_y + 30
    cv2.putText(panel, "FINAL SCORE:", (info_x, final_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(panel, f"{final:.3f}", (info_x + 200, final_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, title_color, 2)

    # 阈值和判断
    cv2.putText(panel, f"(threshold = {threshold:.2f})", (info_x + 320, final_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    # 判断箭头
    if final >= threshold:
        cv2.putText(panel, f"-> {final:.3f} >= {threshold:.2f} => NOISE",
                   (info_x + 500, final_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)
    else:
        cv2.putText(panel, f"-> {final:.3f} < {threshold:.2f} => EDGE",
                   (info_x + 500, final_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 2)

    return panel
