"""
字符切割参数配置

复用自旧代码的 segmentation 参数配置
"""

# 投影修剪配置
PROJECTION_TRIM_CONFIG = {
    'binarize': 'otsu',                 # 投影前的二值化方式
    'adaptive_block': 31,               # 自适应阈值块大小
    'adaptive_C': 3,                    # 自适应阈值常数
    'run_min_coverage_ratio': 0.01,     # run 判定：覆盖度占最大值的比例
    'run_min_coverage_abs': 0.005,      # run 判定：覆盖度绝对下限
    'primary_run_min_mass_ratio': 0.5,  # 主 run 需占投影质量的比例
    'primary_run_min_length_ratio': 0.3,# 主 run 需占宽度的比例
    'tighten_min_coverage': 0.01,       # 主 run 内重新贴边的阈值

    # 检测范围参数 - 在多大范围内寻找内容边界
    'detection_range': {
        'left_ratio': 0.3,      # 左侧检测范围占总宽度的比例
        'right_ratio': 0.3,     # 右侧检测范围占总宽度的比例
        'top_ratio': 0.3,       # 上侧检测范围占总高度的比例
        'bottom_ratio': 0.5,    # 下侧检测范围占总高度的比例
    },

    # 切割限制参数 - 最多允许切掉多少内容（不包括空白）
    'cut_limits': {
        'left_max_ratio': 0.25,      # 左侧最大切割比例
        'right_max_ratio': 0.25,     # 右侧最大切割比例
        'top_max_ratio': 0.25,      # 上侧最大切割比例
        'bottom_max_ratio': 0.3,    # 下侧最大切割比例
    },
}

# 连通组件过滤配置
CC_FILTER_CONFIG = {
    'border_touch_margin': 1,           # 实际触边判定范围（像素）
    'edge_zone_margin': 2,              # 边缘区域判定范围（像素）
    'border_touch_min_area_ratio': 0.04,# 触边组件最小面积比例
    'edge_zone_min_area_ratio': 0.01,   # 边缘区域组件最小面积比例
    'interior_min_area_ratio': 0.002,   # 内部组件最小面积比例
    'max_aspect_for_edge': 6.0,         # 边缘/触边组件最大长宽比
    'min_dim_px': 2,                    # 边缘/触边组件最小尺寸（宽度或高度的最小值，像素）
    'interior_min_dim_px': 1,           # 内部组件最小尺寸（宽度或高度的最小值，像素）

    'debug_visualize': True,            # 是否生成详细的 CC debug 图
}

# 边框去除配置
BORDER_REMOVAL_CONFIG = {
    'enabled': True,                    # 是否启用边框去除
    'max_iterations': 5,                # 最大迭代次数（多次执行以完全去除边框）

    # 预清理：移除靠边的 L/回字框结构（更稳健）
    'frame_removal': {
        'enabled': True,
        'edge_margin_ratio': 0.08,      # 只处理靠边区域（占比）
        'min_length_ratio': 0.7,        # 线段最小长度占比
        'min_length_px': 12,            # 线段最小像素长度
        'max_thickness_px': 3,          # 最大线宽（像素）
        'min_corner_count': 1,          # 至少命中几个角点才触发
        'max_removal_ratio': 0.2,       # 最多允许移除的前景比例
    },

    # 水平边框检测参数
    'border_max_width_ratio': 0.15,      # 最大边框宽度占比（左右两侧检测范围）
    'border_threshold_ratio': 0.35,      # 边框检测阈值（相对于最大投影值的比例）

    # 突变检测参数
    'spike_min_length_ratio': 0.02,     # 异常高值段最小长度占检测范围的比例
    'spike_max_length_ratio': 0.1,     # 异常高值段最大长度占检测范围的比例
    'spike_gradient_threshold': 0.4,    # 突变梯度阈值（相对于最大投影值）
    'spike_prominence_ratio': 0.5,      # 突出度阈值（峰值相对于周围的突出程度）
    'edge_tolerance': 2,                # 允许的边缘偏移像素数

    # 垂直边框处理参数
    'vertical_detection_range': {
        'top_ratio': 0.3,       # 上侧检测范围占总高度的比例
        'bottom_ratio': 0.3,    # 下侧检测范围占总高度的比例
    },

    'vertical_cut_limits': {
        'top_max_ratio': 0.2,       # 上侧最大切割比例
        'bottom_max_ratio': 0.2,    # 下侧最大切割比例
    },

    'debug_verbose': True,              # 是否输出详细的border debug图
}

# 噪声去除配置
NOISE_REMOVAL_CONFIG = {
    'enabled': True,                   # 是否启用杂质色块清理

    # 阈值参数
    'dark_stroke_threshold': 60,       # 深色笔画阈值（<=此值认为是文字主体）
    'light_noise_threshold': 240,      # 淡色杂质阈值（>深色且<此值的区域为杂质候选）
    'min_stroke_area': 6,             # 最小笔画面积（过滤小于此值的深色点，避免误判为笔画）
    'min_noise_area': 2,               # 最小杂质面积（太小的保留）
    'max_noise_area': 200000,          # 最大杂质面积（太大的直接判定为噪声）

    # 综合判断参数
    'noise_threshold': 0.4,            # 综合得分阈值（>此值判定为噪声）

    # 智能去除参数
    'smart_removal_preserve_distance': 1.0,  # 保留距离阈值（像素）：<此距离的像素视为笔画边缘保留

    # === 形态特征详细参数 ===
    'morphology': {
        'aspect_ratio': {
            'edge_threshold': 5.0,      # 长宽比 > 此值 → 狭长边缘特征
            'noise_threshold': 2.0,     # 长宽比 < 此值 → 块状噪声特征
            'weight': 0.4,              # 长宽比在形态特征中的权重
        },
        'solidity': {
            'edge_threshold': 0.8,      # 凸包率 > 此值 → 规则边缘
            'noise_threshold': 0.5,     # 凸包率 < 此值 → 不规则噪声
            'weight': 0.3,              # 凸包率在形态特征中的权重
        },
        'perimeter_area': {
            'edge_threshold': 50.0,     # 周长面积比 > 此值 → 狭长边缘
            'noise_threshold': 20.0,    # 周长面积比 < 此值 → 紧凑噪声
            'weight': 0.3,              # 周长面积比在形态特征中的权重
        },
    },

    # === 距离特征详细参数 ===
    'distance': {
        'mean_distance': {
            'edge_threshold': 1.5,      # 平均距离 < 此值(像素) → 紧贴笔画
            'noise_threshold': 2.0,     # 平均距离 > 此值(像素) → 远离笔画
            'weight': 0.5,              # 平均距离在距离特征中的权重
        },
        'distance_cv': {
            'edge_threshold': 0.2,      # 距离变异系数 < 此值 → 一致分布
            'noise_threshold': 0.25,     # 距离变异系数 > 此值 → 不一致分布
            'weight': 0.5,              # 变异系数在距离特征中的权重
        },
    },

    # === 特征权重（只保留形态和距离） ===
    'feature_weights': {
        'morphology': 0.4,             # 形态特征总权重
        'distance': 0.6,               # 距离特征总权重
    },

    # Debug参数
    'debug_features': False,           # 是否输出每个区域的特征得分（控制台）
    'debug_visualize': True,          # 是否生成可视化debug图像（保存到debug目录）
}


def get_default_params():
    """获取所有默认参数"""
    import copy
    return {
        'noise_removal': copy.deepcopy(NOISE_REMOVAL_CONFIG),
        'projection_trim': copy.deepcopy(PROJECTION_TRIM_CONFIG),
        'cc_filter': copy.deepcopy(CC_FILTER_CONFIG),
        'border_removal': copy.deepcopy(BORDER_REMOVAL_CONFIG),
    }


def merge_params(custom_params):
    """合并自定义参数和默认参数"""
    default = get_default_params()

    if not custom_params:
        return default

    # 深度合并
    for category in ['noise_removal', 'projection_trim', 'cc_filter', 'border_removal']:
        if category in custom_params:
            for key, value in custom_params[category].items():
                if isinstance(value, dict) and isinstance(default[category].get(key), dict):
                    default[category][key].update(value)
                else:
                    default[category][key] = value

    return default
