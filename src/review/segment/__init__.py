"""
字符切割模块

完全使用 ref/charsis/src/segmentation 的实现
"""

from .core import segment_character, adjust_bbox
from .config import (
    NOISE_REMOVAL_CONFIG,
    CC_FILTER_CONFIG,
    PROJECTION_TRIM_CONFIG,
    BORDER_REMOVAL_CONFIG,
    get_default_params,
    merge_params
)

__all__ = [
    'segment_character',
    'adjust_bbox',
    'get_default_params',
    'merge_params',
    'NOISE_REMOVAL_CONFIG',
    'CC_FILTER_CONFIG',
    'PROJECTION_TRIM_CONFIG',
    'BORDER_REMOVAL_CONFIG',
]
