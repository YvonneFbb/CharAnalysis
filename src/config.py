"""
项目配置管理

包含路径配置、预处理参数、OCR 参数等
"""
import os
from pathlib import Path

# ============================================================================
# 路径配置
# ============================================================================

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# 数据目录
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DIR = DATA_DIR / 'raw'
RESULTS_DIR = DATA_DIR / 'results'
PREPROCESSED_DIR = RESULTS_DIR / 'preprocessed'
OCR_DIR = RESULTS_DIR / 'ocr'

# 标准字集文件
STANDARD_CHARS_JSON = DATA_DIR / 'standard_chars.json'
STANDARD_CHARS_TXT = DATA_DIR / 'standard_chars.txt'

# ============================================================================
# 预处理配置
# ============================================================================

# CLAHE (对比度自适应直方图均衡化)
PREPROCESS_CLAHE_CONFIG = {
    'enabled': True,
    'clip_limit': 2.0,     # CLAHE 剪切限制
    'tile_size': (8, 8),   # 网格大小
}

# 墨色保持与增强
PREPROCESS_INK_PRESERVE_CONFIG = {
    'enabled': True,
    'blackhat_kernel': 9,       # 黑帽形态学核尺寸
    'blackhat_strength': 0.6,   # 黑帽增强强度 [0.0-1.0]
    'unsharp_amount': 0.2,      # 反锐化掩膜强度 [0.0-1.0]
}

# 双边滤波去噪（可选）
PREPROCESS_DENOISE_CONFIG = {
    'enabled': False,
    'diameter': 9,         # 滤波直径
    'sigma_color': 75,     # 颜色空间标准差
    'sigma_space': 75,     # 坐标空间标准差
}

# 断笔修补（可选）
PREPROCESS_STROKE_HEAL_CONFIG = {
    'enabled': False,
    'kernel': 3,           # 闭运算核尺寸
    'iterations': 1,       # 迭代次数
    'directions': ['iso', 'h', 'v'],  # 方向：iso(各向同性), h(水平), v(垂直)
}

# ============================================================================
# OCR 配置
# ============================================================================

OCR_CONFIG = {
    'framework': 'livetext',           # 使用 macOS LiveText
    'recognition_level': 'accurate',   # 识别级别: 'fast' 或 'accurate'
    'language_preference': ['zh-Hans'], # 语言偏好：简体中文
}

# ============================================================================
# 配置校验
# ============================================================================

def validate_config():
    """校验配置参数的合理性"""
    errors = []

    # 校验 CLAHE 参数
    if PREPROCESS_CLAHE_CONFIG['clip_limit'] <= 0:
        errors.append("CLAHE clip_limit 必须大于 0")

    # 校验墨色增强参数
    if not (0.0 <= PREPROCESS_INK_PRESERVE_CONFIG['blackhat_strength'] <= 1.0):
        errors.append("blackhat_strength 必须在 [0.0, 1.0] 范围内")

    if not (0.0 <= PREPROCESS_INK_PRESERVE_CONFIG['unsharp_amount'] <= 1.0):
        errors.append("unsharp_amount 必须在 [0.0, 1.0] 范围内")

    # 校验 OCR 参数
    if OCR_CONFIG['recognition_level'] not in ['fast', 'accurate']:
        errors.append("recognition_level 必须是 'fast' 或 'accurate'")

    if errors:
        raise ValueError(f"配置校验失败:\n" + "\n".join(f"  - {e}" for e in errors))

    return True

# ============================================================================
# 配置摘要
# ============================================================================

def config_summary():
    """返回当前配置的摘要信息"""
    validate_config()

    return {
        'paths': {
            'project_root': str(PROJECT_ROOT),
            'raw_dir': str(RAW_DIR),
            'preprocessed_dir': str(PREPROCESSED_DIR),
            'ocr_dir': str(OCR_DIR),
        },
        'preprocess': {
            'clahe_enabled': PREPROCESS_CLAHE_CONFIG['enabled'],
            'ink_preserve_enabled': PREPROCESS_INK_PRESERVE_CONFIG['enabled'],
            'denoise_enabled': PREPROCESS_DENOISE_CONFIG['enabled'],
            'stroke_heal_enabled': PREPROCESS_STROKE_HEAL_CONFIG['enabled'],
        },
        'ocr': OCR_CONFIG,
    }

# 启动时校验配置
validate_config()
