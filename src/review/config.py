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
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 数据目录
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DIR = DATA_DIR / 'raw'
RESULTS_DIR = DATA_DIR / 'results'
PREPROCESSED_DIR = RESULTS_DIR / 'preprocessed'
OCR_DIR = RESULTS_DIR / 'ocr'

# 标准字集文件
STANDARD_CHARS_JSON = DATA_DIR / 'metadata' / 'standard_chars.json'
if not STANDARD_CHARS_JSON.exists():
    STANDARD_CHARS_JSON = PROJECT_ROOT / 'src' / 'standard_chars.json'
STANDARD_CHARS_TXT = DATA_DIR / 'metadata' / 'standard_chars.txt'

# PDF 转换输出目录（默认在 raw 目录下）
PDF_IMAGES_DIR = RESULTS_DIR / 'pdf_images'

# ============================================================================
# 预处理配置
# ============================================================================

# CLAHE (对比度自适应直方图均衡化)
PREPROCESS_CLAHE_CONFIG = {
    'enabled': True,
    'clip_limit': 2.0,     # CLAHE 剪切限制
    'tile_size': (8, 8),   # 网格大小
}

# 对比度和亮度调整
PREPROCESS_CONTRAST_CONFIG = {
    'alpha': 1.5,  # 对比度控制 (1.0-3.0)
    'beta': 10,    # 亮度控制 (0-100)
}

# 墨色保持/增强（关键！避免图像发灰）
PREPROCESS_INK_PRESERVE_CONFIG = {
    'enabled': True,            # 是否启用回墨增强
    'blackhat_kernel': 9,       # 黑帽核尺寸（必须为奇数）
    'blackhat_strength': 0.6,   # 黑帽增强强度 (0.0-1.0)
    'unsharp_amount': 0.2,      # 反锐化系数 (0.0-1.0)
}

# ============================================================================
# OCR 配置
# ============================================================================

OCR_CONFIG = {
    'framework': 'livetext',           # 使用 macOS LiveText
    'recognition_level': 'accurate',   # 识别级别: 'fast' 或 'accurate'
    'language_preference': ['zh-Hant'], # 语言偏好：繁体中文
}

# 预处理/OCR 册数覆盖配置（按书名）
# 例：'04_1131_资治通鉴' 从第 11 册开始取 10 册
VOLUME_OVERRIDES = {
    '04_1131_资治通鉴': {
        'start': 11,
        'count': 10,
    },
}

# ============================================================================
# PaddleOCR 自动筛选配置
# ============================================================================

PADDLE_CONFIG = {
    'url': 'http://172.16.1.154:8000',
    'timeout': 20,
    'topk': 15,
    'min_conf': 0.8,
    'batch_size': 16,
    'workers': 3,
    'require_match': True,
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

    # 校验对比度参数
    if PREPROCESS_CONTRAST_CONFIG['alpha'] <= 0:
        errors.append("对比度 alpha 必须大于 0")

    # 校验墨色保持参数
    if PREPROCESS_INK_PRESERVE_CONFIG['enabled']:
        if PREPROCESS_INK_PRESERVE_CONFIG['blackhat_kernel'] % 2 == 0:
            errors.append("黑帽核 blackhat_kernel 必须为奇数")
        if PREPROCESS_INK_PRESERVE_CONFIG['blackhat_kernel'] <= 0:
            errors.append("黑帽核 blackhat_kernel 必须大于 0")

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
            'clahe_clip_limit': PREPROCESS_CLAHE_CONFIG['clip_limit'],
            'contrast_alpha': PREPROCESS_CONTRAST_CONFIG['alpha'],
            'contrast_beta': PREPROCESS_CONTRAST_CONFIG['beta'],
            'ink_preserve_enabled': PREPROCESS_INK_PRESERVE_CONFIG['enabled'],
            'blackhat_kernel': PREPROCESS_INK_PRESERVE_CONFIG['blackhat_kernel'],
            'blackhat_strength': PREPROCESS_INK_PRESERVE_CONFIG['blackhat_strength'],
            'unsharp_amount': PREPROCESS_INK_PRESERVE_CONFIG['unsharp_amount'],
        },
        'ocr': OCR_CONFIG,
        'volume_overrides': VOLUME_OVERRIDES,
        'paddle': PADDLE_CONFIG,
    }

# 启动时校验配置
validate_config()
