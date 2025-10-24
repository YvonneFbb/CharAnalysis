"""
图像预处理模块

对古籍图片进行增强处理：
- CLAHE 对比度增强
- 黑帽墨色保持
- 可选的双边滤波去噪
- 可选的断笔修补
"""
import cv2
import numpy as np
import os
from pathlib import Path

from src.config import (
    PREPROCESS_CLAHE_CONFIG,
    PREPROCESS_INK_PRESERVE_CONFIG,
    PREPROCESS_DENOISE_CONFIG,
    PREPROCESS_STROKE_HEAL_CONFIG,
    PREPROCESSED_DIR
)
from src.utils.path import ensure_dir


def _build_kernel(size: int, mode: str = 'iso') -> np.ndarray:
    """
    构建形态学核

    Args:
        size: 核尺寸
        mode: 核类型 ('iso': 椭圆, 'h': 水平, 'v': 垂直)

    Returns:
        形态学核
    """
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1

    if mode == 'iso':
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    elif mode == 'h':
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, 1))
    elif mode == 'v':
        return cv2.getStructuringElement(cv2.MORPH_RECT, (1, size))
    else:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def preprocess_image(input_path: str, output_path: str = None) -> bool:
    """
    对输入的古籍图片进行预处理

    Args:
        input_path: 输入图片路径
        output_path: 输出路径（可选，默认保存到 PREPROCESSED_DIR）

    Returns:
        是否处理成功
    """
    # 1. 读取图片
    img = cv2.imread(input_path)
    if img is None:
        print(f"错误：无法读取图片 {input_path}")
        return False

    # 2. 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. 可选：双边滤波去纸纹
    if PREPROCESS_DENOISE_CONFIG['enabled']:
        cfg = PREPROCESS_DENOISE_CONFIG
        gray = cv2.bilateralFilter(
            gray,
            d=cfg['diameter'],
            sigmaColor=cfg['sigma_color'],
            sigmaSpace=cfg['sigma_space']
        )

    # 4. 可选：断笔修补（闭运算填补断笔缝隙）
    if PREPROCESS_STROKE_HEAL_CONFIG['enabled']:
        cfg = PREPROCESS_STROKE_HEAL_CONFIG
        # 取反使墨迹为白，再用闭运算填缝
        inv = 255 - gray
        healed = inv

        for direction in cfg['directions']:
            kernel = _build_kernel(cfg['kernel'], direction)
            healed = cv2.morphologyEx(
                healed,
                cv2.MORPH_CLOSE,
                kernel,
                iterations=cfg['iterations']
            )

        gray = 255 - healed

    # 5. CLAHE 对比度增强
    if PREPROCESS_CLAHE_CONFIG['enabled']:
        cfg = PREPROCESS_CLAHE_CONFIG
        clahe = cv2.createCLAHE(
            clipLimit=cfg['clip_limit'],
            tileGridSize=cfg['tile_size']
        )
        gray = clahe.apply(gray)

    # 6. 墨色保持与增强
    if PREPROCESS_INK_PRESERVE_CONFIG['enabled']:
        cfg = PREPROCESS_INK_PRESERVE_CONFIG

        # 黑帽提取暗笔画分量
        ksize = cfg['blackhat_kernel']
        if ksize % 2 == 0:
            ksize += 1

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

        # 回墨：减去黑帽分量使笔画更黑
        strength = cfg['blackhat_strength']
        gray = cv2.subtract(gray, cv2.convertScaleAbs(blackhat, alpha=strength, beta=0))

        # 可选反锐化掩膜
        amount = cfg['unsharp_amount']
        if amount > 1e-6:
            blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
            gray = cv2.addWeighted(gray, 1 + amount, blur, -amount, 0)

    # 7. 确定输出路径
    if output_path is None:
        # 默认保存到 PREPROCESSED_DIR
        input_path_obj = Path(input_path)
        output_filename = f"{input_path_obj.stem}_preprocessed{input_path_obj.suffix}"
        output_path = os.path.join(PREPROCESSED_DIR, output_filename)

    # 确保输出目录存在
    ensure_dir(os.path.dirname(output_path))

    # 8. 保存处理后的图片
    cv2.imwrite(output_path, gray)
    print(f"✓ 图片已处理并保存至 {output_path}")

    return True


def process_directory(input_dir: str, output_dir: str = None) -> tuple[int, int]:
    """
    批量处理目录下的所有图片

    Args:
        input_dir: 输入目录
        output_dir: 输出目录（默认为 PREPROCESSED_DIR）

    Returns:
        (成功数量, 总数量)
    """
    if output_dir is None:
        output_dir = PREPROCESSED_DIR

    ensure_dir(output_dir)

    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    # 获取所有图片文件
    image_files = []
    if not os.path.exists(input_dir):
        print(f"错误：输入目录不存在 {input_dir}")
        return 0, 0

    for filename in os.listdir(input_dir):
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext in image_extensions:
            image_files.append(filename)

    if not image_files:
        print(f"警告：在 {input_dir} 中未找到任何图片文件")
        return 0, 0

    print(f"\n=== 开始批量预处理 ===")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"找到 {len(image_files)} 个图片文件")
    print("-" * 50)

    success_count = 0
    for i, filename in enumerate(image_files, 1):
        input_path = os.path.join(input_dir, filename)

        # 生成输出路径
        file_name, file_ext = os.path.splitext(filename)
        output_filename = f"{file_name}_preprocessed{file_ext}"
        output_path = os.path.join(output_dir, output_filename)

        print(f"[{i}/{len(image_files)}] 处理: {filename}")

        if preprocess_image(input_path, output_path):
            success_count += 1
        else:
            print(f"失败: {filename}")

    print("-" * 50)
    print(f"=== 批量预处理完成 ===")
    print(f"成功处理: {success_count}/{len(image_files)} 个文件")

    return success_count, len(image_files)
