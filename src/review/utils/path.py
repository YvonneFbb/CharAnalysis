"""
路径工具函数
"""
import os
from pathlib import Path


def ensure_dir(path):
    """确保目录存在，不存在则创建"""
    os.makedirs(path, exist_ok=True)


def get_output_path(input_path, output_dir, suffix='', ext=None):
    """
    根据输入路径生成输出路径

    Args:
        input_path: 输入文件路径
        output_dir: 输出目录
        suffix: 文件名后缀
        ext: 新扩展名（如果需要改变），默认保持原扩展名

    Returns:
        输出文件路径
    """
    input_path = Path(input_path)
    base_name = input_path.stem
    file_ext = ext if ext else input_path.suffix

    output_filename = f"{base_name}{suffix}{file_ext}"
    return os.path.join(output_dir, output_filename)
