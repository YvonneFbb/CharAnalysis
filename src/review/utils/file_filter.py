"""
文件过滤和分组工具

用于处理古籍图片时的文件过滤、册数限制等功能
"""
import os
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path


def extract_volume_number(filepath: str) -> Optional[int]:
    """
    从文件路径中提取册号

    支持的格式：
    - 册01_pages/page_001.png -> 1
    - 册12/image.jpg -> 12
    - book_vol03.png -> 3
    - 第5册/... -> 5

    Args:
        filepath: 文件路径

    Returns:
        册号（整数），如果无法提取则返回 None
    """
    # 尝试匹配各种册号格式
    patterns = [
        r'册(\d+)',           # 册01, 册02
        r'vol(?:ume)?[-_]?(\d+)',  # vol01, volume_02
        r'第(\d+)册',         # 第1册
        r'v(\d+)',            # v1, v2
    ]

    for pattern in patterns:
        match = re.search(pattern, filepath, re.IGNORECASE)
        if match:
            return int(match.group(1))

    return None


def extract_book_name(filepath: str, base_dir: str) -> Optional[str]:
    """
    从文件路径中提取书名（第一级子目录名）

    Args:
        filepath: 文件相对路径（相对于 base_dir）
        base_dir: 基础目录

    Returns:
        书名（目录名），如果文件在根目录或第一级目录是册号目录则返回 None
    """
    # 将路径标准化
    parts = Path(filepath).parts
    if len(parts) > 1:
        first_dir = parts[0]
        # 如果第一级目录看起来像册号目录（包含 册/vol/volume 等），则认为是单本书模式
        if extract_volume_number(first_dir) is not None:
            return None
        # 否则，第一级子目录就是书名
        return first_dir
    return None


def group_files_by_book_and_volume(files: List[str], base_dir: str) -> Dict[str, Dict[int, List[str]]]:
    """
    将文件按书籍和册号分组

    Args:
        files: 文件相对路径列表
        base_dir: 基础目录

    Returns:
        {书名: {册号: [文件列表]}} 的嵌套字典
        书名为 None 的表示根目录的文件（单本书模式）
    """
    book_volume_groups = {}

    for filepath in files:
        # 提取书名
        book_name = extract_book_name(filepath, base_dir)

        # 提取册号
        full_path = os.path.join(base_dir, filepath) if base_dir else filepath
        volume_num = extract_volume_number(full_path)
        if volume_num is None:
            volume_num = -1  # 未识别的文件

        # 初始化嵌套字典
        if book_name not in book_volume_groups:
            book_volume_groups[book_name] = {}

        if volume_num not in book_volume_groups[book_name]:
            book_volume_groups[book_name][volume_num] = []

        book_volume_groups[book_name][volume_num].append(filepath)

    return book_volume_groups


def filter_files_by_max_volumes(files: List[str], max_volumes: Optional[int],
                                 base_dir: str = None,
                                 volume_overrides: Optional[Dict[str, Dict]] = None) -> Tuple[List[str], Dict]:
    """
    根据最大册数过滤文件列表（支持多本书自动识别）

    Args:
        files: 文件名列表
        max_volumes: 每本书的最大册数（None 表示不限制）
        base_dir: 基础目录

    Returns:
        (过滤后的文件列表, 统计信息)
    """
    if not files:
        return [], {'max_volumes': max_volumes, 'books': {}}

    # 按书籍和册号分组
    book_volume_groups = group_files_by_book_and_volume(files, base_dir)

    # 判断是单本书还是多本书
    is_multi_book = len(book_volume_groups) > 1 or (len(book_volume_groups) == 1 and None not in book_volume_groups)

    selected_files = []
    book_stats = {}

    for book_name, volume_groups in book_volume_groups.items():
        # 获取所有册号（排除未识别的）
        volume_numbers = sorted([v for v in volume_groups.keys() if v > 0])

        # 选择前 N 册（支持 per-book 起始册数）
        overrides = volume_overrides or {}
        override_key = book_name if book_name else "(根目录)"
        if override_key not in overrides and base_dir:
            base_name = Path(base_dir).name
            if base_name in overrides:
                override_key = base_name
        override = overrides.get(override_key)
        start_volume = None
        override_count = None
        if isinstance(override, dict):
            start_volume = override.get('start')
            if start_volume is None:
                start_volume = override.get('start_volume')
            override_count = override.get('count')
            if override_count is None:
                override_count = override.get('max_volumes')

        selected_source = volume_numbers
        if start_volume and start_volume > 0:
            selected_source = [v for v in selected_source if v >= start_volume]

        selected_limit = override_count if override_count and override_count > 0 else max_volumes
        if selected_limit and selected_limit > 0:
            selected_volumes = selected_source[:selected_limit]
        else:
            selected_volumes = selected_source

        # 收集选中的文件
        book_files = []
        for vol_num in selected_volumes:
            book_files.extend(volume_groups[vol_num])

        # 添加未识别册号的文件
        if -1 in volume_groups:
            book_files.extend(volume_groups[-1])

        selected_files.extend(book_files)

        # 统计信息
        book_display_name = book_name if book_name else "(根目录)"
        book_stats[book_display_name] = {
            'total_volumes': len(volume_numbers),
            'selected_volumes': len(selected_volumes),
            'volume_numbers': volume_numbers,
            'selected_volume_numbers': selected_volumes,
            'start_volume': start_volume,
            'override_count': override_count,
            'total_files': sum(len(volume_groups[v]) for v in volume_groups),
            'selected_files': len(book_files),
        }

    # 总体统计
    stats = {
        'max_volumes': max_volumes,
        'is_multi_book': is_multi_book,
        'total_books': len(book_volume_groups),
        'total_files': len(files),
        'selected_files': len(selected_files),
        'books': book_stats,
    }

    return selected_files, stats


def group_files_by_volume(files: List[str], base_dir: str = None) -> Dict[int, List[str]]:
    """
    将文件按册号分组（向后兼容，单本书模式）

    Args:
        files: 文件名列表
        base_dir: 基础目录（用于构建完整路径）

    Returns:
        {册号: [文件列表]} 的字典，未识别出册号的文件放在键 -1 下
    """
    volume_groups = {}

    for filename in files:
        # 构建完整路径用于提取册号
        if base_dir:
            full_path = os.path.join(base_dir, filename)
        else:
            full_path = filename

        volume_num = extract_volume_number(full_path)

        if volume_num is None:
            volume_num = -1  # 未识别的文件

        if volume_num not in volume_groups:
            volume_groups[volume_num] = []

        volume_groups[volume_num].append(filename)

    return volume_groups


def find_images_recursive(directory: str, extensions: set = None) -> List[str]:
    """
    递归查找目录下的所有图片文件（相对路径）

    Args:
        directory: 搜索目录
        extensions: 文件扩展名集合（如 {'.jpg', '.png'}）

    Returns:
        相对于 directory 的文件路径列表
    """
    if extensions is None:
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    image_files = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in extensions:
                # 计算相对路径
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, directory)
                image_files.append(rel_path)

    return image_files
