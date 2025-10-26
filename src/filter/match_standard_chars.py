"""
标准字匹配模块 - 按书籍分组

从 OCR 结果中筛选出标准字集中的字符，按书籍组织
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from tqdm import tqdm

from src.config import PROJECT_ROOT


def load_standard_chars(json_path: str) -> Tuple[Set[str], Dict[str, str]]:
    """
    加载标准字集

    Args:
        json_path: standard_chars.json 路径

    Returns:
        (标准字集合, {字: 所属分类名称})
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    standard_chars = set()
    char_to_method = {}

    for method in data['methods']:
        method_name = method['name']
        for char in method['chars']:
            standard_chars.add(char)
            char_to_method[char] = method_name

    return standard_chars, char_to_method


def extract_book_name(ocr_file_path: str, ocr_dir: str) -> str:
    """
    从 OCR 文件路径提取书名

    Args:
        ocr_file_path: OCR 文件完整路径
        ocr_dir: OCR 根目录

    Returns:
        书名，如果是散乱文件则返回 None
    """
    rel_path = os.path.relpath(ocr_file_path, ocr_dir)
    # 路径格式: 书名/册号_pages/page_xxxx_ocr.json
    parts = rel_path.split(os.sep)

    # 过滤掉直接在 ocr_dir 下的散乱文件（如 page_0001_preprocessed_ocr.json）
    if len(parts) == 1:
        # 只有文件名，没有子目录，说明是散乱文件
        return None

    if len(parts) >= 2:
        return parts[0]

    return None


def extract_volume_number(volume_dir: str) -> int:
    """从目录名提取册号"""
    import re
    match = re.search(r'册(\d+)', volume_dir)
    if match:
        return int(match.group(1))
    return 0


def match_ocr_results_by_book(ocr_dir: str, standard_chars: Set[str], char_to_method: Dict[str, str]) -> Dict:
    """
    遍历 OCR 结果，按书籍分组筛选标准字

    Args:
        ocr_dir: OCR 结果目录
        standard_chars: 标准字集合
        char_to_method: 字符到分类的映射

    Returns:
        按书籍分组的匹配结果
    """
    # 递归查找所有 OCR JSON 文件
    ocr_files = []
    for root, dirs, files in os.walk(ocr_dir):
        for file in files:
            if file.endswith('_ocr.json'):
                ocr_files.append(os.path.join(root, file))

    print(f"\n=== 开始匹配标准字（按书籍分组） ===")
    print(f"OCR 结果目录: {ocr_dir}")
    print(f"找到 {len(ocr_files)} 个 OCR 结果文件")
    print(f"标准字数量: {len(standard_chars)}")
    print("-" * 50)

    # 按书籍组织结果
    books_data = {}
    chars_coverage = {}  # 统计每个字在多少本书中出现

    # 遍历所有 OCR 结果
    for ocr_file in tqdm(ocr_files, desc="匹配进度", unit="file"):
        try:
            with open(ocr_file, 'r', encoding='utf-8') as f:
                ocr_data = json.load(f)

            if not ocr_data.get('success'):
                continue

            # 提取书籍信息
            book_name = extract_book_name(ocr_file, ocr_dir)

            # 跳过散乱文件（book_name 为 None）
            if book_name is None:
                continue

            # 初始化书籍数据
            if book_name not in books_data:
                books_data[book_name] = {
                    'book_name': book_name,
                    'total_standard_chars': 0,
                    'total_instances': 0,
                    'chars': {}
                }

            # 提取文件路径信息
            rel_path = os.path.relpath(ocr_file, ocr_dir)
            path_parts = rel_path.split(os.sep)

            if len(path_parts) >= 3:
                volume_dir = path_parts[1]  # 例如 "册01_pages"
                volume_num = extract_volume_number(volume_dir)
                page_file = path_parts[2].replace('_preprocessed_ocr.json', '').replace('_ocr.json', '')
            else:
                volume_num = 0
                page_file = "unknown"

            # 获取源图像路径
            source_image = ocr_data.get('image_path', '')

            # 遍历识别出的字符
            for char_data in ocr_data.get('characters', []):
                char_text = char_data['text']

                # 检查是否是标准字
                if char_text in standard_chars:
                    # 构建匹配记录
                    match_record = {
                        'volume': volume_num,
                        'page': page_file,
                        'char_index': char_data['index'],
                        'source_image': source_image,
                        'bbox': char_data['bbox'],
                        'normalized_bbox': char_data['normalized_bbox'],
                        'confidence': char_data['confidence'],
                        'method': char_to_method.get(char_text, '未知分类'),
                        'ocr_file': os.path.relpath(ocr_file, PROJECT_ROOT),
                    }

                    # 添加到书籍的字符列表
                    if char_text not in books_data[book_name]['chars']:
                        books_data[book_name]['chars'][char_text] = []
                        books_data[book_name]['total_standard_chars'] += 1

                        # 更新覆盖率统计
                        if char_text not in chars_coverage:
                            chars_coverage[char_text] = set()
                        chars_coverage[char_text].add(book_name)

                    books_data[book_name]['chars'][char_text].append(match_record)
                    books_data[book_name]['total_instances'] += 1

        except Exception as e:
            tqdm.write(f"✗ 处理失败: {os.path.relpath(ocr_file, PROJECT_ROOT)} - {str(e)}")

    # 转换 chars_coverage 为数量
    chars_coverage_count = {char: len(books) for char, books in chars_coverage.items()}

    print("-" * 50)
    print(f"=== 匹配完成 ===")
    print(f"处理书籍: {len(books_data)} 本")
    print(f"匹配到标准字: {len(chars_coverage_count)}/{len(standard_chars)} 个")

    # 显示每本书的统计
    print(f"\n各书籍包含的标准字数量:")
    sorted_books = sorted(books_data.items(), key=lambda x: x[1]['total_standard_chars'], reverse=True)
    for book_name, book_data in sorted_books[:10]:
        print(f"  {book_name}: {book_data['total_standard_chars']} 个标准字, {book_data['total_instances']} 个实例")
    if len(sorted_books) > 10:
        print(f"  ... 还有 {len(sorted_books) - 10} 本书")

    # 显示覆盖率最高的字
    print(f"\n出现在最多书籍中的标准字:")
    sorted_chars = sorted(chars_coverage_count.items(), key=lambda x: x[1], reverse=True)
    for char, count in sorted_chars[:10]:
        print(f"  {char}: {count} 本书")

    return {
        'books': books_data,
        'summary': {
            'total_books': len(books_data),
            'total_standard_chars_found': len(chars_coverage_count),
            'chars_coverage': chars_coverage_count
        }
    }


def save_matched_chars(matched_data: Dict, output_path: str):
    """
    保存匹配结果到 JSON

    Args:
        matched_data: 匹配结果（按书籍分组）
        output_path: 输出文件路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(matched_data, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 匹配结果已保存至: {output_path}")
    print(f"  格式: 按书籍分组")
    print(f"  可使用以下命令裁切单本书:")
    print(f"  ./pipeline crop {output_path} --book <书名>")


def main(ocr_dir: str, standard_chars_json: str, output_path: str):
    """
    主函数

    Args:
        ocr_dir: OCR 结果目录
        standard_chars_json: 标准字 JSON 文件路径
        output_path: 输出文件路径
    """
    # 加载标准字
    standard_chars, char_to_method = load_standard_chars(standard_chars_json)

    # 匹配 OCR 结果（按书籍分组）
    matched_data = match_ocr_results_by_book(ocr_dir, standard_chars, char_to_method)

    # 保存结果
    save_matched_chars(matched_data, output_path)

    return matched_data
