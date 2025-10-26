"""
字符裁切模块 - 支持按书籍裁切

根据匹配结果裁切字符图像
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Optional
import cv2
from tqdm import tqdm

from src.config import PROJECT_ROOT
from src.utils.path import ensure_dir


def crop_character(source_image_path: str, bbox: Dict[str, int], output_path: str, padding: int = 5) -> bool:
    """
    从源图像裁切字符

    Args:
        source_image_path: 源图像路径
        bbox: 边界框 {x, y, width, height}
        output_path: 输出路径
        padding: 边界填充像素数

    Returns:
        是否成功
    """
    try:
        # 读取图像
        img = cv2.imread(source_image_path)
        if img is None:
            return False

        h, w = img.shape[:2]

        # 提取边界框信息并添加 padding
        x = max(0, bbox['x'] - padding)
        y = max(0, bbox['y'] - padding)
        x2 = min(w, bbox['x'] + bbox['width'] + padding)
        y2 = min(h, bbox['y'] + bbox['height'] + padding)

        # 裁切
        cropped = img[y:y2, x:x2]

        # 保存
        ensure_dir(os.path.dirname(output_path))
        cv2.imwrite(output_path, cropped)

        return True

    except Exception as e:
        print(f"裁切失败: {output_path} - {str(e)}")
        return False


def crop_single_book(matched_data: Dict, book_name: str, output_dir: str, padding: int = 5) -> Dict[str, int]:
    """
    裁切单本书的匹配字符

    Args:
        matched_data: 匹配结果数据（按书籍分组）
        book_name: 书名
        output_dir: 输出目录
        padding: 边界填充像素数

    Returns:
        裁切统计 {字符: 成功数量}
    """
    if book_name not in matched_data['books']:
        print(f"错误：书籍 '{book_name}' 不存在于匹配结果中")
        available_books = list(matched_data['books'].keys())
        print(f"可用书籍列表 ({len(available_books)} 本):")
        for bk in available_books[:10]:
            print(f"  - {bk}")
        if len(available_books) > 10:
            print(f"  ... 还有 {len(available_books) - 10} 本")
        return {}

    book_data = matched_data['books'][book_name]

    print(f"\n=== 开始裁切字符 (书籍: {book_name}) ===")
    print(f"输出目录: {output_dir}")
    print(f"边界填充: {padding} 像素")
    print(f"标准字数: {book_data['total_standard_chars']} 个")
    print(f"总实例数: {book_data['total_instances']}")
    print("-" * 50)

    crop_stats = {}
    total_success = 0
    total_fail = 0

    # 为该书创建目录
    book_output_dir = os.path.join(output_dir, book_name)
    ensure_dir(book_output_dir)

    # 遍历每个标准字
    for char, instances in tqdm(book_data['chars'].items(), desc="裁切进度", unit="char"):
        success_count = 0

        # 为每个字符创建目录
        char_dir = os.path.join(book_output_dir, char)
        ensure_dir(char_dir)

        # 裁切所有实例
        for instance in instances:
            # 生成输出文件名
            # 格式: 册{册号}_{页号}_{索引}.png
            output_filename = f"册{instance['volume']:02d}_{instance['page']}_{instance['char_index']}.png"
            output_path = os.path.join(char_dir, output_filename)

            # 解析源图像路径（转换为绝对路径）
            source_image = instance['source_image']
            if not os.path.isabs(source_image):
                source_image = os.path.join(PROJECT_ROOT, source_image)

            # 裁切字符
            if crop_character(source_image, instance['bbox'], output_path, padding):
                success_count += 1
                total_success += 1

                # 更新实例记录（添加裁切图像路径）
                instance['cropped_image'] = os.path.relpath(output_path, PROJECT_ROOT)
            else:
                total_fail += 1
                tqdm.write(f"✗ 裁切失败: {char} - {output_filename}")

        crop_stats[char] = success_count

    print("-" * 50)
    print(f"=== 裁切完成 ===")
    print(f"成功: {total_success} 个")
    print(f"失败: {total_fail} 个")

    return crop_stats


def crop_all_books(matched_data: Dict, output_dir: str, padding: int = 5):
    """
    裁切所有书籍的匹配字符

    Args:
        matched_data: 匹配结果数据（按书籍分组）
        output_dir: 输出目录
        padding: 边界填充像素数
    """
    books = matched_data['books']
    print(f"\n=== 开始裁切所有书籍 ===")
    print(f"总书籍数: {len(books)}")
    print(f"输出目录: {output_dir}")
    print("-" * 50)

    for book_name in books.keys():
        print(f"\n>>> 处理书籍: {book_name}")
        crop_single_book(matched_data, book_name, output_dir, padding)


def main(matched_chars_json: str, output_dir: str, padding: int = 5, book_name: Optional[str] = None):
    """
    主函数

    Args:
        matched_chars_json: 匹配结果 JSON 文件路径
        output_dir: 输出目录
        padding: 边界填充像素数
        book_name: 指定书名（None 表示处理所有书籍）
    """
    # 加载匹配结果
    with open(matched_chars_json, 'r', encoding='utf-8') as f:
        matched_data = json.load(f)

    if book_name:
        # 裁切单本书
        crop_stats = crop_single_book(matched_data, book_name, output_dir, padding)

        # 显示统计
        if crop_stats:
            print(f"\n裁切最多的前 10 个字:")
            sorted_stats = sorted(crop_stats.items(), key=lambda x: x[1], reverse=True)
            for char, count in sorted_stats[:10]:
                print(f"  {char}: {count} 个实例")
    else:
        # 裁切所有书籍
        crop_all_books(matched_data, output_dir, padding)

    return True
