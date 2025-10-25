"""
图像预处理模块

对古籍图片进行增强处理：
- CLAHE 对比度增强
- 对比度/亮度调整
- 墨色保持/增强（黑帽 + 反锐化）
"""
import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

from src.config import (
    PREPROCESS_CLAHE_CONFIG,
    PREPROCESS_CONTRAST_CONFIG,
    PREPROCESS_INK_PRESERVE_CONFIG,
    PREPROCESSED_DIR
)
from src.utils.path import ensure_dir
from src.utils.progress import ProgressTracker, get_default_progress_file, _get_relative_path
from src.utils.file_filter import find_images_recursive, filter_files_by_max_volumes


def preprocess_image(input_path: str, output_path: str = None,
                     alpha: float = None, beta: float = None,
                     verbose: bool = True) -> bool:
    """
    对输入的古籍图片进行预处理

    Args:
        input_path: 输入图片路径
        output_path: 输出路径（可选，默认保存到 PREPROCESSED_DIR）
        alpha: 对比度控制 (1.0-3.0)，None 则使用配置文件默认值
        beta: 亮度控制 (0-100)，None 则使用配置文件默认值
        verbose: 是否输出详细信息

    Returns:
        是否处理成功
    """
    # 1. 读取图片
    img = cv2.imread(input_path)
    if img is None:
        if verbose:
            print(f"错误：无法读取图片 {input_path}")
        return False

    # 2. 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. 应用 CLAHE 增强局部对比度
    if PREPROCESS_CLAHE_CONFIG['enabled']:
        cfg = PREPROCESS_CLAHE_CONFIG
        clahe = cv2.createCLAHE(
            clipLimit=cfg['clip_limit'],
            tileGridSize=cfg['tile_size']
        )
        gray = clahe.apply(gray)

    # 4. 对比度和亮度调整
    if alpha is None:
        alpha = PREPROCESS_CONTRAST_CONFIG['alpha']
    if beta is None:
        beta = PREPROCESS_CONTRAST_CONFIG['beta']

    adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    # 5. 墨色保持/增强：黑帽回墨 + 可选反锐化，避免整体发灰
    if PREPROCESS_INK_PRESERVE_CONFIG['enabled']:
        ink_cfg = PREPROCESS_INK_PRESERVE_CONFIG

        # 黑帽增强
        ksize = int(ink_cfg['blackhat_kernel'])
        if ksize % 2 == 0:
            ksize += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))

        # 黑帽提取"暗笔画相对亮背景"的分量
        blackhat = cv2.morphologyEx(adjusted, cv2.MORPH_BLACKHAT, kernel)
        strength = float(ink_cfg['blackhat_strength'])

        # 回墨：把黑帽分量按比例减回去，使笔画更黑
        adjusted = cv2.subtract(adjusted, cv2.convertScaleAbs(blackhat, alpha=strength, beta=0))

        # 可选反锐化（unsharp masking）
        amount = float(ink_cfg['unsharp_amount'])
        if amount > 1e-6:
            blur = cv2.GaussianBlur(adjusted, (0, 0), sigmaX=1.0)
            adjusted = cv2.addWeighted(adjusted, 1 + amount, blur, -amount, 0)

    # 6. 确定输出路径
    if output_path is None:
        # 默认保存到 PREPROCESSED_DIR
        input_path_obj = Path(input_path)
        output_filename = f"{input_path_obj.stem}_preprocessed{input_path_obj.suffix}"
        output_path = os.path.join(PREPROCESSED_DIR, output_filename)

    # 确保输出目录存在
    ensure_dir(os.path.dirname(output_path))

    # 7. 保存处理后的图片
    cv2.imwrite(output_path, adjusted)
    if verbose:
        print(f"✓ 图片已处理并保存至 {output_path}")

    return True


def process_directory(input_dir: str, output_dir: str = None,
                      alpha: float = None, beta: float = None,
                      force: bool = False, max_volumes: int = None) -> tuple[int, int]:
    """
    批量处理目录下的所有图片

    Args:
        input_dir: 输入目录
        output_dir: 输出目录（默认为 PREPROCESSED_DIR）
        alpha: 对比度控制 (1.0-3.0)，None 则使用配置文件默认值
        beta: 亮度控制 (0-100)，None 则使用配置文件默认值
        force: 是否强制重新处理（忽略进度记录）
        max_volumes: 最大册数限制（None 表示不限制）

    Returns:
        (成功数量, 总数量)
    """
    # 确保路径是字符串
    input_dir = str(input_dir)

    if output_dir is None:
        output_dir = str(PREPROCESSED_DIR)
    else:
        output_dir = str(output_dir)

    ensure_dir(output_dir)

    # 使用配置文件默认值
    if alpha is None:
        alpha = PREPROCESS_CONTRAST_CONFIG['alpha']
    if beta is None:
        beta = PREPROCESS_CONTRAST_CONFIG['beta']

    # 递归查找所有图片文件（支持子目录）
    if not os.path.exists(input_dir):
        print(f"错误：输入目录不存在 {input_dir}")
        return 0, 0

    image_files = find_images_recursive(input_dir)

    if not image_files:
        print(f"警告：在 {input_dir} 中未找到任何图片文件")
        return 0, 0

    # 根据最大册数过滤文件
    filtered_files, volume_stats = filter_files_by_max_volumes(
        image_files, max_volumes, base_dir=input_dir
    )

    # 初始化进度跟踪器
    progress_file = get_default_progress_file(output_dir)
    tracker = ProgressTracker(progress_file, 'preprocess')
    tracker.init_session(input_dir, output_dir, force=force)

    # 获取待处理的文件列表
    pending_files = tracker.get_pending_files(filtered_files)
    # 按文件名排序，确保处理顺序一致
    pending_files = sorted(pending_files)
    stats = tracker.get_stats()

    print(f"\n=== 开始批量预处理 ===")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")

    # 显示书籍和册数统计
    if volume_stats.get('is_multi_book'):
        # 多本书模式
        print(f"发现书籍: {volume_stats['total_books']} 本")
        if max_volumes:
            print(f"册数限制: 每本书最多 {max_volumes} 册")
        print(f"总文件数: {volume_stats['total_files']} -> {volume_stats['selected_files']} (过滤后)")
        print("\n各书籍统计:")
        for book_name, book_stat in volume_stats['books'].items():
            if book_stat['total_volumes'] > 0:  # 只显示有册号的书籍
                vol_info = f"{book_stat['selected_volumes']}/{book_stat['total_volumes']} 册"
                file_info = f"{book_stat['selected_files']}/{book_stat['total_files']} 文件"
                vol_list = book_stat['selected_volume_numbers'][:5]  # 只显示前5册
                if len(book_stat['selected_volume_numbers']) > 5:
                    vol_list_str = f"{vol_list}..."
                else:
                    vol_list_str = str(vol_list)
                print(f"  {book_name}: {vol_info} | {file_info} | 册 {vol_list_str}")
    else:
        # 单本书模式
        if volume_stats['books']:
            book_stat = list(volume_stats['books'].values())[0]
            if max_volumes:
                print(f"册数限制: 最多 {max_volumes} 册")
            print(f"发现册数: {book_stat['total_volumes']} 册")
            if book_stat['total_volumes'] > 0:
                print(f"选择册数: {book_stat['selected_volumes']} 册 {book_stat['selected_volume_numbers']}")
                print(f"总文件数: {book_stat['total_files']} -> {book_stat['selected_files']} (过滤后)")
        else:
            print(f"总文件数: {len(filtered_files)}")

    print(f"已完成: {stats['completed']} | 待处理: {len(pending_files)}")
    print(f"参数: alpha={alpha}, beta={beta}")
    if stats['completed'] > 0 and not force:
        print(f"💡 继续上次进度 (最后更新: {stats['last_update']})")
        print(f"💡 使用 --force 强制重新处理所有文件")
    print("-" * 50)

    if not pending_files:
        print("所有文件已处理完成！")
        return stats['completed'], len(filtered_files)

    success_count = stats['completed']  # 从已完成数量开始
    # 使用 tqdm 显示进度条
    for rel_filepath in tqdm(pending_files, desc="预处理进度", unit="file"):
        input_path = os.path.join(input_dir, rel_filepath)

        # 生成输出路径（保持目录结构）
        output_path = os.path.join(output_dir, rel_filepath)
        # 在文件名后添加 _preprocessed 后缀
        output_dir_part, output_filename = os.path.split(output_path)
        name, ext = os.path.splitext(output_filename)
        output_filename = f"{name}_preprocessed{ext}"
        output_path = os.path.join(output_dir_part, output_filename)

        # 获取项目相对路径用于输出
        project_rel_path = _get_relative_path(input_path)

        try:
            if preprocess_image(input_path, output_path, alpha=alpha, beta=beta, verbose=False):
                success_count += 1
                tracker.mark_completed(rel_filepath)
            else:
                tracker.mark_failed(rel_filepath)
                tqdm.write(f"✗ 失败: {project_rel_path}")
        except Exception as e:
            tracker.mark_failed(rel_filepath)
            tqdm.write(f"✗ 失败: {project_rel_path} - {str(e)}")

    print("-" * 50)
    print(f"=== 批量预处理完成 ===")
    print(f"成功处理: {success_count}/{len(filtered_files)} 个文件")

    failed_files = tracker.get_failed_files()
    if failed_files:
        print(f"失败文件: {len(failed_files)} 个")
        for f in failed_files[:5]:  # 只显示前5个
            print(f"  - {f}")
        if len(failed_files) > 5:
            print(f"  ... 还有 {len(failed_files) - 5} 个")

    return success_count, len(filtered_files)
