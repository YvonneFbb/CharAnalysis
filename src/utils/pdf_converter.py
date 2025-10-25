"""
PDF 转换模块

将 PDF 文件转换为图片，便于后续的 OCR 处理
"""
import os
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm

from src.utils.path import ensure_dir
from src.utils.progress import ProgressTracker, get_default_progress_file, _get_relative_path
from src.utils.file_filter import extract_volume_number, extract_book_name


def pdf_to_images(
    pdf_path: str,
    output_dir: str = None,
    dpi: int = 300,
    image_format: str = 'png',
    verbose: bool = True
) -> List[str]:
    """
    将 PDF 文件的每一页转换为图片

    Args:
        pdf_path: PDF 文件路径
        output_dir: 输出目录（默认为 PDF 同目录下的 {pdf_name}_pages/）
        dpi: 分辨率（默认 300，适合 OCR）
        image_format: 图片格式（默认 png）
        verbose: 是否输出详细信息

    Returns:
        生成的图片路径列表
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

    # 打开 PDF
    doc = fitz.open(pdf_path)
    page_count = len(doc)

    # 确定输出目录
    if output_dir is None:
        pdf_path_obj = Path(pdf_path)
        output_dir = pdf_path_obj.parent / f"{pdf_path_obj.stem}_pages"

    ensure_dir(output_dir)

    # 转换参数
    zoom = dpi / 72  # PDF 默认 72 DPI
    mat = fitz.Matrix(zoom, zoom)

    image_paths = []

    if verbose:
        print(f"\n正在转换 PDF: {pdf_path}")
        print(f"总页数: {page_count}")
        print(f"输出目录: {output_dir}")
        print(f"分辨率: {dpi} DPI")
        print("-" * 50)

    for page_num in range(page_count):
        # 获取页面
        page = doc[page_num]

        # 渲染为图片
        pix = page.get_pixmap(matrix=mat)

        # 生成输出文件名
        output_filename = f"page_{page_num + 1:04d}.{image_format}"
        output_path = os.path.join(output_dir, output_filename)

        # 保存图片
        pix.save(output_path)
        image_paths.append(output_path)

        if verbose:
            print(f"[{page_num + 1}/{page_count}] 已转换: {output_filename}")

    doc.close()

    if verbose:
        print("-" * 50)
        print(f"✓ PDF 转换完成，共生成 {len(image_paths)} 张图片")

    return image_paths


def convert_directory(
    input_dir: str,
    output_parent_dir: str = None,
    dpi: int = 300,
    image_format: str = 'png'
) -> dict:
    """
    批量转换目录下的所有 PDF 文件

    Args:
        input_dir: 输入目录
        output_parent_dir: 输出父目录（默认为输入目录）
        dpi: 分辨率
        image_format: 图片格式

    Returns:
        转换结果字典 {pdf_path: [image_paths]}
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    if output_parent_dir is None:
        output_parent_dir = input_dir

    # 查找所有 PDF 文件
    pdf_files = []
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.pdf'):
            pdf_files.append(os.path.join(input_dir, filename))

    if not pdf_files:
        print(f"警告：在 {input_dir} 中未找到任何 PDF 文件")
        return {}

    print(f"\n=== 开始批量转换 PDF ===")
    print(f"输入目录: {input_dir}")
    print(f"找到 {len(pdf_files)} 个 PDF 文件")
    print("=" * 60)

    results = {}

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] 处理: {os.path.basename(pdf_path)}")

        # 生成输出目录
        pdf_name = Path(pdf_path).stem
        output_dir = os.path.join(output_parent_dir, f"{pdf_name}_pages")

        try:
            image_paths = pdf_to_images(pdf_path, output_dir, dpi, image_format)
            results[pdf_path] = image_paths
        except Exception as e:
            print(f"✗ 转换失败: {e}")
            results[pdf_path] = []

    print("\n" + "=" * 60)
    print("=== 批量转换完成 ===")
    success_count = sum(1 for paths in results.values() if paths)
    print(f"成功转换: {success_count}/{len(pdf_files)} 个 PDF 文件")

    return results


def get_pdf_info(pdf_path: str) -> dict:
    """
    获取 PDF 文件的基本信息

    Args:
        pdf_path: PDF 文件路径

    Returns:
        PDF 信息字典
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

    doc = fitz.open(pdf_path)

    info = {
        'path': pdf_path,
        'filename': os.path.basename(pdf_path),
        'page_count': len(doc),
        'metadata': doc.metadata,
    }

    # 获取第一页尺寸
    if len(doc) > 0:
        first_page = doc[0]
        rect = first_page.rect
        info['first_page_size'] = {
            'width': rect.width,
            'height': rect.height,
        }

    doc.close()

    return info


def convert_books_directory(
    input_dir: str,
    dpi: int = 300,
    image_format: str = 'png',
    max_volumes: Optional[int] = None,
    force: bool = False
) -> Tuple[int, int]:
    """
    批量转换多本书的 PDF（支持 max-volumes 限制和进度跟踪）

    Args:
        input_dir: 输入目录（包含多本书的子目录）
        dpi: 分辨率
        image_format: 图片格式
        max_volumes: 每本书最多转换的册数（None 表示不限制）
        force: 是否强制重新转换（忽略进度记录）

    Returns:
        (成功转换的 PDF 数量, 总 PDF 数量)
    """
    input_dir = str(input_dir)

    if not os.path.exists(input_dir):
        print(f"错误：输入目录不存在 {input_dir}")
        return 0, 0

    # 查找所有书籍目录
    book_dirs = []
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            book_dirs.append(item_path)

    if not book_dirs:
        print(f"警告：在 {input_dir} 中未找到任何书籍目录")
        return 0, 0

    # 收集每本书的 PDF 文件并按册号分组
    book_pdf_info = {}  # {book_name: {volume_num: pdf_path}}
    total_books = 0
    total_pdfs_before_filter = 0
    total_pdfs_after_filter = 0

    for book_dir in sorted(book_dirs):
        book_name = os.path.basename(book_dir)
        pdf_files = []

        for filename in os.listdir(book_dir):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(book_dir, filename)
                volume_num = extract_volume_number(pdf_path)
                pdf_files.append((volume_num, pdf_path))
                total_pdfs_before_filter += 1

        if pdf_files:
            # 按册号排序
            pdf_files.sort(key=lambda x: x[0])

            # 应用 max_volumes 限制
            if max_volumes and max_volumes > 0:
                pdf_files = pdf_files[:max_volumes]

            # 构建字典
            volume_dict = {}
            for vol_num, pdf_path in pdf_files:
                volume_dict[vol_num] = pdf_path
                total_pdfs_after_filter += 1

            book_pdf_info[book_name] = volume_dict
            total_books += 1

    # 初始化进度跟踪器
    progress_file = get_default_progress_file(input_dir)
    tracker = ProgressTracker(progress_file, 'pdf_convert')
    tracker.init_session(input_dir, input_dir, force=force)  # 输入输出都是同一目录

    # 获取所有待转换的 PDF（使用相对路径）
    all_pdfs = []
    for book_name, volume_dict in book_pdf_info.items():
        for vol_num, pdf_path in volume_dict.items():
            rel_path = os.path.relpath(pdf_path, input_dir)
            all_pdfs.append(rel_path)

    pending_pdfs = tracker.get_pending_files(all_pdfs)
    stats = tracker.get_stats()

    print(f"\n=== 开始批量转换 PDF ===")
    print(f"输入目录: {input_dir}")
    print(f"发现书籍: {total_books} 本")
    if max_volumes:
        print(f"册数限制: 每本书最多 {max_volumes} 册")
    print(f"总 PDF 数: {total_pdfs_before_filter} -> {total_pdfs_after_filter} (过滤后)")
    print(f"分辨率: {dpi} DPI")

    print(f"\n各书籍统计:")
    for book_name, volume_dict in book_pdf_info.items():
        vol_nums = sorted(volume_dict.keys())
        vol_list_str = str(vol_nums[:5])
        if len(vol_nums) > 5:
            vol_list_str = vol_list_str[:-1] + "...]"
        print(f"  {book_name}: {len(vol_nums)} 册 | 册 {vol_list_str}")

    print(f"\n已完成: {stats['completed']} | 待处理: {len(pending_pdfs)}")
    if stats['completed'] > 0 and not force:
        print(f"💡 继续上次进度 (最后更新: {stats['last_update']})")
        print(f"💡 使用 --force 强制重新处理所有文件")
    print("-" * 50)

    if not pending_pdfs:
        print("所有 PDF 已转换完成！")
        return stats['completed'], len(all_pdfs)

    success_count = stats['completed']

    # 批量转换（带进度条）
    for rel_pdf_path in tqdm(pending_pdfs, desc="转换进度", unit="PDF"):
        pdf_path = os.path.join(input_dir, rel_pdf_path)

        # 生成输出目录（与 PDF 同级）
        pdf_path_obj = Path(pdf_path)
        output_dir = pdf_path_obj.parent / f"{pdf_path_obj.stem}_pages"

        # 获取项目相对路径用于输出
        project_rel_path = _get_relative_path(pdf_path)

        try:
            # 转换 PDF（静默模式）
            image_paths = pdf_to_images(pdf_path, str(output_dir), dpi, image_format, verbose=False)
            if image_paths:
                success_count += 1
                tracker.mark_completed(rel_pdf_path)
            else:
                tracker.mark_failed(rel_pdf_path)
                tqdm.write(f"✗ 失败: {project_rel_path} - 未生成图片")
        except Exception as e:
            tracker.mark_failed(rel_pdf_path)
            tqdm.write(f"✗ 失败: {project_rel_path} - {str(e)}")

    print("-" * 50)
    print(f"=== 批量转换完成 ===")
    print(f"成功转换: {success_count}/{len(all_pdfs)} 个 PDF")

    failed_files = tracker.get_failed_files()
    if failed_files:
        print(f"失败文件: {len(failed_files)} 个")
        for f in failed_files[:5]:
            print(f"  - {f}")
        if len(failed_files) > 5:
            print(f"  ... 还有 {len(failed_files) - 5} 个")

    return success_count, len(all_pdfs)
