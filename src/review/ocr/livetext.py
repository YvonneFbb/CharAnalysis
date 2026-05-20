"""
OCR 模块 - 使用 macOS LiveText

对古籍图片进行整体 OCR 识别，获取文字和位置信息
"""
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from ocrmac import ocrmac
    LIVETEXT_AVAILABLE = True
except ImportError:
    ocrmac = None
    LIVETEXT_AVAILABLE = False
    OCRMAC_IMPORT_ERROR = "ocrmac 或相关依赖未安装。请在 macOS 上运行：pip install ocrmac"
else:
    OCRMAC_IMPORT_ERROR = None

from src.review.config import OCR_CONFIG, OCR_DIR, PREPROCESSED_DIR
from src.review.utils.path import ensure_dir
from src.review.utils.progress import ProgressTracker, get_default_progress_file, _get_relative_path
from src.review.utils.file_filter import find_images_recursive, filter_files_by_max_volumes



def ocr_image(
    image_path: str,
    output_path: str = None,
    verbose: bool = True,
    save_output: bool = True,
) -> Dict[str, Any]:
    """
    对图片进行 OCR 识别

    Args:
        image_path: 输入图片路径
        output_path: 输出 JSON 路径（可选，默认保存到 OCR_DIR）
        verbose: 是否输出详细信息

    Returns:
        OCR 结果字典
    """
    # 检查依赖是否可用
    if not LIVETEXT_AVAILABLE:
        return {
            'success': False,
            'error': OCRMAC_IMPORT_ERROR,
            'image_path': image_path,
        }

    # 检查图片是否存在
    if not os.path.exists(image_path):
        return {
            'success': False,
            'error': f'图片不存在: {image_path}',
            'image_path': image_path,
        }

    try:
        cfg = OCR_CONFIG

        # 使用 ocrmac 库
        ocr_obj = ocrmac.OCR(
            image_path,
            framework=cfg['framework'],
            recognition_level=cfg['recognition_level'],
            language_preference=cfg['language_preference'],
            detail=True
        )
        results = ocr_obj.recognize()

        # 获取图片尺寸
        W, H = int(ocr_obj.image.width), int(ocr_obj.image.height)

        # 解析结果
        if not results:
            return {
                'success': False,
                'error': 'LiveText 无检测结果',
                'image_path': image_path,
                'image_size': {'width': W, 'height': H},
            }

        # 处理每个检测到的文字
        characters = []
        for i, item in enumerate(results):
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                continue

            text, confidence, normalized_box = item

            # 将归一化坐标转换为像素坐标
            # normalized_box: [x, y_bottom, width, height] (归一化值 0-1)
            # 注意：y 坐标系原点在左下角
            x = int(round(normalized_box[0] * W))
            y_bottom = normalized_box[1]
            w = int(round(normalized_box[2] * W))
            h = int(round(normalized_box[3] * H))

            # 转换为左上角坐标
            y_top = int(round((1.0 - y_bottom - normalized_box[3]) * H))

            # 边界检查
            x = max(0, min(W - 1, x))
            y_top = max(0, min(H - 1, y_top))
            w = max(1, min(W - x, w))
            h = max(1, min(H - y_top, h))

            characters.append({
                'index': i,
                'text': str(text),
                'confidence': float(confidence),
                'bbox': {
                    'x': x,
                    'y': y_top,
                    'width': w,
                    'height': h,
                },
                'normalized_bbox': {
                    'x': float(normalized_box[0]),
                    'y': float(normalized_box[1]),
                    'width': float(normalized_box[2]),
                    'height': float(normalized_box[3]),
                },
            })

        # 按位置排序（从上到下，从左到右）
        characters.sort(key=lambda c: (c['bbox']['y'], c['bbox']['x']))

        # 构建结果
        result = {
            'success': True,
            'image_path': image_path,
            'image_size': {'width': W, 'height': H},
            'timestamp': datetime.now().isoformat(),
            'ocr_config': cfg,
            'character_count': len(characters),
            'characters': characters,
            'full_text': ''.join([c['text'] for c in characters]),
        }

        if save_output:
            if output_path is None:
                input_path_obj = Path(image_path)
                output_filename = f"{input_path_obj.stem}_ocr.json"
                output_path = os.path.join(OCR_DIR, output_filename)

            ensure_dir(os.path.dirname(output_path))

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

        if verbose:
            print(f"✓ OCR 完成，识别到 {len(characters)} 个字符")
            if save_output and output_path:
                print(f"✓ 结果已保存至 {output_path}")

        return result

    except Exception as e:
        return {
            'success': False,
            'error': f'OCR 执行失败: {str(e)}',
            'image_path': image_path,
        }


def _ocr_worker(args: Tuple[str, str, str]) -> Dict:
    """
    多进程 worker 函数（需要在模块顶层定义以支持序列化）

    Args:
        args: (rel_filepath, input_dir, output_dir)

    Returns:
        {'success': bool, 'rel_path': str, 'error': str or None}
    """
    rel_filepath, input_dir, output_dir = args

    input_path = os.path.join(input_dir, rel_filepath)
    output_path = os.path.join(output_dir, rel_filepath)
    output_dir_part, output_filename = os.path.split(output_path)
    name, ext = os.path.splitext(output_filename)
    output_filename = f"{name}_ocr.json"
    output_path = os.path.join(output_dir_part, output_filename)

    try:
        result = ocr_image(input_path, output_path, verbose=False)
        if result.get('success'):
            return {'success': True, 'rel_path': rel_filepath, 'error': None}
        else:
            error_msg = result.get('error', 'OCR失败')
            return {'success': False, 'rel_path': rel_filepath, 'error': error_msg}
    except Exception as e:
        return {'success': False, 'rel_path': rel_filepath, 'error': str(e)}


def process_directory(input_dir: str, output_dir: str = None,
                      force: bool = False, max_volumes: int = None,
                      workers: int = 1,
                      volume_overrides: dict = None) -> tuple[int, int]:
    """
    批量处理目录下的所有图片

    Args:
        input_dir: 输入目录
        output_dir: 输出目录（默认为 OCR_DIR）
        force: 是否强制重新处理（忽略进度记录）
        max_volumes: 最大册数限制（None 表示不限制）
        volume_overrides: 每本书的册数起点与数量覆盖配置
        workers: 并发线程数（默认1，即单线程）

    Returns:
        (成功数量, 总数量)
    """
    # 确保路径是字符串
    input_dir = str(input_dir)

    if output_dir is None:
        output_dir = str(OCR_DIR)
    else:
        output_dir = str(output_dir)

    input_dir_path = Path(input_dir).resolve()
    output_dir_path = Path(output_dir).resolve()
    preprocessed_root = PREPROCESSED_DIR.resolve()
    if output_dir_path == OCR_DIR.resolve() and input_dir_path.parent == preprocessed_root:
        output_dir = str(output_dir_path / input_dir_path.name)

    ensure_dir(output_dir)

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
        image_files, max_volumes, base_dir=input_dir, volume_overrides=volume_overrides
    )

    # 初始化进度跟踪器
    progress_file = get_default_progress_file(output_dir)
    tracker = ProgressTracker(progress_file, 'ocr')
    tracker.init_session(input_dir, output_dir, force=force)

    # 获取待处理的文件列表
    pending_files = tracker.get_pending_files(filtered_files)
    # 按文件名排序，确保处理顺序一致
    pending_files = sorted(pending_files)
    stats = tracker.get_stats()

    print(f"\n=== 开始批量 OCR 识别 ===")
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
    if stats['completed'] > 0 and not force:
        print(f"💡 继续上次进度 (最后更新: {stats['last_update']})")
        print(f"💡 使用 --force 强制重新处理所有文件")
    print("-" * 50)

    if not pending_files:
        print("所有文件已处理完成！")
        return stats['completed'], len(filtered_files)

    success_count = stats['completed']  # 从已完成数量开始

    if workers == 1:
        # 单进程模式
        for rel_filepath in tqdm(pending_files, desc="OCR进度", unit="file"):
            input_path = os.path.join(input_dir, rel_filepath)
            output_path = os.path.join(output_dir, rel_filepath)
            output_dir_part, output_filename = os.path.split(output_path)
            name, ext = os.path.splitext(output_filename)
            output_filename = f"{name}_ocr.json"
            output_path = os.path.join(output_dir_part, output_filename)
            project_rel_path = _get_relative_path(input_path)

            try:
                result = ocr_image(input_path, output_path, verbose=False)
                if result.get('success'):
                    success_count += 1
                    tracker.mark_completed(rel_filepath)
                else:
                    tracker.mark_failed(rel_filepath)
                    tqdm.write(f"✗ 失败: {project_rel_path}")
            except Exception as e:
                tracker.mark_failed(rel_filepath)
                tqdm.write(f"✗ 失败: {project_rel_path} - {str(e)}")
    else:
        # 多进程模式
        # 准备参数列表
        task_args = [(rel_filepath, input_dir, output_dir) for rel_filepath in pending_files]

        completed_files = []
        failed_files = []

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_ocr_worker, args): args[0] for args in task_args}
            with tqdm(total=len(pending_files), desc=f"OCR进度 ({workers}进程)", unit="file") as pbar:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result['success']:
                            success_count += 1
                            completed_files.append(result['rel_path'])
                        else:
                            failed_files.append(result['rel_path'])
                            project_rel_path = _get_relative_path(
                                os.path.join(input_dir, result['rel_path'])
                            )
                            tqdm.write(f"✗ 失败: {project_rel_path} - {result['error']}")
                    except Exception as e:
                        rel_path = futures[future]
                        failed_files.append(rel_path)
                        project_rel_path = _get_relative_path(os.path.join(input_dir, rel_path))
                        tqdm.write(f"✗ 失败: {project_rel_path} - {str(e)}")
                    finally:
                        pbar.update(1)

        # 批量更新进度（避免文件锁竞争）
        if completed_files:
            tracker.mark_completed_batch(completed_files)
        if failed_files:
            tracker.mark_failed_batch(failed_files)

    print("-" * 50)
    print(f"=== 批量 OCR 完成 ===")
    print(f"成功处理: {success_count}/{len(filtered_files)} 个文件")

    failed_files = tracker.get_failed_files()
    if failed_files:
        print(f"失败文件: {len(failed_files)} 个")
        for f in failed_files[:5]:  # 只显示前5个
            print(f"  - {f}")
        if len(failed_files) > 5:
            print(f"  ... 还有 {len(failed_files) - 5} 个")

    return success_count, len(filtered_files)
