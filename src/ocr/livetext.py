"""
OCR 模块 - 使用 macOS LiveText

对古籍图片进行整体 OCR 识别，获取文字和位置信息
"""
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

try:
    from ocrmac import ocrmac
except ImportError:
    ocrmac = None
    OCRMAC_IMPORT_ERROR = "ocrmac 未安装。请在 macOS 上运行：pip install ocrmac"
else:
    OCRMAC_IMPORT_ERROR = None

from src.config import OCR_CONFIG, OCR_DIR
from src.utils.path import ensure_dir


def ocr_image(image_path: str, output_path: str = None) -> Dict[str, Any]:
    """
    对图片进行 OCR 识别

    Args:
        image_path: 输入图片路径
        output_path: 输出 JSON 路径（可选，默认保存到 OCR_DIR）

    Returns:
        OCR 结果字典
    """
    # 检查 ocrmac 是否可用
    if ocrmac is None:
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
        # 执行 OCR
        cfg = OCR_CONFIG
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

        # 保存结果
        if output_path is None:
            # 默认保存到 OCR_DIR
            input_path_obj = Path(image_path)
            output_filename = f"{input_path_obj.stem}_ocr.json"
            output_path = os.path.join(OCR_DIR, output_filename)

        ensure_dir(os.path.dirname(output_path))

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"✓ OCR 完成，识别到 {len(characters)} 个字符")
        print(f"✓ 结果已保存至 {output_path}")

        return result

    except Exception as e:
        return {
            'success': False,
            'error': f'OCR 执行失败: {str(e)}',
            'image_path': image_path,
        }


def process_directory(input_dir: str, output_dir: str = None) -> tuple[int, int]:
    """
    批量处理目录下的所有图片

    Args:
        input_dir: 输入目录
        output_dir: 输出目录（默认为 OCR_DIR）

    Returns:
        (成功数量, 总数量)
    """
    if output_dir is None:
        output_dir = OCR_DIR

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

    print(f"\n=== 开始批量 OCR 识别 ===")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"找到 {len(image_files)} 个图片文件")
    print("-" * 50)

    success_count = 0
    for i, filename in enumerate(image_files, 1):
        input_path = os.path.join(input_dir, filename)

        # 生成输出路径
        file_name, _ = os.path.splitext(filename)
        output_filename = f"{file_name}_ocr.json"
        output_path = os.path.join(output_dir, output_filename)

        print(f"[{i}/{len(image_files)}] 处理: {filename}")

        result = ocr_image(input_path, output_path)
        if result.get('success'):
            success_count += 1
        else:
            print(f"失败: {filename} - {result.get('error', '未知错误')}")

    print("-" * 50)
    print(f"=== 批量 OCR 完成 ===")
    print(f"成功处理: {success_count}/{len(image_files)} 个文件")

    return success_count, len(image_files)
