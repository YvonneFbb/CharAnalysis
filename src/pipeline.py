"""
主流程管理模块
"""
import argparse
import os
import sys
from pathlib import Path

from src import config
from src.preprocess import core as preprocess
from src.ocr import livetext as ocr
from src.utils import pdf_converter


def cmd_preprocess(args):
    """预处理命令"""
    input_path = args.input
    force = args.force if hasattr(args, 'force') else False
    max_volumes = args.max_volumes if hasattr(args, 'max_volumes') else None

    if os.path.isfile(input_path):
        # 单文件处理
        output_path = args.output if args.output else None
        success = preprocess.preprocess_image(input_path, output_path)
        return 0 if success else 1

    elif os.path.isdir(input_path):
        # 目录批处理
        output_dir = args.output if args.output else str(config.PREPROCESSED_DIR)
        success_count, total_count = preprocess.process_directory(
            input_path, output_dir, force=force, max_volumes=max_volumes
        )
        return 0 if success_count == total_count else 1

    else:
        print(f"错误：输入路径不存在或无效: {input_path}")
        return 1


def cmd_ocr(args):
    """OCR 识别命令"""
    input_path = args.input
    force = args.force if hasattr(args, 'force') else False
    max_volumes = args.max_volumes if hasattr(args, 'max_volumes') else None

    if os.path.isfile(input_path):
        # 单文件处理
        output_path = args.output if args.output else None
        result = ocr.ocr_image(input_path, output_path)
        return 0 if result.get('success') else 1

    elif os.path.isdir(input_path):
        # 目录批处理
        output_dir = args.output if args.output else str(config.OCR_DIR)
        success_count, total_count = ocr.process_directory(
            input_path, output_dir, force=force, max_volumes=max_volumes
        )
        return 0 if success_count == total_count else 1

    else:
        print(f"错误：输入路径不存在或无效: {input_path}")
        return 1


def cmd_all(args):
    """完整流程：预处理 + OCR"""
    input_path = args.input
    force = args.force if hasattr(args, 'force') else False
    max_volumes = args.max_volumes if hasattr(args, 'max_volumes') else None

    print("\n" + "=" * 60)
    print("步骤 1/2: 预处理")
    print("=" * 60)

    # 预处理
    if os.path.isfile(input_path):
        preprocessed_file = os.path.join(
            config.PREPROCESSED_DIR,
            f"{Path(input_path).stem}_preprocessed{Path(input_path).suffix}"
        )
        success = preprocess.preprocess_image(input_path, preprocessed_file)
        if not success:
            print("\n预处理失败，流程中止")
            return 1

        preprocessed_input = preprocessed_file

    elif os.path.isdir(input_path):
        success_count, total_count = preprocess.process_directory(
            input_path, str(config.PREPROCESSED_DIR), force=force, max_volumes=max_volumes
        )
        if success_count == 0:
            print("\n预处理失败，流程中止")
            return 1

        preprocessed_input = str(config.PREPROCESSED_DIR)

    else:
        print(f"错误：输入路径不存在或无效: {input_path}")
        return 1

    print("\n" + "=" * 60)
    print("步骤 2/2: OCR 识别")
    print("=" * 60)

    # OCR 识别
    if os.path.isfile(preprocessed_input):
        result = ocr.ocr_image(preprocessed_input)
        return 0 if result.get('success') else 1

    else:
        success_count, total_count = ocr.process_directory(
            preprocessed_input, str(config.OCR_DIR), force=force, max_volumes=max_volumes
        )
        return 0 if success_count == total_count else 1


def cmd_pdf2images(args):
    """PDF 转图片命令"""
    input_path = args.input
    dpi = args.dpi if hasattr(args, 'dpi') else 300
    output_dir = args.output if args.output else None

    if os.path.isfile(input_path):
        # 单个 PDF 文件
        if not input_path.lower().endswith('.pdf'):
            print(f"错误：文件不是 PDF 格式: {input_path}")
            return 1

        try:
            image_paths = pdf_converter.pdf_to_images(
                input_path,
                output_dir=output_dir,
                dpi=dpi
            )
            return 0 if image_paths else 1
        except Exception as e:
            print(f"错误：PDF 转换失败: {e}")
            return 1

    elif os.path.isdir(input_path):
        # 目录批处理
        output_parent = output_dir if output_dir else input_path
        try:
            results = pdf_converter.convert_directory(
                input_path,
                output_parent_dir=output_parent,
                dpi=dpi
            )
            success_count = sum(1 for paths in results.values() if paths)
            return 0 if success_count > 0 else 1
        except Exception as e:
            print(f"错误：批量转换失败: {e}")
            return 1

    else:
        print(f"错误：输入路径不存在或无效: {input_path}")
        return 1


def cmd_convert(args):
    """批量转换多本书的 PDF（支持 max-volumes 和进度跟踪）"""
    input_dir = args.input
    dpi = args.dpi if hasattr(args, 'dpi') else 300
    force = args.force if hasattr(args, 'force') else False
    max_volumes = args.max_volumes if hasattr(args, 'max_volumes') else None

    if not os.path.isdir(input_dir):
        print(f"错误：输入路径必须是目录: {input_dir}")
        return 1

    try:
        success_count, total_count = pdf_converter.convert_books_directory(
            input_dir,
            dpi=dpi,
            max_volumes=max_volumes,
            force=force
        )
        return 0 if success_count > 0 else 1
    except Exception as e:
        print(f"错误：批量转换失败: {e}")
        return 1


def cmd_config(args):
    """显示配置信息"""
    summary = config.config_summary()

    print("\n" + "=" * 60)
    print("项目配置信息")
    print("=" * 60)

    print("\n路径配置:")
    for key, value in summary['paths'].items():
        print(f"  {key}: {value}")

    print("\n预处理配置:")
    for key, value in summary['preprocess'].items():
        print(f"  {key}: {value}")

    print("\nOCR 配置:")
    for key, value in summary['ocr'].items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)

    return 0


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='古籍文字识别与分析流水线',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 批量转换多本书的 PDF（推荐）
  ./pipeline convert data/raw/ --max-volumes 5
  ./pipeline convert data/raw/ --max-volumes 5 --force

  # PDF 转图片（单个文件或单本书）
  ./pipeline pdf2images data/raw/01_1127_尚书正义/册01.pdf
  ./pipeline pdf2images data/raw/01_1127_尚书正义/ --dpi 300

  # 预处理
  ./pipeline preprocess data/raw/demo.jpg
  ./pipeline preprocess data/raw/ --max-volumes 5

  # OCR 识别
  ./pipeline ocr data/results/preprocessed/demo_preprocessed.jpg
  ./pipeline ocr data/results/preprocessed/ --max-volumes 5

  # 完整流程
  ./pipeline all data/raw/demo.jpg
  ./pipeline all data/raw/ --max-volumes 5

  # 查看配置
  ./pipeline config
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='子命令')

    # convert 命令（批量转换多本书）
    parser_convert = subparsers.add_parser('convert', help='批量转换多本书的 PDF')
    parser_convert.add_argument('input', help='输入目录（包含多本书）')
    parser_convert.add_argument('--dpi', type=int, default=300, help='分辨率（默认 300）')
    parser_convert.add_argument('--max-volumes', type=int, metavar='N', help='限制每本书最多处理的册数（如 5）')
    parser_convert.add_argument('--force', action='store_true', help='强制重新处理所有文件（忽略进度记录）')

    # pdf2images 命令
    parser_pdf = subparsers.add_parser('pdf2images', help='PDF 转图片')
    parser_pdf.add_argument('input', help='输入 PDF 文件或目录')
    parser_pdf.add_argument('-o', '--output', help='输出目录（可选）')
    parser_pdf.add_argument('--dpi', type=int, default=300, help='分辨率（默认 300）')

    # preprocess 命令
    parser_preprocess = subparsers.add_parser('preprocess', help='图像预处理')
    parser_preprocess.add_argument('input', help='输入图片或目录')
    parser_preprocess.add_argument('-o', '--output', help='输出路径（可选）')
    parser_preprocess.add_argument('--force', action='store_true', help='强制重新处理所有文件（忽略进度记录）')
    parser_preprocess.add_argument('--max-volumes', type=int, metavar='N', help='限制每本书最多处理的册数（如 5）')

    # ocr 命令
    parser_ocr = subparsers.add_parser('ocr', help='OCR 文字识别')
    parser_ocr.add_argument('input', help='输入图片或目录')
    parser_ocr.add_argument('-o', '--output', help='输出路径（可选）')
    parser_ocr.add_argument('--force', action='store_true', help='强制重新处理所有文件（忽略进度记录）')
    parser_ocr.add_argument('--max-volumes', type=int, metavar='N', help='限制每本书最多处理的册数（如 5）')

    # all 命令
    parser_all = subparsers.add_parser('all', help='完整流程（预处理 + OCR）')
    parser_all.add_argument('input', help='输入图片或目录')
    parser_all.add_argument('--force', action='store_true', help='强制重新处理所有文件（忽略进度记录）')
    parser_all.add_argument('--max-volumes', type=int, metavar='N', help='限制每本书最多处理的册数（如 5）')

    # config 命令
    parser_config = subparsers.add_parser('config', help='显示配置信息')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # 执行相应命令
    if args.command == 'convert':
        return cmd_convert(args)
    elif args.command == 'pdf2images':
        return cmd_pdf2images(args)
    elif args.command == 'preprocess':
        return cmd_preprocess(args)
    elif args.command == 'ocr':
        return cmd_ocr(args)
    elif args.command == 'all':
        return cmd_all(args)
    elif args.command == 'config':
        return cmd_config(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
