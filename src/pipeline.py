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


def cmd_preprocess(args):
    """预处理命令"""
    input_path = args.input

    if os.path.isfile(input_path):
        # 单文件处理
        output_path = args.output if args.output else None
        success = preprocess.preprocess_image(input_path, output_path)
        return 0 if success else 1

    elif os.path.isdir(input_path):
        # 目录批处理
        output_dir = args.output if args.output else config.PREPROCESSED_DIR
        success_count, total_count = preprocess.process_directory(input_path, output_dir)
        return 0 if success_count == total_count else 1

    else:
        print(f"错误：输入路径不存在或无效: {input_path}")
        return 1


def cmd_ocr(args):
    """OCR 识别命令"""
    input_path = args.input

    if os.path.isfile(input_path):
        # 单文件处理
        output_path = args.output if args.output else None
        result = ocr.ocr_image(input_path, output_path)
        return 0 if result.get('success') else 1

    elif os.path.isdir(input_path):
        # 目录批处理
        output_dir = args.output if args.output else config.OCR_DIR
        success_count, total_count = ocr.process_directory(input_path, output_dir)
        return 0 if success_count == total_count else 1

    else:
        print(f"错误：输入路径不存在或无效: {input_path}")
        return 1


def cmd_all(args):
    """完整流程：预处理 + OCR"""
    input_path = args.input

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
        success_count, total_count = preprocess.process_directory(input_path, config.PREPROCESSED_DIR)
        if success_count == 0:
            print("\n预处理失败，流程中止")
            return 1

        preprocessed_input = config.PREPROCESSED_DIR

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
        success_count, total_count = ocr.process_directory(preprocessed_input, config.OCR_DIR)
        return 0 if success_count == total_count else 1


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
  # 预处理
  ./pipeline preprocess data/raw/demo.jpg
  ./pipeline preprocess data/raw/

  # OCR 识别
  ./pipeline ocr data/results/preprocessed/demo_preprocessed.jpg
  ./pipeline ocr data/results/preprocessed/

  # 完整流程
  ./pipeline all data/raw/demo.jpg
  ./pipeline all data/raw/

  # 查看配置
  ./pipeline config
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='子命令')

    # preprocess 命令
    parser_preprocess = subparsers.add_parser('preprocess', help='图像预处理')
    parser_preprocess.add_argument('input', help='输入图片或目录')
    parser_preprocess.add_argument('-o', '--output', help='输出路径（可选）')

    # ocr 命令
    parser_ocr = subparsers.add_parser('ocr', help='OCR 文字识别')
    parser_ocr.add_argument('input', help='输入图片或目录')
    parser_ocr.add_argument('-o', '--output', help='输出路径（可选）')

    # all 命令
    parser_all = subparsers.add_parser('all', help='完整流程（预处理 + OCR）')
    parser_all.add_argument('input', help='输入图片或目录')

    # config 命令
    parser_config = subparsers.add_parser('config', help='显示配置信息')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # 执行相应命令
    if args.command == 'preprocess':
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
