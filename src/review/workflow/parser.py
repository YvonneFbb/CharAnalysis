"""
Pipeline CLI parser.

Defines command-line interface only. Command execution is handled in
`src.review.workflow.commands`.
"""

from __future__ import annotations

import argparse

from src.review import config


PIPELINE_EPILOG = """
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

  # 标准字匹配（默认只生成分片 matched_books）
  ./pipeline match data/results/ocr

  # 如需生成聚合文件（供旧 crop 工具使用）
  ./pipeline match data/results/ocr -o data/results/matched_by_book.json
"""


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="古籍文字识别与分析流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=PIPELINE_EPILOG,
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    parser_convert = subparsers.add_parser("convert", help="批量转换多本书的 PDF")
    parser_convert.add_argument("input", help="输入目录（包含多本书）")
    parser_convert.add_argument("--dpi", type=int, default=300, help="分辨率（默认 300）")
    parser_convert.add_argument("--max-volumes", type=int, metavar="N", help="限制每本书最多处理的册数（如 5）")
    parser_convert.add_argument("--force", action="store_true", help="强制重新处理所有文件（忽略进度记录）")
    parser_convert.add_argument("--workers", type=int, default=1, metavar="N", help="并发线程数（默认 1）")

    parser_pdf = subparsers.add_parser("pdf2images", help="PDF 转图片")
    parser_pdf.add_argument("input", help="输入 PDF 文件或目录")
    parser_pdf.add_argument("-o", "--output", help="输出目录（可选）")
    parser_pdf.add_argument("--dpi", type=int, default=300, help="分辨率（默认 300）")

    parser_preprocess = subparsers.add_parser("preprocess", help="图像预处理（如有 PDF 会先转换）")
    parser_preprocess.add_argument("input", help="输入图片或目录")
    parser_preprocess.add_argument("-o", "--output", help="输出路径（可选）")
    parser_preprocess.add_argument("--force", action="store_true", help="强制重新处理所有文件（忽略进度记录）")
    parser_preprocess.add_argument("--max-volumes", type=int, metavar="N", help="限制每本书最多处理的册数（如 5）")
    parser_preprocess.add_argument("--workers", type=int, default=1, metavar="N", help="并发线程数（默认 1）")

    parser_ocr = subparsers.add_parser("ocr", help="OCR 文字识别")
    parser_ocr.add_argument("input", help="输入图片或目录")
    parser_ocr.add_argument("-o", "--output", help="输出路径（可选）")
    parser_ocr.add_argument("--force", action="store_true", help="强制重新处理所有文件（忽略进度记录）")
    parser_ocr.add_argument("--max-volumes", type=int, metavar="N", help="限制每本书最多处理的册数（如 5）")
    parser_ocr.add_argument("--workers", type=int, default=1, metavar="N", help="并发线程数（默认 1）")

    parser_all = subparsers.add_parser("all", help="完整流程（预处理 + OCR）")
    parser_all.add_argument("input", help="输入图片或目录")
    parser_all.add_argument("--force", action="store_true", help="强制重新处理所有文件（忽略进度记录）")
    parser_all.add_argument("--max-volumes", type=int, metavar="N", help="限制每本书最多处理的册数（如 5）")
    parser_all.add_argument("--workers", type=int, default=1, metavar="N", help="并发线程数（默认 1）")

    parser_match = subparsers.add_parser("match", help="匹配标准字")
    parser_match.add_argument("ocr_dir", help="OCR 结果目录")
    parser_match.add_argument("-o", "--output", help="输出聚合 JSON 路径（可选，不填则只生成分片）")
    parser_match.add_argument("--standard-chars", help="标准字 JSON 文件路径（可选）")

    parser_crop = subparsers.add_parser("crop", help="裁切字符图像")
    parser_crop.add_argument("matched_chars_json", help="匹配结果 JSON 文件（聚合或单本书分片）")
    parser_crop.add_argument("-o", "--output", help="输出目录（默认 data/results/chars）")
    parser_crop.add_argument("--padding", type=int, default=5, help="边界填充像素数（默认 5）")
    parser_crop.add_argument("--book", help="指定书名（不指定则处理所有书籍）")

    subparsers.add_parser("config", help="显示配置信息")

    parser_paddle = subparsers.add_parser("paddle", help="Paddle 评分/筛选流程")
    parser_paddle.add_argument("--paddle-url", default=None, help="PaddleOCR HTTP 服务地址（base 或 /ocr/predict_base64 完整地址）")
    parser_paddle.add_argument("--books", nargs="+", help="仅处理指定书籍")
    parser_paddle.add_argument("--topk", type=int, default=config.PADDLE_CONFIG.get("topk", 5), help="每字保留 TopK（默认 5）")
    parser_paddle.add_argument("--timeout", type=int, default=config.PADDLE_CONFIG.get("timeout", 20), help="PaddleOCR 超时（秒）")
    parser_paddle.add_argument(
        "--min-conf",
        type=float,
        default=config.PADDLE_CONFIG.get("min_conf", 0.75),
        help="置信度阈值（默认 0.75，可输入 75）",
    )
    parser_paddle.add_argument("--batch-size", type=int, default=config.PADDLE_CONFIG.get("batch_size", 8), help="Paddle 批处理大小（默认 8）")
    parser_paddle.add_argument("--limit-chars", type=int, default=None, help="仅处理前 N 个字（调试用）")
    parser_paddle.add_argument("--limit-instances", type=int, default=None, help="每个字仅处理前 N 个实例（调试用）")
    parser_paddle.add_argument("--workers", type=int, default=config.PADDLE_CONFIG.get("workers", 1), help="Paddle 并发进程数（默认 1）")

    return parser

