"""Pipeline CLI parser."""

from __future__ import annotations

import argparse

from src.review import config


PIPELINE_DESCRIPTION = """
古籍文字识别与分析流水线

推荐按阶段运行：
  阶段 0  convert / pdf2images   原始 PDF -> 页图
  阶段 1  preprocess             页图预处理
  阶段 2  ocr                    整页 OCR
  阶段 3  match                  标准字匹配
  阶段 4  segment                单字切割并写入 atlas
  阶段 5  reocr                  对切割结果做二次 OCR
"""


PIPELINE_EPILOG = """
主流程示例:
  ./pipeline convert data/raw/ --max-volumes 5
  ./pipeline preprocess data/raw/ --max-volumes 5
  ./pipeline ocr data/results/preprocessed/ --max-volumes 5
  ./pipeline match data/results/ocr
  ./pipeline segment --workers 8
  ./pipeline reocr --workers 8

局部重跑:
  ./pipeline segment --books 01_1127_尚书正义 --chars 宣 意
  ./pipeline reocr --books 01_1127_尚书正义 --engine paddle --paddle-url http://127.0.0.1:8000

补充说明:
  `all` / `prepare-filter` / `paddle` / `crop` 仍可用，但属于兼容或调试入口，不再作为主帮助展示。
"""


VISIBLE_COMMANDS_METAVAR = "{convert,pdf2images,preprocess,ocr,match,segment,reocr,config}"


def _add_hidden_parser(subparsers, name: str, **kwargs):
    parser = subparsers.add_parser(name, help=argparse.SUPPRESS, **kwargs)
    if getattr(subparsers, "_choices_actions", None):
        subparsers._choices_actions.pop()
    return parser


def _add_input_output_args(parser, *, require_input: bool = True) -> None:
    if require_input:
        parser.add_argument("input", help="输入图片或目录")
    parser.add_argument("-o", "--output", help="输出路径（可选）")


def _add_volume_processing_args(parser, *, default_workers: int = 1) -> None:
    parser.add_argument("--force", action="store_true", help="强制重新处理所有文件（忽略进度记录）")
    parser.add_argument("--max-volumes", type=int, metavar="N", help="限制每本书最多处理的册数（如 5）")
    parser.add_argument("--workers", type=int, default=default_workers, metavar="N", help=f"并发线程数（默认 {default_workers}）")


def _add_book_char_scope_args(parser, *, default_workers: int = 8) -> None:
    parser.add_argument("--books", nargs="+", help="仅处理指定书籍")
    parser.add_argument("--chars", nargs="+", help="仅处理指定字符")
    parser.add_argument("--limit-chars", type=int, default=None, help="每本书仅处理前 N 个字（调试用）")
    parser.add_argument("--limit-instances", type=int, default=None, help="每个字仅处理前 N 个实例（调试用）")
    parser.add_argument("--workers", type=int, default=default_workers, metavar="N", help=f"按书并行的进程数（默认 {default_workers}）")


def _register_stage_zero_parsers(subparsers) -> None:
    parser_convert = subparsers.add_parser("convert", help="阶段 0: 批量转换多本书的 PDF")
    parser_convert.add_argument("input", help="输入目录（包含多本书）")
    parser_convert.add_argument("--dpi", type=int, default=300, help="分辨率（默认 300）")
    parser_convert.add_argument("--max-volumes", type=int, metavar="N", help="限制每本书最多处理的册数（如 5）")
    parser_convert.add_argument("--force", action="store_true", help="强制重新处理所有文件（忽略进度记录）")
    parser_convert.add_argument("--workers", type=int, default=1, metavar="N", help="并发线程数（默认 1）")

    parser_pdf = subparsers.add_parser("pdf2images", help="阶段 0: 单 PDF 或单书目录转图片")
    parser_pdf.add_argument("input", help="输入 PDF 文件或目录")
    parser_pdf.add_argument("-o", "--output", help="输出目录（可选）")
    parser_pdf.add_argument("--dpi", type=int, default=300, help="分辨率（默认 300）")


def _register_stage_one_two_parsers(subparsers) -> None:
    parser_preprocess = subparsers.add_parser("preprocess", help="阶段 1: 图像预处理（如有 PDF 会先转换）")
    _add_input_output_args(parser_preprocess)
    _add_volume_processing_args(parser_preprocess, default_workers=1)

    parser_ocr = subparsers.add_parser("ocr", help="阶段 2: OCR 文字识别")
    _add_input_output_args(parser_ocr)
    _add_volume_processing_args(parser_ocr, default_workers=1)


def _register_stage_three_five_parsers(subparsers) -> None:
    parser_match = subparsers.add_parser("match", help="阶段 3: 匹配标准字")
    parser_match.add_argument("ocr_dir", help="OCR 结果目录")
    parser_match.add_argument("-o", "--output", help="输出聚合 JSON 路径（可选，不填则只生成分片）")
    parser_match.add_argument("--standard-chars", help="标准字 JSON 文件路径（可选）")

    parser_segment = subparsers.add_parser("segment", help="阶段 4: 为 matched 实例生成 segment atlas")
    _add_book_char_scope_args(parser_segment, default_workers=8)
    parser_segment.add_argument("--force", action="store_true", help="强制重建指定书籍的 segment atlas")

    parser_reocr = subparsers.add_parser("reocr", help="阶段 5: 对 segment atlas 执行 reOCR")
    _add_book_char_scope_args(parser_reocr, default_workers=8)
    parser_reocr.add_argument("--force", action="store_true", help="强制重跑已存在的 reOCR 结果")
    parser_reocr.add_argument("--engine", choices=["livetext", "paddle"], default="livetext", help="reOCR 引擎（默认 livetext）")
    parser_reocr.add_argument("--pad", type=int, default=4, help="segment crop 外扩像素（默认 4）")
    parser_reocr.add_argument("--paddle-url", default=None, help="PaddleOCR HTTP 服务地址（仅 --engine paddle 生效）")
    parser_reocr.add_argument("--timeout", type=int, default=config.PADDLE_CONFIG.get("timeout", 20), help="reOCR 超时（秒）")
    parser_reocr.add_argument("--batch-size", type=int, default=config.PADDLE_CONFIG.get("batch_size", 32), help="Paddle 批处理大小（默认 32）")


def _register_utility_parsers(subparsers) -> None:
    subparsers.add_parser("config", help="工具: 显示配置信息")


def _register_legacy_parsers(subparsers) -> None:
    parser_all = _add_hidden_parser(subparsers, "all")
    parser_all.add_argument("input", help="输入图片或目录")
    parser_all.add_argument("--force", action="store_true", help="强制重新处理所有文件（忽略进度记录）")
    parser_all.add_argument("--max-volumes", type=int, metavar="N", help="限制每本书最多处理的册数（如 5）")
    parser_all.add_argument("--workers", type=int, default=1, metavar="N", help="并发线程数（默认 1）")

    parser_prepare_filter = _add_hidden_parser(subparsers, "prepare-filter")
    _add_book_char_scope_args(parser_prepare_filter, default_workers=8)
    parser_prepare_filter.add_argument("--force", action="store_true", help="强制重新计算已存在的 preview/reOCR")
    parser_prepare_filter.add_argument("--reocr-engine", choices=["livetext", "paddle"], default="livetext", help="reOCR 引擎（默认 livetext）")
    parser_prepare_filter.add_argument("--reocr-pad", type=int, default=4, help="segment crop 外扩像素（默认 4）")
    parser_prepare_filter.add_argument("--paddle-url", default=None, help="PaddleOCR HTTP 服务地址（仅 paddle 生效）")
    parser_prepare_filter.add_argument("--timeout", type=int, default=config.PADDLE_CONFIG.get("timeout", 20), help="reOCR 超时（秒）")
    parser_prepare_filter.add_argument("--batch-size", type=int, default=config.PADDLE_CONFIG.get("batch_size", 32), help="Paddle 批处理大小（默认 32）")

    parser_crop = _add_hidden_parser(subparsers, "crop")
    parser_crop.add_argument("matched_chars_json", help="匹配结果 JSON 文件（聚合或单本书分片）")
    parser_crop.add_argument("-o", "--output", help="输出目录（默认 data/results/chars）")
    parser_crop.add_argument("--padding", type=int, default=5, help="边界填充像素数（默认 5）")
    parser_crop.add_argument("--book", help="指定书名（不指定则处理所有书籍）")

    parser_paddle = _add_hidden_parser(subparsers, "paddle")
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


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=PIPELINE_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=PIPELINE_EPILOG,
    )
    subparsers = parser.add_subparsers(
        dest="command",
        title="主命令",
        help="按阶段选择子命令",
        metavar=VISIBLE_COMMANDS_METAVAR,
    )

    _register_stage_zero_parsers(subparsers)
    _register_stage_one_two_parsers(subparsers)
    _register_stage_three_five_parsers(subparsers)
    _register_utility_parsers(subparsers)
    _register_legacy_parsers(subparsers)
    return parser
