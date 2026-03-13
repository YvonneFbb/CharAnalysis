"""
Pipeline command handlers.

This module contains executable command logic used by `src.review.pipeline`.
It intentionally preserves existing command behavior and argument semantics.
"""

from __future__ import annotations

import os
from argparse import Namespace
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, TypeVar

from src.review import config


def _import_preprocess():
    from src.review.preprocess import core as preprocess

    return preprocess


def _import_pdf_converter():
    from src.review.utils import pdf_converter

    return pdf_converter


def _import_match_standard_chars():
    from src.review.filter import match_standard_chars

    return match_standard_chars


def _import_crop_characters():
    from src.review.crop import crop_characters

    return crop_characters


def _import_paddle_core():
    from src.review.paddle import core as paddle_core

    return paddle_core


def _common_options(args: Namespace) -> Tuple[bool, Optional[int], int, Optional[dict]]:
    force = bool(getattr(args, "force", False))
    max_volumes = getattr(args, "max_volumes", None)
    workers = int(getattr(args, "workers", 1) or 1)
    volume_overrides = getattr(config, "VOLUME_OVERRIDES", None)
    return force, max_volumes, workers, volume_overrides


T = TypeVar("T")


def _dispatch_input_path(
    input_path: str,
    on_file: Callable[[str], T],
    on_dir: Callable[[str], T],
) -> T:
    if os.path.isfile(input_path):
        return on_file(input_path)
    if os.path.isdir(input_path):
        return on_dir(input_path)
    raise FileNotFoundError(input_path)


def _import_livetext():
    try:
        from src.review.ocr import livetext as ocr
    except Exception as e:
        print(f"错误：OCR 模块不可用（可能仅支持 macOS LiveText）：{e}")
        return None
    return ocr


def _maybe_convert_pdfs(
    input_path: str,
    max_volumes: Optional[int],
    force: bool,
    workers: int,
    volume_overrides: Optional[dict],
) -> bool:
    pdf_converter = _import_pdf_converter()
    if not os.path.isdir(input_path):
        return True

    has_pdf_in_root = any(
        name.lower().endswith(".pdf")
        for name in os.listdir(input_path)
        if os.path.isfile(os.path.join(input_path, name))
    )

    has_pdf = has_pdf_in_root
    if not has_pdf:
        for root, _, files in os.walk(input_path):
            if any(name.lower().endswith(".pdf") for name in files):
                has_pdf = True
                break

    if not has_pdf:
        return True

    print("\n" + "=" * 60)
    print("步骤 0/2: PDF 转图片")
    print("=" * 60)

    try:
        if has_pdf_in_root:
            book_name = Path(input_path).name
            results = pdf_converter.convert_directory(
                input_path,
                output_parent_dir=input_path,
                dpi=300,
                max_volumes=max_volumes,
                force=force,
                volume_overrides=volume_overrides,
                book_name=book_name,
            )
            success_count = sum(1 for paths in results.values() if paths)
            return success_count > 0

        success_count, total_count = pdf_converter.convert_books_directory(
            input_path,
            dpi=300,
            max_volumes=max_volumes,
            force=force,
            workers=workers,
            volume_overrides=volume_overrides,
        )
        return success_count > 0 or total_count == 0
    except Exception as e:
        print(f"错误：批量转换失败: {e}")
        return False


def _run_preprocess_dir(
    input_dir: str,
    output_dir: str,
    force: bool,
    max_volumes: Optional[int],
    workers: int,
    volume_overrides: Optional[dict],
) -> Optional[Tuple[int, int]]:
    preprocess = _import_preprocess()
    if not _maybe_convert_pdfs(input_dir, max_volumes, force, workers, volume_overrides):
        return None
    return preprocess.process_directory(
        input_dir,
        output_dir,
        force=force,
        max_volumes=max_volumes,
        workers=workers,
        volume_overrides=volume_overrides,
    )


def _run_ocr_dir(
    ocr,
    input_dir: str,
    output_dir: str,
    force: bool,
    max_volumes: Optional[int],
    workers: int,
    volume_overrides: Optional[dict],
) -> Tuple[int, int]:
    return ocr.process_directory(
        input_dir,
        output_dir,
        force=force,
        max_volumes=max_volumes,
        workers=workers,
        volume_overrides=volume_overrides,
    )


def _prepare_preprocessed_input_for_all(
    input_path: str,
    force: bool,
    max_volumes: Optional[int],
    workers: int,
    volume_overrides: Optional[dict],
) -> Tuple[int, Optional[str]]:
    preprocess = _import_preprocess()
    try:
        return _dispatch_input_path(
            input_path,
            on_file=lambda path: _prepare_preprocessed_input_for_all_file(preprocess, path),
            on_dir=lambda path: _prepare_preprocessed_input_for_all_dir(
                path, force, max_volumes, workers, volume_overrides
            ),
        )
    except FileNotFoundError:
        print(f"错误：输入路径不存在或无效: {input_path}")
        return 1, None


def _prepare_preprocessed_input_for_all_file(preprocess, input_path: str) -> Tuple[int, Optional[str]]:
    preprocessed_file = os.path.join(
        str(config.PREPROCESSED_DIR),
        f"{Path(input_path).stem}_preprocessed{Path(input_path).suffix}",
    )
    success = preprocess.preprocess_image(input_path, preprocessed_file)
    if not success:
        print("\n预处理失败，流程中止")
        return 1, None
    return 0, preprocessed_file


def _prepare_preprocessed_input_for_all_dir(
    input_path: str,
    force: bool,
    max_volumes: Optional[int],
    workers: int,
    volume_overrides: Optional[dict],
) -> Tuple[int, Optional[str]]:
    result = _run_preprocess_dir(
        input_path,
        str(config.PREPROCESSED_DIR),
        force,
        max_volumes,
        workers,
        volume_overrides,
    )
    if result is None:
        return 1, None
    success_count, _ = result
    if success_count == 0:
        print("\n预处理失败，流程中止")
        return 1, None
    return 0, str(config.PREPROCESSED_DIR)


def cmd_preprocess(args: Namespace) -> int:
    preprocess = _import_preprocess()
    input_path = args.input
    force, max_volumes, workers, volume_overrides = _common_options(args)
    try:
        return _dispatch_input_path(
            input_path,
            on_file=lambda path: 0 if preprocess.preprocess_image(path, args.output if args.output else None) else 1,
            on_dir=lambda path: _cmd_preprocess_dir(
                path,
                output_dir=args.output if args.output else str(config.PREPROCESSED_DIR),
                force=force,
                max_volumes=max_volumes,
                workers=workers,
                volume_overrides=volume_overrides,
            ),
        )
    except FileNotFoundError:
        print(f"错误：输入路径不存在或无效: {input_path}")
        return 1


def _cmd_preprocess_dir(
    input_path: str,
    output_dir: str,
    force: bool,
    max_volumes: Optional[int],
    workers: int,
    volume_overrides: Optional[dict],
) -> int:
    result = _run_preprocess_dir(
        input_path, output_dir, force, max_volumes, workers, volume_overrides
    )
    if result is None:
        return 1
    success_count, total_count = result
    return 0 if success_count == total_count else 1


def cmd_ocr(args: Namespace) -> int:
    ocr = _import_livetext()
    if ocr is None:
        return 1

    input_path = args.input
    force, max_volumes, workers, volume_overrides = _common_options(args)
    try:
        return _dispatch_input_path(
            input_path,
            on_file=lambda path: 0
            if ocr.ocr_image(path, args.output if args.output else None).get("success")
            else 1,
            on_dir=lambda path: _cmd_ocr_dir(
                ocr,
                path,
                output_dir=args.output if args.output else str(config.OCR_DIR),
                force=force,
                max_volumes=max_volumes,
                workers=workers,
                volume_overrides=volume_overrides,
            ),
        )
    except FileNotFoundError:
        print(f"错误：输入路径不存在或无效: {input_path}")
        return 1


def _cmd_ocr_dir(
    ocr,
    input_path: str,
    output_dir: str,
    force: bool,
    max_volumes: Optional[int],
    workers: int,
    volume_overrides: Optional[dict],
) -> int:
    success_count, total_count = _run_ocr_dir(
        ocr,
        input_path,
        output_dir,
        force,
        max_volumes,
        workers,
        volume_overrides,
    )
    return 0 if success_count == total_count else 1


def cmd_all(args: Namespace) -> int:
    ocr = _import_livetext()
    if ocr is None:
        return 1

    input_path = args.input
    force, max_volumes, workers, volume_overrides = _common_options(args)

    print("\n" + "=" * 60)
    print("步骤 1/2: 预处理")
    print("=" * 60)

    rc, preprocessed_input = _prepare_preprocessed_input_for_all(
        input_path=input_path,
        force=force,
        max_volumes=max_volumes,
        workers=workers,
        volume_overrides=volume_overrides,
    )
    if rc != 0 or not preprocessed_input:
        return 1

    print("\n" + "=" * 60)
    print("步骤 2/2: OCR 识别")
    print("=" * 60)

    if os.path.isfile(preprocessed_input):
        result = ocr.ocr_image(preprocessed_input)
        return 0 if result.get("success") else 1

    success_count, total_count = _run_ocr_dir(
        ocr,
        preprocessed_input,
        str(config.OCR_DIR),
        force,
        max_volumes,
        workers,
        volume_overrides,
    )
    return 0 if success_count == total_count else 1


def cmd_pdf2images(args: Namespace) -> int:
    pdf_converter = _import_pdf_converter()
    input_path = args.input
    dpi = int(getattr(args, "dpi", 300) or 300)
    output_dir = args.output if getattr(args, "output", None) else None

    if os.path.isfile(input_path):
        if not input_path.lower().endswith(".pdf"):
            print(f"错误：文件不是 PDF 格式: {input_path}")
            return 1
        try:
            image_paths = pdf_converter.pdf_to_images(input_path, output_dir=output_dir, dpi=dpi)
            return 0 if image_paths else 1
        except Exception as e:
            print(f"错误：PDF 转换失败: {e}")
            return 1

    if os.path.isdir(input_path):
        output_parent = output_dir if output_dir else input_path
        try:
            results = pdf_converter.convert_directory(
                input_path,
                output_parent_dir=output_parent,
                dpi=dpi,
            )
            success_count = sum(1 for paths in results.values() if paths)
            return 0 if success_count > 0 else 1
        except Exception as e:
            print(f"错误：批量转换失败: {e}")
            return 1

    print(f"错误：输入路径不存在或无效: {input_path}")
    return 1


def cmd_convert(args: Namespace) -> int:
    pdf_converter = _import_pdf_converter()
    input_dir = args.input
    dpi = int(getattr(args, "dpi", 300) or 300)
    force, max_volumes, workers, volume_overrides = _common_options(args)

    if not os.path.isdir(input_dir):
        print(f"错误：输入路径必须是目录: {input_dir}")
        return 1

    try:
        success_count, _ = pdf_converter.convert_books_directory(
            input_dir,
            dpi=dpi,
            max_volumes=max_volumes,
            force=force,
            workers=workers,
            volume_overrides=volume_overrides,
        )
        return 0 if success_count > 0 else 1
    except Exception as e:
        print(f"错误：批量转换失败: {e}")
        return 1


def cmd_match(args: Namespace) -> int:
    match_standard_chars = _import_match_standard_chars()
    ocr_dir = args.ocr_dir
    standard_chars_json = args.standard_chars if args.standard_chars else str(config.STANDARD_CHARS_JSON)
    output = args.output if args.output else None
    try:
        match_standard_chars.main(ocr_dir, standard_chars_json, output)
        return 0
    except Exception as e:
        print(f"错误：匹配失败: {e}")
        import traceback

        traceback.print_exc()
        return 1


def cmd_crop(args: Namespace) -> int:
    crop_characters = _import_crop_characters()
    matched_chars_json = args.matched_chars_json
    output_dir = args.output if args.output else "data/results/chars"
    padding = args.padding if args.padding else 5
    book_name = args.book if args.book else None
    try:
        crop_characters.main(matched_chars_json, output_dir, padding, book_name)
        return 0
    except Exception as e:
        print(f"错误：裁切失败: {e}")
        import traceback

        traceback.print_exc()
        return 1


def cmd_config(_: Namespace) -> int:
    summary = config.config_summary()

    print("\n" + "=" * 60)
    print("项目配置信息")
    print("=" * 60)

    print("\n路径配置:")
    for key, value in summary["paths"].items():
        print(f"  {key}: {value}")

    print("\n预处理配置:")
    for key, value in summary["preprocess"].items():
        print(f"  {key}: {value}")

    print("\nOCR 配置:")
    for key, value in summary["ocr"].items():
        print(f"  {key}: {value}")

    print("\nPaddle 配置:")
    for key, value in summary["paddle"].items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    return 0


def cmd_paddle(args: Namespace) -> int:
    paddle_core = _import_paddle_core()
    paddle_url = args.paddle_url or config.PADDLE_CONFIG.get("url")
    if not paddle_url:
        print("错误：缺少 PaddleOCR 服务地址")
        return 1

    books = args.books or paddle_core.list_books()
    if not books:
        print("未找到可处理的书籍")
        return 1

    paddle_core.run_paddle_pipeline(
        books=books,
        paddle_url=paddle_url,
        topk=args.topk,
        timeout=args.timeout,
        limit_chars=args.limit_chars,
        limit_instances=args.limit_instances,
        min_conf=args.min_conf,
        batch_size=args.batch_size,
        workers=args.workers,
        require_match=config.PADDLE_CONFIG.get("require_match", False),
    )
    return 0


CommandHandler = Callable[[Namespace], int]

COMMAND_HANDLERS: Dict[str, CommandHandler] = {
    "convert": cmd_convert,
    "pdf2images": cmd_pdf2images,
    "preprocess": cmd_preprocess,
    "ocr": cmd_ocr,
    "all": cmd_all,
    "match": cmd_match,
    "crop": cmd_crop,
    "config": cmd_config,
    "paddle": cmd_paddle,
}
