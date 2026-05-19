"""Primary stage-oriented command handlers for the review pipeline CLI."""

from __future__ import annotations

import os
from argparse import Namespace

from src.review import config
from src.review.workflow.common import (
    common_options,
    dispatch_input_path,
    import_livetext,
    import_match_standard_chars,
    import_pdf_converter,
    import_preprocess,
    import_reocr_books_stage,
    import_segment_books_stage,
    run_ocr_dir,
    run_preprocess_dir,
)


def cmd_preprocess(args: Namespace) -> int:
    preprocess = import_preprocess()
    input_path = args.input
    force, max_volumes, workers, volume_overrides = common_options(args)
    try:
        return dispatch_input_path(
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
    max_volumes: int | None,
    workers: int,
    volume_overrides: dict | None,
) -> int:
    result = run_preprocess_dir(
        input_path, output_dir, force, max_volumes, workers, volume_overrides
    )
    if result is None:
        return 1
    success_count, total_count = result
    return 0 if success_count == total_count else 1


def cmd_ocr(args: Namespace) -> int:
    ocr = import_livetext()
    if ocr is None:
        return 1

    input_path = args.input
    force, max_volumes, workers, volume_overrides = common_options(args)
    try:
        return dispatch_input_path(
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
    max_volumes: int | None,
    workers: int,
    volume_overrides: dict | None,
) -> int:
    success_count, total_count = run_ocr_dir(
        ocr,
        input_path,
        output_dir,
        force,
        max_volumes,
        workers,
        volume_overrides,
    )
    return 0 if success_count == total_count else 1


def cmd_pdf2images(args: Namespace) -> int:
    pdf_converter = import_pdf_converter()
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
        except Exception as exc:
            print(f"错误：PDF 转换失败: {exc}")
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
        except Exception as exc:
            print(f"错误：批量转换失败: {exc}")
            return 1

    print(f"错误：输入路径不存在或无效: {input_path}")
    return 1


def cmd_convert(args: Namespace) -> int:
    pdf_converter = import_pdf_converter()
    input_dir = args.input
    dpi = int(getattr(args, "dpi", 300) or 300)
    force, max_volumes, workers, volume_overrides = common_options(args)

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
    except Exception as exc:
        print(f"错误：批量转换失败: {exc}")
        return 1


def cmd_match(args: Namespace) -> int:
    match_standard_chars = import_match_standard_chars()
    ocr_dir = args.ocr_dir
    standard_chars_json = args.standard_chars if args.standard_chars else str(config.STANDARD_CHARS_JSON)
    output = args.output if args.output else None
    try:
        match_standard_chars.main(ocr_dir, standard_chars_json, output)
        return 0
    except Exception as exc:
        print(f"错误：匹配失败: {exc}")
        import traceback

        traceback.print_exc()
        return 1


def cmd_segment(args: Namespace) -> int:
    segment_books_stage = import_segment_books_stage()
    try:
        return segment_books_stage.run_segment_books(
            books=args.books,
            chars=args.chars,
            limit_chars=args.limit_chars,
            limit_instances=args.limit_instances,
            workers=args.workers,
            force=bool(args.force),
        )
    except Exception as exc:
        print(f"错误：segment 失败: {exc}")
        import traceback

        traceback.print_exc()
        return 1


def cmd_reocr(args: Namespace) -> int:
    reocr_books_stage = import_reocr_books_stage()
    try:
        return reocr_books_stage.run_reocr_books(
            books=args.books,
            engine=args.engine,
            chars=args.chars,
            limit_chars=args.limit_chars,
            limit_instances=args.limit_instances,
            workers=args.workers,
            force=bool(args.force),
            pad=args.pad,
            paddle_url=args.paddle_url,
            timeout=args.timeout,
            batch_size=args.batch_size,
        )
    except Exception as exc:
        print(f"错误：reocr 失败: {exc}")
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


PRIMARY_COMMAND_HANDLERS = {
    "convert": cmd_convert,
    "pdf2images": cmd_pdf2images,
    "preprocess": cmd_preprocess,
    "ocr": cmd_ocr,
    "match": cmd_match,
    "segment": cmd_segment,
    "reocr": cmd_reocr,
    "config": cmd_config,
}
