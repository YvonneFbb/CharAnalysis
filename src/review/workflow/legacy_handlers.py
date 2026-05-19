"""Compatibility and legacy command handlers for the review pipeline CLI."""

from __future__ import annotations

from argparse import Namespace

from src.review import config
from src.review.workflow.common import (
    common_options,
    import_crop_characters,
    import_livetext,
    import_prepare_filter_stage,
    import_reocr_books_stage,
    prepare_preprocessed_input_for_all,
    run_ocr_dir,
)


def cmd_all(args: Namespace) -> int:
    ocr = import_livetext()
    if ocr is None:
        return 1

    print("提示：`pipeline all` 是兼容入口，推荐显式运行 `preprocess` -> `ocr`。")
    input_path = args.input
    force, max_volumes, workers, volume_overrides = common_options(args)

    print("\n" + "=" * 60)
    print("步骤 1/2: 预处理")
    print("=" * 60)

    rc, preprocessed_input = prepare_preprocessed_input_for_all(
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

    import os

    if os.path.isfile(preprocessed_input):
        result = ocr.ocr_image(preprocessed_input)
        return 0 if result.get("success") else 1

    success_count, total_count = run_ocr_dir(
        ocr,
        preprocessed_input,
        str(config.OCR_DIR),
        force,
        max_volumes,
        workers,
        volume_overrides,
    )
    return 0 if success_count == total_count else 1


def cmd_prepare_filter(args: Namespace) -> int:
    prepare_filter_stage = import_prepare_filter_stage()
    try:
        print("提示：`pipeline prepare-filter` 是兼容入口，推荐显式运行 `segment` -> `reocr`。")
        return prepare_filter_stage.run_prepare_filter(
            books=args.books,
            chars=args.chars,
            limit_chars=args.limit_chars,
            limit_instances=args.limit_instances,
            workers=args.workers,
            force=bool(args.force),
            reocr_engine=args.reocr_engine,
            reocr_pad=args.reocr_pad,
            paddle_url=args.paddle_url,
            timeout=args.timeout,
            batch_size=args.batch_size,
        )
    except Exception as exc:
        print(f"错误：prepare-filter 失败: {exc}")
        import traceback

        traceback.print_exc()
        return 1


def cmd_crop(args: Namespace) -> int:
    crop_characters = import_crop_characters()
    matched_chars_json = args.matched_chars_json
    output_dir = args.output if args.output else "data/results/chars"
    padding = args.padding if args.padding else 5
    book_name = args.book if args.book else None
    try:
        crop_characters.main(matched_chars_json, output_dir, padding, book_name)
        return 0
    except Exception as exc:
        print(f"错误：裁切失败: {exc}")
        import traceback

        traceback.print_exc()
        return 1


def cmd_paddle(args: Namespace) -> int:
    print("提示：`pipeline paddle` 已转为新 reOCR 流程别名，旧 Paddle review 流程已 deprecated。")
    reocr_books_stage = import_reocr_books_stage()
    return reocr_books_stage.run_reocr_books(
        books=args.books,
        engine="paddle",
        chars=None,
        limit_chars=args.limit_chars,
        limit_instances=args.limit_instances,
        workers=args.workers,
        force=False,
        pad=4,
        paddle_url=args.paddle_url or config.PADDLE_CONFIG.get("url"),
        timeout=args.timeout,
        batch_size=args.batch_size,
    )


LEGACY_COMMAND_HANDLERS = {
    "all": cmd_all,
    "prepare-filter": cmd_prepare_filter,
    "crop": cmd_crop,
    "paddle": cmd_paddle,
}
