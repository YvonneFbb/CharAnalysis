"""Shared runtime helpers for review pipeline CLI commands."""

from __future__ import annotations

import os
from argparse import Namespace
from pathlib import Path
from typing import Callable, Optional, Tuple, TypeVar

from src.review import config


def import_preprocess():
    from src.review.preprocess import core as preprocess

    return preprocess


def import_pdf_converter():
    from src.review.utils import pdf_converter

    return pdf_converter


def import_match_standard_chars():
    from src.review.filter import match_standard_chars

    return match_standard_chars


def import_crop_characters():
    from src.review.crop import crop_characters

    return crop_characters


def import_prepare_filter_stage():
    from src.review.filter import prepare_filter_stage

    return prepare_filter_stage


def import_segment_books_stage():
    from src.review.filter import segment_books_stage

    return segment_books_stage


def import_reocr_books_stage():
    from src.review.filter import reocr_books_stage

    return reocr_books_stage


def common_options(args: Namespace) -> Tuple[bool, Optional[int], int, Optional[dict]]:
    force = bool(getattr(args, "force", False))
    max_volumes = getattr(args, "max_volumes", None)
    workers = int(getattr(args, "workers", 1) or 1)
    volume_overrides = getattr(config, "VOLUME_OVERRIDES", None)
    return force, max_volumes, workers, volume_overrides


T = TypeVar("T")


def dispatch_input_path(
    input_path: str,
    on_file: Callable[[str], T],
    on_dir: Callable[[str], T],
) -> T:
    if os.path.isfile(input_path):
        return on_file(input_path)
    if os.path.isdir(input_path):
        return on_dir(input_path)
    raise FileNotFoundError(input_path)


def import_livetext():
    try:
        from src.review.ocr import livetext as ocr
    except Exception as exc:
        print(f"错误：OCR 模块不可用（可能仅支持 macOS LiveText）：{exc}")
        return None
    return ocr


def maybe_convert_pdfs(
    input_path: str,
    max_volumes: Optional[int],
    force: bool,
    workers: int,
    volume_overrides: Optional[dict],
) -> bool:
    pdf_converter = import_pdf_converter()
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
    except Exception as exc:
        print(f"错误：批量转换失败: {exc}")
        return False


def run_preprocess_dir(
    input_dir: str,
    output_dir: str,
    force: bool,
    max_volumes: Optional[int],
    workers: int,
    volume_overrides: Optional[dict],
) -> Optional[Tuple[int, int]]:
    preprocess = import_preprocess()
    if not maybe_convert_pdfs(input_dir, max_volumes, force, workers, volume_overrides):
        return None
    return preprocess.process_directory(
        input_dir,
        output_dir,
        force=force,
        max_volumes=max_volumes,
        workers=workers,
        volume_overrides=volume_overrides,
    )


def run_ocr_dir(
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


def prepare_preprocessed_input_for_all(
    input_path: str,
    force: bool,
    max_volumes: Optional[int],
    workers: int,
    volume_overrides: Optional[dict],
) -> Tuple[int, Optional[str]]:
    preprocess = import_preprocess()
    try:
        return dispatch_input_path(
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
    result = run_preprocess_dir(
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
