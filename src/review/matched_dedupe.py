"""Helpers for deduplicating logically identical matched OCR instances."""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Tuple, TypeVar

from src.review.identity import normalize_to_preprocessed_path


MATCHED_SCHEMA_VERSION = 4
MIN_AXIS_OVERLAP_RATIO = 0.8
MAX_AXIS_SIZE_RATIO = 1.6
T = TypeVar("T")


def _int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _bbox_tuple(bbox: Optional[Dict]) -> Tuple[int, int, int, int]:
    bbox = bbox or {}
    return (
        _int(bbox.get("x")),
        _int(bbox.get("y")),
        _int(bbox.get("width")),
        _int(bbox.get("height")),
    )


def matched_page_group_key(inst: Optional[Dict]) -> Tuple[str, int, str]:
    inst = inst or {}
    return (
        normalize_to_preprocessed_path(str(inst.get("source_image") or "")),
        _int(inst.get("volume")),
        str(inst.get("page") or ""),
    )


def matched_instance_group_key(inst: Optional[Dict]) -> Tuple[str, int, str, int, int, int, int]:
    inst = inst or {}
    x, y, width, height = _bbox_tuple(inst.get("bbox"))
    return (
        normalize_to_preprocessed_path(str(inst.get("source_image") or "")),
        _int(inst.get("volume")),
        str(inst.get("page") or ""),
        x,
        y,
        width,
        height,
    )


def matched_source_group_key(source: Optional[Dict]) -> Tuple[str, int, str, int, int, int, int]:
    return matched_instance_group_key(source)


def _axis_overlap_ratio(start_a: int, size_a: int, start_b: int, size_b: int) -> float:
    if size_a <= 0 or size_b <= 0:
        return 0.0
    end_a = start_a + size_a
    end_b = start_b + size_b
    overlap = min(end_a, end_b) - max(start_a, start_b)
    if overlap <= 0:
        return 0.0
    return float(overlap) / float(min(size_a, size_b))


def _axis_size_ratio(size_a: int, size_b: int) -> float:
    if size_a <= 0 or size_b <= 0:
        return float("inf")
    return float(max(size_a, size_b)) / float(min(size_a, size_b))


def matched_bboxes_nearly_overlap(
    left: Optional[Dict],
    right: Optional[Dict],
    min_axis_overlap_ratio: float = MIN_AXIS_OVERLAP_RATIO,
    max_axis_size_ratio: float = MAX_AXIS_SIZE_RATIO,
) -> bool:
    left = left or {}
    right = right or {}
    if matched_page_group_key(left) != matched_page_group_key(right):
        return False

    lx, ly, lw, lh = _bbox_tuple(left.get("bbox"))
    rx, ry, rw, rh = _bbox_tuple(right.get("bbox"))
    if min(lw, lh, rw, rh) <= 0:
        return False

    x_overlap_ratio = _axis_overlap_ratio(lx, lw, rx, rw)
    y_overlap_ratio = _axis_overlap_ratio(ly, lh, ry, rh)
    width_ratio = _axis_size_ratio(lw, rw)
    height_ratio = _axis_size_ratio(lh, rh)

    return (
        x_overlap_ratio >= float(min_axis_overlap_ratio)
        and y_overlap_ratio >= float(min_axis_overlap_ratio)
        and width_ratio <= float(max_axis_size_ratio)
        and height_ratio <= float(max_axis_size_ratio)
    )


def cluster_records_by_page_overlap(
    records: Iterable[T],
    source_getter: Callable[[T], Optional[Dict]],
) -> List[List[T]]:
    indexed_records = list(enumerate(records))
    page_groups: Dict[Tuple[str, int, str], List[Tuple[int, T, Dict]]] = {}
    ordered_page_keys: List[Tuple[str, int, str]] = []
    cluster_entries: List[Tuple[int, List[T]]] = []

    for order_idx, record in indexed_records:
        source = dict(source_getter(record) or {})
        page_key = matched_page_group_key(source)
        _, _, width, height = _bbox_tuple(source.get("bbox"))
        if not (page_key[0] or page_key[1] or page_key[2]) or width <= 0 or height <= 0:
            cluster_entries.append((order_idx, [record]))
            continue
        if page_key not in page_groups:
            ordered_page_keys.append(page_key)
            page_groups[page_key] = []
        page_groups[page_key].append((order_idx, record, source))

    for page_key in ordered_page_keys:
        page_records = page_groups[page_key]
        count = len(page_records)
        if count == 1:
            order_idx, record, _ = page_records[0]
            cluster_entries.append((order_idx, [record]))
            continue

        parents = list(range(count))

        def find(idx: int) -> int:
            while parents[idx] != idx:
                parents[idx] = parents[parents[idx]]
                idx = parents[idx]
            return idx

        def union(left_idx: int, right_idx: int) -> None:
            left_root = find(left_idx)
            right_root = find(right_idx)
            if left_root != right_root:
                parents[right_root] = left_root

        for left_idx in range(count):
            _, _, left_source = page_records[left_idx]
            for right_idx in range(left_idx + 1, count):
                _, _, right_source = page_records[right_idx]
                if matched_bboxes_nearly_overlap(left_source, right_source):
                    union(left_idx, right_idx)

        component_members: Dict[int, List[int]] = {}
        component_order: List[int] = []
        for member_idx in range(count):
            root = find(member_idx)
            if root not in component_members:
                component_members[root] = []
                component_order.append(root)
            component_members[root].append(member_idx)

        for root in component_order:
            member_indexes = component_members[root]
            cluster = [page_records[idx][1] for idx in member_indexes]
            cluster_order = min(page_records[idx][0] for idx in member_indexes)
            cluster_entries.append((cluster_order, cluster))

    cluster_entries.sort(key=lambda entry: entry[0])
    return [cluster for _, cluster in cluster_entries]


def matched_source_kind(inst: Optional[Dict]) -> str:
    inst = inst or {}
    canonical_source = str(inst.get("canonical_source") or "")
    if canonical_source in {"preprocessed", "raw", "other"}:
        return canonical_source
    source_image = str(inst.get("source_image") or "")
    ocr_file = str(inst.get("ocr_file") or "")
    if (
        "/preprocessed/" in source_image
        or "_preprocessed" in source_image
        or "_preprocessed_ocr.json" in ocr_file
    ):
        return "preprocessed"
    if source_image.startswith("data/raw/") or "/data/raw/" in source_image:
        return "raw"
    return "other"


def _is_preprocessed_instance(inst: Optional[Dict]) -> bool:
    return matched_source_kind(inst) == "preprocessed"


def matched_instance_preference(inst: Optional[Dict]) -> Tuple[int, int, float, int]:
    inst = inst or {}
    char_index = _int(inst.get("char_index"), default=10**9)
    return (
        1 if _is_preprocessed_instance(inst) else 0,
        1 if isinstance(inst.get("normalized_bbox"), dict) and inst.get("normalized_bbox") else 0,
        _float(inst.get("confidence")),
        -char_index,
    )


def _variant_identity(variant: Dict) -> Tuple[str, str, str, int]:
    return (
        str(variant.get("source_kind") or ""),
        str(variant.get("source_image") or ""),
        str(variant.get("ocr_file") or ""),
        _int(variant.get("char_index"), default=-1),
    )


def _variant_snapshot(inst: Optional[Dict]) -> Dict:
    inst = inst or {}
    return {
        "source_kind": matched_source_kind(inst),
        "source_image": str(inst.get("source_image") or ""),
        "normalized_source_image": normalize_to_preprocessed_path(str(inst.get("source_image") or "")),
        "ocr_file": str(inst.get("ocr_file") or ""),
        "char_index": inst.get("char_index"),
        "confidence": _float(inst.get("confidence")),
    }


def _expand_variants(inst: Optional[Dict]) -> List[Dict]:
    inst = inst or {}
    variants = inst.get("ocr_variants")
    if isinstance(variants, list) and variants:
        out: List[Dict] = []
        for variant in variants:
            if not isinstance(variant, dict):
                continue
            normalized_variant = {
                "source_kind": str(variant.get("source_kind") or matched_source_kind(variant)),
                "source_image": str(variant.get("source_image") or ""),
                "normalized_source_image": normalize_to_preprocessed_path(str(variant.get("source_image") or variant.get("normalized_source_image") or "")),
                "ocr_file": str(variant.get("ocr_file") or ""),
                "char_index": variant.get("char_index"),
                "confidence": _float(variant.get("confidence")),
            }
            out.append(normalized_variant)
        if out:
            return out
    return [_variant_snapshot(inst)]


def _annotate_canonical_instance(best_inst: Dict, grouped_instances: List[Dict]) -> Dict:
    canonical = dict(best_inst or {})
    canonical["source_image"] = normalize_to_preprocessed_path(str(canonical.get("source_image") or ""))

    variant_map: Dict[Tuple[str, str, str, int], Dict] = {}
    for inst in grouped_instances:
        for variant in _expand_variants(inst):
            key = _variant_identity(variant)
            if key not in variant_map:
                variant_map[key] = variant

    variants = list(variant_map.values())
    variants.sort(
        key=lambda variant: (
            1 if str(variant.get("source_kind") or "") == "preprocessed" else 0,
            _float(variant.get("confidence")),
            -_int(variant.get("char_index"), default=10**9),
        ),
        reverse=True,
    )

    source_kinds: List[str] = []
    for variant in variants:
        source_kind = str(variant.get("source_kind") or "other")
        if source_kind not in source_kinds:
            source_kinds.append(source_kind)

    canonical["canonical_source"] = matched_source_kind(best_inst)
    canonical["ocr_sources"] = source_kinds
    canonical["ocr_variant_count"] = len(variants)
    canonical["ocr_variants"] = variants
    canonical["matched_schema_version"] = MATCHED_SCHEMA_VERSION
    return canonical


def dedupe_matched_instances(instances: Optional[Iterable[Dict]]) -> List[Dict]:
    deduped: List[Dict] = []
    normalized_instances = [dict(inst) for inst in (instances or []) if isinstance(inst, dict)]
    for group in cluster_records_by_page_overlap(normalized_instances, lambda inst: inst):
        best_inst = max(group, key=matched_instance_preference)
        deduped.append(_annotate_canonical_instance(best_inst, group))
    return deduped


def dedupe_matched_book_data(book_data: Optional[Dict]) -> Optional[Dict]:
    if not isinstance(book_data, dict):
        return None

    chars = book_data.get("chars")
    if not isinstance(chars, dict):
        return dict(book_data)

    deduped_book = dict(book_data)
    deduped_chars: Dict[str, List[Dict]] = {}
    total_instances = 0
    total_ocr_variants = 0

    for char, instances in chars.items():
        if not isinstance(instances, list):
            continue
        deduped_instances = dedupe_matched_instances(instances)
        if not deduped_instances:
            continue
        deduped_chars[char] = deduped_instances
        total_instances += len(deduped_instances)
        total_ocr_variants += sum(_int(inst.get("ocr_variant_count"), default=1) for inst in deduped_instances)

    deduped_book["chars"] = deduped_chars
    deduped_book["total_standard_chars"] = len(deduped_chars)
    deduped_book["total_instances"] = total_instances
    deduped_book["total_ocr_variants"] = total_ocr_variants
    deduped_book["deduped_instances_removed"] = max(0, total_ocr_variants - total_instances)
    deduped_book["matched_schema_version"] = MATCHED_SCHEMA_VERSION
    return deduped_book
