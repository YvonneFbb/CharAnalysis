"""
审查系统后端服务器

提供API：
1. 获取书籍列表和字符数据
2. 按需裁切字符实例图片
3. 返回裁切后的图片文件
4. 字符切割和审查
"""

from flask import Flask, jsonify, send_file, request, render_template, redirect, url_for
import logging
from flask_cors import CORS
import json
import os
from collections import OrderedDict
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
import io
import base64
import sys
import traceback
from datetime import datetime, timezone
import time
from typing import List, Optional, Dict, Tuple

# 添加项目根目录到 sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.review import config as review_config
from src.review.identity import (
    get_confirmed_path as _get_confirmed_path,
    normalize_to_preprocessed_path,
    set_confirmed_path as _set_confirmed_path,
)
from src.review.matched_dedupe import MATCHED_SCHEMA_VERSION, cluster_records_by_page_overlap, dedupe_matched_book_data
from src.review.storage.reocr_books import DEFAULT_REOCR_ENGINE, normalize_engine_name, read_reocr_book
from src.review.storage.segment_books import read_segment_book
from src.review.storage.paddle_books import (
    list_paddle_books,
    read_paddle_book,
    write_paddle_book,
)
from src.review.storage.review_books import (
    FILTER_REOCR_PAD_DEFAULT,
    ensure_char_item as _ensure_char_item,
    iter_accepted_items,
    list_review_books,
    make_empty_char_entry,
    read_all_review_books,
    read_review_book,
    review_book_lock_path,
    source_from_matched_instance as _source_from_matched_instance,
    utc_now_iso,
    write_review_book,
)

# 导入切割模块
from src.review.segment import segment_character, adjust_bbox, get_default_params

TEMPLATE_DIR = PROJECT_ROOT / 'src/review/web/templates'
STATIC_DIR = PROJECT_ROOT / 'src/review/web/static'

app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
    static_folder=str(STATIC_DIR)
)

# Basic debug logging
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)
CORS(app)  # 允许跨域请求

# Flask 配置：支持大数据传输
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB
app.config['JSON_AS_ASCII'] = False

# 全局配置
MATCHED_JSON_PATH = review_config.MATCHED_JSON_PATH
MATCHED_BOOKS_DIR = review_config.MATCHED_BOOKS_DIR
MATCHED_CACHE_DIR = review_config.MATCHED_CACHE_DIR
MATCHED_SHARDS_DIR = review_config.MATCHED_SHARDS_DIR
MATCHED_INDEX_PATH = review_config.MATCHED_INDEX_PATH
STANDARD_CHARS_JSON_PATH = review_config.STANDARD_CHARS_JSON
PREPROCESSED_DIR = review_config.PREPROCESSED_DIR
REVIEW_BOOKS_DIR = review_config.REVIEW_BOOKS_DIR
SEGMENT_BOOKS_DIR = review_config.SEGMENT_BOOKS_DIR
REOCR_BOOKS_DIR = review_config.REOCR_BOOKS_DIR

# 切割相关路径
SEGMENTATION_REVIEW_PATH = review_config.SEGMENTATION_REVIEW_PATH
CONFIRMED_DIR = review_config.CONFIRMED_DIR
PADDLE_CONFIG = review_config.PADDLE_CONFIG

# 标记功能路径
SEGMENT_MARKS_PATH = review_config.SEGMENT_MARKS_PATH

# 创建必要的目录
CONFIRMED_DIR.mkdir(parents=True, exist_ok=True)

# 加载数据（懒加载 / 分片缓存）
matched_data = None  # 旧的整库加载（尽量避免使用）
matched_books_cache: Dict[str, dict] = {}
matched_books_cache_mtime = 0.0
matched_index_cache: Optional[Dict] = None
segment_books_cache: Dict[str, Optional[Dict]] = {}
segment_books_cache_mtime = 0.0
reocr_books_cache: Dict[Tuple[str, str], Optional[Dict]] = {}
reocr_books_cache_mtime: Dict[str, float] = {}
standard_chars_data = None
segment_lookup_data = None  # 内存中的查找索引（随 review_books 分片保存同步写入）
_standard_char_order_map: Optional[Dict[str, int]] = None
MATCHED_INDEX_SCHEMA_VERSION = MATCHED_SCHEMA_VERSION
_atlas_image_cache: "OrderedDict[str, Image.Image]" = OrderedDict()
ATLAS_IMAGE_CACHE_SIZE = 8


def _ensure_cache_dirs():
    MATCHED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MATCHED_SHARDS_DIR.mkdir(parents=True, exist_ok=True)
    MATCHED_BOOKS_DIR.mkdir(parents=True, exist_ok=True)


def _extract_book_payload(payload: Optional[Dict], book_name: Optional[str] = None) -> Optional[Dict]:
    if not isinstance(payload, dict):
        return None
    if isinstance(payload.get('data'), dict):
        return payload.get('data')
    if isinstance(payload.get('chars'), dict):
        return payload
    if isinstance(payload.get('books'), dict) and book_name:
        return payload['books'].get(book_name)
    return None


def _matched_books_mtime() -> float:
    if MATCHED_BOOKS_DIR.exists():
        mtimes = [p.stat().st_mtime for p in MATCHED_BOOKS_DIR.glob('*.json')]
        if mtimes:
            return max(mtimes)
    if MATCHED_JSON_PATH.exists():
        return MATCHED_JSON_PATH.stat().st_mtime
    return 0.0


def _segment_books_mtime() -> float:
    if not SEGMENT_BOOKS_DIR.exists():
        return 0.0
    mtimes = [p.stat().st_mtime for p in SEGMENT_BOOKS_DIR.glob('*.json')]
    return max(mtimes) if mtimes else 0.0


def _reocr_books_mtime(engine: str) -> float:
    engine_dir = REOCR_BOOKS_DIR / normalize_engine_name(engine)
    if not engine_dir.exists():
        return 0.0
    mtimes = [p.stat().st_mtime for p in engine_dir.glob('*.json')]
    return max(mtimes) if mtimes else 0.0


def ensure_matched_index() -> Dict:
    """获取书籍索引与统计信息（从缓存读取，过期则重建）。"""
    global matched_index_cache
    _ensure_cache_dirs()
    src_mtime = _matched_books_mtime()
    if matched_index_cache is not None:
        return matched_index_cache
    # 尝试读缓存
    if MATCHED_INDEX_PATH.exists():
        try:
            with open(MATCHED_INDEX_PATH, 'r', encoding='utf-8') as f:
                idx = json.load(f)
            if (
                idx.get('source_mtime', 0) == src_mtime
                and int(idx.get('schema_version') or 0) == MATCHED_INDEX_SCHEMA_VERSION
            ):
                matched_index_cache = idx
                return matched_index_cache
        except Exception:
            pass
    # 重建索引（优先使用拆分后的 per-book 文件）
    idx_books = {}
    if MATCHED_BOOKS_DIR.exists() and list(MATCHED_BOOKS_DIR.glob('*.json')):
        print(f"[索引] 构建 matched_books 索引：{MATCHED_BOOKS_DIR}")
        for path in MATCHED_BOOKS_DIR.glob('*.json'):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    payload = json.load(f)
                book_name = payload.get('book') or path.stem
                book_data = dedupe_matched_book_data(_extract_book_payload(payload, book_name))
                if not isinstance(book_data, dict):
                    continue
                idx_books[book_name] = {
                    'total_standard_chars': book_data.get('total_standard_chars', 0),
                    'total_instances': book_data.get('total_instances', 0)
                }
            except Exception:
                continue
    elif MATCHED_JSON_PATH.exists():
        print(f"[索引] 构建 matched_by_book 索引（首次较慢）：{MATCHED_JSON_PATH}")
        with open(MATCHED_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        books = data.get('books', {})
        for name, b in books.items():
            book_data = dedupe_matched_book_data(b)
            if not isinstance(book_data, dict):
                continue
            idx_books[name] = {
                'total_standard_chars': book_data.get('total_standard_chars', 0),
                'total_instances': book_data.get('total_instances', 0)
            }

    matched_index_cache = {
        'books': idx_books,
        'source_mtime': src_mtime,
        'schema_version': MATCHED_INDEX_SCHEMA_VERSION,
    }
    with open(MATCHED_INDEX_PATH, 'w', encoding='utf-8') as f:
        json.dump(matched_index_cache, f, ensure_ascii=False, indent=2)
    return matched_index_cache


def _build_shards_once(full_data: Optional[Dict] = None):
    """将整库 matched_by_book.json 分片到 per-book 文件，便于后续按需加载。"""
    _ensure_cache_dirs()
    if full_data is None:
        print(f"[分片] 读取整库以分片（首次较慢）：{MATCHED_JSON_PATH}")
        with open(MATCHED_JSON_PATH, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
    books = full_data.get('books', {})
    for name, b in books.items():
        shard_path = MATCHED_SHARDS_DIR / f"{name}.json"
        try:
            with open(shard_path, 'w', encoding='utf-8') as sf:
                json.dump({ 'book': name, 'data': b }, sf, ensure_ascii=False, indent=2)
        except Exception:
            pass


def ensure_matched_book_data(book_name: str) -> Optional[Dict]:
    """按需加载某本书的匹配数据（优先读取分片，缺失则一次性分片）。"""
    global matched_books_cache_mtime
    if not book_name:
        return None
    _ensure_cache_dirs()
    src_mtime = _matched_books_mtime()
    if src_mtime and src_mtime != matched_books_cache_mtime:
        matched_books_cache.clear()
        matched_books_cache_mtime = src_mtime
    book_path = MATCHED_BOOKS_DIR / f"{book_name}.json"
    if book_path.exists():
        try:
            with open(book_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            data = dedupe_matched_book_data(_extract_book_payload(payload, book_name))
            if data:
                matched_books_cache[book_name] = data
                return data
        except Exception:
            pass
    shard_path = MATCHED_SHARDS_DIR / f"{book_name}.json"
    if book_name in matched_books_cache:
        return matched_books_cache[book_name]
    # 如果分片存在且不过期，直接加载
    shard_ok = False
    if shard_path.exists():
        try:
            shard_ok = (not src_mtime) or (shard_path.stat().st_mtime >= src_mtime)
        except Exception:
            shard_ok = False
    if shard_ok:
        try:
            with open(shard_path, 'r', encoding='utf-8') as f:
                shard = json.load(f)
            data = dedupe_matched_book_data(shard.get('data'))
            if data:
                matched_books_cache[book_name] = data
                return data
        except Exception:
            pass
    # 否则一次性分片（整库加载一次），后续快速
    if MATCHED_JSON_PATH.exists():
        with open(MATCHED_JSON_PATH, 'r', encoding='utf-8') as f:
            full = json.load(f)
        _build_shards_once(full)
        book = dedupe_matched_book_data((full.get('books') or {}).get(book_name))
        if book:
            matched_books_cache[book_name] = book
            return book
    return None


def ensure_segment_book_data(book_name: str) -> Optional[Dict]:
    global segment_books_cache_mtime

    current_mtime = _segment_books_mtime()
    if current_mtime != segment_books_cache_mtime:
        segment_books_cache.clear()
        segment_books_cache_mtime = current_mtime

    if book_name in segment_books_cache:
        return segment_books_cache[book_name]

    data = read_segment_book(book_name)
    segment_books_cache[book_name] = data
    return data


def ensure_reocr_book_data(book_name: str, engine: str = DEFAULT_REOCR_ENGINE) -> Optional[Dict]:
    engine_name = normalize_engine_name(engine)
    current_mtime = _reocr_books_mtime(engine_name)
    cached_mtime = reocr_books_cache_mtime.get(engine_name, 0.0)
    if current_mtime != cached_mtime:
        stale_keys = [key for key in reocr_books_cache if key[0] == engine_name]
        for stale_key in stale_keys:
            reocr_books_cache.pop(stale_key, None)
        reocr_books_cache_mtime[engine_name] = current_mtime

    cache_key = (engine_name, book_name)
    if cache_key in reocr_books_cache:
        return reocr_books_cache[cache_key]

    data = read_reocr_book(book_name, engine_name)
    reocr_books_cache[cache_key] = data
    return data


def _resolve_standard_chars_path() -> Path:
    return STANDARD_CHARS_JSON_PATH


def ensure_standard_chars_data():
    """确保 standard_chars_data 已加载（懒加载）"""
    global standard_chars_data
    if standard_chars_data is None:
        chars_path = _resolve_standard_chars_path()
        print(f"[懒加载] 加载标准字数据：{chars_path}")
        with open(chars_path, 'r', encoding='utf-8') as f:
            standard_chars_data = json.load(f)
        print(f"[懒加载] 标准字数据加载完成：{standard_chars_data['total_methods']} 个分类法")


def _invalidate_lookup_book(book_name: str) -> None:
    if segment_lookup_data is not None:
        segment_lookup_data.get('books', {}).pop(book_name, None)


def _source_from_lookup_entry(instance_id: str, entry: Optional[Dict]) -> Dict:
    entry = entry or {}
    bbox = entry.get('bbox') or {}
    canonical_source = entry.get('canonical_source')
    ocr_sources = list(entry.get('ocr_sources') or [])
    if not ocr_sources and canonical_source:
        ocr_sources = [canonical_source]
    ocr_variant_count = int(entry.get('ocr_variant_count') or 0)
    if ocr_variant_count <= 0:
        ocr_variant_count = max(1, len(ocr_sources)) if ocr_sources else 0
    return {
        'instance_id': instance_id,
        'index': entry.get('index'),
        'bbox': bbox,
        'source_image': normalize_to_preprocessed_path(entry.get('source_image', '')),
        'confidence': entry.get('confidence', 0.0),
        'volume': entry.get('volume'),
        'page': entry.get('page'),
        'char_index': entry.get('char_index'),
        'width': int(entry.get('width') or bbox.get('width') or 0),
        'height': int(entry.get('height') or bbox.get('height') or 0),
        'canonical_source': canonical_source,
        'ocr_sources': ocr_sources,
        'ocr_variant_count': ocr_variant_count,
        'ocr_variants': list(entry.get('ocr_variants') or []),
    }


def _matched_sources_for_char(book_name: str, char: str) -> List[Dict]:
    book_data = ensure_matched_book_data(book_name) or {}
    chars = book_data.get('chars') or {}
    instances = chars.get(char) or []
    out: List[Dict] = []
    for idx, inst in enumerate(instances):
        if not isinstance(inst, dict):
            continue
        out.append(_source_from_matched_instance(inst, index=idx))
    return out


def _find_source_for_instance(book_name: str, char: str, instance_id: str) -> Optional[Dict]:
    book_obj = read_review_book(book_name) or {}
    char_obj = book_obj.get(char) or {}
    item = (char_obj.get('items') or {}).get(instance_id)
    if isinstance(item, dict) and isinstance(item.get('source'), dict):
        source = dict(item.get('source') or {})
        source.setdefault('instance_id', instance_id)
        return source

    for source in _matched_sources_for_char(book_name, char):
        if source.get('instance_id') == instance_id:
            return source

    lookup_book = get_lookup_book(book_name) or {}
    lookup_entry = (lookup_book.get(char) or {}).get(instance_id)
    if isinstance(lookup_entry, dict):
        return _source_from_lookup_entry(instance_id, lookup_entry)
    return None


def _review_state_to_legacy_entry(item: Dict) -> Dict:
    review_state = dict((item or {}).get('review') or {})
    status = review_state.get('status')
    if status == 'confirmed':
        legacy_status = 'confirmed'
    elif status == 'dropped':
        legacy_status = 'dropped'
    else:
        legacy_status = 'unreviewed'
    decision = review_state.get('decision')
    if not decision:
        decision = 'drop' if legacy_status == 'dropped' else 'unknown'
    payload = {
        'status': legacy_status,
        'method': review_state.get('method'),
        'timestamp': review_state.get('timestamp'),
        'decision': decision,
    }
    return _set_confirmed_path(payload, _get_confirmed_path(review_state))


def _review_state_from_legacy_entry(entry: Dict) -> Dict:
    entry = dict(entry or {})
    status = entry.get('status')
    decision = (entry.get('decision') or '').lower()
    if status == 'confirmed':
        review_status = 'confirmed'
        if decision == 'drop':
            decision = 'need'
    elif status == 'dropped' or decision == 'drop':
        review_status = 'dropped'
        decision = 'drop'
    else:
        review_status = 'pending'
        if decision not in ('need', 'unknown'):
            decision = 'unknown'
    payload = {
        'status': review_status,
        'method': entry.get('method'),
        'timestamp': entry.get('timestamp') or utc_now_iso(),
        'decision': decision,
    }
    return _set_confirmed_path(payload, _get_confirmed_path(entry))


def _image_path_to_data_url(path: Optional[Path]) -> Optional[str]:
    if path is None or not path.exists():
        return None
    try:
        raw = path.read_bytes()
    except Exception:
        return None
    return 'data:image/png;base64,' + base64.b64encode(raw).decode('ascii')


def _load_cached_atlas_image(relpath: str) -> Optional[Image.Image]:
    if not relpath:
        return None
    atlas_path = PROJECT_ROOT / relpath
    if not atlas_path.exists():
        return None
    cache_key = str(atlas_path.resolve())
    cached = _atlas_image_cache.get(cache_key)
    if cached is not None:
        _atlas_image_cache.move_to_end(cache_key)
        return cached
    try:
        with Image.open(atlas_path) as img:
            atlas = img.convert('RGB').copy()
    except Exception:
        return None
    _atlas_image_cache[cache_key] = atlas
    _atlas_image_cache.move_to_end(cache_key)
    while len(_atlas_image_cache) > ATLAS_IMAGE_CACHE_SIZE:
        _atlas_image_cache.popitem(last=False)
    return atlas


def _atlas_crop_to_data_url(relpath: Optional[str], bbox: Optional[Dict]) -> Optional[str]:
    atlas = _load_cached_atlas_image(str(relpath or ''))
    if atlas is None:
        return None
    bbox = dict(bbox or {})
    x = int(bbox.get('x') or 0)
    y = int(bbox.get('y') or 0)
    width = int(bbox.get('width') or 0)
    height = int(bbox.get('height') or 0)
    if width <= 0 or height <= 0:
        return None
    try:
        cropped = atlas.crop((x, y, x + width, y + height))
        with io.BytesIO() as buffer:
            cropped.save(buffer, format='PNG')
            return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')
    except Exception:
        return None


def _resolve_filter_preview(
    review_state: Optional[Dict],
    segment_state: Optional[Dict],
    include_image: bool,
) -> Tuple[Optional[str], Optional[str]]:
    review_state = dict(review_state or {})
    segment_state = dict(segment_state or {})

    confirmed_rel = _get_confirmed_path(review_state)
    if confirmed_rel:
        confirmed_abs = PROJECT_ROOT / confirmed_rel
        if confirmed_abs.exists():
            return confirmed_rel, (_image_path_to_data_url(confirmed_abs) if include_image else None)
        return confirmed_rel, None

    atlas_relpath = segment_state.get('atlas_relpath')
    atlas_bbox = segment_state.get('atlas_bbox') or {}
    if atlas_relpath and int(atlas_bbox.get('width') or 0) > 0 and int(atlas_bbox.get('height') or 0) > 0:
        return str(atlas_relpath), (_atlas_crop_to_data_url(atlas_relpath, atlas_bbox) if include_image else None)
    return None, None


def _segment_sort_width(source: Optional[Dict], segment_state: Optional[Dict]) -> int:
    segment_state = dict(segment_state or {})
    width = int(segment_state.get('segmented_width') or 0)
    if width > 0:
        return width
    source = dict(source or {})
    return int(source.get('width') or 0)


def _segment_sort_height(source: Optional[Dict], segment_state: Optional[Dict]) -> int:
    segment_state = dict(segment_state or {})
    height = int(segment_state.get('segmented_height') or 0)
    if height > 0:
        return height
    source = dict(source or {})
    return int(source.get('height') or 0)


def _filter_payload_from_item(
    book_name: str,
    char: str,
    source: Optional[Dict],
    item: Optional[Dict],
    segment_item: Optional[Dict],
    reocr_item: Optional[Dict],
    include_image: bool = True,
    engine: str = DEFAULT_REOCR_ENGINE,
) -> Dict:
    source = dict(source or {})
    item = dict(item or {})
    filter_state = dict(item.get('filter') or {})
    review_state = dict(item.get('review') or {})
    segment_state = dict(segment_item or {})
    reocr_state_data = dict(reocr_item or {})

    if review_state.get('status') == 'dropped' or review_state.get('decision') == 'drop':
        return {}

    preview_rel, preview_image = _resolve_filter_preview(review_state, segment_state, include_image)

    error_message = reocr_state_data.get('error') or segment_state.get('error')
    reocr_state = str(reocr_state_data.get('state') or 'pending')
    if reocr_state not in {'ready', 'error', 'pending'}:
        reocr_state = 'pending'
    if segment_state.get('state') == 'error' and reocr_state == 'pending':
        reocr_state = 'error'

    segmented_width = int(segment_state.get('segmented_width') or 0)
    segmented_height = int(segment_state.get('segmented_height') or 0)
    original_width = int(source.get('width') or 0)
    original_height = int(source.get('height') or 0)

    payload = {
        'book': book_name,
        'char': char,
        'instance_id': source.get('instance_id'),
        'filter_status': filter_state.get('status', 'pending'),
        'review_status': review_state.get('status', 'pending'),
        'review_decision': review_state.get('decision', 'need'),
        'width': original_width,
        'height': original_height,
        'original_width': original_width,
        'original_height': original_height,
        'segmented_width': segmented_width,
        'segmented_height': segmented_height,
        'volume': source.get('volume'),
        'page': source.get('page'),
        'char_index': source.get('char_index'),
        'bbox': source.get('bbox') or {},
        'source_image': source.get('source_image'),
        'canonical_source': source.get('canonical_source'),
        'ocr_sources': list(source.get('ocr_sources') or []),
        'ocr_variant_count': int(source.get('ocr_variant_count') or 0),
        'preview_path': preview_rel,
        'preview_image': preview_image,
        'segment_state': segment_state.get('state') or 'pending',
        'reocr_text': reocr_state_data.get('text'),
        'reocr_confidence': reocr_state_data.get('confidence'),
        'reocr_matches': reocr_state_data.get('matches'),
        'reocr_state': reocr_state,
        'reocr_pad': int(reocr_state_data.get('pad') or FILTER_REOCR_PAD_DEFAULT),
        'reocr_engine': normalize_engine_name(engine),
    }
    if error_message:
        payload['error'] = error_message
    return payload


def _filter_item_visible(payload: Dict, include_mismatch: bool) -> bool:
    if not payload:
        return False
    if payload.get('filter_status') == 'accepted':
        return True
    if payload.get('reocr_matches') is True:
        return True
    if include_mismatch:
        return payload.get('reocr_state') in {'ready', 'error'}
    return False


def _filter_payload_priority(payload: Dict) -> Tuple[int, int]:
    """
    Default filter browsing should surface truly usable samples first.
    Keep accepted items visible, but do not let stale accepted/pending rows
    crowd out fresh reOCR matches on the first page.
    """
    if payload.get('reocr_matches') is True:
        return (0, 0)
    if payload.get('filter_status') == 'accepted':
        state = payload.get('reocr_state')
        if state == 'ready':
            return (1, 0)
        if state == 'error':
            return (2, 0)
        return (3, 0)
    if payload.get('reocr_state') == 'ready':
        return (4, 0)
    if payload.get('reocr_state') == 'error':
        return (5, 0)
    return (6, 0)


def _filter_candidate_preference(
    source: Optional[Dict],
    item: Optional[Dict],
    segment_item: Optional[Dict],
    reocr_item: Optional[Dict],
) -> Tuple[int, int, int, int, str]:
    source = dict(source or {})
    item = dict(item or {})
    filter_state = dict(item.get('filter') or {})
    review_state = dict(item.get('review') or {})
    segment_state = dict(segment_item or {})
    reocr_state = dict(reocr_item or {})

    if review_state.get('status') == 'confirmed' or _get_confirmed_path(review_state):
        state_score = 6
    elif review_state.get('status') == 'dropped' or review_state.get('decision') == 'drop':
        state_score = 5
    elif filter_state.get('status') == 'accepted':
        state_score = 4
    elif filter_state.get('status') == 'rejected':
        state_score = 3
    elif reocr_state.get('matches') is True:
        state_score = 2
    elif reocr_state.get('state') in {'ready', 'error'} or segment_state.get('state') in {'ready', 'error'}:
        state_score = 1
    else:
        state_score = 0

    return (
        state_score,
        _segment_sort_width(source, segment_state),
        -int(source.get('index') or 10**9),
        1 if segment_state.get('state') == 'ready' else 0,
        source.get('instance_id') or '',
    )


def _item_is_review_dropped(item: Optional[Dict]) -> bool:
    item = item or {}
    review_state = item.get('review') or {}
    return review_state.get('status') == 'dropped' or review_state.get('decision') == 'drop'


def _accepted_lookup_from_book(book_obj: Dict) -> Dict:
    out: Dict[str, Dict] = {}
    for char, instance_id, item in iter_accepted_items(book_obj):
        source = dict(item.get('source') or {})
        out.setdefault(char, {})[instance_id] = {
            'bbox': source.get('bbox', {}),
            'source_image': source.get('source_image'),
            'confidence': source.get('confidence'),
            'volume': source.get('volume'),
            'page': source.get('page'),
            'char_index': source.get('char_index'),
            'index': source.get('index'),
            'width': source.get('width'),
            'height': source.get('height'),
            'canonical_source': source.get('canonical_source'),
            'ocr_sources': list(source.get('ocr_sources') or []),
            'ocr_variant_count': int(source.get('ocr_variant_count') or 0),
            'ocr_variants': list(source.get('ocr_variants') or []),
        }
    return out


def _ensure_lookup_for_book(book_name: str) -> Dict:
    ensure_segment_lookup_data()
    book_obj = read_review_book(book_name) or {}
    out_book = _accepted_lookup_from_book(book_obj)
    segment_lookup_data['books'][book_name] = out_book
    return out_book


def ensure_segment_lookup_data():
    """初始化 accepted sample lookup 缓存，用于 review/fixing 快速访问。"""
    global segment_lookup_data
    if segment_lookup_data is None:
        segment_lookup_data = {'version': 3, 'books': {}}


def get_standard_char_order_map() -> Dict[str, int]:
    global _standard_char_order_map
    if _standard_char_order_map is not None:
        return _standard_char_order_map
    path = _resolve_standard_chars_path()
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        _standard_char_order_map = {}
        return _standard_char_order_map
    order = {}
    idx = 0
    methods = data.get('methods') or {}
    if isinstance(methods, list):
        for method in methods:
            chars = method.get('chars') if isinstance(method, dict) else None
            if not chars:
                continue
            for ch in chars:
                if ch not in order:
                    order[ch] = idx
                    idx += 1
    elif isinstance(methods, dict):
        for _, method in methods.items():
            chars = method.get('chars') if isinstance(method, dict) else None
            if not chars:
                continue
            for ch in chars:
                if ch not in order:
                    order[ch] = idx
                    idx += 1
    _standard_char_order_map = order
    return _standard_char_order_map

def get_lookup_book(book_name: str) -> Optional[Dict]:
    """返回某本书当前 filter 接受集合的 lookup 映射。"""
    ensure_segment_lookup_data()
    if not book_name:
        return None
    if book_name in segment_lookup_data['books']:
        return segment_lookup_data['books'][book_name]
    return _ensure_lookup_for_book(book_name)


# ==================== 切割审查状态：单书分片加锁工具（统一存于 review_books/*.json） ====================
def _review_book_lock_path(book_name: str) -> Path:
    return review_book_lock_path(book_name)


def _read_review_results() -> Dict:
    # 兼容旧逻辑：现在统一从分片读取
    return read_all_review_books()

def read_review_data() -> dict:
    """
    兼容接口：返回 review 阶段视图。
    只暴露 filter.accepted 的实例，结构保持旧 segmentation_review.json 兼容。
    """
    rr = read_all_review_books()
    out = {'version': 3, 'books': {}}
    books = rr.get('books') or {}
    for book, book_obj in books.items():
        if not isinstance(book_obj, dict):
            continue
        book_out: Dict[str, Dict] = {}
        for char, instance_id, item in iter_accepted_items(book_obj):
            book_out.setdefault(char, {})[instance_id] = _review_state_to_legacy_entry(item)
        if book_out:
            out['books'][book] = book_out
    return out


def build_combined_book(book_name: str) -> dict:
    """
    构建统一视图：以 filter.accepted 的实例为主，叠加 review 状态。
    返回结构：{ char: { inst_id: {selected, bbox, source_image, ...,
                                 status, decision, confirmed_path, method, timestamp} } }
    """
    book_obj = read_review_book(book_name) or {}
    combined: Dict[str, Dict] = {}
    for char, instance_id, item in iter_accepted_items(book_obj):
        source = dict(item.get('source') or {})
        review_entry = _review_state_to_legacy_entry(item)
        combined.setdefault(char, {})[instance_id] = {
            'selected': True,
            'bbox': source.get('bbox', {}),
            'source_image': source.get('source_image'),
            'confidence': source.get('confidence'),
            'volume': source.get('volume'),
            'page': source.get('page'),
            'char_index': source.get('char_index'),
            'index': source.get('index'),
            'width': source.get('width'),
            'height': source.get('height'),
            'status': review_entry.get('status', 'unreviewed'),
            'decision': review_entry.get('decision'),
            'confirmed_path': _get_confirmed_path(review_entry),
            'segmented_path': _get_confirmed_path(review_entry),
            'method': review_entry.get('method'),
            'timestamp': review_entry.get('timestamp'),
        }
    return combined

def write_review_data(seg_view: dict):
    """
    将 review 视图写回 review_books/*.json 的 items[*].review 字段。
    同时写出兼容视图 segmentation_review.json，便于旧脚本使用。
    """
    for book, chars in (seg_view.get('books') or {}).items():
        if not isinstance(chars, dict):
            continue
        book_obj = read_review_book(book) or {}
        for char, review_map in chars.items():
            if not isinstance(review_map, dict):
                continue
            for instance_id, entry in review_map.items():
                if not isinstance(entry, dict):
                    continue
                source = _find_source_for_instance(book, char, instance_id)
                item = _ensure_char_item(book_obj, char, instance_id, source=source)
                item['review'] = _review_state_from_legacy_entry(entry)
                filter_state = item.setdefault('filter', {})
                if filter_state.get('status') != 'accepted':
                    filter_state['status'] = 'accepted'
                    filter_state['timestamp'] = filter_state.get('timestamp') or utc_now_iso()
            char_obj = book_obj.setdefault(char, make_empty_char_entry())
            if isinstance(char_obj, dict):
                char_obj['updated_at'] = utc_now_iso()
        write_review_book(book, book_obj)
        _invalidate_lookup_book(book)

    # 同步输出派生视图 segmentation_review.json，保持兼容
    try:
        SEGMENTATION_REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp2 = SEGMENTATION_REVIEW_PATH.with_suffix(SEGMENTATION_REVIEW_PATH.suffix + f".tmp.{os.getpid()}-{int(time.time()*1000)}")
        with open(tmp2, 'w', encoding='utf-8') as f:
            json.dump(seg_view, f, ensure_ascii=False, indent=2)
        os.replace(tmp2, SEGMENTATION_REVIEW_PATH)
    except Exception:
        pass

def update_review_entry(book_name: str, char: str, instance_id: str, entry: dict):
    """对 review 状态进行加锁的读-改-写更新（存于 review_books/*.json 的 items[*].review）。"""
    entry = _set_confirmed_path(dict(entry), _get_confirmed_path(entry))
    try:
        import fcntl  # POSIX
    except Exception:
        fcntl = None

    lock_path = _review_book_lock_path(book_name)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, 'a+') as lock_fp:
        if fcntl:
            try:
                fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
            except Exception:
                pass

        book_obj = read_review_book(book_name) or {}
        source = _find_source_for_instance(book_name, char, instance_id)
        item = _ensure_char_item(book_obj, char, instance_id, source=source)
        item['review'] = _review_state_from_legacy_entry(entry)
        filter_state = item.setdefault('filter', {})
        if filter_state.get('status') != 'accepted':
            filter_state['status'] = 'accepted'
            filter_state['timestamp'] = filter_state.get('timestamp') or utc_now_iso()
        char_obj = book_obj.setdefault(char, make_empty_char_entry())
        if isinstance(char_obj, dict):
            char_obj['updated_at'] = utc_now_iso()
            book_obj[char] = char_obj
        write_review_book(book_name, book_obj)

        if fcntl:
            try:
                fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass

@app.route('/api/books', methods=['GET'])
def get_books():
    """获取所有书籍列表（用于第一轮审查界面）"""
    idx = ensure_matched_index()
    books_list = [
        {
            'name': name,
            'total_chars': data.get('total_standard_chars', 0),
            'total_instances': data.get('total_instances', 0)
        }
        for name, data in (idx.get('books') or {}).items()
    ]
    return jsonify({
        'success': True,
        'books': sorted(books_list, key=lambda x: x['name'])
    })


@app.route('/api/books_simple', methods=['GET'])
def get_books_simple():
    """获取书籍列表（轻量级，用于切割审查界面）"""
    books = [name for name in list_review_books() if get_lookup_book(name)]
    books_list = [{'name': name} for name in books]
    return jsonify({
        'success': True,
        'books': sorted(books_list, key=lambda x: x['name'])
    })


@app.route('/api/methods', methods=['GET'])
def get_methods():
    """获取84法分类"""
    ensure_standard_chars_data()  # 懒加载
    return jsonify({
        'success': True,
        'methods': standard_chars_data['methods']
    })


@app.route('/api/characters_structure', methods=['GET'])
def get_characters_structure():
    """获取84法分类的字符结构（用于切割审查界面）"""
    ensure_standard_chars_data()  # 懒加载
    # 将 methods 转换为字典格式，方便前端使用
    structure = {}
    for method in standard_chars_data['methods']:
        structure[method['name']] = method['chars']

    return jsonify({
        'success': True,
        'structure': structure
    })


@app.route('/api/book/<book_name>', methods=['GET'])
def get_book_chars(book_name):
    """获取指定书籍的字符列表（按84法分类）"""
    ensure_standard_chars_data()  # 懒加载
    book_data = ensure_matched_book_data(book_name)
    if not book_data:
        return jsonify({'success': False, 'error': '书籍不存在'}), 404
    book_chars = book_data['chars']

    # 按84法分类组织字符
    methods_with_chars = []
    for method in standard_chars_data['methods']:
        method_chars = []
        for char in method['chars']:
            if char in book_chars:
                method_chars.append({
                    'char': char,
                    'count': len(book_chars[char])
                })

        if method_chars:  # 只包含有字符的分类
            methods_with_chars.append({
                'id': method['id'],
                'name': method['name'],
                'description': method.get('description', ''),
                'chars': method_chars
            })

    return jsonify({
        'success': True,
        'book_name': book_name,
        'total_chars': book_data.get('total_standard_chars', 0),
        'total_instances': book_data.get('total_instances', 0),
        'methods': methods_with_chars
    })


@app.route('/api/filter/books', methods=['GET'])
def api_filter_books():
    return get_books()


@app.route('/api/filter/book/<book_name>', methods=['GET'])
def api_filter_book(book_name: str):
    """返回 filter 阶段按 84 法组织的字符列表与进度统计。"""
    ensure_standard_chars_data()
    book_data = ensure_matched_book_data(book_name)
    if not book_data:
        return jsonify({'success': False, 'error': '书籍不存在'}), 404

    matched_chars = book_data.get('chars') or {}
    review_book = read_review_book(book_name) or {}
    segment_book = ensure_segment_book_data(book_name) or {}
    engine_name = normalize_engine_name(request.args.get('engine') or DEFAULT_REOCR_ENGINE)
    reocr_book = ensure_reocr_book_data(book_name, engine_name) or {}

    methods_with_chars = []
    processed_chars = set()
    accepted_instances = 0
    rejected_instances = 0

    for char, char_entry in review_book.items():
        if not isinstance(char_entry, dict):
            continue
        items = char_entry.get('items') or {}
        char_processed = False
        for item in items.values():
            if not isinstance(item, dict):
                continue
            if _item_is_review_dropped(item):
                continue
            filter_state = item.get('filter') or {}
            status = filter_state.get('status', 'pending')
            if status != 'pending':
                char_processed = True
            if status == 'accepted':
                accepted_instances += 1
            elif status == 'rejected':
                rejected_instances += 1
        if char_processed:
            processed_chars.add(char)

    for method in standard_chars_data['methods']:
        method_chars = []
        for char in method['chars']:
            instances = matched_chars.get(char) or []
            if not instances:
                continue
            items = ((review_book.get(char) or {}).get('items') or {})
            segment_items = ((segment_book.get(char) or {}).get('items') or {})
            reocr_items = ((reocr_book.get(char) or {}).get('items') or {})
            processed = 0
            accepted = 0
            rejected = 0
            prepared = 0
            matched = 0
            failed = 0
            for item in items.values():
                if not isinstance(item, dict):
                    continue
                if _item_is_review_dropped(item):
                    continue
                filter_state = item.get('filter') or {}
                status = filter_state.get('status', 'pending')
                if status != 'pending':
                    processed += 1
                if status == 'accepted':
                    accepted += 1
                elif status == 'rejected':
                    rejected += 1
            for instance_id, segment_item in segment_items.items():
                review_item = items.get(instance_id)
                if _item_is_review_dropped(review_item):
                    continue
                segment_state = str((segment_item or {}).get('state') or 'pending')
                reocr_state = str(((reocr_items.get(instance_id) or {}).get('state')) or 'pending')
                if segment_state in {'ready', 'error'} or reocr_state in {'ready', 'error'}:
                    prepared += 1
                if (reocr_items.get(instance_id) or {}).get('matches') is True:
                    matched += 1
                elif segment_state == 'error' or reocr_state == 'error':
                    failed += 1
            method_chars.append({
                'char': char,
                'count': len(instances),
                'processed': processed,
                'accepted': accepted,
                'rejected': rejected,
                'prepared': prepared,
                'matched': matched,
                'failed': failed,
            })

        if method_chars:
            methods_with_chars.append({
                'id': method['id'],
                'name': method['name'],
                'description': method.get('description', ''),
                'chars': method_chars,
            })

    return jsonify({
        'success': True,
        'book_name': book_name,
        'total_chars': book_data.get('total_standard_chars', 0),
        'total_instances': book_data.get('total_instances', 0),
        'processed_chars': len(processed_chars),
        'accepted_instances': accepted_instances,
        'rejected_instances': rejected_instances,
        'methods': methods_with_chars,
    })


@app.route('/api/filter/items', methods=['GET'])
def api_filter_items():
    """按字符读取预计算 filter 候选，默认只返回 reOCR 匹配项与既有 accepted 项。"""
    try:
        book_name = request.args.get('book')
        char = request.args.get('char')
        page = max(1, int(request.args.get('page', '1') or '1'))
        page_size = max(1, min(100, int(request.args.get('page_size', '20') or '20')))
        sort_mode = (request.args.get('sort', 'width_desc') or 'width_desc').lower()
        include_mismatch = (request.args.get('include_mismatch', '0') or '0').lower() in ('1', 'true', 'yes')
        engine_name = normalize_engine_name(request.args.get('engine') or DEFAULT_REOCR_ENGINE)

        if not book_name or not char:
            return jsonify({'success': False, 'error': '缺少必要参数'}), 400

        review_book = read_review_book(book_name) or {}
        review_items = (((review_book.get(char) or {}).get('items')) or {})
        segment_book = ensure_segment_book_data(book_name) or {}
        segment_items = (((segment_book.get(char) or {}).get('items')) or {})
        reocr_book = ensure_reocr_book_data(book_name, engine_name) or {}
        reocr_items = (((reocr_book.get(char) or {}).get('items')) or {})

        candidates: List[Dict] = []

        def register_candidate(source: Optional[Dict], item: Optional[Dict], segment_item: Optional[Dict], reocr_item: Optional[Dict]) -> None:
            if not isinstance(source, dict):
                return
            candidates.append({
                'source': dict(source),
                'item': item if isinstance(item, dict) else None,
                'segment_item': segment_item if isinstance(segment_item, dict) else None,
                'reocr_item': reocr_item if isinstance(reocr_item, dict) else None,
            })

        for source in _matched_sources_for_char(book_name, char):
            instance_id = source.get('instance_id')
            register_candidate(
                source,
                review_items.get(instance_id) if instance_id else None,
                segment_items.get(instance_id) if instance_id else None,
                reocr_items.get(instance_id) if instance_id else None,
            )
        for item in review_items.values():
            if not isinstance(item, dict):
                continue
            stored_source = item.get('source')
            if isinstance(stored_source, dict):
                instance_id = stored_source.get('instance_id')
                register_candidate(
                    stored_source,
                    item,
                    segment_items.get(instance_id) if instance_id else None,
                    reocr_items.get(instance_id) if instance_id else None,
                )

        grouped_meta: List[Dict] = []
        for group in cluster_records_by_page_overlap(
            candidates,
            source_getter=lambda candidate: candidate.get('source'),
        ):
            best_candidate = max(
                group,
                key=lambda candidate: _filter_candidate_preference(
                    candidate.get('source'),
                    candidate.get('item'),
                    candidate.get('segment_item'),
                    candidate.get('reocr_item'),
                ),
            )
            grouped_meta.append({
                'source': best_candidate.get('source') or {},
                'item': best_candidate.get('item'),
                'segment_item': best_candidate.get('segment_item'),
                'reocr_item': best_candidate.get('reocr_item'),
            })
        if not grouped_meta:
            return jsonify({
                'success': True,
                'book': book_name,
                'char': char,
                'page': page,
                'page_size': page_size,
                'total_candidates': 0,
                'items': [],
                'has_more': False,
                'include_mismatch': include_mismatch,
                'sort': sort_mode,
            })

        if sort_mode == 'width_desc':
            grouped_meta.sort(key=lambda meta: (
                -_segment_sort_width(meta.get('source'), meta.get('segment_item')),
                -_segment_sort_height(meta.get('source'), meta.get('segment_item')),
                (meta['source'] or {}).get('instance_id') or '',
            ))
        else:
            grouped_meta.sort(key=lambda meta: (int((meta['source'] or {}).get('index') or 0), (meta['source'] or {}).get('instance_id') or ''))

        visible_meta: List[Dict] = []
        for meta in grouped_meta:
            source = meta['source']
            item = meta.get('item')
            payload = _filter_payload_from_item(
                book_name,
                char,
                source,
                item,
                meta.get('segment_item'),
                meta.get('reocr_item'),
                include_image=False,
                engine=engine_name,
            )
            if not payload:
                continue
            if _filter_item_visible(payload, include_mismatch):
                visible_meta.append({
                    'source': source,
                    'item': item,
                    'segment_item': meta.get('segment_item'),
                    'reocr_item': meta.get('reocr_item'),
                    'payload': payload,
                })

        if sort_mode == 'width_desc':
            visible_meta.sort(key=lambda meta: (
                _filter_payload_priority(meta['payload']),
                -_segment_sort_width(meta.get('source'), meta.get('segment_item')),
                -_segment_sort_height(meta.get('source'), meta.get('segment_item')),
                (meta['source'] or {}).get('instance_id') or '',
            ))
        else:
            visible_meta.sort(key=lambda meta: (
                _filter_payload_priority(meta['payload']),
                int((meta['source'] or {}).get('index') or 0),
                (meta['source'] or {}).get('instance_id') or '',
            ))

        start = (page - 1) * page_size

        page_meta = visible_meta[start:start + page_size]
        page_items = [
            _filter_payload_from_item(
                book_name,
                char,
                meta['source'],
                meta['item'],
                meta.get('segment_item'),
                meta.get('reocr_item'),
                include_image=True,
                engine=engine_name,
            )
            for meta in page_meta
        ]
        has_more = start + page_size < len(visible_meta)

        return jsonify({
            'success': True,
            'book': book_name,
            'char': char,
            'page': page,
            'page_size': page_size,
            'total_candidates': len(grouped_meta),
            'items': page_items,
            'has_more': has_more,
            'include_mismatch': include_mismatch,
            'sort': sort_mode,
            'engine': engine_name,
            'truncated_scan': False,
        })
    except Exception as e:
        print(f'❌ /api/filter/items 失败: {e}')
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/filter/decision', methods=['POST'])
def api_filter_decision():
    """更新 filter 阶段单个实例的接受状态。"""
    try:
        data = request.get_json() or {}
        book_name = data.get('book')
        char = data.get('char')
        instance_id = data.get('instance_id')
        status = (data.get('status') or '').lower()
        engine_name = normalize_engine_name(data.get('engine') or DEFAULT_REOCR_ENGINE)
        if status not in {'pending', 'accepted', 'rejected'}:
            return jsonify({'success': False, 'error': 'status 只支持 pending/accepted/rejected'}), 400
        if not all([book_name, char, instance_id]):
            return jsonify({'success': False, 'error': '缺少必要参数'}), 400

        try:
            import fcntl
        except Exception:
            fcntl = None

        source = _find_source_for_instance(book_name, char, instance_id)
        lock_path = _review_book_lock_path(book_name)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, 'a+') as lock_fp:
            if fcntl:
                try:
                    fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
                except Exception:
                    pass

            book_obj = read_review_book(book_name) or {}
            item = _ensure_char_item(book_obj, char, instance_id, source=source)
            filter_state = item.setdefault('filter', {})
            filter_state['status'] = status
            filter_state['timestamp'] = utc_now_iso()
            char_obj = book_obj.setdefault(char, make_empty_char_entry())
            if isinstance(char_obj, dict):
                char_obj['updated_at'] = utc_now_iso()
                book_obj[char] = char_obj
            write_review_book(book_name, book_obj)

            if fcntl:
                try:
                    fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass

        _invalidate_lookup_book(book_name)

        source = source or _find_source_for_instance(book_name, char, instance_id) or {'instance_id': instance_id}
        updated_book = read_review_book(book_name) or {}
        updated_item = ((((updated_book.get(char) or {}).get('items')) or {}).get(instance_id)) or {}
        segment_book = ensure_segment_book_data(book_name) or {}
        reocr_book = ensure_reocr_book_data(book_name, engine_name) or {}
        item_payload = _filter_payload_from_item(
            book_name,
            char,
            source,
            updated_item,
            (((segment_book.get(char) or {}).get('items')) or {}).get(instance_id),
            (((reocr_book.get(char) or {}).get('items')) or {}).get(instance_id),
            include_image=True,
            engine=engine_name,
        )
        return jsonify({'success': True, 'item': item_payload})
    except Exception as e:
        print(f'❌ /api/filter/decision 失败: {e}')
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/instances', methods=['GET'])
def get_instances():
    """获取字符的实例列表（支持分页）"""
    book_name = request.args.get('book')
    char = request.args.get('char')
    page = int(request.args.get('page', 1))
    page_size = int(request.args.get('page_size', 20))
    sort_mode = request.args.get('sort', 'default')

    if not book_name or not char:
        return jsonify({'success': False, 'error': '缺少参数'}), 400

    book_data = ensure_matched_book_data(book_name)
    if not book_data:
        return jsonify({'success': False, 'error': '书籍不存在'}), 404
    if char not in book_data['chars']:
        return jsonify({'success': False, 'error': '字符不存在'}), 404

    enumerated = list(enumerate(book_data['chars'][char]))
    if sort_mode == 'width_desc':
        try:
            enumerated.sort(key=lambda pair: pair[1].get('bbox', {}).get('width', 0), reverse=True)
        except Exception:
            pass
    elif sort_mode == 'paddle':
        paddle_book = read_paddle_book(book_name) or {}
        paddle_chars = paddle_book.get('chars') if isinstance(paddle_book, dict) else None
        paddle_entry = paddle_chars.get(char) if isinstance(paddle_chars, dict) else None
        paddle_list = []
        if isinstance(paddle_entry, dict):
            paddle_list = paddle_entry.get('order') or paddle_entry.get('top5') or paddle_entry.get('items') or []
        if isinstance(paddle_list, dict):
            paddle_list = list(paddle_list.keys())
        order_map = {}
        if isinstance(paddle_list, list):
            for idx, inst_id in enumerate(paddle_list):
                if isinstance(inst_id, str) and inst_id not in order_map:
                    order_map[inst_id] = idx
        if order_map:
            def _instance_id(inst: dict) -> Optional[str]:
                try:
                    volume = int(inst.get('volume'))
                    page = inst.get('page', '')
                    char_index = inst.get('char_index')
                    if char_index is None:
                        return None
                    page_suffix = page.split('_')[-1] if page else ''
                    return f"册{volume:02d}_page{page_suffix}_idx{char_index}"
                except Exception:
                    return None
            def _auto_key(pair):
                orig_idx, inst = pair
                inst_id = _instance_id(inst)
                if inst_id in order_map:
                    return (0, order_map[inst_id])
                return (1, orig_idx)
            enumerated.sort(key=_auto_key)
    total = len(enumerated)

    # 分页
    start = (page - 1) * page_size
    end = start + page_size
    page_instances = enumerated[start:end]

    payload = []
    for orig_idx, inst in page_instances:
        inst_payload = dict(inst)
        inst_payload['instance_index'] = orig_idx
        payload.append(inst_payload)

    return jsonify({
        'success': True,
        'book_name': book_name,
        'char': char,
        'total': total,
        'page': page,
        'page_size': page_size,
        'total_pages': (total + page_size - 1) // page_size,
        'instances': payload
    })


@app.route('/api/crop', methods=['POST'])
def crop_image():
    """裁切单个字符实例"""
    data = request.json
    book_name = data.get('book')
    char = data.get('char')
    instance_index = data.get('index')
    padding = data.get('padding', 5)

    if not all([book_name, char, instance_index is not None]):
        return jsonify({'success': False, 'error': '缺少参数'}), 400

    # 获取实例信息
    book_data = ensure_matched_book_data(book_name)
    try:
        instance = book_data['chars'][char][instance_index]
    except Exception:
        return jsonify({'success': False, 'error': f'实例不存在: book={book_name}, char={char}, index={instance_index}'}), 404

    # 使用实例数据中的 source_image 路径（更可靠）
    if 'source_image' in instance:
        source_image_path = PROJECT_ROOT / instance['source_image']
    else:
        # 备用方案：手动构建路径
        volume = instance['volume']
        page = instance['page']
        source_image_path = PREPROCESSED_DIR / book_name / f'册{volume:02d}_pages' / f'{page}_preprocessed.png'

    if not source_image_path.exists():
        return jsonify({'success': False, 'error': f'源图片不存在: {source_image_path}'}), 404

    # 裁切图片
    try:
        img = cv2.imread(str(source_image_path))
        if img is None:
            return jsonify({'success': False, 'error': '无法读取图片'}), 500

        bbox = instance['bbox']
        x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']

        # 添加padding
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img.shape[1], x + w + padding)
        y2 = min(img.shape[0], y + h + padding)

        cropped = img[y1:y2, x1:x2]

        # 转换为base64
        _, buffer = cv2.imencode('.png', cropped)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'success': True,
            'image': f'data:image/png;base64,{img_base64}',
            'size': {'width': cropped.shape[1], 'height': cropped.shape[0]}
        })

    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()}), 500

def validate_data_format(data):
    """严格校验数据格式"""
    import re

    if not data or not isinstance(data, dict):
        return False, '数据不是有效对象'

    if data.get('version') != 2:
        return False, f'不支持的数据版本: {data.get("version")}，期望版本 2'

    if 'books' not in data or not isinstance(data['books'], dict):
        return False, '缺少 books 字段'

    # UTC 时间戳格式（JavaScript 生成的 'Z' 后缀）
    utc_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$')

    for book_name, book_data in data['books'].items():
        if not isinstance(book_data, dict):
            return False, f'无效的书籍数据: {book_name}'

        for char, char_data in book_data.items():
            if not isinstance(char_data, dict):
                return False, f'无效的字符数据: {book_name} - {char}'

            if 'instances' not in char_data or not isinstance(char_data['instances'], dict):
                return False, f'缺少 instances 字段: {book_name} - {char}'

            if 'timestamp' not in char_data:
                return False, f'缺少 timestamp 字段: {book_name} - {char}'

            # 严格检查时间戳格式
            if not utc_pattern.match(char_data['timestamp']):
                return False, f'无效的时间戳格式: {book_name} - {char}: {char_data["timestamp"]} (必须是 UTC 格式，如 "2025-10-27T13:50:12.804Z")'

    return True, None


@app.route('/api/ping', methods=['GET'])
def api_ping():
    return jsonify({
        'success': True,
        'ts': datetime.now(timezone.utc).isoformat(),
        'pid': os.getpid()
    })


@app.route('/api/save_review', methods=['POST'])
def save_review():
    return jsonify({
        'success': False,
        'error': '旧版 /api/save_review 已停用，请使用新的 /api/filter/decision 接口。'
    }), 410


@app.route('/api/load_review', methods=['GET'])
def load_review():
    """兼容接口：返回 review 阶段视图。"""
    try:
        book_only = request.args.get('book')
        review_data = read_review_data()
        if book_only:
            book_chars = (review_data.get('books') or {}).get(book_only)
            if book_chars is None:
                return jsonify({'success': True, 'data': {'version': 3, 'books': {}}})
            return jsonify({'success': True, 'data': {'version': 3, 'books': {book_only: book_chars}}})
        return jsonify({
            'success': True,
            'data': review_data
        })

    except Exception as e:
        import traceback
        print(f'❌ 加载审查结果失败: {str(e)}')
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ==================== 字符切割相关 API ====================

@app.route('/api/segment_instances', methods=['POST'])
def api_segment_instances():
    """
    请求切割字符实例

    Request body:
    {
        "book": "01_1127_尚书正义",
        "char": "宣",
        "instance_id": "册02_page0017_idx156",
        "custom_params": {...}  // 可选
    }
    """
    try:
        # 查找该书的 lookup（惰性构建，避免整库加载）

        data = request.get_json()
        book_name = data.get('book')
        char = data.get('char')
        instance_id = data.get('instance_id')
        custom_params = data.get('custom_params')
        app.logger.info('[API] /segment_instances book=%s char=%s instance=%s custom=%s',
                        book_name, char, instance_id, 'yes' if custom_params else 'no')

        if not all([book_name, char, instance_id]):
            return jsonify({'success': False, 'error': '缺少必要参数'}), 400

        # 从派生的 lookup 结构直接获取实例信息（O(1) 查找）
        lookup_book = get_lookup_book(book_name)
        if not lookup_book:
            return jsonify({'success': False, 'error': f'书籍不存在: {book_name}'}), 404
        if char not in lookup_book:
            return jsonify({'success': False, 'error': f'字符不存在: {char}'}), 404

        lookup_char = lookup_book[char]
        if instance_id not in lookup_char:
            return jsonify({'success': False, 'error': f'实例不存在: {instance_id}（该实例未通过第一轮审查）'}), 404

        instance_info = lookup_char[instance_id]

        # 执行切割
        preprocessed_image = instance_info['source_image']
        bbox = instance_info['bbox']

        # 转换为绝对路径
        if not os.path.isabs(preprocessed_image):
            preprocessed_image = str(PROJECT_ROOT / preprocessed_image)

        roi_img, segmented_img, debug_img, metadata, processed_roi = segment_character(
            preprocessed_image,
            bbox,
            custom_params=custom_params
        )

        # 如果该实例已经有最终保存的切割图片，并且当前不是在应用自定义参数，
        # 则优先使用已保存的图片作为 segmented_img（只在普通查看时覆盖）
        entry_state = {}
        try:
            if not custom_params:
                review_data = read_review_data()
                saved = (
                    review_data.get('books', {})
                              .get(book_name, {})
                              .get(char, {})
                              .get(instance_id, {})
                )
                if isinstance(saved, dict):
                    entry_state = saved
                segmented_rel = _get_confirmed_path(saved) if isinstance(saved, dict) else None
                saved_status = saved.get('status') if isinstance(saved, dict) else None
                if segmented_rel:
                    # 转换为绝对路径
                    seg_abs = PROJECT_ROOT / segmented_rel
                    if seg_abs.exists() and saved_status == 'confirmed':
                        # 使用已保存图片替换 segmented_img
                        saved_img = cv2.imread(str(seg_abs), cv2.IMREAD_UNCHANGED)
                        if saved_img is not None:
                            segmented_img = saved_img
        except Exception as _e:
            # 安静降级，不影响正常流程
            pass

        # 将图片转为 base64
        def img_to_base64(img):
            _, buffer = cv2.imencode('.png', img)
            return base64.b64encode(buffer).decode('utf-8')

        app.logger.info('[API] /segment_instances done: roi=%sx%s seg=%sx%s', roi_img.shape[1], roi_img.shape[0], segmented_img.shape[1], segmented_img.shape[0])
        return jsonify({
            'success': True,
            'roi_image': f"data:image/png;base64,{img_to_base64(roi_img)}",
            'segmented_image': f"data:image/png;base64,{img_to_base64(segmented_img)}",
            'debug_image': f"data:image/png;base64,{img_to_base64(debug_img)}",
            'processed_roi': f"data:image/png;base64,{img_to_base64(processed_roi)}",  # 用于bbox预览
            'metadata': metadata,
            'review_entry': entry_state
        })

    except Exception as e:
        print(f'❌ 切割失败: {str(e)}')
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/save_segmentation', methods=['POST'])
def api_save_segmentation():
    """
    保存切割结果和审查状态（只保存最终切割图片）

    Request body:
    {
        "book": "01_1127_尚书正义",
        "char": "宣",
        "instance_id": "册02_page0017_idx156",
        "status": "confirmed",
        "method": "auto" / "manual_bbox" / "auto_custom_params" / "pending_manual",
        "segmented_image_base64": "..."  // base64 编码的切割结果图片
    }
    """
    try:
        data = request.get_json()
        book_name = data.get('book')
        char = data.get('char')
        instance_id = data.get('instance_id')
        status = data.get('status')
        method = data.get('method', 'auto')
        segmented_b64 = data.get('segmented_image_base64')
        decision = data.get('decision', 'need')
        app.logger.info('[API] /save_segmentation book=%s char=%s instance=%s status=%s method=%s b64_len=%s',
                        book_name, char, instance_id, status, method, (len(segmented_b64) if segmented_b64 else 0))

        if not all([book_name, char, instance_id, status]):
            return jsonify({'success': False, 'error': '缺少必要参数'}), 400

        # 只保存切割后的图片（扁平结构：不使用字符子文件夹）
        segmented_path = None
        if segmented_b64:
            book_dir = CONFIRMED_DIR / book_name
            book_dir.mkdir(parents=True, exist_ok=True)

            # 解码 base64
            img_data = base64.b64decode(segmented_b64.split(',')[-1])
            # 文件名格式：{char}_{instance_id}.png
            segmented_path = book_dir / f"{char}_{instance_id}.png"
            with open(segmented_path, 'wb') as f:
                f.write(img_data)

        # 更新审查状态（单文件加锁写入）
        from datetime import datetime, timezone
        rel_path = f"data/results/manual/confirmed/{book_name}/{char}_{instance_id}.png" if segmented_path else None
        update_review_entry(book_name, char, instance_id, _set_confirmed_path({
            'status': status,
            'method': method,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'decision': decision
        }, rel_path))

        app.logger.info('[API] /save_segmentation saved: %s', rel_path)
        return jsonify({'success': True})

    except Exception as e:
        print(f'❌ 保存切割结果失败: {str(e)}')
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/unconfirm_segmentation', methods=['POST'])
def api_unconfirm_segmentation():
    """
    取消已确认的切割结果（将状态恢复为未审查）

    Request body:
    {
        "book": "01_1127_尚书正义",
        "char": "意",
        "instance_id": "册02_page0031_idx501"
    }
    """
    try:
        data = request.get_json() or {}
        book_name = data.get('book')
        char = data.get('char')
        instance_id = data.get('instance_id')

        if not all([book_name, char, instance_id]):
            return jsonify({'success': False, 'error': '缺少必要参数'}), 400

        # 恢复为未审查（单文件加锁写入）
        from datetime import datetime, timezone
        update_review_entry(book_name, char, instance_id, _set_confirmed_path({
            'status': 'unreviewed',
            'method': None,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'decision': 'unknown'
        }, None))

        app.logger.info('[API] /unconfirm_segmentation done: %s/%s/%s', book_name, char, instance_id)
        return jsonify({'success': True})

    except Exception as e:
        print(f'❌ 取消确认失败: {str(e)}')
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/mark_segmentation_decision', methods=['POST'])
def api_mark_segmentation_decision():
    """
    标记某个实例在第二轮的决策：需要/不需要/未处理
    """
    try:
        data = request.get_json() or {}
        book_name = data.get('book')
        char = data.get('char')
        instance_id = data.get('instance_id')
        decision = (data.get('decision') or '').lower()

        if decision not in ('need', 'drop', 'unknown'):
            return jsonify({'success': False, 'error': 'decision 只支持 need/drop/unknown'}), 400
        if not all([book_name, char, instance_id]):
            return jsonify({'success': False, 'error': '缺少必要参数'}), 400

        review_data = read_review_data()
        entry = (
            review_data.get('books', {})
                      .get(book_name, {})
                      .get(char, {})
                      .get(instance_id, {})
        ) or {}

        status = entry.get('status', 'unreviewed')
        method = entry.get('method')
        segmented_path = _get_confirmed_path(entry)

        if decision == 'drop':
            status = 'dropped'
            method = None
            if segmented_path:
                seg_abs = PROJECT_ROOT / segmented_path
                try:
                    if seg_abs.exists():
                        seg_abs.unlink()
                except Exception:
                    pass
            segmented_path = None
        elif decision == 'need':
            if status == 'dropped':
                status = 'unreviewed'
        else:  # unknown
            status = 'unreviewed'

        update_review_entry(book_name, char, instance_id, _set_confirmed_path({
            'status': status,
            'method': method,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'decision': decision
        }, segmented_path))

        return jsonify({'success': True})

    except Exception as e:
        print(f'❌ 标记决策失败: {str(e)}')
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/adjust_bbox', methods=['POST'])
def api_adjust_bbox():
    """
    手动调整 bbox 重新裁切

    Request body:
    {
        "book": "01_1127_尚书正义",
        "char": "宣",
        "instance_id": "册02_page0017_idx156",
        "adjusted_bbox": {"x": 100, "y": 200, "width": 180, "height": 200}
    }
    """
    try:
        data = request.get_json()
        book_name = data.get('book')
        char = data.get('char')
        instance_id = data.get('instance_id')
        adjusted_bbox = data.get('adjusted_bbox')
        app.logger.info('[API] /adjust_bbox book=%s char=%s instance=%s adjusted=%s', book_name, char, instance_id, adjusted_bbox)

        if not all([book_name, char, instance_id, adjusted_bbox]):
            return jsonify({'success': False, 'error': '缺少必要参数'}), 400

        # 从派生的 lookup 结构获取实例信息（惰性按书加载）
        lookup_book = get_lookup_book(book_name)
        if not lookup_book:
            return jsonify({'success': False, 'error': '书籍不存在'}), 404
        if char not in lookup_book or instance_id not in lookup_book[char]:
            return jsonify({'success': False, 'error': '未找到实例信息'}), 404

        instance_info = lookup_book[char][instance_id]

        # 重新裁切
        preprocessed_image = instance_info['source_image']
        original_bbox = instance_info['bbox']

        # 转换为绝对路径
        if not os.path.isabs(preprocessed_image):
            preprocessed_image = str(PROJECT_ROOT / preprocessed_image)

        # 使用简单的裁切方式（不执行完整的 segmentation 流程）
        img = cv2.imread(preprocessed_image)
        if img is None:
            raise ValueError(f"无法加载图片: {preprocessed_image}")

        h, w = img.shape[:2]

        # 确保 bbox 在图像范围内
        x = max(0, min(w-1, adjusted_bbox['x']))
        y = max(0, min(h-1, adjusted_bbox['y']))
        x2 = max(x+1, min(w, adjusted_bbox['x'] + adjusted_bbox['width']))
        y2 = max(y+1, min(h, adjusted_bbox['y'] + adjusted_bbox['height']))

        segmented_img = img[y:y2, x:x2]

        # 转为 base64
        def img_to_base64(img):
            _, buffer = cv2.imencode('.png', img)
            return base64.b64encode(buffer).decode('utf-8')

        segmented_b64 = img_to_base64(segmented_img)

        # 为了保持一致性，返回原始ROI作为debug图和processed_roi
        # 用户可以看到调整后的切割范围
        roi_x0 = max(0, original_bbox['x'] - 10)
        roi_y0 = max(0, original_bbox['y'] - 10)
        roi_x1 = min(w, original_bbox['x'] + original_bbox['width'] + 10)
        roi_y1 = min(h, original_bbox['y'] + original_bbox['height'] + 10)
        roi_img = img[roi_y0:roi_y1, roi_x0:roi_x1]

        roi_b64 = img_to_base64(roi_img)

        metadata = {
            'original_bbox': original_bbox,
            'adjusted_bbox': adjusted_bbox,
            'method': 'manual_bbox'
        }

        app.logger.info('[API] /adjust_bbox done: seg=%sx%s', segmented_img.shape[1], segmented_img.shape[0])
        return jsonify({
            'success': True,
            'segmented_image': f"data:image/png;base64,{segmented_b64}",
            'debug_image': f"data:image/png;base64,{roi_b64}",
            'processed_roi': f"data:image/png;base64,{roi_b64}",
            'metadata': metadata
        })

    except Exception as e:
        print(f'❌ 调整 bbox 失败: {str(e)}')
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/get_default_params', methods=['GET'])
def api_get_default_params():
    """获取默认切割参数"""
    try:
        params = get_default_params()
        return jsonify({'success': True, 'params': params})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/segmentation_status', methods=['GET'])
def api_segmentation_status():
    """
    获取切割审查状态

    Query params: book (optional)
    """
    try:
        book_name = request.args.get('book')

        review_data = read_review_data()

        if book_name:
            # 只返回指定书籍的数据
            book_data = review_data.get('books', {}).get(book_name, {})
            return jsonify({'success': True, 'data': book_data})
        else:
            # 返回所有数据
            return jsonify({'success': True, 'data': review_data})

    except Exception as e:
        print(f'❌ 获取切割状态失败: {str(e)}')
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/segment_instance_ids', methods=['GET'])
def api_segment_instance_ids():
    """
    获取字符的所有 instance_id 列表（用于切割审查）

    Query params: book, char
    Returns: ["册02_page0017_idx156", "册03_page0042_idx89", ...]
    """
    try:
        book_name = request.args.get('book')
        char = request.args.get('char')

        if not book_name or not char:
            return jsonify({'success': False, 'error': '缺少必要参数'}), 400

        lookup_book = get_lookup_book(book_name)
        if not lookup_book:
            return jsonify({'success': False, 'error': f'书籍不存在: {book_name}'}), 404
        if char not in lookup_book:
            return jsonify({'success': False, 'error': f'字符不存在: {char}'}), 404

        # 返回该字符的所有 instance_id
        instance_ids = list(lookup_book[char].keys())

        return jsonify({
            'success': True,
            'book': book_name,
            'char': char,
            'instance_ids': instance_ids,
            'total': len(instance_ids)
        })

    except Exception as e:
        print(f'❌ 获取实例列表失败: {str(e)}')
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/segment_book_chars', methods=['GET'])
def api_segment_book_chars():
    """
    获取书籍的字符列表和实例统计（用于切割审查界面）

    Query params: book
    Returns: {
        "chars": {"宣": {"count": 25, "confirmed": 5, "pending": 2}, ...}
    }
    """
    try:
        book_name = request.args.get('book')

        if not book_name:
            return jsonify({'success': False, 'error': '缺少必要参数'}), 400

        combined_book = build_combined_book(book_name)
        if combined_book is None:
            return jsonify({'success': False, 'error': f'书籍不存在: {book_name}'}), 404

        # 统计每个字符的实例数和审查状态
        chars_info = {}
        for char, inst_map in combined_book.items():
            total_instances = len(inst_map)
            confirmed = sum(1 for s in inst_map.values() if s.get('status') == 'confirmed')
            dropped = sum(1 for s in inst_map.values() if s.get('decision') == 'drop')
            effective = max(0, total_instances - dropped)

            chars_info[char] = {
                'total': total_instances,
                'effective': effective,
                'count': effective,
                'confirmed': confirmed,
                'not_needed': dropped
            }

        return jsonify({
            'success': True,
            'book': book_name,
            'chars': chars_info
        })

    except Exception as e:
        print(f'❌ 获取书籍字符列表失败: {str(e)}')
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== 标记功能 API ====================

@app.route('/api/mark_instance', methods=['POST'])
def api_mark_instance():
    """
    标记/取消标记实例

    Request body:
    {
        "instance_id": "册02_page0017_idx156",
        "marked": true/false,
        "book": "01_1127_尚书正义",
        "char": "宣"
    }
    """
    try:
        data = request.get_json()
        instance_id = data.get('instance_id')
        marked = data.get('marked', False)
        book = data.get('book')
        char = data.get('char')

        if not all([instance_id, book, char]):
            return jsonify({'success': False, 'error': '缺少必要参数'}), 400

        # 加载现有标记数据
        if SEGMENT_MARKS_PATH.exists():
            with open(SEGMENT_MARKS_PATH, 'r', encoding='utf-8') as f:
                marks_data = json.load(f)
        else:
            marks_data = {
                'version': 1,
                'last_updated': '',
                'marks': {}
            }

        # 更新标记数据
        instance_key = f"{book}_{char}_{instance_id}"
        if marked:
            marks_data['marks'][instance_key] = {
                'book': book,
                'char': char,
                'instance_id': instance_id,
                'marked_at': datetime.now(timezone.utc).isoformat(),
                'note': ''
            }
        else:
            marks_data['marks'].pop(instance_key, None)

        # 保存数据
        marks_data['last_updated'] = datetime.now(timezone.utc).isoformat()
        with open(SEGMENT_MARKS_PATH, 'w', encoding='utf-8') as f:
            json.dump(marks_data, f, ensure_ascii=False, indent=2)

        return jsonify({
            'success': True,
            'marked': marked,
            'instance_key': instance_key
        })

    except Exception as e:
        print(f'❌ 标记实例失败: {str(e)}')
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/marked_instances', methods=['GET'])
def api_marked_instances():
    """获取所有标记的实例"""
    try:
        if SEGMENT_MARKS_PATH.exists():
            with open(SEGMENT_MARKS_PATH, 'r', encoding='utf-8') as f:
                marks_data = json.load(f)

            return jsonify({
                'success': True,
                'marked_instances': marks_data.get('marks', {}),
                'last_updated': marks_data.get('last_updated', ''),
                'total_marked': len(marks_data.get('marks', {}))
            })
        else:
            return jsonify({
                'success': True,
                'marked_instances': {},
                'last_updated': '',
                'total_marked': 0
            })

    except Exception as e:
        print(f'❌ 获取标记实例失败: {str(e)}')
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/export_marked', methods=['GET'])
def api_export_marked():
    """导出所有标记实例的详细信息"""
    try:
        if not SEGMENT_MARKS_PATH.exists():
            return jsonify({
                'success': False,
                'error': '没有标记数据'
            }), 404

        with open(SEGMENT_MARKS_PATH, 'r', encoding='utf-8') as f:
            marks_data = json.load(f)

        marks = marks_data.get('marks', {})

        if not marks:
            return jsonify({
                'success': False,
                'error': '没有标记数据'
            }), 404

        # 加载派生的 lookup 数据获取详细信息
        ensure_segment_lookup_data()

        detailed_marks = {}
        for instance_key, mark_info in marks.items():
            book = mark_info['book']
            char = mark_info['char']
            instance_id = mark_info['instance_id']

            # 获取实例详细信息
            if (book in segment_lookup_data.get('books', {}) and
                char in segment_lookup_data['books'][book] and
                instance_id in segment_lookup_data['books'][book][char]):

                instance_data = segment_lookup_data['books'][book][char][instance_id]

                detailed_marks[instance_key] = {
                    'book': book,
                    'char': char,
                    'instance_id': instance_id,
                    'marked_at': mark_info['marked_at'],
                    'source_image': instance_data['source_image'],
                    'bbox': instance_data['bbox'],
                    'confidence': instance_data.get('confidence', 0.0),
                    'volume': instance_data.get('volume', 0),
                    'page': instance_data.get('page', ''),
                    'char_index': instance_data.get('char_index', 0)
                }
            else:
                # 如果找不到详细信息，保留基本信息
                detailed_marks[instance_key] = mark_info

        return jsonify({
            'success': True,
            'marked_instances': detailed_marks,
            'total_marked': len(detailed_marks),
            'exported_at': datetime.now(timezone.utc).isoformat()
        })

    except Exception as e:
        print(f'❌ 导出标记实例失败: {str(e)}')
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== 审查界面路由 ====================

# ==================== Fixing（问题集中处理）只读 API ====================

def _iter_confirmed_entries(review: Dict, book_name: str) -> List[Tuple[str, str, str]]:
    """返回该书所有已确认的 (char, instance_id, confirmed_rel_path)。"""
    out: List[Tuple[str, str, str]] = []
    books = (review or {}).get('books', {})
    book_obj = books.get(book_name, {}) if isinstance(books, dict) else {}
    for ch, inst_map in book_obj.items():
        if not isinstance(inst_map, dict):
            continue
        for inst_id, entry in inst_map.items():
            if not isinstance(entry, dict):
                continue
            if entry.get('status') != 'confirmed':
                continue
            seg_rel = _get_confirmed_path(entry)
            if not seg_rel:
                continue
            out.append((ch, inst_id, seg_rel))
    # 固定顺序：按字符、实例
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def _compute_book_fixed_box_Lb(book_name: str, review: Dict) -> int:
    """按书籍计算固定框 L_b：P90(长边) × 1.05，向上取整。无数据则返回 0。"""
    try:
        import numpy as _np
        long_sides: List[float] = []
        for _, _, seg_rel in _iter_confirmed_entries(review, book_name):
            abs_path = PROJECT_ROOT / seg_rel
            if not abs_path.exists():
                continue
            try:
                with Image.open(abs_path) as im:
                    w, h = im.size
                long_sides.append(float(max(w, h)))
            except Exception:
                continue
        if not long_sides:
            return 0
        p90 = float(_np.percentile(_np.array(long_sides, dtype=float), 90))
        from math import ceil
        return int(ceil(p90 * 1.05))
    except Exception:
        return 0


def _center_crop_fixed_box(gray_img: Image.Image, Lb: int) -> Image.Image:
    """以图像中心裁取 Lb×Lb，越界补白（灰度 255）。"""
    if gray_img.mode != 'L':
        gray = gray_img.convert('L')
    else:
        gray = gray_img
    if Lb <= 0:
        return gray.copy()
    w, h = gray.size
    cx, cy = w // 2, h // 2
    half = Lb // 2
    x0, y0 = cx - half, cy - half
    x1, y1 = x0 + Lb, y0 + Lb
    out = Image.new('L', (Lb, Lb), 255)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x1), min(h, y1)
    if sx0 < sx1 and sy0 < sy1:
        sub = gray.crop((sx0, sy0, sx1, sy1))
        dx, dy = sx0 - x0, sy0 - y0
        out.paste(sub, (dx, dy))
    return out


def _compute_metrics_for_entry(abs_path: Path, Lb: int, bw_threshold: int = 128) -> Dict:
    """
    计算：
    - geom_face_ratio = ((w+h)/2) / Lb
    - gray stats：黑像素比例（black_ratio in [0,1]）
    返回 {width, height, long_side, geom_face_ratio, black_ratio}
    """
    try:
        with Image.open(abs_path) as im:
            if im.mode != 'L':
                gray = im.convert('L')
            else:
                gray = im.copy()
            w, h = gray.size
            long_side = max(w, h)
            geom_face_ratio = None
            if Lb and Lb > 0:
                geom_face_ratio = ((w + h) / 2.0) / float(Lb)

            # 灰度比例：在固定框窗口内统计
            crop = _center_crop_fixed_box(gray, Lb) if (Lb and Lb > 0) else gray
            arr = np.array(crop, dtype=np.uint8)
            black = int((arr <= bw_threshold).sum())
            total = int(arr.size)
            black_ratio = (black / total) if total > 0 else 0.0
            return {
                'width': w,
                'height': h,
                'long_side': long_side,
                'geom_face_ratio': (round(geom_face_ratio, 4) if geom_face_ratio is not None else None),
                'black_ratio': round(black_ratio, 4)
            }
    except Exception:
        return {
            'width': None,
            'height': None,
            'long_side': None,
            'geom_face_ratio': None,
            'black_ratio': None
        }

def _crop_roi_image(inst_info: Dict) -> Optional[Image.Image]:
    """根据 source_image + bbox 裁切 ROI，用于尚未切割的实例缩略图。"""
    try:
        src = inst_info.get('source_image')
        bbox = inst_info.get('bbox')
        if not src or not bbox:
            return None
        src_path = Path(src)
        if not src_path.is_absolute():
            src_path = PROJECT_ROOT / src_path
        if not src_path.exists():
            return None
        with Image.open(src_path) as im:
            gray = im.convert('L')
            x = max(0, int(bbox.get('x', 0)))
            y = max(0, int(bbox.get('y', 0)))
            w = max(1, int(bbox.get('width', 1)))
            h = max(1, int(bbox.get('height', 1)))
            x2 = min(gray.width, x + w)
            y2 = min(gray.height, y + h)
            if x2 <= x or y2 <= y:
                return None
            return gray.crop((x, y, x2, y2))
    except Exception:
        return None

def _render_thumb_base64(image: Image.Image, tile: int) -> Optional[str]:
    """将任意 PIL Image 缩放到固定大小并返回 base64 数据。"""
    if image is None:
        return None
    try:
        base = image
        canvas = Image.new('RGB', (tile, tile), (255, 255, 255))
        w, h = base.size
        scale = min((tile - 4) / max(1, w), (tile - 4) / max(1, h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = base.resize((new_w, new_h), Image.BICUBIC)
        ox = (tile - new_w) // 2
        oy = (tile - new_h) // 2
        canvas.paste(resized.convert('RGB'), (ox, oy))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([0, 0, tile - 1, tile - 1], outline=(210, 210, 210))
        buf = io.BytesIO()
        canvas.save(buf, format='PNG')
        return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception:
        return None


@app.route('/api/fixing_items', methods=['GET'])
def api_fixing_items():
    """
    返回 Fixing 视图数据（只读）：
    - 参数：book（必填），page（默认1），page_size（默认50），image=thumb|none（默认thumb）
    - 计算 L_b 并为每条返回几何字面率与灰度比例
    - 可选返回小缩略图（最长边 200px）
    """
    try:
        book_name = request.args.get('book')
        page = int(request.args.get('page', '1') or '1')
        page_size = int(request.args.get('page_size', '50') or '50')
        image_mode = (request.args.get('image', 'thumb') or 'thumb').lower()
        try:
            bw_threshold = int(request.args.get('threshold', '128') or '128')
        except Exception:
            bw_threshold = 128
        use_fixed_box = (request.args.get('use_fixed_box', '1') or '1').lower() in ('1', 'true', 'yes')
        try:
            tile = int(request.args.get('tile', '96') or '96')
        except Exception:
            tile = 96
        all_mode = (request.args.get('all', '0') or '0').lower() in ('1', 'true', 'yes')
        if not book_name:
            return jsonify({'success': False, 'error': '缺少参数 book'}), 400

        lookup_book = get_lookup_book(book_name)
        if lookup_book is None:
            lookup_book = {}
        review_all = read_review_data()
        review_book = (review_all.get('books') or {}).get(book_name, {})
        Lb = _compute_book_fixed_box_Lb(book_name, review_all)
        ensure_standard_chars_data()
        all_chars = standard_chars_data.get('all_chars') if isinstance(standard_chars_data, dict) else None
        if not isinstance(all_chars, list):
            order_map = get_standard_char_order_map()
            all_chars = [ch for ch, _ in sorted(order_map.items(), key=lambda kv: kv[1])]
        present_chars = {ch for ch, inst_map in lookup_book.items() if inst_map}
        missing_chars = [ch for ch in all_chars if ch not in present_chars]
        paddle_counts = {}
        paddle_book = read_paddle_book(book_name)
        paddle_chars = paddle_book.get('chars') if isinstance(paddle_book, dict) else None
        if isinstance(paddle_chars, dict):
            for ch, ch_obj in paddle_chars.items():
                if not isinstance(ch_obj, dict):
                    continue
                items = ch_obj.get('top5')
                if not isinstance(items, list):
                    items = ch_obj.get('items')
                count = len(items) if isinstance(items, list) else 0
                if count:
                    paddle_counts[ch] = count

        all_instances = []
        for ch, inst_map in lookup_book.items():
            for inst_id, info in inst_map.items():
                entry = (review_book.get(ch) or {}).get(inst_id, {})
                seg_rel = _get_confirmed_path(entry)
                status = entry.get('status', 'unreviewed')
                method = entry.get('method')
                metrics = {}
                abs_path = PROJECT_ROOT / seg_rel if seg_rel else None
                if seg_rel and abs_path.exists():
                    metrics = _compute_metrics_for_entry(abs_path, Lb, bw_threshold=bw_threshold)
                elif not seg_rel:
                    metrics = {'width': None, 'height': None, 'long_side': None,
                               'geom_face_ratio': None, 'black_ratio': None}
                all_instances.append({
                    'book': book_name,
                    'char': ch,
                    'instance_id': inst_id,
                    'confirmed_path': seg_rel,
                    'segmented_path': seg_rel,
                    'status': status,
                    'method': method,
                    'decision': entry.get('decision', 'unknown'),
                    'info': info,
                    'metrics': metrics
                })

        total = len(all_instances)
        page = max(1, page)
        if all_mode:
            start = 0
            end = total
            page_size = total if total > 0 else 1
        else:
            page_size = max(1, min(200, page_size))
            start = (page - 1) * page_size
            end = min(total, start + page_size)

        items = []
        for item_data in all_instances[start:end]:
            seg_rel = item_data['segmented_path']
            abs_path = PROJECT_ROOT / seg_rel if seg_rel else None
            item = {
                'book': item_data['book'],
                'char': item_data['char'],
                'instance_id': item_data['instance_id'],
                'confirmed_path': seg_rel,
                'segmented_path': seg_rel,
                'status': item_data['status'],
                'method': item_data['method'],
                'decision': item_data.get('decision', 'unknown'),
                'metrics': item_data['metrics']
            }
            if image_mode == 'thumb':
                thumb_img = None
                if abs_path and abs_path.exists():
                    try:
                        with Image.open(abs_path) as im:
                            thumb_img = im.convert('L')
                    except Exception:
                        thumb_img = None
                if thumb_img is None:
                    thumb_img = _crop_roi_image(item_data['info'])
                if thumb_img is not None:
                    if use_fixed_box and Lb and Lb > 0:
                        thumb_source = _center_crop_fixed_box(thumb_img, Lb)
                    else:
                        thumb_source = thumb_img
                    thumb_b64 = _render_thumb_base64(thumb_source, tile)
                    if thumb_b64:
                        item['thumb'] = thumb_b64
            items.append(item)

        return jsonify({
            'success': True,
            'book': book_name,
            'fixed_box': {'L_b': Lb},
            'page': page,
            'page_size': page_size,
            'total': total,
            'items': items,
            'missing_chars': missing_chars,
            'missing_total': len(missing_chars),
            'paddle_counts': paddle_counts
        })
    except Exception as e:
        print(f'❌ /api/fixing_items 失败: {e}')
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/fixing_montage', methods=['GET'])
def api_fixing_montage():
    """返回该书的拼贴图 PNG（二进制）。参数：book，use_fixed_box(0/1)，tile，cols。"""
    try:
        from math import ceil
        book_name = request.args.get('book')
        use_fixed_box = request.args.get('use_fixed_box', '1') in ('1', 'true', 'yes')
        tile = int(request.args.get('tile', '64') or '64')
        cols = int(request.args.get('cols', '50') or '50')
        if not book_name:
            return jsonify({'success': False, 'error': '缺少参数 book'}), 400

        review = read_review_data()
        entries = _iter_confirmed_entries(review, book_name)
        Lb = _compute_book_fixed_box_Lb(book_name, review) if use_fixed_box else 0

        tiles: List[Image.Image] = []
        for _, _, seg_rel in entries:
            abs_path = PROJECT_ROOT / seg_rel
            if not abs_path.exists():
                continue
            try:
                with Image.open(abs_path) as im:
                    img = im.convert('L')
                if use_fixed_box and Lb and Lb > 0:
                    roi = _center_crop_fixed_box(img, Lb)
                    # 缩放到 tile
                    w, h = roi.size
                    scale = min((tile - 2) / max(1, w), (tile - 2) / max(1, h))
                    new_w = max(1, int(round(w * scale)))
                    new_h = max(1, int(round(h * scale)))
                    roi_resized = roi.resize((new_w, new_h), Image.BICUBIC)
                    canvas = Image.new('RGB', (tile, tile), (255, 255, 255))
                    ox = (tile - new_w) // 2
                    oy = (tile - new_h) // 2
                    canvas.paste(roi_resized.convert('RGB'), (ox, oy))
                else:
                    # 普通缩放
                    w, h = img.size
                    scale = min((tile - 2) / max(1, w), (tile - 2) / max(1, h))
                    new_w = max(1, int(round(w * scale)))
                    new_h = max(1, int(round(h * scale)))
                    resized = img.resize((new_w, new_h), Image.BICUBIC)
                    canvas = Image.new('RGB', (tile, tile), (255, 255, 255))
                    ox = (tile - new_w) // 2
                    oy = (tile - new_h) // 2
                    canvas.paste(resized.convert('RGB'), (ox, oy))
                # 淡边框
                draw = ImageDraw.Draw(canvas)
                draw.rectangle([0, 0, tile - 1, tile - 1], outline=(210, 210, 210))
                tiles.append(canvas)
            except Exception:
                continue

        if not tiles:
            return jsonify({'success': False, 'error': '无可用实例'}), 404

        cols = max(1, cols)
        rows = ceil(len(tiles) / cols)
        W = cols * tile
        H = rows * tile
        out = Image.new('RGB', (W, H), (255, 255, 255))
        for idx, timg in enumerate(tiles):
            r = idx // cols
            c = idx % cols
            out.paste(timg, (c * tile, r * tile))

        buf = io.BytesIO()
        out.save(buf, format='PNG')
        buf.seek(0)
        return send_file(buf, mimetype='image/png', as_attachment=False, download_name=f'{book_name}.png')

    except Exception as e:
        print(f'❌ /api/fixing_montage 失败: {e}')
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/paddle/books', methods=['GET'])
def api_paddle_books():
    try:
        books = []
        for name in list_paddle_books():
            data = read_paddle_book(name) or {}
            chars = data.get('chars') or {}
            books.append({
                'name': name,
                'chars': len(chars)
            })
        return jsonify({'success': True, 'books': books})
    except Exception as e:
        print(f'❌ /api/paddle/books 失败: {e}')
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/paddle/char_list', methods=['GET'])
def api_paddle_char_list():
    try:
        book_name = request.args.get('book')
        if not book_name:
            return jsonify({'success': False, 'error': '缺少参数 book'}), 400
        data = read_paddle_book(book_name) or {}
        chars = data.get('chars') or {}
        order_map = get_standard_char_order_map()
        items = []
        for ch, ch_data in chars.items():
            item_map = (ch_data or {}).get('items') or {}
            decisions = [v.get('decision', 'pending') for v in item_map.values() if isinstance(v, dict)]
            items.append({
                'char': ch,
                'total': len(item_map),
                'need': sum(1 for d in decisions if d == 'need'),
                'drop': sum(1 for d in decisions if d == 'drop'),
                'pending': sum(1 for d in decisions if d not in ('need', 'drop')),
            })
        items.sort(key=lambda x: (order_map.get(x['char'], 10**9), x['char']))
        return jsonify({'success': True, 'book': book_name, 'chars': items})
    except Exception as e:
        print(f'❌ /api/paddle/char_list 失败: {e}')
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/paddle/items', methods=['GET'])
def api_paddle_items():
    try:
        book_name = request.args.get('book')
        char = request.args.get('char')
        include_image = (request.args.get('image', '1') or '1').lower() in ('1', 'true', 'yes')
        if not book_name or not char:
            return jsonify({'success': False, 'error': '缺少参数 book/char'}), 400
        data = read_paddle_book(book_name) or {}
        chars = data.get('chars') or {}
        ch_data = chars.get(char) or {}
        items = ch_data.get('items') or {}
        order = ch_data.get('top5') or list(items.keys())
        out = []
        for inst_id in order:
            item = items.get(inst_id)
            if not isinstance(item, dict):
                continue
            row = dict(item)
            if include_image:
                seg_rel = row.get('confirmed_path') or row.get('segmented_path')
                if seg_rel:
                    seg_abs = PROJECT_ROOT / seg_rel
                    if seg_abs.exists():
                        try:
                            with open(seg_abs, 'rb') as f:
                                row['image'] = 'data:image/png;base64,' + base64.b64encode(f.read()).decode('utf-8')
                        except Exception:
                            pass
            out.append(row)
        return jsonify({
            'success': True,
            'book': book_name,
            'char': char,
            'items': out
        })
    except Exception as e:
        print(f'❌ /api/paddle/items 失败: {e}')
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/paddle/decision', methods=['POST'])
def api_paddle_decision():
    try:
        data = request.get_json() or {}
        book = data.get('book')
        char = data.get('char')
        instance_id = data.get('instance_id')
        decision = data.get('decision')
        if not all([book, char, instance_id]):
            return jsonify({'success': False, 'error': '缺少参数'}), 400
        if decision not in ('need', 'drop', 'pending', None, ''):
            return jsonify({'success': False, 'error': 'decision 非法'}), 400
        payload = read_paddle_book(book) or {}
        chars = payload.get('chars') or {}
        ch_data = chars.get(char)
        if not isinstance(ch_data, dict):
            return jsonify({'success': False, 'error': '找不到字符'}), 404
        items = ch_data.get('items') or {}
        item = items.get(instance_id)
        if not isinstance(item, dict):
            return jsonify({'success': False, 'error': '找不到实例'}), 404
        item['decision'] = decision or 'pending'
        items[instance_id] = item
        ch_data['items'] = items
        chars[char] = ch_data
        payload['chars'] = chars
        ok = write_paddle_book(book, payload)
        if not ok:
            return jsonify({'success': False, 'error': '写入失败，请稍后重试'}), 500
        return jsonify({'success': True})
    except Exception as e:
        print(f'❌ /api/paddle/decision 失败: {e}')
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/paddle_review')
def paddle_review_page():
    """Deprecated：保留旧的 Paddle 复核入口。"""
    try:
        return render_template('paddle/paddle_review.html')
    except Exception:
        return 'Paddle Review 页面未生成', 404


@app.route('/fixing')
def fixing_page():
    """兼容旧入口，重定向到 /review。"""
    return redirect(url_for('review_page'))

@app.route('/')
def index():
    """首页：审查系统入口（优先模板，其次兼容旧文件）"""
    try:
        # 优先使用模板
        return render_template('index.html')
    except Exception:
        # 回退到旧的静态文件
        html_path = PROJECT_ROOT / 'data/results/index.html'
        if html_path.exists():
            with open(html_path, 'r', encoding='utf-8') as f:
                return f.read()
        return '入口页面未生成', 404


@app.route('/filter')
def filter_page():
    """第一阶段 filter 界面（OCR + segment + reOCR）。"""
    # 优先模板
    tpl_path = TEMPLATE_DIR / 'manual/ocr_review.html'
    if tpl_path.exists():
        return render_template('manual/ocr_review.html')
    # 兼容：优先新的磁盘文件名，其次旧文件名
    new_path = PROJECT_ROOT / 'data/results/ocr_review.html'
    if new_path.exists():
        with open(new_path, 'r', encoding='utf-8') as f:
            return f.read()
    old_path = PROJECT_ROOT / 'data/results/review_app.html'
    if old_path.exists():
        with open(old_path, 'r', encoding='utf-8') as f:
            return f.read()
    return 'Filter 页面未生成', 404

@app.route('/ocr_review')
def ocr_review_page():
    """兼容旧入口，重定向到 /filter。"""
    return redirect(url_for('filter_page'))


@app.route('/review')
def review_page():
    """第二阶段 review 界面（总览预览 + 问题修正）。"""
    tpl_path = TEMPLATE_DIR / 'manual/fixing.html'
    if tpl_path.exists():
        return render_template('manual/fixing.html')
    html_path = PROJECT_ROOT / 'data/results/fixing.html'
    if html_path.exists():
        with open(html_path, 'r', encoding='utf-8') as f:
            return f.read()
    return 'Review 页面未生成', 404


@app.route('/segment_review')
def segment_review_page():
    """保留旧的单字精修页入口，默认不再在主 UI 中暴露。"""
    tpl_path = TEMPLATE_DIR / 'manual/segment_review_app.html'
    if tpl_path.exists():
        return render_template('manual/segment_review_app.html')
    html_path = PROJECT_ROOT / 'data/results/segment_review_app.html'
    if html_path.exists():
        with open(html_path, 'r', encoding='utf-8') as f:
            return f.read()
    return 'Segment Review 页面未生成', 404


def main():
    """启动服务器"""
    # 使用懒加载机制，启动时不加载数据，只在访问相关页面时才加载
    import socket

    def _collect_lan_ips() -> list:
        ips = set()
        try:
            for info in socket.getaddrinfo(socket.gethostname(), None):
                ip = info[4][0]
                if '.' in ip:
                    ips.add(ip)
        except Exception:
            pass
        try:
            for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
                if '.' in ip:
                    ips.add(ip)
        except Exception:
            pass
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ips.add(s.getsockname()[0])
            s.close()
        except Exception:
            pass
        def _bad(ip: str) -> bool:
            return ip.startswith("127.") or ip.startswith("169.254.") or ip.startswith("0.") or ip.startswith("255.") or ip.startswith("198.18.")
        return sorted([ip for ip in ips if not _bad(ip)])

    # 获取本机局域网 IP（兼容多网卡 / VPN）
    local_ips = _collect_lan_ips()
    local_ip = local_ips[0] if local_ips else "无法获取"

    print("\n" + "=" * 70)
    print("古籍字形审查系统 - 后端服务器")
    print("=" * 70)
    print(f"数据文件：{MATCHED_JSON_PATH}")
    print(f"标准字文件：{_resolve_standard_chars_path()}")
    print(f"图片目录：{PREPROCESSED_DIR}")
    print(f"确认结果目录：{CONFIRMED_DIR}")
    print("查找索引：内存派生自 review_books 分片（不再依赖独立文件）")

    print("\n服务器启动中...")
    print("=" * 70)
    print("\n访问地址：")
    print("  【系统首页】选择审查模式")
    print(f"    本机访问：http://localhost:5001/")
    print(f"    局域网访问：http://{local_ip}:5001/")
    if len(local_ips) > 1:
        print(f"    其他可选地址：{', '.join(local_ips[1:])}")
    print("\n  【Filter】OCR + segment + reOCR 快速筛选")
    print(f"    本机访问：http://localhost:5001/filter")
    print(f"    局域网访问：http://{local_ip}:5001/filter")
    print("\n  【Review】总览预览与问题修正")
    print(f"    本机访问：http://localhost:5001/review")
    print(f"    局域网访问：http://{local_ip}:5001/review")
    print("\n  【Paddle 复核（Deprecated）】保留旧入口")
    print(f"    本机访问：http://localhost:5001/paddle_review")
    print(f"    局域网访问：http://{local_ip}:5001/paddle_review")
    print("\n按 Ctrl+C 停止服务器")
    print("=" * 70 + "\n")

    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False, threaded=False)


if __name__ == '__main__':
    main()
