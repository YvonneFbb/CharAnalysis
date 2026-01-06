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
import shutil
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
from typing import List, Tuple, Optional, Dict

# 添加项目根目录到 sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
RESULTS_DIR = PROJECT_ROOT / 'data/results'
MANUAL_RESULTS_DIR = RESULTS_DIR / 'manual'
PADDLE_RESULTS_DIR = RESULTS_DIR / 'paddle'

MATCHED_JSON_PATH = RESULTS_DIR / 'matched_by_book.json'
MATCHED_CACHE_DIR = RESULTS_DIR / '_cache'
MATCHED_SHARDS_DIR = MATCHED_CACHE_DIR / 'matched_by_book_shards'
MATCHED_INDEX_PATH = MATCHED_CACHE_DIR / 'matched_books_index.json'
STANDARD_CHARS_JSON_PATH = PROJECT_ROOT / 'data/metadata/standard_chars.json'
STANDARD_CHARS_FALLBACK_PATH = PROJECT_ROOT / 'src/standard_chars.json'
PREPROCESSED_DIR = RESULTS_DIR / 'preprocessed'
REVIEW_RESULTS_PATH = MANUAL_RESULTS_DIR / 'review_results.json'
REVIEW_BOOKS_DIR = MANUAL_RESULTS_DIR / 'review_books'
REVIEW_BOOK_BACKUP_KEEP = 5
REVIEW_BOOK_BACKUP_COOLDOWN = 60  # 秒，避免高频写入产生大量备份

# 切割相关路径
SEGMENTATION_REVIEW_PATH = MANUAL_RESULTS_DIR / 'segmentation_review.json'
SEGMENT_LOOKUP_PATH = MANUAL_RESULTS_DIR / 'segment_lookup.json'
SEGMENTED_DIR = MANUAL_RESULTS_DIR / 'segmented'
MANUAL_WORK_DIR = SEGMENTED_DIR / 'manual'  # 手动处理工作区（统一到 segmented 下，扁平：按书籍，不按字符）

# 标记功能路径
SEGMENT_MARKS_PATH = MANUAL_RESULTS_DIR / 'segment_marks.json'

# Paddle 筛选路径
PADDLE_REVIEW_BOOKS_DIR = PADDLE_RESULTS_DIR / 'review_books'
PADDLE_SEGMENTED_DIR = PADDLE_RESULTS_DIR / 'segmented'

# 创建必要的目录
SEGMENTED_DIR.mkdir(parents=True, exist_ok=True)
MANUAL_WORK_DIR.mkdir(parents=True, exist_ok=True)

# 加载数据（懒加载 / 分片缓存）
matched_data = None  # 旧的整库加载（尽量避免使用）
matched_books_cache: Dict[str, dict] = {}
matched_index_cache: Optional[Dict] = None
standard_chars_data = None
segment_lookup_data = None  # 内存中的查找索引（随 review_results.json 保存同步写入）
_standard_char_order_map: Optional[Dict[str, int]] = None


def _ensure_cache_dirs():
    MATCHED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MATCHED_SHARDS_DIR.mkdir(parents=True, exist_ok=True)


def ensure_matched_index() -> Dict:
    """获取书籍索引与统计信息（从缓存读取，过期则重建）。"""
    global matched_index_cache
    _ensure_cache_dirs()
    src_mtime = MATCHED_JSON_PATH.stat().st_mtime if MATCHED_JSON_PATH.exists() else 0
    if matched_index_cache is not None:
        return matched_index_cache
    # 尝试读缓存
    if MATCHED_INDEX_PATH.exists():
        try:
            with open(MATCHED_INDEX_PATH, 'r', encoding='utf-8') as f:
                idx = json.load(f)
            if idx.get('source_mtime', 0) == src_mtime:
                matched_index_cache = idx
                return matched_index_cache
        except Exception:
            pass
    # 重建索引（首次会慢，但只做一次）
    print(f"[索引] 构建 matched_by_book 索引（首次较慢）：{MATCHED_JSON_PATH}")
    if MATCHED_JSON_PATH.exists():
        with open(MATCHED_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        books = data.get('books', {})
        idx_books = {}
        for name, b in books.items():
            idx_books[name] = {
                'total_standard_chars': b.get('total_standard_chars', 0),
                'total_instances': b.get('total_instances', 0)
            }
        matched_index_cache = {
            'books': idx_books,
            'source_mtime': src_mtime
        }
        with open(MATCHED_INDEX_PATH, 'w', encoding='utf-8') as f:
            json.dump(matched_index_cache, f, ensure_ascii=False, indent=2)
        return matched_index_cache
    else:
        matched_index_cache = { 'books': {}, 'source_mtime': src_mtime }
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
    if not book_name:
        return None
    if book_name in matched_books_cache:
        return matched_books_cache[book_name]
    _ensure_cache_dirs()
    shard_path = MATCHED_SHARDS_DIR / f"{book_name}.json"
    src_mtime = MATCHED_JSON_PATH.stat().st_mtime if MATCHED_JSON_PATH.exists() else 0
    # 如果分片存在且不过期，直接加载
    if shard_path.exists():
        try:
            with open(shard_path, 'r', encoding='utf-8') as f:
                shard = json.load(f)
            data = shard.get('data')
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
        book = (full.get('books') or {}).get(book_name)
        if book:
            matched_books_cache[book_name] = book
            return book
    return None


def _resolve_standard_chars_path() -> Path:
    if STANDARD_CHARS_JSON_PATH.exists():
        return STANDARD_CHARS_JSON_PATH
    if STANDARD_CHARS_FALLBACK_PATH.exists():
        return STANDARD_CHARS_FALLBACK_PATH
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


def _derive_lookup_from_review(review: Dict) -> Dict:
    """
    从 review_results.json 的 instances 与 matched_by_book.json 推导 lookup 结构：
    { 'books': { book: { char: { instance_id: {bbox, source_image, ...} } } } }
    - 如果 review 内部已有 'lookup' 字段，则优先直接拷贝
    - 否则按 instances + matched_by_book 补齐
    """
    out = { 'version': 1, 'books': {} }
    books = (review or {}).get('books', {})
    for book_name, book_obj in books.items():
        matched_book = ensure_matched_book_data(book_name)
        if not matched_book:
            continue
        out_book = {}
        for char, char_obj in book_obj.items():
            if not isinstance(char_obj, dict):
                continue
            # 如果已有 lookup，直接使用
            if 'lookup' in char_obj and isinstance(char_obj['lookup'], dict):
                out_book[char] = char_obj['lookup']
                continue
            if char not in matched_book.get('chars', {}):
                continue
            matched_list = matched_book['chars'][char]
            instances = char_obj.get('instances', {}) or {}
            look = {}
            for idx_str, selected in instances.items():
                if not selected:
                    continue
                try:
                    idx = int(idx_str)
                    if idx < 0 or idx >= len(matched_list):
                        continue
                except Exception:
                    continue
                inst = matched_list[idx]
                instance_id = f"册{inst['volume']:02d}_page{inst['page'].split('_')[-1]}_idx{inst['char_index']}"
                src = normalize_to_preprocessed_path(inst.get('source_image', ''))
                look[instance_id] = {
                    'bbox': inst.get('bbox', {}),
                    'source_image': src,
                    'confidence': inst.get('confidence', 0.0),
                    'volume': inst.get('volume'),
                    'page': inst.get('page'),
                    'char_index': inst.get('char_index')
                }
            if look:
                out_book[char] = look
        if out_book:
            out['books'][book_name] = out_book
    return out


def _make_instance_id(inst: Dict) -> str:
    try:
        vol = int(inst.get('volume', 0))
    except Exception:
        vol = 0
    page = inst.get('page', '')
    page_suffix = page.split('_')[-1] if page else ''
    char_index = inst.get('char_index', 0)
    return f"册{vol:02d}_page{page_suffix}_idx{char_index}"


def _parse_instance_id(instance_id: str):
    """从 instance_id 中解析 volume, page_suffix, char_index。匹配失败返回 None。"""
    import re
    m = re.match(r'^册(\d+)_page(\d+)_idx(\d+)$', instance_id)
    if not m:
        return None
    vol = int(m.group(1))
    page = m.group(2)
    idx = int(m.group(3))
    return vol, page, idx


def _sync_lookup_for_char(book_name: str, char: str, char_obj: Dict) -> set:
    """确保指定字符的 lookup 与 instances 同步，返回被移除的实例 ID 集合。"""
    ensure_segment_lookup_data()
    matched_book = ensure_matched_book_data(book_name)
    if not matched_book:
        char_obj['lookup'] = {}
        return set()

    matched_list = matched_book.get('chars', {}).get(char)
    if not matched_list:
        char_obj['lookup'] = {}
        return set()

    instances = char_obj.get('instances', {}) or {}
    new_lookup = {}
    selected_ids = set()

    for idx_str, selected in instances.items():
        if not selected:
            continue
        try:
            idx = int(idx_str)
            if idx < 0 or idx >= len(matched_list):
                continue
            inst = matched_list[idx]
        except Exception:
            continue

        inst_id = _make_instance_id(inst)
        selected_ids.add(inst_id)
        src = normalize_to_preprocessed_path(inst.get('source_image', ''))
        new_lookup[inst_id] = {
            'bbox': inst.get('bbox', {}),
            'source_image': src,
            'confidence': inst.get('confidence', 0.0),
            'volume': inst.get('volume'),
            'page': inst.get('page'),
            'char_index': inst.get('char_index'),
            'index': idx
        }

    old_lookup = char_obj.get('lookup') or {}
    removed_ids = set(old_lookup.keys()) - selected_ids
    char_obj['lookup'] = new_lookup

    book_cache = segment_lookup_data.setdefault('books', {}).setdefault(book_name, {})
    book_cache[char] = new_lookup

    return removed_ids


def _ensure_lookup_for_book(book_name: str) -> Dict:
    """确保书内所有字符 lookup 可用（按需补齐），返回 lookup 映射。"""
    ensure_segment_lookup_data()
    book_obj = read_review_book(book_name) or {}
    if not isinstance(book_obj, dict):
        segment_lookup_data['books'][book_name] = {}
        return segment_lookup_data['books'][book_name]

    updated = False
    removed_by_char: Dict[str, set] = {}
    for char, char_obj in book_obj.items():
        if not isinstance(char_obj, dict):
            continue
        lookup_map = char_obj.get('lookup') or {}
        seg_map = char_obj.get('segments') or {}
        instances = char_obj.get('instances') or {}
        need_sync = char_obj.get('_lookup_dirty') or ('lookup' not in char_obj)
        if not need_sync:
            if instances and not lookup_map:
                need_sync = True
            elif lookup_map and not instances:
                need_sync = True
            elif seg_map and any(k not in lookup_map for k in seg_map.keys()):
                need_sync = True
        if not need_sync:
            continue
        removed_ids = _sync_lookup_for_char(book_name, char, char_obj)
        lookup_ids = set((char_obj.get('lookup') or {}).keys())
        if seg_map:
            stale_ids = set(seg_map.keys()) - lookup_ids
            if stale_ids:
                for inst_id in stale_ids:
                    seg_map.pop(inst_id, None)
                if seg_map:
                    char_obj['segments'] = seg_map
                else:
                    char_obj.pop('segments', None)
                removed_ids = set(removed_ids) | stale_ids
        char_obj.pop('_lookup_dirty', None)
        book_obj[char] = char_obj
        updated = True
        if removed_ids:
            removed_by_char[char] = removed_ids

    if updated:
        try:
            write_review_book(book_name, book_obj, skip_backup=True)
        except Exception:
            pass
        for char, removed_ids in removed_by_char.items():
            _remove_segmentation_entries(book_name, char, removed_ids)

    out_book = {}
    for char, char_obj in book_obj.items():
        if not isinstance(char_obj, dict):
            continue
        out_book[char] = char_obj.get('lookup') or {}
    segment_lookup_data['books'][book_name] = out_book
    return out_book


def _remove_segmentation_entries(book_name: str, char: str, instance_ids: set):
    """删除指定实例的切割状态，并删除对应图片。"""
    if not instance_ids:
        return
    try:
        import fcntl
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
        changed = False

        if book_obj and isinstance(book_obj, dict):
            char_obj = book_obj.get(char) or {}
            seg_map = char_obj.get('segments') or {}
            for inst_id in instance_ids:
                entry = seg_map.pop(inst_id, None)
                if entry:
                    seg_rel = entry.get('segmented_path')
                    if seg_rel:
                        seg_abs = PROJECT_ROOT / seg_rel
                        try:
                            if seg_abs.exists():
                                seg_abs.unlink()
                        except Exception:
                            pass
                    changed = True
            if seg_map:
                char_obj['segments'] = seg_map
                book_obj[char] = char_obj
            else:
                if 'segments' in char_obj:
                    char_obj.pop('segments', None)
                    book_obj[char] = char_obj

        if changed:
            try:
                write_review_book(book_name, book_obj)
            except Exception:
                pass

        if fcntl:
            try:
                fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass


def ensure_segment_lookup_data():
    """初始化内存 lookup 缓存，用于 Fixing/Segment 快速访问。"""
    global segment_lookup_data
    if segment_lookup_data is None:
        segment_lookup_data = { 'version': 1, 'books': {} }


def _review_book_path(book_name: str) -> Path:
    safe_name = book_name.replace('/', '_')
    return REVIEW_BOOKS_DIR / f'{safe_name}.json'


def _review_book_backup_dir(book_name: str) -> Path:
    safe_name = book_name.replace('/', '_')
    return REVIEW_BOOKS_DIR / '_backups' / safe_name


def _read_review_payload(path: Path) -> Optional[Dict]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _extract_review_book_chars(payload: Dict, book_name: str) -> Optional[Dict]:
    if not isinstance(payload, dict):
        return None
    if 'chars' in payload and isinstance(payload['chars'], dict):
        return payload['chars']
    if 'books' in payload and isinstance(payload['books'], dict):
        return payload['books'].get(book_name)
    return None


def _read_latest_review_backup(book_name: str) -> Optional[Dict]:
    backup_dir = _review_book_backup_dir(book_name)
    if not backup_dir.exists():
        return None
    candidates = sorted(backup_dir.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        payload = _read_review_payload(path)
        if payload is not None:
            print(f'⚠️  {book_name}: 主文件损坏，回退到备份 {path.name}')
            return payload
    return None


def _rotate_review_backups(backup_dir: Path, keep: int):
    try:
        backups = sorted(backup_dir.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in backups[keep:]:
            old.unlink(missing_ok=True)
    except Exception:
        pass


def _maybe_backup_review_book(book_name: str, book_path: Path):
    if not book_path.exists():
        return
    backup_dir = _review_book_backup_dir(book_name)
    backup_dir.mkdir(parents=True, exist_ok=True)
    backups = sorted(backup_dir.glob('*.json'), key=lambda p: p.stat().st_mtime)
    if backups:
        latest_mtime = backups[-1].stat().st_mtime
        if time.time() - latest_mtime < REVIEW_BOOK_BACKUP_COOLDOWN:
            return
    ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup_path = backup_dir / f'{ts}.json'
    try:
        shutil.copy2(book_path, backup_path)
    except Exception:
        return
    _rotate_review_backups(backup_dir, REVIEW_BOOK_BACKUP_KEEP)


def _fsync_dir(path: Path):
    try:
        dir_fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception:
        pass


def read_review_book(book_name: str) -> Optional[Dict]:
    """读取单本书的 OCR 数据（字符级），返回 {char: data}。"""
    path = _review_book_path(book_name)
    if not path.exists():
        return None
    payload = _read_review_payload(path)
    if payload is None:
        payload = _read_latest_review_backup(book_name)
    if payload is None:
        return None
    return _extract_review_book_chars(payload, book_name)


def write_review_book(book_name: str, book_data: Dict, skip_backup: bool = False):
    """写入单本书的 OCR 数据（字符级）。"""
    REVIEW_BOOKS_DIR.mkdir(parents=True, exist_ok=True)
    book_path = _review_book_path(book_name)
    if not skip_backup:
        _maybe_backup_review_book(book_name, book_path)
    payload = {
        'version': 2,
        'book': book_name,
        'chars': book_data
    }
    tmp = book_path.with_suffix(f'.json.tmp.{os.getpid()}-{int(time.time()*1000)}')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, book_path)
    _fsync_dir(book_path.parent)


def list_review_books() -> list:
    if not REVIEW_BOOKS_DIR.exists():
        return []
    return sorted([p.stem for p in REVIEW_BOOKS_DIR.glob('*.json')])


def read_all_review_books() -> Dict:
    """聚合所有分片为 {version:2, books:{book: chars}} 的内存视图。"""
    out = {'version': 2, 'books': {}}
    for book_name in list_review_books():
        chars = read_review_book(book_name)
        if isinstance(chars, dict):
            out['books'][book_name] = chars
    return out


def _paddle_book_path(book_name: str) -> Path:
    safe_name = book_name.replace('/', '_')
    return PADDLE_REVIEW_BOOKS_DIR / f'{safe_name}.json'


def _paddle_book_lock_path(book_name: str) -> Path:
    safe_name = book_name.replace('/', '_')
    return PADDLE_REVIEW_BOOKS_DIR / f'{safe_name}.json.lock'


def list_paddle_books() -> list:
    if not PADDLE_REVIEW_BOOKS_DIR.exists():
        return []
    return sorted([p.stem for p in PADDLE_REVIEW_BOOKS_DIR.glob('*.json')])


def read_paddle_book(book_name: str) -> Optional[Dict]:
    path = _paddle_book_path(book_name)
    if not path.exists():
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def write_paddle_book(book_name: str, payload: Dict) -> bool:
    PADDLE_REVIEW_BOOKS_DIR.mkdir(parents=True, exist_ok=True)
    book_path = _paddle_book_path(book_name)
    lock_path = _paddle_book_lock_path(book_name)
    with open(lock_path, 'a+', encoding='utf-8') as lock_fp:
        if not _acquire_book_lock(lock_fp, timeout_sec=2.0):
            return False
        tmp = book_path.with_suffix(book_path.suffix + f'.tmp.{os.getpid()}-{int(time.time()*1000)}')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, book_path)
        _fsync_dir(book_path.parent)
    return True


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
    """返回某本书在内存中的 lookup 映射（直接来自 OCR 结果，不再补齐切割实例）。"""
    ensure_segment_lookup_data()
    if not book_name:
        return None
    if book_name in segment_lookup_data['books']:
        return segment_lookup_data['books'][book_name]
    return _ensure_lookup_for_book(book_name)


# ==================== 切割审查状态：单文件加锁工具（统一存于 review_results.json） ====================
def _review_lock_path() -> Path:
    # 兼容旧逻辑的全局锁（尽量避免使用）
    return REVIEW_RESULTS_PATH.with_suffix(REVIEW_RESULTS_PATH.suffix + '.lock')


def _review_book_lock_path(book_name: str) -> Path:
    safe_name = book_name.replace('/', '_')
    return REVIEW_BOOKS_DIR / f'{safe_name}.json.lock'


def _acquire_book_lock(lock_fp, timeout_sec: float = 2.0) -> bool:
    try:
        import fcntl
    except Exception:
        return True
    start = time.time()
    while True:
        try:
            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            if time.time() - start >= timeout_sec:
                return False
            time.sleep(0.05)
        except Exception:
            return True
def _read_review_results() -> Dict:
    # 兼容旧逻辑：现在统一从分片读取
    return read_all_review_books()

def read_review_data() -> dict:
    """
    兼容接口：返回切割状态视图（源自 review_results.json 的 segments 字段）。
    结构与旧 segmentation_review.json 保持一致：{version, books{book{char{inst_id: entry}}}}
    """
    rr = read_all_review_books()
    out = {'version': rr.get('version', 2), 'books': {}}
    books = rr.get('books', {})
    for book, chars in books.items():
        if not isinstance(chars, dict):
            continue
        book_out = {}
        for char, char_obj in chars.items():
            if not isinstance(char_obj, dict):
                continue
            seg_map = char_obj.get('segments', {})
            if seg_map:
                book_out[char] = seg_map
        if book_out:
            out['books'][book] = book_out
    return out


def build_combined_book(book_name: str) -> dict:
    """
    构建统一视图：以 OCR 选中的 lookup 为主，叠加切割状态。
    返回结构：{ char: { inst_id: {selected, bbox, source_image, ...,
                                 status, decision, segmented_path, method, timestamp} } }
    """
    lookup_book = get_lookup_book(book_name) or {}
    seg_data = read_review_data()
    seg_book = (seg_data.get('books') or {}).get(book_name, {})

    combined = {}
    for char, inst_map in lookup_book.items():
        seg_char = seg_book.get(char, {})
        combined_char = {}
        for inst_id, info in inst_map.items():
            seg_entry = seg_char.get(inst_id, {})
            merged = {
                'selected': True,
                'bbox': info.get('bbox', {}),
                'source_image': info.get('source_image'),
                'confidence': info.get('confidence'),
                'volume': info.get('volume'),
                'page': info.get('page'),
                'char_index': info.get('char_index'),
                'index': info.get('index'),
                'status': seg_entry.get('status', 'unreviewed'),
                'decision': seg_entry.get('decision'),
                'segmented_path': seg_entry.get('segmented_path'),
                'method': seg_entry.get('method'),
                'timestamp': seg_entry.get('timestamp')
            }
            combined_char[inst_id] = merged
        combined[char] = combined_char
    return combined

def write_review_data(seg_view: dict):
    """
    将切割状态写回 review_results.json 的 segments 字段。
    仅更新 segments，不改动 instances/lookup/timestamp。
    同时写出兼容文件 segmentation_review.json（派生视图），便于旧脚本使用。
    """
    # 按书写入分片文件的 segments 字段
    for book, chars in (seg_view.get('books') or {}).items():
        if not isinstance(chars, dict):
            continue
        book_obj = read_review_book(book) or {}
        for char, seg_map in chars.items():
            if not isinstance(seg_map, dict):
                continue
            char_obj = book_obj.setdefault(char, {})
            char_obj['segments'] = seg_map
        try:
            write_review_book(book, book_obj)
        except Exception:
            pass

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
    """对切割状态进行加锁的读-改-写更新（存于 review_results.json 的 segments）。"""
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
        char_obj = book_obj.setdefault(char, {})
        seg_map = char_obj.setdefault('segments', {})
        seg_map[instance_id] = entry
        char_obj['segments'] = seg_map
        book_obj[char] = char_obj
        write_review_book(book_name, book_obj)

        if fcntl:
            try:
                fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass

def normalize_to_preprocessed_path(raw_or_mixed_path: str) -> str:
    """
    将路径统一转换为 preprocessed 格式

    data/raw/{book}/册XX_pages/page_XXXX.png
    → data/results/preprocessed/{book}/册XX_pages/page_XXXX_preprocessed.png
    """
    import re
    from pathlib import Path

    # 如果已经是 preprocessed 路径，直接返回
    if '/preprocessed/' in raw_or_mixed_path and '_preprocessed.png' in raw_or_mixed_path:
        return raw_or_mixed_path

    # 解析路径
    # 模式：data/raw/{book}/册XX_pages/page_XXXX.png
    match = re.search(r'data/raw/([^/]+)/(册\d+_pages)/(page_\d+)\.png', raw_or_mixed_path)

    if match:
        book, volume_dir, page_name = match.groups()
        preprocessed_path = f'data/results/preprocessed/{book}/{volume_dir}/{page_name}_preprocessed.png'
        return preprocessed_path

    # 如果无法匹配，返回原路径并警告
    print(f"⚠️  无法标准化路径: {raw_or_mixed_path}")
    return raw_or_mixed_path


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
    books = list_review_books()
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
    """保存审查结果到服务器（v2格式：字符级别时间戳合并）"""
    try:
        global segment_lookup_data
        req_start = time.perf_counter()
        # 记录请求信息
        content_length = request.content_length
        size_kb = (content_length / 1024.0) if content_length else 0
        app.logger.info(
            '[save_review] from=%s origin=%s referer=%s ua=%s size=%.2fKB',
            request.remote_addr,
            request.headers.get('Origin'),
            request.headers.get('Referer'),
            request.headers.get('User-Agent'),
            size_kb
        )

        # 解析 JSON（设置超时和大小限制）
        try:
            client_results = request.get_json(force=True, silent=False)
        except Exception as e:
            print(f'❌ 解析 JSON 失败: {e}')
            import traceback
            print(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': f'无法解析请求数据: {str(e)}'
            }), 400

        if not client_results:
            print('❌ 请求数据为空')
            return jsonify({'success': False, 'error': '请求数据为空'}), 400

        # 严格校验客户端数据
        valid, error = validate_data_format(client_results)
        if not valid:
            print(f'❌ 客户端数据格式错误: {error}')
            return jsonify({'success': False, 'error': f'客户端数据格式错误: {error}'}), 400

        # 逐书加锁写入，避免全局覆盖
        client_books = client_results.get('books', {})
        app.logger.info('[save_review] books=%s', ','.join(client_books.keys()))

        def merge_char(server_char_data: dict, client_char_data: dict):
            """返回合并后的字符数据，保留服务器 segments 与 lookup/instances 一致性。"""
            merged = dict(client_char_data)
            # 如果客户端没有 segments，则沿用服务器的 segments
            if 'segments' not in merged and isinstance(server_char_data, dict):
                if 'segments' in server_char_data:
                    merged['segments'] = server_char_data['segments']
            else:
                # 双方都有时，合并（客户端覆盖同名实例）
                if isinstance(server_char_data, dict):
                    srv_seg = server_char_data.get('segments', {}) or {}
                    cli_seg = merged.get('segments', {}) or {}
                    merged['segments'] = {**srv_seg, **cli_seg}
            return merged

        for book_name, client_book in client_books.items():
            lock_path = _review_book_lock_path(book_name)
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with open(lock_path, 'a+') as lock_fp:
                lock_ok = _acquire_book_lock(lock_fp, timeout_sec=2.0)
                if not lock_ok:
                    app.logger.warning('[save_review] lock busy book=%s', book_name)
                    return jsonify({'success': False, 'error': f'书籍 "{book_name}" 正在被占用，稍后再试'}), 423

                book_start = time.perf_counter()
                server_book = read_review_book(book_name) or {}

                # 字符级别比较时间戳
                for char, client_char_data in client_book.items():
                    server_char_data = server_book.get(char)

                    # 如果服务器没有这个字符，直接添加
                    if not server_char_data:
                        client_char_data['_lookup_dirty'] = True
                        client_char_data.pop('lookup', None)
                        server_book[char] = client_char_data
                        print(f'  {book_name}: 新增字符 "{char}"')
                        continue

                    # 比较时间戳（统一转换为 UTC 时间戳进行比较）
                    from datetime import datetime, timezone

                    def parse_timestamp(ts_str):
                        """解析 ISO 格式时间戳，统一转换为 UTC 时间戳（秒）"""
                        if not ts_str:
                            return 0
                        try:
                            # 替换 'Z' 为 '+00:00' 以兼容 Python < 3.11
                            ts_str = ts_str.replace('Z', '+00:00')
                            dt = datetime.fromisoformat(ts_str)
                            # 如果是 naive datetime，假定为 UTC
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=timezone.utc)
                            return dt.timestamp()
                        except Exception as e:
                            print(f'    解析时间戳失败: {ts_str}, 错误: {e}')
                            return 0

                    client_timestamp = parse_timestamp(client_char_data.get('timestamp', '1970-01-01T00:00:00'))
                    server_timestamp = parse_timestamp(server_char_data.get('timestamp', '1970-01-01T00:00:00'))

                    # 如果实例集合有变化，优先接受客户端更新（避免时间戳漂移导致不更新）
                    def instance_keys(data):
                        inst = (data or {}).get('instances') or {}
                        return {k for k, v in inst.items() if v}

                    client_keys = instance_keys(client_char_data)
                    server_keys = instance_keys(server_char_data)

                    if client_keys != server_keys or client_timestamp > server_timestamp:
                        merged_char = merge_char(server_char_data, client_char_data)
                        if client_keys != server_keys:
                            # 强制刷新时间戳，避免客户端时间戳未更新导致“看起来没保存”
                            merged_char['timestamp'] = datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')
                            merged_char['_lookup_dirty'] = True
                            merged_char.pop('lookup', None)
                        server_book[char] = merged_char
                    else:
                        # 服务器更新较新，保留服务器；但可选择合并客户端 segments（此处保留服务器为准）
                        pass

                try:
                    write_review_book(book_name, server_book)
                except Exception:
                    pass

                try:
                    import fcntl
                    fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
                app.logger.info('[save_review] book=%s done in %.2fms', book_name, (time.perf_counter() - book_start) * 1000)

        # 刷新缓存：确保 OCR 修改立即反映到切割/修正界面
        if segment_lookup_data is not None:
            for book_name in client_books.keys():
                segment_lookup_data.get('books', {}).pop(book_name, None)

        app.logger.info('[save_review] total %.2fms', (time.perf_counter() - req_start) * 1000)
        return jsonify({
            'success': True,
            'message': '审查结果已保存 (v2格式)'
        })

    except Exception as e:
        import traceback
        print(f'保存审查结果失败: {str(e)}')
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/load_review', methods=['GET'])
def load_review():
    """从服务器加载审查结果（v2格式，严格校验）"""
    try:
        book_only = request.args.get('book')
        if book_only:
            # 优先读取单书分片文件
            book_chars = read_review_book(book_only)
            if book_chars is None:
                return jsonify({'success': True, 'data': {'version': 2, 'books': {}}})
            return jsonify({'success': True, 'data': {'version': 2, 'books': {book_only: book_chars}}})

        results = read_all_review_books()
        return jsonify({
            'success': True,
            'data': results
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
                segmented_rel = saved.get('segmented_path') if isinstance(saved, dict) else None
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
            book_dir = SEGMENTED_DIR / book_name
            book_dir.mkdir(parents=True, exist_ok=True)

            # 解码 base64
            img_data = base64.b64decode(segmented_b64.split(',')[-1])
            # 文件名格式：{char}_{instance_id}.png
            segmented_path = book_dir / f"{char}_{instance_id}.png"
            with open(segmented_path, 'wb') as f:
                f.write(img_data)

        # 更新审查状态（单文件加锁写入）
        from datetime import datetime, timezone
        rel_path = f"data/results/manual/segmented/{book_name}/{char}_{instance_id}.png" if segmented_path else None
        update_review_entry(book_name, char, instance_id, {
            'status': status,
            'method': method,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'segmented_path': rel_path,
            'decision': decision
        })

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
        update_review_entry(book_name, char, instance_id, {
            'status': 'unreviewed',
            'method': None,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'segmented_path': None,
            'decision': 'unknown'
        })

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
        segmented_path = entry.get('segmented_path')

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

        update_review_entry(book_name, char, instance_id, {
            'status': status,
            'method': method,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'segmented_path': segmented_path,
            'decision': decision
        })

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


@app.route('/api/export_roi', methods=['GET'])
def api_export_roi():
    """
    导出原图 ROI（供手动 PS 处理）

    Query params: book, char, instance_id
    """
    try:
        book_name = request.args.get('book')
        char = request.args.get('char')
        instance_id = request.args.get('instance_id')

        if not all([book_name, char, instance_id]):
            return jsonify({'success': False, 'error': '缺少必要参数'}), 400

        # 从派生的 lookup 结构获取实例信息
        lookup_book = get_lookup_book(book_name)
        if not lookup_book:
            return '书籍不存在', 404
        if char not in lookup_book or instance_id not in lookup_book[char]:
            return '未找到实例信息', 404

        instance_info = lookup_book[char][instance_id]
        app.logger.info('[API] /export_roi book=%s char=%s instance=%s', book_name, char, instance_id)

        # 裁切 ROI
        source_image_path = instance_info['source_image']

        # 转换为绝对路径
        if not os.path.isabs(source_image_path):
            source_image_path = str(PROJECT_ROOT / source_image_path)

        img = cv2.imread(source_image_path)
        bbox = instance_info['bbox']
        padding = 10

        h, w = img.shape[:2]
        x = max(0, bbox['x'] - padding)
        y = max(0, bbox['y'] - padding)
        x2 = min(w, bbox['x'] + bbox['width'] + padding)
        y2 = min(h, bbox['y'] + bbox['height'] + padding)

        roi = img[y:y2, x:x2]

        # 保存到 manual 工作区（扁平：manual/{book}/{char}_{instance_id}.png）
        manual_dir = MANUAL_WORK_DIR / book_name
        manual_dir.mkdir(parents=True, exist_ok=True)

        roi_path = manual_dir / f"{char}_{instance_id}.png"
        cv2.imwrite(str(roi_path), roi)
        app.logger.info('[API] /export_roi saved ROI: %s size=%sx%s', str(roi_path), roi.shape[1], roi.shape[0])

        # 返回文件
        return send_file(roi_path, as_attachment=True, download_name=f"{book_name}_{char}_{instance_id}.png")

    except Exception as e:
        print(f'❌ 导出 ROI 失败: {str(e)}')
        print(traceback.format_exc())
        return str(e), 500


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
            pending = sum(1 for s in inst_map.values() if s.get('status') == 'pending_manual')
            dropped = sum(1 for s in inst_map.values() if s.get('decision') == 'drop')
            effective = max(0, total_instances - dropped)

            chars_info[char] = {
                'total': total_instances,
                'effective': effective,
                'count': effective,
                'confirmed': confirmed,
                'pending': pending,
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


# ==================== 手动处理工作流 API ====================

@app.route('/api/list_manual_todo', methods=['GET'])
def api_list_manual_todo():
    """
    列出手动处理工作区的所有项目

    返回 manual/ 工作区中的所有文件（待处理或已处理）
    """
    try:
        todo_items = []

        # 遍历 manual 工作区（扁平：manual/{book}/{char}_{instance_id}.png）
        if not MANUAL_WORK_DIR.exists():
            return jsonify({
                'success': True,
                'todo_items': [],
                'total': 0
            })

        for book_dir in MANUAL_WORK_DIR.iterdir():
            if not book_dir.is_dir():
                continue

            book_name = book_dir.name

            # 直接扫描该书籍目录下的 png 文件：{char}_{instance_id}.png
            for img_file in book_dir.glob('*.png'):
                stem = img_file.stem
                # 拆分出 char 和 instance_id（按第一个下划线分割）
                if '_' in stem:
                    char, instance_id = stem.split('_', 1)
                else:
                    # 非法命名，跳过
                    continue

                todo_items.append({
                    'book': book_name,
                    'char': char,
                    'instance_id': instance_id,
                    'file_path': str(img_file.relative_to(PROJECT_ROOT))
                })

        app.logger.info('[API] /list_manual_todo total=%d', len(todo_items))
        return jsonify({
            'success': True,
            'todo_items': todo_items,
            'total': len(todo_items)
        })

    except Exception as e:
        print(f'❌ 列出待处理项目失败: {str(e)}')
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/import_manual_results', methods=['POST'])
def api_import_manual_results():
    """
    导入手动处理的结果（简化版）

    扫描 manual/ 工作区，将图片移动到 segmented/ 目录并更新审查状态

    Request body (可选):
    {
        "book": "...",  // 只导入指定书籍
        "char": "..."   // 只导入指定字符
    }
    """
    try:
        data = request.get_json() or {}
        filter_book = data.get('book')
        filter_char = data.get('char')

        imported_count = 0
        errors = []

        # 不再预读全量，逐实例更新（加锁）

        import shutil

        # 遍历 manual 工作区（扁平：manual/{book}/{char}_{instance_id}.png）
        if not MANUAL_WORK_DIR.exists():
            return jsonify({
                'success': True,
                'imported': 0,
                'message': 'manual 工作区不存在'
            })

        for book_dir in MANUAL_WORK_DIR.iterdir():
            if not book_dir.is_dir():
                continue

            book_name = book_dir.name
            if filter_book and book_name != filter_book:
                continue

            # 查找所有 .png 文件：{char}_{instance_id}.png
            for img_file in book_dir.glob('*.png'):
                stem = img_file.stem
                if '_' not in stem:
                    continue
                char, instance_id = stem.split('_', 1)

                if filter_char and char != filter_char:
                    continue

                try:
                    # 移动文件到 segmented 目录（扁平结构）
                    target_dir = SEGMENTED_DIR / book_name
                    target_dir.mkdir(parents=True, exist_ok=True)

                    target_file = target_dir / f"{char}_{instance_id}.png"
                    shutil.move(str(img_file), str(target_file))

                    # 更新审查状态（加锁写入）
                    rel_path = f"data/results/manual/segmented/{book_name}/{char}_{instance_id}.png"
                    update_review_entry(book_name, char, instance_id, {
                        'status': 'confirmed',
                        'method': 'manual',
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'segmented_path': rel_path
                    })

                    imported_count += 1
                    app.logger.info('[API] /import_manual_results imported %s/%s/%s', book_name, char, instance_id)

                except Exception as e:
                    errors.append(f'{book_name}/{char}/{instance_id}: {str(e)}')
                    continue

            # 删除空目录
            try:
                if not any(book_dir.iterdir()):
                    book_dir.rmdir()
            except:
                pass

        # 已逐条更新，无需统一保存

        app.logger.info('[API] /import_manual_results done imported=%d errors=%d', imported_count, len(errors))
        return jsonify({
            'success': True,
            'imported': imported_count,
            'errors': errors,
            'error_count': len(errors)
        })

    except Exception as e:
        print(f'❌ 导入手动处理结果失败: {str(e)}')
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== 审查界面路由 ====================

# ==================== Fixing（问题集中处理）只读 API ====================

def _iter_confirmed_entries(review: Dict, book_name: str) -> List[Tuple[str, str, str]]:
    """返回该书所有已确认的 (char, instance_id, segmented_rel_path)。"""
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
            seg_rel = entry.get('segmented_path')
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
                seg_rel = entry.get('segmented_path')
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
                seg_rel = row.get('segmented_path')
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
    try:
        return render_template('paddle/paddle_review.html')
    except Exception:
        return 'Paddle Review 页面未生成', 404


@app.route('/fixing')
def fixing_page():
    """Fixing 页面（问题集中处理，读取 segmentation_review.json）"""
    try:
        return render_template('manual/fixing.html')
    except Exception:
        return 'Fixing 页面未生成', 404

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


@app.route('/ocr_review')
def ocr_review_page():
    """第一轮审查界面（OCR/文字审查）——优先模板，其次兼容旧文件名"""
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
    return 'OCR 审查界面未生成', 404

@app.route('/review')
def review_page_compat():
    """兼容旧路径，重定向到 /ocr_review"""
    return redirect(url_for('ocr_review_page'))


@app.route('/segment_review')
def segment_review_page():
    """第二轮切割审查界面（优先模板，其次兼容旧文件）"""
    tpl_path = TEMPLATE_DIR / 'manual/segment_review_app.html'
    if tpl_path.exists():
        return render_template('manual/segment_review_app.html')
    html_path = PROJECT_ROOT / 'data/results/segment_review_app.html'
    if html_path.exists():
        with open(html_path, 'r', encoding='utf-8') as f:
            return f.read()
    return '切割审查界面未生成', 404


def main():
    """启动服务器"""
    # 使用懒加载机制，启动时不加载数据，只在访问相关页面时才加载
    import socket

    # 获取本机局域网 IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except:
        local_ip = "无法获取"

    print("\n" + "=" * 70)
    print("古籍字形审查系统 - 后端服务器")
    print("=" * 70)
    print(f"数据文件：{MATCHED_JSON_PATH}")
    print(f"标准字文件：{_resolve_standard_chars_path()}")
    print(f"图片目录：{PREPROCESSED_DIR}")
    print(f"切割结果目录：{SEGMENTED_DIR}")
    print("查找索引：内存派生自 review_results.json（不再依赖独立文件）")

    print("\n服务器启动中...")
    print("=" * 70)
    print("\n访问地址：")
    print("  【系统首页】选择审查模式")
    print(f"    本机访问：http://localhost:5001/")
    print(f"    局域网访问：http://{local_ip}:5001/")
    print("\n  【第一轮审查】文字筛选（OCR 审查）")
    print(f"    本机访问：http://localhost:5001/ocr_review")
    print(f"    局域网访问：http://{local_ip}:5001/ocr_review")
    print("\n  【第二轮审查】字符切割")
    print(f"    本机访问：http://localhost:5001/segment_review")
    print(f"    局域网访问：http://{local_ip}:5001/segment_review")
    print("\n  【Paddle 复核】Paddle 自动筛选后复核")
    print(f"    本机访问：http://localhost:5001/paddle_review")
    print(f"    局域网访问：http://{local_ip}:5001/paddle_review")
    print("\n按 Ctrl+C 停止服务器")
    print("=" * 70 + "\n")

    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)


if __name__ == '__main__':
    main()
