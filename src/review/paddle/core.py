from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.review.segment import segment_character

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "data/results"
PADDLE_RESULTS_DIR = RESULTS_DIR / "paddle"
MATCHED_JSON = RESULTS_DIR / "matched_by_book.json"
MATCHED_BOOKS_DIR = RESULTS_DIR / "matched_books"
MATCHED_SHARDS = RESULTS_DIR / "_cache/matched_by_book_shards"
PREPROCESSED_DIR = RESULTS_DIR / "preprocessed"
SEGMENTED_DIR = PADDLE_RESULTS_DIR / "segmented"
REVIEW_PADDLE_DIR = PADDLE_RESULTS_DIR / "review_books"


def normalize_to_preprocessed_path(raw_or_mixed_path: str) -> str:
    if not raw_or_mixed_path:
        return raw_or_mixed_path
    if "/preprocessed/" in raw_or_mixed_path and "_preprocessed.png" in raw_or_mixed_path:
        return raw_or_mixed_path
    match = re.search(r"data/raw/([^/]+)/(册\\d+_pages)/(page_\\d+)\\.png", raw_or_mixed_path)
    if match:
        book, volume_dir, page_name = match.groups()
        return f"data/results/preprocessed/{book}/{volume_dir}/{page_name}_preprocessed.png"
    return raw_or_mixed_path


def load_matched_book(book: str) -> Optional[Dict]:
    book_path = MATCHED_BOOKS_DIR / f"{book}.json"
    if book_path.exists():
        try:
            payload = json.loads(book_path.read_text(encoding="utf-8"))
            data = payload.get("data") if isinstance(payload, dict) else None
            if isinstance(data, dict):
                return data
            if isinstance(payload.get("chars"), dict):
                return payload
        except Exception:
            pass
    shard_path = MATCHED_SHARDS / f"{book}.json"
    if shard_path.exists():
        try:
            payload = json.loads(shard_path.read_text(encoding="utf-8"))
            data = payload.get("data")
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    if not MATCHED_JSON.exists():
        return None
    with open(MATCHED_JSON, "r", encoding="utf-8") as f:
        full = json.load(f)
    return (full.get("books") or {}).get(book)


def list_books() -> List[str]:
    if MATCHED_BOOKS_DIR.exists() and list(MATCHED_BOOKS_DIR.glob("*.json")):
        return sorted([p.stem for p in MATCHED_BOOKS_DIR.glob("*.json")])
    if MATCHED_SHARDS.exists():
        return sorted([p.stem for p in MATCHED_SHARDS.glob("*.json")])
    if MATCHED_JSON.exists():
        data = json.loads(MATCHED_JSON.read_text(encoding="utf-8"))
        return sorted((data.get("books") or {}).keys())
    return []


def make_instance_id(inst: Dict) -> str:
    try:
        vol = int(inst.get("volume", 0))
    except Exception:
        vol = 0
    page = inst.get("page", "")
    page_suffix = page.split("_")[-1] if page else ""
    char_index = inst.get("char_index", 0)
    return f"册{vol:02d}_page{page_suffix}_idx{char_index}"


def encode_png_b64(img_bgr) -> str:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise ValueError("PNG 编码失败")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def encode_png_bytes(img_bgr) -> bytes:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise ValueError("PNG 编码失败")
    return buf.tobytes()


def parse_paddle_response(payload: Dict) -> Tuple[str, float]:
    if not isinstance(payload, dict):
        return ("", 0.0)
    if "text" in payload:
        try:
            return (str(payload.get("text") or ""), float(payload.get("confidence") or payload.get("score") or 0.0))
        except Exception:
            return (str(payload.get("text") or ""), 0.0)
    if "text_regions" in payload and isinstance(payload["text_regions"], list) and payload["text_regions"]:
        best = None
        for item in payload["text_regions"]:
            if not isinstance(item, dict):
                continue
            conf = item.get("confidence") if item.get("confidence") is not None else item.get("score")
            try:
                conf = float(conf)
            except Exception:
                conf = 0.0
            if best is None or conf > best[1]:
                best = (str(item.get("text") or ""), conf)
        if best:
            return best
    if "result" in payload and isinstance(payload["result"], list) and payload["result"]:
        item = payload["result"][0]
        if isinstance(item, dict):
            try:
                return (str(item.get("text") or item.get("transcription") or ""), float(item.get("confidence") or item.get("score") or 0.0))
            except Exception:
                return (str(item.get("text") or item.get("transcription") or ""), 0.0)
    if "data" in payload and isinstance(payload["data"], list) and payload["data"]:
        item = payload["data"][0]
        if isinstance(item, dict):
            try:
                return (str(item.get("text") or ""), float(item.get("score") or item.get("confidence") or 0.0))
            except Exception:
                return (str(item.get("text") or ""), 0.0)
    return ("", 0.0)


def _resolve_paddle_url(paddle_url: str) -> str:
    url = paddle_url.strip()
    if "/ocr/" in url:
        return url
    return url.rstrip("/") + "/ocr/predict_base64"


def call_paddle_ocr(img_bgr, paddle_url: str, timeout: int) -> Tuple[str, float]:
    b64 = encode_png_b64(img_bgr)
    form = urllib.parse.urlencode({"image_base64": b64}).encode("utf-8")
    url = _resolve_paddle_url(paddle_url)
    req = urllib.request.Request(url, data=form, headers={"Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    payload = json.loads(data.decode("utf-8"))
    return parse_paddle_response(payload)


def _resolve_paddle_batch_url(paddle_url: str) -> str:
    url = paddle_url.strip()
    if url.endswith("/ocr/batch"):
        return url
    if url.endswith("/ocr/predict_base64") or url.endswith("/ocr/predict"):
        return url.rsplit("/", 1)[0] + "/batch"
    if "/ocr/" in url:
        base = url.split("/ocr/")[0]
        return base.rstrip("/") + "/ocr/batch"
    return url.rstrip("/") + "/ocr/batch"


def _encode_multipart(files: List[bytes]) -> Tuple[str, bytes]:
    boundary = "----PaddleBoundary" + uuid.uuid4().hex
    body = bytearray()
    for idx, data in enumerate(files):
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="files"; filename="img_{idx}.png"\r\n'.encode("utf-8"))
        body.extend(b"Content-Type: image/png\r\n\r\n")
        body.extend(data)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    return boundary, bytes(body)


def _parse_batch_payload(payload: object, count: int) -> List[Tuple[str, float]]:
    items = None
    if isinstance(payload, dict):
        if isinstance(payload.get("results"), list):
            items = payload.get("results")
        elif isinstance(payload.get("data"), list):
            items = payload.get("data")
    elif isinstance(payload, list):
        items = payload
    if not items:
        items = [payload] if payload else []
    results = [parse_paddle_response(item) for item in items]
    if count and len(results) < count:
        results.extend([("", 0.0)] * (count - len(results)))
    return results[:count]


def call_paddle_batch(images: List[bytes], paddle_url: str, timeout: int) -> List[Tuple[str, float]]:
    if not images:
        return []
    if len(images) == 1:
        dummy = cv2.imdecode(np.frombuffer(images[0], dtype=np.uint8), cv2.IMREAD_COLOR)
        if dummy is None:
            return [("", 0.0)]
        return [call_paddle_ocr(dummy, paddle_url, timeout)]
    boundary, body = _encode_multipart(images)
    url = _resolve_paddle_batch_url(paddle_url)
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    payload = json.loads(data.decode("utf-8"))
    return _parse_batch_payload(payload, len(images))

def rank_candidates(candidates: List[Dict], topk: int | None) -> List[Dict]:
    candidates.sort(
        key=lambda x: (
            -int(x.get("width") or 0),
            -int(1 if x.get("match") else 0),
            -float(x.get("paddle_conf") or 0.0),
            x.get("instance_id") or ""
        )
    )
    if topk is None or topk <= 0:
        return candidates
    return candidates[:topk]


def write_book_paddle(book: str, data: Dict) -> None:
    REVIEW_PADDLE_DIR.mkdir(parents=True, exist_ok=True)
    safe = book.replace("/", "_")
    out_path = REVIEW_PADDLE_DIR / f"{safe}.json"
    tmp = out_path.with_suffix(out_path.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, out_path)


def _normalize_min_conf(min_conf: float) -> float:
    try:
        val = float(min_conf)
    except Exception:
        return 0.75
    if val > 1.0:
        val = val / 100.0
    return max(0.0, min(1.0, val))


def _normalize_text(text: str) -> str:
    if text is None:
        return ""
    return "".join(str(text).split())


def _is_single_char(text: str) -> bool:
    cleaned = _normalize_text(text)
    return len(cleaned) == 1


def _prepare_instance(ch: str, inst: Dict, stats: Dict[str, int], error_samples: Dict[str, List[str]], sample_fn):
    bbox = inst.get("bbox") or {}
    source_image = normalize_to_preprocessed_path(inst.get("source_image") or "")
    instance_id = make_instance_id(inst)
    if not source_image:
        stats["skipped_source"] += 1
        sample_fn(error_samples["missing_source"], f"{ch}:{instance_id} 缺少 source")
        return None
    preprocessed = PROJECT_ROOT / source_image
    if not preprocessed.exists():
        stats["skipped_source"] += 1
        sample_fn(error_samples["missing_source"], f"{ch}:{instance_id} {source_image}")
        return None
    try:
        seg_out = segment_character(str(preprocessed), bbox)
        if isinstance(seg_out, tuple) and len(seg_out) >= 2:
            segmented_img = seg_out[1]
        else:
            stats["seg_fail"] += 1
            sample_fn(error_samples["seg_fail"], f"{ch}:{instance_id}")
            return None
    except Exception:
        stats["seg_fail"] += 1
        sample_fn(error_samples["seg_fail"], f"{ch}:{instance_id}")
        return None
    try:
        img_bytes = encode_png_bytes(segmented_img)
    except Exception:
        stats["encode_fail"] += 1
        sample_fn(error_samples["encode_fail"], f"{ch}:{instance_id}")
        return None
    h, w = segmented_img.shape[:2]
    return {
        "char": ch,
        "inst": inst,
        "instance_id": instance_id,
        "bbox": bbox,
        "source_image": source_image,
        "img_bytes": img_bytes,
        "width": int(w),
        "height": int(h),
    }


def process_book(book: str, paddle_url: str, topk: int, timeout: int, limit_chars: Optional[int], limit_instances: Optional[int], min_conf: float, batch_size: int, require_match: bool) -> int:
    matched = load_matched_book(book)
    if not matched:
        print(f"跳过 {book}：无 matched 数据")
        return 0
    chars = matched.get("chars") or {}
    out_chars: Dict[str, Dict] = {}
    seg_dir = SEGMENTED_DIR / book
    seg_dir.mkdir(parents=True, exist_ok=True)
    min_conf = _normalize_min_conf(min_conf)

    char_keys = list(chars.keys())
    if limit_chars is not None:
        char_keys = char_keys[:limit_chars]

    stats = {
        "instances": 0,
        "candidates": 0,
        "saved": 0,
        "skipped_source": 0,
        "seg_fail": 0,
        "encode_fail": 0,
        "paddle_fail": 0,
        "filtered_non_single": 0,
        "filtered_low_conf": 0,
        "filtered_mismatch": 0,
    }
    error_samples = {
        "missing_source": [],
        "seg_fail": [],
        "encode_fail": [],
        "paddle_fail": [],
    }
    sample_limit = 3
    def _maybe_sample(bucket: List[str], msg: str):
        if len(bucket) < sample_limit:
            bucket.append(msg)
    total_chars = len(char_keys)
    batch_size = max(1, int(batch_size or 1))
    states: Dict[str, Dict] = {}
    for ch in char_keys:
        instances = chars.get(ch) or []
        if limit_instances is not None:
            instances = instances[:limit_instances]
        instances = sorted(instances, key=lambda inst: -int((inst.get("bbox") or {}).get("width") or 0))
        stats["instances"] += len(instances)
        states[ch] = {
            "instances": instances,
            "idx": 0,
            "candidates": [],
        }

    progress = tqdm(total=total_chars, desc=f"Paddle {book}", unit="char")
    done_chars = set()
    active_chars = [ch for ch in char_keys if states[ch]["instances"]]
    for ch in char_keys:
        if not states[ch]["instances"]:
            done_chars.add(ch)
            progress.update(1)

    pending: List[Dict] = []
    cursor = 0

    def _mark_done(ch: str):
        if ch not in done_chars:
            done_chars.add(ch)
            progress.update(1)

    while active_chars:
        if len(pending) < batch_size:
            if cursor >= len(active_chars):
                cursor = 0
            ch = active_chars[cursor]
            state = states[ch]
            if topk > 0 and len(state["candidates"]) >= topk:
                active_chars.pop(cursor)
                _mark_done(ch)
                continue
            if state["idx"] >= len(state["instances"]):
                active_chars.pop(cursor)
                _mark_done(ch)
                continue
            inst = state["instances"][state["idx"]]
            state["idx"] += 1
            prepared = _prepare_instance(ch, inst, stats, error_samples, _maybe_sample)
            if prepared:
                pending.append(prepared)
            cursor += 1
            continue

        try:
            results = call_paddle_batch([b["img_bytes"] for b in pending], paddle_url, timeout=timeout)
        except Exception as e:
            stats["paddle_fail"] += len(pending)
            msg = f"{type(e).__name__}: {e}"
            _maybe_sample(error_samples["paddle_fail"], msg)
            results = [("", 0.0)] * len(pending)

        for item, (paddle_text, paddle_conf) in zip(pending, results):
            ch = item["char"]
            state = states[ch]
            if topk > 0 and len(state["candidates"]) >= topk:
                continue
            if not _is_single_char(paddle_text):
                stats["filtered_non_single"] += 1
                continue
            cleaned_text = _normalize_text(paddle_text)
            if float(paddle_conf or 0.0) < min_conf:
                stats["filtered_low_conf"] += 1
                continue
            match = cleaned_text == ch
            if require_match and not match:
                stats["filtered_mismatch"] += 1
                continue
            state["candidates"].append({
                "instance_id": item["instance_id"],
                "bbox": item["bbox"],
                "source_image": item["source_image"],
                "segmented_path": None,
                "paddle_text": cleaned_text,
                "paddle_conf": float(paddle_conf),
                "width": item["width"],
                "height": item["height"],
                "decision": "pending",
                "match": match,
                "png_bytes": item["img_bytes"],
            })
            stats["candidates"] += 1
        pending = []
        progress.set_postfix({
            "saved": stats["saved"],
            "cand": stats["candidates"],
            "seg_fail": stats["seg_fail"],
            "paddle_fail": stats["paddle_fail"],
        })

    if pending:
        try:
            results = call_paddle_batch([b["img_bytes"] for b in pending], paddle_url, timeout=timeout)
        except Exception as e:
            stats["paddle_fail"] += len(pending)
            msg = f"{type(e).__name__}: {e}"
            _maybe_sample(error_samples["paddle_fail"], msg)
            results = [("", 0.0)] * len(pending)
        for item, (paddle_text, paddle_conf) in zip(pending, results):
            ch = item["char"]
            state = states[ch]
            if topk > 0 and len(state["candidates"]) >= topk:
                continue
            if not _is_single_char(paddle_text):
                stats["filtered_non_single"] += 1
                continue
            cleaned_text = _normalize_text(paddle_text)
            if float(paddle_conf or 0.0) < min_conf:
                stats["filtered_low_conf"] += 1
                continue
            match = cleaned_text == ch
            if require_match and not match:
                stats["filtered_mismatch"] += 1
                continue
            state["candidates"].append({
                "instance_id": item["instance_id"],
                "bbox": item["bbox"],
                "source_image": item["source_image"],
                "segmented_path": None,
                "paddle_text": cleaned_text,
                "paddle_conf": float(paddle_conf),
                "width": item["width"],
                "height": item["height"],
                "decision": "pending",
                "match": match,
                "png_bytes": item["img_bytes"],
            })
            stats["candidates"] += 1
        pending = []
        progress.set_postfix({
            "saved": stats["saved"],
            "cand": stats["candidates"],
            "seg_fail": stats["seg_fail"],
            "paddle_fail": stats["paddle_fail"],
        })

    progress.close()

    for ch in char_keys:
        candidates = states[ch]["candidates"]
        if not candidates:
            out_chars[ch] = {
                "items": {},
                "top5": [],
                "order": [],
                "scores": {},
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            continue
        sorted_all = rank_candidates(candidates, topk=None)
        top_items = sorted_all[:topk] if topk and topk > 0 else []
        top_ids = {it.get("instance_id") for it in top_items if it.get("instance_id")}
        for item in sorted_all:
            if item.get("instance_id") not in top_ids:
                item.pop("png_bytes", None)
        for item in top_items:
            instance_id = item.get("instance_id")
            png_bytes = item.pop("png_bytes", None)
            if not instance_id or not png_bytes:
                item["segmented_path"] = None
                continue
            seg_path = seg_dir / f"{ch}_{instance_id}.png"
            try:
                with open(seg_path, "wb") as f:
                    f.write(png_bytes)
                item["segmented_path"] = f"data/results/paddle/segmented/{book}/{ch}_{instance_id}.png"
                stats["saved"] += 1
            except Exception:
                item["segmented_path"] = None

        order = [it.get("instance_id") for it in sorted_all if it.get("instance_id")]
        scores = {
            it["instance_id"]: {
                "paddle_text": it.get("paddle_text", ""),
                "paddle_conf": it.get("paddle_conf", 0.0),
                "match": bool(it.get("match")),
                "width": it.get("width"),
                "height": it.get("height"),
            }
            for it in sorted_all
            if it.get("instance_id")
        }

        out_chars[ch] = {
            "items": {it["instance_id"]: it for it in top_items},
            "top5": [it["instance_id"] for it in top_items],
            "order": order,
            "scores": scores,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    if not out_chars:
        return 0

    payload = {
        "version": 1,
        "book": book,
        "chars": out_chars,
    }
    write_book_paddle(book, payload)
    print(
        f"✓ {book}: chars={len(out_chars)}/{total_chars} "
        f"instances={stats['instances']} saved={stats['saved']} "
        f"candidates={stats['candidates']} skipped_src={stats['skipped_source']} "
        f"seg_fail={stats['seg_fail']} paddle_fail={stats['paddle_fail']} "
        f"filtered(non_single={stats['filtered_non_single']},low_conf={stats['filtered_low_conf']},mismatch={stats['filtered_mismatch']})"
    )
    if error_samples["missing_source"]:
        print(f"  - 缺少图片样例: {', '.join(error_samples['missing_source'])}")
    if error_samples["seg_fail"]:
        print(f"  - 切割失败样例: {', '.join(error_samples['seg_fail'])}")
    if error_samples["encode_fail"]:
        print(f"  - 编码失败样例: {', '.join(error_samples['encode_fail'])}")
    if error_samples["paddle_fail"]:
        print(f"  - Paddle 失败样例: {', '.join(error_samples['paddle_fail'])}")
    return len(out_chars)


def run_paddle_pipeline(books: List[str], paddle_url: str, topk: int, timeout: int, limit_chars: Optional[int], limit_instances: Optional[int], min_conf: float, batch_size: int, workers: int, require_match: bool) -> int:
    count = 0
    if workers and int(workers) != 1:
        print("⚠️ Paddle 单 GPU：已强制 workers=1（顺序处理）")
    for book in books:
        count += process_book(
            book,
            paddle_url=paddle_url,
            topk=topk,
            timeout=timeout,
            limit_chars=limit_chars,
            limit_instances=limit_instances,
            min_conf=min_conf,
            batch_size=batch_size,
            require_match=require_match,
        )
    return count
