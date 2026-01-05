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
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.review.segment import segment_character

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "data/results"
AUTO_RESULTS_DIR = RESULTS_DIR / "auto"
MATCHED_JSON = RESULTS_DIR / "matched_by_book.json"
MATCHED_SHARDS = RESULTS_DIR / "_cache/matched_by_book_shards"
PREPROCESSED_DIR = RESULTS_DIR / "preprocessed"
SEGMENTED_DIR = AUTO_RESULTS_DIR / "segmented"
REVIEW_AUTO_DIR = AUTO_RESULTS_DIR / "review_books"


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

def rank_candidates(candidates: List[Dict], topk: int) -> List[Dict]:
    candidates.sort(
        key=lambda x: (
            -int(1 if x.get("match") else 0),
            -float(x.get("paddle_conf") or 0.0),
            -int(x.get("width") or 0),
            x.get("instance_id") or ""
        )
    )
    return candidates[:topk]


def write_book_auto(book: str, data: Dict) -> None:
    REVIEW_AUTO_DIR.mkdir(parents=True, exist_ok=True)
    safe = book.replace("/", "_")
    out_path = REVIEW_AUTO_DIR / f"{safe}.json"
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

    for ch in char_keys:
        instances = chars.get(ch) or []
        if limit_instances is not None:
            instances = instances[:limit_instances]
        instances = sorted(instances, key=lambda inst: -int((inst.get("bbox") or {}).get("width") or 0))
        candidates: List[Dict] = []

        idx = 0
        batch_size = max(1, int(batch_size or 1))
        while idx < len(instances):
            if len(candidates) >= topk:
                break
            batch_items = []
            while idx < len(instances) and len(batch_items) < batch_size:
                inst = instances[idx]
                idx += 1
                bbox = inst.get("bbox") or {}
                source_image = normalize_to_preprocessed_path(inst.get("source_image") or "")
                if not source_image:
                    continue
                preprocessed = PROJECT_ROOT / source_image
                if not preprocessed.exists():
                    continue
                try:
                    seg_out = segment_character(str(preprocessed), bbox)
                    if isinstance(seg_out, tuple) and len(seg_out) >= 2:
                        segmented_img = seg_out[1]
                    else:
                        continue
                except Exception:
                    continue
                try:
                    img_bytes = encode_png_bytes(segmented_img)
                except Exception:
                    continue
                h, w = segmented_img.shape[:2]
                batch_items.append({
                    "inst": inst,
                    "bbox": bbox,
                    "source_image": source_image,
                    "img_bytes": img_bytes,
                    "width": int(w),
                    "height": int(h),
                })

            if not batch_items:
                continue

            try:
                results = call_paddle_batch([b["img_bytes"] for b in batch_items], paddle_url, timeout=timeout)
            except Exception:
                results = [("", 0.0)] * len(batch_items)

            for item, (paddle_text, paddle_conf) in zip(batch_items, results):
                if not _is_single_char(paddle_text):
                    continue
                cleaned_text = _normalize_text(paddle_text)
                if float(paddle_conf or 0.0) < min_conf:
                    continue
                inst = item["inst"]
                instance_id = make_instance_id(inst)
                match = cleaned_text == ch
                if require_match and not match:
                    continue
                candidates.append({
                    "instance_id": instance_id,
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
                if len(candidates) >= topk:
                    break

        if not candidates:
            out_chars[ch] = {
                "items": {},
                "top5": [],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            continue

        top_items = rank_candidates(candidates, topk=topk)
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
                item["segmented_path"] = f"data/results/auto/segmented/{book}/{ch}_{instance_id}.png"
            except Exception:
                item["segmented_path"] = None

        out_chars[ch] = {
            "items": {it["instance_id"]: it for it in top_items},
            "top5": [it["instance_id"] for it in top_items],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    if not out_chars:
        return 0

    payload = {
        "version": 1,
        "book": book,
        "chars": out_chars,
    }
    write_book_auto(book, payload)
    print(f"✓ {book}: chars={len(out_chars)}")
    return len(out_chars)


def run_auto_pipeline(books: List[str], paddle_url: str, topk: int, timeout: int, limit_chars: Optional[int], limit_instances: Optional[int], min_conf: float, batch_size: int, require_match: bool) -> int:
    count = 0
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
