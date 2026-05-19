"""Common filesystem helpers for review storage."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional


def fsync_dir(path: Path) -> None:
    try:
        dir_fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception:
        pass


def read_json_file(path: Path) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}-{int(time.time() * 1000)}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


def acquire_file_lock(lock_fp, timeout_sec: float = 2.0) -> bool:
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
