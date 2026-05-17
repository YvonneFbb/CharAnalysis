"""Common filesystem helpers for review storage."""

from __future__ import annotations

import os
import time
from pathlib import Path


def fsync_dir(path: Path) -> None:
    try:
        dir_fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception:
        pass


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

