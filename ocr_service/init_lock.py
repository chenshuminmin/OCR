"""Cross-process lock for ONNX Runtime CUDA session initialization.

Eight uvicorn workers creating CUDA contexts at the same time cause a large
transient commit-charge spike on Windows (and cuDNN host-allocation failures
when no pagefile is configured). Serializing just the one-off engine creation
removes that spike; it does NOT serialize inference, so the workers still run
in parallel once initialized.
"""
from __future__ import annotations

import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

try:
    import msvcrt  # Windows only
except ImportError:  # pragma: no cover
    msvcrt = None

_DEFAULT_LOCK = Path(tempfile.gettempdir()) / "ocr_gpu_init.lock"


def _enabled() -> bool:
    return os.getenv("OCR_SERIALIZE_INIT", "1").strip().lower() in {
        "1", "true", "yes", "on"
    }


@contextmanager
def gpu_init_lock(timeout_seconds: float = 180.0):
    """Serialize engine creation across worker processes (no-op on POSIX)."""
    if msvcrt is None or not _enabled():
        yield
        return

    path = Path(os.getenv("OCR_INIT_LOCK_FILE", str(_DEFAULT_LOCK)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+b") as handle:
        # msvcrt.locking needs at least one byte inside the file.
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)

        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out after {timeout_seconds:.0f}s waiting for the "
                        f"OCR GPU init lock ({path})"
                    )
                time.sleep(0.15)

        try:
            yield
        finally:
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
