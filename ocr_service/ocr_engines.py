"""Native OCR engine creation, timeouts and text extraction.

Adapted from the Amazon listing pipeline ``utils/image/ocr_engines.py``. The
only structural change is that image loading now comes from this package's
``image_assets`` module instead of the listing project's ``gma_runtime``.
All timeout / warm-up / circuit-breaker behaviour is kept as-is.
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import time
from functools import lru_cache
from typing import Any, List, Tuple

import numpy as np
from PIL import Image
from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR

from .image_assets import get_image_asset, load_image as load_shared_image

logger = logging.getLogger(__name__)


def _env_positive_float(name: str, default: float, minimum: float) -> float:
    try:
        return max(float(minimum), float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return float(default)


def _ocr_cooldown_seconds() -> float:
    """How long local OCR stays suspended after a hard failure (per process)."""
    return max(30.0, _env_positive_float("AMAZON_OCR_DISABLE_COOLDOWN_SECONDS", 600, 30))


def _ocr_is_currently_disabled() -> bool:
    return time.monotonic() < _OCR_DISABLED_UNTIL


def _ocr_mark_disabled() -> None:
    global _OCR_DISABLED_UNTIL
    with _OCR_DISABLE_GUARD:
        _OCR_DISABLED_UNTIL = max(
            _OCR_DISABLED_UNTIL, time.monotonic() + _ocr_cooldown_seconds()
        )


# RapidOCR/PaddleOCR native sessions are not safe for concurrent inference.
# Keep one caller in the native layer and disable it after a hard timeout so a
# hung native thread cannot cause every later image to queue indefinitely.
_OCR_ENGINE_INIT_LOCK = threading.RLock()
_OCR_INFERENCE_LOCK = threading.RLock()
# Recoverable OCR disable (per process): a hard failure only suspends local OCR
# for a cooldown window, then the next listing retries automatically.
_OCR_DISABLE_GUARD = threading.Lock()
_OCR_DISABLED_UNTIL = 0.0  # monotonic timestamp; OCR disabled until this time
_OCR_WARMUP_GUARD = threading.Lock()
_OCR_WARMUP_GRACE_UNTIL = 0.0
_OCR_ENGINE_TIMEOUT_SECONDS = _env_positive_float(
    "AMAZON_OCR_ENGINE_TIMEOUT_SECONDS", 15, 5
)
_OCR_LOCK_WAIT_SECONDS = _env_positive_float(
    "AMAZON_OCR_LOCK_WAIT_SECONDS", 5, 1
)
_OCR_INITIALIZATION_TIMEOUT_SECONDS = _env_positive_float(
    "AMAZON_OCR_INITIALIZATION_TIMEOUT_SECONDS", 90, 5
)
# First real inference through a freshly created ONNX session pays a large one-off
# cost. A tiny warm-up image absorbs that cost, and the grace window keeps the
# first real images on the wider budget too.
_OCR_WARMUP_TIMEOUT_SECONDS = _env_positive_float(
    "AMAZON_OCR_WARMUP_TIMEOUT_SECONDS", 180, 5
)
_OCR_WARMUP_GRACE_SECONDS = _env_positive_float(
    "AMAZON_OCR_WARMUP_GRACE_SECONDS", 120, 0
)
_OCR_MAX_IMAGE_SIDE = max(
    512,
    int(_env_positive_float("AMAZON_OCR_MAX_IMAGE_SIDE", 2500, 512)),
)
_OCR_IMAGE_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
)


class OCRTimeoutError(TimeoutError):
    """Raised when native OCR initialization or inference exceeds its deadline."""


class OCRBusyError(TimeoutError):
    """Raised when another OCR call owns the process-local inference lease."""


def _is_fatal_ocr_error(exc: BaseException) -> bool:
    message = str(exc or "").lower()
    return isinstance(exc, MemoryError) or any(
        marker in message
        for marker in (
            "bad allocation",
            "out of memory",
            "not enough memory",
            "failed to allocate",
        )
    )


def ensure_ocr_process_owner() -> bool:
    """Provide a hook for runners that need to refuse duplicate OCR owners."""
    return True


def _initialize_engine(factory, label: str):
    if _ocr_is_currently_disabled():
        raise OCRTimeoutError("local OCR is disabled after an earlier fatal error")
    if not ensure_ocr_process_owner():
        raise OCRBusyError(
            "another process owns the local OCR models; use cloud fallback"
        )

    with _OCR_ENGINE_INIT_LOCK:
        outcome: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

        def invoke() -> None:
            try:
                outcome.put((True, factory()), block=False)
            except BaseException as exc:
                if _is_fatal_ocr_error(exc):
                    _ocr_mark_disabled()
                outcome.put((False, exc), block=False)

        threading.Thread(
            target=invoke,
            name=f"amazon-ocr-init-{label}",
            daemon=True,
        ).start()
        try:
            succeeded, value = outcome.get(timeout=_OCR_INITIALIZATION_TIMEOUT_SECONDS)
        except queue.Empty as exc:
            _ocr_mark_disabled()
            logger.error(
                "Amazon OCR %s model initialization exceeded %.1fs; pausing local OCR (auto-retry after cooldown)",
                label,
                _OCR_INITIALIZATION_TIMEOUT_SECONDS,
            )
            raise OCRTimeoutError(
                f"OCR {label} model initialization timed out after "
                f"{_OCR_INITIALIZATION_TIMEOUT_SECONDS:.1f}s"
            ) from exc

        if succeeded:
            if value is None:
                raise RuntimeError(f"OCR {label} factory returned no engine")
            _mark_warmup_window()
            warm_up_engine(value)
            return value
        if _is_fatal_ocr_error(value):
            _ocr_mark_disabled()
            raise OCRTimeoutError(
                f"OCR {label} model initialization failed due to memory pressure"
            ) from value
        raise value


def _is_image_url(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(("http://", "https://"))


def _limit_image_array(image: Any) -> Any:
    if not isinstance(image, np.ndarray) or image.ndim < 2:
        return image
    height, width = image.shape[:2]
    if max(height, width) <= _OCR_MAX_IMAGE_SIDE:
        return image
    try:
        source = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
        source.thumbnail(
            (_OCR_MAX_IMAGE_SIDE, _OCR_MAX_IMAGE_SIDE),
            Image.Resampling.LANCZOS,
        )
        return np.array(source, copy=True)
    except Exception as exc:
        logger.warning("Unable to resize oversized OCR image: %s", exc)
        return image


def download_image_array(
    url: str,
    *,
    timeout: int = 15,
    max_retries: int = 2,
) -> "np.ndarray | None":
    """Download and validate an image before passing it to native OCR."""
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            with load_shared_image(url, timeout=timeout) as image:
                if max(image.size) > _OCR_MAX_IMAGE_SIDE:
                    image.thumbnail(
                        (_OCR_MAX_IMAGE_SIDE, _OCR_MAX_IMAGE_SIDE),
                        Image.Resampling.LANCZOS,
                    )
                return np.array(image.convert("RGB"), copy=True)
        except Exception as exc:
            last_error = exc
            logger.warning(
                "OCR image download failed [URL=%s] (%s/%s): %s",
                url,
                attempt,
                max_retries,
                exc,
            )
            if attempt < max_retries:
                time.sleep(0.5 * attempt)
    logger.error("OCR image download failed permanently [URL=%s]: %s", url, last_error)
    return None


def create_text_presence_ocr():
    """Return the process-wide lightweight OCR engine."""
    return _initialize_engine(_create_text_presence_ocr, "presence")


@lru_cache(maxsize=1)
def _create_text_presence_ocr():
    return RapidOCR(params={
        "Rec.ocr_version": OCRVersion.PPOCRV6,
        "Rec.lang_type": LangRec.CH,
        "Rec.model_type": ModelType.SMALL,
        "Rec.batch_num": 1,
    })


def create_ai_text_ocr():
    """Return the process-wide optional full-text OCR engine."""
    return _initialize_engine(_create_ai_text_ocr, "text")


@lru_cache(maxsize=1)
def _create_ai_text_ocr():
    paddle = _create_paddle_ocr()
    if paddle is not None:
        return paddle

    logger.warning("PaddleOCR unavailable; falling back to RapidOCR PP-OCRv6 medium")
    return RapidOCR(params={
        "Rec.ocr_version": OCRVersion.PPOCRV6,
        "Rec.lang_type": LangRec.CH,
        "Rec.model_type": ModelType.MEDIUM,
        "Rec.batch_num": 1,
    })


def _create_paddle_ocr():
    try:
        from paddleocr import PaddleOCR
    except Exception:
        return None

    configs = [
        {
            "ocr_version": "PP-OCRv6",
            "lang": "ch",
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        },
        {
            "ocr_version": "PP-OCRv6",
            "lang": "ch",
            "use_angle_cls": False,
        },
        {"lang": "ch"},
    ]
    for cfg in configs:
        try:
            engine = PaddleOCR(**cfg)
            logger.info("PaddleOCR initialized: %s", cfg)
            return engine
        except Exception as exc:
            if _is_fatal_ocr_error(exc):
                raise
            logger.debug("PaddleOCR initialization failed: cfg=%s, error=%s", cfg, exc)
    return None


def _run_ocr_sync(engine: Any, image: Any):
    if engine is None:
        raise RuntimeError("OCR engine is not initialized")
    if hasattr(engine, "predict"):
        return engine.predict(image)
    if hasattr(engine, "ocr"):
        try:
            return engine.ocr(image, cls=False)
        except TypeError:
            return engine.ocr(image)
    return engine(image)


def _mark_warmup_window() -> None:
    """Keep the first real listing images on the wider inference budget."""
    global _OCR_WARMUP_GRACE_UNTIL
    if _OCR_WARMUP_GRACE_SECONDS <= 0:
        return
    with _OCR_WARMUP_GUARD:
        _OCR_WARMUP_GRACE_UNTIL = time.monotonic() + _OCR_WARMUP_GRACE_SECONDS


def _resolve_inference_timeout(requested: float) -> float:
    """Return the effective per-image inference budget."""
    requested = max(0.01, float(requested))
    if _OCR_WARMUP_GRACE_SECONDS <= 0:
        return requested
    with _OCR_WARMUP_GUARD:
        graced = time.monotonic() < _OCR_WARMUP_GRACE_UNTIL
    return max(requested, _OCR_WARMUP_TIMEOUT_SECONDS) if graced else requested


def _new_warmup_image() -> Any:
    """A tiny synthetic image that exercises detection, cls and recognition."""
    image = np.full((64, 256, 3), 255, dtype=np.uint8)
    image[12:52, 12:118] = 0
    image[20:44, 140:244] = 40
    return image


def warm_up_engine(engine: Any, *, timeout_seconds: float | None = None) -> bool:
    """Absorb the one-off first-inference cost outside the request budget.

    A failure here is deliberately soft: it must never disable local OCR.
    """
    timeout_seconds = (
        _OCR_WARMUP_TIMEOUT_SECONDS if timeout_seconds is None else float(timeout_seconds)
    )
    if engine is None:
        return False

    acquired_local = _OCR_INFERENCE_LOCK.acquire(timeout=max(0.0, timeout_seconds))
    if not acquired_local:
        logger.warning(
            "Amazon OCR warm-up skipped: inference busy for %.0fs", timeout_seconds
        )
        return False

    outcome: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def invoke() -> None:
        try:
            _run_ocr_sync(engine, _new_warmup_image())
            outcome.put((True, None), block=False)
        except BaseException as exc:
            outcome.put((False, exc), block=False)

    started = time.monotonic()
    try:
        threading.Thread(
            target=invoke,
            name="amazon-ocr-warmup",
            daemon=True,
        ).start()
        try:
            succeeded, value = outcome.get(timeout=max(1.0, timeout_seconds))
        except queue.Empty:
            logger.warning(
                "Amazon OCR warm-up did not finish within %.0fs; the first listing "
                "image stays on the wider warm-up budget",
                timeout_seconds,
            )
            return False
        if not succeeded:
            logger.warning("Amazon OCR warm-up failed (ignored): %s", value)
            return False
        logger.info(
            "Amazon OCR warm-up completed in %.2fs (first-image overhead absorbed)",
            time.monotonic() - started,
        )
        return True
    finally:
        _OCR_INFERENCE_LOCK.release()


def _run_native_ocr_with_timeout(
    engine: Any,
    image: Any,
    *,
    timeout_seconds: float,
    lock_timeout_seconds: float,
):
    if _ocr_is_currently_disabled():
        raise OCRTimeoutError("local OCR is disabled after an earlier fatal error")

    acquired = _OCR_INFERENCE_LOCK.acquire(
        timeout=max(0.0, float(lock_timeout_seconds))
    )
    if not acquired:
        raise OCRBusyError(
            f"OCR inference is busy for more than {lock_timeout_seconds:.1f}s"
        )

    try:
        if _ocr_is_currently_disabled():
            raise OCRTimeoutError("local OCR is disabled after an earlier fatal error")

        outcome: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

        def invoke() -> None:
            try:
                outcome.put((True, _run_ocr_sync(engine, image)), block=False)
            except BaseException as exc:
                outcome.put((False, exc), block=False)

        threading.Thread(
            target=invoke,
            name="amazon-ocr-inference",
            daemon=True,
        ).start()
        try:
            succeeded, value = outcome.get(timeout=max(0.01, float(timeout_seconds)))
        except queue.Empty as exc:
            _ocr_mark_disabled()
            logger.error(
                "Amazon OCR native inference exceeded %.1fs; pausing local OCR (auto-retry after cooldown)",
                timeout_seconds,
            )
            raise OCRTimeoutError(
                f"OCR native inference timed out after {timeout_seconds:.1f}s"
            ) from exc

        if succeeded:
            return value
        if _is_fatal_ocr_error(value):
            _ocr_mark_disabled()
            logger.error(
                "Amazon OCR native inference ran out of memory; pausing local OCR (auto-retry after cooldown)"
            )
            raise OCRTimeoutError(
                "local OCR disabled after native memory allocation failure"
            ) from value
        raise value
    finally:
        _OCR_INFERENCE_LOCK.release()


def run_ocr(
    engine: Any,
    image: Any,
    *,
    ocr_timeout_retries: int = 1,
    engine_timeout_seconds: float | None = None,
    timeout_seconds: float | None = None,
    lock_timeout_seconds: float | None = None,
) -> Tuple[List[str], List[float]]:
    """Run native OCR without allowing a hung call to stall the batch."""
    if timeout_seconds is not None:
        engine_timeout_seconds = timeout_seconds
    if engine_timeout_seconds is None:
        engine_timeout_seconds = _OCR_ENGINE_TIMEOUT_SECONDS
    if lock_timeout_seconds is None:
        lock_timeout_seconds = _OCR_LOCK_WAIT_SECONDS
    engine_timeout_seconds = _resolve_inference_timeout(engine_timeout_seconds)
    lock_timeout_seconds = _resolve_inference_timeout(lock_timeout_seconds)
    if _ocr_is_currently_disabled():
        raise OCRTimeoutError("local OCR is disabled after an earlier fatal error")

    if _is_image_url(image):
        image = download_image_array(image)
        if image is None:
            return [], []
    image = _limit_image_array(image)

    attempts = max(1, int(ocr_timeout_retries))
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            result = _run_native_ocr_with_timeout(
                engine,
                image,
                timeout_seconds=float(engine_timeout_seconds),
                lock_timeout_seconds=float(lock_timeout_seconds),
            )
            return _extract_ocr_texts(result)
        except (OCRTimeoutError, OCRBusyError):
            raise
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Amazon OCR engine call failed (attempt %s/%s): %s",
                attempt,
                attempts,
                exc,
            )
            if attempt < attempts:
                time.sleep(0.5 * attempt)

    logger.error("Amazon OCR engine call failed permanently: %s", last_error)
    raise last_error or RuntimeError("Amazon OCR engine call failed")


def _extract_ocr_texts(value: Any) -> Tuple[List[str], List[float]]:
    texts: List[str] = []
    scores: List[float] = []

    def add(text, score=1.0):
        text = str(text or "").strip()
        if text:
            texts.append(text)
            try:
                scores.append(float(score))
            except Exception:
                scores.append(1.0)

    def walk(node):
        if node is None:
            return
        if hasattr(node, "txts"):
            txts = node.txts or ()
            node_scores = node.scores or ()
            for idx, text in enumerate(txts):
                score = node_scores[idx] if idx < len(node_scores) else 1.0
                add(text, score)
            return
        if isinstance(node, dict):
            rec_texts = node.get("rec_texts") or node.get("texts")
            rec_scores = node.get("rec_scores") or node.get("scores") or []
            if isinstance(rec_texts, list):
                for idx, text in enumerate(rec_texts):
                    score = rec_scores[idx] if idx < len(rec_scores) else 1.0
                    add(text, score)
            else:
                text = node.get("text") or node.get("transcription")
                if text:
                    add(text, node.get("score", 1.0))
            for child in node.values():
                if isinstance(child, (dict, list, tuple)):
                    walk(child)
            return
        if isinstance(node, (list, tuple)):
            if len(node) == 2 and isinstance(node[1], (list, tuple)) and node[1]:
                text = node[1][0]
                score = node[1][1] if len(node[1]) > 1 else 1.0
                if isinstance(text, str):
                    add(text, score)
                    return
            for child in node:
                walk(child)

    walk(value)
    return texts, scores


def filter_texts(
    texts: List[str],
    scores: List[float],
    min_score: float = 0.45,
) -> List[str]:
    return [
        text
        for idx, text in enumerate(texts)
        if text and (scores[idx] if idx < len(scores) else 1.0) >= min_score
    ]


def filter_text_pairs(
    texts: List[str],
    scores: List[float],
    min_score: float,
) -> Tuple[List[str], List[float]]:
    """Same filter as ``filter_texts`` but keeps the matching scores."""
    kept_texts: List[str] = []
    kept_scores: List[float] = []
    for idx, text in enumerate(texts):
        score = scores[idx] if idx < len(scores) else 1.0
        if text and score >= min_score:
            kept_texts.append(text)
            kept_scores.append(float(score))
    return kept_texts, kept_scores


def ocr_status() -> dict:
    """Runtime status of the process-local circuit breaker (for /health)."""
    return {
        "local_ocr_disabled": _ocr_is_currently_disabled(),
        "cooldown_remaining_seconds": round(max(0.0, _OCR_DISABLED_UNTIL - time.monotonic()), 1),
        "warmup_grace_active": time.monotonic() < _OCR_WARMUP_GRACE_UNTIL,
        "engine_timeout_seconds": _OCR_ENGINE_TIMEOUT_SECONDS,
        "initialization_timeout_seconds": _OCR_INITIALIZATION_TIMEOUT_SECONDS,
        "max_image_side": _OCR_MAX_IMAGE_SIDE,
    }


def clear_ocr_engine_cache() -> None:
    """Clear native engine caches and reset the process-local circuit breaker."""
    global _OCR_DISABLED_UNTIL, _OCR_WARMUP_GRACE_UNTIL
    with _OCR_ENGINE_INIT_LOCK:
        _create_text_presence_ocr.cache_clear()
        _create_ai_text_ocr.cache_clear()
        _OCR_DISABLED_UNTIL = 0.0
        _OCR_WARMUP_GRACE_UNTIL = 0.0
