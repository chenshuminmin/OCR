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
from typing import Any, Callable, List, Tuple

import numpy as np
from PIL import Image
from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR

from .image_assets import get_image_asset, load_image as load_shared_image
from .init_lock import gpu_init_lock
from .runtime_env import (
    REQUIRE_GPU,
    engine_params,
    expected_provider,
    runtime_info,
)

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


class OCREngineExhaustedError(RuntimeError):
    """Raised when the GPU memory arena is full and the engine must be rebuilt."""


# Filled in when an engine is created; exposed through /health so a silent
# ONNX Runtime CPU fallback can never masquerade as GPU inference.
_GPU_SESSION_PROVIDERS: dict[str, list[str]] = {}


def _session_providers(engine: Any) -> dict[str, list[str]]:
    """Read the *actual* providers of each RapidOCR ONNX session."""
    found: dict[str, list[str]] = {}
    for name in ("text_det", "text_cls", "text_rec", "det", "cls", "rec"):
        part = getattr(engine, name, None)
        if part is None:
            continue
        holder = getattr(part, "session", None)
        session = getattr(holder, "session", None) or holder
        getter = getattr(session, "get_providers", None)
        if callable(getter):
            try:
                found[name] = list(getter())
            except Exception:
                continue
    return found


def _assert_engine_on_gpu(engine: Any, label: str) -> dict[str, list[str]]:
    """Fail loudly when the CUDA/DML provider was requested but not used."""
    global _GPU_SESSION_PROVIDERS
    providers = _session_providers(engine)
    if providers:
        _GPU_SESSION_PROVIDERS = providers

    wanted = expected_provider()
    if wanted is None or not REQUIRE_GPU:
        return providers
    if not providers:
        if REQUIRE_GPU and wanted:
            raise RuntimeError(
                f"GPU inference is required but {label} exposed no ONNX sessions. "
                f"The engine failed to initialize (check CUDA DLLs / commit memory)."
            )
        logger.warning("OCR %s: cannot introspect session providers", label)
        return providers

    fallback = {k: v for k, v in providers.items() if not v or v[0] != wanted}
    if fallback:
        raise RuntimeError(
            f"GPU inference is required but {label} sessions fell back to CPU: "
            f"{fallback}. Check the CUDA runtime DLLs (OCR_CUDA_DLL_DIR) "
            f"and the onnxruntime-gpu / CUDA version match."
        )
    logger.info("OCR %s engine running on %s: %s", label, wanted, providers)
    return providers


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


def is_arena_exhaustion_error(exc: BaseException) -> bool:
    """Detect ORT CUDA BFC-arena exhaustion specifically.

    The BFC arena never returns memory to the GPU, so under sustained large
    image traffic it eventually fills its cap. That failure is *transient* for
    the process (rebuilding the engine releases everything), unlike a true
    out-of-VRAM condition.
    """
    message = str(exc or "").lower()
    return (
        "bfc_arena" in message
        or "allocaterawinternal" in message
        or ("failed to allocate" in message and "arena" in message)
    )


def ensure_ocr_process_owner() -> bool:
    """Provide a hook for runners that need to refuse duplicate OCR owners."""
    return True


def _initialize_engine(factory, label: str):
    """Serialize the one-off CUDA session creation across worker processes.

    Inference itself is NOT serialized; each worker keeps its own engine.
    """
    with gpu_init_lock():
        return _initialize_engine_unlocked(factory, label)


def _initialize_engine_unlocked(factory, label: str):
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
            # RapidOCR creates its ONNX sessions lazily on the first inference,
            # so the real execution providers only exist after warm-up.
            warm_up_engine(value)
            try:
                _assert_engine_on_gpu(value, label)
            except RuntimeError:
                _ocr_mark_disabled()
                raise
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
    return RapidOCR(params=engine_params({
        "Rec.ocr_version": OCRVersion.PPOCRV6,
        "Rec.lang_type": LangRec.CH,
        "Rec.model_type": ModelType.SMALL,
        "Rec.batch_num": 1,
    }))


def create_ai_text_ocr():
    """Return the process-wide optional full-text OCR engine."""
    return _initialize_engine(_create_ai_text_ocr, "text")


@lru_cache(maxsize=1)
def _create_ai_text_ocr():
    paddle = _create_paddle_ocr()
    if paddle is not None:
        return paddle

    logger.warning("PaddleOCR unavailable; falling back to RapidOCR PP-OCRv6 medium")
    return RapidOCR(params=engine_params({
        "Rec.ocr_version": OCRVersion.PPOCRV6,
        "Rec.lang_type": LangRec.CH,
        "Rec.model_type": ModelType.MEDIUM,
        "Rec.batch_num": 1,
    }))


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
        if is_arena_exhaustion_error(value):
            # The arena is full but the process is healthy; raising the
            # dedicated error lets run_ocr rebuild the engine (freeing all
            # GPU memory) and retry, instead of returning empty results.
            # NOTE: must run before _is_fatal_ocr_error — the "Available
            # memory ... smaller than requested" variant contains neither
            # "failed to allocate" nor "out of memory".
            raise OCREngineExhaustedError(str(value)) from value
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
    on_engine_exhausted: Callable[[], Any] | None = None,
) -> Tuple[List[str], List[float]]:
    """Run native OCR without allowing a hung call to stall the batch.

    ``on_engine_exhausted`` is called when the GPU memory arena is full; it
    must return a freshly built engine (dropping the old one releases the
    arena), and the call is then retried instead of failing.
    """
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
    rebuild_budget = 2 if on_engine_exhausted is not None else 0
    attempt = 1
    last_error: Exception | None = None
    while True:
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
        except OCREngineExhaustedError as exc:
            last_error = exc
            if rebuild_budget <= 0:
                raise
            rebuild_budget -= 1
            logger.warning(
                "OCR GPU arena exhausted; rebuilding engine and retrying "
                "(rebuild %s/2): %s",
                2 - rebuild_budget,
                exc,
            )
            engine = on_engine_exhausted()
            time.sleep(0.5)
            continue
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Amazon OCR engine call failed (attempt %s/%s): %s",
                attempt,
                attempts,
                exc,
            )
            if attempt >= attempts:
                break
            attempt += 1
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
    info = runtime_info()
    wanted = expected_provider()
    sessions = _GPU_SESSION_PROVIDERS
    gpu_ready = bool(wanted and sessions and all(
        providers and providers[0] == wanted for providers in sessions.values()
    ))
    return {
        "local_ocr_disabled": _ocr_is_currently_disabled(),
        "cooldown_remaining_seconds": round(max(0.0, _OCR_DISABLED_UNTIL - time.monotonic()), 1),
        "warmup_grace_active": time.monotonic() < _OCR_WARMUP_GRACE_UNTIL,
        "engine_timeout_seconds": _OCR_ENGINE_TIMEOUT_SECONDS,
        "initialization_timeout_seconds": _OCR_INITIALIZATION_TIMEOUT_SECONDS,
        "max_image_side": _OCR_MAX_IMAGE_SIDE,
        "ep": info["ep"],
        "ort_version": info["ort_version"],
        "ort_device": info["ort_device"],
        "available_providers": info["available_providers"],
        "session_providers": sessions,
        "gpu_ready": gpu_ready,
        "gpu_device_id": info["gpu_device_id"],
    }


def clear_ocr_engine_cache() -> None:
    """Clear native engine caches and reset the process-local circuit breaker."""
    global _OCR_DISABLED_UNTIL, _OCR_WARMUP_GRACE_UNTIL
    with _OCR_ENGINE_INIT_LOCK:
        _create_text_presence_ocr.cache_clear()
        _create_ai_text_ocr.cache_clear()
        _OCR_DISABLED_UNTIL = 0.0
        _OCR_WARMUP_GRACE_UNTIL = 0.0
