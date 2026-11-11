"""Lazy, serialized OCR wrapper used by the OCR HTTP service.

Adapted from the Amazon listing pipeline's ``OCRProcessor``. Changes:
* image loading comes from this package (no ``gma_runtime`` / rapidocr.utils);
* ``detail`` can be overridden per request;
* the engine used and its filtered scores are exposed as ``last_engine`` /
  ``last_scores`` for the HTTP response;
* the old ``_ocr_unavailable`` permanent kill-switch is removed. A timeout now
  only fails that one call; the engine layer's cooldown circuit breaker decides
  when local OCR resumes, otherwise one hiccup would disable OCR until restart.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import List, Union

from .image_assets import load_image
from .ocr_engines import (
    OCRBusyError,
    OCRTimeoutError,
    clear_ocr_engine_cache,
    create_ai_text_ocr,
    create_text_presence_ocr,
    download_image_array,
    filter_text_pairs,
    run_ocr,
)


class OCRProcessor:
    """Lazy, serialized OCR wrapper."""

    def __init__(self):
        self.presence_ocr = None
        self.ocr = None
        self.load_image = load_image
        self._engine_lock = threading.Lock()
        self._detail_ocr_enabled = str(
            os.getenv("AMAZON_OCR_DETAIL_TEXT_MODEL", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.last_engine: str | None = None
        self.last_scores: List[float] = []
        logging.info(
            "Amazon OCR initialized lazily (detail model enabled=%s)",
            self._detail_ocr_enabled,
        )

    @property
    def detail_model_enabled(self) -> bool:
        return self._detail_ocr_enabled

    def _ensure_presence_ocr(self):
        if self.presence_ocr is not None:
            return self.presence_ocr
        with self._engine_lock:
            if self.presence_ocr is None:
                self.presence_ocr = create_text_presence_ocr()
        if self.presence_ocr is None:
            raise RuntimeError("OCR presence engine was not initialized")
        return self.presence_ocr

    def _ensure_detail_ocr(self):
        if self.ocr is not None:
            return self.ocr
        with self._engine_lock:
            if self.ocr is None:
                self.ocr = create_ai_text_ocr()
        if self.ocr is None:
            raise RuntimeError("OCR detail engine was not initialized")
        return self.ocr

    def warm_up(self) -> bool:
        """Initialize the presence engine in the background at service startup."""
        try:
            self._ensure_presence_ocr()
            return True
        except Exception as exc:
            logging.warning("OCR pre-warm skipped: %s", exc)
            return False

    def _rebuild_presence_ocr(self):
        """Rebuild the presence engine after GPU-arena exhaustion.

        Dropping the old RapidOCR instance releases its ONNX sessions, which
        frees the whole GPU arena; a fresh engine then starts with an empty
        pool and the failed request is retried.
        """
        self.presence_ocr = None
        clear_ocr_engine_cache()
        return self._ensure_presence_ocr()

    def _rebuild_detail_ocr(self):
        self.ocr = None
        clear_ocr_engine_cache()
        return self._ensure_detail_ocr()

    def recognize_text(
        self,
        image,
        min_score=0.45,
        max_retries=2,
        detail: bool | None = None,
    ) -> tuple[bool, Union[List[str], None], bool]:
        """Return ``(has_text, texts, local_ocr_success)``.

        ``success=False`` means the caller should fall back to another OCR path
        (e.g. cloud translation); it does not poison later requests.
        """
        self.last_engine = None
        self.last_scores = []

        if isinstance(image, str) and image.startswith(("http://", "https://")):
            image_url = image
            image = download_image_array(image_url, max_retries=max_retries)
            if image is None:
                logging.error("OCR skipped [URL=%s]: image download failed", image_url)
                return False, [], False

        try:
            presence_ocr = self._ensure_presence_ocr()
            presence_texts, presence_scores = run_ocr(
                presence_ocr,
                image,
                on_engine_exhausted=self._rebuild_presence_ocr,
            )
            presence_filtered, presence_filtered_scores = filter_text_pairs(
                presence_texts, presence_scores, min_score=0.35
            )
        except (OCRTimeoutError, OCRBusyError) as exc:
            logging.warning("Local OCR unavailable; use cloud OCR: %s", exc)
            return False, [], False
        except Exception as exc:
            logging.warning("Local OCR failed; use cloud OCR: %s", exc)
            return False, [], False

        self.last_engine = "presence"
        self.last_scores = presence_filtered_scores
        if not presence_filtered:
            return False, [], True

        # Presence OCR is enough to decide whether text exists. The heavier
        # detail model is opt-in so a product batch cannot exhaust RAM.
        detail_enabled = self._detail_ocr_enabled if detail is None else bool(detail)
        if not detail_enabled:
            return True, presence_filtered, True

        try:
            detail_ocr = self._ensure_detail_ocr()
            texts, scores = run_ocr(
                detail_ocr,
                image,
                on_engine_exhausted=self._rebuild_detail_ocr,
            )
            filtered, filtered_scores = filter_text_pairs(texts, scores, min_score)
        except Exception as exc:
            self._detail_ocr_enabled = False
            logging.warning(
                "Detail OCR unavailable; keep presence OCR result: %s", exc
            )
            return True, presence_filtered, True

        self.last_engine = "detail"
        self.last_scores = filtered_scores or presence_filtered_scores
        return True, filtered or presence_filtered, True

    def extract_text(self, image, min_score=0.45):
        _, texts, _ = self.recognize_text(image, min_score=min_score)
        return "\n".join(texts or [])
