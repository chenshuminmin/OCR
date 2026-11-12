"""OCR HTTP service — RapidOCR (PP-OCRv6) with timeout / warm-up / circuit breaker.

Run:
    .venv\\Scripts\\python.exe server.py          # default 0.0.0.0:6575
    set PORT=7000 && .venv\\Scripts\\python.exe server.py

Endpoints:
    GET  /                 service info
    GET  /health           liveness + engine / circuit-breaker status
    GET  /sample.png       a generated text image for quick testing
    POST /ocr              JSON  {image_url | image_base64, ...}
    POST /ocr/upload       multipart {file, ...}
    POST /ocr/reset        clear the local-OCR cooldown (recovery)
"""
from __future__ import annotations

import base64
import binascii
import io
import logging
import os
import threading
import time
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel, Field

from ocr_service.ocr_engines import clear_ocr_engine_cache, ocr_status
from ocr_service.processor import OCRProcessor
from ocr_service.sample_image import make_sample_image

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ocr-service")

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "6575"))
# Each worker owns one ONNX session set and one inference lock, so effective
# OCR concurrency == worker count. Eight workers by default.
WORKERS = max(1, int(os.getenv("WORKERS", os.getenv("WEB_CONCURRENCY", "8"))))
MAX_UPLOAD_BYTES = int(os.getenv("OCR_MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
REQUEST_LOCK_TIMEOUT = float(os.getenv("OCR_REQUEST_LOCK_TIMEOUT", "200"))
PREWARM = os.getenv("OCR_PREWARM", "1").strip().lower() in {
    "1", "true", "yes", "on"
}

processor = OCRProcessor()
# Native ONNX sessions are not safe for concurrent inference inside a process.
# Multiple workers are separate processes, each with its own session + lock.
_request_lock = threading.Lock()

# ORT CUDA arenas (BFC), cuDNN workspaces and pinned host buffers only grow
# during the worker's lifetime, so commit charge creeps up until the whole
# system runs out of virtual memory (which crashed the NVIDIA driver once).
# Recycle the worker process when its private commit exceeds a cap; uvicorn
# respawns dead workers automatically, so the port never goes down.
WORKER_MAX_COMMIT_GB = float(os.getenv("OCR_WORKER_MAX_COMMIT_GB", "8"))
MEMORY_CHECK_SECONDS = float(os.getenv("OCR_MEMORY_CHECK_SECONDS", "60"))


def _process_private_bytes() -> int:
    """Private (commit) bytes of the current process, via psapi."""
    import ctypes
    from ctypes import wintypes

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    psapi = ctypes.WinDLL("psapi")
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    handle = wintypes.HANDLE(-1)  # GetCurrentProcess pseudo-handle
    if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
        return 0
    return counters.PagefileUsage


def _memory_recycle_loop():
    limit_bytes = WORKER_MAX_COMMIT_GB * 1024 ** 3
    while True:
        time.sleep(MEMORY_CHECK_SECONDS)
        try:
            private = _process_private_bytes()
        except Exception:
            continue
        if not private or private <= limit_bytes:
            continue
        # Acquiring the request lock guarantees no request is mid-flight.
        if _request_lock.acquire(timeout=30):
            _request_lock.release()
            logger.critical(
                "OCR worker pid=%s private memory %.2fGB > %.2fGB cap; "
                "recycling process (uvicorn will respawn it)",
                os.getpid(), private / 1024 ** 3, WORKER_MAX_COMMIT_GB,
            )
            os._exit(0)

app = FastAPI(
    title="OCR Service",
    version="1.0.0",
    description="图片文字识别（本地 RapidOCR PP-OCRv6，含超时/预热/熔断）",
)


class OCRRequest(BaseModel):
    image_url: str | None = Field(
        default=None, description="HTTP(S) 图片地址，服务端自动下载"
    )
    image_base64: str | None = Field(
        default=None, description="图片 base64，可带 data:image/...;base64, 前缀"
    )
    min_score: float = Field(default=0.45, ge=0.0, le=1.0, description="文本置信度阈值")
    detail: bool | None = Field(
        default=None,
        description="是否启用高精度模型；不传则跟随环境变量 AMAZON_OCR_DETAIL_TEXT_MODEL",
    )
    max_retries: int = Field(default=2, ge=1, le=5, description="图片下载重试次数")


def _bytes_to_array(raw: bytes) -> np.ndarray:
    if not raw:
        raise HTTPException(status_code=400, detail="图片内容为空")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"图片过大：{len(raw)} > {MAX_UPLOAD_BYTES} bytes",
        )
    try:
        with Image.open(io.BytesIO(raw)) as image:
            return np.array(image.convert("RGB"), copy=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"无法解析图片: {exc}") from exc


def _decode_base64(value: str) -> np.ndarray:
    payload = value.split(",", 1)[1] if value.startswith("data:") else value
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"base64 解码失败: {exc}") from exc
    return _bytes_to_array(raw)


def _do_ocr(
    image: Any,
    *,
    min_score: float,
    detail: bool | None,
    max_retries: int,
) -> dict:
    acquired = _request_lock.acquire(timeout=REQUEST_LOCK_TIMEOUT)
    if not acquired:
        raise HTTPException(status_code=503, detail="OCR 服务繁忙，请稍后重试")
    try:
        started = time.perf_counter()
        has_text, texts, success = processor.recognize_text(
            image,
            min_score=min_score,
            max_retries=max_retries,
            detail=detail,
        )
        texts = list(texts or [])
        scores = list(processor.last_scores)
        engine = processor.last_engine
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    finally:
        _request_lock.release()

    result = {
        "success": success,
        "has_text": bool(has_text),
        "texts": texts,
        "text": "\n".join(texts),
        "scores": scores,
        "engine": engine,
        "elapsed_ms": elapsed_ms,
    }
    if not success:
        result["hint"] = (
            "本地 OCR 不可用（初始化失败/超时/熔断冷却中）。"
            "可稍后重试、调用 /ocr/reset 清除冷却，或改用云端 OCR。"
        )
    return result


@app.get("/")
def root():
    return {
        "service": "OCR Service",
        "version": "1.1.0",
        "port": PORT,
        "workers": WORKERS,
        "ep": ocr_status()["ep"],
        "endpoints": ["/health", "/sample.png", "/ocr", "/ocr/upload", "/ocr/reset", "/docs"],
    }


@app.get("/health")
def health():
    status = ocr_status()
    status.update(
        {
            "status": "healthy",
            "presence_engine_ready": processor.presence_ocr is not None,
            "detail_engine_ready": processor.ocr is not None,
            "detail_model_enabled": processor.detail_model_enabled,
            "port": PORT,
            "workers": WORKERS,
            "pid": os.getpid(),
        }
    )
    return status


@app.get("/sample.png", response_class=Response)
def sample_png():
    return Response(content=make_sample_image(), media_type="image/png")


@app.post("/ocr")
def ocr_json(req: OCRRequest):
    """JSON 入参：image_url 或 image_base64 二选一。"""
    if req.image_url:
        image: Any = req.image_url
    elif req.image_base64:
        image = _decode_base64(req.image_base64)
    else:
        raise HTTPException(
            status_code=400, detail="必须提供 image_url 或 image_base64"
        )
    return _do_ocr(
        image,
        min_score=req.min_score,
        detail=req.detail,
        max_retries=req.max_retries,
    )


@app.post("/ocr/upload")
def ocr_upload(
    file: UploadFile = File(..., description="图片文件（jpg/png/webp...）"),
    min_score: float = Form(0.45, ge=0.0, le=1.0),
    detail: str = Form("", description="true/false，留空=跟随环境变量"),
    max_retries: int = Form(2, ge=1, le=5),
):
    """multipart 文件上传入参。"""
    raw = file.file.read(MAX_UPLOAD_BYTES + 1)
    image = _bytes_to_array(raw)
    detail_value: bool | None = None
    if detail.strip():
        detail_value = detail.strip().lower() in {"1", "true", "yes", "on"}
    return _do_ocr(
        image,
        min_score=min_score,
        detail=detail_value,
        max_retries=max_retries,
    )


@app.post("/ocr/reset")
def ocr_reset():
    """清除本地 OCR 熔断冷却状态（不重启进程即可恢复）。"""
    clear_ocr_engine_cache()
    return {"reset": True, **ocr_status()}


@app.on_event("startup")
def _startup():
    if PREWARM:
        def _warm():
            logger.info("Warming up presence OCR engine in background ...")
            processor.warm_up()

        threading.Thread(target=_warm, name="ocr-prewarm", daemon=True).start()
    if WORKER_MAX_COMMIT_GB > 0:
        threading.Thread(
            target=_memory_recycle_loop, name="ocr-mem-recycle", daemon=True
        ).start()
    logger.info(
        "OCR worker pid=%s listening on %s:%s (workers=%s)",
        os.getpid(), HOST, PORT, WORKERS,
    )


if __name__ == "__main__":
    import uvicorn

    # Windows spawns worker processes, which must be able to import the app by
    # name; passing the ``app`` object is rejected when workers > 1.
    app_target = "server:app" if WORKERS > 1 else app
    uvicorn.run(
        app_target,
        host=HOST,
        port=PORT,
        workers=WORKERS,
        reload=False,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
