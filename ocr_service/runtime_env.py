"""GPU runtime bootstrap for the OCR service.

This module MUST run before ``import onnxruntime`` / ``import rapidocr``.

On Windows, ONNX Runtime loads ``onnxruntime_providers_cuda.dll`` lazily when a
session is created. That DLL needs the CUDA 12 runtime libraries
(``cublas64_12.dll``, ``cublasLt64_12.dll``, ``cudart64_12.dll``,
``cudnn64_9.dll``, ``cufft64_11.dll``) on the DLL search path. If any of them is
missing, ORT does NOT fail: it prints a warning and silently falls back to
``CPUExecutionProvider``. That is why the service verifies both the advertised
providers and the *actual session providers* (see ocr_engines.py).

The DLLs are taken from, in priority order:
  1. ``OCR_CUDA_DLL_DIR`` (one or more dirs separated by ``;``)
  2. ``nvidia/*/bin`` and ``torch/lib`` inside this venv
  3. ``torch/lib`` of a sibling project's venv (e.g. the RMBG service's
     PyTorch wheel, which bundles CUDA 12.4 + cuDNN 9)
  4. A locally installed CUDA toolkit ``.../CUDA/v*/bin``

Set ``OCR_EP=dml`` to use DirectML instead of CUDA (no CUDA DLLs needed), or
``OCR_EP=cpu`` together with ``OCR_REQUIRE_GPU=0`` for a deliberate CPU fallback.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

EP = os.getenv("OCR_EP", "cuda").strip().lower()
if EP not in {"cuda", "dml", "cpu"}:
    raise RuntimeError(f"OCR_EP must be cuda|dml|cpu, got {EP!r}")

REQUIRE_GPU = os.getenv("OCR_REQUIRE_GPU", "1").strip().lower() in {
    "1", "true", "yes", "on"
}
try:
    GPU_DEVICE_ID = int(os.getenv("OCR_GPU_DEVICE_ID", "0"))
except ValueError:
    GPU_DEVICE_ID = 0

_EXPECTED_PROVIDER = {
    "cuda": "CUDAExecutionProvider",
    "dml": "DmlExecutionProvider",
}

_configured = False
_info: dict[str, Any] = {}


def _candidate_dll_dirs() -> list[Path]:
    found: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        try:
            key = str(path.resolve()).lower()
        except OSError:
            return
        if path.is_dir() and key not in seen:
            seen.add(key)
            found.append(path)

    for chunk in os.getenv("OCR_CUDA_DLL_DIR", "").split(os.pathsep):
        if chunk.strip():
            add(Path(chunk.strip()))

    site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    for pattern in ("torch/lib", "nvidia/*/bin"):
        for path in sorted(site_packages.glob(pattern)):
            add(path)

    # Sibling projects in the same workspace: H:/AI/<proj>/.venv/... and
    # H:/AI/<group>/<proj>/.venv/...
    workspace = Path(__file__).resolve().parents[2]
    for pattern in ("*/.venv/Lib/site-packages/torch/lib",
                    "*/*/.venv/Lib/site-packages/torch/lib",
                    "*/Lib/site-packages/torch/lib"):
        for path in sorted(workspace.glob(pattern)):
            add(path)

    cuda_root = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
    for path in sorted(cuda_root.glob("v*/bin")):
        add(path)

    return found


def configure() -> dict[str, Any]:
    """Add CUDA DLL directories, then import and validate ONNX Runtime."""
    global _configured, _info
    if _configured:
        return _info
    _configured = True

    dll_dirs: list[Path] = []
    if os.name == "nt" and EP == "cuda":
        dll_dirs = _candidate_dll_dirs()
        for path in dll_dirs:
            try:
                os.add_dll_directory(str(path))
            except (OSError, AttributeError):
                pass
        if dll_dirs:
            os.environ["PATH"] = (
                os.pathsep.join(str(p) for p in dll_dirs)
                + os.pathsep
                + os.environ.get("PATH", "")
            )
            logger.info("CUDA DLL search dirs: %s", [str(p) for p in dll_dirs])

    import onnxruntime as ort

    _info = {
        "ep": EP,
        "ort_version": ort.__version__,
        "ort_device": ort.get_device(),
        "available_providers": list(ort.get_available_providers()),
        "gpu_device_id": GPU_DEVICE_ID,
        "require_gpu": REQUIRE_GPU,
        "dll_dirs": [str(p) for p in dll_dirs],
    }

    if EP == "cpu":
        if REQUIRE_GPU:
            raise RuntimeError(
                "OCR_EP=cpu but OCR_REQUIRE_GPU=1. Set OCR_REQUIRE_GPU=0 to "
                "explicitly allow CPU inference."
            )
        logger.warning("OCR running on CPU by explicit configuration")
        return _info

    expected = _EXPECTED_PROVIDER[EP]
    problems = []
    if expected not in _info["available_providers"]:
        problems.append(
            f"{expected} missing from {_info['available_providers']} "
            f"(install onnxruntime-gpu / onnxruntime-directml)"
        )
    if EP == "cuda" and _info["ort_device"] != "GPU":
        problems.append(f"onnxruntime.get_device()={_info['ort_device']!r}, expected GPU")

    if problems and REQUIRE_GPU:
        raise RuntimeError(
            "GPU inference is required but unavailable: " + "; ".join(problems)
        )
    if problems:
        logger.warning("GPU inference unavailable (continuing on CPU): %s", problems)

    logger.info(
        "OCR runtime: ep=%s ort=%s device=%s providers=%s",
        EP, _info["ort_version"], _info["ort_device"], _info["available_providers"],
    )
    return _info


def runtime_info() -> dict[str, Any]:
    """Runtime details for /health; never raises after a successful configure()."""
    return dict(configure())


def expected_provider() -> str | None:
    return _EXPECTED_PROVIDER.get(EP)


def engine_params(base: dict[str, Any]) -> dict[str, Any]:
    """Inject the GPU execution-provider config into RapidOCR params."""
    params = dict(base)
    if EP == "cuda":
        params["EngineConfig.onnxruntime.use_cuda"] = True
        params["EngineConfig.onnxruntime.cuda_ep_cfg.device_id"] = GPU_DEVICE_ID
        # EXHAUSTIVE (rapidocr default) benchmarks every conv algorithm on the
        # first run and is slow / memory hungry, especially with 8 workers.
        params["EngineConfig.onnxruntime.cuda_ep_cfg.cudnn_conv_algo_search"] = "DEFAULT"
        params["EngineConfig.onnxruntime.cuda_ep_cfg.arena_extend_strategy"] = "kSameAsRequested"
        # A per-arena cap keeps runaway allocations bounded, but it must be
        # generous: the BFC arena never returns memory to CUDA, and large
        # (~2500px) detection inputs need several hundred MB. A 512MB cap
        # made every OCR call return empty once the arena filled up
        # (BFCArena::AllocateRawInternal "available < requested"), while the
        # GPU itself still had >11GB free. 2GB per arena is a safe ceiling
        # for 4 workers on a 16GB GPU.
        params["EngineConfig.onnxruntime.cuda_ep_cfg.gpu_mem_limit"] = (
            int(os.getenv("OCR_GPU_MEM_LIMIT_MB", "2048")) * 1024 * 1024
        )
        params["EngineConfig.onnxruntime.cuda_ep_cfg.cudnn_conv_use_max_workspace"] = 0
    elif EP == "dml":
        params["EngineConfig.onnxruntime.use_dml"] = True
        params["EngineConfig.onnxruntime.dml_ep_cfg.device_id"] = GPU_DEVICE_ID
    return params
