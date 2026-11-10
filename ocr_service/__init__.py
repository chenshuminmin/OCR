"""Standalone OCR service package (RapidOCR, GPU inference by default)."""
from __future__ import annotations

# Importing this package configures the GPU runtime BEFORE anything else can
# import onnxruntime / rapidocr. On Windows the CUDA provider DLL is resolved
# lazily, so the CUDA runtime DLLs must be on the search path first.
from .runtime_env import configure

configure()
