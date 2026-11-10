"""Self-contained image loading helpers.

The original pipeline imported these from ``gma_runtime.image_assets``. This
standalone service ships a minimal equivalent so it has no dependency on the
listing project: HTTP(S) URL or local path -> PIL image context manager.
"""
from __future__ import annotations

import io
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import requests
from PIL import Image

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
)


@contextmanager
def load_image(
    source: str,
    *,
    timeout: float = 15,
    headers: dict | None = None,
) -> Iterator[Image.Image]:
    """Yield a PIL image loaded from a URL or a local file, closing it on exit."""
    if str(source).startswith(("http://", "https://")):
        merged = {"User-Agent": _USER_AGENT, "Accept": "image/*,*/*"}
        if headers:
            merged.update(headers)
        response = requests.get(source, timeout=timeout, headers=merged)
        response.raise_for_status()
        payload = response.content
    else:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"image asset not found: {source}")
        payload = path.read_bytes()

    image = Image.open(io.BytesIO(payload))
    image.load()
    try:
        yield image
    finally:
        image.close()


def get_image_asset(name: str) -> str:
    """Resolve a bundled asset (or an OCR_ASSETS_DIR override) to a path."""
    root = Path(os.getenv("OCR_ASSETS_DIR", str(Path(__file__).resolve().parent.parent / "assets")))
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(f"image asset not found: {path}")
    return str(path)
