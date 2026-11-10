"""Generate a small image containing text for /sample.png and the test client."""
from __future__ import annotations

import io
import os

from PIL import Image, ImageDraw, ImageFont

_FONT_CANDIDATES = (
    ("C:/Windows/Fonts/msyh.ttc", True),
    ("C:/Windows/Fonts/simhei.ttf", True),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", True),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", False),
)


def _pick_font(size: int):
    for path, has_cjk in _FONT_CANDIDATES:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size), has_cjk
            except OSError:
                continue
    return ImageFont.load_default(), False


def make_sample_image() -> bytes:
    """Return PNG bytes: white background with three lines of black text."""
    font, has_cjk = _pick_font(34)
    lines = (
        ("新品上市 限时优惠", "OCR 服务测试 2026", "Hello World 12345")
        if has_cjk
        else ("OCR SERVICE TEST", "Hello World 2026", "Min score 0.45")
    )
    image = Image.new("RGB", (640, 220), "white")
    draw = ImageDraw.Draw(image)
    for index, line in enumerate(lines):
        draw.text((24, 18 + index * 66), line, fill=(0, 0, 0), font=font)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
