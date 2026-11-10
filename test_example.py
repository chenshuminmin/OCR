"""OCR 服务测试示例 —— 只依赖 requests，可在本机或同 WiFi 的笔记本上直接运行。

用法：
    python test_example.py                                  # 默认 http://127.0.0.1:6575
    python test_example.py http://192.168.1.14:6575         # 从笔记本访问
    python test_example.py http://192.168.1.14:6575 图片.jpg  # 额外测试自己的图片

覆盖 3 种入参方式：image_url / image_base64 / multipart 文件上传。
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import requests

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:6575").rstrip("/")
LOCAL_IMAGE = sys.argv[2] if len(sys.argv) > 2 else None


def show(title: str, response: requests.Response) -> None:
    print(f"\n===== {title} =====")
    print("HTTP", response.status_code)
    try:
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
    except Exception:
        print(response.text[:500])


def main() -> None:
    session = requests.Session()

    # 0) 健康检查
    show("GET /health", session.get(f"{BASE}/health", timeout=30))

    # 1) 取服务端内置样例图（白底黑字，含中文+英文）
    sample = session.get(f"{BASE}/sample.png", timeout=30)
    sample.raise_for_status()
    image_bytes = sample.content
    print(f"\n样例图大小: {len(image_bytes)} bytes")

    # 2) JSON 入参 image_url —— 让服务端自己下载图片
    show(
        "POST /ocr  {image_url}",
        session.post(
            f"{BASE}/ocr",
            json={
                "image_url": f"{BASE}/sample.png",
                "min_score": 0.45,
                "max_retries": 2,
            },
            timeout=120,
        ),
    )

    # 3) JSON 入参 image_base64 —— 客户端上传图内容
    show(
        "POST /ocr  {image_base64}",
        session.post(
            f"{BASE}/ocr",
            json={
                "image_base64": base64.b64encode(image_bytes).decode("ascii"),
                "min_score": 0.35,
            },
            timeout=120,
        ),
    )

    # 4) multipart 文件上传（支持 detail / min_score / max_retries 表单字段）
    upload_bytes = Path(LOCAL_IMAGE).read_bytes() if LOCAL_IMAGE else image_bytes
    filename = Path(LOCAL_IMAGE).name if LOCAL_IMAGE else "sample.png"
    show(
        f"POST /ocr/upload  file={filename}",
        session.post(
            f"{BASE}/ocr/upload",
            files={"file": (filename, upload_bytes, "image/png")},
            data={"min_score": "0.45", "detail": "", "max_retries": "2"},
            timeout=120,
        ),
    )

    # 5) 等价 curl 命令
    print("\n===== 等价 curl 命令 =====")
    print('curl -X POST "%s/ocr" -H "Content-Type: application/json" '
          '-d "{\\"image_url\\": \\"%s/sample.png\\"}"' % (BASE, BASE))
    print('curl -X POST "%s/ocr/upload" -F "file=@1.jpg" -F "min_score=0.45"' % BASE)


if __name__ == "__main__":
    main()
