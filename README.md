# OCR Service

基于 **RapidOCR PP-OCRv6**（本地 ONNX，CPU 推理，模型随 pip 包自带、完全离线）的图片文字识别 HTTP 服务。
引擎层移植自 Amazon 上品链路的 `ocr_engines.py` / `OCRProcessor`，保留超时保护、首次推理预热、串行化推理、可恢复熔断。

- **对外端口：`6575`**（监听 `0.0.0.0`，`PORT` 环境变量可改）
- 本机：`http://127.0.0.1:6575` ｜ 同 WiFi 笔记本：`http://192.168.1.14:6575`
- 在线接口文档：`http://127.0.0.1:6575/docs`

---

## 1. 接口与入参

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | 服务信息 |
| GET | `/health` | 健康检查 + 引擎/熔断状态 |
| GET | `/sample.png` | 内置测试图（白底黑字，中英文），供 URL 入参直接测试 |
| POST | `/ocr` | JSON 入参 |
| POST | `/ocr/upload` | multipart 文件上传 |
| POST | `/ocr/reset` | 清除本地 OCR 熔断冷却（不重启即可恢复） |

### `POST /ocr`（`Content-Type: application/json`）

| 字段 | 类型 | 必填 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `image_url` | string | 二选一 | — | HTTP(S) 图片地址，服务端自动下载、限制最大边 2500px |
| `image_base64` | string | 二选一 | — | 图片 base64，可带 `data:image/png;base64,` 前缀 |
| `min_score` | float | 否 | `0.45` | 文本置信度阈值（0~1），仅对 detail 模型生效 |
| `detail` | bool\|null | 否 | `null` | 是否启用高精度模型；不传跟随环境变量 `AMAZON_OCR_DETAIL_TEXT_MODEL`（默认关闭，省内存） |
| `max_retries` | int | 否 | `2` | `image_url` 下载重试次数（1~5） |

> `image_url` 与 `image_base64` **二选一**；都不传返回 400。

### `POST /ocr/upload`（`multipart/form-data`）

| 字段 | 类型 | 必填 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `file` | file | 是 | — | jpg/png/webp/bmp 等，上限 20MB（`OCR_MAX_UPLOAD_BYTES`） |
| `min_score` | float | 否 | `0.45` | 同上 |
| `detail` | string | 否 | `""` | `true`/`false`；留空 = 跟随环境变量 |
| `max_retries` | int | 否 | `2` | 同上 |

### 返回（统一 JSON）

```json
{
  "success": true,
  "has_text": true,
  "texts": ["新品上市限时优惠", "OCR 服务测试 2026", "Hello World 12345"],
  "text": "新品上市限时优惠\nOCR 服务测试 2026\nHello World 12345",
  "scores": [0.99976, 0.99218, 0.99972],
  "engine": "presence",
  "elapsed_ms": 187.4
}
```

| 字段 | 含义 |
| --- | --- |
| `success` | 本地 OCR 是否成功执行；`false` = 初始化失败/超时/熔断冷却中，应走云端 OCR 兜底 |
| `has_text` | 图片中是否检出达标文字（对应原链路的"是否需要翻译"判断） |
| `texts` / `text` | 识别文本列表 / 换行拼接文本 |
| `scores` | 各文本置信度 |
| `engine` | 实际使用的引擎：`presence`（轻量）/ `detail`（高精度）/ `null`（失败） |
| `elapsed_ms` | 单次耗时；`success=false` 时附带 `hint` 提示 |

> `success` / `has_text` / `texts` 一一对应原 `OCRProcessor.recognize_text()` 的三元组 `(has_text, texts, success)`。

---

## 2. 启动 / 停止

```bat
:: 双击 start_service.bat，或在项目根目录执行
.venv\Scripts\python.exe server.py
```

首次部署：

```bat
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt -i https://repo.huaweicloud.com/repository/pypi/simple
```

停止：在服务窗口按 `Ctrl+C`。服务单进程单 worker，模型只加载一份。

### 环境变量

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `PORT` / `HOST` | `6575` / `0.0.0.0` | 监听地址 |
| `AMAZON_OCR_DETAIL_TEXT_MODEL` | `0` | 全局默认是否启用高精度模型 |
| `AMAZON_OCR_ENGINE_TIMEOUT_SECONDS` | `15` | 单图推理超时 |
| `AMAZON_OCR_LOCK_WAIT_SECONDS` | `5` | 推理锁等待 |
| `AMAZON_OCR_INITIALIZATION_TIMEOUT_SECONDS` | `90` | 引擎初始化超时 |
| `AMAZON_OCR_WARMUP_TIMEOUT_SECONDS` / `..._GRACE_SECONDS` | `180` / `120` | 预热预算 / 首图宽限窗口 |
| `AMAZON_OCR_DISABLE_COOLDOWN_SECONDS` | `600` | 硬失败后本地 OCR 暂停时长（自动恢复） |
| `AMAZON_OCR_MAX_IMAGE_SIDE` | `2500` | 图片最大边，超过自动缩放 |
| `OCR_MAX_UPLOAD_BYTES` | `20971520` | 上传大小上限 |
| `OCR_PREWARM` | `1` | 启动时后台预热引擎 |
| `LOG_LEVEL` | `INFO` | 日志级别 |

---

## 3. Test 示例

### 一键自测（覆盖 URL / base64 / 文件上传三种入参）

```bat
.venv\Scripts\python.exe test_example.py
:: 从笔记本跑（同一 WiFi）
python test_example.py http://192.168.1.14:6575
:: 测自己的图片
python test_example.py http://192.168.1.14:6575 D:\pics\1.jpg
```

### curl

```bash
# 健康检查
curl http://127.0.0.1:6575/health

# ① URL 入参（用内置样例图）
curl -X POST http://127.0.0.1:6575/ocr \
  -H "Content-Type: application/json" \
  -d "{\"image_url\": \"http://127.0.0.1:6575/sample.png\", \"min_score\": 0.45}"

# ② 文件上传
curl -X POST http://127.0.0.1:6575/ocr/upload \
  -F "file=@1.jpg" -F "min_score=0.45" -F "max_retries=2"

# ③ 熔断恢复
curl -X POST http://127.0.0.1:6575/ocr/reset
```

### Python（业务接入：判断图片是否含文字 → 决定是否翻译）

```python
import requests

BASE = "http://192.168.1.14:6575"

def need_translation(image_url: str) -> tuple[bool, list[str]]:
    r = requests.post(
        f"{BASE}/ocr",
        json={"image_url": image_url, "min_score": 0.45, "max_retries": 2},
        timeout=60,
    ).json()
    if not r["success"]:
        # 本地 OCR 挂了 → 这里接云端 OCR / 翻译兜底
        return False, []
    return r["has_text"], r["texts"]

has_text, texts = need_translation("https://example.com/product.jpg")
print(has_text, texts)
```

---

## 4. 同 WiFi 笔记本访问

服务已监听 `0.0.0.0`。Windows 防火墙默认拦截入站连接，**双击 `allow_lan.bat` 并在 UAC 弹窗点「是」** 即可放行 TCP 6575（仅本地子网，不影响其他端口）。
访问地址 `http://192.168.1.14:6575/docs`；若 IP 变化，在主机执行 `ipconfig` 查 WLAN 网卡 IPv4。

**连不上的排查顺序：** ① `ping 192.168.1.14` 不通 → 路由器 AP 隔离/访客网络；② 主机 `netstat -ano | findstr :6575` 无 LISTEN → 服务没起，跑 `start_service.bat`；③ 第三方杀软自带防火墙也要放行 6575；④ 确认笔记本连的是 `801-5G` 同一个网段。

---

## 5. 实现说明

```
server.py                     FastAPI 入口，端口 6575，请求级串行锁
ocr_service/
  image_assets.py             图片加载（替换原 gma_runtime.image_assets）
  ocr_engines.py              引擎创建/超时/预热/熔断/文本提取（原逻辑）
  processor.py                OCRProcessor（懒加载、presence→detail）
  sample_image.py             生成内置测试图
test_example.py               测试示例
requirements.txt / start_service.bat / allow_lan.bat
```

相对原代码的两处必要改动：
1. 原 `OCRProcessor` 超时后把 `_ocr_unavailable` 永久置 `True`，一次抖动就导致本地 OCR 直到进程重启都不可用；改为只让本次调用失败，由引擎层的**可恢复冷却**决定何时重试（也可 `POST /ocr/reset` 立即恢复）。
2. `detail` 改为可按请求覆盖，并在响应中返回 `engine` / `scores`，便于调用方判断是否走了高精度模型。

`model_type=MEDIUM`（detail 模型）不在 rapidocr wheel 内，首次启用需联网下载；默认关闭，只用自带的 `SMALL` 模型（31.7MB，完全离线）。若 `detail=true` 因下载超时触发引擎熔断（本地 OCR 暂停 600s），调用 `POST /ocr/reset` 即可立即恢复，无需重启进程。
