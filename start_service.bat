@echo off
chcp 65001 >nul
REM Start the OCR service (GPU / CUDA, 8 workers) on 0.0.0.0:6575.
REM Override with PORT / HOST / WORKERS / OCR_EP env vars before launching.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] 未找到虚拟环境 .venv，请先执行：
  echo   python -m venv .venv
  echo   .venv\Scripts\python.exe -m pip install -r requirements.txt -i https://repo.huaweicloud.com/repository/pypi/simple
  pause
  exit /b 1
)

if "%PORT%"=="" set PORT=6575
if "%HOST%"=="" set HOST=0.0.0.0
if "%WORKERS%"=="" set WORKERS=8
if "%OCR_EP%"=="" set OCR_EP=cuda
echo OCR service starting at http://%HOST%:%PORT%  (workers=%WORKERS%, ep=%OCR_EP%, docs: /docs)
".venv\Scripts\python.exe" server.py
pause
