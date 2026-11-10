@echo off
chcp 65001 >nul
REM Start the OCR service on 0.0.0.0:6575 (override with PORT/HOST env vars).
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
echo OCR service starting at http://%HOST%:%PORT%  (docs: /docs)
".venv\Scripts\python.exe" server.py
pause
