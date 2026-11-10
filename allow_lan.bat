@echo off
chcp 65001 >nul
REM Allow same-WiFi devices (e.g. laptop) to reach the OCR service on TCP 6575.
REM Double-click and accept the UAC prompt (admin rights are required for firewall rules).
cd /d "%~dp0"
net session >nul 2>&1
if errorlevel 1 (
  echo Requesting administrator privileges, please click "Yes" on the UAC prompt...
  powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
  exit /b
)

powershell -NoProfile -ExecutionPolicy Bypass -Command "& { $n = 'OCR Service TCP 6575'; Get-NetFirewallRule -DisplayName $n -ErrorAction SilentlyContinue | Remove-NetFirewallRule; New-NetFirewallRule -DisplayName $n -Direction Inbound -Protocol TCP -LocalPort 6575 -Action Allow -Profile Any -RemoteAddress LocalSubnet -Description 'Allow LAN clients to reach OCR API' | Out-Null; $r = Get-NetFirewallRule -DisplayName $n; Write-Host ('Rule: ' + $r.DisplayName + ' | Enabled=' + $r.Enabled + ' | Action=' + $r.Action); Write-Host '--- LAN URLs ---'; Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -ne '127.0.0.1' -and $_.IPAddress -notlike '169.*' } | ForEach-Object { Write-Host ('  http://' + $_.IPAddress + ':6575/health') } }"

echo.
pause
