@echo off
setlocal
title PINN Space Debris Dashboard
cd /d "%~dp0"

set "PINN_DASHBOARD_PYTHON=%~dp0.venv\Scripts\python.exe"
set "PINN_DASHBOARD_SCRIPT=%~dp0dashboard_server.py"

if not exist "%PINN_DASHBOARD_PYTHON%" (
    echo ERROR: The project Python environment is missing.
    echo Create .venv and install requirements-lock.txt as described in README.md.
    pause
    exit /b 1
)
if not exist "%PINN_DASHBOARD_SCRIPT%" (
    echo ERROR: dashboard_server.py is missing. Keep this BAT file in the project folder.
    pause
    exit /b 1
)

echo Dashboard: http://127.0.0.1:8787/
echo Starting the dashboard. Your browser will open when it is ready.
echo Keep this window open while using the dashboard. Press Ctrl+C to stop it.
echo.

powershell.exe -NoLogo -NoProfile -Command ^
  "$ErrorActionPreference = 'Stop'; $dashboardUrl = 'http://127.0.0.1:8787/'; $page = $null;" ^
  "try { $page = Invoke-WebRequest -Uri $dashboardUrl -UseBasicParsing -TimeoutSec 2 } catch {}" ^
  "if ($null -ne $page) { if ($page.Content -notmatch '<title>Orbital Watch') { Write-Host 'ERROR: Port 8787 is being used by a different application.'; exit 1 }; Write-Host 'Dashboard is already running. Opening it.'; Start-Process $dashboardUrl; exit 0 };" ^
  "$browserJob = Start-Job -ArgumentList $dashboardUrl -ScriptBlock { param($url) for ($attempt = 0; $attempt -lt 120; $attempt++) { try { $ready = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 1; if ($ready.Content -match '<title>Orbital Watch') { Start-Process $url; return } } catch {}; Start-Sleep -Milliseconds 250 } };" ^
  "try { & $env:PINN_DASHBOARD_PYTHON $env:PINN_DASHBOARD_SCRIPT; $serverExit = $LASTEXITCODE } finally { Stop-Job -Job $browserJob -ErrorAction SilentlyContinue; Remove-Job -Job $browserJob -Force -ErrorAction SilentlyContinue }; exit $serverExit"

set "PINN_DASHBOARD_EXIT=%ERRORLEVEL%"
if not "%PINN_DASHBOARD_EXIT%"=="0" (
    echo.
    echo Dashboard startup stopped. Check the error message above.
    pause
)
exit /b %PINN_DASHBOARD_EXIT%
