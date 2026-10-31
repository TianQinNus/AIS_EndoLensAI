@echo off
setlocal
cd /d "%~dp0.."
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
python -u e12_negative_windows\run_e12.py smoke
if errorlevel 1 exit /b 1
python -u e12_negative_windows\run_e12.py preflight
exit /b %ERRORLEVEL%
