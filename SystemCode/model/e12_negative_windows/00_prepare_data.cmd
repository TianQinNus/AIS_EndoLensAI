@echo off
setlocal
cd /d "%~dp0.."
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
python -u e12_negative_windows\run_e12.py prepare
set "RUN_EXIT=%ERRORLEVEL%"
if not "%RUN_EXIT%"=="0" echo E12 preparation failed. Preserve existing files and inspect the error.
exit /b %RUN_EXIT%
