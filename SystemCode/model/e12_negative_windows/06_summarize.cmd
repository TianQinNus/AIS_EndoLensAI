@echo off
setlocal
cd /d "%~dp0.."
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
python -u e12_negative_windows\run_e12.py report
exit /b %ERRORLEVEL%
