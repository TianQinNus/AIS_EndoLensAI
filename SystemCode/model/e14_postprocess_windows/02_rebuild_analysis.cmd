@echo off
setlocal
cd /d "%~dp0.."
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
python -u e14_postprocess_windows\run_e14.py analyze
exit /b %ERRORLEVEL%
