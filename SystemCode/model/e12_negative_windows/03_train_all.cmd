@echo off
setlocal
cd /d "%~dp0.."
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
echo Four controlled runs will execute sequentially. Do not start another E12 training window.
python -u e12_negative_windows\run_e12.py train-all --workers 2
exit /b %ERRORLEVEL%
