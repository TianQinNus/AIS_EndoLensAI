@echo off
setlocal
cd /d "%~dp0.."
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
python -u e13_joint_windows\run_e13.py evaluate --arm mixed50 --tta rotate4
exit /b %ERRORLEVEL%
