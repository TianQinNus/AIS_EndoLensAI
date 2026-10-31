@echo off
setlocal
cd /d "%~dp0.."
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
echo Secondary comparison only. All seven models use the same four rotations and 352 input.
python -u e12_negative_windows\run_e12.py evaluate-test --tta rotate4
if errorlevel 1 exit /b 1
python -u e12_negative_windows\run_e12.py report
exit /b %ERRORLEVEL%
