@echo off
setlocal
cd /d "%~dp0.."
python e4_windows\run_e4.py compare
if errorlevel 1 exit /b 1
echo REVIEW DEVELOPMENT RESULTS BEFORE FINAL RETRAINING.
endlocal
