@echo off
setlocal
cd /d "%~dp0.."
python e4_windows\run_e4.py train
if errorlevel 1 exit /b 1
endlocal
