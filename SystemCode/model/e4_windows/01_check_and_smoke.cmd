@echo off
setlocal
cd /d "%~dp0.."
python e4_windows\test_e4.py
if errorlevel 1 exit /b 1
python e4_windows\core\test_e2.py
if errorlevel 1 exit /b 1
python e4_windows\run_e4.py check-gpu
if errorlevel 1 exit /b 1
python e4_windows\run_e4.py smoke
if errorlevel 1 exit /b 1
endlocal
