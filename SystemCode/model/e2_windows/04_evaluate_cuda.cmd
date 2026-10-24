@echo off
setlocal
cd /d "%~dp0.."
python e2_windows\run_e2.py evaluate --device cuda --run-dir runs\e2_final_seed42 --e0-run runs\e0_20261004_221345_872444
if errorlevel 1 exit /b 1
