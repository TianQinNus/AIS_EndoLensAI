@echo off
setlocal
cd /d "%~dp0.."
python v3_windows\run_v3.py --output "runs\v3_fusion_equal_seed42"
if errorlevel 1 exit /b 1
endlocal
