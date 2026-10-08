@echo off
setlocal
cd /d "%~dp0.."
python v3_windows\test_v3.py
if errorlevel 1 exit /b 1
python v3_windows\run_v3.py --audit-only --output "runs\v3_audit_seed42"
if errorlevel 1 exit /b 1
python v3_windows\run_v3.py --smoke --output "runs\v3_smoke_seed42"
if errorlevel 1 exit /b 1
endlocal
