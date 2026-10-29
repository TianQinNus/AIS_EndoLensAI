@echo off
setlocal
cd /d "%~dp0.."
python e2_windows\check_device.py --device cuda --output runs\e2_gpu_device_proof.json
if errorlevel 1 exit /b 1
python e2_windows\test_e2.py
if errorlevel 1 exit /b 1
python e2_windows\run_e2.py download-weights
if errorlevel 1 exit /b 1
python e2_windows\run_e2.py train --experiment pretrained --device cuda --smoke --output runs\e2_smoke_C_seed42
if errorlevel 1 exit /b 1
echo GPU SMOKE COMPLETE. This is not a formal experiment.
