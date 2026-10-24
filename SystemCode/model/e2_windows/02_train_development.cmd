@echo off
setlocal
cd /d "%~dp0.."
python e2_windows\check_device.py --device cuda --output runs\e2_gpu_device_proof.json
if errorlevel 1 exit /b 1
python e2_windows\run_e2.py train --experiment original --device cuda --output runs\e2_dev_A_original_seed42
if errorlevel 1 exit /b 1
python e2_windows\run_e2.py train --experiment random --device cuda --output runs\e2_dev_B_resnet_random_seed42
if errorlevel 1 exit /b 1
python e2_windows\run_e2.py train --experiment pretrained --device cuda --output runs\e2_dev_C_resnet_pretrained_seed42
if errorlevel 1 exit /b 1
echo ALL THREE DEVELOPMENT EXPERIMENTS COMPLETE. Return runs to Mac for analysis.
