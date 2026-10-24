@echo off
setlocal
cd /d "%~dp0.."
python e2_windows\run_e2.py select --runs runs\e2_dev_A_original_seed42 runs\e2_dev_B_resnet_random_seed42 runs\e2_dev_C_resnet_pretrained_seed42
if errorlevel 1 exit /b 1
python e2_windows\run_e2.py train-final --device cuda --output runs\e2_final_seed42
if errorlevel 1 exit /b 1
echo FINAL TRAINING COMPLETE. Evaluate on CUDA or return final run to Mac.
