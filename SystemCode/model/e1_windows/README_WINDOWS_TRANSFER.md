# E1 从 Mac 迁移到 Windows 并返回 Mac

本包的 `e1_windows/` 同时可在 Windows CUDA 与 Mac CPU/MPS 上使用。它保留原 E1 模型与训练设置，只修改相对路径保存方式和迁移后的 E0 本地路径解析。原 `e1_mac/` 不需要覆盖；保留它可恢复先前 Mac 运行。

## 1 先停止 Mac 训练

在实际运行训练的 VS Code 终端内点击一下，再按 Control+C（不是 Command+C），等待回到 shell 提示符。脚本会尽力记录 INTERRUPTED；每轮保存的 `checkpoints/last.pth` 仍在。当前未完成轮次不保存为完成轮次。

Kill Terminal / 垃圾桶按钮会关闭终端，并通常终止它的前台训练进程，但不会保证脚本有机会更新运行状态；隐藏底部终端面板、切换 tab 或关闭终端视图不等于终止。后台启动或脱离终端的进程应单独核查。

在新的 Mac 终端检查：

```bash
pgrep -fl 'run_e1.py.*train'
```

无输出表示没有匹配的训练进程；若仍有输出，在活动监视器查看对应 PID。不要依靠可能陈旧的 `run_info.json` 状态判断进程是否仍存在。

保留 `runs/e1_unet_mps_seed42/`。Windows E1 从 seed 42 开始新运行，不直接 resume Mac 的 MPS 检查点，以保持当前恢复规则和实验记录一致。

## 2 Mac 复制到 Windows

以 Windows `D:\Projects\Polyp-PVT\` 为工程根目录。路径可以自行更换，Python 无需改源码。

必须复制：

```text
Polyp-PVT/
  dataset/
    TrainDataset/                     # images + masks，全量1450对
    TestDataset/                      # 整个目录，包括test合并集与五个来源子集
  runs/
    e0_20261004_221345_872444/
      run_info.json
      manifest.json
      per_image.csv
      audit.json
      summary.csv
      metrics.json
  e1_windows/                         # 本代码包完整解压出的文件夹
```

E1 训练/评估真正读取的 E0 文件为 `run_info.json`、`manifest.json`、`per_image.csv`，其余三个保留用于实验追溯。不要编辑 manifest 或重编码数据图像；哈希必须与 Mac E0 一致。

现阶段无需复制：原始 Polyp-PVT `lib/`、`model_pth/`、`pretrained_pth/`、`Train.py`、`Test.py`、`result_map/`、`Figs/`、旧可视化、Mac conda 环境或 Mac E1 部分训练目录。E0 的 `probabilities/`、`masks/`、`legacy_minmax/` 不被 E1 读取，本阶段也可不复制；Mac 上已有这些文件应保留，后续融合会用到 E0 的原始概率。

如果想省去筛选，可以直接复制整个项目，再添加 `e1_windows/`，但文件更大；仍需独立安装 Windows 环境。

## 3 Windows CUDA 环境

更新 NVIDIA 驱动到支持 RTX 5060 的稳定版。推荐原生 Windows Python 3.11，独立 conda 环境。可以在 Anaconda Prompt 或已配置 conda 的 VS Code PowerShell 内运行：

```powershell
cd "D:\Projects\Polyp-PVT"
conda create -n endolens-e1-win python=3.11 -y
conda activate endolens-e1-win
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r e1_windows/requirements-windows.txt
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
python e1_windows/check_device.py --device cuda
python -m pip freeze > e1_windows/environment-windows.txt
```

上述 PyTorch 2.9.1 / CUDA 12.8 配对来自 [PyTorch 官方历史版本安装说明](https://pytorch.org/get-started/previous-versions/#v291)。安装该 CUDA wheel 无需为本包另外编译 CUDA 源码或安装完整 CUDA Toolkit，但需要兼容的 NVIDIA 驱动。不要使用 E0 的 torch 2.2.2 运行 RTX 50 系列。

设备探针应报告 cuda PASS，GPU 名称含 RTX 5060。它使用小模型；实际内存还需标准 smoke 检查。GPU 驱动、CUDA torch 版本及运行环境均记录到 run_info.json。这个安装版本组合尚未在此交付环境的 Windows GPU 上实测。

## 4 Windows 先检查，再正式训练

全部使用 PowerShell/Anaconda Prompt 单行命令，避免复制 Bash 的反斜杠续行语法。

```powershell
python e1_windows/test_e1.py
python e1_windows/run_e1.py train --device cuda --e0-run "runs/e0_20261004_221345_872444" --smoke
```

如果 smoke 成功，正式训练：

```powershell
python e1_windows/run_e1.py train --device cuda --e0-run "runs/e0_20261004_221345_872444" --output "runs/e1_unet_cuda_seed42" --train-only
```

保持默认 U-Net、352×352、100轮、AdamW lr=1e-4/weight decay=1e-4、BCE+Dice、seed=42、物理batch=2、累积4、float32。无混合精度；无按test选模型。接通电源并设置运行期间不自动睡眠。

若显存不足，换一个新目录，以 batch=1、累积8重跑：

```powershell
python e1_windows/run_e1.py train --device cuda --batch-size 1 --accumulation 8 --e0-run "runs/e0_20261004_221345_872444" --output "runs/e1_unet_cuda_b1_seed42" --train-only
```

Windows 内部恢复（原batch=2配置）：

```powershell
python e1_windows/run_e1.py train --device cuda --resume "runs/e1_unet_cuda_seed42" --train-only
```

如果改变了 batch/accumulation 等原配置，恢复必须传相同参数；设备、源码和库版本必须一致。Windows 运行名称独立于原 Mac 运行。

## 5 Windows 完成后可先做全量评估

```powershell
python e1_windows/run_e1.py evaluate --project "D:\Projects\Polyp-PVT" --run-dir "runs/e1_unet_cuda_seed42" --e0-run "runs/e0_20261004_221345_872444" --device cuda
```

训练 `--train-only` 完成时根run_info状态为 TRAINED；评估全部798条目完成才为 COMPLETED。评估输出包含分子集指标、与E0分数差值、原尺寸 float32 概率图和二值掩膜。CUDA与E0的Mac CPU耗时不直接作加速倍数。

## 6 Windows 拷回 Mac

最省事的方式是完整复制这两个目录：

```text
e1_windows/                           # 必须是训练使用的同一份代码
runs/e1_unet_cuda_seed42/              # 保留Windows run名称
```

复制前等待训练/评估结束，避免拿到正在写入的临时文件。不覆盖 Mac 的 `e1_mac/`、`runs/e1_unet_mps_seed42/` 或 E0 结果。

如果需要减少传输量，至少保留：

- `checkpoints/unet_final.pth`：最终权重和配置，Mac评估必需，约118 MiB。
- `run_info.json`：最终checkpoint哈希、环境、状态和运行来源，Mac评估必需。
- `test_manifest.json`：冻结测试样本和哈希，Mac评估必需。
- `checkpoints/last.pth`：含AdamW状态，约355 MiB；为后续继续训练E2保留，只有推理时可省。
- `train_manifest.json`、`train_history.csv`、`data_audit.json`、`test_audit.json`：训练追溯与报告分析建议全部保留。
- `evaluation_cuda_.../`：若在Windows评估，复制其 CSV、metrics.json、run_info.json，以及 probabilities/、masks/。全量复制便于保持一次评估的完整结果。
- `e1_windows/environment-windows.txt`：若已生成，也复制回Mac。

没有必要拷回同一份训练/测试数据、E0结果、Windows conda环境；它们已在Mac上或不跨系统使用。

`.pth` 权重保存为 CPU tensor，Mac可加载CUDA训练的模型；GPU训练不意味着推理也必须使用同种GPU。当前train resume规则要求同设备，但evaluate允许CPU/MPS/CUDA。

## 7 Mac 加载 Windows 训练结果并统一 CPU 评估

将目录放回同一个Mac工程根目录，然后：

```bash
cd "/Users/yucheng/Desktop/NUS-ISS/ISY5002-Practice Module/Polyp-PVT"
conda activate polyp-pvt-mac
python e1_windows/run_e1.py evaluate \
  --project "/Users/yucheng/Desktop/NUS-ISS/ISY5002-Practice Module/Polyp-PVT" \
  --run-dir "runs/e1_unet_cuda_seed42" \
  --e0-run "runs/e0_20261004_221345_872444" \
  --device cpu --threads 4
```

明确传本机 `--project` 和 `--e0-run`；源运行记录保留Windows路径用于追溯，不手动改JSON。新评估记录会单独保存本机项目路径、E0参考路径和设备环境。

必须使用拷回的 `e1_windows/` 同一份源码。若误用原 `e1_mac/`，源码hash检查会拒绝加载，而不是让修改过的代码悄悄改变实验。

## 8 改动与验证范围

- 模型、损失、seed、默认epoch/batch/lr/阈值未改变。
- `e0_protocol.py` 的相对路径保存统一为 `as_posix()`，避免Windows反斜杠导致冻结manifest签名不一致。
- `run_e1.py` 训练清单同样保存POSIX相对路径。
- evaluate新增 `--e0-run`，解析并验证本机E0清单，与Windows训练保存的test签名一致才评估。
- CPU五项测试、真实数据小规模训练/评估、模拟Windows绝对路径迁回Mac评估验证见VALIDATION.md。
- 未在真实Windows/RTX5060硬件运行，必须先跑本机CUDA探针和标准smoke；没有提前执行正式100轮。
