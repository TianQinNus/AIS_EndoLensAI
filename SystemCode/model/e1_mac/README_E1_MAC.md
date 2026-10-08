# E1 本地运行说明

E1 在现有 1450 对训练数据上，从随机初始化训练标准 U-Net，建立自己的单模型基线。使用完成的 E0 运行冻结测试清单，测试仍为五个子集共 798 个条目。当前阶段不融合、不加载 Polyp-PVT 权重，不利用测试集挑选 epoch 或超参数。

## 1 文件放置

将 ZIP 解压出的整个 `e1_mac` 文件夹放到现有 `Polyp-PVT/` 根目录：

```text
Polyp-PVT/
  dataset/TrainDataset/images/ 和 masks/
  dataset/TestDataset/test/images/ 和 masks/
  dataset/TestDataset/Kvasir/、CVC-ClinicDB/、CVC-ColonDB/、ETIS-LaribPolypDB/、CVC-300/
  runs/e0_20261004_221345_872444/
  e1_mac/
    run_e1.py
    unet.py
    e0_protocol.py
    check_device.py
    test_e1.py
    requirements-e1-mac.txt
    README_E1_MAC.md
    VALIDATION.md
```

也可以将代码保存在别处，所有命令附上 `--project "/完整路径/Polyp-PVT"` 即可。本包不依赖原仓库的 `Train.py`、`timm`、`torchvision` 或预训练权重，也不修改其代码。

## 2 推荐环境与 M5 GPU 检查

建议 E1 建立独立环境。原 E0 环境为 PyTorch 2.2.2，保留它用于复核 E0 和 CPU 耗时比较。E1 使用近期稳定版 PyTorch，以适配较新的 Apple GPU。无需 CUDA。

在本地 Terminal 中执行：

```bash
cd "/Users/yucheng/Desktop/NUS-ISS/ISY5002-Practice Module/Polyp-PVT"
conda create -n endolens-e1 python=3.11 -y
conda activate endolens-e1
python -m pip install -r e1_mac/requirements-e1-mac.txt
python e1_mac/check_device.py --device mps
python -m pip freeze > e1_mac/environment-local.txt
```

检查输出应包含 `Architecture: arm64`、`MPS built: True`、`MPS available: True`，以及在 mps 上 forward、backward、AdamW 的 PASS。MPS 使用 Apple 芯片的 GPU，不是 Neural Engine。探针使用小模型检查算子和设备可用性，实际 352×352 模型还需下一步 smoke 验证。

安装说明依据 [Apple 的 PyTorch Metal 文档](https://developer.apple.com/metal/pytorch/) 和 [PyTorch MPS 文档](https://docs.pytorch.org/docs/stable/notes/mps.html)。Apple 当前页面给出的要求为 Apple silicon、macOS 14 或以上、Python 3.10 或以上及 Xcode command-line tools。如缺少命令行工具，运行 `xcode-select --install`。本包选择 Python 3.11，并将实际 torch/NumPy/Pillow 版本写入运行记录，不把动态安装的版本伪称为锁定环境。

本次代码生成环境检测到了 M5 / 16 GB，但执行进程中的旧 PyTorch 未发现可用 MPS。因此 CPU 路径已实测；新环境的 M5 GPU 路径由你在本地执行上述探针和 smoke 验证。此包未声称已完成 GPU 训练。

如果 MPS 不可用：

- 检查已激活 `endolens-e1`，且 `python -c "import platform; print(platform.machine())"` 为 `arm64`。
- 在 Mac 本地 Terminal 运行检查，避免通过不支持 GPU 的隔离执行环境运行。
- 查看 `python -c "import torch; print(torch.__version__, torch.backends.mps.is_built(), torch.backends.mps.is_available())"`。
- 暂时用 `--device cpu` 验证流程；100 轮 CPU 训练耗时需要以第一轮实测估计。

`--device mps` 不可用时会明确报错；`--device auto` 依次选择 MPS、CUDA、CPU，并打印实际设备。默认不开启 MPS CPU 算子回退、不使用混合精度。不要为绕过内存错误关闭 MPS 内存上限。

## 3 先做小规模端到端验证

```bash
python e1_mac/test_e1.py
python e1_mac/run_e1.py train --device mps --smoke
```

第二条使用正式模型宽度 64、输入 352×352，训练 8 张图 1 轮，并对每个测试子集各 1 张做推理。它验证读取、训练、梯度累积、保存权重、恢复原尺寸、指标计算和结果文件，结果标记为 `SMOKE`，不作为正式 E1 指标。它只更新一次权重，预测全背景、Dice 为 0 也可能出现，不能据此判断 100 轮训练的效果。

如需先做 CPU 快速功能检查，可缩小 smoke 模型：

```bash
python e1_mac/run_e1.py train --device cpu --smoke --input-size 64 --base-channels 8
```

只有正式实验使用标准模型和全量训练。每个新实验自动使用独立目录；自定义目录必须尚不存在。

## 4 正式 E1 训练

M5 / 16 GB 推荐物理 batch=2，梯度累积=4，有效 batch=8：

```bash
python e1_mac/run_e1.py train \
  --device mps \
  --e0-run "runs/e0_20261004_221345_872444" \
  --output "runs/e1_unet_mps_seed42" \
  --train-only
```

所有默认训练设置已写入代码：

| 项目 | E1 设置 |
|---|---|
| 训练数据 | 原有 TrainDataset 的全部 1450 对，不另拆验证集 |
| 测试数据 | E0 冻结清单，798 个条目，按五个来源分别汇总 |
| 模型 | 四层下采样/上采样 U-Net；通道 64/128/256/512/1024 |
| 卷积 | 每级两次 3×3 Conv + ReLU，padding=1；无 BatchNorm、无 dropout |
| 上采样 | 2×2 ConvTranspose2d；与编码器 skip feature 拼接 |
| 初始化 | PyTorch 模块默认随机初始化；无预训练编码器 |
| 输入 | RGB 352×352，Pillow bilinear，/255，ImageNet mean/std |
| 标签 | 灰度掩膜二值化后 nearest resize，前景=1 |
| 数据增强 | 图像与标签同步水平/垂直翻转，各 p=0.5 |
| 训练轮数 | 100，固定学习率，无 scheduler、无早停 |
| 优化器 | AdamW，lr=1e-4，weight decay=1e-4，betas=(0.9,0.999) |
| 损失 | BCEWithLogits + 每图 soft Dice loss，系数各 1，smooth=1 |
| Batch | 物理 2 × 累积 4 = 有效 8；不丢弃最后不足整组的样本 |
| Seed | 42；每轮样本顺序与每图翻转可复现 |
| 精度 | float32，不启用 AMP |
| 检查点选择 | 第 100 轮最终权重，不按 test 分数选 best |

原始 U-Net 的论文实现细节存在不同版本。本包将上述具体实现固定为项目基线，不声称逐行复现原始 U-Net 论文。无 BatchNorm/dropout，因此采用累积梯度时没有微批统计或 dropout 带来的额外差异；浮点计算顺序仍可能导致轻微数值差异。

1450 张图每轮产生 182 次 optimizer 更新；最后一组仅 2 张。损失按实际组内样本数加权，避免末尾梯度被错误缩小。代码记录实际 batch、累积步数与更新时间。

如果 GPU smoke 或训练提示内存不足，使用新目录从头运行：

```bash
python e1_mac/run_e1.py train --device mps \
  --batch-size 1 --accumulation 8 \
  --e0-run "runs/e0_20261004_221345_872444" \
  --output "runs/e1_unet_mps_b1_seed42" --train-only
```

有效 batch 仍为 8，模型与分辨率保持一致。不要在正式实验中无记录地缩小图像或通道数。`--epochs`、`--input-size`、`--base-channels` 等改变预定基线时，运行标记为 `CUSTOM_E1`，避免与正式 100 轮结果混淆。

第一轮结束会显示该轮耗时与粗略剩余训练时间。估计不包含检查点写入，不能当作完整训练时间。训练需要持续供电，可使用 `caffeinate -i python ...` 包住正式训练命令，避免系统休眠导致暂停。

## 5 完成后评估 798 张并比较 E0

为比较两种模型的 CPU 推理时间，可以切回 E0 原环境执行评估：

```bash
conda activate polyp-pvt-mac
python e1_mac/run_e1.py evaluate \
  --run-dir "runs/e1_unet_mps_seed42" \
  --device cpu --threads 4
```

E1 推理与 E0 使用相同的 RGB 预处理、原图坐标、原尺寸 logits bilinear 插值、sigmoid、概率阈值 0.5 和固定阈值 Dice/IoU。U-Net 是单路 logits；Polyp-PVT 的两路 logits 相加属于它自身的输出方式。均不做逐图 min–max。

如果希望评估 GPU 推理速度，在 `endolens-e1` 环境执行同一命令并改为 `--device mps`。不同设备与库版本的耗时应分开报告；GPU 结果不能直接与 E0 的 CPU 耗时计算“加速倍数”。训练精度对比要保留评估环境记录。

`train` 省略 `--train-only` 时，训练结束立即在当前训练设备评估。测试仅在训练完成后运行，训练过程中不查看其分数。额外的跨设备评估用于数值与性能验证，不用于挑选更高分结果。

成功结束需满足根目录 `run_info.json` 的 `run_kind=FULL_E1`、`status=COMPLETED`、`completed_epochs=100`、`test_processed_count=798`。`--train-only` 结束时为 `TRAINED`，仍需要执行评估。

## 6 中断与恢复

每轮结束原子保存 `checkpoints/last.pth`，其中包含模型、AdamW 状态、轮数、训练日志、数据签名和配置。Ctrl+C、设备报错或系统中断后，可从最近完成的整轮恢复；未完成轮次会重跑，不进行轮内恢复。

```bash
conda activate endolens-e1
python e1_mac/run_e1.py train --device mps \
  --resume "runs/e1_unet_mps_seed42" --train-only
```

恢复使用原 run 目录，不再传 `--output`。若原运行改变了 batch 等参数，恢复时需提供同样设置。恢复要求配置、代码、数据、设备及 torch/NumPy/Pillow 版本一致，避免混合实验。epoch 顺序与增强按 seed/epoch/sample index 生成；CPU 的合成小实验已验证中断恢复与连续训练最终参数逐元素一致，但不同后端/驱动间不承诺 bitwise 相同。

若训练已完成而评估失败，不需要重新训练，重新执行 `evaluate` 即可；每次评估创建新目录，不覆盖已有输出。

## 7 输出文件

```text
runs/e1_unet_mps_seed42/
  run_info.json
  train_manifest.json
  test_manifest.json
  data_audit.json
  test_audit.json
  train_history.csv
  checkpoints/
    last.pth                 # 最近完成整轮，含模型与优化器，供恢复/E2继续训练
    unet_final.pth           # 最终轮模型与配置，供评估，未按测试集选best
  evaluation_cpu_日期时间/
    run_info.json
    per_image.csv
    summary.csv              # ALL + 五个子集
    metrics.json
    comparison_e0_e1.csv      # 同清单，E1减E0，Dice/IoU
    comparison_per_image.csv
    probabilities/数据集/样本.npz
    masks/数据集/样本.png
```

NPZ 的 `probability` 为 float32 原图 H×W 连续概率，后续 E2 应沿用该输出口径，E3 对齐 sample_id 后再融合。二值 PNG 不是融合输入。

`forward_ms` 为 batch=1 且同步设备的模型前向时间，不含读图、预处理、插值、sigmoid、指标及保存。`analysis_loop_seconds` 包含完整图像评估循环但不含数据审计、模型加载、预热，不能称为网页或视频端到端延迟。

## 8 数据审计与已验证范围

训练前自动验证训练集一一配对、1450 对计数、尺寸、哈希；重新审计 798 个测试条目与五个子集，并严格与 E0 冻结 manifest 比较。检查训练与测试的完全相同解码 RGB 图像；发现重叠即停止，不自动删图或改划分。

本次实际检查：训练集 1450 张不同 RGB 图像，训练与测试完全相同图像重叠数为 0。测试中已知重复内容继续保留原基准清单。这个检查不能替代患者/序列层级的独立性验证。

完整 CPU 验证、已测试环境、检查点恢复及输出一致性验证记录见 `VALIDATION.md`。此阶段不自动执行 100 轮训练或宣称模型优于 E0。
