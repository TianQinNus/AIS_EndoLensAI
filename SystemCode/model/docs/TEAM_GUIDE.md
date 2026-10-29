# 组员安装、运行和跨平台同步

## 1. GitHub 上传/同步思路

仓库只保存可评审、可版本管理的代码与文档；图片、权重、原尺寸概率以及恢复训练所需的完整运行目录通过组内单独共享。保持课程四目录结构，模型工程统一在 `SystemCode/model/`；冻结 E0–E4 源码，不为了新目录而修改模型或训练引擎。

本次同步内容：E0/E1/E2/E3/E4 Python 与测试、环境依赖、CMD 入口、固定1160/290划分与798测试清单、E0轻量参考元数据、训练历史/逐图指标/结果摘要、E0–E4 Word报告、组员指南和第三方归属。既有HDB课程示例归档保留，不作为EndoLens成果。

禁止整份本地Polyp-PVT文件夹直接上传：它包含数据、虚拟环境、模型、大量运行产物和重复快照。`.gitignore` 排除这些内容；不要强制添加。

## 2. Clone 与恢复共享元数据

```bash
git clone https://github.com/TianQinNus/EndoLensAI.git
cd EndoLensAI/SystemCode/model
python scripts/bootstrap_artifacts.py
python scripts/verify_imported_code.py
```

bootstrap不需要第三方依赖。第一次恢复7份固定bundle文件及E0轻量元数据；重复执行相同内容不覆盖，不同内容会报错。已有完整E0目录不要用bootstrap覆盖：先在干净checkout执行bootstrap，再将授权的完整运行目录整份迁入；现有完整运行的run_info可能含不同机器路径，必须保持原始产物一致。

`Miscellaneous/model_results/` 是只读分享的指标，不是代码运行目录；不要整份复制为 `runs/` 后假设可以 resume/fusion。

## 3. 恢复图片与权重

向实验负责人获取已验证的数据，原样放在：

```text
SystemCode/model/
  dataset/
    TrainDataset/images/                  1450张
    TrainDataset/masks/
    TestDataset/
      Kvasir/images/ masks/                100对
      CVC-ClinicDB/images/ masks/           62对
      CVC-ColonDB/images/ masks/            380对
      ETIS-LaribPolypDB/images/ masks/      196对
      CVC-300/images/ masks/                60对
      test/images/ masks/                  798合并条目
  model_pth/PolypPVT.pth                   仅重新推理E0需要
  pretrained_pth/pvt_v2_b2.pth             仅重新推理E0需要
  e2_weights/resnet34-b627a593.pth         E2/E4编码器初始化
  runs/                                    完整产物另行恢复
```

不要把图像重新保存、改变stem、或使用另外划分的Kvasir/CVC文件。脚本验证文件哈希。已有匹配数据不用重拷贝。完整实验依赖见 [资产表](../../../Miscellaneous/model_artifacts/README.md)。

## 4. Windows CUDA 环境与功能检查

以下在Anaconda Prompt运行，先进入 `EndoLensAI\SystemCode\model`。CMD用 `cd /d "你的目录"`，PowerShell用 `Set-Location "你的目录"`。

```bat
conda create -n endolens-e2-win python=3.11 -y
conda activate endolens-e2-win
python -m pip install -r e2_windows/requirements-windows.txt --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r e2_windows/requirements-common.txt
python scripts/run_checks.py
python e2_windows/check_device.py --device cuda --output runs/e2_gpu_device_proof.json
python e2_windows/run_e2.py download-weights
```

固定torch2.9.1/torchvision0.24.1（cu128）、NumPy1.26.4、Pillow10.2.0，对应原Windows成功环境。不要只凭CPU单元测试或任务管理器3D曲线判断训练设备。探针必须实际完成CUDA前向、反向、参数更新；训练日志应显示model/images/masks/logits位于cuda:0。

无CUDA时E2/E4正式训练直接失败；数据加载用CPU是正常行为。`--workers 0`可处理Windows进程加载问题，不会切换模型到CPU。以上安装及完整GPU流程仍须在组员自己的电脑验证。

## 5. E2 从头复现

使用已恢复的 `runs/e2_data_seed42/`，不要重新prepare划分。

```bat
python e2_windows/run_e2.py train --experiment pretrained --device cuda --smoke --output runs/e2_smoke_C_seed42
python e2_windows/run_e2.py train --experiment original --device cuda --output runs/e2_dev_A_original_seed42
python e2_windows/run_e2.py train --experiment random --device cuda --output runs/e2_dev_B_resnet_random_seed42
python e2_windows/run_e2.py train --experiment pretrained --device cuda --output runs/e2_dev_C_resnet_pretrained_seed42
python e2_windows/run_e2.py select --runs runs/e2_dev_A_original_seed42 runs/e2_dev_B_resnet_random_seed42 runs/e2_dev_C_resnet_pretrained_seed42
```

查看内部验证选择记录后，再执行：

```bat
python e2_windows/run_e2.py train-final
python e2_windows/run_e2.py evaluate --device cuda --e0-run runs/e0_20261004_221345_872444
```

每个输出必须是新目录，拒绝覆盖；上述默认名如果已有产物就不要重复执行。重跑使用新输出名，并同步修改后续`--runs`、`--recipe`、`--run-dir`参数。显卡/种子/版本可能引入非完全相同数值，不能保证新运行复现逐位一致。

中断后使用原运行配置：

```bat
python e2_windows/run_e2.py resume --run-dir runs/e2_dev_C_resnet_pretrained_seed42 --device cuda
```

续训需恢复完整权重/优化器/RNG等信息和相同源码环境，不要跨平台迁移后直接resume。详细历史操作说明：[E2_操作说明](../e2_windows/E2_操作说明.md)，其中旧路径需替换为本仓库模型目录。

## 6. E4 增强对照

先恢复匹配的 E2-C **完整运行**，尤其 `run_info.json`、`train_history.csv`、`validation_best_per_image.csv`、`checkpoints/best.pth`，以及相同bundle/初始化权重。仅有GitHub摘要时preflight不能通过。

```bat
python e4_windows/run_e4.py check-gpu
python e4_windows/run_e4.py preflight
python e4_windows/run_e4.py smoke
python e4_windows/run_e4.py train
python e4_windows/run_e4.py compare
```

分析内部验证后，若选择strong且决定重训：

```bat
python e4_windows/run_e4.py train-final
python e4_windows/run_e4.py evaluate
```

本轮已完成的结果可以直接分析，不要求组员重新训练。`core/`的E2阶段标识为兼容设计；E4由strong配置、目录和e4_stage区分。参阅 [E4原操作说明](../e4_windows/README.md)。

## 7. E3/V3 离线融合

恢复完整E0/E2概率、掩膜、来源元数据、E2最终检查点和合并测试数据。GitHub中没有这些大文件。

```bat
python v3_windows/run_v3.py --audit-only --output runs/v3_audit_seed42
python v3_windows/run_v3.py --smoke --output runs/v3_smoke_seed42
python v3_windows/run_v3.py --output runs/v3_fusion_equal_seed42
```

该步骤是NumPy CPU概率加法，没有训练，也不需要CUDA参数；默认E2评估名为 `evaluation_cuda_20261005_231641_102143`。目录更名时显式传`--e0-run`、`--e2-run`、`--e2-eval`。它仍是E0＋E2，不要给它传E4目录后把元数据当成正确的E5实验。E5需要独立实现与审计，目前只规划。

## 8. Mac使用范围

可在已有 `polyp-pvt-mac` 环境复用CPU结果分析；E0参考环境原为Python3.10、torch2.2.2、torchvision0.17.2、timm0.4.12。依赖见 `e0_mac/requirements-e0-mac.txt`。不要用Windows CUDA安装命令安装Mac环境。

E2最终模型CPU评估入口（恢复完整final run和相同测试数据后）：

```bash
python e2_windows/run_e2.py evaluate --run-dir runs/e2_final_seed42 --e0-run runs/e0_20261004_221345_872444 --device cpu
```

E4 wrapper强制CUDA评估；Mac必要时使用冻结核心的通用评估入口：

```bash
python e4_windows/core/run_e2.py evaluate --run-dir runs/e4_final_seed42 --e0-run runs/e0_20261004_221345_872444 --device cpu
```

新评估写入新的evaluation目录，与已归档CUDA结果分开。Mac推理耗时不能替代Windows GPU部署耗时。E1 Mac旧代码仅作基线保留，当前正式实验路线为E2/E4。

## 9. Windows训练后迁回哪些内容

开发阶段：整个对应dev运行目录、固定bundle、GPU设备证明、比较目录和selection JSON/CSV。最终阶段：整个final目录，包括checkpoints、配置、日志、清单及完整evaluation目录（概率、掩膜、CSV/JSON）；融合阶段：整个fusion目录。同时保留实际使用源码与初始化权重备份。

迁回Mac的位置是该checkout的 `SystemCode/model/runs/`；如果继续用旧Polyp-PVT项目分析，也可放在其原runs位置。不要只拷贝summary或只有final.pth，否则无法完成来源校验、恢复或后续融合。

## 10. 日常协作命令

从仓库根目录：

```bash
git pull --ff-only
git switch -c feature/your-task
# 编辑代码/文档，在SystemCode/model运行相应检查
git status --short
git diff --stat
git add <明确的代码或文档路径>
git diff --cached --stat
git commit -m "Describe the change"
git push -u origin feature/your-task
```

在GitHub创建pull request给组员审核，合并后其他成员pull。不上传数据/权重，不强制推送master，不在训练期间编辑被冻结的Python。视频/前端接口可以新增模块，不需改历史实验文件。
