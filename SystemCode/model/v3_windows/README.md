# EndoLens AI V3 固定等权概率融合

本阶段对应此前讨论的 E3/Fusion，按本次约定命名为 V3。公式预先固定：

`p_fusion = 0.5 * p_E0 + 0.5 * p_E2`，`mask = p_fusion >= 0.5`。

融合原尺寸 float32 sigmoid 概率，不融合 PNG 二值掩膜、不平均模型参数、不进行逐图 min–max、不搜索权重/阈值。两个分支都被冻结。V3 **没有训练步骤**，不需要再次加载模型前向；NumPy CPU 加法即能完成。这不是 GPU 训练速度或视频 FPS 实验。E2 原预测已在 Windows CUDA 上产生，复用它们可保持对比使用完全相同的分支结果。

## 1 放置代码与迁移内容

将本文件夹 `v3_windows/` 放在 Windows 工程根目录，即包含 `dataset/` 与 `runs/` 的目录。示例：

```
F:\Semester1\ISY5002-PracticeModule\Polyp-PVT\Polyp-PVT\
  v3_windows\
    run_v3.py
    test_v3.py
    requirements.txt
    README.md
    01_check_and_smoke.cmd
    02_run_full.cmd
  dataset\TestDataset\test\
    images\
    masks\
  runs\
    e0_20261004_221345_872444\
    e2_final_seed42\
```

Windows 上已经存在且与 Mac 相同的文件不用重拷贝。建议保留完整 E0/E2 结果文件夹；最低必需内容如下。

### E0：`runs/e0_20261004_221345_872444/`

- `run_info.json`、`manifest.json`、`per_image.csv`
- `probabilities/`：798 份原始概率 NPZ
- `masks/`：798 份对应 PNG，用于校验概率与掩膜一致性

不要用 `legacy_minmax/` 代替 `probabilities/`。V3 不需要 E0 权重或原始模型源码，因为不重新推理。

### E2：`runs/e2_final_seed42/`

- `run_info.json`、`test_manifest.json`
- `checkpoints/final.pth`：校验最终模型和评估来源一致，不用于再次前向
- `evaluation_cuda_20261005_231641_102143/` 下的 `run_info.json`、`per_image.csv`、完整 `probabilities/` 与 `masks/`

若采用其他评估目录，需要显式传 `--e2-eval`，不能使用开发模型或 smoke 输出。

### 测试数据

- `dataset/TestDataset/test/images/`
- `dataset/TestDataset/test/masks/`

只需清单所引用的合并测试目录即可完成本次融合评估，不需要复制训练集、三个开发实验或 `e2_data_seed42/`。完整数据留在 Windows 也可以。脚本会对每张图像和标注做文件哈希检查，因此复制到新平台后不能重新编码图片。训练与测试划分不改变。

## 2 环境

推荐继续使用已安装 NumPy/Pillow 的 `endolens-e2-win` 环境，Python 3.11。不需要更新 PyTorch/CUDA。

在 Anaconda Prompt 或已可运行 conda 的终端：

```bat
conda activate endolens-e2-win
cd /d "F:\Semester1\ISY5002-PracticeModule\Polyp-PVT\Polyp-PVT"
python -m pip install -r v3_windows\requirements.txt
python -c "import sys,numpy,PIL; print(sys.executable); print('NumPy',numpy.__version__,'Pillow',PIL.__version__)"
```

`cd /d` 用于 CMD/Anaconda Prompt；如果使用 PowerShell，替换为：

```powershell
Set-Location "F:\Semester1\ISY5002-PracticeModule\Polyp-PVT\Polyp-PVT"
```

如不想复用环境，可建立只用于离线融合的轻量环境：

```bat
conda create -n endolens-v3-win python=3.11 -y
conda activate endolens-v3-win
python -m pip install -r v3_windows\requirements.txt
```

## 3 先测试代码与全量输入审计

在工程根目录，依次执行：

```bat
python v3_windows\test_v3.py
python v3_windows\run_v3.py --audit-only --output "runs\v3_audit_seed42"
```

测试涵盖等权概率与阈值边界、异常概率拒绝、空掩膜规则、完整 smoke 输出、数据篡改拒绝、不覆盖既有运行。输入审计检查全部 798 条目，不生成融合分数。

期望看到 `AUDIT PASSED: all 798 source entries verified`。Dice/IoU/计数校验容差为 1e-12；float32 MAE 在不同 NumPy 求和实现间可有微小差异，单独使用绝对容差 2e-6。若失败，查看审计目录 `run_info.json` 的错误和终端 traceback；不要跳过校验或换成 legacy 图。

## 4 Smoke 测试

```bat
python v3_windows\run_v3.py --smoke --output "runs\v3_smoke_seed42"
```

每个子集按 sample_id 排序选第一张，共 5 张，仅验证流程。结果标记 `SMOKE_ONLY`，不能作为实验结论。概率融合无需学习，smoke 分数也不要求为零。

## 5 正式 V3

```bat
python v3_windows\run_v3.py --output "runs\v3_fusion_equal_seed42"
```

默认路径已匹配当前 E0/E2 运行。完整显式命令如下，若工程或文件夹更名只改相应路径：

```bat
python v3_windows\run_v3.py --project "F:\Semester1\ISY5002-PracticeModule\Polyp-PVT\Polyp-PVT" --e0-run "runs\e0_20261004_221345_872444" --e2-run "runs\e2_final_seed42" --e2-eval "runs\e2_final_seed42\evaluation_cuda_20261005_231641_102143" --output "runs\v3_fusion_equal_seed42"
```

或在已激活环境的 CMD 中运行：

```bat
v3_windows\01_check_and_smoke.cmd
v3_windows\02_run_full.cmd
```

脚本按代码所在位置确定工程，路径参数可使用 Windows 路径；输出元数据内部使用 POSIX 相对路径以方便迁移。正式输出必须是 `FULL_V3`、`COMPLETED`、`processed_count=798`，ALL 与五个子集均有结果。

程序拒绝覆盖已经存在的输出目录，包括失败/审计/smoke 目录。再次运行时更换 `--output`（例如 `runs\v3_fusion_equal_seed42_retry1`），避免混用旧结果。自动 CMD 同样遵守此规则；不要因为第二次运行报目录存在而误认为代码失效。

实际耗时取决于磁盘解压 NPZ、哈希、PNG/NPZ 写入，预计明显短于重新训练；没有给出未经测量的固定时间承诺。终端每处理 50 张显示进度。显示 CPU 是预期行为：不会创建训练进程，也没有 `--device cuda` 参数。以后若需要部署速度，应另做两模型 CUDA 联合前向与视频端到端测试。

## 6 查看输出

`runs/v3_fusion_equal_seed42/`：

- `run_info.json`：阶段/状态/数量、固定配方、源码和来源哈希、执行环境
- `summary.csv`：ALL/五子集，E0/E2/fusion 的 Dice、IoU、MAE、中位数、低分与零分数量、融合相对两个模型的差值及胜/负/平数量
- `per_image.csv`：每图三个模型的 TP/FP/FN、Dice/IoU/MAE；融合相对两者的 Dice 变化
- `metrics.json`：分组统计与实验边界
- `manifest.json`、`input_fingerprints.csv`：实际样本清单与输入概率哈希
- `probabilities/`、`masks/`：798 份最终融合原尺寸概率和二值掩膜
- `comparison.html`、`visualizations/`、`visualization_selection.json`：可离线打开的代表样本可视化

用浏览器打开 `comparison.html` 即可。每个子集按固定规则展示中间分数、相对 E0 增益最大与下降最大条目，重复样本只展示一次。HTML 是离线结果页，不是最终视频产品。

Δ Dice/IoU >0 表示融合更高；Δ MAE <0 表示误差更小。优先判断是否超过 E0，而不是只超过较弱的 E2；同时查看 ETIS/ColonDB、低分条目及退步。没有提高也是有效实验结果。

## 7 迁回 Mac

复制 **完整** `runs/v3_fusion_equal_seed42/` 到：

`/Users/yucheng/Desktop/NUS-ISS/ISY5002-Practice Module/Polyp-PVT/runs/v3_fusion_equal_seed42/`

若正式运行采用其他名字，保留该真实名字并告知。将 `v3_windows/` 源码一并保留/复制回工程，便于复跑核查。审计/smoke 文件夹可作为流程记录迁回，但不能替代完整正式结果。

没有新训练权重、优化器或训练日志需要迁回。不要只拷贝 summary.csv：后续分析与视频集成还需要逐图结果、概率和掩膜。已有 Mac E0/E2 原目录无需重复复制。

## 8 科研表述与下一步

本轮是预先固定等权的决策层概率融合；不能写成新架构或已训练融合网络。测试结果可用于评价本轮配方，但不要查看结果后在同一测试集搜索最优权重/阈值并把最高分当成无偏结果。

如果融合改善整体且重要子集没有明显退步，可继续受控 GPU 性能和视频功能验证；如融合下降，先分析两个分支如何贡献错误，再决定是否改善 E2（如强增强对照）。本轮不自动修改 E2 配方，也不生成任何临床检测结论。
