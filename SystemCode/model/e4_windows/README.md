# EndoLens AI E4 强数据增强对照

## 实验边界

本轮复用经过 E2 验证的预训练 ResNet34–U-Net，**只改变数据增强**：E2-C 的 `flips` 与 E4 的 `strong`。固定 seed42、1160训练/290验证、352输入、100轮预算、batch2×累积4、AdamW lr1e-4、原损失和scheduler、AMP训练/FP32验证。strong保留同步水平/垂直翻转，新增同步±15°旋转和仅图像的亮度/对比度0.85–1.15。

E4重新从ImageNet编码器和随机解码器开始；不从E2最终模型或开发best继续训练。`core/` 是逐字保留的E2核心，源码哈希与原开发C一致，以避免训练管线变化混入增强对照。core运行元数据仍使用 `E2_DEVELOPMENT` / `E2_FINAL`，这是兼容协议标识；E4以目录名、strong配置和附加 `e4_stage.json` 区分，并非误跑E2。

## 1 从Mac迁往Windows

**本次新生成、必须复制的只有 `e4_windows/`**，包括其完整`core/`子目录。放在Windows工程根目录，与dataset、runs平级。不要把core文件单独搬到其他目录。

Windows原E2工程保留如下文件时，无需从Mac重复传输：

| 内容 | 用途与最低要求 |
|---|---|
| `dataset/TrainDataset/images/`与`masks/` | 全1450个训练条目；E4需要训练数据，区别于E3仅需测试数据 |
| `dataset/TestDataset/test/images/`与`masks/` | 冻结bundle审计及最终798条目评估所需的合并测试目录 |
| `runs/e2_data_seed42/`整个文件夹 | 沿用原1160/290划分和数据签名；不得重新prepare或重新划分 |
| `runs/e2_dev_C_resnet_pretrained_seed42/` | `run_info.json`、`train_history.csv`、`validation_best_per_image.csv`、`checkpoints/best.pth`；作为匹配控制，推荐保留整目录 |
| `e2_weights/resnet34-b627a593.pth` | 原ImageNet初始化权重；不是E2训练出的final.pth |
| `runs/e0_20261004_221345_872444/` | 最终评估需要run_info、manifest、per_image；保留完整目录也可 |

Mac目前没有工程根下的 `e2_weights/`，因此不要假设能从Mac复制该文件。Windows之前成功训练过E2-C，通常已保留它。若确实缺失，在工程根目录用已有E2环境运行：

```bat
python e4_windows\core\run_e2.py download-weights --project "."
```

该命令下载并校验官方预训练权重；网络不可用时需把原Windows权重备份放回对应位置。不要用E2最终模型替代初始化权重。

不需要迁移E3融合概率、E3 Word报告、Mac分析图或E1结果；它们不参与E4开发。三个E2开发方案中仅C作为本轮控制，不需重新训练A/B/C。

## 2 环境

继续复用 `endolens-e2-win`。原环境为Python3.11、torch2.9.1+cu128、torchvision0.24.1+cu128、NumPy1.26.4、Pillow10.2.0，不需要重装CUDA或PyTorch。

在 **Anaconda Prompt/CMD**：

```bat
conda activate endolens-e2-win
cd /d "F:\Semester1\ISY5002-PracticeModule\Polyp-PVT\Polyp-PVT"
python -m pip install -r e4_windows\requirements-common.txt
python -c "import sys,torch,torchvision; print(sys.executable); print(torch.__version__,torchvision.__version__); print('CUDA available:',torch.cuda.is_available())"
```

PowerShell中目录切换改为 `Set-Location "F:\Semester1\ISY5002-PracticeModule\Polyp-PVT\Polyp-PVT"`。

## 3 代码测试和CUDA确认

```bat
python e4_windows\test_e4.py
python e4_windows\core\test_e2.py
python e4_windows\run_e4.py check-gpu
python e4_windows\run_e4.py preflight
```

两个test脚本是CPU功能测试，不是正式CPU训练。check-gpu会用实际352输入执行CUDA前向、BCE/Dice反向和AdamW参数更新，生成 `runs/e4_gpu_device_proof.json`。应显示真实GPU及 `PASS: REAL CUDA FORWARD, BACKWARD AND PARAMETER UPDATE. NO CPU FALLBACK.`

preflight强制CUDA、核验原C配置和best检查点、完整数据bundle、290验证ID与初始化权重。遇到源码、数据或配置不匹配应修复迁移文件，不要跳过检查。`--skip-gpu`只用于Mac只读preflight，不允许正式CPU训练；Mac也必须具备所有输入才能完成该核验，实际不必在Mac执行。

## 4 Smoke

```bat
python e4_windows\run_e4.py smoke
```

输出 `runs/e4_smoke_seed42/`。它只训练一轮的小批次并验证管线，不使用798测试集选择方案；内部标记GPU_SMOKE，不能用于正式结果。

## 5 正式开发训练

```bat
python e4_windows\run_e4.py train
```

固定输出 `runs/e4_dev_strong_seed42/`，完成100轮1160训练/290验证。日志应显示CUDA/RTX5060；run_info里的模型及逐轮images、masks、logits应为cuda:0，AMP forward float16、loss float32。CPU负责读图/增强是正常的，不代表模型用CPU训练。

中断后，从最近完整epoch继续：

```bat
python e4_windows\run_e4.py resume --run-dir "runs\e4_dev_strong_seed42"
```

若Windows DataLoader报spawn/worker错误，可在首次运行或resume加 `--workers 0`；这只调整加载进程，模型仍强制CUDA。不要为了OOM自行改batch/AMP/lr，否则不再是当前预设的匹配对照；先保存错误信息再调整实验设计。

Windows单次总时间不能预先保证。E2-C开发约52分30秒，E4有额外增强开销，可把此作为粗略参考；以实际日志为准。

## 6 对比内部验证并停止在决策点

完成100轮后执行：

```bat
python e4_windows\run_e4.py compare
```

生成：

- `runs/e4_comparison_seed42/comparison.csv`、`comparison.json`
- `runs/e4_comparison_seed42/validation_paired.csv`
- `runs/e4_selection_seed42.json`与`.csv`

比较原C与E4的best验证Dice/IoU、中位数、低分/零分条目、第100轮和末20轮均值、AMP跳过情况。方案和轮次选择仅用内部验证Dice最高值，持平取更早轮次；best方案相同分数时原C保留。

**先把开发结果迁回Mac进行分析，不要自动接着跑最终训练。** 若只有极小提升、末期波动或失败尾部恶化，需先判断是否值得复验。只有E4出现可信改善才考虑全量重训；若未改善，保留E2/E3。compare会保存预选recipe，但不意味着已批准任何科学结论。

输出不可覆盖。若开发使用自定义目录，例如首次目录失败后改用 `--output "runs\e4_dev_strong_seed42_retry1"`，比较命令必须相应传 `--run-dir "runs\e4_dev_strong_seed42_retry1"`。compare的输出用新目录，selection文件仍固定 `runs/e4_selection_seed42.json`；已有recipe会拒绝覆盖，应保留已有完整比较，不要重复选择。

## 7 仅在决定采用strong后进行最终重训与评估

```bat
python e4_windows\run_e4.py train-final
python e4_windows\run_e4.py evaluate
```

输出 `runs/e4_final_seed42/`。train-final只有recipe选中strong才允许执行；从原ImageNet编码器和随机解码器重新开始，全部1450条目，轮数使用E4最佳内部验证epoch，保留100轮scheduler周期，不直接续训开发best。

evaluate在CUDA上评估全798条目，自动生成 `evaluation_cuda_日期时间/`，包括逐图、汇总、E0对比和原尺寸概率/掩膜。不要在测试集上回溯best轮次、调阈值、调增强。目录和core的E2协议标识是兼容设计，结果应归档为E4。

最终训练如中断：

```bat
python e4_windows\run_e4.py resume --run-dir "runs\e4_final_seed42"
```

本代码包不自动跑新融合。先分析E4单模型，再决定是否进行E0+E4的固定融合。

## 8 哪些文件迁回Mac

### 开发阶段结束后先迁回

复制完整目录/文件至Mac工程下同名位置：

- `runs/e4_dev_strong_seed42/`（完整，尤其train_history、run_info、validation_best_per_image、checkpoints/best.pth与last.pth）
- `runs/e4_comparison_seed42/`
- `runs/e4_selection_seed42.json`、`runs/e4_selection_seed42.csv`
- `runs/e4_gpu_device_proof.json`
- 本次使用的完整 `e4_windows/` 源码（建议保留）

Mac根目录：

`/Users/yucheng/Desktop/NUS-ISS/ISY5002-Practice Module/Polyp-PVT/`

这时即可分析开发对照；没有最终模型也正常，不需要为了凑齐e4_final而强制重训。

### 决定重训并完成评估后再迁回

- 完整 `runs/e4_final_seed42/`，含最终检查点、可恢复last、训练日志、配置、清单及**完整 evaluation_cuda_.../**（per_image/summary/probabilities/masks等）
- 如原Mac缺失，补一份 `e2_weights/resnet34-b627a593.pth`，方便完整preflight/未来重训；只分析CSV不需要该初始化权重。

不用重复复制训练数据、E0/E2/E3结果；Mac已有相同版本即可。不要只复制summary.csv，否则无法核验best、分析验证曲线或开展后续概率融合。

## 9 已验证与未验证

本包的E2核心与原C记录源码一致，E4专项测试验证“只改增强”、强增强标签几何和确定性、smoke隔离。Mac可完成CPU功能与真实baseline核验；Mac不能验证RTX5060实跑，Windows需执行check-gpu、preflight和smoke。没有把CPU测试当作GPU训练成功，也未运行正式E4或生成E4实验分数。
