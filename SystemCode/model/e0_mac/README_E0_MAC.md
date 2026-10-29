# E0：Mac 本地运行说明

本包复用现有 Polyp-PVT 代码和权重，执行固定模型的测试，不训练模型、不改变权重、不修改原仓库脚本。默认 CPU，固定输入 352×352、batch=1、阈值 0.5。

## 1. 文件如何放置

将整个 `e0_mac` 文件夹放到 `Polyp-PVT/` 下：

```
Polyp-PVT/
  lib/pvt.py
  lib/pvtv2.py
  model_pth/PolypPVT.pth
  pretrained_pth/pvt_v2_b2.pth
  dataset/TestDataset/
    test/images/ 和 test/masks/
    Kvasir/、CVC-ClinicDB/、CVC-ColonDB/、ETIS-LaribPolypDB/、CVC-300/
  e0_mac/
    run_e0.py
    requirements-e0-mac.txt
    test_e0.py
    README_E0_MAC.md
```

脚本可放在其他位置。使用 `--project "/完整路径/Polyp-PVT"` 指向项目即可，不需要修改 Python 源码。

## 2. 环境配置

优先复用已有环境：

```bash
conda activate polyp-pvt-mac
python -c "import sys, torch, torchvision, timm, numpy, PIL; print(sys.version); print(torch.__version__, torchvision.__version__, timm.__version__, numpy.__version__, PIL.__version__)"
```

已在本机发现的环境：Python 3.10.21、PyTorch 2.2.2、torchvision 0.17.2、timm 0.4.12、NumPy 1.26.4、Pillow 10.2.0，ARM64。不要为了跑 E0 升级它。

如需建立独立环境，在 `Polyp-PVT` 目录执行：

```bash
conda create -n endolens-e0 python=3.10 -y
conda activate endolens-e0
python -m pip install -r e0_mac/requirements-e0-mac.txt
```

不需要 CUDA、OpenCV、MATLAB、scipy 或 matplotlib。本包固定旧版本是为了兼容原仓库 timm 导入接口，并保持已跑通环境一致。torch 2.2.2 / torchvision 0.17.2 的 Mac 版本配对见官方说明：
https://pytorch.org/get-started/previous-versions/#v222

不要直接改成 Python 3.13 或 NumPy 2.x。权重文件需要是当前仓库里的 state_dict 文件；使用 strict=True 加载，不忽略缺失参数。

## 3. 推荐运行顺序

```bash
cd "/Users/yucheng/Desktop/NUS-ISS/ISY5002-Practice Module/Polyp-PVT"
conda activate polyp-pvt-mac
```

先运行单元测试：

```bash
python e0_mac/test_e0.py
```

只审核数据，不加载模型（可选，因为每次推理前都会自动审核）：

```bash
python e0_mac/run_e0.py --audit-only
```

每个数据集各选一张做真实推理，共 5 张：

```bash
python e0_mac/run_e0.py --device cpu --smoke
```

正式 E0，推理全部 798 张：

```bash
python e0_mac/run_e0.py --device cpu
```

每次自动创建新的 `runs/e0_日期时间/`，不覆盖已有结果。`--smoke` 的结果明确标记 SMOKE，不能当作正式 E0 分数。

从任意文件夹运行的写法：

```bash
python "/保存位置/e0_mac/run_e0.py" \
  --project "/Users/yucheng/Desktop/NUS-ISS/ISY5002-Practice Module/Polyp-PVT" \
  --device cpu
```

可自定义新输出目录及 CPU 线程数：

```bash
python e0_mac/run_e0.py --device cpu --threads 4 --output "runs/e0_cpu_run1"
```

输出目录必须尚不存在。若中断，会保存 FAILED 或 INTERRUPTED 状态以及已有逐图记录；当前版本不支持断点续跑，请换一个新目录重跑，避免混合不同运行记录。

## 4. 数据审核内容

读取五个命名子集，按原仓库的合并命名规则 `原文件名主体 + 数据集名称` 找到 `test` 内的样本。要求：

- 图像和 mask 文件名一一配对；
- 五个子集分别为 100 / 62 / 380 / 196 / 60 对，总共 798 对；
- 所有合并样本都能且只能映射到一个来源；
- 合并图像与对应源图像的解码 RGB 像素完全一致；
- 二值 GT mask 完全一致、图像与 mask 尺寸一致。

不能只比较文件哈希，因为相同图像可以具有不同的 PNG 编码。审核不通过就停止，不会默默丢弃失败样本。审核只验证目录合并正确，不证明患者级独立或既有 checkpoint 没看过测试集。

## 5. 本轮固定评估协议

1. RGB 图像用 Pillow bilinear resize 到 352×352，转 float32 /255。
2. 使用 ImageNet mean `[0.485,0.456,0.406]`、std `[0.229,0.224,0.225]`。
3. 调用原模型，取两路 logits 相加。
4. 使用 bilinear、`align_corners=False` 将合并 logits 恢复原图尺寸。
5. sigmoid 得到 foreground probability，不做逐图 min–max。
6. 保存原尺寸 float32 概率图，以 `>=0.5` 生成二值 mask。
7. 计算每张图的 Dice / IoU，再作图像等权平均。

图像等权总体均值不是五个子集均值的等权平均。默认同时保存原有 min–max + 四舍五入 8 位 PNG 的参考结果，并单独标为 legacy；两种协议不可混用。

GT 为灰度 0/1 编码时使用 `>0`，普通 8 位 mask 使用 `>127.5`。本协议双方均为空时 Dice / IoU=1，仅一方为空时为0。legacy 沿用旧脚本的零交集记0约定。当前测试数据均为息肉图像，负样本实验需另外设计。

本包输出固定阈值 Dice / IoU，不计算在测试集上搜索最佳阈值的 maxDic，也不将 256 个阈值平均的 meanDic 混称为固定阈值 Dice。

## 6. 输出文件

```
runs/e0_日期时间/
  audit.json              # 全量数据审核
  manifest.json           # 798个样本、来源、路径、尺寸和SHA256
  run_info.json           # 环境、状态、权重/代码hash、参数、时间范围
  per_image.csv           # 每张图的新协议和legacy指标、耗时、结果路径
  summary.csv             # ALL + 五个子集的汇总，适合直接做报告表格
  metrics.json            # 同一汇总的JSON版本及完整协议
  probabilities/数据集/样本.npz  # float32原始概率，后续fusion用这个
  masks/数据集/样本.png          # 新协议二值mask，0/255
  legacy_minmax/数据集/样本.png  # 旧协议可视化概率图，不能用于fusion
```

`run_info.json` 中 `status=COMPLETED`、`run_kind=FULL_E0`、`processed_count=798` 才表示正式全量结束。失败运行的部分 CSV 不能作为全量报告。

读取融合用概率：

```python
import numpy as np
with np.load("runs/e0_.../probabilities/Kvasir/样本名.npz", allow_pickle=False) as f:
    p_pvt = f["probability"]  # float32, H×W, [0,1]
# E2输出应使用相同sample_id、原图坐标和前景语义。
# p_fused = 0.5 * p_unet + 0.5 * p_pvt
```

### 耗时如何解释

`forward_ms` 仅包含模型前向，batch=1，默认3次warmup，在计时前后同步设备；不含解码、预处理、后处理、评价或写文件。`analysis_loop_seconds` 包含所有选定样本的这些操作，但不含数据审核、加载权重和warmup。不能把 forward 时间称作网页端到端实时延迟。

## 7. Mac 设备选择

CPU 为已验证参考路径。`--device auto` 在 Mac 上仍使用 CPU，避免旧模型的 MPS 算子兼容性差异。

若你的环境显示 `torch.backends.mps.is_available()` 为 True，可选尝试：

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python e0_mac/run_e0.py --device mps --smoke
```

MPS 是可选路径，未与 CPU 一样经过此包的完整验证；若报算子不支持，直接用 CPU。即使成功也应与 CPU 比较同样本输出，并在报告中记录设备和回退设置。不要将混合 CPU 回退的时间描述为纯 GPU 时间。

## 8. 常见问题

- `No module named timm`：确认已激活正确环境，使用该环境的 `python -m pip install -r ...`。
- `Not a Polyp-PVT repository`：进入项目根目录，或显式提供 `--project`。
- `Dataset audit failed`：读取本次 `audit.json`，不要绕过数据不匹配。
- `FileExistsError`：自定义输出目录已存在，换一个新名称即可。
- 权重加载错误：核对两个原有 `.pth` 文件，脚本不会自动下载或接受部分参数加载。
- 新 Dice 与旧0.9204不同：主协议去除了逐图 min–max，变化是预期可能性；legacy列用于核对旧流程。
- 首次 `lib/pvt.py` 的 `upsample` 弃用提示：来自原实现，脚本保留原架构，可继续运行。

正式实验完成后，优先查看 `summary.csv`、`run_info.json` 和 `audit.json`，再决定 E1 的下一步。

## 9. 本机数据的额外审核发现

当前 798 个条目中存在 76 组完全相同的解码图像，共对应 701 种不同的解码图像内容。其中 58 组跨 CVC-300 与 CVC-ColonDB，其余重复发生在子集内部。这里统计的是图像内容，不是患者数。

按本次约定，脚本保留全部原始测试条目，不自动去重、不更改划分；`ALL` 是 798 条目的加权汇总，不能解释为 798 个独立观察。分数据集结果仍分别输出。`audit.json` 记录重复组数量，`manifest.json` 中的 `decoded_rgb_sha256`、height、width 支持进一步定位重复来源。正式报告应说明这一情况。
