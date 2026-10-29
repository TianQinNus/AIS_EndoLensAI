# EndoLens AI
**Hybrid Polyp Segmentation and Interactive Colonoscopy Video Analysis**

面向肠镜视频的息肉分割与交互分析课程项目。我们的技术路线是训练 U-Net 基线，研究预训练 ResNet34 编码器与数据增强，再与现有 Polyp-PVT 参考模型进行决策层概率融合；后续将模型接入视频抽帧、候选帧展示、mask 人工修订和系统结果报告。

**当前进度（2026-10-08）：E0–E4 图像模型实验已完成；E0＋E4 融合为下一步计划。视频处理、实时性能和完整网页尚未实现/验证。** 现有结果页用于离线实验比较，系统不生成临床诊断。

## 1. 新成员从这里开始

```bash
git clone https://github.com/TianQinNus/EndoLensAI.git
cd EndoLensAI/SystemCode/model
python scripts/bootstrap_artifacts.py
```

`bootstrap_artifacts.py` 只将已提交的固定划分与 E0 参考元数据复制到本地 `runs/`，检查文件哈希，并拒绝覆盖不同内容。它不会下载数据或模型，也不会启动训练。

- [组员安装、运行和迁移指南](SystemCode/model/docs/TEAM_GUIDE.md)
- [模型结构、损失与评估协议](SystemCode/model/docs/MODEL_PROTOCOL.md)
- [当前实验结果与后续计划](SystemCode/model/docs/EXPERIMENT_STATUS.md)
- [单独共享的数据、权重和完整运行目录清单](Miscellaneous/model_artifacts/README.md)
- [组员协作约定](CONTRIBUTING.md)

**只有源码即可运行合成数据功能测试；正式训练需要恢复数据图片和预训练权重。** E4 开发还需要 E2-C 的完整参考运行及 `best.pth`；离线融合需要两个分支的完整概率和掩膜。GitHub 中的指标摘要不能代替这些运行目录。

## 2. 仓库结构

保留课程提交的四个顶层目录。模型脚本保持原来的相对结构，避免破坏实验源码哈希与恢复检查。

```text
EndoLensAI/
├── SystemCode/
│   └── model/                      所有模型命令的工作目录
│       ├── e0_mac/                 E0：固定 Polyp-PVT 审计、推理与评估
│       ├── lib/                    来自原 Polyp-PVT 的模型源码（保留归属）
│       ├── e1_mac/                 E1：原始 U-Net，Mac MPS/CPU 版本
│       ├── e1_windows/             E1：CUDA 训练与跨平台评估版本
│       ├── e2_windows/             E2：三方案开发对照、选型、全量训练
│       ├── v3_windows/             E3/V3：E0＋E2 固定等权离线概率融合
│       ├── e4_windows/             E4：强增强对照；core 是冻结的 E2 引擎
│       ├── scripts/                恢复共享元数据、检查源码、运行功能测试
│       ├── docs/                   环境、协议、实验状态、上游说明
│       ├── dataset/                本地恢复；Git 忽略
│       ├── runs/                   本地恢复/生成；Git 忽略
│       └── e2_weights/             本地 ResNet34 初始化权重；Git 忽略
├── ProjectReport/                  E0–E4 Word 实验报告
├── Miscellaneous/
│   ├── model_artifacts/            固定数据清单、E0 参考元数据、代码哈希
│   ├── model_results/              精选 CSV/JSON 指标与训练历史
│   └── template_reference/         原 HDB 课程模板归档，与本项目无关
├── Video/                          后续项目演示视频（当前未提交）
├── .gitignore
├── CONTRIBUTING.md
└── THIRD_PARTY_NOTICES.md
```

## 3. 当前模型路线与阶段结果

| 阶段 | 工作 | 当前状态 |
|---|---|---|
| E0 | 现有 Polyp-PVT 固定权重参考；统一概率和评估协议 | 完成 |
| E1 | 原始 U-Net 随机初始化，完整 1450 条目训练 100 轮 | 完成 |
| E2 | 原始 U-Net / 随机 ResNet34-U-Net / ImageNet 预训练 ResNet34-U-Net；内部验证选型 | 完成；预训练方案，第 67 轮预算，重新全量训练 |
| E3（V3） | E0＋E2 原尺寸概率等权融合，无新模型训练 | 完成；整体未超过 E0 |
| E4 | 相同预训练 ResNet34-U-Net 加强增强；100 轮开发、76 轮全量重训 | 完成；改善存在子集取舍 |
| E5（计划） | 固定 E0＋E4 等权概率融合，与 E0/E4/E3 对比 | 尚未运行；当前仓库未实现专用 E5 入口 |

| 模型 / 方案 | 798 条目平均 Dice | 平均 IoU |
|---|---:|---:|
| E0 Polyp-PVT 参考 | 0.8368 | 0.7630 |
| E2 预训练 ResNet34-U-Net | 0.7549 | 0.6862 |
| E3 E0＋E2 等权融合 | 0.8329 | 0.7606 |
| E4 强增强 ResNet34-U-Net | 0.7609 | 0.6903 |

E4 在 ETIS/Kvasir 改善，但其他三个子集下降，背景误分增加。暂时保留 E2/E4 两个候选，不把 E4 或融合自动设为最佳方案。[详细分析与后续判断标准](SystemCode/model/docs/EXPERIMENT_STATUS.md)。这些是本项目固定二值指标协议下的结果，不能直接与论文 MATLAB 分数等同。

## 4. Windows GPU 训练入口

从 `SystemCode/model/` 运行。建议 Python 3.11，采用已成功训练的 CUDA 12.8 环境：

```bat
conda create -n endolens-e2-win python=3.11 -y
conda activate endolens-e2-win
python -m pip install -r e2_windows/requirements-windows.txt --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r e2_windows/requirements-common.txt
python scripts/bootstrap_artifacts.py
python scripts/verify_imported_code.py
python scripts/run_checks.py
python e2_windows/check_device.py --device cuda --output runs/e2_gpu_device_proof.json
python e2_windows/run_e2.py download-weights
```

功能测试使用 CPU 合成数据，这是预期行为。E2/E4 正式训练强制 CUDA，没有 CPU 回退；需在真实 Windows GPU 上完成设备探针、数据核验和 smoke。Mac 可做结果分析/部分评估，不能运行这两阶段的正式训练。[完整训练与迁移命令](SystemCode/model/docs/TEAM_GUIDE.md)。

## 5. 数据、资产与使用边界

外层固定 1450 训练 / 798 测试条目，内部 1160 开发训练 / 290 验证。测试由 Kvasir 100、CVC-ClinicDB 62、CVC-ColonDB 380、ETIS 196、CVC-300 60 组成。合并 `TestDataset/test/` 也是现有审计的必要输入。

数据来源：[Polyp-PVT 官方准备说明](https://github.com/DengPingFan/Polyp-PVT#42-data-preparation)、[Kvasir-SEG](https://datasets.simula.no/kvasir-seg/)、[CVC-ClinicDB](https://polyp.grand-challenge.org/CVCClinicDB/)。为复用固定哈希，优先获取组内已有的同一版本数据，避免图片重编码与随意重新划分。数据/权重具体权限以其原始条款为准。

无完全相同训练/测试 RGB 重叠不证明患者级独立；798 测试条目包含重复内容且已被多轮查看。测试不用于搜索最佳阈值、轮次或融合权重。当前全部测试图像含息肉，尚不能评价无息肉视频帧误报或临床检测效能。

## 6. 项目报告、演示与团队

[ProjectReport](ProjectReport/) 收录 E0–E4 完整 Word 实验报告；[Miscellaneous/model_results](Miscellaneous/model_results/) 保存可直接查看的指标与历史。报告中旧机器路径属于历史记录，不是新成员的安装路径。

演示视频与最终项目报告待补充。成员姓名、学号及职责由小组确认后填写；没有沿用模板中的虚构成员信息。原模板内容保存在 [template_reference](Miscellaneous/template_reference/)。

## 7. 归属与许可

Polyp-PVT 作为既有参考模型，ResNet34/U-Net 为已有架构的项目实现；本项目不声称原创这些架构。我们的工作是训练对照、统一评估、决策融合与应用集成。

请阅读 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。Polyp-PVT 原 README 限制源码为科研和教育用途，商业使用需正式许可。仓库当前未额外授予统一 MIT/Apache 许可；第三方代码、数据与权重仍受各自条款约束。
