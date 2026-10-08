# 本次交付验证

日期：2026-10-04。

- 环境：本机已有的 `polyp-pvt-mac`，Python 3.10.21，ARM64。
- 包版本：torch 2.2.2、torchvision 0.17.2、timm 0.4.12、numpy 1.26.4、Pillow 10.2.0。
- 数据：现有 `Polyp-PVT/dataset/TestDataset`。
- 798 对合并样本通过一一映射、解码图像像素、二值标注和尺寸检查。
- 使用真实 `PolypPVT.pth` 在 CPU 上完成五张推理，每个子集一张。
- 复读五个保存的 float32 NPZ，核对概率尺寸、二值 mask、逐图 Dice 与 CSV 一致。
- smoke 中 Kvasir 样本的 legacy Dice 与此前保存的对应样本结果一致。
- 五个单元测试通过：已知重叠、空 mask 约定、min–max 改变阈值含义、图像等权汇总、尺寸错误。

本次仅为代码交付验证，未执行全部 798 张模型推理，也未测试 MPS。五张 smoke 的汇总不可用作 E0 正式结果。使用 README 中的全量命令执行正式实验。

数据审核还发现76组完全相同的解码图像，798条目对应701种不同图像内容；按用户指定保持原测试划分，不去重。详见 README 第9节。
