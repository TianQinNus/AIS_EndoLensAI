# 模型与评估协议

## 数据与选择

原Polyp-PVT外层划分：1450训练、798测试条目。固定seed42 bundle：1160开发训练/290内部验证；开发选型完成后从对应初始参数重新训练全部1450，而不是续训开发best。E2选择67轮，E4选择76轮；100轮scheduler时间轴保留。测试不用于选择轮次、权重或阈值。

完全相同RGB内容分组可以减少直接重复跨分区，但缺少可靠患者/序列身份，因此不声称患者级独立；测试798条目对应701种解码RGB内容。内部验证可能已包含在原PVT训练数据中，不能将其作为两个分支共同独立的融合调参集。

## 模型

- E0：现有Polyp-PVT，352输入，两个输出logits相加后评估，固定参考权重。
- E1 / E2-A：原始U-Net，随机初始化，无归一化层。
- E2-B/C：ResNet34编码器＋U-Net解码器；B随机编码器，C为官方ImageNet初始化。编码器BatchNorm更新，解码器GroupNorm(8)，24,436,241参数。B/C是相同架构不同初始化，不是两个新架构。
- E4：同E2-C模型与开发预算，改变增强。最终E2/E4轮数不同，最终测试不能全部独立归因于增强。

输入RGB、Pillow bilinear缩放352×352、/255及ImageNet mean/std。标签最近邻插值；水平和垂直翻转各p=0.5。strong增加同步±15°旋转及仅图像亮度/对比度0.85–1.15；旋转不expand，边缘可能裁切/填充。

## 损失和优化

E2/E4：`loss = BCEWithLogits(logits, mask) + mean(per-image soft Dice loss)`，两项权重1:1。soft Dice采用sigmoid概率、smooth=1；训练损失为FP32，不使用阈值二值mask反向传播。

AdamW lr1e-4、weight_decay1e-4；batch2/累积4；5轮warmup后cosine；AMP float16前向与FP32损失，验证/测试FP32。梯度累积不扩大BatchNorm物理统计批量。

## 概率与指标接口

恢复原尺寸logits（bilinear、align_corners=False），再sigmoid；原尺寸float32 NPZ的键为`probability`，二维HxW、有限数值且位于[0,1]。二值mask是`probability >= 0.5`，PNG为0/255。不要将legacy逐图min–max图作为概率。

逐图 `Dice=2TP/(2TP+FP+FN)`、`IoU=TP/(TP+FP+FN)`，两者都空记1、仅一方空记0。分组算术平均；ALL按条目数量加权，不是五子集均值等权。MAE使用连续概率与二值GT。

E3：`p=0.5*p_E0+0.5*p_E2`，阈值0.5，分支冻结，不训练、不搜索配方。E5拟将E2替换E4并复用固定规则，尚未实现。

## 应用接口与待验证项

视频模块应保留timestamp/frame_index、原尺寸、模型标识、预处理版本、概率和mask；修订mask与原预测分开存储。抽帧间隔可配置，候选帧数不能当成病灶数量。含阴性帧的视频、事件覆盖、误报、两模型联合推理和端到端延迟尚未验证，不以单图forward均值宣称实时视频FPS。
