# 模型资产与组内单独迁移清单

此目录共享固定数据清单和E0参考元数据，不包含图片、权重或预测概率。`e2_data_seed42/`七份JSON逐字保留；描述符及各文件的SHA可验证。E0的manifest/per_image/summary/audit逐字保留，run_info仅隐藏机器绝对路径；模型/清单哈希、指标和协议未变。指标分享目录的run_info也隐藏了机器路径，不用于恢复训练。

初始化后本地运行目录为 `SystemCode/model/`。资产保管人目前在组内提供原目录迁移；本仓库**没有假设已有云盘或Release下载链接**。获取后原样恢复以下相对路径：

| 工作 | 除源码和共享元数据之外需要恢复的内容 |
|---|---|
| CPU合成功能测试 | 不需要外部资产；安装依赖即可 |
| 查看进度/指标 | 不需要外部资产；读取model_results和ProjectReport |
| E2重新训练 | 完整dataset/；下载e2_weights/resnet34-b627a593.pth |
| E0重新推理 | dataset/TestDataset/五子集及合并test；model_pth/PolypPVT.pth；pretrained_pth/pvt_v2_b2.pth；E0依赖环境 |
| E4开发 | 上述E2输入＋完整runs/e2_dev_C_resnet_pretrained_seed42/，含checkpoints/best.pth；仅CSV不能代替 |
| E2/E4最终评估 | 相应完整final目录（checkpoints/final.pth、run_info、test_manifest等）、测试图片、E0参考元数据 |
| E3/V3复跑 | 完整E0目录（798概率NPZ、mask、manifest、per_image、run_info）；完整E2 final及其evaluation_cuda_20261005_231641_102143；合并test图片和mask |
| 下一步E5 | 完整E0与E4 final/evaluation_cuda_20261006_235221_022843概率和来源记录；专用E5代码尚未实现 |
| 原设备中断后resume | 对应完整run（last/best、优化器/RNG状态、历史等）、相同源码/数据/初始化及匹配环境；不要只迁final.pth或跨平台直接恢复 |

外层数据1450训练、798测试；内部1160/290。恢复原图，不重编码，不换同名但不同内容文件；校验失败应核对资产，不改代码绕过检查。

`WEIGHT_REGISTRY.json`记录既有模型/编码器检查点哈希及必要相对路径，**不是下载链接**。完整运行由负责实验的组员提供；后续可增加经过权限核对的组内共享位置及下载后校验说明。模型权重和数据原始许可见根目录THIRD_PARTY_NOTICES。

`CODE_IMPORT_MANIFEST.json`记录导入时的源码大小和哈希。`verify_imported_code.py`只核对冻结Python文件，文档可以正常更新。

完整runs默认不提交Git。完成新实验后应保留完整目录作本地/组内备份，再将经过审核的小型指标与报告复制到分享目录；不要把分享目录当成可resume的完整run。
