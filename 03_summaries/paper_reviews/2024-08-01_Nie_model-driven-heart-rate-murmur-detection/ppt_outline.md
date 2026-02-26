## 1. 标题页

- Model-Driven Heart Rate Estimation and Heart Murmur Detection Based on Phonocardiogram
- Jingping Nie et al., IEEE MLSP 2024
- 一句话结论：2dCNN在PCG上实现低MAE心率估计，并在MTL下保持95%+杂音检测准确率。

## 2. 研究问题

- 目标：在真实噪声环境下，同时做HR估计和murmur检测。
- 背景：数字听诊器/远程监测需要低侵入且稳定的声学算法。
- 缺口：传统方法抗噪与分布偏移能力弱。

## 3. 核心思路

- 用Mel/MFCC/PSD/RMS构建多视角声学输入。
- 比较TCNN-LSTM、2dCNN、2dCNN-Fusion后选2dCNN。
- 扩展2dCNN-MTL联合优化HR与murmur任务。

## 4. 方法框架

- 输入：5秒PCG片段（stride=1秒）。
- 输出：HR预测 + murmur分类。
- 流程图占位：`[插入方法流程图]`
- 关键模块：特征提取、共享卷积骨干、双任务头。

## 5. 实验设置

- 数据：CirCor DigiScope，3163录音，滑窗后23381片段。
- 划分：train/val/test=80/10/10（按受试者隔离）。
- 指标：MAE（HR），ACC/Precision/Recall（murmur）。
- 训练：对比不同任务权重与LR scheduler。

## 6. 关键结果

- 单任务2dCNN最佳：MAE=1.312 bpm。
- MTL：HR MAE约1.338-1.636 bpm，同时murmur ACC >95%。
- 最佳murmur准确率可达97.49%。
- 图表占位：`[插入主结果图表]`

## 7. 机制与消融

- 四特征联合优于单特征/部分组合。
- 2D卷积优于TCNN-LSTM（该任务下）。
- LR scheduler是关键增益来源之一。

## 8. 局限与风险

- 单数据集验证，跨域泛化证据不足。
- 低HR区间误差偏大，数据分布不均衡。
- 噪声分离与非稳态场景覆盖不足。

## 9. 是否值得投入

- 评分摘要：相关性4/创新3/严谨4/充分3/复现3/应用4。
- 结论：可关注，建议先做低成本复现再投入。

## 10. 下一步行动

- 30分钟：核验核心数字和复现要素。
- 1天：复现2dCNN四特征并对照LR调度。
- 1周：实现MTL并补域泛化实验。

## 风格约束

- 每页只讲一个核心信息。
- 优先使用可量化结果。
- 缺失信息统一标注：`论文未说明`。
