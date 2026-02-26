# CNIBP Paper Reproduction (Colab-first)

本目录用于复现论文：`10.1038/s41598-025-07087-2`，优先保证与论文设置一致。

## 1. 项目结构
- `configs/paper_repro.json`: 论文复现配置
- `src/cnibp_repro/io_mat.py`: 读取 Google Drive 中 `Part_0.mat` 到 `Part_4.mat`
- `src/cnibp_repro/preprocess.py`: 预处理与清洗（去趋势、时长过滤、ABP阈值、平线检测、切片）
- `src/cnibp_repro/model.py`: 3CNN + 2BiLSTM + Attention
- `src/cnibp_repro/train.py`: 5-fold 训练与评估（训练内10%验证）
- `src/cnibp_repro/qa.py`: 抽样波形图、标签分布、预处理统计
- `src/cnibp_repro/run_repro.py`: 主入口
- `scripts/run_colab.sh`: Colab 一键执行脚本
- `notebooks/cnibp_colab_repro.ipynb`: Colab notebook 入口

## 2. 与论文对齐的关键设置
- 采样率：125 Hz
- 记录过滤：`ABP > 200 mmHg` 删除；`duration < 8 min` 删除
- 切窗：`8.192s`，`75% overlap`
- 标签：每窗 ABP 的 systolic/diastolic 点均值
- 网络：3CNN + 2BiLSTM + Attention
- 训练：5-fold，每折 `80% train / 20% test`，训练内再划 `10%` 做验证

## 3. 运行方式（Colab 推荐）
在 Colab 中：

```bash
# 1) 挂载 Google Drive（在 notebook 中执行）
# 2) 保证仓库已在 /content/Lab

cd /content/Lab/06_experiments/cnibp/repro_ppg_bp
bash scripts/run_colab.sh
```

默认读取：`/content/drive/MyDrive/bp_kachuee_cach/Part_0.mat ... Part_4.mat`

默认输出：`/content/drive/MyDrive/cnibp_repro_outputs/run_YYYYMMDD_HHMMSS/`

## 4. 输出工件
每次运行会生成：
- `resolved_config.json`
- `preprocess/preprocess_stats.json`
- `preprocess/segments_meta.csv`
- `preprocess/qa/sample_waveforms.png`
- `preprocess/qa/target_distribution.png`
- `preprocessed_dataset.npz`
- `train/fold_*/history.csv`
- `train/fold_*/test_metrics.json`
- `train/fold_metrics.csv`
- `train/summary.json`

## 5. 质量检查节点（你要求的 Codex 检查环节）
- 节点A（读盘后）：确认 MAT 键结构可解析
- 节点B（预处理后）：检查清洗统计与抽样波形图
- 节点C（训练前）：检查目标分布是否异常偏斜
- 节点D（每折完成）：检查 `fold_metrics.csv` 是否稳定
- 节点E（最终）：汇总 MAE/MSE 与论文值对齐偏差

## 6. 关于“Codex 直接控制 VSCode/Colab 插件”
当前我不能直接操作你本地 VSCode 图形界面，也不能直接点击 Colab 插件按钮。
可行模式是：
1. 我在本地仓库生成和修改代码。
2. 你在 Colab 执行 notebook/命令。
3. 你把报错或日志贴给我，我继续自动修复。

这个模式下仍然可以做到“高自动化 + 你必要时手工介入”。

## 7. 下一步建议
先按论文原始策略跑通一版（`subject_level_split=false`）。跑完后再开第二版严格切分（`subject_level_split=true`）评估是否存在性能回落。
