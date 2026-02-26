# Colab 一键运行（不依赖本地路径映射）

你只需要做两次上传：

1. 上传代码包：
`/Users/codex/Lab/06_experiments/cnibp/repro_ppg_bp_colab_bundle.zip`

2. 确保数据在 Google Drive：
`/content/drive/MyDrive/bp_kachuee_cach/Part_0.mat ... Part_4.mat`

然后在 Colab 依次运行：

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
from google.colab import files
uploaded = files.upload()  # 选择 repro_ppg_bp_colab_bundle.zip
```

```bash
!unzip -o repro_ppg_bp_colab_bundle.zip -d /content
!python -m pip install -r /content/repro_ppg_bp/requirements_colab.txt
```

```python
import os
os.environ['PYTHONPATH'] = '/content/repro_ppg_bp/src:' + os.environ.get('PYTHONPATH','')
```

```bash
!python -m cnibp_repro.run_repro \
  --drive_root /content/drive/MyDrive/bp_kachuee_cach \
  --config /content/repro_ppg_bp/configs/paper_repro.json \
  --output_root /content/drive/MyDrive/cnibp_repro_outputs
```

输出会在：
`/content/drive/MyDrive/cnibp_repro_outputs/run_YYYYMMDD_HHMMSS/`

## Git 免上传版本（推荐）
使用 notebook：
`/Users/codex/Lab/06_experiments/cnibp/repro_ppg_bp/notebooks/cnibp_colab_git_run.ipynb`

只需在第一个单元设置：
- `GIT_REPO`
- `GIT_BRANCH`
- `PROJECT_SUBDIR`

然后顺序运行即可，不需要上传 zip。

## 自动调试友好模式
新版 notebook 已增加：
- 数据文件存在性预检查（Part_0~Part_4）
- 统一 `set -euo pipefail`，尽早失败
- 运行日志自动写入：`/content/drive/MyDrive/cnibp_repro_outputs/last_run.log`

如果报错，你只需要把 `last_run.log` 内容贴给我，我会继续自动修复代码。
