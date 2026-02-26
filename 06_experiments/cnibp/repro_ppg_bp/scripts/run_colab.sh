#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/content/Lab/06_experiments/cnibp/repro_ppg_bp"
DRIVE_DATA_ROOT="/content/drive/MyDrive/bp_kachuee_cach"
OUT_ROOT="/content/drive/MyDrive/cnibp_repro_outputs"

cd "$PROJECT_ROOT"

python -m pip install -r requirements_colab.txt
python -m pip install -e .

python -m cnibp_repro.run_repro \
  --drive_root "$DRIVE_DATA_ROOT" \
  --config "$PROJECT_ROOT/configs/paper_repro.json" \
  --output_root "$OUT_ROOT"
