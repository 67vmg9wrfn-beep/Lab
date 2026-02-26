from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from .io_mat import load_records
from .preprocess import PreprocessConfig, preprocess_records
from .qa import save_meta, save_preprocess_report, save_sample_waveforms, save_target_hist
from .train import TrainConfig, train_5fold


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce CNIBP paper with Colab/Drive workflow")
    p.add_argument("--drive_root", type=str, required=True, help="Google Drive folder containing Part_0.mat ... Part_4.mat")
    p.add_argument("--parts", type=str, default="Part_0.mat,Part_1.mat,Part_2.mat,Part_3.mat,Part_4.mat")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--output_root", type=str, required=True)
    p.add_argument("--reuse_preprocessed", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    out_root = Path(args.output_root)
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    parts = [x.strip() for x in args.parts.split(",") if x.strip()]
    mat_files = [Path(args.drive_root) / p for p in parts]

    npz_path = run_dir / "preprocessed_dataset.npz"
    meta_path = run_dir / "preprocess" / "segments_meta.csv"

    if args.reuse_preprocessed and npz_path.exists() and meta_path.exists():
        data = np.load(npz_path)
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.float32)
        import pandas as pd

        meta = pd.read_csv(meta_path)
    else:
        records = load_records(mat_files)

        p_cfg = PreprocessConfig(
            fs=int(cfg["fs"]),
            window_sec=float(cfg["window_sec"]),
            overlap=float(cfg["overlap"]),
            min_duration_sec=int(cfg["min_duration_sec"]),
            abp_max=float(cfg["abp_max"]),
            flatline_std_threshold=float(cfg["flatline_std_threshold"]),
        )
        X, y, meta, stats = preprocess_records(records, p_cfg)

        pre_dir = run_dir / "preprocess"
        save_preprocess_report(stats, pre_dir)
        save_meta(meta, pre_dir)
        save_sample_waveforms(X, y, pre_dir / "qa", fs=p_cfg.fs)
        save_target_hist(y, pre_dir / "qa")

        np.savez_compressed(npz_path, X=X, y=y)

    t_cfg = TrainConfig(
        seed=int(cfg["seed"]),
        n_splits=int(cfg["n_splits"]),
        batch_size=int(cfg["batch_size"]),
        max_epochs=int(cfg["max_epochs"]),
        patience=int(cfg["patience"]),
        lr=float(cfg["lr"]),
        subject_level_split=bool(cfg["subject_level_split"]),
    )

    groups = meta["record_id"].to_numpy() if t_cfg.subject_level_split else None
    metrics = train_5fold(X, y, out_dir=run_dir / "train", cfg=t_cfg, groups=groups)
    print(metrics)
    print(f"Run artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
