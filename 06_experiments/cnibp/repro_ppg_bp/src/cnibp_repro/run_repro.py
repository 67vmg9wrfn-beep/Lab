from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

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
    pre_dir = run_dir / "preprocess"
    meta_path = pre_dir / "segments_meta.csv"

    if args.reuse_preprocessed and npz_path.exists() and meta_path.exists():
        data = np.load(npz_path)
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.float32)

        meta = pd.read_csv(meta_path)
    else:
        p_cfg = PreprocessConfig(
            fs=int(cfg["fs"]),
            window_sec=float(cfg["window_sec"]),
            overlap=float(cfg["overlap"]),
            min_duration_sec=int(cfg["min_duration_sec"]),
            abp_max=float(cfg["abp_max"]),
            flatline_std_threshold=float(cfg["flatline_std_threshold"]),
        )
        chunks_dir = pre_dir / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)

        stats_total = {
            "records_total": 0,
            "records_kept": 0,
            "records_removed_short": 0,
            "records_removed_abp_high": 0,
            "records_removed_flatline": 0,
            "segments_total": 0,
            "segments_kept": 0,
            "segments_removed_bad_target": 0,
        }
        chunk_paths = []
        meta_parts = []

        # Process each MAT part independently to reduce memory peak.
        for i, mf in enumerate(mat_files, start=1):
            print(f"[INFO] loading {mf} ({i}/{len(mat_files)})", flush=True)
            records_i = load_records([mf])
            print(f"[INFO] records parsed from {mf.name}: {len(records_i)}", flush=True)
            X_i, y_i, meta_i, stats_i = preprocess_records(records_i, p_cfg)
            print(
                f"[INFO] segments kept from {mf.name}: {len(X_i)} (shape={X_i.shape})",
                flush=True,
            )

            for k in stats_total:
                stats_total[k] += int(stats_i.get(k, 0))

            chunk_path = chunks_dir / f"chunk_{i:02d}_{mf.stem}.npz"
            np.savez_compressed(chunk_path, X=X_i, y=y_i)
            chunk_paths.append(chunk_path)
            meta_parts.append(meta_i)

            # Release per-file arrays before next file.
            del records_i, X_i, y_i, meta_i, stats_i

        print(f"[INFO] combining {len(chunk_paths)} preprocessed chunks", flush=True)
        X_parts = []
        y_parts = []
        for cp in chunk_paths:
            d = np.load(cp)
            X_parts.append(d["X"].astype(np.float32))
            y_parts.append(d["y"].astype(np.float32))

        X = np.concatenate(X_parts, axis=0)
        y = np.concatenate(y_parts, axis=0)
        meta = pd.concat(meta_parts, ignore_index=True)

        save_preprocess_report(stats_total, pre_dir)
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
