from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from .io_mat import load_records
from . import __version__
from .preprocess import PreprocessConfig, preprocess_records
from .qa import save_meta, save_preprocess_report, save_sample_waveforms, save_target_hist
from .train import TrainConfig, train_5fold


def _cfg_sig_pre(cfg: dict) -> dict:
    keys = [
        "fs",
        "window_sec",
        "overlap",
        "min_duration_sec",
        "abp_max",
        "flatline_std_threshold",
        "ppg_norm_mode",
        "zscore_eps",
        "abp_label_mode",
        "abp_filter_mode",
        "abp_lowpass_hz",
        "abp_peak_distance_sec",
        "abp_peak_prominence",
        "sbp_min",
        "sbp_max_label",
        "dbp_min",
        "dbp_max_label",
        "pulse_pressure_min",
        "pulse_pressure_max",
    ]
    return {k: cfg.get(k) for k in keys}


def _cfg_sig_train(cfg: dict) -> dict:
    keys = [
        "seed",
        "n_splits",
        "batch_size",
        "max_epochs",
        "patience",
        "lr",
        "subject_level_split",
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "prefetch_factor",
        "use_amp",
    ]
    return {k: cfg.get(k) for k in keys}


def _load_run_cfg(run_dir: Path) -> dict | None:
    p = run_dir / "resolved_config.json"
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _detect_git_sha() -> str:
    try:
        cur = Path(__file__).resolve()
        repo_root = None
        for p in [cur.parent, *cur.parents]:
            if (p / ".git").exists():
                repo_root = p
                break
        if repo_root is None:
            return "unknown"
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return out if out else "unknown"
    except Exception:
        return "unknown"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce CNIBP paper with Colab/Drive workflow")
    p.add_argument("--drive_root", type=str, required=True, help="Google Drive folder containing Part_0.mat ... Part_4.mat")
    p.add_argument("--parts", type=str, default="Part_0.mat,Part_1.mat,Part_2.mat,Part_3.mat,Part_4.mat")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--output_root", type=str, required=True)
    p.add_argument("--reuse_preprocessed", action="store_true")
    p.add_argument(
        "--reuse_from_run_dir",
        type=str,
        default="",
        help="Reuse preprocessing artifacts from an existing run dir (contains preprocess/manifest+meta).",
    )
    p.add_argument(
        "--resume_from_run_dir",
        type=str,
        default="",
        help="Resume training in an existing run dir (skip completed folds and continue interrupted fold).",
    )
    return p.parse_args()


def _build_memmaps_from_chunks(chunk_paths: list[Path], pre_dir: Path) -> Tuple[np.memmap, np.memmap]:
    counts = []
    feat_dim = None
    tgt_dim = 2
    for cp in chunk_paths:
        d = np.load(cp)
        x = d["X"]
        y = d["y"]
        if feat_dim is None:
            feat_dim = int(x.shape[1])
        tgt_dim = int(y.shape[1])
        counts.append(int(x.shape[0]))

    total_n = int(sum(counts))
    x_path = pre_dir / "X.float16.mmap"
    y_path = pre_dir / "y.float32.mmap"
    manifest = {
        "x_path": str(x_path),
        "x_dtype": "float16",
        "x_shape": [total_n, int(feat_dim)],
        "y_path": str(y_path),
        "y_dtype": "float32",
        "y_shape": [total_n, int(tgt_dim)],
    }

    X_m = np.memmap(x_path, dtype=np.float16, mode="w+", shape=(total_n, int(feat_dim)))
    y_m = np.memmap(y_path, dtype=np.float32, mode="w+", shape=(total_n, int(tgt_dim)))

    s = 0
    for cp in chunk_paths:
        d = np.load(cp)
        x = d["X"].astype(np.float16, copy=False)
        y = d["y"].astype(np.float32, copy=False)
        e = s + len(x)
        X_m[s:e] = x
        y_m[s:e] = y
        s = e

    X_m.flush()
    y_m.flush()
    with (pre_dir / "preprocessed_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return X_m, y_m


def _load_memmaps_from_manifest(pre_dir: Path) -> Tuple[np.memmap, np.memmap]:
    with (pre_dir / "preprocessed_manifest.json").open("r", encoding="utf-8") as f:
        m = json.load(f)
    if not Path(m["x_path"]).exists() or not Path(m["y_path"]).exists():
        raise FileNotFoundError(
            f"Memmap files not found for manifest in {pre_dir}: {m.get('x_path')} , {m.get('y_path')}"
        )
    X_m = np.memmap(m["x_path"], dtype=np.float16, mode="r", shape=tuple(m["x_shape"]))
    y_m = np.memmap(m["y_path"], dtype=np.float32, mode="r", shape=tuple(m["y_shape"]))
    return X_m, y_m


def main() -> None:
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    print(
        f"[VERSION] cnibp_repro={__version__} git={_detect_git_sha()} file={Path(__file__).resolve()}",
        flush=True,
    )

    out_root = Path(args.output_root)

    # Validate reuse/resume config compatibility to avoid cross-experiment contamination.
    reuse_from_run_dir = Path(args.reuse_from_run_dir) if args.reuse_from_run_dir else None
    if reuse_from_run_dir is not None:
        prev_cfg = _load_run_cfg(reuse_from_run_dir)
        if prev_cfg is None or _cfg_sig_pre(prev_cfg) != _cfg_sig_pre(cfg):
            print(
                f"[WARN] reuse_from_run_dir incompatible with current preprocessing config; ignoring: {reuse_from_run_dir}",
                flush=True,
            )
            reuse_from_run_dir = None

    resume_from_run_dir = Path(args.resume_from_run_dir) if args.resume_from_run_dir else None
    if resume_from_run_dir is not None:
        prev_cfg = _load_run_cfg(resume_from_run_dir)
        if prev_cfg is None or _cfg_sig_pre(prev_cfg) != _cfg_sig_pre(cfg) or _cfg_sig_train(prev_cfg) != _cfg_sig_train(cfg):
            print(
                f"[WARN] resume_from_run_dir incompatible with current config; ignoring: {resume_from_run_dir}",
                flush=True,
            )
            resume_from_run_dir = None

    if resume_from_run_dir is not None:
        run_dir = resume_from_run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        run_dir = out_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    parts = [x.strip() for x in args.parts.split(",") if x.strip()]
    drive_root = Path(args.drive_root)
    mat_files: list[Path] = []
    for p in parts:
        pp = Path(p)
        cand = pp if pp.is_absolute() else (drive_root / pp)
        if cand.exists():
            mat_files.append(cand)
            continue

        # Fallback: resolve by basename under drive_root recursively.
        matches = list(drive_root.rglob(pp.name))
        if len(matches) == 1:
            print(f"[WARN] part not found at expected path; resolved by search: {matches[0]}", flush=True)
            mat_files.append(matches[0])
            continue
        if len(matches) > 1:
            print(f"[WARN] multiple matches for {pp.name}; using first: {matches[0]}", flush=True)
            mat_files.append(matches[0])
            continue
        raise FileNotFoundError(
            f"Part file not found: {p}. Checked: {cand} and recursive search under {drive_root}"
        )

    pre_dir = run_dir / "preprocess"
    reuse_pre_dir = pre_dir
    if reuse_from_run_dir is not None:
        reuse_pre_dir = reuse_from_run_dir / "preprocess"
    elif resume_from_run_dir is not None:
        reuse_pre_dir = resume_from_run_dir / "preprocess"

    meta_path = reuse_pre_dir / "segments_meta.csv"
    manifest_path = reuse_pre_dir / "preprocessed_manifest.json"

    can_reuse = False
    if (args.reuse_preprocessed or reuse_from_run_dir is not None) and manifest_path.exists() and meta_path.exists():
        try:
            print(f"[INFO] reusing preprocessed data from: {reuse_pre_dir}", flush=True)
            X, y = _load_memmaps_from_manifest(reuse_pre_dir)
            meta = pd.read_csv(meta_path)
            can_reuse = True
        except Exception as e:
            print(f"[WARN] reuse requested but not usable ({e}); fallback to fresh preprocessing.", flush=True)

    if not can_reuse:
        p_cfg = PreprocessConfig(
            fs=int(cfg["fs"]),
            window_sec=float(cfg["window_sec"]),
            overlap=float(cfg["overlap"]),
            min_duration_sec=int(cfg["min_duration_sec"]),
            abp_max=float(cfg["abp_max"]),
            flatline_std_threshold=float(cfg["flatline_std_threshold"]),
            ppg_norm_mode=str(cfg.get("ppg_norm_mode", "none")),
            zscore_eps=float(cfg.get("zscore_eps", 1e-6)),
            abp_label_mode=str(cfg.get("abp_label_mode", "paper")),
            abp_filter_mode=str(cfg.get("abp_filter_mode", "none")),
            abp_lowpass_hz=float(cfg.get("abp_lowpass_hz", 12.0)),
            abp_peak_distance_sec=float(cfg.get("abp_peak_distance_sec", 0.25)),
            abp_peak_prominence=float(cfg.get("abp_peak_prominence", 2.0)),
            sbp_min=float(cfg.get("sbp_min", 70.0)),
            sbp_max_label=float(cfg.get("sbp_max_label", 220.0)),
            dbp_min=float(cfg.get("dbp_min", 40.0)),
            dbp_max_label=float(cfg.get("dbp_max_label", 140.0)),
            pulse_pressure_min=float(cfg.get("pulse_pressure_min", 10.0)),
            pulse_pressure_max=float(cfg.get("pulse_pressure_max", 120.0)),
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
            "segments_removed_flatline": 0,
            "segments_removed_bad_label_qc": 0,
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
        X, y = _build_memmaps_from_chunks(chunk_paths, pre_dir)
        meta = pd.concat(meta_parts, ignore_index=True)

        save_preprocess_report(stats_total, pre_dir)
        save_meta(meta, pre_dir)
        save_sample_waveforms(X, y, pre_dir / "qa", fs=p_cfg.fs)
        save_target_hist(y, pre_dir / "qa")

    t_cfg = TrainConfig(
        seed=int(cfg["seed"]),
        n_splits=int(cfg["n_splits"]),
        batch_size=int(cfg["batch_size"]),
        max_epochs=int(cfg["max_epochs"]),
        patience=int(cfg["patience"]),
        lr=float(cfg["lr"]),
        subject_level_split=bool(cfg["subject_level_split"]),
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=bool(cfg.get("pin_memory", True)),
        persistent_workers=bool(cfg.get("persistent_workers", True)),
        prefetch_factor=int(cfg.get("prefetch_factor", 2)),
        use_amp=bool(cfg.get("use_amp", True)),
        resume=bool(resume_from_run_dir),
        max_folds=int(cfg.get("max_folds", 0)),
    )

    groups = meta["record_id"].to_numpy() if t_cfg.subject_level_split else None
    metrics = train_5fold(X, y, out_dir=run_dir / "train", cfg=t_cfg, groups=groups)
    print(metrics)
    print(f"Run artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
