from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, KFold
from torch.utils.data import DataLoader, Dataset

from .model import CNNBiLSTMAttnRegressor


class IndexedArrayDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, indices: np.ndarray):
        self.X = X
        self.y = y
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        x_np = np.array(self.X[idx], dtype=np.float32, copy=True)
        y_np = np.array(self.y[idx], dtype=np.float32, copy=True)
        np.nan_to_num(x_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(y_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(x_np), torch.from_numpy(y_np), idx


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Post-hoc analysis for an existing CNIBP run")
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--batch_size", type=int, default=256)
    return p.parse_args()


def _load_manifest(pre_dir: Path) -> tuple[np.memmap, np.memmap]:
    manifest_path = pre_dir / "preprocessed_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        m = json.load(f)
    X = np.memmap(m["x_path"], dtype=np.float16, mode="r", shape=tuple(m["x_shape"]))
    y = np.memmap(m["y_path"], dtype=np.float32, mode="r", shape=tuple(m["y_shape"]))
    return X, y


def _split_generator(seed: int, n_splits: int, n: int, groups: np.ndarray | None):
    if groups is None:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        yield from splitter.split(np.arange(n))
    else:
        splitter = GroupKFold(n_splits=n_splits)
        yield from splitter.split(np.arange(n), groups=groups)


def _to_loader(X: np.ndarray, y: np.ndarray, indices: np.ndarray, batch_size: int) -> DataLoader:
    ds = IndexedArrayDataset(X, y, indices)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)


def _pick_device(user_device: str) -> str:
    if user_device:
        return user_device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _predict(model: torch.nn.Module, loader: DataLoader, device: str) -> pd.DataFrame:
    rows = []
    model.eval()
    with torch.no_grad():
        for xb, yb, idxb in loader:
            xb = xb.to(device)
            pred, _ = model(xb)
            pred_np = pred.detach().cpu().numpy()
            y_np = yb.detach().cpu().numpy()
            idx_np = idxb.detach().cpu().numpy()
            for idx, yt, yp in zip(idx_np, y_np, pred_np):
                rows.append(
                    {
                        "sample_index": int(idx),
                        "actual_sbp": float(yt[0]),
                        "actual_dbp": float(yt[1]),
                        "pred_sbp": float(yp[0]),
                        "pred_dbp": float(yp[1]),
                    }
                )
    df = pd.DataFrame(rows)
    df["err_sbp"] = df["pred_sbp"] - df["actual_sbp"]
    df["err_dbp"] = df["pred_dbp"] - df["actual_dbp"]
    df["abs_err_sbp"] = df["err_sbp"].abs()
    df["abs_err_dbp"] = df["err_dbp"].abs()
    return df


def _summarize_bins(df: pd.DataFrame, value_col: str, err_col: str, bins: Iterable[float], labels: list[str]) -> pd.DataFrame:
    work = df.copy()
    work["range"] = pd.cut(work[value_col], bins=bins, labels=labels, right=False, include_lowest=True)
    out = (
        work.groupby("range", observed=False)[err_col]
        .agg(["count", "mean", "std"])
        .reset_index()
        .rename(columns={"mean": "mae", "std": "mae_std"})
    )
    return out


def _load_existing_predictions(analysis_dir: Path, fold: int) -> pd.DataFrame | None:
    pred_path = analysis_dir / f"fold_{fold}_predictions.csv.gz"
    if not pred_path.exists():
        return None
    df = pd.read_csv(pred_path)
    if "fold" not in df.columns:
        df["fold"] = fold
    return df


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    pre_dir = run_dir / "preprocess"
    train_dir = run_dir / "train"
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "resolved_config.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    meta = pd.read_csv(pre_dir / "segments_meta.csv")
    X, y = _load_manifest(pre_dir)
    groups = meta["record_id"].to_numpy() if bool(cfg.get("subject_level_split", False)) else None
    device = _pick_device(args.device)

    all_fold_metrics = []
    all_preds = []

    for fold, (_, test_idx) in enumerate(
        _split_generator(int(cfg["seed"]), int(cfg["n_splits"]), len(meta), groups),
        start=1,
    ):
        existing_pred_df = _load_existing_predictions(analysis_dir, fold)
        if existing_pred_df is not None:
            pred_df = existing_pred_df
            fold_metric = {
                "fold": fold,
                "mae_sbp": float(pred_df["abs_err_sbp"].mean()),
                "mae_dbp": float(pred_df["abs_err_dbp"].mean()),
                "me_sbp": float(pred_df["err_sbp"].mean()),
                "me_dbp": float(pred_df["err_dbp"].mean()),
                "rmse_sbp": float(np.sqrt(np.mean(pred_df["err_sbp"] ** 2))),
                "rmse_dbp": float(np.sqrt(np.mean(pred_df["err_dbp"] ** 2))),
                "n_test": int(len(pred_df)),
            }
            all_fold_metrics.append(fold_metric)
            all_preds.append(pred_df)
            print(f"[INFO] fold {fold} analysis already exists; skipping recompute.", flush=True)
            continue

        fold_dir = train_dir / f"fold_{fold}"
        best_path = fold_dir / "best.pt"
        if not best_path.exists():
            continue

        model = CNNBiLSTMAttnRegressor().to(device)
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state)

        test_loader = _to_loader(X, y, np.asarray(test_idx, dtype=np.int64), args.batch_size)
        pred_df = _predict(model, test_loader, device)
        pred_df["fold"] = fold
        pred_df = pred_df.merge(
            meta.reset_index(names="sample_index")[["sample_index", "source_file", "record_id", "start", "end"]],
            on="sample_index",
            how="left",
        )

        sbp_bins = [0, 90, 120, 140, 160, 300]
        sbp_labels = ["<90", "90-119", "120-139", "140-159", ">=160"]
        dbp_bins = [0, 60, 80, 90, 100, 200]
        dbp_labels = ["<60", "60-79", "80-89", "90-99", ">=100"]

        sbp_summary = _summarize_bins(pred_df, "actual_sbp", "abs_err_sbp", sbp_bins, sbp_labels)
        sbp_summary["fold"] = fold
        sbp_summary["target"] = "SBP"
        dbp_summary = _summarize_bins(pred_df, "actual_dbp", "abs_err_dbp", dbp_bins, dbp_labels)
        dbp_summary["fold"] = fold
        dbp_summary["target"] = "DBP"

        fold_metric = {
            "fold": fold,
            "mae_sbp": float(pred_df["abs_err_sbp"].mean()),
            "mae_dbp": float(pred_df["abs_err_dbp"].mean()),
            "me_sbp": float(pred_df["err_sbp"].mean()),
            "me_dbp": float(pred_df["err_dbp"].mean()),
            "rmse_sbp": float(np.sqrt(np.mean(pred_df["err_sbp"] ** 2))),
            "rmse_dbp": float(np.sqrt(np.mean(pred_df["err_dbp"] ** 2))),
            "n_test": int(len(pred_df)),
        }
        all_fold_metrics.append(fold_metric)
        all_preds.append(pred_df)

        pred_df.to_csv(analysis_dir / f"fold_{fold}_predictions.csv.gz", index=False, compression="gzip")
        pd.concat([sbp_summary, dbp_summary], ignore_index=True).to_csv(
            analysis_dir / f"fold_{fold}_range_error.csv", index=False
        )
        print(f"[INFO] fold {fold} analysis saved.", flush=True)

    if not all_preds:
        raise RuntimeError(f"No fold predictions generated from {run_dir}")

    all_pred_df = pd.concat(all_preds, ignore_index=True)
    all_pred_df.to_csv(analysis_dir / "all_folds_predictions.csv.gz", index=False, compression="gzip")

    fold_metrics_df = pd.DataFrame(all_fold_metrics)
    fold_metrics_df.to_csv(analysis_dir / "fold_level_error_summary.csv", index=False)

    sbp_bins = [0, 90, 120, 140, 160, 300]
    sbp_labels = ["<90", "90-119", "120-139", "140-159", ">=160"]
    dbp_bins = [0, 60, 80, 90, 100, 200]
    dbp_labels = ["<60", "60-79", "80-89", "90-99", ">=100"]

    global_sbp = _summarize_bins(all_pred_df, "actual_sbp", "abs_err_sbp", sbp_bins, sbp_labels)
    global_sbp["target"] = "SBP"
    global_dbp = _summarize_bins(all_pred_df, "actual_dbp", "abs_err_dbp", dbp_bins, dbp_labels)
    global_dbp["target"] = "DBP"
    pd.concat([global_sbp, global_dbp], ignore_index=True).to_csv(
        analysis_dir / "global_range_error_summary.csv", index=False
    )

    overall = {
        "mae_sbp_mean": float(all_pred_df["abs_err_sbp"].mean()),
        "mae_dbp_mean": float(all_pred_df["abs_err_dbp"].mean()),
        "me_sbp_mean": float(all_pred_df["err_sbp"].mean()),
        "me_dbp_mean": float(all_pred_df["err_dbp"].mean()),
        "rmse_sbp_mean": float(np.sqrt(np.mean(all_pred_df["err_sbp"] ** 2))),
        "rmse_dbp_mean": float(np.sqrt(np.mean(all_pred_df["err_dbp"] ** 2))),
        "n_total": int(len(all_pred_df)),
    }
    with (analysis_dir / "overall_error_summary.json").open("w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)

    print(f"[OK] analysis saved to: {analysis_dir}")
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
