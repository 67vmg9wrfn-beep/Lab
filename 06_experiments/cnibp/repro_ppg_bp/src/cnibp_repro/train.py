from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .model import CNNBiLSTMAttnRegressor


@dataclass
class TrainConfig:
    seed: int = 42
    n_splits: int = 5
    batch_size: int = 64
    max_epochs: int = 500
    patience: int = 30
    lr: float = 1e-3
    device: str = "cuda"
    subject_level_split: bool = False
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    use_amp: bool = True
    resume: bool = False
    max_folds: int = 0  # 0 means run all folds


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    mae = np.abs(err)
    mse = err ** 2
    return {
        "mae_sbp": float(mae[:, 0].mean()),
        "mae_dbp": float(mae[:, 1].mean()),
        "mse_sbp": float(mse[:, 0].mean()),
        "mse_dbp": float(mse[:, 1].mean()),
    }


class IndexedArrayDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, indices: np.ndarray):
        self.X = X
        self.y = y
        self.indices = indices.astype(np.int64)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        # Keep source arrays lightweight (possibly memmap/float16), cast per batch sample.
        x = torch.from_numpy(np.asarray(self.X[idx], dtype=np.float32))
        t = torch.from_numpy(np.asarray(self.y[idx], dtype=np.float32))
        return x, t


def _to_loader(
    X: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    cfg: TrainConfig,
) -> DataLoader:
    ds = IndexedArrayDataset(X, y, indices)
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": False,
        "num_workers": int(cfg.num_workers),
    }
    if cfg.num_workers > 0:
        kwargs["pin_memory"] = bool(cfg.pin_memory)
        kwargs["persistent_workers"] = bool(cfg.persistent_workers)
        kwargs["prefetch_factor"] = int(cfg.prefetch_factor)
    return DataLoader(ds, **kwargs)


def _run_epoch(model, loader, criterion, optimizer, device, train: bool, scaler, use_amp: bool):
    model.train(mode=train)
    losses = []
    ys = []
    ps = []
    amp_enabled = bool(use_amp and str(device).startswith("cuda"))

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            pred, _ = model(xb)
            loss = criterion(pred, yb)

        if train:
            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        losses.append(loss.detach().item())
        ys.append(yb.detach().cpu().numpy())
        ps.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(ps, axis=0)
    m = _metrics(y_true, y_pred)
    m["loss"] = float(np.mean(losses))
    return m


def _split_generator(cfg: TrainConfig, n: int, groups: np.ndarray | None):
    if cfg.subject_level_split:
        if groups is None:
            raise ValueError("subject_level_split=True requires group ids")
        splitter = GroupKFold(n_splits=cfg.n_splits)
        yield from splitter.split(np.arange(n), groups=groups)
    else:
        splitter = KFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
        yield from splitter.split(np.arange(n))


def train_5fold(
    X: np.ndarray,
    y: np.ndarray,
    out_dir: Path,
    cfg: TrainConfig,
    groups: np.ndarray | None = None,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)

    device = cfg.device if torch.cuda.is_available() else "cpu"
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.use_amp and str(device).startswith("cuda")))

    fold_rows: List[Dict[str, float]] = []
    existing_metrics_path = out_dir / "fold_metrics.csv"
    if cfg.resume and existing_metrics_path.exists():
        prev = pd.read_csv(existing_metrics_path)
        fold_rows.extend(prev.to_dict(orient="records"))

    for fold, (train_idx, test_idx) in enumerate(_split_generator(cfg, len(X), groups), start=1):
        if int(cfg.max_folds) > 0 and fold > int(cfg.max_folds):
            break
        fold_dir = out_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_metric_path = fold_dir / "test_metrics.json"
        if cfg.resume and fold_metric_path.exists():
            with fold_metric_path.open("r", encoding="utf-8") as f:
                row = json.load(f)
            # Replace same fold if present.
            fold_rows = [r for r in fold_rows if int(r.get("fold", -1)) != fold]
            fold_rows.append(row)
            print(f"[INFO] fold {fold} already completed; skipping.", flush=True)
            continue

        # 训练集中再划10%做验证，复现论文流程。
        tr_idx, val_idx = train_test_split(
            train_idx, test_size=0.1, random_state=cfg.seed, shuffle=True
        )

        train_loader = _to_loader(X, y, tr_idx, cfg.batch_size, shuffle=True, cfg=cfg)
        val_loader = _to_loader(X, y, val_idx, cfg.batch_size, shuffle=False, cfg=cfg)
        test_loader = _to_loader(X, y, test_idx, cfg.batch_size, shuffle=False, cfg=cfg)

        model = CNNBiLSTMAttnRegressor().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

        best_val = float("inf")
        best_epoch = 0
        wait = 0
        history = []
        best_state = None
        start_epoch = 1
        last_ckpt_path = fold_dir / "last.ckpt"

        if cfg.resume and last_ckpt_path.exists():
            try:
                ckpt = torch.load(last_ckpt_path, map_location=device)
                model.load_state_dict(ckpt["model_state"])
                optimizer.load_state_dict(ckpt["optimizer_state"])
                if "scaler_state" in ckpt and ckpt["scaler_state"] is not None:
                    scaler.load_state_dict(ckpt["scaler_state"])
                best_val = float(ckpt.get("best_val", best_val))
                best_epoch = int(ckpt.get("best_epoch", best_epoch))
                wait = int(ckpt.get("wait", wait))
                start_epoch = int(ckpt.get("epoch", 0)) + 1
                if (fold_dir / "best.pt").exists():
                    best_state = torch.load(fold_dir / "best.pt", map_location="cpu")
                if (fold_dir / "history.csv").exists():
                    history = pd.read_csv(fold_dir / "history.csv").to_dict(orient="records")
                print(f"[INFO] resumed fold {fold} from epoch {start_epoch}.", flush=True)
            except Exception as e:
                print(f"[WARN] failed to resume fold {fold} from checkpoint: {e}", flush=True)
                start_epoch = 1
                history = []
                best_state = None
                best_val = float("inf")
                best_epoch = 0
                wait = 0

        for epoch in tqdm(range(start_epoch, cfg.max_epochs + 1), desc=f"fold{fold}"):
            train_m = _run_epoch(
                model, train_loader, criterion, optimizer, device, train=True, scaler=scaler, use_amp=cfg.use_amp
            )
            val_m = _run_epoch(
                model, val_loader, criterion, optimizer, device, train=False, scaler=scaler, use_amp=cfg.use_amp
            )

            row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_m.items()}, **{f"val_{k}": v for k, v in val_m.items()}}
            history.append(row)

            val_loss = float(val_m["loss"])
            if np.isfinite(val_loss) and val_loss < best_val:
                best_val = val_m["loss"]
                best_epoch = epoch
                wait = 0
                # Keep in-memory best weights so eval can continue even if Drive write fails.
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                try:
                    torch.save(best_state, fold_dir / "best.pt")
                except Exception as e:
                    print(f"[WARN] failed to save best.pt for fold {fold}: {e}", flush=True)
            else:
                wait += 1

            if wait >= cfg.patience:
                break

            try:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scaler_state": scaler.state_dict() if scaler is not None else None,
                        "best_val": best_val,
                        "best_epoch": best_epoch,
                        "wait": wait,
                    },
                    last_ckpt_path,
                )
            except Exception as e:
                print(f"[WARN] failed to save last checkpoint for fold {fold}: {e}", flush=True)

        hist_df = pd.DataFrame(history)
        hist_df.to_csv(fold_dir / "history.csv", index=False)

        best_path = fold_dir / "best.pt"
        if best_state is not None:
            model.load_state_dict(best_state)
        elif best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location=device))
        else:
            print(
                f"[WARN] no finite validation improvement in fold {fold}; evaluating last-epoch weights.",
                flush=True,
            )
        test_m = _run_epoch(
            model, test_loader, criterion, optimizer, device, train=False, scaler=scaler, use_amp=cfg.use_amp
        )

        fold_row = {
            "fold": fold,
            "best_epoch": best_epoch,
            **test_m,
            "n_train": int(len(tr_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
        }
        fold_rows.append(fold_row)

        with fold_metric_path.open("w", encoding="utf-8") as f:
            json.dump(fold_row, f, indent=2)

    # Deduplicate by fold (keep latest row for each fold)
    fold_rows_sorted = {}
    for r in fold_rows:
        fold_rows_sorted[int(r["fold"])] = r
    res = pd.DataFrame([fold_rows_sorted[k] for k in sorted(fold_rows_sorted.keys())])
    res.to_csv(out_dir / "fold_metrics.csv", index=False)

    avg = {
        "mae_sbp_mean": float(res["mae_sbp"].mean()),
        "mae_dbp_mean": float(res["mae_dbp"].mean()),
        "mse_sbp_mean": float(res["mse_sbp"].mean()),
        "mse_dbp_mean": float(res["mse_dbp"].mean()),
        "mae_sbp_std": float(res["mae_sbp"].std(ddof=0)),
        "mae_dbp_std": float(res["mae_dbp"].std(ddof=0)),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(avg, f, indent=2)

    return res
