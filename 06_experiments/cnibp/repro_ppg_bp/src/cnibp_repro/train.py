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


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def _to_loader(X: np.ndarray, y: np.ndarray, indices: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = IndexedArrayDataset(X, y, indices)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(mode=train)
    losses = []
    ys = []
    ps = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)

        pred, _ = model(xb)
        loss = criterion(pred, yb)

        if train:
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

    fold_rows: List[Dict[str, float]] = []

    for fold, (train_idx, test_idx) in enumerate(_split_generator(cfg, len(X), groups), start=1):
        fold_dir = out_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # 训练集中再划10%做验证，复现论文流程。
        tr_idx, val_idx = train_test_split(
            train_idx, test_size=0.1, random_state=cfg.seed, shuffle=True
        )

        train_loader = _to_loader(X, y, tr_idx, cfg.batch_size, shuffle=True)
        val_loader = _to_loader(X, y, val_idx, cfg.batch_size, shuffle=False)
        test_loader = _to_loader(X, y, test_idx, cfg.batch_size, shuffle=False)

        model = CNNBiLSTMAttnRegressor().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

        best_val = float("inf")
        best_epoch = 0
        wait = 0
        history = []

        for epoch in tqdm(range(1, cfg.max_epochs + 1), desc=f"fold{fold}"):
            train_m = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
            val_m = _run_epoch(model, val_loader, criterion, optimizer, device, train=False)

            row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_m.items()}, **{f"val_{k}": v for k, v in val_m.items()}}
            history.append(row)

            if val_m["loss"] < best_val:
                best_val = val_m["loss"]
                best_epoch = epoch
                wait = 0
                torch.save(model.state_dict(), fold_dir / "best.pt")
            else:
                wait += 1

            if wait >= cfg.patience:
                break

        hist_df = pd.DataFrame(history)
        hist_df.to_csv(fold_dir / "history.csv", index=False)

        model.load_state_dict(torch.load(fold_dir / "best.pt", map_location=device))
        test_m = _run_epoch(model, test_loader, criterion, optimizer, device, train=False)

        fold_row = {
            "fold": fold,
            "best_epoch": best_epoch,
            **test_m,
            "n_train": int(len(tr_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
        }
        fold_rows.append(fold_row)

        with (fold_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(fold_row, f, indent=2)

    res = pd.DataFrame(fold_rows)
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
