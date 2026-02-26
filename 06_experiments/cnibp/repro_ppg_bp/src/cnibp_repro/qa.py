from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_preprocess_report(stats: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "preprocess_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def save_meta(meta: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta.to_csv(out_dir / "segments_meta.csv", index=False)


def save_sample_waveforms(X: np.ndarray, y: np.ndarray, out_dir: Path, fs: int = 125, n: int = 12, seed: int = 42) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)

    t = np.arange(X.shape[1]) / fs
    fig, axes = plt.subplots(len(idx), 1, figsize=(10, 2.2 * len(idx)), sharex=True)
    if len(idx) == 1:
        axes = [axes]

    for ax, i in zip(axes, idx):
        ax.plot(t, X[i], linewidth=1.0)
        ax.set_ylabel("PPG")
        ax.grid(True, alpha=0.25)
        ax.set_title(f"sample={i} SBP={y[i,0]:.1f} DBP={y[i,1]:.1f}", fontsize=9)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    fig.savefig(out_dir / "sample_waveforms.png", dpi=160)
    plt.close(fig)


def save_target_hist(y: np.ndarray, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].hist(y[:, 0], bins=50)
    axes[0].set_title("SBP distribution")
    axes[1].hist(y[:, 1], bins=50)
    axes[1].set_title("DBP distribution")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_dir / "target_distribution.png", dpi=160)
    plt.close(fig)
