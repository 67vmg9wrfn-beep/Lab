from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .io_mat import Record


@dataclass
class PreprocessConfig:
    fs: int = 125
    window_sec: float = 8.192
    overlap: float = 0.75
    min_duration_sec: int = 8 * 60
    abp_max: float = 200.0
    flatline_std_threshold: float = 1e-3


def detrend_linear(x: np.ndarray) -> np.ndarray:
    t = np.arange(len(x), dtype=np.float32)
    c = np.polyfit(t, x, 1)
    trend = c[0] * t + c[1]
    return (x - trend).astype(np.float32)


def is_flatline(x: np.ndarray, std_threshold: float = 1e-3) -> bool:
    return float(np.nanstd(x)) < std_threshold


def _segment_indices(length: int, window: int, stride: int) -> List[Tuple[int, int]]:
    out = []
    i = 0
    while i + window <= length:
        out.append((i, i + window))
        i += stride
    return out


def _extract_sbp_dbp_from_abp(abp_seg: np.ndarray) -> tuple[float, float] | None:
    peaks, _ = find_peaks(abp_seg, distance=20)
    troughs, _ = find_peaks(-abp_seg, distance=20)
    if len(peaks) < 2 or len(troughs) < 2:
        return None
    sbp = float(np.mean(abp_seg[peaks]))
    dbp = float(np.mean(abp_seg[troughs]))
    if not np.isfinite(sbp) or not np.isfinite(dbp):
        return None
    return sbp, dbp


def preprocess_records(records: List[Record], cfg: PreprocessConfig):
    win = int(cfg.window_sec * cfg.fs)
    stride = int(win * (1.0 - cfg.overlap))
    if stride <= 0:
        raise ValueError("overlap too high; stride <= 0")

    X: List[np.ndarray] = []
    y: List[np.ndarray] = []
    rows: List[Dict] = []

    stats = {
        "records_total": len(records),
        "records_kept": 0,
        "records_removed_short": 0,
        "records_removed_abp_high": 0,
        "records_removed_flatline": 0,
        "segments_total": 0,
        "segments_kept": 0,
        "segments_removed_bad_target": 0,
    }

    min_len = cfg.min_duration_sec * cfg.fs

    for rec in records:
        ppg = rec.ppg.astype(np.float32).reshape(-1)
        abp = rec.abp.astype(np.float32).reshape(-1)
        n = min(len(ppg), len(abp))
        if n < min_len:
            stats["records_removed_short"] += 1
            continue

        ppg = ppg[:n]
        abp = abp[:n]

        if float(np.nanmax(abp)) > cfg.abp_max:
            stats["records_removed_abp_high"] += 1
            continue

        if is_flatline(ppg, cfg.flatline_std_threshold) or is_flatline(abp, cfg.flatline_std_threshold):
            stats["records_removed_flatline"] += 1
            continue

        ppg = detrend_linear(ppg)
        stats["records_kept"] += 1

        for start, end in _segment_indices(n, win, stride):
            stats["segments_total"] += 1
            p_seg = ppg[start:end]
            a_seg = abp[start:end]
            target = _extract_sbp_dbp_from_abp(a_seg)
            if target is None:
                stats["segments_removed_bad_target"] += 1
                continue

            X.append(p_seg)
            y.append(np.asarray([target[0], target[1]], dtype=np.float32))
            rows.append(
                {
                    "source_file": rec.source_file,
                    "record_id": rec.record_index,
                    "start": start,
                    "end": end,
                    "len": len(p_seg),
                }
            )
            stats["segments_kept"] += 1

    if not X:
        raise RuntimeError("No valid segments after preprocessing. Check MAT parsing and thresholds.")

    X_np = np.stack(X).astype(np.float32)
    y_np = np.stack(y).astype(np.float32)
    meta = pd.DataFrame(rows)
    return X_np, y_np, meta, stats
