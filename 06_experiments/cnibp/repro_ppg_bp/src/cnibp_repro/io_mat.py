from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import h5py
import numpy as np
from scipy.io import loadmat


@dataclass
class Record:
    source_file: str
    record_index: int
    ppg: np.ndarray
    abp: np.ndarray


def _is_numeric_array(x: Any) -> bool:
    return isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.number)


def _choose_channels(arr2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Heuristic for unknown column ordering in Kachuee MAT variants.
    if arr2d.ndim != 2:
        raise ValueError("Expected 2D array")

    # Some MAT variants store channels on axis 0 (C x T).
    if arr2d.shape[0] <= 4 and arr2d.shape[1] > arr2d.shape[0]:
        arr2d = arr2d.T

    if arr2d.shape[1] < 2:
        raise ValueError("Expected 2D array with >=2 channels")

    q95 = np.nanpercentile(arr2d, 95, axis=0)
    q05 = np.nanpercentile(arr2d, 5, axis=0)
    span = q95 - q05

    # ABP usually has large mmHg dynamic range; pick largest span as ABP.
    abp_idx = int(np.argmax(span))
    candidate = [i for i in range(arr2d.shape[1]) if i != abp_idx]
    if not candidate:
        raise ValueError("Could not infer PPG channel")
    # PPG tends to have smaller range than ABP; pick smallest remaining span.
    ppg_idx = int(candidate[np.argmin(span[candidate])])

    ppg = arr2d[:, ppg_idx].astype(np.float32)
    abp = arr2d[:, abp_idx].astype(np.float32)
    return ppg, abp


def _records_from_numeric_array(obj: np.ndarray, source_file: str, start_index: int) -> List[Record]:
    out: List[Record] = []
    idx = start_index

    if obj.ndim == 2 and obj.shape[1] >= 2 and obj.shape[0] > 128:
        ppg, abp = _choose_channels(obj)
        out.append(Record(source_file=source_file, record_index=idx, ppg=ppg, abp=abp))
        return out

    if obj.ndim == 3 and obj.shape[-1] >= 2:
        for i in range(obj.shape[0]):
            arr2d = np.asarray(obj[i])
            if arr2d.ndim != 2:
                continue
            ppg, abp = _choose_channels(arr2d)
            out.append(Record(source_file=source_file, record_index=idx, ppg=ppg, abp=abp))
            idx += 1
    return out


def _walk_obj(obj: Any, source_file: str, start_index: int = 0) -> List[Record]:
    out: List[Record] = []
    idx = start_index

    if _is_numeric_array(obj):
        recs = _records_from_numeric_array(np.asarray(obj), source_file, idx)
        out.extend(recs)
        idx += len(recs)
        return out

    if isinstance(obj, np.ndarray) and obj.dtype == object:
        for item in obj.flat:
            recs = _walk_obj(item, source_file, idx)
            out.extend(recs)
            idx += len(recs)
        return out

    if hasattr(obj, "__dict__"):
        d = vars(obj)
        lower = {k.lower(): k for k in d.keys()}
        if "ppg" in lower and "abp" in lower:
            ppg = np.asarray(d[lower["ppg"]]).astype(np.float32).reshape(-1)
            abp = np.asarray(d[lower["abp"]]).astype(np.float32).reshape(-1)
            out.append(Record(source_file=source_file, record_index=idx, ppg=ppg, abp=abp))
            return out
        for v in d.values():
            recs = _walk_obj(v, source_file, idx)
            out.extend(recs)
            idx += len(recs)
        return out

    if isinstance(obj, dict):
        lower = {k.lower(): k for k in obj.keys()}
        if "ppg" in lower and "abp" in lower:
            ppg = np.asarray(obj[lower["ppg"]]).astype(np.float32).reshape(-1)
            abp = np.asarray(obj[lower["abp"]]).astype(np.float32).reshape(-1)
            out.append(Record(source_file=source_file, record_index=idx, ppg=ppg, abp=abp))
            return out
        for v in obj.values():
            recs = _walk_obj(v, source_file, idx)
            out.extend(recs)
            idx += len(recs)
        return out

    return out


def _h5_ref_to_py(x: Any, f: h5py.File, visited: set[str]) -> Any:
    if isinstance(x, h5py.Reference):
        if not x:
            return None
        return _h5_obj_to_py(f[x], f, visited)
    if isinstance(x, np.ndarray):
        return _h5_data_to_py(x, f, visited)
    return x


def _h5_data_to_py(data: Any, f: h5py.File, visited: set[str]) -> Any:
    if isinstance(data, np.ndarray):
        if h5py.check_dtype(ref=data.dtype) is not None or data.dtype == object:
            flat = [_h5_ref_to_py(x, f, visited) for x in data.flat]
            arr = np.empty(len(flat), dtype=object)
            arr[:] = flat
            return arr.reshape(data.shape)
        return np.asarray(data)
    if isinstance(data, h5py.Reference):
        return _h5_ref_to_py(data, f, visited)
    return data


def _h5_obj_to_py(obj: Any, f: h5py.File, visited: set[str]) -> Any:
    name = getattr(obj, "name", None)
    if isinstance(name, str) and name in visited:
        return None
    if isinstance(name, str):
        visited.add(name)

    if isinstance(obj, h5py.Dataset):
        return _h5_data_to_py(obj[()], f, visited)

    if isinstance(obj, h5py.Group):
        return {k: _h5_obj_to_py(obj[k], f, visited) for k in obj.keys()}

    return obj


def _load_mat_file(path: Path) -> Dict[str, Any]:
    try:
        data = loadmat(path, squeeze_me=True, struct_as_record=False)
        return {k: v for k, v in data.items() if not k.startswith("__")}
    except NotImplementedError:
        out: Dict[str, Any] = {}
        with h5py.File(path, "r") as f:
            for k in f.keys():
                if k.startswith("#"):
                    continue
                out[k] = _h5_obj_to_py(f[k], f, set())
        return out


def load_records(mat_files: Iterable[Path]) -> List[Record]:
    records: List[Record] = []
    rec_idx = 0

    for path in mat_files:
        content = _load_mat_file(Path(path))
        found_before = len(records)
        for v in content.values():
            recs = _walk_obj(v, source_file=path.name, start_index=rec_idx)
            records.extend(recs)
            rec_idx += len(recs)

        if len(records) == found_before:
            keys = ", ".join(content.keys())
            raise RuntimeError(
                f"Could not parse records from {path}. Available MAT keys: {keys}"
            )

    return records
