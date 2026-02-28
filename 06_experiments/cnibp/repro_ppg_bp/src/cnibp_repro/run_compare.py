from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One-click compare: no-zscore vs segment-zscore")
    p.add_argument("--drive_root", type=str, required=True)
    p.add_argument("--parts", type=str, required=True, help="Comma-separated MAT filenames or absolute paths.")
    p.add_argument("--output_root", type=str, required=True)
    p.add_argument("--config_a", type=str, required=True)
    p.add_argument("--config_b", type=str, required=True)
    p.add_argument("--label_a", type=str, default="baseline")
    p.add_argument("--label_b", type=str, default="enhanced")
    return p.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _cfg_signature(cfg: dict[str, Any]) -> str:
    # Full config signature avoids mixing smoke/full runs during reuse/resume.
    return json.dumps(cfg, sort_keys=True, ensure_ascii=False)


def _valid_preprocess(run_dir: Path) -> bool:
    pre = run_dir / "preprocess"
    return (
        (pre / "preprocessed_manifest.json").exists()
        and (pre / "segments_meta.csv").exists()
        and (pre / "X.float16.mmap").exists()
        and (pre / "y.float32.mmap").exists()
    )


def _mode_of_run(run_dir: Path) -> str | None:
    rc = run_dir / "resolved_config.json"
    if not rc.exists():
        return None
    try:
        cfg = _load_json(rc)
    except Exception:
        return None
    return str(cfg.get("ppg_norm_mode", "none"))


def _latest_run_for_signature(
    output_root: Path, *, target_sig: str, require_pre: bool, require_train: bool
) -> Path | None:
    runs = sorted(output_root.glob("run_*"))
    for run in reversed(runs):
        rc = run / "resolved_config.json"
        if not rc.exists():
            continue
        try:
            cfg = _load_json(rc)
        except Exception:
            continue
        if _cfg_signature(cfg) != target_sig:
            continue
        if require_pre and not _valid_preprocess(run):
            continue
        if require_train and not (run / "train").exists():
            continue
        return run
    return None


def _run_single(
    *,
    run_label: str,
    config_path: Path,
    drive_root: str,
    parts: str,
    output_root: Path,
    compare_dir: Path,
) -> Path:
    target_cfg = _load_json(config_path)
    target_sig = _cfg_signature(target_cfg)
    reuse_from = _latest_run_for_signature(output_root, target_sig=target_sig, require_pre=True, require_train=False)
    resume_from = _latest_run_for_signature(output_root, target_sig=target_sig, require_pre=False, require_train=True)

    cmd = [
        "python",
        "-m",
        "cnibp_repro.run_repro",
        "--drive_root",
        drive_root,
        "--parts",
        parts,
        "--config",
        str(config_path),
        "--output_root",
        str(output_root),
    ]
    if reuse_from is not None:
        cmd += ["--reuse_from_run_dir", str(reuse_from)]
    if resume_from is not None:
        cmd += ["--resume_from_run_dir", str(resume_from)]

    log_file = compare_dir / f"{run_label}.log"
    print(f"[COMPARE] label={run_label}")
    print(f"[COMPARE] config={config_path}")
    print(f"[COMPARE] reuse_from={reuse_from if reuse_from else '(none)'}")
    print(f"[COMPARE] resume_from={resume_from if resume_from else '(none)'}")

    with log_file.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"label={run_label} failed, see log: {log_file}")

    if resume_from is not None:
        run_dir = resume_from
    else:
        run_dir = _latest_run_for_signature(output_root, target_sig=target_sig, require_pre=False, require_train=True)
        if run_dir is None:
            raise RuntimeError(f"label={run_label} completed but run dir not found under {output_root}")
    return run_dir


def _load_summary(run_dir: Path) -> dict[str, Any]:
    p = run_dir / "train" / "summary.json"
    if not p.exists():
        raise FileNotFoundError(f"summary not found: {p}")
    return _load_json(p)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    compare_dir = output_root / "compare_runs" / f"compare_{ts}"
    compare_dir.mkdir(parents=True, exist_ok=True)

    run_a = _run_single(
        run_label=args.label_a,
        config_path=Path(args.config_a),
        drive_root=args.drive_root,
        parts=args.parts,
        output_root=output_root,
        compare_dir=compare_dir,
    )
    run_b = _run_single(
        run_label=args.label_b,
        config_path=Path(args.config_b),
        drive_root=args.drive_root,
        parts=args.parts,
        output_root=output_root,
        compare_dir=compare_dir,
    )

    s_a = _load_summary(run_a)
    s_b = _load_summary(run_b)

    rows = [
        {"mode": args.label_a, "run_dir": str(run_a), **s_a},
        {"mode": args.label_b, "run_dir": str(run_b), **s_b},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(compare_dir / "compare_summary.csv", index=False)

    delta = {
        "mae_sbp_mean_delta_b_minus_a": float(s_b["mae_sbp_mean"] - s_a["mae_sbp_mean"]),
        "mae_dbp_mean_delta_b_minus_a": float(s_b["mae_dbp_mean"] - s_a["mae_dbp_mean"]),
        "mse_sbp_mean_delta_b_minus_a": float(s_b["mse_sbp_mean"] - s_a["mse_sbp_mean"]),
        "mse_dbp_mean_delta_b_minus_a": float(s_b["mse_dbp_mean"] - s_a["mse_dbp_mean"]),
        "better_by_mae_sum": (
            args.label_b
            if (s_b["mae_sbp_mean"] + s_b["mae_dbp_mean"]) < (s_a["mae_sbp_mean"] + s_a["mae_dbp_mean"])
            else args.label_a
        ),
    }

    report = {
        "compare_dir": str(compare_dir),
        "run_a": str(run_a),
        "run_b": str(run_b),
        "label_a": args.label_a,
        "label_b": args.label_b,
        "summary_a": s_a,
        "summary_b": s_b,
        "delta_b_minus_a": delta,
    }
    with (compare_dir / "compare_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n[COMPARE DONE]")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
