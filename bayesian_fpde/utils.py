from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd


def load_yaml(path: str | Path) -> Dict[str, Any]:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required for config files. Install with `pip install pyyaml`.") from exc
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return data


def mode_config(config: Mapping[str, Any], mode: str) -> Dict[str, Any]:
    run_hash = config_hash({"mode": mode, "config": config})
    merged = dict(config)
    modes = merged.pop("modes", {}) or {}
    selected = modes.get(mode, {}) or {}
    if not isinstance(selected, dict):
        raise ValueError(f"mode config must be a mapping: {mode}")
    merged.update(selected)
    merged["mode"] = mode
    merged["run_config_hash"] = run_hash
    merged["job_config_hash"] = config_hash(merged)
    # Backward compatibility: config_hash means run-level config hash.
    merged["config_hash"] = run_hash
    return merged


def config_hash(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def ensure_dirs(*paths: str | Path) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def base_metadata(**extra: Any) -> Dict[str, Any]:
    out = {
        "method": "",
        "dataset_name": "",
        "task_id": "",
        "seed": "",
        "fold": "",
        "split_id": "",
        "mode": "",
        "config_hash": "",
        "run_config_hash": "",
        "job_config_hash": "",
        "timestamp": now_iso(),
        "git_commit": git_commit(),
        "status": "ok",
        "error_message": "",
    }
    out.update(extra)
    return out


def normalize_result_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "error" in out.columns and "error_message" not in out.columns:
        out = out.rename(columns={"error": "error_message"})
    required = [
        "method",
        "dataset_name",
        "task_id",
        "seed",
        "fold",
        "split_id",
        "mode",
        "config_hash",
        "run_config_hash",
        "job_config_hash",
        "timestamp",
        "git_commit",
        "status",
        "error_message",
    ]
    for col in required:
        if col not in out.columns:
            out[col] = ""
    front = [col for col in required if col in out.columns]
    rest = [col for col in out.columns if col not in front]
    return out[front + rest]


def setup_logging(log_dir: str | Path = "logs", name: str = "bayesian_fpde") -> logging.Logger:
    ensure_dirs(log_dir)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        stream = logging.StreamHandler()
        stream.setFormatter(fmt)
        logger.addHandler(stream)
        file_handler = logging.FileHandler(Path(log_dir) / f"{name}.log", encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    return logger


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ensure_dirs(path.parent)
    normalize_result_columns(df).to_csv(path, index=False, lineterminator="\n")


def write_parquet_or_csv(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    ensure_dirs(path.parent)
    try:
        normalize_result_columns(df).to_parquet(path, index=False)
        return path
    except Exception:
        fallback = path.with_suffix(path.suffix + ".csv")
        normalize_result_columns(df).to_csv(fallback, index=False, lineterminator="\n")
        return fallback


def write_json(data: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dirs(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def rng(seed: Optional[int] = None) -> np.random.Generator:
    return np.random.default_rng(seed)


def dense_float_array(x: Any) -> np.ndarray:
    if hasattr(x, "toarray"):
        x = x.toarray()
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D array, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("array contains NaN or infinity")
    return arr
