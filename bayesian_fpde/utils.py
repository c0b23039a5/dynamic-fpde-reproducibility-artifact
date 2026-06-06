from __future__ import annotations

import hashlib
import json
import logging
import os
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


HASH_METADATA_KEYS = {
    "config_hash",
    "experiment_config_hash",
    "job_config_hash",
    "run_config_hash",
    "runner_invocation_hash",
    "workflow_run_id",
    "workflow_run_attempt",
    "workflow_name",
    "workflow_ref",
    "workflow_sha",
}

METADATA_STRING_COLUMNS = [
    "method",
    "dataset_name",
    "task_id",
    "seed",
    "fold",
    "split_id",
    "mode",
    "config_hash",
    "experiment_config_hash",
    "workflow_run_id",
    "workflow_run_attempt",
    "workflow_name",
    "workflow_ref",
    "workflow_sha",
    "runner_invocation_hash",
    "run_config_hash",
    "job_config_hash",
    "timestamp",
    "git_commit",
    "status",
    "error_message",
    "metric_direction",
]


def _hash_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in config.items() if k not in HASH_METADATA_KEYS}


def mode_config(config: Mapping[str, Any], mode: str, *, runner_name: str = "") -> Dict[str, Any]:
    hash_config = _hash_config(config)
    explicit_experiment_hash = str(
        config.get("experiment_config_hash")
        or os.environ.get("BAYESIAN_FPDE_EXPERIMENT_CONFIG_HASH", "")
        # Backward compatibility for configs produced before experiment_config_hash existed.
        or config.get("run_config_hash")
        or os.environ.get("BAYESIAN_FPDE_RUN_CONFIG_HASH", "")
    ).strip()
    experiment_hash = explicit_experiment_hash or config_hash({"mode": mode, "runner_name": runner_name, "config": hash_config})
    workflow_run_id = str(config.get("workflow_run_id") or os.environ.get("BAYESIAN_FPDE_WORKFLOW_RUN_ID", "") or os.environ.get("GITHUB_RUN_ID", "")).strip()
    workflow_run_attempt = str(config.get("workflow_run_attempt") or os.environ.get("BAYESIAN_FPDE_WORKFLOW_RUN_ATTEMPT", "") or os.environ.get("GITHUB_RUN_ATTEMPT", "")).strip()
    workflow_name = str(config.get("workflow_name") or os.environ.get("BAYESIAN_FPDE_WORKFLOW_NAME", "") or os.environ.get("GITHUB_WORKFLOW", "")).strip()
    workflow_ref = str(config.get("workflow_ref") or os.environ.get("BAYESIAN_FPDE_WORKFLOW_REF", "") or os.environ.get("GITHUB_REF", "")).strip()
    workflow_sha = str(config.get("workflow_sha") or os.environ.get("BAYESIAN_FPDE_WORKFLOW_SHA", "") or os.environ.get("GITHUB_SHA", "")).strip()
    runner_hash = str(config.get("runner_invocation_hash") or os.environ.get("BAYESIAN_FPDE_RUNNER_INVOCATION_HASH", "")).strip()
    if not runner_hash:
        runner_hash = config_hash(
            {
                "mode": mode,
                "runner_name": runner_name,
                "experiment_config_hash": experiment_hash,
                "workflow_run_id": workflow_run_id,
                "config": hash_config,
            }
        )
    merged = dict(config)
    modes = merged.pop("modes", {}) or {}
    selected = modes.get(mode, {}) or {}
    if not isinstance(selected, dict):
        raise ValueError(f"mode config must be a mapping: {mode}")
    merged.update(selected)
    merged["mode"] = mode
    merged["experiment_config_hash"] = experiment_hash
    merged["workflow_run_id"] = workflow_run_id
    merged["workflow_run_attempt"] = workflow_run_attempt
    merged["workflow_name"] = workflow_name
    merged["workflow_ref"] = workflow_ref
    merged["workflow_sha"] = workflow_sha
    merged["runner_invocation_hash"] = runner_hash
    # Backward compatibility: run_config_hash now identifies the runner
    # invocation, not the paper-level experiment.
    merged["run_config_hash"] = runner_hash
    # This is only a mode-level placeholder. Experiment runners should replace
    # it with a dataset/seed/fold-specific job_config_hash for result rows.
    merged["job_config_hash"] = config_hash({k: v for k, v in merged.items() if k != "job_config_hash"})
    # Backward compatibility: config_hash means paper-level experiment hash.
    merged["config_hash"] = experiment_hash
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
            ["git", "rev-parse", "HEAD"],
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
        "experiment_config_hash": "",
        "workflow_run_id": "",
        "workflow_run_attempt": "",
        "workflow_name": "",
        "workflow_ref": "",
        "workflow_sha": "",
        "runner_invocation_hash": "",
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
        "experiment_config_hash",
        "workflow_run_id",
        "workflow_run_attempt",
        "workflow_name",
        "workflow_ref",
        "workflow_sha",
        "runner_invocation_hash",
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
    for col in METADATA_STRING_COLUMNS:
        if col in out.columns:
            out[col] = out[col].astype("string")
    front = [col for col in required if col in out.columns]
    rest = [col for col in out.columns if col not in front]
    return out[front + rest]


def read_csv_preserve_metadata(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    path = Path(path)
    header = pd.read_csv(path, nrows=0)
    dtype = dict(kwargs.pop("dtype", {}) or {})
    for col in METADATA_STRING_COLUMNS:
        if col in header.columns:
            dtype[col] = "string"
    return pd.read_csv(path, dtype=dtype, **kwargs)


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
