from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_synthetic_smoke_cli(tmp_path: Path):
    cmd = [
        sys.executable,
        "-m",
        "experiments.run_synthetic_calibration",
        "--config",
        "configs/synthetic.yaml",
        "--mode",
        "smoke",
    ]
    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    assert result.returncode == 0, result.stderr
    assert (Path(__file__).resolve().parents[1] / "results" / "synthetic_calibration_summary.csv").exists()


def test_openml_local_smoke_cli():
    cmd = [
        sys.executable,
        "-m",
        "experiments.run_openml_benchmark",
        "--config",
        "configs/openml_cc18.yaml",
        "--mode",
        "smoke",
    ]
    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    assert result.returncode == 0, result.stderr
    assert (Path(__file__).resolve().parents[1] / "results" / "openml_metrics.csv").exists()
