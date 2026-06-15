from __future__ import annotations

import csv
import math
import tomllib
from pathlib import Path

import numpy as np
import pytest

from experiments.dynamic_fpde_audio.datasets import read_esc50_metadata


ROOT = Path(__file__).resolve().parents[1]


def test_project_installs_fpde_from_dynamic_branch():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = pyproject["project"]["dependencies"]
    assert "fpde @ git+https://github.com/fpde-xai/fpde.git@dynamic" in deps
    assert "librosa" in pyproject["project"]["optional-dependencies"]["dynamic-audio"]
    assert "fpde @ git+https://github.com/fpde-xai/fpde.git@dynamic" in (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "fpde @ git+https://github.com/fpde-xai/fpde.git@dynamic" in (ROOT / "environment.yml").read_text(encoding="utf-8")


def test_missing_esc50_dataset_reports_expected_structure(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Expected structure"):
        read_esc50_metadata(tmp_path / "ESC-50")


def _write_wav(path: Path, *, frequency: float, sr: int = 22050) -> None:
    soundfile = pytest.importorskip("soundfile")
    t = np.linspace(0.0, 0.08, int(sr * 0.08), endpoint=False)
    y = 0.2 * np.sin(2.0 * math.pi * frequency * t)
    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(path, y, sr)


def _make_tiny_esc50(root: Path) -> None:
    pytest.importorskip("fpde")
    pytest.importorskip("librosa")
    pytest.importorskip("matplotlib")
    rows = []
    categories = ["class_a", "class_b"]
    for class_idx, category in enumerate(categories):
        for fold in (1, 2, 3, 4):
            filename = f"{fold}-{class_idx}-{category}.wav"
            _write_wav(root / "audio" / filename, frequency=220.0 + 110.0 * class_idx + 10.0 * fold)
            rows.append({"filename": filename, "fold": fold, "category": category})
    meta = root / "meta" / "esc50.csv"
    meta.parent.mkdir(parents=True, exist_ok=True)
    with meta.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "fold", "category"])
        writer.writeheader()
        writer.writerows(rows)


def test_smoke_cli_runs_on_synthetic_esc50_layout(tmp_path: Path):
    from experiments.dynamic_fpde_audio.run_esc50_dynamic_fpde import main

    dataset_root = tmp_path / "ESC-50"
    output_dir = tmp_path / "outputs"
    _make_tiny_esc50(dataset_root)

    exit_code = main(
        [
            "--dataset-root",
            str(dataset_root),
            "--output-dir",
            str(output_dir),
            "--mode",
            "smoke",
            "--fold",
            "1",
            "--seed",
            "7",
            "--prototype-length",
            "8",
            "--lambda-grid",
            "0.0,0.5,1.0",
            "--n-fft",
            "256",
            "--hop-length",
            "128",
            "--steps",
            "2",
            "--make-figures",
        ]
    )

    assert exit_code == 0
    sample_metrics = output_dir / "results" / "dynamic_fpde_sample_metrics.csv"
    summary = output_dir / "results" / "dynamic_fpde_summary_by_method.csv"
    lambdas = output_dir / "results" / "dynamic_fpde_lambda_selection.csv"
    additivity = output_dir / "results" / "dynamic_fpde_additivity_summary.csv"
    assert sample_metrics.exists()
    assert summary.exists()
    assert lambdas.exists()
    assert additivity.exists()
    assert (output_dir / "tables" / "table_dynamic_fpde_main_results.tex").exists()
    figures = output_dir / "figures"
    assert list(figures.glob("example_time_importance_*.png"))
    assert list(figures.glob("example_attribution_heatmap_*.png"))
    assert list(figures.glob("deletion_insertion_*.png"))
    assert (figures / "combined_score_by_method.png").exists()
    assert (figures / "lambda_selection.png").exists()

    with sample_metrics.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["method"] for row in rows} == {
        "dynamic_diff",
        "dynamic_cos",
        "dynamic_hyb",
        "energy_baseline",
        "random_baseline",
    }
    assert all(row["T"] and row["F"] for row in rows)
