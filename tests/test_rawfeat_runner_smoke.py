from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.dynamic_fpde_audio.run_esc50_rawfeat_dynamic_fpde import main


def _make_tiny_esc50(root: Path) -> None:
    sf = pytest.importorskip("soundfile")
    (root / "audio").mkdir(parents=True)
    (root / "meta").mkdir(parents=True)
    rows = []
    sample_rate = 2000
    for category_index, category in enumerate(("alpha", "beta")):
        for fold in (1, 2, 3):
            filename = f"{fold}-{category}.wav"
            t = np.arange(400, dtype=float) / sample_rate
            frequency = 120.0 + 180.0 * category_index + fold
            waveform = 0.25 * np.sin(2.0 * np.pi * frequency * t)
            sf.write(root / "audio" / filename, waveform, sample_rate)
            rows.append({"filename": filename, "fold": fold, "category": category})
    with (root / "meta" / "esc50.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "fold", "category"])
        writer.writeheader()
        writer.writerows(rows)


def test_rawfeat_runner_smoke_with_generation_audit(tmp_path: Path):
    dataset = tmp_path / "ESC-50"
    output = tmp_path / "output"
    _make_tiny_esc50(dataset)

    exit_code = main(
        [
            "--dataset-root",
            str(dataset),
            "--output-dir",
            str(output),
            "--mode",
            "smoke",
            "--fold",
            "1",
            "--seed",
            "0",
            "--target-sr",
            "2000",
            "--frame-length",
            "64",
            "--hop-length",
            "32",
            "--lambda-hyb",
            "0.5",
            "--normalize",
            "l1",
            "--generation-scope",
            "selected",
            "--summary-scaling",
            "standard",
            "--noise-scale",
            "0.0",
        ]
    )

    assert exit_code == 0
    assert (output / "rawfeat_config.json").exists()
    sample_csv = output / "results" / "rawfeat_sample_metrics.csv"
    generation_csv = output / "results" / "rawfeat_generation_metrics.csv"
    sample_rows = list(csv.DictReader(sample_csv.open(encoding="utf-8")))
    generation_rows = list(csv.DictReader(generation_csv.open(encoding="utf-8")))
    assert len(sample_rows) == 2
    assert len(generation_rows) == 2
    assert all(row["shape_match"] == "True" for row in sample_rows)
    assert all(float(row["abs_exactness_residual"]) < 1e-9 for row in sample_rows)
    assert all(float(row["generated_abs_exactness_residual"]) < 1e-9 for row in generation_rows)
    for row in sample_rows:
        method_dir = output / "samples" / row["sample_id"] / "rawfeat_hyb_lambda_0.5"
        assert (method_dir / "generated_target.wav").exists()
        metrics = json.loads((method_dir / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["generation_audit"]["shape_match"] is True
        assert (output / "samples" / row["sample_id"] / "summary.csv").exists()
