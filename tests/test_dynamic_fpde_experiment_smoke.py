from __future__ import annotations

import csv
import math
import subprocess
import sys
import types
import tomllib
from pathlib import Path

import numpy as np
import pytest

from experiments.dynamic_fpde_audio.datasets import read_esc50_metadata


ROOT = Path(__file__).resolve().parents[1]


def test_project_installs_fpde_from_dynamic_branch():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["name"] == "dynamic-fpde-reproducibility-artifact"
    assert (
        pyproject["project"]["description"]
        == "Reproducible Dynamic-FPDE experiments for time-resolved prototype-directional audio explanations."
    )
    deps = pyproject["project"]["dependencies"]
    assert "fpde @ git+https://github.com/fpde-xai/fpde.git@dynamic" in deps
    assert "librosa" in pyproject["project"]["optional-dependencies"]["dynamic-audio"]
    assert "matplotlib" in pyproject["project"]["optional-dependencies"]["plot"]
    assert "openml" not in pyproject["project"]["optional-dependencies"]
    assert "baselines" not in pyproject["project"]["optional-dependencies"]
    assert pyproject["tool"]["setuptools"]["packages"]["find"]["include"] == ["experiments*"]
    assert "fpde @ git+https://github.com/fpde-xai/fpde.git@dynamic" in (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "fpde @ git+https://github.com/fpde-xai/fpde.git@dynamic" in (ROOT / "environment.yml").read_text(encoding="utf-8")


def test_cli_accepts_comma_separated_folds():
    from experiments.dynamic_fpde_audio.run_esc50_dynamic_fpde import build_parser

    args = build_parser().parse_args(["--dataset-root", "ESC-50", "--folds", "1,2,3,4,5"])

    assert args.folds == "1,2,3,4,5"
    assert args.backend == "cpu"


def test_cli_accepts_explicit_cuda_backend():
    from experiments.dynamic_fpde_audio.run_esc50_dynamic_fpde import build_parser

    args = build_parser().parse_args(["--dataset-root", "ESC-50", "--backend", "cuda"])

    assert args.backend == "cuda"


def test_cuda_backend_error_is_not_silently_downgraded(monkeypatch: pytest.MonkeyPatch):
    from experiments.dynamic_fpde_audio import run_esc50_dynamic_fpde as runner

    def fail_cuda() -> None:
        raise RuntimeError("CUDA backend requested, but CuPy is not installed")

    monkeypatch.setattr(runner, "_ensure_cuda_backend", fail_cuda)

    with pytest.raises(RuntimeError, match="CUDA backend requested"):
        runner._ensure_backend("cuda")


def test_cuda_backend_uses_local_cupy_path_without_fpde_dynamic_cuda(monkeypatch: pytest.MonkeyPatch):
    from fpde import prepare_dynamic_fpde_context

    from experiments.dynamic_fpde_audio.datasets import ESCSample
    from experiments.dynamic_fpde_audio.run_esc50_dynamic_fpde import (
        _explain_batch_cpu,
        _explain_batch_cuda,
        _resolve_dynamic_input,
    )

    class _FakeRuntime:
        @staticmethod
        def getDeviceCount() -> int:
            return 1

    fake_cupy = types.SimpleNamespace(
        abs=np.abs,
        asarray=np.asarray,
        asnumpy=np.asarray,
        cuda=types.SimpleNamespace(runtime=_FakeRuntime()),
        float64=np.float64,
        sqrt=np.sqrt,
        sum=np.sum,
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)

    X_train = [
        np.array([[0.0, 0.2], [0.5, 0.4], [1.0, 0.7]], dtype=float),
        np.array([[0.1, 0.1], [0.4, 0.5], [0.9, 0.8]], dtype=float),
        np.array([[1.0, 0.3], [0.4, 0.2], [0.0, 0.1]], dtype=float),
        np.array([[0.9, 0.4], [0.3, 0.3], [0.1, 0.0]], dtype=float),
    ]
    context = prepare_dynamic_fpde_context(X_train, ["class_a", "class_a", "class_b", "class_b"], prototype_length=3)
    samples = [
        ESCSample("sample_a", Path("sample_a.wav"), "sample_a.wav", 1, "class_a"),
        ESCSample("sample_b", Path("sample_b.wav"), "sample_b.wav", 1, "class_a"),
    ]
    resolved = [
        _resolve_dynamic_input(
            X,
            context,
            sample=sample,
            target_label="class_a",
            rival_label="class_b",
        )
        for sample, X in zip(
            samples,
            [
                np.array([[0.2, 0.2], [0.6, 0.4], [0.8, 0.6]], dtype=float),
                np.array([[0.0, 0.3], [0.5, 0.3], [1.1, 0.7]], dtype=float),
            ],
            strict=True,
        )
    ]

    for mode in ("dynamic_diff", "dynamic_cos", "dynamic_hyb"):
        cpu = _explain_batch_cpu(resolved, mode=mode, lambda_hyb=0.35)
        cuda = _explain_batch_cuda(resolved, mode=mode, lambda_hyb=0.35)

        for sample in samples:
            cpu_explanation = cpu[sample.sample_id][0]
            cuda_explanation = cuda[sample.sample_id][0]
            np.testing.assert_allclose(cuda_explanation.attributions, cpu_explanation.attributions, rtol=1e-8, atol=1e-10)
            assert cuda_explanation.evidence == pytest.approx(cpu_explanation.evidence, rel=1e-8, abs=1e-10)
            assert cuda_explanation.rival_label == cpu_explanation.rival_label


def test_gitignore_excludes_dynamic_audio_feature_cache_paths():
    result = subprocess.run(
        ["git", "check-ignore", "outputs/demo/cache/features/example.npz", "outputs/demo/cache/features/"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    ignored = set(result.stdout.splitlines())
    assert "outputs/demo/cache/features/example.npz" in ignored
    assert "outputs/demo/cache/features/" in ignored


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
    positive_summary = output_dir / "results" / "dynamic_fpde_summary_positive_margin_by_method.csv"
    lambdas = output_dir / "results" / "dynamic_fpde_lambda_selection.csv"
    additivity = output_dir / "results" / "dynamic_fpde_additivity_summary.csv"
    assert sample_metrics.exists()
    assert summary.exists()
    assert positive_summary.exists()
    assert lambdas.exists()
    assert additivity.exists()
    assert (output_dir / "tables" / "table_dynamic_fpde_main_results.tex").exists()
    assert (output_dir / "tables" / "table_dynamic_fpde_positive_margin_results.tex").exists()
    assert (output_dir / "tables" / "table_dynamic_fpde_margin_summary.tex").exists()
    figures = output_dir / "figures"
    assert list(figures.glob("example_time_importance_*.png"))
    assert list(figures.glob("example_attribution_heatmap_*.png"))
    assert list(figures.glob("deletion_insertion_*.png"))
    assert (figures / "combined_score_by_method.png").exists()
    assert (figures / "lambda_selection.png").exists()

    with sample_metrics.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {
        "prototype_margin",
        "prototype_margin_positive",
        "prototype_margin_sign",
        "selection_margin",
        "selection_margin_positive",
        "selection_margin_sign",
        "selection_margin_source",
        "common_rival_label",
        "evaluation_evidence",
        "evaluation_margin",
        "evidence_role",
        "aggregation_unit",
    }.issubset(rows[0])
    assert {row["prototype_margin_sign"] for row in rows}.issubset({"positive", "zero", "negative"})
    assert {row["selection_margin_sign"] for row in rows}.issubset({"positive", "zero", "negative"})
    assert {row["selection_margin_source"] for row in rows} == {"dynamic_diff"}
    assert {row["aggregation_unit"] for row in rows} == {"sample", "sample_repetition"}
    assert all(row["prototype_margin"] == row["evidence"] for row in rows)
    baseline_rows = [row for row in rows if row["method"].endswith("_baseline")]
    assert baseline_rows
    assert {row["evidence_role"] for row in baseline_rows} == {"evaluation_margin"}
    assert all(row["evaluation_evidence"] == row["evidence"] for row in baseline_rows)
    assert all(row["evaluation_margin"] == row["prototype_margin"] for row in baseline_rows)
    assert {row["method"] for row in rows} == {
        "dynamic_diff",
        "dynamic_cos",
        "dynamic_hyb",
        "energy_baseline",
        "random_baseline",
    }
    assert all(row["T"] and row["F"] for row in rows)
    for sample_id in {row["sample_id"] for row in rows}:
        sample_group = [row for row in rows if row["sample_id"] == sample_id]
        assert len({row["target_label"] for row in sample_group}) == 1
        assert len({row["rival_label"] for row in sample_group}) == 1
        assert {row["rival_label"] for row in sample_group} == {row["common_rival_label"] for row in sample_group}
        assert len({row["selection_margin"] for row in sample_group}) == 1

    with positive_summary.open("r", encoding="utf-8", newline="") as handle:
        positive_rows = list(csv.DictReader(handle))
    positive_counts = {row["method"]: int(row["n"]) for row in positive_rows}
    assert len(set(positive_counts.values())) == 1

    with summary.open("r", encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert {"n_unique_samples", "n_rows", "random_repetitions_mean"}.issubset(summary_rows[0])
    random_summary = next(row for row in summary_rows if row["method"] == "random_baseline")
    assert int(random_summary["n"]) == int(random_summary["n_unique_samples"])
    assert int(random_summary["n_rows"]) == int(random_summary["n_unique_samples"])
    assert float(random_summary["random_repetitions_mean"]) == pytest.approx(1.0)


def _cupy_or_skip():
    cp = pytest.importorskip("cupy")
    try:
        if int(cp.cuda.runtime.getDeviceCount()) < 1:
            pytest.skip("CUDA device is unavailable")
    except Exception as exc:
        pytest.skip(f"CUDA runtime is unavailable: {exc}")
    return cp


def test_dynamic_fpde_cuda_backend_matches_cpu_on_synthetic_batch(tmp_path: Path):
    _cupy_or_skip()
    from fpde import prepare_dynamic_fpde_context

    from experiments.dynamic_fpde_audio.datasets import ESCSample
    from experiments.dynamic_fpde_audio.run_esc50_dynamic_fpde import (
        _explain_batch_cpu,
        _explain_batch_cuda,
        _resolve_dynamic_input,
    )

    X_train = [
        np.array([[0.0, 0.2], [0.5, 0.4], [1.0, 0.7]], dtype=float),
        np.array([[0.1, 0.1], [0.4, 0.5], [0.9, 0.8]], dtype=float),
        np.array([[1.0, 0.3], [0.4, 0.2], [0.0, 0.1]], dtype=float),
        np.array([[0.9, 0.4], [0.3, 0.3], [0.1, 0.0]], dtype=float),
    ]
    y_train = ["class_a", "class_a", "class_b", "class_b"]
    context = prepare_dynamic_fpde_context(X_train, y_train, prototype_length=3)
    samples = [
        ESCSample("sample_a", tmp_path / "sample_a.wav", "sample_a.wav", 1, "class_a"),
        ESCSample("sample_b", tmp_path / "sample_b.wav", "sample_b.wav", 1, "class_a"),
    ]
    X_test = [
        np.array([[0.2, 0.2], [0.6, 0.4], [0.8, 0.6]], dtype=float),
        np.array([[0.0, 0.3], [0.5, 0.3], [1.1, 0.7]], dtype=float),
    ]
    resolved = [
        _resolve_dynamic_input(X, context, sample=sample, target_label="class_a", rival_label="class_b")
        for sample, X in zip(samples, X_test, strict=True)
    ]

    for mode in ("dynamic_diff", "dynamic_cos", "dynamic_hyb"):
        cpu = _explain_batch_cpu(resolved, mode=mode, lambda_hyb=0.35)
        cuda = _explain_batch_cuda(resolved, mode=mode, lambda_hyb=0.35)

        for sample in samples:
            cpu_explanation = cpu[sample.sample_id][0]
            cuda_explanation = cuda[sample.sample_id][0]
            np.testing.assert_allclose(cuda_explanation.attributions, cpu_explanation.attributions, rtol=1e-8, atol=1e-10)
            np.testing.assert_allclose(cuda_explanation.time_importance, cpu_explanation.time_importance, rtol=1e-8, atol=1e-10)
            np.testing.assert_allclose(cuda_explanation.feature_importance, cpu_explanation.feature_importance, rtol=1e-8, atol=1e-10)
            assert cuda_explanation.evidence == pytest.approx(cpu_explanation.evidence, rel=1e-8, abs=1e-10)
            assert cuda_explanation.rival_label == cpu_explanation.rival_label
