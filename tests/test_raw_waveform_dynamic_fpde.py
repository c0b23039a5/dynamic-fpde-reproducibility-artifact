from __future__ import annotations

import csv
import inspect
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def _write_wav(path: Path, data: np.ndarray, *, sr: int = 1000) -> None:
    soundfile = pytest.importorskip("soundfile")
    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(path, data.astype(np.float32, copy=False), sr)


def _make_raw_esc50(root: Path) -> None:
    rows = []
    for category, sign in (("class_a", 1.0), ("class_b", -1.0)):
        for fold in (1, 2, 3, 4):
            filename = f"{fold}-{category}.wav"
            if fold == 4 and category == "class_a":
                waveform = np.array([[sign, sign * 0.5]], dtype=float)
            elif fold == 1 and category == "class_a":
                waveform = np.concatenate([np.ones(30), -np.ones(30)])
            else:
                t = np.linspace(0.0, 0.08, 80, endpoint=False)
                waveform = sign * (0.8 + 0.1 * np.sin(2.0 * math.pi * 40.0 * t))
            _write_wav(root / "audio" / filename, np.asarray(waveform, dtype=float), sr=1000)
            rows.append({"filename": filename, "fold": fold, "category": category})

    meta = root / "meta" / "esc50.csv"
    meta.parent.mkdir(parents=True, exist_ok=True)
    with meta.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "fold", "category"])
        writer.writeheader()
        writer.writerows(rows)


def _cuda_available() -> bool:
    try:
        import cupy as cp  # type: ignore[import-not-found]

        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


def test_raw_fpde_api_imports_from_refreshed_dynamic_package():
    import fpde

    assert "site-packages" in str(Path(fpde.__file__))
    assert "fpde-dynamic-check" not in str(Path(fpde.__file__))
    assert hasattr(fpde, "prepare_raw_waveform_fpde_context")
    assert hasattr(fpde, "raw_waveform_fpde_explain_one")
    assert hasattr(fpde, "save_raw_waveform_fpde_results")


def test_raw_cli_defaults_and_lambda_grid():
    from experiments.dynamic_fpde_audio.run_esc50_raw_waveform_fpde import build_parser, parse_lambda_grid

    args = build_parser().parse_args(["--dataset-root", "ESC-50"])

    assert args.target_sr == 16000
    assert args.segment_sec == pytest.approx(0.5)
    assert args.hop_sec == pytest.approx(0.1)
    assert args.device == "cuda"
    assert args.context_device == "cpu"
    assert args.prototype_selection == "exact_medoid"
    assert args.medoid_block_size == 128
    assert args.max_prototype_candidates == 0
    assert args.retain_segment_banks is False
    assert args.save_plots is True
    assert not hasattr(args, "normalize")
    assert parse_lambda_grid(args.lambda_grid) == tuple(i / 10.0 for i in range(11))


def test_raw_runner_does_not_reference_feature_extraction_or_standardization():
    import experiments.dynamic_fpde_audio.run_esc50_raw_waveform_fpde as runner

    source = inspect.getsource(runner).lower()

    assert "extract_frame_features" not in source
    assert "fit_standardizer" not in source
    assert "transform_features" not in source
    assert "librosa" not in source
    assert "torchaudio" not in source
    assert "experiments.dynamic_fpde_audio.features" not in source


def test_raw_waveform_context_handles_stereo_short_padding_and_masking():
    from fpde import prepare_raw_waveform_fpde_context, raw_waveform_fpde_explain_one

    context = prepare_raw_waveform_fpde_context(
        [
            np.array([[2.0, 4.0], [6.0, 8.0]], dtype=float),
            np.array([-2.0, -4.0], dtype=float),
        ],
        ["class_a", "class_b"],
        sample_rates=[1000, 1000],
        target_sr=1000,
        segment_sec=0.004,
        hop_sec=0.002,
    )

    np.testing.assert_allclose(context.segment_banks["class_a"][0], [3.0, 7.0, 0.0, 0.0])
    np.testing.assert_array_equal(context.segment_masks["class_a"][0], [True, True, False, False])
    explanation = raw_waveform_fpde_explain_one(
        np.array([3.0, 7.0], dtype=float),
        context,
        sample_rate=1000,
        target_label="class_a",
        rival_label="class_b",
        lambda_grid=[0.5],
    )

    result = explanation.lambda_results[0.5]
    assert result["phi"].shape == explanation.waveform.shape
    assert np.all(np.isfinite(result["phi"]))
    np.testing.assert_array_equal(result["effective_window_masks"][0], [True, True, False, False])


def test_fast_raw_context_uses_masked_mean_medoid_and_discards_banks():
    import fpde

    from experiments.dynamic_fpde_audio.raw_waveform_context import prepare_fast_raw_waveform_fpde_context

    result = prepare_fast_raw_waveform_fpde_context(
        fpde,
        [
            np.array([0.0, 0.0]),
            np.array([10.0, 10.0]),
            np.array([1.0, 1.0]),
            np.array([-5.0, -5.0]),
        ],
        ["class_a", "class_a", "class_a", "class_b"],
        sample_rates=[1000, 1000, 1000, 1000],
        sample_ids=["a0", "a10", "a1", "b"],
        target_sr=1000,
        segment_sec=0.002,
        hop_sec=0.002,
        prototype_selection="exact_medoid",
        medoid_block_size=2,
        retain_segment_banks=False,
    )

    context = result.context
    np.testing.assert_allclose(context.prototypes["class_a"], [1.0, 1.0])
    assert context.prototype_indices["class_a"] == 2
    assert context.segment_banks == {}
    assert context.details["distance_metric"] == "masked_mean_squared_distance"
    assert context.details["resample_method"] == "scipy.signal.resample_poly"
    assert context.details["prototype_provenance"]["class_a"]["source_sample_id"] == "a1"
    assert result.timings["medoid_runtime_sec"] >= 0.0


def test_exact_medoid_ignores_candidate_cap_and_sampled_seed_is_stable_across_processes():
    import fpde

    from experiments.dynamic_fpde_audio.raw_waveform_context import prepare_fast_raw_waveform_fpde_context

    result = prepare_fast_raw_waveform_fpde_context(
        fpde,
        [
            np.array([0.0, 0.0]),
            np.array([10.0, 10.0]),
            np.array([1.0, 1.0]),
            np.array([-5.0, -5.0]),
        ],
        ["class_a", "class_a", "class_a", "class_b"],
        sample_rates=[1000, 1000, 1000, 1000],
        target_sr=1000,
        segment_sec=0.002,
        hop_sec=0.002,
        prototype_selection="exact_medoid",
        medoid_block_size=2,
        max_candidates_per_label=1,
    )

    assert result.context.prototype_indices["class_a"] == 2
    assert result.context.details["medoid_details"]["class_a"]["n_candidates"] == 3

    code = (
        "import json; "
        "from experiments.dynamic_fpde_audio.raw_waveform_context import _candidate_indices; "
        "print(json.dumps(_candidate_indices(20, max_candidates_per_label=5, "
        "prototype_selection='sampled_medoid', seed=123, label='class_a').tolist()))"
    )
    first = subprocess.check_output([sys.executable, "-c", code], cwd=Path.cwd(), text=True).strip()
    second = subprocess.check_output([sys.executable, "-c", code], cwd=Path.cwd(), text=True).strip()
    assert json.loads(first) == json.loads(second)


def test_raw_context_cache_roundtrip_preserves_prototypes(tmp_path: Path):
    import fpde

    from experiments.dynamic_fpde_audio.raw_waveform_context import (
        load_raw_context_cache,
        prepare_fast_raw_waveform_fpde_context,
        save_raw_context_cache,
    )

    built = prepare_fast_raw_waveform_fpde_context(
        fpde,
        [np.array([0.0, 0.0]), np.array([1.0, 1.0]), np.array([-1.0, -1.0])],
        ["class_a", "class_a", "class_b"],
        sample_rates=[1000, 1000, 1000],
        target_sr=1000,
        segment_sec=0.002,
        hop_sec=0.002,
    ).context
    cache_path = tmp_path / "context.npz"
    save_raw_context_cache(cache_path, built)
    loaded = load_raw_context_cache(cache_path, fpde)

    for label in built.prototype_labels.tolist():
        np.testing.assert_allclose(loaded.prototypes[label], built.prototypes[label])
        np.testing.assert_array_equal(loaded.prototype_masks[label], built.prototype_masks[label])
    assert loaded.segment_banks == {}


@pytest.mark.skipif(not _cuda_available(), reason="CuPy CUDA device is not available")
def test_fast_raw_context_cpu_cuda_medoid_match():
    import fpde

    from experiments.dynamic_fpde_audio.raw_waveform_context import prepare_fast_raw_waveform_fpde_context

    waveforms = [
        np.linspace(0.0, 1.0, 12),
        np.linspace(0.1, 1.1, 12),
        np.linspace(4.0, 5.0, 12),
        -np.linspace(0.0, 1.0, 12),
    ]
    labels = ["class_a", "class_a", "class_a", "class_b"]
    kwargs = dict(
        sample_rates=[1000] * len(waveforms),
        target_sr=1000,
        segment_sec=0.004,
        hop_sec=0.002,
        prototype_selection="exact_medoid",
        medoid_block_size=2,
    )
    cpu = prepare_fast_raw_waveform_fpde_context(fpde, waveforms, labels, context_device="cpu", **kwargs).context
    cuda = prepare_fast_raw_waveform_fpde_context(fpde, waveforms, labels, context_device="cuda", **kwargs).context

    assert cpu.prototype_indices == cuda.prototype_indices
    for label in cpu.prototype_labels.tolist():
        np.testing.assert_allclose(cpu.prototypes[label], cuda.prototypes[label])


@pytest.mark.skipif(not _cuda_available(), reason="CuPy CUDA device is not available")
def test_raw_waveform_cpu_cuda_phi_match():
    import fpde

    from experiments.dynamic_fpde_audio.raw_waveform_context import prepare_fast_raw_waveform_fpde_context

    context = prepare_fast_raw_waveform_fpde_context(
        fpde,
        [np.linspace(0.0, 1.0, 20), -np.linspace(0.0, 1.0, 20)],
        ["class_a", "class_b"],
        sample_rates=[1000, 1000],
        target_sr=1000,
        segment_sec=0.006,
        hop_sec=0.003,
    ).context
    waveform = np.linspace(0.0, 1.0, 20)
    cpu = fpde.raw_waveform_fpde_explain_one(
        waveform,
        context,
        sample_rate=1000,
        target_label="class_a",
        lambda_grid=[0.0, 0.5, 1.0],
        device="cpu",
    )
    cuda = fpde.raw_waveform_fpde_explain_one(
        waveform,
        context,
        sample_rate=1000,
        target_label="class_a",
        lambda_grid=[0.0, 0.5, 1.0],
        device="cuda",
    )

    for lambda_hyb in (0.0, 0.5, 1.0):
        np.testing.assert_allclose(cpu.lambda_results[lambda_hyb]["phi"], cuda.lambda_results[lambda_hyb]["phi"], atol=1e-10)
        np.testing.assert_allclose(
            cpu.lambda_results[lambda_hyb]["window_evidence"],
            cuda.lambda_results[lambda_hyb]["window_evidence"],
            atol=1e-10,
        )


def test_raw_runner_smoke_outputs_schema_and_generator_hook(tmp_path: Path):
    from experiments.dynamic_fpde_audio.datasets import read_esc50_metadata
    from experiments.dynamic_fpde_audio.run_esc50_raw_waveform_fpde import RAW_SAMPLE_FIELDS, run_fold

    dataset_root = tmp_path / "ESC-50"
    output_dir = tmp_path / "outputs"
    _make_raw_esc50(dataset_root)
    samples = read_esc50_metadata(dataset_root)
    calls = []

    def generator(label, lambda_hyb, segment, sample_rate, role, metadata):
        assert role in {"target", "rival"}
        assert "evidence" in metadata
        assert segment.size > 0
        calls.append((label, lambda_hyb, role, metadata["start_sample"], metadata["end_sample"]))
        return segment

    rows, method_rows, errors = run_fold(
        samples,
        fold=1,
        output_dir=output_dir,
        mode="smoke",
        seed=3,
        target_sr=1000,
        segment_sec=0.02,
        hop_sec=0.01,
        lambda_grid=(0.0, 0.5, 1.0),
        top_k_segments=1,
        device="cpu",
        raw_generator=generator,
        skip_errors=False,
        save_plots=False,
    )

    assert errors == []
    assert rows
    assert method_rows
    assert set(RAW_SAMPLE_FIELDS).issuperset(rows[0])
    assert {row["lambda_hyb"] for row in rows} == {0.0, 0.5, 1.0}
    assert {row["shape_match"] for row in rows} == {True}
    assert all(np.isfinite(float(row["evidence"])) for row in rows)
    assert {"evidence_per_window", "evidence_per_valid_sample", "raw_diff_unscaled_evidence", "raw_cos_unscaled_evidence"}.issubset(rows[0])
    assert {"raw_diff_unscaled", "raw_cos_unscaled", "raw_hyb_l1_lambda_0.5"}.issubset({row["method"] for row in method_rows})
    sample_ids = {row["sample_id"] for row in rows}
    assert sum(1 for row in method_rows if row["method"] == "raw_diff_unscaled") == len(sample_ids)
    assert sum(1 for row in method_rows if row["method"] == "raw_cos_unscaled") == len(sample_ids)
    assert sum(1 for row in method_rows if str(row["method"]).startswith("raw_hyb_l1_lambda_")) == len(sample_ids) * 3
    assert all(float(row["sample_total_runtime_sec"]) >= 0.0 for row in rows)
    assert all(float(row["context_runtime_amortized_sec"]) >= 0.0 for row in rows)
    assert calls
    assert {call[2] for call in calls}.issubset({"target", "rival"})

    sample_dir = output_dir / "samples" / "1-class_a"
    assert (sample_dir / "summary.csv").exists()
    assert (sample_dir / "raw_hyb_lambda_0.5" / "window_evidence.csv").exists()
    assert (sample_dir / "raw_hyb_lambda_0.5" / "metrics.json").exists()
    assert (sample_dir / "raw_hyb_lambda_0.5" / "top_positive_segment.wav").exists()
    assert (sample_dir / "raw_hyb_lambda_0.5" / "top_negative_segment.wav").exists()
    assert (sample_dir / "raw_hyb_lambda_0.5" / "generated_target_lambda_0.5.wav").exists()
    assert (sample_dir / "raw_hyb_lambda_0.5" / "generated_rival_lambda_0.5.wav").exists()


def test_raw_cli_writes_dataset_level_outputs(tmp_path: Path):
    from experiments.dynamic_fpde_audio.run_esc50_raw_waveform_fpde import main

    dataset_root = tmp_path / "ESC-50"
    output_dir = tmp_path / "raw_outputs"
    _make_raw_esc50(dataset_root)

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
            "5",
            "--target-sr",
            "1000",
            "--segment-sec",
            "0.02",
            "--hop-sec",
            "0.01",
            "--lambda-grid",
            "0.0,0.5,1.0",
            "--device",
            "cpu",
            "--no-plots",
        ]
    )

    assert exit_code == 0
    config = output_dir / "raw_waveform_config.json"
    sample_metrics = output_dir / "results" / "raw_waveform_sample_metrics.csv"
    method_metrics = output_dir / "results" / "raw_waveform_method_metrics.csv"
    summary = output_dir / "results" / "raw_waveform_summary_by_lambda.csv"
    assert config.exists()
    assert sample_metrics.exists()
    assert method_metrics.exists()
    assert summary.exists()

    with sample_metrics.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert {
        "sample_id",
        "fold",
        "target_label",
        "rival_label",
        "lambda_hyb",
        "evidence",
        "n_windows",
        "input_length",
        "phi_shape",
        "shape_match",
        "generation_status",
        "top_positive_start_sample",
        "top_negative_start_sample",
        "evidence_per_window",
        "evidence_per_valid_sample",
        "positive_window_rate",
        "negative_window_rate",
        "n_valid_samples",
        "coverage_rate",
        "raw_diff_unscaled_evidence",
        "raw_cos_unscaled_evidence",
        "prototype_selection",
        "medoid_runtime_sec",
        "fold_context_runtime_sec",
        "context_runtime_amortized_sec",
        "sample_total_runtime_sec",
        "timings_overlap",
    }.issubset(rows[0])
    assert {row["generation_status"] for row in rows} == {'{"rival": "skipped", "target": "skipped"}'}

    with method_metrics.open("r", encoding="utf-8", newline="") as handle:
        method_rows = list(csv.DictReader(handle))
    assert method_rows
    assert {"raw_diff_unscaled", "raw_cos_unscaled", "raw_hyb_l1_lambda_0.5"}.issubset({row["method"] for row in method_rows})

    with summary.open("r", encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert summary_rows
    assert {
        "n_unique_samples",
        "evidence_per_window_mean",
        "evidence_per_valid_sample_mean",
        "absolute_evidence_mean",
        "positive_window_rate_mean",
        "negative_window_rate_mean",
        "coverage_rate_mean",
    }.issubset(summary_rows[0])


def test_raw_resume_does_not_duplicate_rows_and_retain_banks_disables_cache(tmp_path: Path):
    from experiments.dynamic_fpde_audio.run_esc50_raw_waveform_fpde import main

    dataset_root = tmp_path / "ESC-50"
    output_dir = tmp_path / "raw_outputs"
    _make_raw_esc50(dataset_root)

    args = [
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
        "--target-sr",
        "1000",
        "--segment-sec",
        "0.02",
        "--hop-sec",
        "0.01",
        "--lambda-grid",
        "0.0,0.5,1.0",
        "--device",
        "cpu",
        "--resume",
        "--no-plots",
    ]
    assert main(args) == 0
    assert main(args) == 0

    sample_metrics = output_dir / "results" / "raw_waveform_sample_metrics.csv"
    method_metrics = output_dir / "results" / "raw_waveform_method_metrics.csv"
    with sample_metrics.open("r", encoding="utf-8", newline="") as handle:
        sample_rows = list(csv.DictReader(handle))
    with method_metrics.open("r", encoding="utf-8", newline="") as handle:
        method_rows = list(csv.DictReader(handle))
    sample_ids = {row["sample_id"] for row in sample_rows}
    assert len(sample_rows) == len(sample_ids) * 3
    assert sum(1 for row in method_rows if row["method"] == "raw_diff_unscaled") == len(sample_ids)
    assert sum(1 for row in method_rows if row["method"] == "raw_cos_unscaled") == len(sample_ids)

    retain_output = tmp_path / "retain_outputs"
    assert (
        main(
            [
                "--dataset-root",
                str(dataset_root),
                "--output-dir",
                str(retain_output),
                "--mode",
                "smoke",
                "--fold",
                "1",
                "--seed",
                "8",
                "--target-sr",
                "1000",
                "--segment-sec",
                "0.02",
                "--hop-sec",
                "0.01",
                "--lambda-grid",
                "0.0",
                "--device",
                "cpu",
                "--retain-segment-banks",
                "--no-plots",
            ]
        )
        == 0
    )
    assert not list((retain_output / "cache" / "raw_context").glob("*.npz"))


def test_corrupt_raw_context_cache_is_rebuilt(tmp_path: Path):
    from experiments.dynamic_fpde_audio.run_esc50_raw_waveform_fpde import main

    dataset_root = tmp_path / "ESC-50"
    output_dir = tmp_path / "raw_outputs"
    _make_raw_esc50(dataset_root)
    args = [
        "--dataset-root",
        str(dataset_root),
        "--output-dir",
        str(output_dir),
        "--mode",
        "smoke",
        "--fold",
        "1",
        "--seed",
        "9",
        "--target-sr",
        "1000",
        "--segment-sec",
        "0.02",
        "--hop-sec",
        "0.01",
        "--lambda-grid",
        "0.0",
        "--device",
        "cpu",
        "--no-plots",
    ]
    assert main(args) == 0
    cache_files = list((output_dir / "cache" / "raw_context").glob("*.npz"))
    assert len(cache_files) == 1
    cache_files[0].write_text("not a valid npz", encoding="utf-8")

    assert main(args) == 0
    with np.load(cache_files[0], allow_pickle=False) as data:
        assert "metadata" in data.files
