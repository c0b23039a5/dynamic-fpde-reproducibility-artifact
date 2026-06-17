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
    assert args.alignment_mode == "none"
    assert args.shift_max_ms == pytest.approx(20.0)
    assert args.coarse_step_ms == pytest.approx(1.0)
    assert args.fine_radius_ms == pytest.approx(2.0)
    assert args.fine_step_samples == 1
    assert args.coarse_top_k == 3
    assert args.minimum_overlap_ratio == pytest.approx(0.8)
    assert args.alignment_temperature == pytest.approx(0.05)
    assert args.overlap_penalty_weight == pytest.approx(1.0)
    assert args.generation_scope == "none"
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


def test_shift_with_mask_and_masked_distances_are_non_circular():
    from experiments.dynamic_fpde_audio.shift_robust_raw_waveform import (
        masked_shift_cosine,
        masked_shift_mse,
        shift_with_mask,
    )

    waveform = np.array([1.0, 2.0, 3.0, 4.0])
    mask = np.array([True, True, True, True])
    original = waveform.copy()

    shifted_pos, mask_pos = shift_with_mask(waveform, mask, 1)
    shifted_neg, mask_neg = shift_with_mask(waveform, mask, -1)
    shifted_max, mask_max = shift_with_mask(waveform, mask, 4)

    np.testing.assert_allclose(shifted_pos, [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(mask_pos, [False, True, True, True])
    np.testing.assert_allclose(shifted_neg, [2.0, 3.0, 4.0, 0.0])
    np.testing.assert_array_equal(mask_neg, [True, True, True, False])
    np.testing.assert_allclose(shifted_max, [0.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(mask_max, [False, False, False, False])
    np.testing.assert_allclose(waveform, original)

    window = np.array([0.0, 1.0, 2.0, 3.0])
    mse, overlap = masked_shift_mse(window, mask, waveform, mask, 1)
    assert mse == pytest.approx(0.0)
    assert overlap == pytest.approx(0.75)
    assert masked_shift_mse(window, np.zeros(4, dtype=bool), waveform, mask, 1)[0] == math.inf
    cos_dist, _ = masked_shift_cosine(window, mask, waveform, mask, 1)
    assert cos_dist == pytest.approx(0.0)
    zero_cos, _ = masked_shift_cosine(np.zeros(4), mask, np.zeros(4), mask, 0)
    assert zero_cos == pytest.approx(0.0)
    one_zero_cos, _ = masked_shift_cosine(np.ones(4), mask, np.zeros(4), mask, 0)
    assert one_zero_cos == pytest.approx(0.5)


def test_coarse_to_fine_soft_alignment_recovers_known_lag():
    from experiments.dynamic_fpde_audio.shift_robust_raw_waveform import (
        ShiftAlignmentConfig,
        align_prototype_to_window,
        generate_coarse_lags,
        masked_shift_mse,
    )

    sample_rate = 1000
    prototype = np.array([1.0, 2.0, 3.0, 4.0, 0.0, 0.0])
    prototype_mask = np.array([True, True, True, True, False, False])
    window = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 0.0])
    window_mask = np.array([True, True, True, True, True, False])
    config = ShiftAlignmentConfig(
        alignment_mode="soft_bounded",
        shift_max_ms=5.0,
        coarse_step_ms=2.0,
        fine_radius_ms=2.0,
        fine_step_samples=1,
        coarse_top_k=2,
        minimum_overlap_ratio=0.5,
        alignment_temperature=0.01,
    )

    coarse = generate_coarse_lags(sample_rate, 5.0, 2.0)
    assert {-5, 0, 5}.issubset(set(coarse.tolist()))
    result = align_prototype_to_window(
        window,
        window_mask,
        prototype,
        prototype_mask,
        sample_rate=sample_rate,
        lambda_hyb=1.0,
        config=config,
    )

    assert result.valid is True
    assert result.best_lag_samples == 1
    assert np.isfinite(result.costs).any()
    assert np.isfinite(result.entropy)
    assert np.isfinite(result.confidence)
    assert result.weights.sum() == pytest.approx(1.0)
    best_mse, _ = masked_shift_mse(window, window_mask, prototype, prototype_mask, result.best_lag_samples)
    zero_mse, _ = masked_shift_mse(window, window_mask, prototype, prototype_mask, 0)
    assert best_mse < zero_mse


def test_vectorized_lag_metrics_match_scalar_metrics():
    from experiments.dynamic_fpde_audio.shift_robust_raw_waveform import (
        build_shifted_batch,
        masked_shift_cosine,
        masked_shift_mse,
        precompute_lag_metrics,
        resolve_backend,
    )

    backend = resolve_backend("cpu")
    window = np.array([0.0, 1.0, 2.0, 3.0])
    window_mask = np.array([True, True, True, False])
    prototype = np.array([1.0, 2.0, 3.0, 4.0])
    prototype_mask = np.array([True, True, True, True])
    lags = np.array([-1, 0, 1], dtype=int)

    shifted, shifted_masks = build_shifted_batch(prototype, prototype_mask, lags, backend=backend)
    metrics = precompute_lag_metrics(window, window_mask, prototype, prototype_mask, lags, backend=backend)

    assert shifted.shape == (3, 4)
    assert shifted_masks.shape == (3, 4)
    for i, lag in enumerate(lags):
        mse, overlap = masked_shift_mse(window, window_mask, prototype, prototype_mask, int(lag))
        cosine, _ = masked_shift_cosine(window, window_mask, prototype, prototype_mask, int(lag))
        assert metrics.mse[i] == pytest.approx(mse)
        assert metrics.cosine_distance[i] == pytest.approx(cosine)
        assert metrics.overlap_ratio[i] == pytest.approx(overlap)


def test_hard_alignment_tie_break_uses_abs_lag_then_signed_lag():
    from experiments.dynamic_fpde_audio.shift_robust_raw_waveform import _hard_best_index

    assert _hard_best_index(np.array([1.0, 0.0, 0.0, 1.0]), np.array([-10, -2, 2, 10])) == 1
    assert _hard_best_index(np.array([0.0, 0.0, 0.0]), np.array([-3, 0, 3])) == 1


def test_alignment_fallback_preserves_real_lag0_cost_and_invalid_state():
    from experiments.dynamic_fpde_audio.shift_robust_raw_waveform import ShiftAlignmentConfig, align_prototype_to_window, masked_shift_mse

    window = np.array([1.0, 0.0, 0.0, 0.0])
    prototype = np.array([2.0, 0.0, 0.0, 0.0])
    mask = np.array([True, False, False, False])
    config = ShiftAlignmentConfig(
        alignment_mode="hard_bounded",
        shift_max_ms=2.0,
        coarse_step_ms=1.0,
        fine_radius_ms=1.0,
        minimum_overlap_ratio=0.9,
    )

    result = align_prototype_to_window(window, mask, prototype, mask, sample_rate=1000, lambda_hyb=1.0, config=config)
    expected_mse, expected_overlap = masked_shift_mse(window, mask, prototype, mask, 0)

    assert result.valid is False
    assert result.fallback_used is True
    assert result.fallback_reason == "minimum_overlap_not_met"
    assert result.best_lag_samples == 0
    assert result.best_cost == pytest.approx(expected_mse + config.overlap_penalty_weight * (1.0 - expected_overlap))
    assert result.best_cost != 0.0
    assert result.weights.sum() == pytest.approx(1.0)


def test_lag_metric_precompute_is_not_repeated_per_lambda(monkeypatch: pytest.MonkeyPatch):
    import experiments.dynamic_fpde_audio.shift_robust_raw_waveform as sr

    calls = {"count": 0}
    original = sr.precompute_lag_metrics

    def counted(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(sr, "precompute_lag_metrics", counted)
    config = sr.ShiftAlignmentConfig(
        alignment_mode="soft_bounded",
        shift_max_ms=3.0,
        coarse_step_ms=1.0,
        fine_radius_ms=1.0,
        coarse_top_k=2,
        minimum_overlap_ratio=0.5,
        alignment_temperature=0.01,
    )
    sr.explain_shift_robust_window(
        np.array([0.0, 1.0, 2.0, 3.0]),
        np.ones(4, dtype=bool),
        np.array([1.0, 2.0, 3.0, 0.0]),
        np.ones(4, dtype=bool),
        -np.array([1.0, 2.0, 3.0, 0.0]),
        np.ones(4, dtype=bool),
        sample_rate=1000,
        lambda_grid=sr.LAMBDA_GRID,
        config=config,
        backend=sr.resolve_backend("cpu"),
    )

    assert calls["count"] <= 4


def test_shift_robust_raw_explanation_preserves_shape_and_lambda_endpoints():
    import fpde

    from experiments.dynamic_fpde_audio.raw_waveform_context import prepare_fast_raw_waveform_fpde_context
    from experiments.dynamic_fpde_audio.shift_robust_raw_waveform import (
        LAMBDA_GRID,
        ShiftAlignmentConfig,
        explain_shift_robust_raw_waveform,
    )

    context = prepare_fast_raw_waveform_fpde_context(
        fpde,
        [np.array([1.0, 2.0, 3.0, 4.0]), -np.array([1.0, 2.0, 3.0, 4.0])],
        ["class_a", "class_b"],
        sample_rates=[1000, 1000],
        target_sr=1000,
        segment_sec=0.006,
        hop_sec=0.003,
    ).context
    waveform = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    explanation = explain_shift_robust_raw_waveform(
        fpde,
        waveform,
        context,
        sample_rate=1000,
        target_label="class_a",
        lambda_grid=LAMBDA_GRID,
        top_k_segments=1,
        device="cpu",
        config=ShiftAlignmentConfig(
            alignment_mode="soft_bounded",
            shift_max_ms=5.0,
            coarse_step_ms=2.0,
            fine_radius_ms=2.0,
            minimum_overlap_ratio=0.5,
            alignment_temperature=0.01,
            generation_scope="none",
        ),
    )

    assert tuple(explanation.lambda_results) == LAMBDA_GRID
    assert explanation.best_lambda is None
    assert explanation.details["diagnostic_only"] is True
    assert explanation.details["not_used_for_evaluation"] is True
    for lambda_hyb, result in explanation.lambda_results.items():
        assert result["phi"].shape == waveform.shape
        assert np.all(np.isfinite(result["phi"]))
        assert result["details"]["alignment_mode"] == "soft_bounded"
        assert result["details"]["alignment_valid_rate"] > 0.0
    np.testing.assert_allclose(
        explanation.lambda_results[0.0]["window_evidence"],
        np.sum(explanation.lambda_results[0.0]["window_attributions"], axis=1),
    )
    np.testing.assert_allclose(
        explanation.lambda_results[1.0]["window_evidence"],
        np.sum(explanation.lambda_results[1.0]["window_attributions"], axis=1),
    )


def test_generation_scope_selected_only_calls_requested_lambdas():
    import fpde

    from experiments.dynamic_fpde_audio.raw_waveform_context import prepare_fast_raw_waveform_fpde_context
    from experiments.dynamic_fpde_audio.shift_robust_raw_waveform import ShiftAlignmentConfig, explain_shift_robust_raw_waveform, resolve_backend

    backend = resolve_backend("cuda")
    assert backend.name == "cupy_cuda"
    assert backend.is_cuda is True

    context = prepare_fast_raw_waveform_fpde_context(
        fpde,
        [np.array([1.0, 2.0, 3.0, 4.0]), -np.array([1.0, 2.0, 3.0, 4.0])],
        ["class_a", "class_b"],
        sample_rates=[1000, 1000],
        target_sr=1000,
        segment_sec=0.006,
        hop_sec=0.003,
    ).context
    calls = []

    def generator(label, lambda_hyb, segment, sample_rate, role, metadata):
        calls.append((float(lambda_hyb), role))
        return segment

    explanation = explain_shift_robust_raw_waveform(
        fpde,
        np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        context,
        sample_rate=1000,
        target_label="class_a",
        lambda_grid=[0.0, 0.5, 1.0],
        top_k_segments=1,
        generator=generator,
        device="cpu",
        config=ShiftAlignmentConfig(
            alignment_mode="soft_bounded",
            shift_max_ms=5.0,
            coarse_step_ms=2.0,
            fine_radius_ms=2.0,
            minimum_overlap_ratio=0.5,
            alignment_temperature=0.01,
            generation_scope="selected",
            generation_selected_lambdas=(0.5,),
        ),
    )

    assert {item[0] for item in calls} == {0.5}
    assert explanation.lambda_results[0.0]["details"]["generation_selected"] is False
    assert explanation.lambda_results[0.5]["details"]["generation_selected"] is True
    assert explanation.lambda_results[1.0]["details"]["generation_selected"] is False


def test_sensitivity_spearman_uses_average_tie_ranks():
    from scipy.stats import rankdata

    from experiments.dynamic_fpde_audio.run_shift_robust_sensitivity import _pearson, _spearman

    a = np.array([1.0, 1.0, 3.0, 4.0])
    b = np.array([2.0, 3.0, 3.0, 5.0])
    expected = _pearson(rankdata(a, method="average"), rankdata(b, method="average"))

    assert _spearman(a, b) == pytest.approx(expected)
    assert _spearman(np.ones(4), np.ones(4)) == pytest.approx(1.0)
    assert _spearman(np.ones(4), np.arange(4.0)) == pytest.approx(0.0)


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


@pytest.mark.skipif(not _cuda_available(), reason="CuPy CUDA device is not available")
def test_shift_robust_raw_waveform_cpu_cuda_phi_match():
    import fpde

    from experiments.dynamic_fpde_audio.raw_waveform_context import prepare_fast_raw_waveform_fpde_context
    from experiments.dynamic_fpde_audio.shift_robust_raw_waveform import ShiftAlignmentConfig, explain_shift_robust_raw_waveform

    context = prepare_fast_raw_waveform_fpde_context(
        fpde,
        [np.array([1.0, 2.0, 3.0, 4.0]), -np.array([1.0, 2.0, 3.0, 4.0])],
        ["class_a", "class_b"],
        sample_rates=[1000, 1000],
        target_sr=1000,
        segment_sec=0.006,
        hop_sec=0.003,
    ).context
    waveform = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    config = ShiftAlignmentConfig(
        alignment_mode="soft_bounded",
        shift_max_ms=5.0,
        coarse_step_ms=2.0,
        fine_radius_ms=2.0,
        minimum_overlap_ratio=0.5,
        alignment_temperature=0.01,
    )
    cpu = explain_shift_robust_raw_waveform(
        fpde,
        waveform,
        context,
        sample_rate=1000,
        target_label="class_a",
        lambda_grid=[0.0, 0.5, 1.0],
        device="cpu",
        config=config,
    )
    cuda = explain_shift_robust_raw_waveform(
        fpde,
        waveform,
        context,
        sample_rate=1000,
        target_label="class_a",
        lambda_grid=[0.0, 0.5, 1.0],
        device="cuda",
        config=config,
    )

    for lambda_hyb in (0.0, 0.5, 1.0):
        np.testing.assert_allclose(cpu.lambda_results[lambda_hyb]["phi"], cuda.lambda_results[lambda_hyb]["phi"], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(
            cpu.lambda_results[lambda_hyb]["window_evidence"],
            cuda.lambda_results[lambda_hyb]["window_evidence"],
            rtol=1e-10,
            atol=1e-10,
        )
        assert cpu.lambda_results[lambda_hyb]["details"]["alignment_valid_rate"] == pytest.approx(
            cuda.lambda_results[lambda_hyb]["details"]["alignment_valid_rate"]
        )
        assert cuda.lambda_results[lambda_hyb]["details"]["resolved_backend"] == "cupy_cuda"


def test_raw_runner_smoke_outputs_schema_and_generator_hook(tmp_path: Path):
    from experiments.dynamic_fpde_audio.datasets import read_esc50_metadata
    from experiments.dynamic_fpde_audio.run_esc50_raw_waveform_fpde import RAW_SAMPLE_FIELDS, run_fold
    from experiments.dynamic_fpde_audio.shift_robust_raw_waveform import ShiftAlignmentConfig

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
        alignment_config=ShiftAlignmentConfig(generation_scope="selected"),
    )

    assert errors == []
    assert rows
    assert method_rows
    assert set(RAW_SAMPLE_FIELDS).issuperset(rows[0])
    assert {row["lambda_hyb"] for row in rows} == {0.0, 0.5, 1.0}
    assert {row["shape_match"] for row in rows} == {True}
    assert all(np.isfinite(float(row["evidence"])) for row in rows)
    assert {"evidence_per_window", "evidence_per_valid_sample", "raw_diff_unscaled_evidence", "raw_cos_unscaled_evidence"}.issubset(rows[0])
    assert {"raw_diff_unscaled_no_alignment", "raw_cos_unscaled_no_alignment", "raw_hyb_l1_no_alignment_lambda_0.5"}.issubset({row["method"] for row in method_rows})
    sample_ids = {row["sample_id"] for row in rows}
    assert sum(1 for row in method_rows if row["method"] == "raw_diff_unscaled_no_alignment") == len(sample_ids)
    assert sum(1 for row in method_rows if row["method"] == "raw_cos_unscaled_no_alignment") == len(sample_ids)
    assert sum(1 for row in method_rows if str(row["method"]).startswith("raw_hyb_l1_no_alignment_lambda_")) == len(sample_ids) * 3
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
            "--generation-scope",
            "selected",
            "--no-plots",
        ]
    )

    assert exit_code == 0
    config = output_dir / "raw_waveform_config.json"
    sample_metrics = output_dir / "results" / "raw_waveform_sample_metrics.csv"
    method_metrics = output_dir / "results" / "raw_waveform_method_metrics.csv"
    alignment_metrics = output_dir / "results" / "window_alignment_metrics.csv"
    summary = output_dir / "results" / "raw_waveform_summary_by_lambda.csv"
    assert config.exists()
    assert sample_metrics.exists()
    assert method_metrics.exists()
    assert not alignment_metrics.exists()
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
    assert {"raw_diff_unscaled_no_alignment", "raw_cos_unscaled_no_alignment", "raw_hyb_l1_no_alignment_lambda_0.5"}.issubset({row["method"] for row in method_rows})

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


def test_raw_cli_soft_bounded_outputs_alignment_metrics(tmp_path: Path):
    from experiments.dynamic_fpde_audio.datasets import read_esc50_metadata, split_esc50, get_mode_config
    from experiments.dynamic_fpde_audio.run_esc50_raw_waveform_fpde import _raw_context_cache_key, main
    from experiments.dynamic_fpde_audio.shift_robust_raw_waveform import ShiftAlignmentConfig

    dataset_root = tmp_path / "ESC-50"
    output_dir = tmp_path / "shift_outputs"
    _make_raw_esc50(dataset_root)
    samples = read_esc50_metadata(dataset_root)
    train_samples, _, _ = split_esc50(samples, fold=1, mode_config=get_mode_config("smoke"), seed=11)
    none_key = _raw_context_cache_key(
        train_samples,
        fold=1,
        seed=11,
        target_sr=1000,
        segment_sec=0.02,
        hop_sec=0.01,
        prototype_selection="exact_medoid",
        medoid_block_size=128,
        max_prototype_candidates=None,
        context_device="cpu",
    )
    soft_key = _raw_context_cache_key(
        train_samples,
        fold=1,
        seed=11,
        target_sr=1000,
        segment_sec=0.02,
        hop_sec=0.01,
        prototype_selection="exact_medoid",
        medoid_block_size=128,
        max_prototype_candidates=None,
        context_device="cpu",
    )
    assert none_key == soft_key

    assert (
        main(
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
                "11",
                "--target-sr",
                "1000",
                "--segment-sec",
                "0.02",
                "--hop-sec",
                "0.01",
                "--lambda-grid",
                "0.0,1.0",
                "--alignment-mode",
                "soft_bounded",
                "--shift-max-ms",
                "5",
                "--coarse-step-ms",
                "2",
                "--fine-radius-ms",
                "2",
                "--fine-step-samples",
                "1",
                "--coarse-top-k",
                "2",
                "--minimum-overlap-ratio",
                "0.5",
                "--alignment-temperature",
                "0.01",
                "--device",
                "cpu",
                "--save-alignment-details",
                "--no-plots",
            ]
        )
        == 0
    )

    with (output_dir / "results" / "raw_waveform_sample_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
        sample_rows = list(csv.DictReader(handle))
    with (output_dir / "results" / "raw_waveform_method_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
        method_rows = list(csv.DictReader(handle))
    part_paths = sorted((output_dir / "results" / "window_alignment").glob("part-*"))
    assert part_paths
    if part_paths[0].suffix == ".parquet":
        import pandas as pd

        alignment_rows = pd.read_parquet(part_paths[0]).to_dict("records")
    else:
        with part_paths[0].open("r", encoding="utf-8", newline="") as handle:
            alignment_rows = list(csv.DictReader(handle))

    assert sample_rows
    assert alignment_rows
    assert {row["alignment_mode"] for row in sample_rows} == {"soft_bounded"}
    assert {row["alignment_mode"] for row in alignment_rows} == {"soft_bounded"}
    assert {"mean_abs_target_lag_ms", "alignment_valid_rate", "target_alignment_confidence_mean"}.issubset(sample_rows[0])
    assert {"shift_robust_raw_diff_lambda_1.0", "shift_robust_raw_cos_lambda_1.0", "shift_robust_raw_hyb_lambda_1.0"}.issubset(
        {row["method"] for row in method_rows}
    )
    assert all(np.isfinite(float(row["target_alignment_confidence"])) for row in alignment_rows)


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
    assert sum(1 for row in method_rows if row["method"] == "raw_diff_unscaled_no_alignment") == len(sample_ids)
    assert sum(1 for row in method_rows if row["method"] == "raw_cos_unscaled_no_alignment") == len(sample_ids)

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


def test_raw_context_cache_hit_skips_training_wav_loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from experiments.dynamic_fpde_audio.datasets import read_esc50_metadata
    import experiments.dynamic_fpde_audio.run_esc50_raw_waveform_fpde as runner
    from experiments.dynamic_fpde_audio.run_esc50_raw_waveform_fpde import run_fold

    dataset_root = tmp_path / "ESC-50"
    _make_raw_esc50(dataset_root)
    samples = read_esc50_metadata(dataset_root)
    cache_dir = tmp_path / "cache"

    run_fold(
        samples,
        fold=1,
        output_dir=tmp_path / "first",
        mode="smoke",
        seed=13,
        target_sr=1000,
        segment_sec=0.02,
        hop_sec=0.01,
        lambda_grid=(0.0,),
        top_k_segments=1,
        device="cpu",
        context_cache_dir=cache_dir,
        save_plots=False,
    )

    def fail_loader(_samples):
        raise AssertionError("training WAV loader should not run on a context cache hit")

    monkeypatch.setattr(runner, "_load_waveforms", fail_loader)
    rows, _method_rows, errors = run_fold(
        samples,
        fold=1,
        output_dir=tmp_path / "second",
        mode="smoke",
        seed=13,
        target_sr=1000,
        segment_sec=0.02,
        hop_sec=0.01,
        lambda_grid=(0.0,),
        top_k_segments=1,
        device="cpu",
        context_cache_dir=cache_dir,
        save_plots=False,
    )

    assert errors == []
    assert rows
    assert {row["context_cache_hit"] for row in rows} == {True}
    assert {row["training_audio_load_skipped"] for row in rows} == {True}
    assert {float(row["fold_audio_load_runtime_sec"]) for row in rows} == {0.0}


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
