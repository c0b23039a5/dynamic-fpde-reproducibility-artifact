from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from experiments.dynamic_fpde_audio.features import FeatureConfig
from experiments.dynamic_fpde_audio.native_prototypes import (
    select_class_exemplar_prototypes,
    select_native_exemplar_prototype,
)


def test_native_time_phi_preserves_variable_clip_shapes():
    from fpde import native_dynamic_fpde_explain_batch

    X_list = [np.zeros((3, 2)), np.ones((7, 2))]
    explanations = native_dynamic_fpde_explain_batch(
        X_list,
        p_targets=np.zeros(2),
        p_rivals=np.ones(2),
        target_labels=["a", "a"],
        rival_labels=["b", "b"],
        mode="dynamic_diff",
    )

    assert [exp.attributions.shape for exp in explanations] == [(3, 2), (7, 2)]


def test_prototype_selection_returns_real_exemplar_frame_with_metadata():
    matrices = {
        "a1": np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float),
        "a2": np.array([[0.2, 0.2], [3.0, 3.0]], dtype=float),
        "b1": np.array([[9.0, 9.0]], dtype=float),
    }
    labels = {"a1": "a", "a2": "a", "b1": "b"}

    proto = select_native_exemplar_prototype(
        matrices,
        labels,
        label="a",
        feature_config=FeatureConfig(target_sr=100, hop_length=10),
        feature_names=["x", "y"],
    )

    candidate_frames = [tuple(frame) for sample_id in ("a1", "a2") for frame in matrices[sample_id]]
    assert tuple(proto.vector) in candidate_frames
    assert {
        "source_sample_id",
        "source_frame_index",
        "source_time_sec",
        "label",
        "prototype_mode",
        "selection_rule",
        "feature_names",
    }.issubset(proto.metadata)
    assert proto.metadata["label"] == "a"


def test_class_prototypes_are_feature_vectors_not_time_series():
    matrices = {
        "a": np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float),
        "b": np.array([[9.0, 9.0], [8.0, 8.0]], dtype=float),
    }
    prototypes = select_class_exemplar_prototypes(matrices, {"a": "a", "b": "b"})

    assert {label: proto.vector.shape for label, proto in prototypes.items()} == {"a": (2,), "b": (2,)}


def test_class_prototypes_are_real_frames_with_complete_metadata():
    matrices = {
        "a1": np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0]], dtype=float),
        "a2": np.array([[0.2, 0.2, 1.2]], dtype=float),
        "b1": np.array([[9.0, 9.0, 0.0], [8.0, 8.0, 0.5]], dtype=float),
    }
    labels = {"a1": "a", "a2": "a", "b1": "b"}
    feature_names = ["x", "y", "z"]

    prototypes = select_class_exemplar_prototypes(
        matrices,
        labels,
        feature_config=FeatureConfig(target_sr=100, hop_length=10),
        feature_names=feature_names,
    )

    for label, proto in prototypes.items():
        assert proto.vector.shape == (3,)
        class_frames = [tuple(frame) for sample_id, X in matrices.items() if labels[sample_id] == label for frame in X]
        assert tuple(proto.vector) in class_frames
        assert {
            "source_sample_id",
            "source_frame_index",
            "source_time_sec",
            "label",
            "prototype_mode",
            "selection_rule",
            "feature_names",
        }.issubset(proto.metadata)
        assert proto.metadata["label"] == label
        assert proto.metadata["feature_names"] == feature_names


def test_native_diff_exactness_and_hyb_endpoints():
    from fpde import native_dynamic_cos_fpde, native_dynamic_diff_fpde, native_dynamic_hyb_fpde

    rng = np.random.default_rng(0)
    X = rng.normal(size=(11, 4))
    p_target = rng.normal(size=4)
    p_rival = rng.normal(size=4)

    diff_attr, diff_evidence = native_dynamic_diff_fpde(X, p_target, p_rival)
    cos_attr, _ = native_dynamic_cos_fpde(X, p_target, p_rival)
    hyb_diff, _, _ = native_dynamic_hyb_fpde(X, p_target, p_rival, lambda_hyb=1.0, normalize="none")
    hyb_cos, _, _ = native_dynamic_hyb_fpde(X, p_target, p_rival, lambda_hyb=0.0, normalize="none")

    assert diff_evidence == pytest.approx(float(np.sum(diff_attr)))
    np.testing.assert_allclose(hyb_diff, diff_attr)
    np.testing.assert_allclose(hyb_cos, cos_attr)


def test_native_cos_has_no_nan_or_inf_for_zero_and_near_zero_vectors():
    from fpde import native_dynamic_cos_fpde

    cases = [
        (np.zeros((5, 3)), np.zeros(3), np.zeros(3)),
        (np.full((5, 3), 1e-300), np.full(3, 1e-300), np.full(3, -1e-300)),
    ]
    for X, p_target, p_rival in cases:
        attr, evidence = native_dynamic_cos_fpde(X, p_target, p_rival)
        assert np.all(np.isfinite(attr))
        assert np.isfinite(evidence)


def test_runner_native_path_does_not_reference_legacy_resampling_call():
    from experiments.dynamic_fpde_audio import run_esc50_dynamic_fpde as runner

    source = inspect.getsource(runner)
    assert "resample_time_series_linear(" not in source
    assert "_explain_batch_cuda(api, [item]" not in source


def test_sample_fields_exclude_legacy_temporal_lengths():
    from experiments.dynamic_fpde_audio.run_esc50_dynamic_fpde import SAMPLE_FIELDS

    assert "prototype_length" not in SAMPLE_FIELDS
    assert "resampled_length" not in SAMPLE_FIELDS
    assert "shape_preserved" in SAMPLE_FIELDS
    assert "phi_shape" in SAMPLE_FIELDS
    assert "x_shape" in SAMPLE_FIELDS


def test_cuda_batching_groups_only_exact_native_shapes(monkeypatch):
    from fpde import NativeTimeDynamicFPDEExplanation

    from experiments.dynamic_fpde_audio.datasets import ESCSample
    from experiments.dynamic_fpde_audio.run_esc50_dynamic_fpde import _ResolvedNativeInput, _explain_batch_cuda
    from experiments.dynamic_fpde_audio import run_esc50_dynamic_fpde as runner

    calls: list[tuple[int, int, int]] = []

    def fake_cuda_available() -> None:
        return None

    def fake_diff(X_batch, target_batch, rival_batch):
        calls.append(tuple(X_batch.shape))
        return np.zeros_like(X_batch, dtype=float), np.zeros(X_batch.shape[0], dtype=float)

    monkeypatch.setattr(runner, "_ensure_cuda_backend", fake_cuda_available)
    monkeypatch.setattr(runner, "_native_diff_fpde_cuda", fake_diff)
    api = type("Api", (), {"NativeTimeDynamicFPDEExplanation": NativeTimeDynamicFPDEExplanation})()
    items = []
    for sample_id, T in [("a", 3), ("b", 7), ("c", 3)]:
        items.append(
            _ResolvedNativeInput(
                sample=ESCSample(sample_id, Path(f"{sample_id}.wav"), f"{sample_id}.wav", 1, "class_a"),
                X=np.ones((T, 2), dtype=float),
                p_target=np.zeros(2, dtype=float),
                p_rival=np.ones(2, dtype=float),
                anchor=np.zeros(2, dtype=float),
                target_label="class_a",
                rival_label="class_b",
                target_metadata={"label": "class_a"},
                rival_metadata={"label": "class_b"},
            )
        )

    out = _explain_batch_cuda(api, items, mode="dynamic_diff", lambda_hyb=0.5, normalize="none")

    assert set(out) == {"a", "b", "c"}
    assert sorted(calls) == [(1, 7, 2), (2, 3, 2)]
