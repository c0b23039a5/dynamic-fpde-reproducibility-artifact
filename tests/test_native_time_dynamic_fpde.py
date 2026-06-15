from __future__ import annotations

import inspect

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

