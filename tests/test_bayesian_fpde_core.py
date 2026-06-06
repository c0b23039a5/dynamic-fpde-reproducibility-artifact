from __future__ import annotations

import builtins
import sys
import types

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from bayesian_fpde.baselines import explain_shap, optional_baseline
from bayesian_fpde.bayesian_fpde import BayesianFPDEConfig, explain_bayesian_fpde
from bayesian_fpde.datasets import generate_synthetic_gaussian
from bayesian_fpde.fpde import FPDEConfig, class_prototypes, explain_fpde, true_fpde_attribution
from bayesian_fpde.metrics import calibration_metrics, deletion_insertion_metrics, replacement_values, sign_reliability_bins
from bayesian_fpde.prototypes import NIGPrior, fit_nig_posteriors, sample_posterior_prototypes


def _model_data():
    data = generate_synthetic_gaussian(n_samples=80, n_features=6, n_informative=3, random_seed=7)
    model = RandomForestClassifier(n_estimators=20, random_state=7).fit(data.X, data.y)
    return data, model


def test_nig_posterior_update_and_sampling_shape():
    data, _ = _model_data()
    post = fit_nig_posteriors(data.X, data.y, NIGPrior())
    assert sorted(post) == [0, 1]
    draws, labels = sample_posterior_prototypes(post, 11, seed=1)
    assert draws.shape == (11, 2, 6)
    assert labels.tolist() == [0, 1]
    assert np.all(np.isfinite(draws))


def test_fpde_scoring_and_true_attribution_shape():
    data, model = _model_data()
    prototypes, labels = class_prototypes(data.X, data.y)
    attr, c_plus, c_minus = explain_fpde(model, data.X[0], prototypes, labels, config=FPDEConfig(mode="hyb", lambda_hyb=0.5))
    truth = true_fpde_attribution(data.X[0], data.true_prototypes, data.labels, positive_label=c_plus, negative_label=c_minus)
    assert attr.shape == (6,)
    assert truth.shape == (6,)
    assert np.all(np.isfinite(attr))


def test_bayesian_ci_summary_contains_requested_columns():
    data, model = _model_data()
    result = explain_bayesian_fpde(
        model,
        data.X[0],
        data.X,
        data.y,
        config=BayesianFPDEConfig(n_posterior_samples=25, top_k=3),
        feature_names=data.feature_names,
        seed=4,
    )
    expected = {
        "posterior_mean",
        "posterior_std",
        "ci_lower_95",
        "ci_upper_95",
        "p_positive",
        "p_negative",
        "p_abs_gt_tau",
        "rank_mean",
        "rank_std",
        "rank_probability_top_k",
    }
    assert expected.issubset(result.summary.columns)
    assert result.samples.shape == (25, 6)


def test_deletion_metric_and_synthetic_coverage():
    data, model = _model_data()
    result = explain_bayesian_fpde(model, data.X[0], data.X, data.y, config=BayesianFPDEConfig(n_posterior_samples=25), seed=5)
    attr = result.summary["posterior_mean"].to_numpy(dtype=float)
    metrics = deletion_insertion_metrics(model, data.X[0], attr, np.mean(data.X, axis=0), target_label=result.positive_label)
    assert {"deletion_auc", "insertion_auc", "faithfulness_correlation", "faithfulness_delta_mean", "faithfulness_delta_abs_mean"}.issubset(metrics)
    truth = true_fpde_attribution(data.X[0], data.true_prototypes, data.labels, positive_label=result.positive_label, negative_label=result.negative_label)
    cal = calibration_metrics(result.summary, truth, top_k=3)
    assert 0.0 <= cal["coverage_95"] <= 1.0
    assert {"sign_brier_score", "sign_ece", "sign_accuracy_at_confidence_0_8", "sign_accuracy_at_confidence_0_9"}.issubset(cal)
    bins = sign_reliability_bins(result.summary, truth, n_bins=5)
    assert len(bins) == 5


def test_replacement_values_class_conditional_and_sampling():
    X = np.array([[1.0, 10.0], [3.0, 30.0], [100.0, 200.0]])
    y = np.array([0, 0, 1])
    assert np.allclose(replacement_values(X, strategy="mean"), [104.0 / 3.0, 80.0])
    assert np.allclose(replacement_values(X, y, target_label=0, strategy="class_conditional_mean"), [2.0, 20.0])
    assert replacement_values(X, strategy="marginal_sampling", rng=np.random.default_rng(1)).shape == (2,)
    assert replacement_values(X, strategy="permutation", rng=np.random.default_rng(1)).shape == (2,)


def test_optional_baseline_skip_record_for_unknown_dependency_path():
    data, model = _model_data()
    result = optional_baseline("aime", model, data.X[0], data.X, data.y, data.feature_names, int(model.classes_[0]), seed=0)
    assert result.status in {"skipped", "error"}
    assert result.attribution is None
    assert result.error_message
    assert result.dependency_available in {True, False}


def test_bayesshap_and_bayeslime_are_not_mapped_to_standard_adapters():
    data, model = _model_data()
    for method, expected in [
        ("bayesshap", "BayesSHAP adapter is not implemented in this artifact"),
        ("bayeslime", "BayesLIME adapter is not implemented in this artifact"),
        ("bayesian_aime", "Bayesian-AIME adapter is not implemented in this artifact"),
    ]:
        result = optional_baseline(method, model, data.X[0], data.X, data.y, data.feature_names, int(model.classes_[0]), seed=0)
        assert result.status == "skipped"
        assert result.attribution is None
        assert result.error_message == expected
        assert result.n_model_calls == 0


def test_shap_skipped_when_unavailable(monkeypatch):
    data, model = _model_data()
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "shap":
            raise ImportError("blocked shap for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    result = optional_baseline("shap", model, data.X[0], data.X, data.y, data.feature_names, int(model.classes_[0]), seed=0)
    assert result.status == "skipped"
    assert "SHAP is not installed" in result.error_message


def test_shap_uses_training_background_when_available(monkeypatch):
    data, model = _model_data()
    captured = {}

    class FakeValues:
        def __init__(self, n_features: int, n_classes: int):
            self.values = np.zeros((1, n_features, n_classes), dtype=float)

    class FakeExplainer:
        def __init__(self, predict_fn, background):
            captured["background"] = np.asarray(background)
            self.predict_fn = predict_fn

        def __call__(self, x):
            proba = self.predict_fn(x)
            return FakeValues(np.asarray(x).shape[1], proba.shape[1])

    monkeypatch.setitem(sys.modules, "shap", types.SimpleNamespace(Explainer=FakeExplainer))
    attr = explain_shap(model, data.X[0], data.X, int(model.classes_[0]), seed=2, max_background=10)
    assert attr.shape == data.X[0].shape
    assert captured["background"].shape[0] == 10
    assert not np.array_equal(captured["background"], data.X[0].reshape(1, -1))

    baseline = optional_baseline("shap", model, data.X[0], data.X, data.y, data.feature_names, int(model.classes_[0]), seed=2, max_background=10)
    assert baseline.status == "ok"
    assert baseline.background_size == 10
    assert baseline.max_background == 10
