from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from bayesian_fpde.baselines import optional_baseline
from bayesian_fpde.bayesian_fpde import BayesianFPDEConfig, explain_bayesian_fpde
from bayesian_fpde.datasets import generate_synthetic_gaussian
from bayesian_fpde.fpde import FPDEConfig, class_prototypes, explain_fpde, true_fpde_attribution
from bayesian_fpde.metrics import calibration_metrics, deletion_insertion_metrics
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
    assert {"deletion_auc", "insertion_auc", "faithfulness_correlation"}.issubset(metrics)
    truth = true_fpde_attribution(data.X[0], data.true_prototypes, data.labels, positive_label=result.positive_label, negative_label=result.negative_label)
    cal = calibration_metrics(result.summary, truth, top_k=3)
    assert 0.0 <= cal["coverage_95"] <= 1.0


def test_optional_baseline_skip_record_for_unknown_dependency_path():
    data, model = _model_data()
    result = optional_baseline("aime", model, data.X[0], data.X, data.y, data.feature_names, int(model.classes_[0]), seed=0)
    assert result.status in {"skipped", "error"}
    assert result.attribution is None
