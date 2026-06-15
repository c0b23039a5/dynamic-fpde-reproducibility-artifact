from __future__ import annotations

import numpy as np
import pytest

from experiments.dynamic_fpde_audio.baselines import energy_frame_scores, random_frame_scores
from experiments.dynamic_fpde_audio.features import fit_standardizer, transform_features


def test_feature_standardization_uses_train_statistics_and_stays_finite():
    train = [
        np.array([[1.0, 10.0], [3.0, 10.0]], dtype=float),
        np.array([[5.0, 10.0]], dtype=float),
    ]
    test = np.array([[7.0, 10.0], [9.0, 10.0]], dtype=float)

    standardizer = fit_standardizer(train, ["rms", "mfcc_1"])
    transformed_train = [transform_features(X, standardizer) for X in train]
    transformed_test = transform_features(test, standardizer)

    stacked_train = np.vstack(transformed_train)
    np.testing.assert_allclose(np.mean(stacked_train, axis=0), [0.0, 0.0], atol=1e-12)
    assert np.all(np.isfinite(stacked_train))
    assert np.all(np.isfinite(transformed_test))
    np.testing.assert_allclose(transformed_test[:, 1], [0.0, 0.0])


def test_energy_baseline_ranks_higher_rms_frames_first():
    X = np.array([[0.2, 1.0], [0.9, 0.0], [0.4, 2.0]], dtype=float)

    scores = energy_frame_scores(X, ["rms", "mfcc_1"])
    order = np.argsort(scores)[::-1]

    assert order.tolist() == [1, 2, 0]
    with pytest.raises(ValueError, match="rms"):
        energy_frame_scores(X, ["energy", "mfcc_1"])


def test_random_baseline_is_deterministic_for_seed_and_repetition():
    scores_a = random_frame_scores(8, seed=123, repetition=2)
    scores_b = random_frame_scores(8, seed=123, repetition=2)
    scores_c = random_frame_scores(8, seed=123, repetition=3)

    np.testing.assert_allclose(scores_a, scores_b)
    assert not np.allclose(scores_a, scores_c)

