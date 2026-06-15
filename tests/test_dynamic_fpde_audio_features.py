from __future__ import annotations

import numpy as np
import pytest

from experiments.dynamic_fpde_audio.baselines import energy_frame_scores, random_frame_scores
from experiments.dynamic_fpde_audio.features import extract_frame_features, fit_standardizer, transform_features


def test_acoustic_feature_standardization_uses_train_statistics_and_stays_finite():
    train = [
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
        np.array([[5.0, 6.0]], dtype=float),
    ]
    test = np.array([[7.0, 8.0], [9.0, 10.0]], dtype=float)

    standardizer = fit_standardizer(train, ["a", "b"])
    transformed_train = [transform_features(X, standardizer) for X in train]
    transformed_test = transform_features(test, standardizer)

    stacked_train = np.vstack(transformed_train)
    np.testing.assert_allclose(np.mean(stacked_train, axis=0), [0.0, 0.0], atol=1e-12)
    assert np.all(np.isfinite(stacked_train))
    assert np.all(np.isfinite(transformed_test))


def test_energy_baseline_ranks_larger_acoustic_feature_norm_first():
    X = np.array([[0.2, 0.0], [0.6, 0.8], [0.4, 0.1]], dtype=float)

    scores = energy_frame_scores(X, ["a", "b"])
    order = np.argsort(scores)[::-1]

    assert order.tolist() == [1, 2, 0]


def test_random_baseline_is_deterministic_for_seed_and_repetition():
    scores_a = random_frame_scores(8, seed=123, repetition=2)
    scores_b = random_frame_scores(8, seed=123, repetition=2)
    scores_c = random_frame_scores(8, seed=123, repetition=3)

    np.testing.assert_allclose(scores_a, scores_b)
    assert not np.allclose(scores_a, scores_c)


def _write_wav(path, y: np.ndarray, *, sr: int = 8000) -> None:
    soundfile = pytest.importorskip("soundfile")
    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(path, y.astype(np.float32, copy=False), sr)


def test_feature_extraction_preserves_duration_differences(tmp_path):
    short = tmp_path / "short.wav"
    long = tmp_path / "long.wav"
    _write_wav(short, np.sin(np.linspace(0.0, 4.0, 900, endpoint=False)))
    _write_wav(long, np.sin(np.linspace(0.0, 8.0, 1800, endpoint=False)))

    X_short, names_short = extract_frame_features(short, target_sr=8000, frame_length=256, hop_length=128)
    X_long, names_long = extract_frame_features(long, target_sr=8000, frame_length=256, hop_length=128)

    assert names_short == names_long
    assert X_short.shape[1] == X_long.shape[1]
    assert X_short.shape[0] != X_long.shape[0]


def test_sub_frame_clip_gets_one_intra_clip_analysis_frame(tmp_path):
    tiny = tmp_path / "tiny.wav"
    _write_wav(tiny, np.ones(64, dtype=np.float32), sr=8000)

    X, _ = extract_frame_features(tiny, target_sr=8000, frame_length=256, hop_length=128)

    assert X.shape[0] == 1


def test_feature_extractor_does_not_force_global_fixed_T(tmp_path):
    paths = []
    for name, n_samples in [("a.wav", 512), ("b.wav", 640), ("c.wav", 1024)]:
        path = tmp_path / name
        _write_wav(path, np.linspace(-1.0, 1.0, n_samples, dtype=np.float32), sr=8000)
        paths.append(path)

    lengths = [
        extract_frame_features(path, target_sr=8000, frame_length=256, hop_length=128)[0].shape[0]
        for path in paths
    ]

    assert len(set(lengths)) > 1
