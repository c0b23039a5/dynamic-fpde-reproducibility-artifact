"""Frame-ranking baselines for Native-Time Dynamic-FPDE audio experiments."""

from __future__ import annotations

import numpy as np


def frame_norm_scores(matrix: np.ndarray, feature_names: object | None = None) -> np.ndarray:
    """Rank frames by feature-vector norm.

    ``feature_names`` is accepted for compatibility with older call sites but
    is not used.
    """

    X = np.asarray(matrix, dtype=float)
    if X.ndim != 2:
        raise ValueError("matrix must be 2D")
    return np.linalg.norm(X, axis=1).astype(float, copy=True)


def raw_feature_norm_scores(matrix: np.ndarray, feature_names: object | None = None) -> np.ndarray:
    """Rank raw acoustic feature frames by vector norm."""

    return frame_norm_scores(matrix, feature_names)


def standardized_feature_norm_scores(matrix: np.ndarray, feature_names: object | None = None) -> np.ndarray:
    """Rank standardized acoustic feature frames by vector norm."""

    return frame_norm_scores(matrix, feature_names)


def energy_frame_scores(matrix: np.ndarray, feature_names: object | None = None) -> np.ndarray:
    """Backward-compatible alias for ``frame_norm_scores``."""

    return frame_norm_scores(matrix, feature_names)


def random_frame_scores(n_frames: int, *, seed: int, repetition: int = 0) -> np.ndarray:
    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    rng = np.random.default_rng(int(seed) + 1009 * int(repetition))
    return rng.random(int(n_frames))
