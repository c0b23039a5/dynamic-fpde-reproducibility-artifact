"""Frame-ranking baselines for Dynamic-FPDE audio experiments."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def energy_frame_scores(matrix: np.ndarray, feature_names: Sequence[str]) -> np.ndarray:
    X = np.asarray(matrix, dtype=float)
    if X.ndim != 2:
        raise ValueError("matrix must be 2D")
    try:
        rms_idx = list(feature_names).index("rms")
    except ValueError as exc:
        raise ValueError("feature_names must include 'rms' for the energy baseline") from exc
    return X[:, rms_idx].astype(float, copy=True)


def random_frame_scores(n_frames: int, *, seed: int, repetition: int = 0) -> np.ndarray:
    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    rng = np.random.default_rng(int(seed) + 1009 * int(repetition))
    return rng.random(int(n_frames))

