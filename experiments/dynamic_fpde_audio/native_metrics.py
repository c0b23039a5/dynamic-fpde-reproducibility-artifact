"""Native-Time prototype-evidence diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np


def _rank_scores(scores: np.ndarray, rank_mode: str) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1:
        raise ValueError("ranking scores must be a 1D frame vector")
    if rank_mode == "signed":
        ranking = values
    elif rank_mode == "abs":
        ranking = np.abs(values)
    elif rank_mode == "positive":
        ranking = np.maximum(values, 0.0)
    else:
        raise ValueError("rank_mode must be signed, abs, or positive")
    return np.argsort(ranking, kind="mergesort")[::-1]


def native_temporal_deletion_insertion_curves(
    explanation: Any,
    *,
    ranking_scores: np.ndarray | None = None,
    evidence_by_frame: np.ndarray | None = None,
    steps: int = 20,
    rank_mode: str = "signed",
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Return normalized prototype-evidence removal/recovery diagnostics.

    The diagnostic works on native frame indices. It does not resample, pad, or
    query a black-box model, so it should not be interpreted as causal
    faithfulness.
    """

    if steps <= 0:
        raise ValueError("steps must be positive")
    attr = np.asarray(explanation.attributions, dtype=float)
    if attr.ndim != 2:
        raise ValueError("explanation.attributions must be 2D")
    T = attr.shape[0]
    if T <= 0:
        raise ValueError("explanation must contain at least one frame")

    frame_evidence = np.asarray(evidence_by_frame, dtype=float) if evidence_by_frame is not None else np.sum(attr, axis=1)
    if frame_evidence.shape != (T,):
        raise ValueError(f"evidence_by_frame must have shape {(T,)}, got {frame_evidence.shape}")
    if not np.all(np.isfinite(frame_evidence)):
        raise ValueError("frame evidence contains NaN or inf")

    scores = np.asarray(ranking_scores, dtype=float) if ranking_scores is not None else frame_evidence
    if scores.shape != (T,):
        raise ValueError(f"ranking_scores must have shape {(T,)}, got {scores.shape}")
    if not np.all(np.isfinite(scores)):
        raise ValueError("ranking scores contain NaN or inf")

    order = _rank_scores(scores, rank_mode)
    fractions = np.linspace(0.0, 1.0, int(steps) + 1)
    base_evidence = float(np.sum(frame_evidence))
    normalizer = max(abs(base_evidence), float(np.sum(np.abs(frame_evidence))), float(eps))
    deletion_drop_curve: list[float] = []
    insertion_gain_curve: list[float] = []
    native_frame_indices_by_step: list[list[int]] = []
    for fraction in fractions:
        k = int(round(float(fraction) * T))
        selected = order[:k]
        removed = float(np.sum(frame_evidence[selected])) if k else 0.0
        remaining = base_evidence - removed
        recovered = removed
        deletion_drop_curve.append(float((base_evidence - remaining) / normalizer))
        insertion_gain_curve.append(float(recovered / normalizer))
        native_frame_indices_by_step.append([int(idx) for idx in selected.tolist()])

    deletion_auc = float(np.trapezoid(deletion_drop_curve, fractions))
    insertion_auc = float(np.trapezoid(insertion_gain_curve, fractions))
    return {
        "fractions": [float(value) for value in fractions.tolist()],
        "deletion_drop_curve": deletion_drop_curve,
        "insertion_gain_curve": insertion_gain_curve,
        "deletion_drop_auc": deletion_auc,
        "insertion_gain_auc": insertion_auc,
        "combined_score": float((deletion_auc + insertion_auc) / 2.0),
        "rank_mode": rank_mode,
        "diagnostic": "normalized_native_time_prototype_evidence_removal_recovery",
        "native_frame_indices_by_step": native_frame_indices_by_step,
        "causal_faithfulness": False,
        "black_box_faithfulness": False,
    }
