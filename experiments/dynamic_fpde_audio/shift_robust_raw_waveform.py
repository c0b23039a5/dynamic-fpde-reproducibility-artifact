"""Shift-robust Raw-Waveform Dynamic-FPDE helpers.

This module implements bounded, non-circular lag alignment for the artifact's
raw-waveform runner.  The legacy no-alignment path remains delegated to the
installed ``fpde`` package by the runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.dynamic_fpde_audio.raw_waveform_context import _as_raw_waveform, _raw_windows


LAMBDA_GRID = tuple(i / 10.0 for i in range(11))
EPS = 1e-12


@dataclass(frozen=True)
class ShiftAlignmentConfig:
    alignment_mode: str = "none"
    shift_max_ms: float = 20.0
    coarse_step_ms: float = 1.0
    fine_radius_ms: float = 2.0
    fine_step_samples: int = 1
    coarse_top_k: int = 3
    minimum_overlap_ratio: float = 0.8
    alignment_temperature: float = 0.05
    overlap_penalty_weight: float = 1.0
    save_alignment_details: bool = False
    generation_scope: str = "none"


@dataclass(frozen=True)
class ShiftAlignmentResult:
    candidate_lags_samples: np.ndarray
    candidate_lags_ms: np.ndarray
    costs: np.ndarray
    weights: np.ndarray
    best_lag_samples: int
    best_lag_ms: float
    best_cost: float
    overlap_ratio: float
    entropy: float
    confidence: float
    valid: bool


@dataclass(frozen=True)
class ShiftRobustWindowExplanation:
    phi_diff_by_lambda: dict[float, np.ndarray]
    phi_cos_by_lambda: dict[float, np.ndarray]
    phi_hyb_by_lambda: dict[float, np.ndarray]
    evidence_by_lambda: dict[float, float]
    target_alignment_by_lambda: dict[float, ShiftAlignmentResult]
    rival_alignment_by_lambda: dict[float, ShiftAlignmentResult]
    mask: np.ndarray


def validate_alignment_config(config: ShiftAlignmentConfig) -> ShiftAlignmentConfig:
    if config.alignment_mode not in {"none", "hard_bounded", "soft_bounded"}:
        raise ValueError("alignment_mode must be none, hard_bounded, or soft_bounded")
    if config.generation_scope not in {"none", "selected", "all"}:
        raise ValueError("generation_scope must be none, selected, or all")
    for name in ("shift_max_ms", "coarse_step_ms", "fine_radius_ms", "minimum_overlap_ratio"):
        value = float(getattr(config, name))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if config.coarse_step_ms <= 0.0:
        raise ValueError("coarse_step_ms must be positive")
    if int(config.fine_step_samples) <= 0:
        raise ValueError("fine_step_samples must be positive")
    if int(config.coarse_top_k) <= 0:
        raise ValueError("coarse_top_k must be positive")
    if not (0.0 <= float(config.minimum_overlap_ratio) <= 1.0):
        raise ValueError("minimum_overlap_ratio must be in [0, 1]")
    if config.alignment_mode == "soft_bounded" and float(config.alignment_temperature) <= 0.0:
        raise ValueError("alignment_temperature must be positive for soft_bounded")
    if not np.isfinite(float(config.overlap_penalty_weight)) or float(config.overlap_penalty_weight) < 0.0:
        raise ValueError("overlap_penalty_weight must be finite and non-negative")
    return config


def shift_with_mask(waveform: np.ndarray, mask: np.ndarray, lag_samples: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(waveform, dtype=float)
    valid = np.asarray(mask, dtype=bool)
    if values.shape != valid.shape:
        raise ValueError("waveform and mask must have the same shape")
    shifted = np.zeros_like(values, dtype=float)
    shifted_mask = np.zeros_like(valid, dtype=bool)
    lag = int(lag_samples)
    n = int(values.shape[0])
    if abs(lag) >= n:
        return shifted, shifted_mask
    if lag >= 0:
        shifted[lag:] = values[: n - lag]
        shifted_mask[lag:] = valid[: n - lag]
    else:
        k = -lag
        shifted[: n - k] = values[k:]
        shifted_mask[: n - k] = valid[k:]
    return shifted, shifted_mask


def masked_shift_mse(
    window: np.ndarray,
    window_mask: np.ndarray,
    prototype: np.ndarray,
    prototype_mask: np.ndarray,
    lag_samples: int,
) -> tuple[float, float]:
    w = np.asarray(window, dtype=float)
    wm = np.asarray(window_mask, dtype=bool)
    p = np.asarray(prototype, dtype=float)
    pm = np.asarray(prototype_mask, dtype=bool)
    if w.shape != wm.shape or w.shape != p.shape or p.shape != pm.shape:
        raise ValueError("window, prototype, and masks must share shape")
    shifted, shifted_mask = shift_with_mask(p, pm, lag_samples)
    valid = wm & shifted_mask
    overlap_ratio = float(np.count_nonzero(valid) / max(1, w.shape[0]))
    if not np.any(valid):
        return float("inf"), overlap_ratio
    diff = w[valid] - shifted[valid]
    return float(np.mean(diff * diff)), overlap_ratio


def masked_shift_cosine(
    window: np.ndarray,
    window_mask: np.ndarray,
    prototype: np.ndarray,
    prototype_mask: np.ndarray,
    lag_samples: int,
    eps: float = EPS,
) -> tuple[float, float]:
    w = np.asarray(window, dtype=float)
    wm = np.asarray(window_mask, dtype=bool)
    p = np.asarray(prototype, dtype=float)
    pm = np.asarray(prototype_mask, dtype=bool)
    if w.shape != wm.shape or w.shape != p.shape or p.shape != pm.shape:
        raise ValueError("window, prototype, and masks must share shape")
    shifted, shifted_mask = shift_with_mask(p, pm, lag_samples)
    valid = wm & shifted_mask
    overlap_ratio = float(np.count_nonzero(valid) / max(1, w.shape[0]))
    if not np.any(valid):
        return 1.0, overlap_ratio
    wv = w[valid]
    pv = shifted[valid]
    w_norm = float(np.linalg.norm(wv))
    p_norm = float(np.linalg.norm(pv))
    if w_norm <= eps and p_norm <= eps:
        cos = 1.0
    elif w_norm <= eps or p_norm <= eps:
        cos = 0.0
    else:
        cos = float(np.dot(wv, pv) / ((w_norm * p_norm) + eps))
    cos = float(np.clip(cos, -1.0, 1.0))
    dist = (1.0 - cos) / 2.0
    return float(dist if np.isfinite(dist) else 1.0), overlap_ratio


def generate_coarse_lags(sample_rate: int, shift_max_ms: float, coarse_step_ms: float) -> np.ndarray:
    sr = int(sample_rate)
    if sr <= 0:
        raise ValueError("sample_rate must be positive")
    shift_max_samples = int(round(sr * float(shift_max_ms) / 1000.0))
    coarse_step_samples = max(1, int(round(sr * float(coarse_step_ms) / 1000.0)))
    values = list(range(-shift_max_samples, shift_max_samples + 1, coarse_step_samples))
    values.extend([-shift_max_samples, 0, shift_max_samples])
    return np.asarray(sorted(set(int(v) for v in values)), dtype=np.intp)


def _fine_lags(
    coarse_lags: np.ndarray,
    coarse_cost: np.ndarray,
    *,
    sample_rate: int,
    shift_max_ms: float,
    fine_radius_ms: float,
    fine_step_samples: int,
    coarse_top_k: int,
) -> np.ndarray:
    finite_idx = np.flatnonzero(np.isfinite(coarse_cost))
    if finite_idx.size == 0:
        centers = np.array([0], dtype=np.intp)
    else:
        order = sorted(
            finite_idx.tolist(),
            key=lambda i: (float(coarse_cost[i]), abs(int(coarse_lags[i])), int(coarse_lags[i])),
        )
        centers = coarse_lags[order[: int(coarse_top_k)]]
    shift_max_samples = int(round(int(sample_rate) * float(shift_max_ms) / 1000.0))
    radius = int(round(int(sample_rate) * float(fine_radius_ms) / 1000.0))
    step = max(1, int(fine_step_samples))
    values: set[int] = set()
    for center in centers.tolist():
        values.update(range(int(center) - radius, int(center) + radius + 1, step))
    values.update([-shift_max_samples, 0, shift_max_samples])
    return np.asarray(sorted(v for v in values if -shift_max_samples <= v <= shift_max_samples), dtype=np.intp)


def _precompute_shift_metrics(
    window: np.ndarray,
    window_mask: np.ndarray,
    prototype: np.ndarray,
    prototype_mask: np.ndarray,
    lags: np.ndarray,
    *,
    minimum_overlap_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mse = np.empty(lags.shape[0], dtype=float)
    cosine = np.empty(lags.shape[0], dtype=float)
    overlap = np.empty(lags.shape[0], dtype=float)
    for i, lag in enumerate(lags.tolist()):
        mse_value, overlap_ratio = masked_shift_mse(window, window_mask, prototype, prototype_mask, int(lag))
        cos_value, _ = masked_shift_cosine(window, window_mask, prototype, prototype_mask, int(lag))
        if overlap_ratio < float(minimum_overlap_ratio):
            mse_value = float("inf")
            cos_value = float("inf")
        mse[i] = mse_value
        cosine[i] = cos_value
        overlap[i] = overlap_ratio
    return mse, cosine, overlap


def _alignment_cost(mse: np.ndarray, cosine: np.ndarray, overlap: np.ndarray, lambda_hyb: float, penalty_weight: float) -> np.ndarray:
    finite_mse = mse[np.isfinite(mse)]
    scale = float(np.median(finite_mse)) if finite_mse.size else 1.0
    if not np.isfinite(scale) or scale <= EPS:
        scale = 1.0
    lambda_value = float(lambda_hyb)
    mse_term = np.zeros_like(mse, dtype=float) if lambda_value == 0.0 else mse / (scale + EPS)
    cos_term = np.zeros_like(cosine, dtype=float) if lambda_value == 1.0 else cosine
    cost = lambda_value * mse_term
    cost = cost + (1.0 - lambda_value) * cos_term
    cost = cost + float(penalty_weight) * (1.0 - overlap)
    cost[~np.isfinite(mse) | ~np.isfinite(cosine)] = np.inf
    return cost


def _weights_from_cost(cost: np.ndarray, mode: str, temperature: float) -> tuple[np.ndarray, float, float]:
    weights = np.zeros_like(cost, dtype=float)
    valid_idx = np.flatnonzero(np.isfinite(cost))
    if valid_idx.size == 0:
        return weights, 0.0, 0.0
    if mode == "hard_bounded" or valid_idx.size == 1:
        best = min(valid_idx.tolist(), key=lambda i: (float(cost[i]), i))
        weights[best] = 1.0
    else:
        stable = -cost[valid_idx] / float(temperature)
        stable = stable - float(np.max(stable))
        exp_values = np.exp(stable)
        total = float(np.sum(exp_values))
        if not np.isfinite(total) or total <= 0.0:
            best = min(valid_idx.tolist(), key=lambda i: (float(cost[i]), i))
            weights[best] = 1.0
        else:
            weights[valid_idx] = exp_values / total
    nonzero = weights[weights > 0.0]
    entropy = float(-np.sum(nonzero * np.log(nonzero + EPS))) if nonzero.size else 0.0
    confidence = 1.0 if nonzero.size <= 1 else float(1.0 - entropy / np.log(nonzero.size))
    return weights, entropy, float(np.clip(confidence, 0.0, 1.0))


def align_prototype_to_window(
    window: np.ndarray,
    window_mask: np.ndarray,
    prototype: np.ndarray,
    prototype_mask: np.ndarray,
    *,
    sample_rate: int,
    lambda_hyb: float,
    config: ShiftAlignmentConfig,
) -> ShiftAlignmentResult:
    cfg = validate_alignment_config(config)
    coarse_lags = generate_coarse_lags(sample_rate, cfg.shift_max_ms, cfg.coarse_step_ms)
    coarse_mse, coarse_cos, coarse_overlap = _precompute_shift_metrics(
        window,
        window_mask,
        prototype,
        prototype_mask,
        coarse_lags,
        minimum_overlap_ratio=cfg.minimum_overlap_ratio,
    )
    coarse_cost = _alignment_cost(coarse_mse, coarse_cos, coarse_overlap, lambda_hyb, cfg.overlap_penalty_weight)
    fine = _fine_lags(
        coarse_lags,
        coarse_cost,
        sample_rate=sample_rate,
        shift_max_ms=cfg.shift_max_ms,
        fine_radius_ms=cfg.fine_radius_ms,
        fine_step_samples=cfg.fine_step_samples,
        coarse_top_k=cfg.coarse_top_k,
    )
    mse, cosine, overlap = _precompute_shift_metrics(
        window,
        window_mask,
        prototype,
        prototype_mask,
        fine,
        minimum_overlap_ratio=cfg.minimum_overlap_ratio,
    )
    cost = _alignment_cost(mse, cosine, overlap, lambda_hyb, cfg.overlap_penalty_weight)
    weights, entropy, confidence = _weights_from_cost(cost, cfg.alignment_mode, cfg.alignment_temperature)
    valid = bool(np.any(weights > 0.0))
    if not valid:
        zero_idx = np.where(fine == 0)[0]
        if zero_idx.size:
            zero_overlap = masked_shift_mse(window, window_mask, prototype, prototype_mask, 0)[1]
            if zero_overlap > 0.0:
                weights[zero_idx[0]] = 1.0
                cost[zero_idx[0]] = 0.0
                overlap[zero_idx[0]] = zero_overlap
        valid = bool(np.any(weights > 0.0))
    if valid:
        best_idx = min(
            np.flatnonzero(weights > 0.0).tolist() if cfg.alignment_mode == "hard_bounded" else np.flatnonzero(np.isfinite(cost)).tolist(),
            key=lambda i: (float(cost[i]), abs(int(fine[i])), int(fine[i])),
        )
        best_lag = int(fine[best_idx])
        best_cost = float(cost[best_idx]) if np.isfinite(cost[best_idx]) else float("inf")
        best_overlap = float(overlap[best_idx])
    else:
        best_lag = 0
        best_cost = float("inf")
        best_overlap = 0.0
    return ShiftAlignmentResult(
        candidate_lags_samples=fine.astype(np.intp, copy=True),
        candidate_lags_ms=fine.astype(float) * 1000.0 / float(sample_rate),
        costs=cost.astype(float, copy=True),
        weights=weights.astype(float, copy=True),
        best_lag_samples=best_lag,
        best_lag_ms=float(best_lag) * 1000.0 / float(sample_rate),
        best_cost=best_cost,
        overlap_ratio=best_overlap,
        entropy=float(entropy),
        confidence=float(confidence if valid else 0.0),
        valid=valid,
    )


def _weighted_shifted(prototype: np.ndarray, mask: np.ndarray, alignment: ShiftAlignmentResult) -> list[tuple[float, np.ndarray, np.ndarray]]:
    out: list[tuple[float, np.ndarray, np.ndarray]] = []
    for weight, lag in zip(alignment.weights.tolist(), alignment.candidate_lags_samples.tolist(), strict=True):
        if weight <= 0.0:
            continue
        shifted, shifted_mask = shift_with_mask(prototype, mask, int(lag))
        out.append((float(weight), shifted, shifted_mask))
    return out


def _weighted_squared_error(
    window: np.ndarray,
    window_mask: np.ndarray,
    prototype: np.ndarray,
    prototype_mask: np.ndarray,
    alignment: ShiftAlignmentResult,
) -> np.ndarray:
    num = np.zeros_like(window, dtype=float)
    den = np.zeros_like(window, dtype=float)
    for weight, shifted, shifted_mask in _weighted_shifted(prototype, prototype_mask, alignment):
        valid = window_mask & shifted_mask
        values = np.zeros_like(window, dtype=float)
        values[valid] = (window[valid] - shifted[valid]) ** 2
        num[valid] += weight * values[valid]
        den[valid] += weight
    out = np.zeros_like(window, dtype=float)
    valid_den = den > 0.0
    out[valid_den] = num[valid_den] / den[valid_den]
    return out


def _weighted_cos_contribution(
    window: np.ndarray,
    window_mask: np.ndarray,
    prototype: np.ndarray,
    prototype_mask: np.ndarray,
    alignment: ShiftAlignmentResult,
) -> np.ndarray:
    num = np.zeros_like(window, dtype=float)
    den = np.zeros_like(window, dtype=float)
    for weight, shifted, shifted_mask in _weighted_shifted(prototype, prototype_mask, alignment):
        valid = window_mask & shifted_mask
        if not np.any(valid):
            continue
        w_norm = float(np.linalg.norm(window[valid]))
        p_norm = float(np.linalg.norm(shifted[valid]))
        contrib = np.zeros_like(window, dtype=float)
        if w_norm > EPS and p_norm > EPS:
            contrib[valid] = window[valid] * shifted[valid] / ((w_norm * p_norm) + EPS)
        elif w_norm <= EPS and p_norm <= EPS:
            contrib[valid] = 1.0 / float(np.count_nonzero(valid))
        num[valid] += weight * contrib[valid]
        den[valid] += weight
    out = np.zeros_like(window, dtype=float)
    valid_den = den > 0.0
    out[valid_den] = num[valid_den] / den[valid_den]
    return out


def explain_shift_robust_window(
    window: np.ndarray,
    window_mask: np.ndarray,
    p_target: np.ndarray,
    target_mask: np.ndarray,
    p_rival: np.ndarray,
    rival_mask: np.ndarray,
    *,
    sample_rate: int,
    lambda_grid: Sequence[float],
    config: ShiftAlignmentConfig,
) -> ShiftRobustWindowExplanation:
    phi_diff_by_lambda: dict[float, np.ndarray] = {}
    phi_cos_by_lambda: dict[float, np.ndarray] = {}
    phi_hyb_by_lambda: dict[float, np.ndarray] = {}
    evidence_by_lambda: dict[float, float] = {}
    target_alignment_by_lambda: dict[float, ShiftAlignmentResult] = {}
    rival_alignment_by_lambda: dict[float, ShiftAlignmentResult] = {}
    for lambda_hyb in lambda_grid:
        lambda_value = float(lambda_hyb)
        target_alignment = align_prototype_to_window(
            window,
            window_mask,
            p_target,
            target_mask,
            sample_rate=sample_rate,
            lambda_hyb=lambda_value,
            config=config,
        )
        rival_alignment = align_prototype_to_window(
            window,
            window_mask,
            p_rival,
            rival_mask,
            sample_rate=sample_rate,
            lambda_hyb=lambda_value,
            config=config,
        )
        if not target_alignment.valid or not rival_alignment.valid:
            phi_diff = np.zeros_like(window, dtype=float)
            phi_cos = np.zeros_like(window, dtype=float)
        else:
            target_sq = _weighted_squared_error(window, window_mask, p_target, target_mask, target_alignment)
            rival_sq = _weighted_squared_error(window, window_mask, p_rival, rival_mask, rival_alignment)
            phi_diff = rival_sq - target_sq
            target_cos = _weighted_cos_contribution(window, window_mask, p_target, target_mask, target_alignment)
            rival_cos = _weighted_cos_contribution(window, window_mask, p_rival, rival_mask, rival_alignment)
            phi_cos = target_cos - rival_cos
            phi_diff[~window_mask] = 0.0
            phi_cos[~window_mask] = 0.0
        diff_scale = float(np.sum(np.abs(phi_diff[window_mask])) + EPS)
        cos_scale = float(np.sum(np.abs(phi_cos[window_mask])) + EPS)
        phi_hyb = lambda_value * (phi_diff / diff_scale) + (1.0 - lambda_value) * (phi_cos / cos_scale)
        phi_hyb[~window_mask] = 0.0
        if not np.all(np.isfinite(phi_hyb)):
            phi_hyb = np.zeros_like(window, dtype=float)
        phi_diff_by_lambda[lambda_value] = phi_diff
        phi_cos_by_lambda[lambda_value] = phi_cos
        phi_hyb_by_lambda[lambda_value] = phi_hyb
        evidence_by_lambda[lambda_value] = float(np.sum(phi_hyb[window_mask]))
        target_alignment_by_lambda[lambda_value] = target_alignment
        rival_alignment_by_lambda[lambda_value] = rival_alignment
    return ShiftRobustWindowExplanation(
        phi_diff_by_lambda=phi_diff_by_lambda,
        phi_cos_by_lambda=phi_cos_by_lambda,
        phi_hyb_by_lambda=phi_hyb_by_lambda,
        evidence_by_lambda=evidence_by_lambda,
        target_alignment_by_lambda=target_alignment_by_lambda,
        rival_alignment_by_lambda=rival_alignment_by_lambda,
        mask=window_mask.astype(bool, copy=True),
    )


def _overlap_add(window_attrs: np.ndarray, masks: np.ndarray, starts: np.ndarray, n_samples: int) -> np.ndarray:
    values = np.zeros(n_samples, dtype=float)
    counts = np.zeros(n_samples, dtype=float)
    for attr, mask, start in zip(window_attrs, masks, starts, strict=True):
        valid_idx = np.where(mask)[0]
        sample_idx = valid_idx + int(start)
        keep = sample_idx < n_samples
        sample_idx = sample_idx[keep]
        valid_idx = valid_idx[keep]
        values[sample_idx] += attr[valid_idx]
        counts[sample_idx] += 1.0
    out = np.zeros(n_samples, dtype=float)
    covered = counts > 0.0
    out[covered] = values[covered] / counts[covered]
    return out


def _top_segments(
    waveform: np.ndarray,
    starts: np.ndarray,
    lengths: np.ndarray,
    evidence: np.ndarray,
    *,
    top_k: int,
    positive: bool,
    lambda_hyb: float,
    target_alignment: Sequence[ShiftAlignmentResult],
    rival_alignment: Sequence[ShiftAlignmentResult],
) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []
    order = np.argsort(evidence)
    if positive:
        order = order[::-1]
        keep = [idx for idx in order.tolist() if evidence[idx] > 0.0]
        role = "positive"
    else:
        keep = [idx for idx in order.tolist() if evidence[idx] < 0.0]
        role = "negative"
    rows: list[dict[str, Any]] = []
    for rank, idx in enumerate(keep[:top_k], start=1):
        start = int(starts[idx])
        end = min(start + int(lengths[idx]), waveform.shape[0])
        ta = target_alignment[idx]
        ra = rival_alignment[idx]
        rows.append(
            {
                "rank": int(rank),
                "window_index": int(idx),
                "role": role,
                "lambda_hyb": float(lambda_hyb),
                "sign": role,
                "start_sample": start,
                "end_sample": int(end),
                "start_time_sec": start / 1.0,
                "end_time_sec": end / 1.0,
                "evidence": float(evidence[idx]),
                "target_best_lag_samples": int(ta.best_lag_samples),
                "target_best_lag_ms": float(ta.best_lag_ms),
                "rival_best_lag_samples": int(ra.best_lag_samples),
                "rival_best_lag_ms": float(ra.best_lag_ms),
                "target_alignment_confidence": float(ta.confidence),
                "rival_alignment_confidence": float(ra.confidence),
                "target_overlap_ratio": float(ta.overlap_ratio),
                "rival_overlap_ratio": float(ra.overlap_ratio),
                "segment": waveform[start:end].astype(float, copy=True),
            }
        )
    return rows


def _call_generator(
    generator: Any,
    *,
    label: Any,
    lambda_hyb: float,
    segment: np.ndarray,
    sample_rate: int,
    role: str,
    metadata: Mapping[str, Any],
) -> tuple[np.ndarray | None, str]:
    if generator is None:
        return None, "skipped"
    generated = generator(label, float(lambda_hyb), segment.copy(), int(sample_rate), role, metadata)
    return _as_raw_waveform(f"generated_{role}", generated), "ok"


def _labels_without_target(labels: np.ndarray, target_label: Any) -> list[Any]:
    return [label for label in labels.tolist() if label != target_label]


def _select_rival_label(windows: np.ndarray, masks: np.ndarray, context: Any, target_label: Any) -> Any:
    candidates = _labels_without_target(context.prototype_labels, target_label)
    if not candidates:
        raise ValueError("rival_label is None, but no non-target raw prototypes exist")
    scores: list[float] = []
    for label in candidates:
        proto = np.asarray(context.prototypes[label], dtype=float)
        proto_mask = np.asarray(context.prototype_masks[label], dtype=bool)
        values = [masked_shift_mse(windows[i], masks[i], proto, proto_mask, 0)[0] for i in range(windows.shape[0])]
        finite = [value for value in values if np.isfinite(value)]
        scores.append(float(np.mean(finite)) if finite else float("inf"))
    return candidates[int(np.argmin(np.asarray(scores, dtype=float)))]


def explain_shift_robust_raw_waveform(
    fpde: Any,
    waveform: np.ndarray | Sequence[float] | Sequence[Sequence[float]],
    context: Any,
    *,
    sample_rate: int,
    target_label: Any,
    rival_label: Any | None = None,
    lambda_grid: Sequence[float] | None = None,
    top_k_segments: int = 1,
    generator: Any = None,
    device: str = "cpu",
    details: dict[str, Any] | None = None,
    config: ShiftAlignmentConfig | None = None,
) -> Any:
    cfg = validate_alignment_config(config or ShiftAlignmentConfig(alignment_mode="soft_bounded"))
    if cfg.alignment_mode == "none":
        raise ValueError("explain_shift_robust_raw_waveform is only for bounded alignment modes")
    if device == "cuda":
        import cupy as cp  # type: ignore[import-not-found]

        cp.cuda.runtime.getDeviceCount()
        resolved_device = "cuda_host_numpy"
    elif device == "auto":
        try:
            import cupy as cp  # type: ignore[import-not-found]

            cp.cuda.runtime.getDeviceCount()
            resolved_device = "cuda_host_numpy"
        except Exception:
            resolved_device = "cpu"
    else:
        resolved_device = "cpu"
    waveform_arr = _as_raw_waveform("waveform", waveform)
    if int(sample_rate) != int(context.target_sr):
        raise ValueError("shift-robust runner expects the waveform already resampled to context.target_sr")
    windows, masks, starts, lengths = _raw_windows(waveform_arr, int(context.segment_length), int(context.hop_length))
    if target_label not in context.prototypes:
        raise ValueError(f"no raw prototype found for target_label={target_label!r}")
    resolved_rival = _select_rival_label(windows, masks, context, target_label) if rival_label is None else rival_label
    if resolved_rival == target_label:
        raise ValueError("rival_label must differ from target_label")
    if resolved_rival not in context.prototypes:
        raise ValueError(f"no raw prototype found for rival_label={resolved_rival!r}")
    lambdas = tuple(float(value) for value in (LAMBDA_GRID if lambda_grid is None else lambda_grid))
    p_target = np.asarray(context.prototypes[target_label], dtype=float)
    target_mask = np.asarray(context.prototype_masks[target_label], dtype=bool)
    p_rival = np.asarray(context.prototypes[resolved_rival], dtype=float)
    rival_mask = np.asarray(context.prototype_masks[resolved_rival], dtype=bool)

    per_window = [
        explain_shift_robust_window(
            windows[i],
            masks[i],
            p_target,
            target_mask,
            p_rival,
            rival_mask,
            sample_rate=int(context.target_sr),
            lambda_grid=lambdas,
            config=cfg,
        )
        for i in range(windows.shape[0])
    ]
    lambda_results: dict[float, dict[str, Any]] = {}
    for lambda_value in lambdas:
        window_attrs = np.stack([item.phi_hyb_by_lambda[lambda_value] for item in per_window], axis=0)
        window_diff = np.stack([item.phi_diff_by_lambda[lambda_value] for item in per_window], axis=0)
        window_cos = np.stack([item.phi_cos_by_lambda[lambda_value] for item in per_window], axis=0)
        window_evidence = np.asarray([item.evidence_by_lambda[lambda_value] for item in per_window], dtype=float)
        phi = _overlap_add(window_attrs, masks, starts, waveform_arr.shape[0])
        if phi.shape != waveform_arr.shape:
            raise RuntimeError(f"raw waveform attribution shape mismatch: expected {waveform_arr.shape}, got {phi.shape}")
        if not np.all(np.isfinite(phi)):
            raise RuntimeError("shift-robust phi contains NaN or inf")
        target_alignments = [item.target_alignment_by_lambda[lambda_value] for item in per_window]
        rival_alignments = [item.rival_alignment_by_lambda[lambda_value] for item in per_window]
        top_positive = _top_segments(
            waveform_arr,
            starts,
            lengths,
            window_evidence,
            top_k=top_k_segments,
            positive=True,
            lambda_hyb=lambda_value,
            target_alignment=target_alignments,
            rival_alignment=rival_alignments,
        )
        top_negative = _top_segments(
            waveform_arr,
            starts,
            lengths,
            window_evidence,
            top_k=top_k_segments,
            positive=False,
            lambda_hyb=lambda_value,
            target_alignment=target_alignments,
            rival_alignment=rival_alignments,
        )
        for rows in (top_positive, top_negative):
            for row in rows:
                row["start_time_sec"] = float(row["start_sample"]) / float(context.target_sr)
                row["end_time_sec"] = float(row["end_sample"]) / float(context.target_sr)
        generation_status = {"target": "skipped", "rival": "skipped"}
        generated_target = None
        generated_rival = None
        if cfg.generation_scope in {"selected", "all"}:
            if top_positive:
                metadata = {key: value for key, value in top_positive[0].items() if key != "segment"}
                generated_target, generation_status["target"] = _call_generator(
                    generator,
                    label=target_label,
                    lambda_hyb=lambda_value,
                    segment=top_positive[0]["segment"],
                    sample_rate=int(context.target_sr),
                    role="target",
                    metadata=metadata,
                )
            if top_negative:
                metadata = {key: value for key, value in top_negative[0].items() if key != "segment"}
                generated_rival, generation_status["rival"] = _call_generator(
                    generator,
                    label=resolved_rival,
                    lambda_hyb=lambda_value,
                    segment=top_negative[0]["segment"],
                    sample_rate=int(context.target_sr),
                    role="rival",
                    metadata=metadata,
                )
        alignment_valid = np.asarray([ta.valid and ra.valid for ta, ra in zip(target_alignments, rival_alignments, strict=True)], dtype=bool)
        target_lags = np.asarray([ta.best_lag_ms for ta in target_alignments], dtype=float)
        rival_lags = np.asarray([ra.best_lag_ms for ra in rival_alignments], dtype=float)
        shift_max_samples = int(round(int(context.target_sr) * cfg.shift_max_ms / 1000.0))
        lambda_results[float(lambda_value)] = {
            "phi": phi.astype(float, copy=True),
            "window_attributions": window_attrs.astype(float, copy=True),
            "window_evidence": window_evidence,
            "window_starts": starts.copy(),
            "window_lengths": lengths.copy(),
            "window_masks": masks.copy(),
            "effective_window_masks": masks.copy(),
            "evidence": float(np.sum(window_evidence)),
            "top_positive_segments": top_positive,
            "top_negative_segments": top_negative,
            "generated_target": generated_target,
            "generated_rival": generated_rival,
            "generation_status": generation_status,
            "details": {
                "lambda_hyb": float(lambda_value),
                "alignment_mode": cfg.alignment_mode,
                "shift_max_ms": float(cfg.shift_max_ms),
                "minimum_overlap_ratio": float(cfg.minimum_overlap_ratio),
                "alignment_temperature": float(cfg.alignment_temperature),
                "diff_evidence": np.sum(window_diff, axis=1).astype(float).tolist(),
                "cos_evidence": np.sum(window_cos, axis=1).astype(float).tolist(),
                "diff_scale": (np.sum(np.abs(window_diff), axis=1) + EPS).astype(float).tolist(),
                "cos_scale": (np.sum(np.abs(window_cos), axis=1) + EPS).astype(float).tolist(),
                "target_lags_by_lambda": target_lags.tolist(),
                "rival_lags_by_lambda": rival_lags.tolist(),
                "target_alignment_confidence_by_lambda": [float(item.confidence) for item in target_alignments],
                "rival_alignment_confidence_by_lambda": [float(item.confidence) for item in rival_alignments],
                "alignment_valid_rate_by_lambda": float(np.mean(alignment_valid)) if alignment_valid.size else 0.0,
                "mean_abs_target_lag_ms": float(np.mean(np.abs(target_lags))) if target_lags.size else 0.0,
                "mean_abs_rival_lag_ms": float(np.mean(np.abs(rival_lags))) if rival_lags.size else 0.0,
                "max_abs_target_lag_ms": float(np.max(np.abs(target_lags))) if target_lags.size else 0.0,
                "max_abs_rival_lag_ms": float(np.max(np.abs(rival_lags))) if rival_lags.size else 0.0,
                "target_boundary_hit_rate": float(np.mean([abs(ta.best_lag_samples) >= shift_max_samples for ta in target_alignments])) if target_alignments else 0.0,
                "rival_boundary_hit_rate": float(np.mean([abs(ra.best_lag_samples) >= shift_max_samples for ra in rival_alignments])) if rival_alignments else 0.0,
                "alignment_confidence_mean": float(np.mean([0.5 * (ta.confidence + ra.confidence) for ta, ra in zip(target_alignments, rival_alignments, strict=True)])) if target_alignments else 0.0,
                "alignment_valid_rate": float(np.mean(alignment_valid)) if alignment_valid.size else 0.0,
                "device": resolved_device,
                "input_shape": tuple(waveform_arr.shape),
                "output_shape": tuple(phi.shape),
                "target_alignment": target_alignments,
                "rival_alignment": rival_alignments,
            },
        }
    best_lambda = max(lambdas, key=lambda value: float(lambda_results[float(value)]["evidence"])) if lambdas else None
    merged_details = {} if details is None else dict(details)
    merged_details.update(
        {
            "time_mode": "raw_waveform",
            "uses_acoustic_features": False,
            "uses_spectrogram": False,
            "uses_mfcc": False,
            "waveform_normalization": False,
            "target_sr": int(context.target_sr),
            "segment_length": int(context.segment_length),
            "hop_length": int(context.hop_length),
            "lambda_grid": tuple(float(value) for value in lambdas),
            "device": resolved_device,
            "target_label": target_label,
            "rival_label": resolved_rival,
            "alignment_mode": cfg.alignment_mode,
            "shift_max_ms": float(cfg.shift_max_ms),
            "minimum_overlap_ratio": float(cfg.minimum_overlap_ratio),
            "alignment_temperature": float(cfg.alignment_temperature),
        }
    )
    return fpde.RawWaveformFPDEExplanation(
        mode="shift_robust_raw_hyb",
        target_label=target_label,
        rival_label=resolved_rival,
        waveform=waveform_arr.astype(float, copy=True),
        sample_rate=int(context.target_sr),
        lambda_results=lambda_results,
        best_lambda=None if best_lambda is None else float(best_lambda),
        details=merged_details,
    )


def alignment_result_to_rows(
    *,
    dataset: str,
    fold: int,
    seed: int,
    sample_id: str,
    target_label: Any,
    rival_label: Any,
    lambda_hyb: float,
    alignment_mode: str,
    result: dict[str, Any],
) -> list[dict[str, object]]:
    details = dict(result.get("details", {}))
    target_items = details.get("target_alignment", [])
    rival_items = details.get("rival_alignment", [])
    if not target_items or not rival_items:
        return []
    rows: list[dict[str, object]] = []
    for idx, (ta, ra) in enumerate(zip(target_items, rival_items, strict=True)):
        rows.append(
            {
                "dataset": dataset,
                "fold": int(fold),
                "seed": int(seed),
                "sample_id": sample_id,
                "target_label": target_label,
                "rival_label": rival_label,
                "window_index": int(idx),
                "lambda_hyb": float(lambda_hyb),
                "alignment_mode": alignment_mode,
                "target_best_lag_samples": int(ta.best_lag_samples),
                "target_best_lag_ms": float(ta.best_lag_ms),
                "rival_best_lag_samples": int(ra.best_lag_samples),
                "rival_best_lag_ms": float(ra.best_lag_ms),
                "target_alignment_cost": float(ta.best_cost),
                "rival_alignment_cost": float(ra.best_cost),
                "target_overlap_ratio": float(ta.overlap_ratio),
                "rival_overlap_ratio": float(ra.overlap_ratio),
                "target_alignment_entropy": float(ta.entropy),
                "rival_alignment_entropy": float(ra.entropy),
                "target_alignment_confidence": float(ta.confidence),
                "rival_alignment_confidence": float(ra.confidence),
                "target_alignment_valid": bool(ta.valid),
                "rival_alignment_valid": bool(ra.valid),
            }
        )
    return rows
