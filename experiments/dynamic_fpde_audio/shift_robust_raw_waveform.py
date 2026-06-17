"""Shift-robust Raw-Waveform Dynamic-FPDE helpers.

The implementation keeps raw samples as the explanation domain and supports a
real NumPy/CuPy backend boundary for bounded local lag alignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.dynamic_fpde_audio.raw_waveform_context import _as_raw_waveform, _raw_windows


LAMBDA_GRID = tuple(i / 10.0 for i in range(11))
EPS = 1e-12


@dataclass(frozen=True)
class ArrayBackend:
    name: str
    xp: Any
    is_cuda: bool
    cuda_device_name: str = ""
    cupy_version: str = ""
    cuda_runtime_version: str = ""
    backend_fallback_used: bool = False


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
    generation_selected_lambdas: tuple[float, ...] = (0.0, 0.5, 1.0)
    alignment_lag_block_size: int = 128
    rival_selection_mode: str = "shift_robust_neutral"
    rival_selection_lambda: float = 0.5
    alignment_details_format: str = "parquet"


@dataclass(frozen=True)
class PrecomputedLagMetrics:
    lags_samples: Any
    shifted_prototypes: Any
    shifted_masks: Any
    mse: Any
    cosine_distance: Any
    overlap_ratio: Any


@dataclass(frozen=True)
class LambdaAlignmentBatch:
    lambda_grid: Any
    costs: Any
    weights: Any
    best_indices: Any
    entropy: Any
    confidence: Any
    valid: Any
    fallback_used: Any
    fallback_reason: tuple[str, ...]


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
    fallback_used: bool = False
    fallback_reason: str = ""


@dataclass(frozen=True)
class ShiftRobustWindowExplanation:
    phi_diff_by_lambda: dict[float, np.ndarray]
    phi_cos_by_lambda: dict[float, np.ndarray]
    phi_hyb_by_lambda: dict[float, np.ndarray]
    evidence_by_lambda: dict[float, float]
    target_alignment_by_lambda: dict[float, ShiftAlignmentResult]
    rival_alignment_by_lambda: dict[float, ShiftAlignmentResult]
    effective_mask_by_lambda: dict[float, np.ndarray]
    mask: np.ndarray
    timings: dict[str, float]


def resolve_backend(device: str) -> ArrayBackend:
    if device == "cpu":
        return ArrayBackend(name="numpy_cpu", xp=np, is_cuda=False)
    if device not in {"cuda", "auto"}:
        raise ValueError("device must be cpu, cuda, or auto")
    try:
        import cupy as cp  # type: ignore[import-not-found]

        cp.cuda.runtime.getDeviceCount()
        device_id = cp.cuda.runtime.getDevice()
        props = cp.cuda.runtime.getDeviceProperties(device_id)
        raw_name = props.get("name", b"") if isinstance(props, dict) else b""
        if isinstance(raw_name, bytes):
            cuda_device_name = raw_name.decode("utf-8", errors="replace")
        else:
            cuda_device_name = str(raw_name)
        return ArrayBackend(
            name="cupy_cuda",
            xp=cp,
            is_cuda=True,
            cuda_device_name=cuda_device_name,
            cupy_version=getattr(cp, "__version__", ""),
            cuda_runtime_version=str(cp.cuda.runtime.runtimeGetVersion()),
        )
    except Exception as exc:
        if device == "cuda":
            raise RuntimeError("CUDA/CuPy backend requested but unavailable") from exc
        return ArrayBackend(name="numpy_cpu", xp=np, is_cuda=False, backend_fallback_used=True)


def asnumpy(value: Any, backend: ArrayBackend) -> np.ndarray:
    return backend.xp.asnumpy(value) if backend.is_cuda else np.asarray(value)


def validate_alignment_config(config: ShiftAlignmentConfig) -> ShiftAlignmentConfig:
    if config.alignment_mode not in {"none", "hard_bounded", "soft_bounded"}:
        raise ValueError("alignment_mode must be none, hard_bounded, or soft_bounded")
    if config.generation_scope not in {"none", "selected", "all"}:
        raise ValueError("generation_scope must be none, selected, or all")
    if config.rival_selection_mode not in {"zero_lag_mse", "shift_robust_neutral"}:
        raise ValueError("rival_selection_mode must be zero_lag_mse or shift_robust_neutral")
    if config.alignment_details_format not in {"parquet", "csv"}:
        raise ValueError("alignment_details_format must be parquet or csv")
    for name in ("shift_max_ms", "coarse_step_ms", "fine_radius_ms", "minimum_overlap_ratio", "rival_selection_lambda"):
        value = float(getattr(config, name))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if config.coarse_step_ms <= 0.0:
        raise ValueError("coarse_step_ms must be positive")
    if int(config.fine_step_samples) <= 0:
        raise ValueError("fine_step_samples must be positive")
    if int(config.coarse_top_k) <= 0:
        raise ValueError("coarse_top_k must be positive")
    if int(config.alignment_lag_block_size) <= 0:
        raise ValueError("alignment_lag_block_size must be positive")
    if not (0.0 <= float(config.minimum_overlap_ratio) <= 1.0):
        raise ValueError("minimum_overlap_ratio must be in [0, 1]")
    if not (0.0 <= float(config.rival_selection_lambda) <= 1.0):
        raise ValueError("rival_selection_lambda must be in [0, 1]")
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


def build_shifted_batch(
    prototype: Any,
    prototype_mask: Any,
    lags: Any,
    *,
    backend: ArrayBackend,
) -> tuple[Any, Any]:
    xp = backend.xp
    p = xp.asarray(prototype, dtype=float)
    pm = xp.asarray(prototype_mask, dtype=bool)
    lag_arr = xp.asarray(lags, dtype=xp.int64)
    length = int(p.shape[0])
    positions = xp.arange(length, dtype=xp.int64)[None, :]
    source = positions - lag_arr[:, None]
    in_range = (source >= 0) & (source < length)
    source = xp.clip(source, 0, max(0, length - 1))
    shifted = xp.where(in_range, p[source], 0.0)
    shifted_mask = xp.where(in_range, pm[source], False)
    return shifted, shifted_mask


def precompute_lag_metrics(
    window: Any,
    window_mask: Any,
    prototype: Any,
    prototype_mask: Any,
    lags: Any,
    *,
    backend: ArrayBackend,
    lag_block_size: int = 128,
) -> PrecomputedLagMetrics:
    xp = backend.xp
    lag_arr = xp.asarray(lags, dtype=xp.int64)
    w = xp.asarray(window, dtype=float)
    wm = xp.asarray(window_mask, dtype=bool)
    shifted_parts = []
    mask_parts = []
    mse_parts = []
    cosine_parts = []
    overlap_parts = []
    for start in range(0, int(lag_arr.shape[0]), int(lag_block_size)):
        block_lags = lag_arr[start : start + int(lag_block_size)]
        shifted, shifted_mask = build_shifted_batch(prototype, prototype_mask, block_lags, backend=backend)
        valid = wm[None, :] & shifted_mask
        valid_f = valid.astype(float)
        valid_count = xp.sum(valid_f, axis=1)
        overlap = valid_count / float(w.shape[0])
        diff = xp.where(valid, w[None, :] - shifted, 0.0)
        mse = xp.sum(diff * diff, axis=1) / xp.maximum(valid_count, 1.0)
        mse = xp.where(valid_count > 0, mse, xp.inf)
        dot = xp.sum(xp.where(valid, w[None, :] * shifted, 0.0), axis=1)
        w_norm = xp.sqrt(xp.sum(xp.where(valid, w[None, :] * w[None, :], 0.0), axis=1))
        p_norm = xp.sqrt(xp.sum(xp.where(valid, shifted * shifted, 0.0), axis=1))
        both_zero = (w_norm <= EPS) & (p_norm <= EPS) & (valid_count > 0)
        one_zero = ((w_norm <= EPS) ^ (p_norm <= EPS)) & (valid_count > 0)
        cos = dot / (w_norm * p_norm + EPS)
        cos = xp.clip(cos, -1.0, 1.0)
        cos = xp.where(both_zero, 1.0, xp.where(one_zero, 0.0, cos))
        cosine_distance = xp.where(valid_count > 0, (1.0 - cos) / 2.0, xp.inf)
        shifted_parts.append(shifted)
        mask_parts.append(shifted_mask)
        mse_parts.append(mse)
        cosine_parts.append(cosine_distance)
        overlap_parts.append(overlap)
    return PrecomputedLagMetrics(
        lags_samples=lag_arr,
        shifted_prototypes=xp.concatenate(shifted_parts, axis=0),
        shifted_masks=xp.concatenate(mask_parts, axis=0),
        mse=xp.concatenate(mse_parts, axis=0),
        cosine_distance=xp.concatenate(cosine_parts, axis=0),
        overlap_ratio=xp.concatenate(overlap_parts, axis=0),
    )


def masked_shift_mse(
    window: np.ndarray,
    window_mask: np.ndarray,
    prototype: np.ndarray,
    prototype_mask: np.ndarray,
    lag_samples: int,
) -> tuple[float, float]:
    shifted, shifted_mask = shift_with_mask(prototype, prototype_mask, lag_samples)
    valid = np.asarray(window_mask, dtype=bool) & shifted_mask
    overlap_ratio = float(np.count_nonzero(valid) / max(1, np.asarray(window).shape[0]))
    if not np.any(valid):
        return float("inf"), overlap_ratio
    diff = np.asarray(window, dtype=float)[valid] - shifted[valid]
    return float(np.mean(diff * diff)), overlap_ratio


def masked_shift_cosine(
    window: np.ndarray,
    window_mask: np.ndarray,
    prototype: np.ndarray,
    prototype_mask: np.ndarray,
    lag_samples: int,
    eps: float = EPS,
) -> tuple[float, float]:
    shifted, shifted_mask = shift_with_mask(prototype, prototype_mask, lag_samples)
    valid = np.asarray(window_mask, dtype=bool) & shifted_mask
    overlap_ratio = float(np.count_nonzero(valid) / max(1, np.asarray(window).shape[0]))
    if not np.any(valid):
        return 1.0, overlap_ratio
    wv = np.asarray(window, dtype=float)[valid]
    pv = shifted[valid]
    w_norm = float(np.linalg.norm(wv))
    p_norm = float(np.linalg.norm(pv))
    if w_norm <= eps and p_norm <= eps:
        cos = 1.0
    elif w_norm <= eps or p_norm <= eps:
        cos = 0.0
    else:
        cos = float(np.dot(wv, pv) / ((w_norm * p_norm) + eps))
    return float((1.0 - np.clip(cos, -1.0, 1.0)) / 2.0), overlap_ratio


def generate_coarse_lags(sample_rate: int, shift_max_ms: float, coarse_step_ms: float) -> np.ndarray:
    sr = int(sample_rate)
    if sr <= 0:
        raise ValueError("sample_rate must be positive")
    shift_max_samples = int(round(sr * float(shift_max_ms) / 1000.0))
    coarse_step_samples = max(1, int(round(sr * float(coarse_step_ms) / 1000.0)))
    values = list(range(-shift_max_samples, shift_max_samples + 1, coarse_step_samples))
    values.extend([-shift_max_samples, 0, shift_max_samples])
    return np.asarray(sorted(set(int(v) for v in values)), dtype=np.intp)


def _cost_matrix(
    metrics: PrecomputedLagMetrics,
    lambda_grid: Any,
    config: ShiftAlignmentConfig,
    backend: ArrayBackend,
    *,
    enforce_minimum_overlap: bool = True,
) -> Any:
    xp = backend.xp
    lambdas = xp.asarray(lambda_grid, dtype=float)[:, None]
    mse = metrics.mse[None, :]
    cosine = metrics.cosine_distance[None, :]
    overlap = metrics.overlap_ratio[None, :]
    finite_mse = metrics.mse[xp.isfinite(metrics.mse)]
    if int(finite_mse.shape[0]) == 0:
        mse_scale = xp.asarray(1.0)
    else:
        mse_scale = xp.median(finite_mse)
        mse_scale = xp.where((~xp.isfinite(mse_scale)) | (mse_scale <= EPS), 1.0, mse_scale)
    mse_term = xp.where(lambdas == 0.0, 0.0, mse / (mse_scale + EPS))
    cos_term = xp.where(lambdas == 1.0, 0.0, cosine)
    cost = lambdas * mse_term + (1.0 - lambdas) * cos_term + float(config.overlap_penalty_weight) * (1.0 - overlap)
    invalid = (~xp.isfinite(mse)) | (~xp.isfinite(cosine))
    if enforce_minimum_overlap:
        invalid = invalid | (overlap < float(config.minimum_overlap_ratio))
    return xp.where(invalid, xp.inf, cost)


def _fine_lags_from_costs(
    coarse_lags: np.ndarray,
    coarse_costs: np.ndarray,
    *,
    sample_rate: int,
    config: ShiftAlignmentConfig,
) -> np.ndarray:
    shift_max_samples = int(round(int(sample_rate) * float(config.shift_max_ms) / 1000.0))
    radius = int(round(int(sample_rate) * float(config.fine_radius_ms) / 1000.0))
    step = max(1, int(config.fine_step_samples))
    values: set[int] = {-shift_max_samples, 0, shift_max_samples}
    for row in coarse_costs:
        finite_idx = np.flatnonzero(np.isfinite(row))
        if finite_idx.size == 0:
            centers = np.asarray([0], dtype=np.intp)
        else:
            order = sorted(finite_idx.tolist(), key=lambda i: (float(row[i]), abs(int(coarse_lags[i])), int(coarse_lags[i])))
            centers = coarse_lags[order[: int(config.coarse_top_k)]]
        for center in centers.tolist():
            values.update(range(int(center) - radius, int(center) + radius + 1, step))
    return np.asarray(sorted(v for v in values if -shift_max_samples <= v <= shift_max_samples), dtype=np.intp)


def _hard_best_index(cost_row: np.ndarray, lags: np.ndarray) -> int:
    finite_idx = np.flatnonzero(np.isfinite(cost_row))
    if finite_idx.size == 0:
        return -1
    return int(min(finite_idx.tolist(), key=lambda i: (float(cost_row[i]), abs(int(lags[i])), int(lags[i]))))


def _alignment_batch(
    metrics: PrecomputedLagMetrics,
    lambda_grid: Sequence[float],
    *,
    config: ShiftAlignmentConfig,
    backend: ArrayBackend,
    lag0_metrics: PrecomputedLagMetrics | None = None,
) -> LambdaAlignmentBatch:
    xp = backend.xp
    costs = _cost_matrix(metrics, lambda_grid, config, backend)
    costs_np = asnumpy(costs, backend)
    lags_np = asnumpy(metrics.lags_samples, backend).astype(int, copy=False)
    weights_np = np.zeros_like(costs_np, dtype=float)
    best_indices = np.full(costs_np.shape[0], -1, dtype=np.intp)
    entropy = np.zeros(costs_np.shape[0], dtype=float)
    confidence = np.zeros(costs_np.shape[0], dtype=float)
    valid = np.zeros(costs_np.shape[0], dtype=bool)
    fallback_used = np.zeros(costs_np.shape[0], dtype=bool)
    fallback_reason = [""] * costs_np.shape[0]
    lag0_index = int(np.where(lags_np == 0)[0][0]) if np.any(lags_np == 0) else -1
    lag0_costs_np = None
    lag0_overlap_np = None
    if lag0_metrics is not None:
        lag0_costs_np = asnumpy(_cost_matrix(lag0_metrics, lambda_grid, config, backend, enforce_minimum_overlap=False), backend)[:, 0]
        lag0_overlap_np = asnumpy(lag0_metrics.overlap_ratio, backend)[0]
    for row_idx, row in enumerate(costs_np):
        finite_idx = np.flatnonzero(np.isfinite(row))
        if finite_idx.size == 0:
            fallback_used[row_idx] = True
            fallback_reason[row_idx] = "minimum_overlap_not_met"
            if lag0_index >= 0 and lag0_overlap_np is not None and lag0_overlap_np > 0.0:
                weights_np[row_idx, lag0_index] = 1.0
                best_indices[row_idx] = lag0_index
                costs_np[row_idx, lag0_index] = float(lag0_costs_np[row_idx])
            continue
        valid[row_idx] = True
        if config.alignment_mode == "hard_bounded" or finite_idx.size == 1:
            best = _hard_best_index(row, lags_np)
            weights_np[row_idx, best] = 1.0
            best_indices[row_idx] = best
            confidence[row_idx] = 1.0
            continue
        stable = -row[finite_idx] / float(config.alignment_temperature)
        stable = stable - float(np.max(stable))
        exp_values = np.exp(stable)
        total = float(np.sum(exp_values))
        if not np.isfinite(total) or total <= 0.0:
            best = _hard_best_index(row, lags_np)
            weights_np[row_idx, best] = 1.0
            best_indices[row_idx] = best
            confidence[row_idx] = 1.0
            continue
        weights_np[row_idx, finite_idx] = exp_values / total
        best_indices[row_idx] = _hard_best_index(row, lags_np)
        nonzero = weights_np[row_idx, weights_np[row_idx] > 0.0]
        entropy[row_idx] = float(-np.sum(nonzero * np.log(nonzero + EPS)))
        confidence[row_idx] = 1.0 if nonzero.size <= 1 else float(np.clip(1.0 - entropy[row_idx] / np.log(nonzero.size), 0.0, 1.0))
    return LambdaAlignmentBatch(
        lambda_grid=xp.asarray(lambda_grid, dtype=float),
        costs=xp.asarray(costs_np, dtype=float),
        weights=xp.asarray(weights_np, dtype=float),
        best_indices=xp.asarray(best_indices, dtype=xp.int64),
        entropy=xp.asarray(entropy, dtype=float),
        confidence=xp.asarray(confidence, dtype=float),
        valid=xp.asarray(valid, dtype=bool),
        fallback_used=xp.asarray(fallback_used, dtype=bool),
        fallback_reason=tuple(fallback_reason),
    )


def precompute_alignment_for_lambdas(
    window: Any,
    window_mask: Any,
    prototype: Any,
    prototype_mask: Any,
    *,
    sample_rate: int,
    lambda_grid: Sequence[float],
    config: ShiftAlignmentConfig,
    backend: ArrayBackend,
) -> tuple[PrecomputedLagMetrics, LambdaAlignmentBatch, dict[str, float]]:
    timings: dict[str, float] = {}
    start = perf_counter()
    coarse_lags = generate_coarse_lags(sample_rate, config.shift_max_ms, config.coarse_step_ms)
    timings["lag_batch_build_runtime_sec"] = perf_counter() - start
    start = perf_counter()
    coarse_metrics = precompute_lag_metrics(
        window,
        window_mask,
        prototype,
        prototype_mask,
        coarse_lags,
        backend=backend,
        lag_block_size=config.alignment_lag_block_size,
    )
    timings["coarse_metric_runtime_sec"] = perf_counter() - start
    start = perf_counter()
    coarse_costs = _cost_matrix(coarse_metrics, lambda_grid, config, backend)
    timings["lambda_cost_runtime_sec"] = perf_counter() - start
    fine_lags = _fine_lags_from_costs(asnumpy(coarse_metrics.lags_samples, backend).astype(int), asnumpy(coarse_costs, backend), sample_rate=sample_rate, config=config)
    start = perf_counter()
    fine_metrics = precompute_lag_metrics(
        window,
        window_mask,
        prototype,
        prototype_mask,
        fine_lags,
        backend=backend,
        lag_block_size=config.alignment_lag_block_size,
    )
    fine_costs = _cost_matrix(fine_metrics, lambda_grid, config, backend)
    needs_lag0_fallback = bool(np.any(np.all(~np.isfinite(asnumpy(fine_costs, backend)), axis=1)))
    lag0_metrics = (
        precompute_lag_metrics(
            window,
            window_mask,
            prototype,
            prototype_mask,
            np.asarray([0], dtype=np.intp),
            backend=backend,
            lag_block_size=1,
        )
        if needs_lag0_fallback
        else None
    )
    timings["fine_metric_runtime_sec"] = perf_counter() - start
    start = perf_counter()
    batch = _alignment_batch(fine_metrics, lambda_grid, config=config, backend=backend, lag0_metrics=lag0_metrics)
    timings["alignment_weight_runtime_sec"] = perf_counter() - start
    return fine_metrics, batch, timings


def _alignment_results(metrics: PrecomputedLagMetrics, batch: LambdaAlignmentBatch, sample_rate: int, backend: ArrayBackend) -> dict[float, ShiftAlignmentResult]:
    lags = asnumpy(metrics.lags_samples, backend).astype(int)
    costs = asnumpy(batch.costs, backend)
    weights = asnumpy(batch.weights, backend)
    best_indices = asnumpy(batch.best_indices, backend).astype(int)
    entropy = asnumpy(batch.entropy, backend)
    confidence = asnumpy(batch.confidence, backend)
    valid = asnumpy(batch.valid, backend).astype(bool)
    fallback_used = asnumpy(batch.fallback_used, backend).astype(bool)
    overlap = asnumpy(metrics.overlap_ratio, backend)
    out: dict[float, ShiftAlignmentResult] = {}
    for row_idx, lambda_value in enumerate(asnumpy(batch.lambda_grid, backend).astype(float).tolist()):
        best = int(best_indices[row_idx])
        if best >= 0:
            best_lag = int(lags[best])
            best_cost = float(costs[row_idx, best])
            best_overlap = float(overlap[best])
        else:
            best_lag = 0
            best_cost = float("inf")
            best_overlap = 0.0
        out[float(lambda_value)] = ShiftAlignmentResult(
            candidate_lags_samples=lags.copy(),
            candidate_lags_ms=lags.astype(float) * 1000.0 / float(sample_rate),
            costs=costs[row_idx].astype(float, copy=True),
            weights=weights[row_idx].astype(float, copy=True),
            best_lag_samples=best_lag,
            best_lag_ms=float(best_lag) * 1000.0 / float(sample_rate),
            best_cost=best_cost,
            overlap_ratio=best_overlap,
            entropy=float(entropy[row_idx]),
            confidence=float(confidence[row_idx]),
            valid=bool(valid[row_idx]),
            fallback_used=bool(fallback_used[row_idx]),
            fallback_reason=batch.fallback_reason[row_idx],
        )
    return out


def align_prototype_to_window(
    window: np.ndarray,
    window_mask: np.ndarray,
    prototype: np.ndarray,
    prototype_mask: np.ndarray,
    *,
    sample_rate: int,
    lambda_hyb: float,
    config: ShiftAlignmentConfig,
    backend: ArrayBackend | None = None,
) -> ShiftAlignmentResult:
    resolved = backend or resolve_backend("cpu")
    metrics, batch, _ = precompute_alignment_for_lambdas(
        window,
        window_mask,
        prototype,
        prototype_mask,
        sample_rate=sample_rate,
        lambda_grid=(float(lambda_hyb),),
        config=config,
        backend=resolved,
    )
    return _alignment_results(metrics, batch, sample_rate, resolved)[float(lambda_hyb)]


def _weighted_squared_error_backend(window: Any, window_mask: Any, metrics: PrecomputedLagMetrics, weights: Any, backend: ArrayBackend) -> tuple[Any, Any]:
    xp = backend.xp
    valid = xp.asarray(window_mask, dtype=bool)[None, :] & metrics.shifted_masks
    valid_f = valid.astype(float)
    weighted_valid = weights[:, None] * valid_f
    den = xp.sum(weighted_valid, axis=0)
    diff = xp.where(valid, xp.asarray(window, dtype=float)[None, :] - metrics.shifted_prototypes, 0.0)
    num = xp.sum(weights[:, None] * diff * diff * valid_f, axis=0)
    out = xp.where(den > 0, num / xp.maximum(den, EPS), 0.0)
    return out, den > 0


def _weighted_cos_backend(window: Any, window_mask: Any, metrics: PrecomputedLagMetrics, weights: Any, backend: ArrayBackend) -> tuple[Any, Any]:
    xp = backend.xp
    w = xp.asarray(window, dtype=float)
    valid = xp.asarray(window_mask, dtype=bool)[None, :] & metrics.shifted_masks
    valid_f = valid.astype(float)
    w_norm = xp.sqrt(xp.sum(xp.where(valid, w[None, :] * w[None, :], 0.0), axis=1))
    p_norm = xp.sqrt(xp.sum(xp.where(valid, metrics.shifted_prototypes * metrics.shifted_prototypes, 0.0), axis=1))
    valid_count = xp.sum(valid_f, axis=1)
    contrib = xp.where(valid, w[None, :] * metrics.shifted_prototypes / ((w_norm * p_norm)[:, None] + EPS), 0.0)
    both_zero = (w_norm <= EPS) & (p_norm <= EPS) & (valid_count > 0)
    contrib = xp.where(both_zero[:, None] & valid, 1.0 / xp.maximum(valid_count[:, None], 1.0), contrib)
    den = xp.sum(weights[:, None] * valid_f, axis=0)
    num = xp.sum(weights[:, None] * contrib * valid_f, axis=0)
    out = xp.where(den > 0, num / xp.maximum(den, EPS), 0.0)
    return out, den > 0


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
    backend: ArrayBackend | None = None,
) -> ShiftRobustWindowExplanation:
    resolved = backend or resolve_backend("cpu")
    cfg = validate_alignment_config(config)
    target_metrics, target_batch, target_timings = precompute_alignment_for_lambdas(
        window, window_mask, p_target, target_mask, sample_rate=sample_rate, lambda_grid=lambda_grid, config=cfg, backend=resolved
    )
    rival_metrics, rival_batch, rival_timings = precompute_alignment_for_lambdas(
        window, window_mask, p_rival, rival_mask, sample_rate=sample_rate, lambda_grid=lambda_grid, config=cfg, backend=resolved
    )
    timings: dict[str, float] = {}
    for key in set(target_timings) | set(rival_timings):
        timings[key] = float(target_timings.get(key, 0.0)) + float(rival_timings.get(key, 0.0))
    xp = resolved.xp
    phi_diff_by_lambda: dict[float, np.ndarray] = {}
    phi_cos_by_lambda: dict[float, np.ndarray] = {}
    phi_hyb_by_lambda: dict[float, np.ndarray] = {}
    evidence_by_lambda: dict[float, float] = {}
    effective_mask_by_lambda: dict[float, np.ndarray] = {}
    target_results = _alignment_results(target_metrics, target_batch, sample_rate, resolved)
    rival_results = _alignment_results(rival_metrics, rival_batch, sample_rate, resolved)
    window_mask_x = xp.asarray(window_mask, dtype=bool)
    for row_idx, lambda_value in enumerate(lambda_grid):
        target_weights = target_batch.weights[row_idx]
        rival_weights = rival_batch.weights[row_idx]
        start = perf_counter()
        target_sq, target_eff = _weighted_squared_error_backend(window, window_mask, target_metrics, target_weights, resolved)
        rival_sq, rival_eff = _weighted_squared_error_backend(window, window_mask, rival_metrics, rival_weights, resolved)
        timings["aligned_diff_runtime_sec"] = timings.get("aligned_diff_runtime_sec", 0.0) + perf_counter() - start
        start = perf_counter()
        target_cos, target_cos_eff = _weighted_cos_backend(window, window_mask, target_metrics, target_weights, resolved)
        rival_cos, rival_cos_eff = _weighted_cos_backend(window, window_mask, rival_metrics, rival_weights, resolved)
        timings["aligned_cos_runtime_sec"] = timings.get("aligned_cos_runtime_sec", 0.0) + perf_counter() - start
        effective = window_mask_x & (target_eff | rival_eff | target_cos_eff | rival_cos_eff)
        both_valid = bool(target_results[float(lambda_value)].valid and rival_results[float(lambda_value)].valid)
        if not both_valid and not (target_results[float(lambda_value)].fallback_used or rival_results[float(lambda_value)].fallback_used):
            phi_diff_x = xp.zeros_like(xp.asarray(window, dtype=float))
            phi_cos_x = xp.zeros_like(phi_diff_x)
        else:
            phi_diff_x = xp.where(effective, rival_sq - target_sq, 0.0)
            phi_cos_x = xp.where(effective, target_cos - rival_cos, 0.0)
        diff_scale = xp.sum(xp.abs(phi_diff_x[effective])) + EPS
        cos_scale = xp.sum(xp.abs(phi_cos_x[effective])) + EPS
        lambda_float = float(lambda_value)
        phi_hyb_x = xp.where(effective, lambda_float * (phi_diff_x / diff_scale) + (1.0 - lambda_float) * (phi_cos_x / cos_scale), 0.0)
        phi_hyb = asnumpy(phi_hyb_x, resolved).astype(float, copy=True)
        phi_diff = asnumpy(phi_diff_x, resolved).astype(float, copy=True)
        phi_cos = asnumpy(phi_cos_x, resolved).astype(float, copy=True)
        eff_np = asnumpy(effective, resolved).astype(bool, copy=True)
        phi_diff_by_lambda[lambda_float] = phi_diff
        phi_cos_by_lambda[lambda_float] = phi_cos
        phi_hyb_by_lambda[lambda_float] = np.where(np.isfinite(phi_hyb), phi_hyb, 0.0)
        evidence_by_lambda[lambda_float] = float(np.sum(phi_hyb_by_lambda[lambda_float][eff_np]))
        effective_mask_by_lambda[lambda_float] = eff_np
    return ShiftRobustWindowExplanation(
        phi_diff_by_lambda=phi_diff_by_lambda,
        phi_cos_by_lambda=phi_cos_by_lambda,
        phi_hyb_by_lambda=phi_hyb_by_lambda,
        evidence_by_lambda=evidence_by_lambda,
        target_alignment_by_lambda=target_results,
        rival_alignment_by_lambda=rival_results,
        effective_mask_by_lambda=effective_mask_by_lambda,
        mask=np.asarray(window_mask, dtype=bool).copy(),
        timings=timings,
    )


def _overlap_add(window_attrs: np.ndarray, masks: np.ndarray, starts: np.ndarray, n_samples: int) -> np.ndarray:
    values = np.zeros(n_samples, dtype=float)
    counts = np.zeros(n_samples, dtype=float)
    for attr, mask, start in zip(window_attrs, masks, starts, strict=True):
        valid_idx = np.where(mask)[0]
        sample_idx = valid_idx + int(start)
        keep = sample_idx < n_samples
        values[sample_idx[keep]] += attr[valid_idx[keep]]
        counts[sample_idx[keep]] += 1.0
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
    sample_rate: int,
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
    rows = []
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
                "start_time_sec": float(start) / float(sample_rate),
                "end_time_sec": float(end) / float(sample_rate),
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


def _select_rival_label(
    windows: np.ndarray,
    masks: np.ndarray,
    context: Any,
    target_label: Any,
    *,
    config: ShiftAlignmentConfig,
    backend: ArrayBackend,
) -> tuple[Any, dict[str, float | str]]:
    candidates = _labels_without_target(context.prototype_labels, target_label)
    if not candidates:
        raise ValueError("rival_label is None, but no non-target raw prototypes exist")
    scores = []
    for label in candidates:
        proto = np.asarray(context.prototypes[label], dtype=float)
        proto_mask = np.asarray(context.prototype_masks[label], dtype=bool)
        values = []
        for i in range(windows.shape[0]):
            if config.rival_selection_mode == "zero_lag_mse":
                value = masked_shift_mse(windows[i], masks[i], proto, proto_mask, 0)[0]
            else:
                metrics, batch, _ = precompute_alignment_for_lambdas(
                    windows[i],
                    masks[i],
                    proto,
                    proto_mask,
                    sample_rate=int(context.target_sr),
                    lambda_grid=(float(config.rival_selection_lambda),),
                    config=config,
                    backend=backend,
                )
                costs = asnumpy(batch.costs, backend)[0]
                finite = costs[np.isfinite(costs)]
                value = float(np.min(finite)) if finite.size else float("inf")
            if np.isfinite(value):
                values.append(float(value))
        scores.append(float(np.mean(values)) if values else float("inf"))
    order = np.argsort(np.asarray(scores, dtype=float))
    best_i = int(order[0])
    second = float(scores[int(order[1])]) if len(order) > 1 else float("inf")
    best = float(scores[best_i])
    return candidates[best_i], {
        "rival_selection_mode": config.rival_selection_mode,
        "rival_selection_lambda": float(config.rival_selection_lambda),
        "rival_selection_cost": best,
        "rival_selection_second_best_cost": second,
        "rival_selection_margin": second - best if np.isfinite(second) and np.isfinite(best) else "",
    }


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
    input_mask: np.ndarray | None = None,
) -> Any:
    cfg = validate_alignment_config(config or ShiftAlignmentConfig(alignment_mode="soft_bounded"))
    if cfg.alignment_mode == "none":
        raise ValueError("explain_shift_robust_raw_waveform is only for bounded alignment modes")
    backend = resolve_backend(device)
    waveform_arr = _as_raw_waveform("waveform", waveform)
    if int(sample_rate) != int(context.target_sr):
        raise ValueError("shift-robust runner expects the waveform already resampled to context.target_sr")
    if input_mask is None:
        sample_mask = np.ones(waveform_arr.shape, dtype=bool)
    else:
        sample_mask = np.asarray(input_mask, dtype=bool)
        if sample_mask.shape != waveform_arr.shape:
            raise ValueError("input_mask must match waveform shape")
    windows, base_masks, starts, lengths = _raw_windows(waveform_arr, int(context.segment_length), int(context.hop_length))
    mask_windows, _, _, _ = _raw_windows(sample_mask.astype(float), int(context.segment_length), int(context.hop_length))
    masks = base_masks & (mask_windows > 0.5)
    if target_label not in context.prototypes:
        raise ValueError(f"no raw prototype found for target_label={target_label!r}")
    rival_start = perf_counter()
    if rival_label is None:
        resolved_rival, rival_details = _select_rival_label(windows, masks, context, target_label, config=cfg, backend=backend)
    else:
        resolved_rival = rival_label
        rival_details = {
            "rival_selection_mode": "provided",
            "rival_selection_lambda": "",
            "rival_selection_cost": "",
            "rival_selection_second_best_cost": "",
            "rival_selection_margin": "",
        }
    rival_runtime = perf_counter() - rival_start
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
            backend=backend,
        )
        for i in range(windows.shape[0])
    ]
    aggregate_timings: dict[str, float] = {
        "lag_batch_build_runtime_sec": 0.0,
        "coarse_metric_runtime_sec": 0.0,
        "fine_metric_runtime_sec": 0.0,
        "lambda_cost_runtime_sec": 0.0,
        "alignment_weight_runtime_sec": 0.0,
        "aligned_diff_runtime_sec": 0.0,
        "aligned_cos_runtime_sec": 0.0,
        "backend_transfer_runtime_sec": 0.0,
    }
    for item in per_window:
        for key, value in item.timings.items():
            aggregate_timings[key] = aggregate_timings.get(key, 0.0) + float(value)
    lambda_results: dict[float, dict[str, Any]] = {}
    fallback_rates = []
    for lambda_value in lambdas:
        window_attrs = np.stack([item.phi_hyb_by_lambda[lambda_value] for item in per_window], axis=0)
        window_diff = np.stack([item.phi_diff_by_lambda[lambda_value] for item in per_window], axis=0)
        window_cos = np.stack([item.phi_cos_by_lambda[lambda_value] for item in per_window], axis=0)
        effective_masks = np.stack([item.effective_mask_by_lambda[lambda_value] for item in per_window], axis=0)
        window_evidence = np.asarray([item.evidence_by_lambda[lambda_value] for item in per_window], dtype=float)
        phi = _overlap_add(window_attrs, effective_masks, starts, waveform_arr.shape[0])
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
            sample_rate=int(context.target_sr),
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
            sample_rate=int(context.target_sr),
            target_alignment=target_alignments,
            rival_alignment=rival_alignments,
        )
        selected_lambdas = set(round(v, 10) for v in cfg.generation_selected_lambdas)
        generation_selected = cfg.generation_scope == "all" or (cfg.generation_scope == "selected" and round(float(lambda_value), 10) in selected_lambdas)
        generation_status = {"target": "skipped", "rival": "skipped"}
        generated_target = None
        generated_rival = None
        if generation_selected:
            if top_positive:
                metadata = {key: value for key, value in top_positive[0].items() if key != "segment"}
                generated_target, generation_status["target"] = _call_generator(
                    generator, label=target_label, lambda_hyb=lambda_value, segment=top_positive[0]["segment"], sample_rate=int(context.target_sr), role="target", metadata=metadata
                )
            if top_negative:
                metadata = {key: value for key, value in top_negative[0].items() if key != "segment"}
                generated_rival, generation_status["rival"] = _call_generator(
                    generator, label=resolved_rival, lambda_hyb=lambda_value, segment=top_negative[0]["segment"], sample_rate=int(context.target_sr), role="rival", metadata=metadata
                )
        alignment_valid = np.asarray([ta.valid and ra.valid for ta, ra in zip(target_alignments, rival_alignments, strict=True)], dtype=bool)
        fallback_used = np.asarray([ta.fallback_used or ra.fallback_used for ta, ra in zip(target_alignments, rival_alignments, strict=True)], dtype=bool)
        fallback_rate = float(np.mean(fallback_used)) if fallback_used.size else 0.0
        fallback_rates.append(fallback_rate)
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
            "effective_window_masks": effective_masks.copy(),
            "effective_window_masks_by_lambda": {float(lambda_value): effective_masks.copy()},
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
                **rival_details,
                "rival_selection_runtime_sec": rival_runtime,
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
                "fallback_rate": fallback_rate,
                "generation_scope": cfg.generation_scope,
                "generation_selected": generation_selected,
                "generation_selection_source": "generation_selected_lambdas" if cfg.generation_scope == "selected" else cfg.generation_scope,
                "requested_device": device,
                "resolved_backend": backend.name,
                "cuda_device_name": backend.cuda_device_name,
                "cupy_version": backend.cupy_version,
                "cuda_runtime_version": backend.cuda_runtime_version,
                "backend_fallback_used": backend.backend_fallback_used,
                "device": backend.name,
                "runtime_scope": "per_sample_all_lambdas",
                **aggregate_timings,
                "input_shape": tuple(waveform_arr.shape),
                "output_shape": tuple(phi.shape),
                "target_alignment": target_alignments if cfg.save_alignment_details else [],
                "rival_alignment": rival_alignments if cfg.save_alignment_details else [],
            },
        }
    diagnostic_best = max(lambdas, key=lambda value: float(lambda_results[float(value)]["evidence"])) if lambdas else None
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
            "device": backend.name,
            "requested_device": device,
            "resolved_backend": backend.name,
            "cuda_device_name": backend.cuda_device_name,
            "cupy_version": backend.cupy_version,
            "cuda_runtime_version": backend.cuda_runtime_version,
            "backend_fallback_used": backend.backend_fallback_used,
            "target_label": target_label,
            "rival_label": resolved_rival,
            "alignment_mode": cfg.alignment_mode,
            "shift_max_ms": float(cfg.shift_max_ms),
            "minimum_overlap_ratio": float(cfg.minimum_overlap_ratio),
            "alignment_temperature": float(cfg.alignment_temperature),
            "diagnostic_max_evidence_lambda": diagnostic_best,
            "diagnostic_only": True,
            "not_used_for_evaluation": True,
            "fallback_rate": float(np.mean(fallback_rates)) if fallback_rates else 0.0,
            **rival_details,
        }
    )
    return fpde.RawWaveformFPDEExplanation(
        mode="shift_robust_raw_hyb",
        target_label=target_label,
        rival_label=resolved_rival,
        waveform=waveform_arr.astype(float, copy=True),
        sample_rate=int(context.target_sr),
        lambda_results=lambda_results,
        best_lambda=None,
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
                "target_fallback_used": bool(ta.fallback_used),
                "rival_fallback_used": bool(ra.fallback_used),
                "target_fallback_reason": ta.fallback_reason,
                "rival_fallback_reason": ra.fallback_reason,
            }
        )
    return rows
