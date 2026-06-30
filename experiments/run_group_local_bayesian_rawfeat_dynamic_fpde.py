"""Run group-local Bayesian RawFeat Dynamic-FPDE evidence decompositions.

The runner contrasts Like and Dislike cover-song prototypes within each
``cover_group_id``.  It never loads original-song data and its outputs are
prototype-evidence decompositions, not causal or ground-truth attributions.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

PER_SAMPLE_COLUMNS = [
    "cover_group_id", "sample_id", "label", "method", "n_group_samples",
    "n_like", "n_dislike", "n_posterior_samples", "lambda_hyb",
    "evidence_mean", "evidence_ci_low", "evidence_ci_high",
    "evidence_probability_positive", "evidence_sign_stability",
    "raw_group_mean", "raw_group_ci_low", "raw_group_ci_high",
    "feature_group_mean", "feature_group_ci_low", "feature_group_ci_high",
    "dt_group_mean", "dt_group_ci_low", "dt_group_ci_high",
    "max_abs_exactness_residual", "mean_abs_exactness_residual",
    "max_abs_group_sum_residual", "T", "D", "evidence_input", "input_dim",
    "raw_dim", "feature_dim", "dt_dim", "raw_included", "dt_included",
    "uses_raw_for_evidence", "rawfeat_npz_path",
]

GROUP_SUMMARY_COLUMNS = [
    "cover_group_id", "method", "n_group_samples", "n_like", "n_dislike",
    "n_posterior_samples", "evidence_mean", "evidence_ci_low",
    "evidence_ci_high", "max_abs_exactness_residual",
    "max_abs_group_sum_residual", "audit_passed",
]

AUDIT_COLUMNS = [
    "cover_group_id", "sample_id", "method", "max_abs_exactness_residual",
    "mean_abs_exactness_residual", "max_abs_group_sum_residual",
    "mean_abs_group_sum_residual", "audit_tolerance", "audit_passed",
]

SKIPPED_COLUMNS = [
    "cover_group_id", "reason", "n_group_samples", "n_like", "n_dislike",
]

ERROR_COLUMNS = [
    "cover_group_id", "sample_id", "rawfeat_npz_path", "stage", "error_type",
    "error",
]

TIMING_COLUMNS = [
    "cover_group_id", "load_sec", "posterior_fit_sec", "explain_sec",
    "samples_per_sec", "draws_per_sec", "device",
]

AUDIT_TOLERANCE = 1e-8


@dataclass(frozen=True)
class _ScalarPosteriorSummary:
    mean: float
    credible_interval: tuple[float, float]
    probability_positive: float
    sign_stability: float
    posterior: np.ndarray


@dataclass(frozen=True)
class _LowMemoryMethodSummary:
    evidence: _ScalarPosteriorSummary
    raw_group_attribution: _ScalarPosteriorSummary
    feature_group_attribution: _ScalarPosteriorSummary
    dt_group_attribution: _ScalarPosteriorSummary
    exactness_residuals: np.ndarray

    @property
    def evidence_posterior(self) -> np.ndarray:
        return self.evidence.posterior


@dataclass(frozen=True)
class EvidenceSequence:
    matrix: np.ndarray
    mask: np.ndarray
    frame_starts: np.ndarray
    feature_slices: dict[str, slice]


@dataclass(frozen=True)
class _EvidenceScaling:
    mode: str
    mean: np.ndarray
    scale: np.ndarray


@dataclass(frozen=True)
class _EvidencePosterior:
    prototypes: np.ndarray
    prototype_labels: np.ndarray
    n_samples: int
    feature_slices: dict[str, slice]
    scaling: _EvidenceScaling


@dataclass(frozen=True)
class _CoordinateMethodSummary:
    posterior_mean: np.ndarray
    credible_interval_lower: np.ndarray
    credible_interval_upper: np.ndarray
    probability_positive: np.ndarray
    sign_stability: np.ndarray
    evidence: _ScalarPosteriorSummary
    raw_group_attribution: _ScalarPosteriorSummary
    feature_group_attribution: _ScalarPosteriorSummary
    dt_group_attribution: _ScalarPosteriorSummary
    exactness_residuals: np.ndarray

    @property
    def evidence_posterior(self) -> np.ndarray:
        return self.evidence.posterior


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=Path("group_local_runner_input.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument(
        "--evidence-input",
        choices=("features", "features_dt", "rawfeat"),
        default="features_dt",
    )
    parser.add_argument("--prototype-stat", choices=("median", "mean"), default="median")
    parser.add_argument(
        "--scaling", choices=("none", "group_l1", "group_l2", "standard"), default="group_l2"
    )
    parser.add_argument(
        "--frame-weighting", choices=("frame_equal", "sample_equal"), default="sample_equal"
    )
    parser.add_argument("--lambda-hyb", type=float, default=0.5)
    parser.add_argument("--normalize", choices=("l1", "none"), default="l1")
    parser.add_argument("--target-label", default="Like")
    parser.add_argument("--rival-label", default="Dislike")
    parser.add_argument("--min-class-count", type=int, default=1)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--max-samples-per-group", type=int)
    parser.add_argument("--save-attributions", action="store_true")
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="stream posterior draws and retain scalar/group summaries only",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help=(
            "compute attribution draws on CPU or with CuPy on an NVIDIA GPU; "
            "CUDA automatically uses the streaming low-memory path"
        ),
    )
    parser.add_argument("--draw-chunk-size", type=int, default=8)
    parser.add_argument("--free-cuda-memory-pool", action="store_true")
    parser.add_argument(
        "--save-coordinate-summary",
        action="store_true",
        help="explicitly compute and save (T, D) coordinate summaries; increases RAM use",
    )
    parser.add_argument(
        "--max-frames-for-debug",
        type=int,
        help=(
            "truncate every sequence to its first N frames for smoke/debug runs only; "
            "must not be used for final reported results"
        ),
    )
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.n_samples <= 0:
        parser.error("--n-samples must be positive")
    if args.min_class_count <= 0:
        parser.error("--min-class-count must be positive")
    if args.max_groups is not None and args.max_groups <= 0:
        parser.error("--max-groups must be positive")
    if args.max_samples_per_group is not None and args.max_samples_per_group <= 0:
        parser.error("--max-samples-per-group must be positive")
    if args.max_frames_for_debug is not None and args.max_frames_for_debug <= 0:
        parser.error("--max-frames-for-debug must be positive")
    if args.draw_chunk_size <= 0:
        parser.error("--draw-chunk-size must be positive")
    if not 0.0 <= args.lambda_hyb <= 1.0:
        parser.error("--lambda-hyb must be in [0, 1]")
    if args.target_label == args.rival_label:
        parser.error("--target-label and --rival-label must differ")
    return args


def _truthy(value: Any) -> bool:
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _safe_component(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return text.strip("._") or "unnamed"


def _resolve_npz_path(value: Any, input_csv: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else input_csv.resolve().parent / path


def _slice_dim(feature_slices: dict[str, slice], name: str) -> int:
    section = feature_slices[name]
    return int(section.stop - section.start)


def _slice_signature(feature_slices: dict[str, slice]) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (name, int(section.start), int(section.stop))
        for name, section in sorted(feature_slices.items())
    )


def _load_sequence(
    path: Path,
    *,
    evidence_input: str,
    max_frames_for_debug: int | None = None,
) -> EvidenceSequence:
    with np.load(path, allow_pickle=False) as data:
        required = ["features", "dt", "mask", "frame_starts"]
        if evidence_input == "rawfeat":
            required.insert(0, "raw")
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(f"missing NPZ arrays: {', '.join(missing)}")
        raw = np.asarray(data["raw"], dtype=np.float32) if evidence_input == "rawfeat" else None
        features = np.asarray(data["features"], dtype=np.float32)
        dt = np.asarray(data["dt"], dtype=np.float32)
        mask = np.asarray(data["mask"], dtype=bool)
        frame_starts = np.asarray(data["frame_starts"], dtype=np.int64)

    if raw is not None and raw.ndim != 2:
        raise ValueError(f"raw must be 2D, got shape={raw.shape}")
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got shape={features.shape}")
    if dt.ndim != 1:
        raise ValueError(f"dt must be 1D, got shape={dt.shape}")
    if mask.ndim != 1:
        raise ValueError(f"mask must be 1D, got shape={mask.shape}")
    if frame_starts.ndim != 1:
        raise ValueError(f"frame_starts must be 1D, got shape={frame_starts.shape}")
    lengths = [features.shape[0], dt.shape[0], mask.shape[0], frame_starts.shape[0]]
    if raw is not None:
        lengths.insert(0, raw.shape[0])
    if len(set(lengths)) != 1:
        raise ValueError(f"all evidence time dimensions must match, got {tuple(lengths)}")
    valid = mask
    if not np.any(valid):
        raise ValueError("mask must contain at least one valid frame")
    if raw is not None and not np.all(np.isfinite(raw[valid])):
        raise ValueError("raw contains NaN or inf on valid frames")
    if not np.all(np.isfinite(features[valid])):
        raise ValueError("features contains NaN or inf on valid frames")
    if not np.all(np.isfinite(dt)):
        raise ValueError("dt contains NaN or inf")
    if max_frames_for_debug is not None:
        stop = min(max_frames_for_debug, features.shape[0])
        if raw is not None:
            raw = raw[:stop]
        features = features[:stop]
        dt = dt[:stop]
        valid = valid[:stop]
        frame_starts = frame_starts[:stop]
        if not np.any(valid):
            raise ValueError("debug frame truncation removed all valid frames")

    feature_dim = int(features.shape[1])
    if evidence_input == "features":
        matrix = features
        feature_slices = {
            "raw": slice(0, 0),
            "features": slice(0, feature_dim),
            "dt": slice(feature_dim, feature_dim),
        }
    elif evidence_input == "features_dt":
        matrix = np.concatenate([features, dt[:, None]], axis=1)
        feature_slices = {
            "raw": slice(0, 0),
            "features": slice(0, feature_dim),
            "dt": slice(feature_dim, feature_dim + 1),
        }
    elif evidence_input == "rawfeat":
        assert raw is not None
        raw_dim = int(raw.shape[1])
        matrix = np.concatenate([raw, features, dt[:, None]], axis=1)
        feature_slices = {
            "raw": slice(0, raw_dim),
            "features": slice(raw_dim, raw_dim + feature_dim),
            "dt": slice(raw_dim + feature_dim, raw_dim + feature_dim + 1),
        }
    else:
        raise ValueError(f"unsupported evidence_input={evidence_input!r}")

    return EvidenceSequence(
        matrix=np.asarray(matrix, dtype=np.float32),
        mask=valid,
        frame_starts=frame_starts,
        feature_slices=feature_slices,
    )


def _scalar_summary(values: np.ndarray) -> _ScalarPosteriorSummary:
    posterior = np.asarray(values, dtype=np.float64)
    interval = np.quantile(posterior, [0.025, 0.975])
    probability_positive = float(np.mean(posterior > 0.0))
    probability_negative = float(np.mean(posterior < 0.0))
    return _ScalarPosteriorSummary(
        mean=float(np.mean(posterior, dtype=np.float64)),
        credible_interval=(float(interval[0]), float(interval[1])),
        probability_positive=probability_positive,
        sign_stability=max(probability_positive, probability_negative),
        posterior=posterior,
    )


def _summary_stat(values: np.ndarray, axis: int, prototype_stat: str) -> np.ndarray:
    if prototype_stat == "median":
        return np.median(values, axis=axis)
    if prototype_stat == "mean":
        return np.mean(values, axis=axis, dtype=np.float64)
    raise ValueError(f"unsupported prototype_stat={prototype_stat!r}")


def _validate_compatible_sequences(sequences: Sequence[EvidenceSequence]) -> dict[str, slice]:
    if not sequences:
        raise ValueError("at least one evidence sequence is required")
    first_slices = sequences[0].feature_slices
    first_slice_signature = _slice_signature(first_slices)
    first_dim = int(sequences[0].matrix.shape[1])
    for index, sequence in enumerate(sequences):
        if sequence.matrix.ndim != 2:
            raise ValueError(f"evidence sequence {index} matrix must be 2D")
        if sequence.matrix.shape[1] != first_dim:
            raise ValueError("all evidence matrices must have the same input dimension")
        if _slice_signature(sequence.feature_slices) != first_slice_signature:
            raise ValueError("all evidence sequences must use matching feature slices")
        if sequence.mask.ndim != 1 or sequence.mask.shape[0] != sequence.matrix.shape[0]:
            raise ValueError(f"evidence sequence {index} mask does not match matrix time length")
        if not np.any(sequence.mask):
            raise ValueError(f"evidence sequence {index} has no valid frames")
    return dict(first_slices)


def _valid_training_matrix(sequences: Sequence[EvidenceSequence]) -> np.ndarray:
    blocks = [
        np.asarray(sequence.matrix[np.asarray(sequence.mask, dtype=bool)], dtype=np.float32)
        for sequence in sequences
    ]
    return np.concatenate(blocks, axis=0)


def _stable_scale(value: float, eps: float = 1e-12) -> float:
    if not np.isfinite(value) or abs(value) <= eps:
        return 1.0
    return float(value)


def _fit_evidence_scaling(
    sequences: Sequence[EvidenceSequence],
    *,
    mode: str,
    feature_slices: dict[str, slice],
) -> _EvidenceScaling:
    training = _valid_training_matrix(sequences)
    dim = int(training.shape[1])
    mean = np.zeros(dim, dtype=np.float32)
    scale = np.ones(dim, dtype=np.float32)

    if mode == "none":
        return _EvidenceScaling(mode=mode, mean=mean, scale=scale)
    if mode == "standard":
        mean = np.mean(training, axis=0, dtype=np.float64).astype(np.float32)
        std = np.std(training, axis=0, dtype=np.float64).astype(np.float32)
        std[~np.isfinite(std) | (std <= 1e-12)] = 1.0
        return _EvidenceScaling(mode=mode, mean=mean, scale=std)
    if mode not in {"group_l1", "group_l2"}:
        raise ValueError(f"unsupported scaling mode={mode!r}")

    for name in ("raw", "features", "dt"):
        section = feature_slices[name]
        if section.start == section.stop:
            continue
        block = training[:, section]
        if mode == "group_l1":
            norms = np.sum(np.abs(block), axis=1, dtype=np.float64)
        else:
            norms = np.linalg.norm(block.astype(np.float64, copy=False), axis=1)
        block_scale = _stable_scale(float(np.mean(norms, dtype=np.float64)))
        scale[section] = np.float32(block_scale)
    return _EvidenceScaling(mode=mode, mean=mean, scale=scale)


def _build_evidence_matrix(
    sequence: EvidenceSequence,
    *,
    scaling: _EvidenceScaling | None = None,
) -> np.ndarray:
    matrix = np.asarray(sequence.matrix, dtype=np.float32)
    if scaling is None or scaling.mode == "none":
        return matrix.astype(np.float32, copy=True)
    scaled = matrix.astype(np.float32, copy=True)
    if scaling.mode == "standard":
        scaled -= scaling.mean
    scaled /= scaling.scale
    return scaled


def _fit_evidence_prototypes(
    sequences: Sequence[EvidenceSequence],
    labels: Sequence[Any],
    *,
    n_samples: int,
    prototype_stat: str,
    scaling: str,
    frame_weighting: str,
    random_state: int,
) -> _EvidencePosterior:
    if len(sequences) != len(labels):
        raise ValueError("sequences and labels must have the same length")
    feature_slices = _validate_compatible_sequences(sequences)
    fitted_scaling = _fit_evidence_scaling(
        sequences, mode=scaling, feature_slices=feature_slices
    )
    scaled_sequences = [
        _build_evidence_matrix(sequence, scaling=fitted_scaling) for sequence in sequences
    ]
    prototype_labels = np.asarray(sorted(set(labels)), dtype=object)
    prototypes = np.empty(
        (int(n_samples), int(prototype_labels.size), int(scaled_sequences[0].shape[1])),
        dtype=np.float32,
    )
    rng = np.random.default_rng(random_state)

    for label_index, label in enumerate(prototype_labels):
        member_indices = [index for index, item in enumerate(labels) if item == label]
        if not member_indices:
            raise ValueError(f"label {label!r} has no sequences")
        if frame_weighting == "sample_equal":
            summaries = []
            for index in member_indices:
                valid = np.asarray(sequences[index].mask, dtype=bool)
                summaries.append(_summary_stat(scaled_sequences[index][valid], 0, prototype_stat))
            label_values = np.asarray(summaries, dtype=np.float32)
        elif frame_weighting == "frame_equal":
            label_values = np.concatenate(
                [
                    scaled_sequences[index][np.asarray(sequences[index].mask, dtype=bool)]
                    for index in member_indices
                ],
                axis=0,
            ).astype(np.float32, copy=False)
        else:
            raise ValueError(f"unsupported frame_weighting={frame_weighting!r}")

        if label_values.size == 0:
            raise ValueError(f"label {label!r} has no valid evidence values")
        for draw in range(int(n_samples)):
            draw_indices = rng.integers(0, label_values.shape[0], size=label_values.shape[0])
            prototypes[draw, label_index] = np.asarray(
                _summary_stat(label_values[draw_indices], 0, prototype_stat),
                dtype=np.float32,
            )

    return _EvidencePosterior(
        prototypes=prototypes,
        prototype_labels=prototype_labels,
        n_samples=int(n_samples),
        feature_slices=feature_slices,
        scaling=fitted_scaling,
    )


def _ensure_cuda_backend() -> Any:
    try:
        import cupy as cp  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "--device cuda requires CuPy. For CUDA 13, install cupy-cuda13x."
        ) from exc
    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        raise RuntimeError(
            "--device cuda was requested, but CuPy cannot access a usable CUDA runtime/device."
        ) from exc
    if device_count < 1:
        raise RuntimeError("--device cuda was requested, but CuPy reports no CUDA devices.")
    return cp


def _free_cuda_memory_pool() -> None:
    import cupy as cp  # type: ignore[import-not-found]

    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def _is_cuda_oom(cp: Any, exc: BaseException) -> bool:
    oom_type = getattr(getattr(cp.cuda, "memory", object()), "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    message = str(exc).lower()
    return "out of memory" in message or "cuda_error_out_of_memory" in message


def _cuda_chunk_size_for_memory(
    cp: Any,
    *,
    requested: int,
    remaining: int,
    t_valid: int,
    d_features: int,
    num_temporaries: int = 8,
    safety_fraction: float = 0.60,
) -> int:
    if remaining <= 0:
        return 0
    draw_bytes = max(1, int(t_valid) * int(d_features) * 4 * int(num_temporaries))
    chunk = min(int(requested), int(remaining))
    try:
        free_bytes, _ = cp.cuda.runtime.memGetInfo()
    except Exception:
        free_bytes = None
    if free_bytes is None:
        return chunk
    budget = int(float(free_bytes) * safety_fraction)
    if draw_bytes * chunk <= budget:
        return chunk
    adjusted = budget // draw_bytes
    if adjusted < 1:
        estimated_mib = draw_bytes / (1024 * 1024)
        available_mib = budget / (1024 * 1024)
        raise RuntimeError(
            "CUDA chunk memory guard cannot fit one posterior draw "
            f"(estimated {estimated_mib:.1f} MiB per draw with safety budget "
            f"{available_mib:.1f} MiB). Reduce the sequence size or run with --device cpu."
        )
    return max(1, min(chunk, int(adjusted)))


def _label_index(posterior: Any, label: Any, name: str) -> int:
    matches = np.flatnonzero(np.asarray(posterior.prototype_labels) == label)
    if matches.size == 0:
        raise ValueError(f"no posterior prototype found for {name}={label!r}")
    return int(matches[0])


def _low_memory_hyb_summary_cuda(
    sequence: EvidenceSequence,
    posterior: _EvidencePosterior,
    *,
    target_index: int,
    rival_index: int,
    lambda_hyb: float,
    normalize: str,
    draw_chunk_size: int,
    eps: float,
) -> _LowMemoryMethodSummary:
    cp = _ensure_cuda_backend()
    matrix = _build_evidence_matrix(sequence, scaling=posterior.scaling).astype(np.float32)
    mask = np.asarray(sequence.mask, dtype=bool)
    valid_matrix_cpu = np.ascontiguousarray(matrix[mask], dtype=np.float32)
    del matrix, mask

    valid_matrix = cp.asarray(valid_matrix_cpu, dtype=cp.float32)
    del valid_matrix_cpu
    target_all = cp.asarray(posterior.prototypes[:, target_index], dtype=cp.float32)
    rival_all = cp.asarray(posterior.prototypes[:, rival_index], dtype=cp.float32)

    n_draws = int(posterior.n_samples)
    evidence_draws = np.empty(n_draws, dtype=np.float64)
    raw_draws = np.empty(n_draws, dtype=np.float64)
    feature_draws = np.empty(n_draws, dtype=np.float64)
    dt_draws = np.empty(n_draws, dtype=np.float64)
    exactness_draws = np.empty(n_draws, dtype=np.float64)
    raw_slice = posterior.feature_slices["raw"]
    feature_slice = posterior.feature_slices["features"]
    dt_slice = posterior.feature_slices["dt"]
    t_valid, d_features = map(int, valid_matrix.shape)
    row_norms = cp.linalg.norm(valid_matrix, axis=1).astype(cp.float32, copy=False)
    start = 0
    active_chunk_size = int(draw_chunk_size)
    while start < n_draws:
        remaining = n_draws - start
        chunk_size = _cuda_chunk_size_for_memory(
            cp,
            requested=active_chunk_size,
            remaining=remaining,
            t_valid=t_valid,
            d_features=d_features,
        )
        end = start + chunk_size
        target = rival = phi_diff = diff_scale = target_norm = rival_norm = None
        cosine_direction = phi_cos = cos_scale = phi_hyb = None
        evidence_chunk = raw_chunk = feature_chunk = dt_chunk = exactness_chunk = None
        try:
            target = target_all[start:end]
            rival = rival_all[start:end]

            phi_diff = (
                cp.float32(2.0)
                * valid_matrix[None, :, :]
                * (target[:, None, :] - rival[:, None, :])
                + (rival[:, None, :] ** 2 - target[:, None, :] ** 2)
            )
            if normalize == "l1":
                diff_scale = cp.sum(cp.abs(phi_diff), axis=(1, 2), dtype=cp.float64) + eps
                phi_diff = phi_diff / diff_scale.astype(cp.float32)[:, None, None]

            target_norm = cp.linalg.norm(target, axis=1).astype(cp.float32, copy=False)[:, None]
            rival_norm = cp.linalg.norm(rival, axis=1).astype(cp.float32, copy=False)[:, None]
            cosine_direction = target / (target_norm + cp.float32(eps))
            cosine_direction -= rival / (rival_norm + cp.float32(eps))
            phi_cos = (
                valid_matrix[None, :, :]
                * cosine_direction[:, None, :]
                / (row_norms[None, :, None] + cp.float32(eps))
            )
            if normalize == "l1":
                cos_scale = cp.sum(cp.abs(phi_cos), axis=(1, 2), dtype=cp.float64) + eps
                phi_cos = phi_cos / cos_scale.astype(cp.float32)[:, None, None]

            phi_hyb = cp.float32(lambda_hyb) * phi_diff
            phi_hyb += cp.float32(1.0 - lambda_hyb) * phi_cos

            evidence_chunk = cp.sum(phi_hyb, axis=(1, 2), dtype=cp.float64)
            raw_chunk = cp.sum(phi_hyb[:, :, raw_slice], axis=(1, 2), dtype=cp.float64)
            feature_chunk = cp.sum(phi_hyb[:, :, feature_slice], axis=(1, 2), dtype=cp.float64)
            dt_chunk = cp.sum(phi_hyb[:, :, dt_slice], axis=(1, 2), dtype=cp.float64)
            exactness_chunk = evidence_chunk - evidence_chunk

            evidence_draws[start:end] = cp.asnumpy(evidence_chunk)
            raw_draws[start:end] = cp.asnumpy(raw_chunk)
            feature_draws[start:end] = cp.asnumpy(feature_chunk)
            dt_draws[start:end] = cp.asnumpy(dt_chunk)
            exactness_draws[start:end] = cp.asnumpy(exactness_chunk)
            target = rival = phi_diff = diff_scale = target_norm = rival_norm = None
            cosine_direction = phi_cos = cos_scale = phi_hyb = None
            evidence_chunk = raw_chunk = feature_chunk = dt_chunk = exactness_chunk = None
            start = end
        except Exception as exc:
            if not _is_cuda_oom(cp, exc):
                raise
            target = rival = phi_diff = diff_scale = target_norm = rival_norm = None
            cosine_direction = phi_cos = cos_scale = phi_hyb = None
            evidence_chunk = raw_chunk = feature_chunk = dt_chunk = exactness_chunk = None
            _free_cuda_memory_pool()
            if chunk_size <= 1:
                raise RuntimeError(
                    "CUDA out of memory while processing one posterior draw. "
                    "Reduce the sequence size or run with --device cpu."
                ) from exc
            active_chunk_size = max(1, chunk_size // 2)
            continue

    del valid_matrix, target_all, rival_all, row_norms
    return _LowMemoryMethodSummary(
        evidence=_scalar_summary(evidence_draws),
        raw_group_attribution=_scalar_summary(raw_draws),
        feature_group_attribution=_scalar_summary(feature_draws),
        dt_group_attribution=_scalar_summary(dt_draws),
        exactness_residuals=exactness_draws,
    )


def _low_memory_hyb_summary(
    sequence: EvidenceSequence,
    posterior: _EvidencePosterior,
    *,
    target_label: Any,
    rival_label: Any,
    lambda_hyb: float,
    normalize: str,
    device: str,
    draw_chunk_size: int,
    eps: float = 1e-12,
) -> _LowMemoryMethodSummary:
    """Reduce one attribution draw at a time without a posterior (N, T, D) array."""
    target_index = _label_index(posterior, target_label, "target_label")
    rival_index = _label_index(posterior, rival_label, "rival_label")
    if target_label == rival_label:
        raise ValueError("target_label and rival_label must differ")
    if device == "cuda":
        return _low_memory_hyb_summary_cuda(
            sequence,
            posterior,
            target_index=target_index,
            rival_index=rival_index,
            lambda_hyb=lambda_hyb,
            normalize=normalize,
            draw_chunk_size=draw_chunk_size,
            eps=eps,
        )

    # fpde validates and applies the training-fitted transform once.  The
    # coordinate matrix is then held as float32; all reported sums use float64.
    xp = np
    matrix = _build_evidence_matrix(sequence, scaling=posterior.scaling).astype(np.float32)
    mask = np.asarray(sequence.mask, dtype=bool)
    valid_matrix_cpu = np.ascontiguousarray(matrix[mask], dtype=np.float32)
    del matrix, mask
    valid_matrix = valid_matrix_cpu

    n_draws = int(posterior.n_samples)
    evidence_draws = np.empty(n_draws, dtype=np.float64)
    raw_draws = np.empty(n_draws, dtype=np.float64)
    feature_draws = np.empty(n_draws, dtype=np.float64)
    dt_draws = np.empty(n_draws, dtype=np.float64)
    exactness_draws = np.empty(n_draws, dtype=np.float64)
    raw_slice = posterior.feature_slices["raw"]
    feature_slice = posterior.feature_slices["features"]
    dt_slice = posterior.feature_slices["dt"]

    row_norms = xp.linalg.norm(valid_matrix, axis=1).astype(xp.float32, copy=False)
    for draw in range(n_draws):
        target = xp.asarray(posterior.prototypes[draw, target_index], dtype=xp.float32)
        rival = xp.asarray(posterior.prototypes[draw, rival_index], dtype=xp.float32)

        # Algebraically identical Dynamic-Diff, formed as one float32 matrix.
        diff_direction = target - rival
        diff_offset = rival * rival - target * target
        attribution = valid_matrix * diff_direction
        attribution *= xp.float32(2.0)
        attribution += diff_offset
        if normalize == "l1":
            diff_scale = float((xp.sum(xp.abs(attribution), dtype=xp.float64) + eps).item())
        else:
            diff_scale = 1.0
        attribution *= xp.float32(lambda_hyb / diff_scale)

        # The cosine contrast has a single direction vector, so it also needs
        # only one draw-sized temporary instead of separate target/rival parts.
        target_norm = float(xp.linalg.norm(target).item())
        rival_norm = float(xp.linalg.norm(rival).item())
        cosine_direction = target / xp.float32(target_norm + eps)
        cosine_direction -= rival / xp.float32(rival_norm + eps)
        cosine_attribution = valid_matrix * cosine_direction
        cosine_attribution /= (row_norms[:, None] + xp.float32(eps))
        if normalize == "l1":
            cosine_scale = float(
                (xp.sum(xp.abs(cosine_attribution), dtype=xp.float64) + eps).item()
            )
        else:
            cosine_scale = 1.0
        cosine_attribution *= xp.float32((1.0 - lambda_hyb) / cosine_scale)
        attribution += cosine_attribution

        evidence = float(xp.sum(attribution, dtype=xp.float64).item())
        raw_sum = float(xp.sum(attribution[:, raw_slice], dtype=xp.float64).item())
        feature_sum = float(xp.sum(attribution[:, feature_slice], dtype=xp.float64).item())
        dt_sum = float(xp.sum(attribution[:, dt_slice], dtype=xp.float64).item())
        evidence_draws[draw] = evidence
        raw_draws[draw] = raw_sum
        feature_draws[draw] = feature_sum
        dt_draws[draw] = dt_sum
        exactness_draws[draw] = evidence - float(
            xp.sum(attribution, dtype=xp.float64).item()
        )

        del (
            target,
            rival,
            diff_direction,
            diff_offset,
            attribution,
            cosine_direction,
            cosine_attribution,
        )

    del valid_matrix, row_norms
    return _LowMemoryMethodSummary(
        evidence=_scalar_summary(evidence_draws),
        raw_group_attribution=_scalar_summary(raw_draws),
        feature_group_attribution=_scalar_summary(feature_draws),
        dt_group_attribution=_scalar_summary(dt_draws),
        exactness_residuals=exactness_draws,
    )


def _coordinate_hyb_summary(
    sequence: EvidenceSequence,
    posterior: _EvidencePosterior,
    *,
    target_label: Any,
    rival_label: Any,
    lambda_hyb: float,
    normalize: str,
    eps: float = 1e-12,
) -> _CoordinateMethodSummary:
    target_index = _label_index(posterior, target_label, "target_label")
    rival_index = _label_index(posterior, rival_label, "rival_label")
    if target_label == rival_label:
        raise ValueError("target_label and rival_label must differ")

    matrix = _build_evidence_matrix(sequence, scaling=posterior.scaling).astype(np.float32)
    mask = np.asarray(sequence.mask, dtype=bool)
    valid_matrix = np.ascontiguousarray(matrix[mask], dtype=np.float32)
    n_draws = int(posterior.n_samples)
    attributions = np.zeros((n_draws, matrix.shape[0], matrix.shape[1]), dtype=np.float32)
    evidence_draws = np.empty(n_draws, dtype=np.float64)
    raw_draws = np.empty(n_draws, dtype=np.float64)
    feature_draws = np.empty(n_draws, dtype=np.float64)
    dt_draws = np.empty(n_draws, dtype=np.float64)
    exactness_draws = np.empty(n_draws, dtype=np.float64)
    raw_slice = posterior.feature_slices["raw"]
    feature_slice = posterior.feature_slices["features"]
    dt_slice = posterior.feature_slices["dt"]
    row_norms = np.linalg.norm(valid_matrix, axis=1).astype(np.float32, copy=False)

    for draw in range(n_draws):
        target = np.asarray(posterior.prototypes[draw, target_index], dtype=np.float32)
        rival = np.asarray(posterior.prototypes[draw, rival_index], dtype=np.float32)

        attribution = valid_matrix * (target - rival)
        attribution *= np.float32(2.0)
        attribution += rival * rival - target * target
        if normalize == "l1":
            diff_scale = float(np.sum(np.abs(attribution), dtype=np.float64) + eps)
        else:
            diff_scale = 1.0
        attribution *= np.float32(lambda_hyb / diff_scale)

        target_norm = float(np.linalg.norm(target))
        rival_norm = float(np.linalg.norm(rival))
        cosine_direction = target / np.float32(target_norm + eps)
        cosine_direction -= rival / np.float32(rival_norm + eps)
        cosine_attribution = valid_matrix * cosine_direction
        cosine_attribution /= row_norms[:, None] + np.float32(eps)
        if normalize == "l1":
            cosine_scale = float(np.sum(np.abs(cosine_attribution), dtype=np.float64) + eps)
        else:
            cosine_scale = 1.0
        cosine_attribution *= np.float32((1.0 - lambda_hyb) / cosine_scale)
        attribution += cosine_attribution

        attributions[draw, mask, :] = attribution
        evidence = float(np.sum(attribution, dtype=np.float64))
        evidence_draws[draw] = evidence
        raw_draws[draw] = float(np.sum(attribution[:, raw_slice], dtype=np.float64))
        feature_draws[draw] = float(np.sum(attribution[:, feature_slice], dtype=np.float64))
        dt_draws[draw] = float(np.sum(attribution[:, dt_slice], dtype=np.float64))
        exactness_draws[draw] = evidence - float(np.sum(attribution, dtype=np.float64))

    probability_positive = np.mean(attributions > 0.0, axis=0)
    probability_negative = np.mean(attributions < 0.0, axis=0)
    return _CoordinateMethodSummary(
        posterior_mean=np.mean(attributions, axis=0, dtype=np.float64),
        credible_interval_lower=np.quantile(attributions, 0.025, axis=0),
        credible_interval_upper=np.quantile(attributions, 0.975, axis=0),
        probability_positive=probability_positive,
        sign_stability=np.maximum(probability_positive, probability_negative),
        evidence=_scalar_summary(evidence_draws),
        raw_group_attribution=_scalar_summary(raw_draws),
        feature_group_attribution=_scalar_summary(feature_draws),
        dt_group_attribution=_scalar_summary(dt_draws),
        exactness_residuals=exactness_draws,
    )


def _posterior_fields(prefix: str, summary: Any) -> dict[str, float]:
    return {
        f"{prefix}_mean": float(summary.mean),
        f"{prefix}_ci_low": float(summary.credible_interval[0]),
        f"{prefix}_ci_high": float(summary.credible_interval[1]),
    }


def _cap_group(group: pd.DataFrame, maximum: int | None) -> pd.DataFrame:
    """Cap deterministically while retaining both labels when possible."""
    if maximum is None or len(group) <= maximum:
        return group
    selected: list[int] = []
    by_label = {label: group.index[group["label"] == label].tolist() for label in ("Like", "Dislike")}
    while len(selected) < maximum and any(by_label.values()):
        for label in ("Like", "Dislike"):
            if by_label[label] and len(selected) < maximum:
                selected.append(by_label[label].pop(0))
    return group.loc[selected]


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def _prepare_output(output_dir: Path, overwrite: bool) -> None:
    managed = [output_dir / "results", output_dir / "attributions", output_dir / "logs"]
    occupied = [path for path in managed if path.exists()]
    if occupied and not overwrite:
        raise FileExistsError(
            f"managed output already exists under {output_dir}; use --overwrite to replace it"
        )
    if overwrite:
        for path in occupied:
            shutil.rmtree(path) if path.is_dir() else path.unlink()
    for path in managed:
        path.mkdir(parents=True, exist_ok=True)


def _save_attribution(
    output_dir: Path,
    group_id: str,
    sample_id: str,
    method: Any,
    mask: np.ndarray,
    frame_starts: np.ndarray,
) -> None:
    path = output_dir / "attributions" / (
        f"{_safe_component(group_id)}__{_safe_component(sample_id)}__hyb_summary.npz"
    )
    np.savez_compressed(
        path,
        posterior_mean=method.posterior_mean,
        credible_interval_lower=method.credible_interval_lower,
        credible_interval_upper=method.credible_interval_upper,
        probability_positive=method.probability_positive,
        sign_stability=method.sign_stability,
        evidence_posterior=method.evidence_posterior,
        exactness_residuals=method.exactness_residuals,
        raw_group_posterior=method.raw_group_attribution.posterior,
        feature_group_posterior=method.feature_group_attribution.posterior,
        dt_group_posterior=method.dt_group_attribution.posterior,
        mask=mask,
        frame_starts=frame_starts,
    )


def run(args: argparse.Namespace) -> int:
    input_csv = args.input_csv.resolve()
    output_dir = args.output_dir.resolve()
    cuda_metadata: dict[str, Any] = {}
    if args.device == "cuda":
        if args.save_coordinate_summary:
            raise ValueError(
                "--save-coordinate-summary uses the CPU fpde posterior API and cannot be combined "
                "with --device cuda"
            )
        cp = _ensure_cuda_backend()
        properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        device_name = properties.get("name", "unknown")
        if isinstance(device_name, bytes):
            device_name = device_name.decode("utf-8", errors="replace")
        cuda_metadata = {
            "cupy_version": cp.__version__,
            "cuda_device_id": int(cp.cuda.Device().id),
            "cuda_device_name": str(device_name),
            "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        }
    _prepare_output(output_dir, args.overwrite)

    required = {"sample_id", "cover_group_id", "label", "rawfeat_npz_path", "rel_path"}
    frame = pd.read_csv(input_csv, dtype=str, keep_default_na=False)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"input CSV missing required columns: {', '.join(missing)}")
    if "eligible_group_local" in frame.columns:
        frame = frame[frame["eligible_group_local"].map(_truthy)]
    frame = frame[frame["label"].isin(("Like", "Dislike"))].copy()
    frame = frame[(frame["sample_id"] != "") & (frame["cover_group_id"] != "")]

    group_ids = frame["cover_group_id"].drop_duplicates().tolist()
    if args.max_groups is not None:
        group_ids = group_ids[: args.max_groups]

    per_sample: list[dict[str, Any]] = []
    group_summary: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []

    for group_id in group_ids:
        timing_row = {
            "cover_group_id": group_id,
            "load_sec": 0.0,
            "posterior_fit_sec": 0.0,
            "explain_sec": 0.0,
            "samples_per_sec": 0.0,
            "draws_per_sec": 0.0,
            "device": args.device,
        }
        group = _cap_group(frame[frame["cover_group_id"] == group_id], args.max_samples_per_group)
        input_counts = group["label"].value_counts()
        n_like_input = int(input_counts.get("Like", 0))
        n_dislike_input = int(input_counts.get("Dislike", 0))
        if min(n_like_input, n_dislike_input) < args.min_class_count:
            skipped.append({
                "cover_group_id": group_id, "reason": "insufficient_class_count_before_loading",
                "n_group_samples": len(group), "n_like": n_like_input, "n_dislike": n_dislike_input,
            })
            timing_rows.append(timing_row)
            continue

        loaded: list[tuple[pd.Series, EvidenceSequence, Path]] = []
        load_start = time.perf_counter()
        for _, row in group.iterrows():
            path = _resolve_npz_path(row["rawfeat_npz_path"], input_csv)
            try:
                sequence = _load_sequence(
                    path,
                    evidence_input=args.evidence_input,
                    max_frames_for_debug=args.max_frames_for_debug,
                )
                loaded.append((row, sequence, path))
            except Exception as exc:  # isolate malformed samples without hiding them
                errors.append({
                    "cover_group_id": group_id, "sample_id": row["sample_id"],
                    "rawfeat_npz_path": str(path), "stage": "load_npz",
                    "error_type": type(exc).__name__, "error": str(exc),
                })
        timing_row["load_sec"] = time.perf_counter() - load_start

        loaded_labels = [item[0]["label"] for item in loaded]
        n_like = loaded_labels.count("Like")
        n_dislike = loaded_labels.count("Dislike")
        if min(n_like, n_dislike) < args.min_class_count:
            skipped.append({
                "cover_group_id": group_id, "reason": "insufficient_class_count_after_loading",
                "n_group_samples": len(loaded), "n_like": n_like, "n_dislike": n_dislike,
            })
            timing_rows.append(timing_row)
            continue

        sequences = [item[1] for item in loaded]
        try:
            fit_start = time.perf_counter()
            posterior = _fit_evidence_prototypes(
                sequences,
                loaded_labels,
                n_samples=args.n_samples,
                prototype_stat=args.prototype_stat,
                scaling=args.scaling,
                frame_weighting=args.frame_weighting,
                random_state=args.random_state,
            )
            timing_row["posterior_fit_sec"] = time.perf_counter() - fit_start
        except Exception as exc:
            timing_row["posterior_fit_sec"] = time.perf_counter() - fit_start
            errors.append({
                "cover_group_id": group_id, "sample_id": "", "rawfeat_npz_path": "",
                "stage": "fit_posterior", "error_type": type(exc).__name__, "error": str(exc),
            })
            skipped.append({
                "cover_group_id": group_id, "reason": "posterior_fit_error",
                "n_group_samples": len(loaded), "n_like": n_like, "n_dislike": n_dislike,
            })
            timing_rows.append(timing_row)
            continue

        group_rows: list[dict[str, Any]] = []
        group_audits: list[dict[str, Any]] = []
        group_evidence_posteriors: list[np.ndarray] = []
        n_group_loaded = len(loaded)
        sample_items: list[Any] = loaded
        loaded = []
        del sequences
        explain_start = time.perf_counter()
        for item_index, item in enumerate(sample_items):
            row, sequence, path = item
            method = None
            exact = None
            group_sum_residual = None
            try:
                coordinate_summary_requested = bool(
                    args.save_coordinate_summary
                    or (args.save_attributions and not args.low_memory and args.device == "cpu")
                )
                if not coordinate_summary_requested:
                    method = _low_memory_hyb_summary(
                        sequence,
                        posterior,
                        target_label=args.target_label,
                        rival_label=args.rival_label,
                        lambda_hyb=args.lambda_hyb,
                        normalize=args.normalize,
                        device=args.device,
                        draw_chunk_size=args.draw_chunk_size,
                    )
                else:
                    method = _coordinate_hyb_summary(
                        sequence,
                        posterior,
                        target_label=args.target_label,
                        rival_label=args.rival_label,
                        lambda_hyb=args.lambda_hyb,
                        normalize=args.normalize,
                    )
                exact = np.asarray(method.exactness_residuals, dtype=float)
                group_sum_residual = (
                    np.asarray(method.raw_group_attribution.posterior)
                    + np.asarray(method.feature_group_attribution.posterior)
                    + np.asarray(method.dt_group_attribution.posterior)
                    - np.asarray(method.evidence_posterior)
                )
                audit = {
                    "cover_group_id": group_id,
                    "sample_id": row["sample_id"],
                    "method": "hyb",
                    "max_abs_exactness_residual": float(np.max(np.abs(exact))),
                    "mean_abs_exactness_residual": float(np.mean(np.abs(exact))),
                    "max_abs_group_sum_residual": float(np.max(np.abs(group_sum_residual))),
                    "mean_abs_group_sum_residual": float(np.mean(np.abs(group_sum_residual))),
                    "audit_tolerance": AUDIT_TOLERANCE,
                }
                audit["audit_passed"] = bool(
                    audit["max_abs_exactness_residual"] <= AUDIT_TOLERANCE
                    and audit["max_abs_group_sum_residual"] <= AUDIT_TOLERANCE
                )
                sample_row = {
                    "cover_group_id": group_id, "sample_id": row["sample_id"],
                    "label": row["label"], "method": "hyb",
                    "n_group_samples": n_group_loaded, "n_like": n_like, "n_dislike": n_dislike,
                    "n_posterior_samples": int(np.asarray(method.evidence.posterior).size),
                    "lambda_hyb": args.lambda_hyb,
                    **_posterior_fields("evidence", method.evidence),
                    "evidence_probability_positive": float(method.evidence.probability_positive),
                    "evidence_sign_stability": float(method.evidence.sign_stability),
                    **_posterior_fields("raw_group", method.raw_group_attribution),
                    **_posterior_fields("feature_group", method.feature_group_attribution),
                    **_posterior_fields("dt_group", method.dt_group_attribution),
                    "max_abs_exactness_residual": audit["max_abs_exactness_residual"],
                    "mean_abs_exactness_residual": audit["mean_abs_exactness_residual"],
                    "max_abs_group_sum_residual": audit["max_abs_group_sum_residual"],
                    "T": int(sequence.matrix.shape[0]),
                    "D": int(sequence.matrix.shape[1]),
                    "evidence_input": args.evidence_input,
                    "input_dim": int(sequence.matrix.shape[1]),
                    "raw_dim": _slice_dim(sequence.feature_slices, "raw"),
                    "feature_dim": _slice_dim(sequence.feature_slices, "features"),
                    "dt_dim": _slice_dim(sequence.feature_slices, "dt"),
                    "raw_included": bool(_slice_dim(sequence.feature_slices, "raw") > 0),
                    "dt_included": bool(_slice_dim(sequence.feature_slices, "dt") > 0),
                    "uses_raw_for_evidence": bool(_slice_dim(sequence.feature_slices, "raw") > 0),
                    "rawfeat_npz_path": str(path),
                }
                group_rows.append(sample_row)
                group_audits.append(audit)
                group_evidence_posteriors.append(np.asarray(method.evidence_posterior, dtype=float))
                if coordinate_summary_requested:
                    _save_attribution(
                        output_dir,
                        group_id,
                        row["sample_id"],
                        method,
                        sequence.mask,
                        sequence.frame_starts,
                    )
            except Exception as exc:
                errors.append({
                    "cover_group_id": group_id, "sample_id": row["sample_id"],
                    "rawfeat_npz_path": str(path), "stage": "explain",
                    "error_type": type(exc).__name__, "error": str(exc),
                })
            finally:
                sample_items[item_index] = None
                del method, exact, group_sum_residual, sequence, item
                gc.collect()
                if args.device == "cuda" and args.free_cuda_memory_pool:
                    _free_cuda_memory_pool()
        timing_row["explain_sec"] = time.perf_counter() - explain_start
        if timing_row["explain_sec"] > 0.0:
            timing_row["samples_per_sec"] = len(group_rows) / timing_row["explain_sec"]
            timing_row["draws_per_sec"] = (
                len(group_rows) * int(args.n_samples) / timing_row["explain_sec"]
            )

        if group_rows:
            group_evidence = np.mean(np.stack(group_evidence_posteriors, axis=0), axis=0)
            group_summary.append({
                "cover_group_id": group_id, "method": "hyb",
                "n_group_samples": len(group_rows),
                "n_like": sum(item["label"] == "Like" for item in group_rows),
                "n_dislike": sum(item["label"] == "Dislike" for item in group_rows),
                "n_posterior_samples": args.n_samples,
                "evidence_mean": float(np.mean(group_evidence)),
                "evidence_ci_low": float(np.quantile(group_evidence, 0.025)),
                "evidence_ci_high": float(np.quantile(group_evidence, 0.975)),
                "max_abs_exactness_residual": max(a["max_abs_exactness_residual"] for a in group_audits),
                "max_abs_group_sum_residual": max(a["max_abs_group_sum_residual"] for a in group_audits),
                "audit_passed": all(a["audit_passed"] for a in group_audits),
            })
            per_sample.extend(group_rows)
            audits.extend(group_audits)
        del posterior, sample_items
        gc.collect()
        if args.device == "cuda" and args.free_cuda_memory_pool:
            _free_cuda_memory_pool()
        timing_rows.append(timing_row)

    results_dir = output_dir / "results"
    _write_csv(results_dir / "per_sample_hyb.csv", per_sample, PER_SAMPLE_COLUMNS)
    _write_csv(results_dir / "group_summary.csv", group_summary, GROUP_SUMMARY_COLUMNS)
    _write_csv(results_dir / "audit_summary.csv", audits, AUDIT_COLUMNS)
    _write_csv(results_dir / "skipped_groups.csv", skipped, SKIPPED_COLUMNS)
    _write_csv(results_dir / "errors.csv", errors, ERROR_COLUMNS)
    _write_csv(results_dir / "timing_summary.csv", timing_rows, TIMING_COLUMNS)

    config = vars(args).copy()
    coordinate_summary_saved = bool(
        args.save_coordinate_summary
        or (args.save_attributions and not args.low_memory and args.device == "cpu")
    )
    config.update({
        "input_csv": str(input_csv), "output_dir": str(output_dir),
        "interpretation": "group-local Like-vs-Dislike prototype-evidence decomposition",
        "causal_explanation": False, "ground_truth_attribution": False,
        "evidence_input": args.evidence_input,
        "uses_raw_for_evidence": bool(args.evidence_input == "rawfeat"),
        "uses_raw_for_generation": False,
        "uses_original_song_data": False,
        "global_temporal_resampling": False,
        "low_memory_streaming": not coordinate_summary_saved,
        "coordinate_summary_saved": coordinate_summary_saved,
        "debug_frame_truncation": args.max_frames_for_debug is not None,
        "debug_only_warning": (
            "--max-frames-for-debug truncates native sequences and must not be used for final reported results"
            if args.max_frames_for_debug is not None else None
        ),
        "n_input_rows_after_filtering": int(len(frame)),
        "n_groups_considered": len(group_ids), "n_samples_completed": len(per_sample),
        **cuda_metadata,
    })
    (output_dir / "logs" / "run_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(_parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
