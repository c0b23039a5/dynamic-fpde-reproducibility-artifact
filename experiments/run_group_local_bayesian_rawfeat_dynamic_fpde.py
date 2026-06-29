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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from fpde.dynamic import (
    RawFeatSequence,
    bayesian_rawfeat_dynamic_fpde_explain_one,
    build_rawfeat_matrix,
    fit_bayesian_rawfeat_prototypes,
)


PER_SAMPLE_COLUMNS = [
    "cover_group_id", "sample_id", "label", "method", "n_group_samples",
    "n_like", "n_dislike", "n_posterior_samples", "lambda_hyb",
    "evidence_mean", "evidence_ci_low", "evidence_ci_high",
    "evidence_probability_positive", "evidence_sign_stability",
    "raw_group_mean", "raw_group_ci_low", "raw_group_ci_high",
    "feature_group_mean", "feature_group_ci_low", "feature_group_ci_high",
    "dt_group_mean", "dt_group_ci_low", "dt_group_ci_high",
    "max_abs_exactness_residual", "mean_abs_exactness_residual",
    "max_abs_group_sum_residual", "T", "D", "raw_dim", "feature_dim",
    "dt_dim", "rawfeat_npz_path",
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


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=Path("group_local_runner_input.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=100)
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


def _load_sequence(
    path: Path, max_frames_for_debug: int | None = None
) -> tuple[RawFeatSequence, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        missing = [name for name in ("raw", "features", "dt", "mask", "frame_starts") if name not in data]
        if missing:
            raise ValueError(f"missing NPZ arrays: {', '.join(missing)}")
        raw = np.asarray(data["raw"], dtype=np.float32)
        features = np.asarray(data["features"], dtype=np.float32)
        dt = np.asarray(data["dt"], dtype=float)
        mask = np.asarray(data["mask"])
        frame_starts = np.asarray(data["frame_starts"])

    if raw.ndim != 2:
        raise ValueError(f"raw must be 2D, got shape={raw.shape}")
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got shape={features.shape}")
    if dt.ndim != 1:
        raise ValueError(f"dt must be 1D, got shape={dt.shape}")
    if mask.ndim != 1:
        raise ValueError(f"mask must be 1D, got shape={mask.shape}")
    if frame_starts.ndim != 1:
        raise ValueError(f"frame_starts must be 1D, got shape={frame_starts.shape}")
    lengths = (raw.shape[0], features.shape[0], dt.shape[0], mask.shape[0], frame_starts.shape[0])
    if len(set(lengths)) != 1:
        raise ValueError(f"all RawFeat time dimensions must match, got {lengths}")
    valid = mask.astype(bool)
    if not np.any(valid):
        raise ValueError("mask must contain at least one valid frame")
    if not np.all(np.isfinite(raw[valid])):
        raise ValueError("raw contains NaN or inf on valid frames")
    if not np.all(np.isfinite(features[valid])):
        raise ValueError("features contains NaN or inf on valid frames")
    if not np.all(np.isfinite(dt[valid])):
        raise ValueError("dt contains NaN or inf on valid frames")
    if max_frames_for_debug is not None:
        stop = min(max_frames_for_debug, raw.shape[0])
        raw = raw[:stop]
        features = features[:stop]
        dt = dt[:stop]
        valid = valid[:stop]
        frame_starts = frame_starts[:stop]
        if not np.any(valid):
            raise ValueError("debug frame truncation removed all valid frames")
    return RawFeatSequence(raw=raw, features=features, dt=dt, mask=valid), frame_starts


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


def _label_index(posterior: Any, label: Any, name: str) -> int:
    matches = np.flatnonzero(np.asarray(posterior.prototype_labels) == label)
    if matches.size == 0:
        raise ValueError(f"no posterior prototype found for {name}={label!r}")
    return int(matches[0])


def _low_memory_hyb_summary(
    sequence: RawFeatSequence,
    posterior: Any,
    *,
    target_label: Any,
    rival_label: Any,
    lambda_hyb: float,
    normalize: str,
    eps: float = 1e-12,
) -> _LowMemoryMethodSummary:
    """Reduce one attribution draw at a time without a posterior (N, T, D) array."""
    target_index = _label_index(posterior, target_label, "target_label")
    rival_index = _label_index(posterior, rival_label, "rival_label")
    if target_label == rival_label:
        raise ValueError("target_label and rival_label must differ")

    # fpde validates and applies the training-fitted transform once.  The
    # coordinate matrix is then held as float32; all reported sums use float64.
    matrix = build_rawfeat_matrix(sequence, scaling=posterior.scaling).astype(np.float32)
    mask = np.asarray(sequence.mask, dtype=bool)
    valid_matrix = np.ascontiguousarray(matrix[mask], dtype=np.float32)
    del matrix

    n_draws = int(posterior.n_samples)
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

        # Algebraically identical Dynamic-Diff, formed as one float32 matrix.
        diff_direction = target - rival
        diff_offset = rival * rival - target * target
        attribution = valid_matrix * diff_direction
        attribution *= np.float32(2.0)
        attribution += diff_offset
        if normalize == "l1":
            diff_scale = float(np.sum(np.abs(attribution), dtype=np.float64) + eps)
        else:
            diff_scale = 1.0
        attribution *= np.float32(lambda_hyb / diff_scale)

        # The cosine contrast has a single direction vector, so it also needs
        # only one draw-sized temporary instead of separate target/rival parts.
        target_norm = float(np.linalg.norm(target))
        rival_norm = float(np.linalg.norm(rival))
        cosine_direction = target / np.float32(target_norm + eps)
        cosine_direction -= rival / np.float32(rival_norm + eps)
        cosine_attribution = valid_matrix * cosine_direction
        cosine_attribution /= (row_norms[:, None] + np.float32(eps))
        if normalize == "l1":
            cosine_scale = float(np.sum(np.abs(cosine_attribution), dtype=np.float64) + eps)
        else:
            cosine_scale = 1.0
        cosine_attribution *= np.float32((1.0 - lambda_hyb) / cosine_scale)
        attribution += cosine_attribution

        evidence = float(np.sum(attribution, dtype=np.float64))
        raw_sum = float(np.sum(attribution[:, raw_slice], dtype=np.float64))
        feature_sum = float(np.sum(attribution[:, feature_slice], dtype=np.float64))
        dt_sum = float(np.sum(attribution[:, dt_slice], dtype=np.float64))
        evidence_draws[draw] = evidence
        raw_draws[draw] = raw_sum
        feature_draws[draw] = feature_sum
        dt_draws[draw] = dt_sum
        exactness_draws[draw] = evidence - float(np.sum(attribution, dtype=np.float64))

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

    for group_id in group_ids:
        group = _cap_group(frame[frame["cover_group_id"] == group_id], args.max_samples_per_group)
        input_counts = group["label"].value_counts()
        n_like_input = int(input_counts.get("Like", 0))
        n_dislike_input = int(input_counts.get("Dislike", 0))
        if min(n_like_input, n_dislike_input) < args.min_class_count:
            skipped.append({
                "cover_group_id": group_id, "reason": "insufficient_class_count_before_loading",
                "n_group_samples": len(group), "n_like": n_like_input, "n_dislike": n_dislike_input,
            })
            continue

        loaded: list[tuple[pd.Series, RawFeatSequence, np.ndarray, Path]] = []
        for _, row in group.iterrows():
            path = _resolve_npz_path(row["rawfeat_npz_path"], input_csv)
            try:
                sequence, frame_starts = _load_sequence(path, args.max_frames_for_debug)
                loaded.append((row, sequence, frame_starts, path))
            except Exception as exc:  # isolate malformed samples without hiding them
                errors.append({
                    "cover_group_id": group_id, "sample_id": row["sample_id"],
                    "rawfeat_npz_path": str(path), "stage": "load_npz",
                    "error_type": type(exc).__name__, "error": str(exc),
                })

        loaded_labels = [item[0]["label"] for item in loaded]
        n_like = loaded_labels.count("Like")
        n_dislike = loaded_labels.count("Dislike")
        if min(n_like, n_dislike) < args.min_class_count:
            skipped.append({
                "cover_group_id": group_id, "reason": "insufficient_class_count_after_loading",
                "n_group_samples": len(loaded), "n_like": n_like, "n_dislike": n_dislike,
            })
            continue

        sequences = [item[1] for item in loaded]
        try:
            posterior = fit_bayesian_rawfeat_prototypes(
                sequences,
                loaded_labels,
                n_samples=args.n_samples,
                prototype_stat=args.prototype_stat,
                scaling=args.scaling,
                frame_weighting=args.frame_weighting,
                random_state=args.random_state,
            )
        except Exception as exc:
            errors.append({
                "cover_group_id": group_id, "sample_id": "", "rawfeat_npz_path": "",
                "stage": "fit_posterior", "error_type": type(exc).__name__, "error": str(exc),
            })
            skipped.append({
                "cover_group_id": group_id, "reason": "posterior_fit_error",
                "n_group_samples": len(loaded), "n_like": n_like, "n_dislike": n_dislike,
            })
            continue

        group_rows: list[dict[str, Any]] = []
        group_audits: list[dict[str, Any]] = []
        group_evidence_posteriors: list[np.ndarray] = []
        n_group_loaded = len(loaded)
        sample_items: list[Any] = loaded
        loaded = []
        del sequences
        for item_index, item in enumerate(sample_items):
            row, sequence, frame_starts, path = item
            result = None
            method = None
            exact = None
            group_sum_residual = None
            try:
                coordinate_summary_requested = bool(
                    args.save_coordinate_summary or (args.save_attributions and not args.low_memory)
                )
                if args.low_memory and not coordinate_summary_requested:
                    result = None
                    method = _low_memory_hyb_summary(
                        sequence,
                        posterior,
                        target_label=args.target_label,
                        rival_label=args.rival_label,
                        lambda_hyb=args.lambda_hyb,
                        normalize=args.normalize,
                    )
                else:
                    result = bayesian_rawfeat_dynamic_fpde_explain_one(
                        sequence,
                        posterior,
                        target_label=args.target_label,
                        rival_label=args.rival_label,
                        lambda_hyb=args.lambda_hyb,
                        normalize=args.normalize,
                    )
                    method = result.hyb
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
                    "T": int(sequence.raw.shape[0]),
                    "D": int(sequence.raw.shape[1] + sequence.features.shape[1] + 1),
                    "raw_dim": int(sequence.raw.shape[1]),
                    "feature_dim": int(sequence.features.shape[1]),
                    "dt_dim": 1,
                    "rawfeat_npz_path": str(path),
                }
                group_rows.append(sample_row)
                group_audits.append(audit)
                group_evidence_posteriors.append(np.asarray(method.evidence_posterior, dtype=float))
                if coordinate_summary_requested:
                    assert result is not None
                    _save_attribution(
                        output_dir, group_id, row["sample_id"], method, result.mask, frame_starts
                    )
            except Exception as exc:
                errors.append({
                    "cover_group_id": group_id, "sample_id": row["sample_id"],
                    "rawfeat_npz_path": str(path), "stage": "explain",
                    "error_type": type(exc).__name__, "error": str(exc),
                })
            finally:
                sample_items[item_index] = None
                del result, method, exact, group_sum_residual, sequence, frame_starts, item
                gc.collect()

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

    results_dir = output_dir / "results"
    _write_csv(results_dir / "per_sample_hyb.csv", per_sample, PER_SAMPLE_COLUMNS)
    _write_csv(results_dir / "group_summary.csv", group_summary, GROUP_SUMMARY_COLUMNS)
    _write_csv(results_dir / "audit_summary.csv", audits, AUDIT_COLUMNS)
    _write_csv(results_dir / "skipped_groups.csv", skipped, SKIPPED_COLUMNS)
    _write_csv(results_dir / "errors.csv", errors, ERROR_COLUMNS)

    config = vars(args).copy()
    config.update({
        "input_csv": str(input_csv), "output_dir": str(output_dir),
        "interpretation": "group-local Like-vs-Dislike prototype-evidence decomposition",
        "causal_explanation": False, "ground_truth_attribution": False,
        "uses_original_song_data": False, "global_temporal_resampling": False,
        "low_memory_streaming": bool(args.low_memory and not args.save_coordinate_summary),
        "coordinate_summary_saved": bool(
            args.save_coordinate_summary or (args.save_attributions and not args.low_memory)
        ),
        "debug_frame_truncation": args.max_frames_for_debug is not None,
        "debug_only_warning": (
            "--max-frames-for-debug truncates native sequences and must not be used for final reported results"
            if args.max_frames_for_debug is not None else None
        ),
        "n_input_rows_after_filtering": int(len(frame)),
        "n_groups_considered": len(group_ids), "n_samples_completed": len(per_sample),
    })
    (output_dir / "logs" / "run_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(_parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
