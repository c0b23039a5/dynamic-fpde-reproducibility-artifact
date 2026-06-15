"""Run ESC-50 Dynamic-FPDE experiments with fpde-xai/fpde@dynamic."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fpde import (
    DynamicFPDEExplanation,
    dynamic_cos_fpde,
    dynamic_diff_fpde,
    dynamic_fpde_explain_one,
    dynamic_hyb_fpde,
    prepare_dynamic_fpde_context,
    resample_time_series_linear,
    select_dynamic_lambda,
    temporal_deletion_insertion_curves,
)

from experiments.dynamic_fpde_audio.aggregate import (
    aggregate_additivity,
    aggregate_by_method,
    average_random_repetitions,
    positive_margin_rows,
    write_csv,
)
from experiments.dynamic_fpde_audio.baselines import energy_frame_scores, random_frame_scores
from experiments.dynamic_fpde_audio.datasets import (
    ESCSample,
    get_mode_config,
    labels_for,
    parse_folds,
    read_esc50_metadata,
    split_esc50,
)
from experiments.dynamic_fpde_audio.features import FeatureConfig, fit_standardizer, load_or_extract_features, transform_features
from experiments.dynamic_fpde_audio.tables import generate_tables


SAMPLE_FIELDS = [
    "dataset",
    "fold",
    "seed",
    "sample_id",
    "true_label",
    "target_label",
    "rival_label",
    "common_rival_label",
    "method",
    "lambda_hyb",
    "evidence",
    "evidence_role",
    "evaluation_evidence",
    "evaluation_margin",
    "prototype_margin",
    "prototype_margin_positive",
    "prototype_margin_sign",
    "selection_margin",
    "selection_margin_positive",
    "selection_margin_sign",
    "selection_margin_source",
    "exactness_residual",
    "abs_exactness_residual",
    "deletion_drop_auc",
    "insertion_gain_auc",
    "combined_score",
    "runtime_sec",
    "T",
    "F",
    "random_repetition",
    "aggregation_unit",
]


@dataclass(frozen=True)
class _ResolvedDynamicInput:
    sample: ESCSample
    X: np.ndarray
    target_proto: np.ndarray
    rival_proto: np.ndarray
    anchor: np.ndarray
    target_idx: int
    rival_idx: int
    target_label: Any
    rival_label: Any
    diff_positive_score: float
    diff_negative_score: float
    cos_positive_score: float
    cos_negative_score: float


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _package_version(package: str) -> str | None:
    try:
        import importlib.metadata

        return importlib.metadata.version(package)
    except Exception:
        return None


def _load_feature_map(
    samples: list[ESCSample],
    *,
    output_dir: Path,
    config: FeatureConfig,
    skip_errors: bool,
) -> tuple[dict[str, np.ndarray], list[str], list[dict[str, str]]]:
    cache_dir = output_dir / "cache" / "features"
    features: dict[str, np.ndarray] = {}
    feature_names: list[str] | None = None
    errors: list[dict[str, str]] = []
    for sample in samples:
        try:
            X, names, _ = load_or_extract_features(
                sample.audio_path,
                dataset="esc50",
                sample_id=sample.sample_id,
                cache_dir=cache_dir,
                config=config,
            )
        except Exception as exc:
            if not skip_errors:
                raise
            errors.append({"sample_id": sample.sample_id, "audio_path": str(sample.audio_path), "error": str(exc)})
            continue
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise ValueError(f"feature names changed for sample {sample.sample_id}")
        features[sample.sample_id] = X
    if feature_names is None:
        raise RuntimeError("no features were extracted")
    return features, feature_names, errors


def _baseline_explanation(base: DynamicFPDEExplanation, scores: np.ndarray, *, method: str) -> DynamicFPDEExplanation:
    return DynamicFPDEExplanation(
        mode=method,
        evidence=base.evidence,
        attributions=np.zeros_like(base.attributions),
        time_importance=np.asarray(scores, dtype=float),
        feature_importance=np.zeros(base.attributions.shape[1], dtype=float),
        positive_score=base.positive_score,
        negative_score=base.negative_score,
        target_label=base.target_label,
        rival_label=base.rival_label,
        exactness_residual=np.nan,
        details={"baseline": method, "ranking_only": True},
    )


def _row_from_result(
    *,
    fold: int,
    seed: int,
    sample: ESCSample,
    method: str,
    lambda_hyb: float | str,
    explanation: DynamicFPDEExplanation,
    selection_explanation: DynamicFPDEExplanation,
    curves: dict[str, Any],
    runtime_sec: float,
    random_repetition: int | str = "",
) -> dict[str, object]:
    residual = "" if not np.isfinite(explanation.exactness_residual) else float(explanation.exactness_residual)
    prototype_margin = float(explanation.evidence)
    selection_margin = float(selection_explanation.evidence)
    margin_sign = _margin_sign(prototype_margin)
    selection_margin_sign = _margin_sign(selection_margin)
    evidence_role = "evaluation_margin" if method.endswith("_baseline") else "explanation_margin"
    return {
        "dataset": "esc50",
        "fold": int(fold),
        "seed": int(seed),
        "sample_id": sample.sample_id,
        "true_label": sample.category,
        "target_label": explanation.target_label,
        "rival_label": explanation.rival_label,
        "common_rival_label": selection_explanation.rival_label,
        "method": method,
        "lambda_hyb": lambda_hyb,
        "evidence": float(explanation.evidence),
        "evidence_role": evidence_role,
        "evaluation_evidence": float(explanation.evidence),
        "evaluation_margin": float(explanation.evidence),
        "prototype_margin": prototype_margin,
        "prototype_margin_positive": bool(prototype_margin > 0.0),
        "prototype_margin_sign": margin_sign,
        "selection_margin": selection_margin,
        "selection_margin_positive": bool(selection_margin > 0.0),
        "selection_margin_sign": selection_margin_sign,
        "selection_margin_source": "dynamic_diff",
        "exactness_residual": residual,
        "abs_exactness_residual": "" if residual == "" else abs(float(residual)),
        "deletion_drop_auc": float(curves["deletion_drop_auc"]),
        "insertion_gain_auc": float(curves["insertion_gain_auc"]),
        "combined_score": float(curves["combined_score"]),
        "runtime_sec": float(runtime_sec),
        "T": int(explanation.attributions.shape[0]),
        "F": int(explanation.attributions.shape[1]),
        "random_repetition": random_repetition,
        "aggregation_unit": "sample_repetition" if random_repetition != "" else "sample",
    }


def _margin_sign(value: float) -> str:
    if value > 0.0:
        return "positive"
    if value < 0.0:
        return "negative"
    return "zero"


def _regularized_matrix_norm(values: np.ndarray, eps: float) -> float:
    return float(np.sqrt(np.sum(values * values) + eps))


def _regularized_matrix_cosine(a: np.ndarray, b: np.ndarray, eps: float) -> float:
    return float(np.sum(a * b) / (_regularized_matrix_norm(a, eps) * _regularized_matrix_norm(b, eps)))


def _scale_value(value: float, scale: float) -> float:
    return float(value / scale) if scale > 0.0 else 0.0


def _prototype_index(labels: np.ndarray, label: Any, name: str) -> int:
    matches = np.where(labels == label)[0]
    if matches.size == 0:
        raise ValueError(f"no prototype found for {name}={label!r}")
    return int(matches[0])


def _ensure_cuda_backend() -> None:
    try:
        import cupy as cp  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "CUDA backend requested, but CuPy is not installed. Install a CUDA-matched CuPy package "
            "or the fpde CUDA extra before rerunning with --backend cuda."
        ) from exc
    try:
        from fpde.dynamic_cuda import dynamic_cos_fpde_gpu, dynamic_diff_fpde_gpu, dynamic_hyb_fpde_gpu  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "CUDA backend requested, but fpde.dynamic_cuda could not be imported. "
            "Install fpde from https://github.com/fpde-xai/fpde.git@dynamic with CUDA support."
        ) from exc
    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        raise RuntimeError("CUDA backend requested, but CuPy cannot access a usable CUDA runtime/device.") from exc
    if device_count < 1:
        raise RuntimeError("CUDA backend requested, but CuPy reports no CUDA devices.")


def _ensure_backend(backend: str) -> None:
    if backend == "cuda":
        _ensure_cuda_backend()


def _resolve_dynamic_input(
    X: np.ndarray,
    context: Any,
    *,
    sample: ESCSample,
    target_label: Any,
    rival_label: Any | None,
    eps: float = 1e-12,
) -> _ResolvedDynamicInput:
    X_arr = np.asarray(X, dtype=float)
    target_idx = _prototype_index(context.prototype_labels, target_label, "target_label")
    target_proto = resample_time_series_linear(context.prototypes[target_idx], X_arr.shape[0])
    if rival_label is None:
        candidate_indices = np.where(context.prototype_labels != target_label)[0]
        if candidate_indices.size == 0:
            raise ValueError("rival_label is None, but no non-target prototypes exist")
        distances = []
        for idx in candidate_indices:
            candidate = resample_time_series_linear(context.prototypes[int(idx)], X_arr.shape[0])
            distances.append(float(np.sum((X_arr - candidate) ** 2)))
        rival_idx = int(candidate_indices[int(np.argmin(np.asarray(distances, dtype=float)))])
    else:
        rival_idx = _prototype_index(context.prototype_labels, rival_label, "rival_label")
        if context.prototype_labels[rival_idx] == target_label:
            raise ValueError("rival_label must differ from target_label")
    resolved_rival_label = context.prototype_labels[rival_idx]
    rival_proto = resample_time_series_linear(context.prototypes[rival_idx], X_arr.shape[0])
    anchor = resample_time_series_linear(context.mean_anchor, X_arr.shape[0])

    z = X_arr - anchor
    q_target = target_proto - anchor
    q_rival = rival_proto - anchor
    return _ResolvedDynamicInput(
        sample=sample,
        X=X_arr,
        target_proto=target_proto,
        rival_proto=rival_proto,
        anchor=anchor,
        target_idx=target_idx,
        rival_idx=rival_idx,
        target_label=target_label,
        rival_label=resolved_rival_label,
        diff_positive_score=-float(np.sum((X_arr - target_proto) ** 2)),
        diff_negative_score=-float(np.sum((X_arr - rival_proto) ** 2)),
        cos_positive_score=_regularized_matrix_cosine(z, q_target, eps),
        cos_negative_score=_regularized_matrix_cosine(z, q_rival, eps),
    )


def _scalar_detail(details: dict[str, Any], key: str, index: int | None) -> Any:
    value = details[key]
    if index is None:
        return value
    arr = np.asarray(value)
    if arr.ndim == 0:
        return float(arr)
    return arr[index]


def _explanation_from_tensors(
    resolved: _ResolvedDynamicInput,
    *,
    mode: str,
    attributions: np.ndarray,
    evidence: float,
    diff_evidence: float | None = None,
    cos_evidence: float | None = None,
    hyb_details: dict[str, Any] | None = None,
) -> DynamicFPDEExplanation:
    attr = np.asarray(attributions, dtype=float)
    details: dict[str, Any] = {
        "target_prototype_index": int(resolved.target_idx),
        "rival_prototype_index": int(resolved.rival_idx),
        "anchor_strategy": "mean",
        "eps": 1e-12,
        "diff_positive_score": float(resolved.diff_positive_score),
        "diff_negative_score": float(resolved.diff_negative_score),
        "cos_positive_score": float(resolved.cos_positive_score),
        "cos_negative_score": float(resolved.cos_negative_score),
    }
    if diff_evidence is not None:
        details["diff_evidence"] = float(diff_evidence)
    if cos_evidence is not None:
        details["cos_evidence"] = float(cos_evidence)

    if mode == "dynamic_diff":
        positive_score = resolved.diff_positive_score
        negative_score = resolved.diff_negative_score
    elif mode == "dynamic_cos":
        positive_score = resolved.cos_positive_score
        negative_score = resolved.cos_negative_score
    elif mode == "dynamic_hyb" and hyb_details is not None:
        lambda_value = float(hyb_details["lambda_hyb"])
        diff_scale = float(hyb_details["diff_scale"])
        cos_scale = float(hyb_details["cos_scale"])
        details.update(
            {
                "diff_evidence": float(hyb_details["diff_evidence"]),
                "cos_evidence": float(hyb_details["cos_evidence"]),
                "lambda_hyb": lambda_value,
                "normalize": hyb_details["normalize"],
                "diff_scale": diff_scale,
                "cos_scale": cos_scale,
                "hyb_score_definition": "weighted Diff/Cos scores divided by each component attribution L1 scale",
            }
        )
        positive_score = lambda_value * _scale_value(resolved.diff_positive_score, diff_scale)
        positive_score += (1.0 - lambda_value) * _scale_value(resolved.cos_positive_score, cos_scale)
        negative_score = lambda_value * _scale_value(resolved.diff_negative_score, diff_scale)
        negative_score += (1.0 - lambda_value) * _scale_value(resolved.cos_negative_score, cos_scale)
        details["hyb_positive_score"] = float(positive_score)
        details["hyb_negative_score"] = float(negative_score)
    else:
        raise ValueError(f"unsupported Dynamic-FPDE mode: {mode}")

    time_importance = np.sum(attr, axis=1)
    feature_importance = np.sum(attr, axis=0)
    residual = float(evidence - np.sum(attr))
    return DynamicFPDEExplanation(
        mode=mode,
        evidence=float(evidence),
        attributions=attr.astype(float, copy=True),
        time_importance=time_importance.astype(float, copy=True),
        feature_importance=feature_importance.astype(float, copy=True),
        positive_score=float(positive_score),
        negative_score=float(negative_score),
        target_label=resolved.target_label,
        rival_label=resolved.rival_label,
        exactness_residual=residual,
        details=details,
    )


def _explain_and_score(
    X: np.ndarray,
    context: Any,
    *,
    sample: ESCSample,
    mode: str,
    rival_label: str | None,
    lambda_hyb: float = 0.5,
    steps: int,
) -> tuple[DynamicFPDEExplanation, dict[str, Any], float]:
    start = time.perf_counter()
    explanation = dynamic_fpde_explain_one(
        X,
        context,
        target_label=sample.category,
        rival_label=rival_label,
        mode=mode,
        lambda_hyb=lambda_hyb,
    )
    curves = temporal_deletion_insertion_curves(
        X,
        explanation,
        context,
        target_label=sample.category,
        rival_label=explanation.rival_label,
        steps=steps,
    )
    return explanation, curves, time.perf_counter() - start


def _explain_batch_cuda(
    resolved_items: list[_ResolvedDynamicInput],
    *,
    mode: str,
    lambda_hyb: float = 0.5,
) -> dict[str, tuple[DynamicFPDEExplanation, float]]:
    if not resolved_items:
        return {}
    _ensure_cuda_backend()
    from fpde.dynamic_cuda import dynamic_cos_fpde_gpu, dynamic_diff_fpde_gpu, dynamic_hyb_fpde_gpu

    groups: dict[tuple[int, int], list[_ResolvedDynamicInput]] = {}
    for item in resolved_items:
        groups.setdefault(tuple(item.X.shape), []).append(item)

    out: dict[str, tuple[DynamicFPDEExplanation, float]] = {}
    for group_items in groups.values():
        X_batch = np.stack([item.X for item in group_items], axis=0)
        target_batch = np.stack([item.target_proto for item in group_items], axis=0)
        rival_batch = np.stack([item.rival_proto for item in group_items], axis=0)
        anchor_batch = np.stack([item.anchor for item in group_items], axis=0)

        start = time.perf_counter()
        if mode == "dynamic_diff":
            attrs, evidences = dynamic_diff_fpde_gpu(X_batch, target_batch, rival_batch)
            details = None
        elif mode == "dynamic_cos":
            attrs, evidences = dynamic_cos_fpde_gpu(X_batch, target_batch, rival_batch, anchor=anchor_batch)
            details = None
        elif mode == "dynamic_hyb":
            attrs, evidences, details = dynamic_hyb_fpde_gpu(
                X_batch,
                target_batch,
                rival_batch,
                lambda_hyb=lambda_hyb,
                anchor=anchor_batch,
            )
        else:
            raise ValueError(f"unsupported Dynamic-FPDE mode: {mode}")
        per_sample_runtime = (time.perf_counter() - start) / len(group_items)

        attrs_arr = np.asarray(attrs, dtype=float)
        evidences_arr = np.asarray(evidences, dtype=float)
        for idx, item in enumerate(group_items):
            hyb_details = None
            if details is not None:
                hyb_details = {
                    "diff_evidence": _scalar_detail(details, "diff_evidence", idx),
                    "cos_evidence": _scalar_detail(details, "cos_evidence", idx),
                    "lambda_hyb": details["lambda_hyb"],
                    "normalize": details["normalize"],
                    "diff_scale": _scalar_detail(details, "diff_scale", idx),
                    "cos_scale": _scalar_detail(details, "cos_scale", idx),
                }
            explanation = _explanation_from_tensors(
                item,
                mode=mode,
                attributions=attrs_arr[idx],
                evidence=float(evidences_arr[idx]),
                diff_evidence=float(evidences_arr[idx]) if mode == "dynamic_diff" else None,
                cos_evidence=float(evidences_arr[idx]) if mode == "dynamic_cos" else None,
                hyb_details=hyb_details,
            )
            out[item.sample.sample_id] = (explanation, per_sample_runtime)
    return out


def _explain_batch_cpu(
    resolved_items: list[_ResolvedDynamicInput],
    *,
    mode: str,
    lambda_hyb: float = 0.5,
) -> dict[str, tuple[DynamicFPDEExplanation, float]]:
    out: dict[str, tuple[DynamicFPDEExplanation, float]] = {}
    for item in resolved_items:
        start = time.perf_counter()
        if mode == "dynamic_diff":
            attr, evidence = dynamic_diff_fpde(item.X, item.target_proto, item.rival_proto)
            explanation = _explanation_from_tensors(
                item,
                mode=mode,
                attributions=attr,
                evidence=evidence,
                diff_evidence=evidence,
            )
        elif mode == "dynamic_cos":
            attr, evidence = dynamic_cos_fpde(item.X, item.target_proto, item.rival_proto, anchor=item.anchor)
            explanation = _explanation_from_tensors(
                item,
                mode=mode,
                attributions=attr,
                evidence=evidence,
                cos_evidence=evidence,
            )
        elif mode == "dynamic_hyb":
            attr, evidence, details = dynamic_hyb_fpde(
                item.X,
                item.target_proto,
                item.rival_proto,
                lambda_hyb=lambda_hyb,
                anchor=item.anchor,
            )
            explanation = _explanation_from_tensors(
                item,
                mode=mode,
                attributions=attr,
                evidence=evidence,
                hyb_details=details,
            )
        else:
            raise ValueError(f"unsupported Dynamic-FPDE mode: {mode}")
        out[item.sample.sample_id] = (explanation, time.perf_counter() - start)
    return out


def _prepare_cuda_explanations(
    test_samples: list[ESCSample],
    standardized: dict[str, np.ndarray],
    context: Any,
    *,
    selected_lambda: float,
) -> dict[tuple[str, str], tuple[DynamicFPDEExplanation, float]]:
    selection_inputs = [
        _resolve_dynamic_input(
            standardized[sample.sample_id],
            context,
            sample=sample,
            target_label=sample.category,
            rival_label=None,
        )
        for sample in test_samples
    ]
    diff_explanations = _explain_batch_cuda(selection_inputs, mode="dynamic_diff")
    common_inputs = [
        _resolve_dynamic_input(
            standardized[item.sample.sample_id],
            context,
            sample=item.sample,
            target_label=item.sample.category,
            rival_label=diff_explanations[item.sample.sample_id][0].rival_label,
        )
        for item in selection_inputs
    ]
    # The common-rival Diff explanation is numerically identical to the
    # selection explanation; keep it keyed for the method row without a second
    # GPU launch.
    by_method: dict[tuple[str, str], tuple[DynamicFPDEExplanation, float]] = {
        (sample_id, "selection"): value for sample_id, value in diff_explanations.items()
    }
    by_method.update({(sample_id, "dynamic_diff"): value for sample_id, value in diff_explanations.items()})
    for mode in ("dynamic_cos", "dynamic_hyb"):
        batch = _explain_batch_cuda(common_inputs, mode=mode, lambda_hyb=selected_lambda)
        by_method.update({(sample_id, mode): value for sample_id, value in batch.items()})
    return by_method


def _score_baseline(
    X: np.ndarray,
    context: Any,
    *,
    sample: ESCSample,
    scores: np.ndarray,
    method: str,
    rival_label: str | None,
    steps: int,
) -> tuple[DynamicFPDEExplanation, dict[str, Any], float]:
    start = time.perf_counter()
    base = dynamic_fpde_explain_one(X, context, target_label=sample.category, rival_label=rival_label, mode="dynamic_diff")
    explanation = _baseline_explanation(base, scores, method=method)
    curves = temporal_deletion_insertion_curves(
        X,
        explanation,
        context,
        target_label=sample.category,
        rival_label=explanation.rival_label,
        steps=steps,
    )
    return explanation, curves, time.perf_counter() - start


def _stable_sample_seed(seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    return int(seed) + int(digest[:8], 16)


def _safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)


def _maybe_write_example_figures(output_dir: Path, sample: ESCSample, explanation: DynamicFPDEExplanation, curves: dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
        from fpde import plot_dynamic_attribution_heatmap, plot_dynamic_time_importance

        from experiments.dynamic_fpde_audio.plots import save_deletion_insertion_plot
    except ImportError as exc:
        print(f"Skipping optional example figures: {exc}", file=sys.stderr)
        return
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    sample_id = _safe_filename(sample.sample_id)
    for plotter, stem in (
        (plot_dynamic_time_importance, f"example_time_importance_{sample_id}"),
        (plot_dynamic_attribution_heatmap, f"example_attribution_heatmap_{sample_id}"),
    ):
        ax = plotter(explanation, title=None)
        fig = ax.figure
        fig.tight_layout()
        fig.savefig(figures_dir / f"{stem}.png", dpi=160)
        fig.savefig(figures_dir / f"{stem}.pdf")
        plt.close(fig)
    save_deletion_insertion_plot(curves, figures_dir / f"deletion_insertion_{sample_id}")


def run_fold(
    samples: list[ESCSample],
    *,
    fold: int,
    output_dir: Path,
    mode: str,
    seed: int,
    prototype_length: int | None,
    lambda_grid: list[float],
    feature_config: FeatureConfig,
    skip_errors: bool,
    steps: int,
    random_repetitions: int,
    backend: str = "cpu",
    make_figures: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, str]]]:
    _ensure_backend(backend)
    mode_config = get_mode_config(mode)
    proto_len = prototype_length or mode_config.prototype_length
    train_samples, val_samples, test_samples = split_esc50(samples, fold=fold, mode_config=mode_config, seed=seed)
    all_samples = train_samples + val_samples + test_samples
    raw_features, feature_names, feature_errors = _load_feature_map(
        all_samples,
        output_dir=output_dir,
        config=feature_config,
        skip_errors=skip_errors,
    )
    train_samples = [sample for sample in train_samples if sample.sample_id in raw_features]
    val_samples = [sample for sample in val_samples if sample.sample_id in raw_features]
    test_samples = [sample for sample in test_samples if sample.sample_id in raw_features]
    if not train_samples or not val_samples or not test_samples:
        raise RuntimeError("train, validation, and test splits must all contain extracted features")

    standardizer = fit_standardizer([raw_features[sample.sample_id] for sample in train_samples], feature_names)
    standardized = {
        sample.sample_id: transform_features(raw_features[sample.sample_id], standardizer)
        for sample in all_samples
        if sample.sample_id in raw_features
    }
    context = prepare_dynamic_fpde_context(
        [standardized[sample.sample_id] for sample in train_samples],
        labels_for(train_samples),
        prototype_length=proto_len,
    )
    selection = select_dynamic_lambda(
        [standardized[sample.sample_id] for sample in val_samples],
        labels_for(val_samples),
        context,
        lambda_grid=lambda_grid,
        steps=steps,
    )
    selected_lambda = float(selection["best_lambda"])
    lambda_rows = [{"dataset": "esc50", "fold": int(fold), "seed": int(seed), **row} for row in selection["rows"]]
    sample_rows: list[dict[str, object]] = []
    first_plot_payload: tuple[ESCSample, DynamicFPDEExplanation, dict[str, Any]] | None = None
    cuda_explanations = (
        _prepare_cuda_explanations(test_samples, standardized, context, selected_lambda=selected_lambda)
        if backend == "cuda"
        else {}
    )

    for sample in test_samples:
        X = standardized[sample.sample_id]
        raw_X = raw_features[sample.sample_id]
        if backend == "cuda":
            selection_explanation, _ = cuda_explanations[(sample.sample_id, "selection")]
        else:
            selection_explanation = dynamic_fpde_explain_one(
                X,
                context,
                target_label=sample.category,
                rival_label=None,
                mode="dynamic_diff",
            )
        common_rival_label = selection_explanation.rival_label
        for method in ("dynamic_diff", "dynamic_cos"):
            if backend == "cuda":
                explanation, runtime = cuda_explanations[(sample.sample_id, method)]
                curve_start = time.perf_counter()
                curves = temporal_deletion_insertion_curves(
                    X,
                    explanation,
                    context,
                    target_label=sample.category,
                    rival_label=explanation.rival_label,
                    steps=steps,
                )
                runtime += time.perf_counter() - curve_start
            else:
                explanation, curves, runtime = _explain_and_score(
                    X,
                    context,
                    sample=sample,
                    mode=method,
                    rival_label=common_rival_label,
                    steps=steps,
                )
            sample_rows.append(
                _row_from_result(
                    fold=fold,
                    seed=seed,
                    sample=sample,
                    method=method,
                    lambda_hyb="",
                    explanation=explanation,
                    selection_explanation=selection_explanation,
                    curves=curves,
                    runtime_sec=runtime,
                )
            )

        if backend == "cuda":
            explanation, runtime = cuda_explanations[(sample.sample_id, "dynamic_hyb")]
            curve_start = time.perf_counter()
            curves = temporal_deletion_insertion_curves(
                X,
                explanation,
                context,
                target_label=sample.category,
                rival_label=explanation.rival_label,
                steps=steps,
            )
            runtime += time.perf_counter() - curve_start
        else:
            explanation, curves, runtime = _explain_and_score(
                X,
                context,
                sample=sample,
                mode="dynamic_hyb",
                rival_label=common_rival_label,
                lambda_hyb=selected_lambda,
                steps=steps,
            )
        if first_plot_payload is None:
            first_plot_payload = (sample, explanation, curves)
        sample_rows.append(
            _row_from_result(
                fold=fold,
                seed=seed,
                sample=sample,
                method="dynamic_hyb",
                lambda_hyb=selected_lambda,
                explanation=explanation,
                selection_explanation=selection_explanation,
                curves=curves,
                runtime_sec=runtime,
            )
        )

        energy_scores = energy_frame_scores(raw_X, feature_names)
        explanation, curves, runtime = _score_baseline(
            X,
            context,
            sample=sample,
            scores=energy_scores,
            method="energy_baseline",
            rival_label=common_rival_label,
            steps=steps,
        )
        sample_rows.append(
            _row_from_result(
                fold=fold,
                seed=seed,
                sample=sample,
                method="energy_baseline",
                lambda_hyb="",
                explanation=explanation,
                selection_explanation=selection_explanation,
                curves=curves,
                runtime_sec=runtime,
            )
        )

        for repetition in range(random_repetitions):
            random_scores = random_frame_scores(
                X.shape[0],
                seed=_stable_sample_seed(seed, sample.sample_id),
                repetition=repetition,
            )
            explanation, curves, runtime = _score_baseline(
                X,
                context,
                sample=sample,
                scores=random_scores,
                method="random_baseline",
                rival_label=common_rival_label,
                steps=steps,
            )
            sample_rows.append(
                _row_from_result(
                    fold=fold,
                    seed=seed,
                    sample=sample,
                    method="random_baseline",
                    lambda_hyb="",
                    explanation=explanation,
                    selection_explanation=selection_explanation,
                    curves=curves,
                    runtime_sec=runtime,
                    random_repetition=repetition,
                )
            )

    _write_json(
        output_dir / f"feature_config_fold_{fold}.json",
        {
            "feature_config": asdict(feature_config),
            "feature_names": feature_names,
            "standardizer": standardizer.to_json_dict(),
            "non_finite_policy": "np.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)",
        },
    )
    if first_plot_payload is not None:
        sample, explanation, curves = first_plot_payload
        _write_json(
            output_dir / "cache" / f"example_plot_payload_fold_{fold}.json",
            {
                "sample_id": sample.sample_id,
                "curves": curves,
                "time_importance": explanation.time_importance.tolist(),
            },
        )
        if make_figures:
            _maybe_write_example_figures(output_dir, sample, explanation, curves)
    return sample_rows, lambda_rows, feature_errors


def _maybe_write_figures(output_dir: Path, sample_rows: list[dict[str, object]], lambda_rows: list[dict[str, object]]) -> None:
    try:
        from experiments.dynamic_fpde_audio.plots import save_combined_score_plot, save_lambda_selection_plot
    except ImportError as exc:
        print(f"Skipping optional figures: {exc}", file=sys.stderr)
        return
    summary_rows = aggregate_by_method(average_random_repetitions(sample_rows))
    figures_dir = output_dir / "figures"
    try:
        save_combined_score_plot(summary_rows, figures_dir / "combined_score_by_method")
        save_lambda_selection_plot(lambda_rows, figures_dir / "lambda_selection")
    except ImportError as exc:
        print(f"Skipping optional figures: {exc}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Dynamic-FPDE on ESC-50 frame-level audio features.")
    parser.add_argument("--dataset", default="esc50", choices=["esc50"])
    parser.add_argument("--dataset-root", type=Path, default=Path("data/ESC-50"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dynamic_fpde_esc50_smoke"))
    parser.add_argument("--mode", default="smoke", choices=["smoke", "pilot", "full"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--folds", default=None, help="Comma-separated ESC-50 folds. Overrides --fold.")
    parser.add_argument("--prototype-length", type=int, default=None)
    parser.add_argument("--lambda-grid", default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--n-mfcc", type=int, default=13)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--skip-errors", action="store_true")
    parser.add_argument("--make-figures", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _ensure_backend(args.backend)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = read_esc50_metadata(args.dataset_root)
    folds = parse_folds(args.fold, args.folds)
    lambda_grid = [float(value.strip()) for value in args.lambda_grid.split(",") if value.strip()]
    feature_config = FeatureConfig(sr=args.sr, n_fft=args.n_fft, hop_length=args.hop_length, n_mfcc=args.n_mfcc)
    random_repetitions = 1 if args.mode == "smoke" else 5

    _write_json(
        output_dir / "run_config.json",
        {
            "dataset": args.dataset,
            "dataset_root": args.dataset_root,
            "output_dir": output_dir,
            "mode": args.mode,
            "seed": args.seed,
            "folds": folds,
            "prototype_length": args.prototype_length,
            "lambda_grid": lambda_grid,
            "backend": args.backend,
            "steps": args.steps,
            "random_repetitions": random_repetitions,
            "fpde_source": "git+https://github.com/fpde-xai/fpde.git@dynamic",
        },
    )
    _write_json(
        output_dir / "environment_info.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "fpde": _package_version("fpde"),
            "backend": args.backend,
            "librosa": _package_version("librosa"),
            "soundfile": _package_version("soundfile"),
            "seed": args.seed,
        },
    )
    _write_json(
        output_dir / "feature_config.json",
        {
            "feature_config": asdict(feature_config),
            "feature_set": "rms, zcr, spectral centroid, bandwidth, rolloff, flatness, MFCC 1..n_mfcc",
            "input_space": "frame-level acoustic features, not raw waveform samples",
            "non_finite_policy": "np.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)",
        },
    )

    all_sample_rows: list[dict[str, object]] = []
    all_lambda_rows: list[dict[str, object]] = []
    all_errors: list[dict[str, str]] = []
    for fold in folds:
        sample_rows, lambda_rows, errors = run_fold(
            samples,
            fold=fold,
            output_dir=output_dir,
            mode=args.mode,
            seed=args.seed,
            prototype_length=args.prototype_length,
            lambda_grid=lambda_grid,
            feature_config=feature_config,
            skip_errors=args.skip_errors,
            steps=args.steps,
            random_repetitions=random_repetitions,
            backend=args.backend,
            make_figures=args.make_figures,
        )
        all_sample_rows.extend(sample_rows)
        all_lambda_rows.extend(lambda_rows)
        all_errors.extend(errors)

    results_dir = output_dir / "results"
    summary_rows = average_random_repetitions(all_sample_rows)
    write_csv(results_dir / "dynamic_fpde_sample_metrics.csv", all_sample_rows, SAMPLE_FIELDS)
    write_csv(results_dir / "dynamic_fpde_summary_by_method.csv", aggregate_by_method(summary_rows))
    write_csv(
        results_dir / "dynamic_fpde_summary_positive_margin_by_method.csv",
        aggregate_by_method(positive_margin_rows(summary_rows)),
    )
    write_csv(results_dir / "dynamic_fpde_lambda_selection.csv", all_lambda_rows)
    write_csv(results_dir / "dynamic_fpde_additivity_summary.csv", aggregate_additivity(all_sample_rows))
    if all_errors:
        write_csv(output_dir / "feature_errors.csv", all_errors)
    generate_tables(results_dir, output_dir / "tables")
    if args.make_figures:
        _maybe_write_figures(output_dir, all_sample_rows, all_lambda_rows)
    print(f"Wrote Dynamic-FPDE audio experiment outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
