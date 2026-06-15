"""Run ESC-50 Native-Time Dynamic-FPDE experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

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
from experiments.dynamic_fpde_audio.native_metrics import native_temporal_deletion_insertion_curves
from experiments.dynamic_fpde_audio.native_prototypes import NativePrototype, select_class_exemplar_prototypes, validate_feature_matrix
from experiments.dynamic_fpde_audio.tables import generate_tables


REQUIRED_NATIVE_FPDE_API = (
    "NativeTimeDynamicFPDEExplanation",
    "native_dynamic_diff_fpde",
    "native_dynamic_cos_fpde",
    "native_dynamic_hyb_fpde",
    "native_dynamic_fpde_explain_one",
    "native_dynamic_fpde_explain_batch",
)


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
    "normalize",
    "anchor",
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
    "native_fpde_runtime_sec",
    "diagnostic_runtime_sec",
    "total_runtime_sec",
    "T",
    "F",
    "phi_shape",
    "x_shape",
    "shape_preserved",
    "target_prototype_source_sample_id",
    "target_prototype_source_frame_index",
    "target_prototype_source_time_sec",
    "target_prototype_label",
    "rival_prototype_source_sample_id",
    "rival_prototype_source_frame_index",
    "rival_prototype_source_time_sec",
    "rival_prototype_label",
    "prototype_mode",
    "prototype_selection_rule",
    "random_repetition",
    "aggregation_unit",
]


@dataclass(frozen=True)
class _ResolvedNativeInput:
    sample: ESCSample
    X: np.ndarray
    p_target: np.ndarray
    p_rival: np.ndarray
    anchor: np.ndarray
    target_label: str
    rival_label: str
    target_metadata: dict[str, object]
    rival_metadata: dict[str, object]


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


def _require_native_fpde() -> SimpleNamespace:
    try:
        import fpde
    except ImportError as exc:
        raise RuntimeError(
            "Native-Time Dynamic-FPDE requires fpde from git+https://github.com/fpde-xai/fpde.git@dynamic. "
            "Install the artifact dependencies before running the ESC-50 runner."
        ) from exc
    missing = [name for name in REQUIRED_NATIVE_FPDE_API if not hasattr(fpde, name)]
    if missing:
        raise RuntimeError(
            "Installed fpde package does not expose the Native-Time Dynamic-FPDE API "
            f"required by this artifact: {', '.join(missing)}. Reinstall with "
            '`python -m pip install --force-reinstall "fpde @ git+https://github.com/fpde-xai/fpde.git@dynamic"`.'
        )
    return SimpleNamespace(**{name: getattr(fpde, name) for name in REQUIRED_NATIVE_FPDE_API})


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
        validate_feature_matrix(X)
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise ValueError(f"feature names changed for sample {sample.sample_id}")
        features[sample.sample_id] = X
    if feature_names is None:
        raise RuntimeError("no acoustic feature matrices were loaded")
    return features, feature_names, errors


def _ensure_cuda_backend() -> None:
    try:
        import cupy as cp  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "CUDA backend requested, but CuPy is not installed. Install a CUDA-matched CuPy package "
            "such as cupy-cuda13x before rerunning with --backend cuda."
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


def _margin_sign(value: float) -> str:
    if value > 0.0:
        return "positive"
    if value < 0.0:
        return "negative"
    return "zero"


def _is_ranking_baseline(method: str) -> bool:
    return method in {"energy_baseline_raw", "feature_norm_baseline_standardized", "random_baseline"}


def _stable_sample_seed(seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    return int(seed) + int(digest[:8], 16)


def _safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)


def _class_means(
    train_features: dict[str, np.ndarray],
    train_samples: list[ESCSample],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    by_label: dict[str, list[np.ndarray]] = {}
    all_frames: list[np.ndarray] = []
    for sample in train_samples:
        X = validate_feature_matrix(train_features[sample.sample_id])
        by_label.setdefault(sample.category, []).append(X)
        all_frames.append(X)
    class_means = {label: np.mean(np.vstack(items), axis=0) for label, items in by_label.items()}
    return class_means, np.mean(np.vstack(all_frames), axis=0)


def _anchor_vector(anchor_mode: str, *, target_label: str, class_means: dict[str, np.ndarray], global_mean: np.ndarray, F: int) -> np.ndarray:
    if anchor_mode == "zero":
        return np.zeros(F, dtype=float)
    if anchor_mode == "class_mean":
        return np.asarray(class_means[target_label], dtype=float).copy()
    if anchor_mode == "global_mean":
        return np.asarray(global_mean, dtype=float).copy()
    raise ValueError("anchor must be zero, class_mean, or global_mean")


def _resolve_native_input(
    sample: ESCSample,
    X: np.ndarray,
    prototypes: dict[str, NativePrototype],
    *,
    rival_label: str | None,
    anchor_mode: str,
    class_means: dict[str, np.ndarray],
    global_mean: np.ndarray,
) -> _ResolvedNativeInput:
    X_arr = validate_feature_matrix(X)
    target_label = sample.category
    if target_label not in prototypes:
        raise ValueError(f"no target prototype for label={target_label!r}")
    if rival_label is None:
        distances = {
            label: float(np.mean(np.sum((X_arr - proto.vector) ** 2, axis=1)))
            for label, proto in prototypes.items()
            if label != target_label
        }
        if not distances:
            raise ValueError("at least one non-target prototype is required")
        resolved_rival_label = min(distances, key=distances.get)
    else:
        resolved_rival_label = str(rival_label)
        if resolved_rival_label == target_label:
            raise ValueError("rival_label must differ from target_label")
        if resolved_rival_label not in prototypes:
            raise ValueError(f"no rival prototype for label={resolved_rival_label!r}")

    p_target = np.asarray(prototypes[target_label].vector, dtype=float)
    p_rival = np.asarray(prototypes[resolved_rival_label].vector, dtype=float)
    F = X_arr.shape[1]
    if p_target.shape != (F,):
        raise ValueError(f"target prototype has shape {p_target.shape}, expected {(F,)}")
    if p_rival.shape != (F,):
        raise ValueError(f"rival prototype has shape {p_rival.shape}, expected {(F,)}")
    anchor = _anchor_vector(anchor_mode, target_label=target_label, class_means=class_means, global_mean=global_mean, F=F)
    if anchor.shape != (F,):
        raise ValueError(f"anchor has shape {anchor.shape}, expected {(F,)}")
    return _ResolvedNativeInput(
        sample=sample,
        X=X_arr,
        p_target=p_target,
        p_rival=p_rival,
        anchor=anchor,
        target_label=target_label,
        rival_label=resolved_rival_label,
        target_metadata=dict(prototypes[target_label].metadata),
        rival_metadata=dict(prototypes[resolved_rival_label].metadata),
    )


def _explain_one_cpu(
    api: SimpleNamespace,
    item: _ResolvedNativeInput,
    *,
    mode: str,
    lambda_hyb: float,
    normalize: str,
    feature_names: list[str],
    feature_config: FeatureConfig,
) -> Any:
    timestamps = [feature_config.frame_time_sec(i) for i in range(item.X.shape[0])]
    details = {
        "target_prototype_metadata": item.target_metadata,
        "rival_prototype_metadata": item.rival_metadata,
        "anchor": "zero" if np.allclose(item.anchor, 0.0) else "provided_vector",
    }
    return api.native_dynamic_fpde_explain_one(
        item.X,
        p_target=item.p_target,
        p_rival=item.p_rival,
        target_label=item.target_label,
        rival_label=item.rival_label,
        mode=mode,
        lambda_hyb=float(lambda_hyb),
        normalize=normalize,
        anchor=item.anchor,
        feature_names=feature_names,
        timestamps_sec=timestamps,
        details=details,
    )


def _native_diff_fpde_cuda(X_batch: np.ndarray, target_batch: np.ndarray, rival_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import cupy as cp  # type: ignore[import-not-found]

    X_gpu = cp.asarray(X_batch, dtype=cp.float64)
    target_gpu = cp.asarray(target_batch, dtype=cp.float64)[:, None, :]
    rival_gpu = cp.asarray(rival_batch, dtype=cp.float64)[:, None, :]
    attrs = (X_gpu - rival_gpu) ** 2 - (X_gpu - target_gpu) ** 2
    evidences = cp.sum(attrs, axis=(1, 2))
    return cp.asnumpy(attrs), cp.asnumpy(evidences)


def _native_cos_fpde_cuda(
    X_batch: np.ndarray,
    target_batch: np.ndarray,
    rival_batch: np.ndarray,
    anchor_batch: np.ndarray,
    *,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    import cupy as cp  # type: ignore[import-not-found]

    X_gpu = cp.asarray(X_batch, dtype=cp.float64)
    target_gpu = cp.asarray(target_batch, dtype=cp.float64)
    rival_gpu = cp.asarray(rival_batch, dtype=cp.float64)
    anchor_gpu = cp.asarray(anchor_batch, dtype=cp.float64)
    z = X_gpu - anchor_gpu[:, None, :]
    q_target = target_gpu - anchor_gpu
    q_rival = rival_gpu - anchor_gpu
    z_norm = cp.linalg.norm(z, axis=2, keepdims=True)
    target_norm = cp.linalg.norm(q_target, axis=1)[:, None, None]
    rival_norm = cp.linalg.norm(q_rival, axis=1)[:, None, None]
    attrs = (z * q_target[:, None, :]) / ((z_norm + float(eps)) * (target_norm + float(eps)))
    attrs -= (z * q_rival[:, None, :]) / ((z_norm + float(eps)) * (rival_norm + float(eps)))
    evidences = cp.sum(attrs, axis=(1, 2))
    return cp.asnumpy(attrs), cp.asnumpy(evidences)


def _native_hyb_fpde_cuda(
    X_batch: np.ndarray,
    target_batch: np.ndarray,
    rival_batch: np.ndarray,
    anchor_batch: np.ndarray,
    *,
    lambda_hyb: float,
    normalize: str,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import cupy as cp  # type: ignore[import-not-found]

    diff_attr, diff_evidence = _native_diff_fpde_cuda(X_batch, target_batch, rival_batch)
    cos_attr, cos_evidence = _native_cos_fpde_cuda(X_batch, target_batch, rival_batch, anchor_batch, eps=eps)
    diff_gpu = cp.asarray(diff_attr, dtype=cp.float64)
    cos_gpu = cp.asarray(cos_attr, dtype=cp.float64)
    if normalize == "l1":
        diff_scale = cp.sum(cp.abs(diff_gpu), axis=(1, 2), keepdims=True) + float(eps)
        cos_scale = cp.sum(cp.abs(cos_gpu), axis=(1, 2), keepdims=True) + float(eps)
        diff_part = diff_gpu / diff_scale
        cos_part = cos_gpu / cos_scale
    elif normalize == "none":
        diff_scale = cp.ones((diff_gpu.shape[0], 1, 1), dtype=cp.float64)
        cos_scale = cp.ones((cos_gpu.shape[0], 1, 1), dtype=cp.float64)
        diff_part = diff_gpu
        cos_part = cos_gpu
    else:
        raise ValueError("normalize must be none or l1")
    attrs = float(lambda_hyb) * diff_part + (1.0 - float(lambda_hyb)) * cos_part
    evidences = cp.sum(attrs, axis=(1, 2))
    return (
        cp.asnumpy(attrs),
        cp.asnumpy(evidences),
        {
            "diff_evidence": np.asarray(diff_evidence, dtype=float),
            "cos_evidence": np.asarray(cos_evidence, dtype=float),
            "diff_scale": cp.asnumpy(diff_scale.reshape(-1)),
            "cos_scale": cp.asnumpy(cos_scale.reshape(-1)),
        },
    )


def _explanation_from_cuda(
    api: SimpleNamespace,
    item: _ResolvedNativeInput,
    *,
    mode: str,
    attributions: np.ndarray,
    evidence: float,
    lambda_hyb: float,
    normalize: str,
    hyb_details: dict[str, Any] | None = None,
) -> Any:
    attr = np.asarray(attributions, dtype=float)
    time_importance = np.sum(attr, axis=1)
    feature_importance = np.sum(attr, axis=0)
    details: dict[str, Any] = {
        "time_mode": "native",
        "temporal_resampling": False,
        "temporal_pooling": False,
        "prototype_kind": "feature_vector",
        "input_shape": tuple(item.X.shape),
        "output_shape": tuple(attr.shape),
        "target_prototype_metadata": item.target_metadata,
        "rival_prototype_metadata": item.rival_metadata,
        "normalize": normalize,
        "lambda_hyb": float(lambda_hyb) if mode == "dynamic_hyb" else None,
        "backend": "cuda",
    }
    if hyb_details:
        details.update(hyb_details)
    diff_positive = -float(np.sum((item.X - item.p_target) ** 2))
    diff_negative = -float(np.sum((item.X - item.p_rival) ** 2))
    positive_score = diff_positive
    negative_score = diff_negative
    if mode == "dynamic_cos":
        positive_score = float(np.sum(attr + 0.0))
        negative_score = 0.0
    elif mode == "dynamic_hyb" and normalize == "none":
        positive_score = float(lambda_hyb) * diff_positive
        negative_score = float(lambda_hyb) * diff_negative
    return api.NativeTimeDynamicFPDEExplanation(
        mode=mode,
        evidence=float(evidence),
        attributions=attr.astype(float, copy=True),
        time_importance=time_importance.astype(float, copy=True),
        feature_importance=feature_importance.astype(float, copy=True),
        positive_score=float(positive_score),
        negative_score=float(negative_score),
        target_label=item.target_label,
        rival_label=item.rival_label,
        exactness_residual=float(evidence - np.sum(attr)),
        details=details,
    )


def _explain_batch_cuda(
    api: SimpleNamespace,
    items: list[_ResolvedNativeInput],
    *,
    mode: str,
    lambda_hyb: float,
    normalize: str,
) -> dict[str, tuple[Any, float]]:
    if not items:
        return {}
    _ensure_cuda_backend()
    groups: dict[tuple[int, int], list[_ResolvedNativeInput]] = {}
    for item in items:
        groups.setdefault(tuple(item.X.shape), []).append(item)
    out: dict[str, tuple[Any, float]] = {}
    for group_items in groups.values():
        X_batch = np.stack([item.X for item in group_items], axis=0)
        target_batch = np.stack([item.p_target for item in group_items], axis=0)
        rival_batch = np.stack([item.p_rival for item in group_items], axis=0)
        anchor_batch = np.stack([item.anchor for item in group_items], axis=0)
        start = time.perf_counter()
        if mode == "dynamic_diff":
            attrs, evidences = _native_diff_fpde_cuda(X_batch, target_batch, rival_batch)
            details = None
        elif mode == "dynamic_cos":
            attrs, evidences = _native_cos_fpde_cuda(X_batch, target_batch, rival_batch, anchor_batch)
            details = None
        elif mode == "dynamic_hyb":
            attrs, evidences, details = _native_hyb_fpde_cuda(
                X_batch,
                target_batch,
                rival_batch,
                anchor_batch,
                lambda_hyb=lambda_hyb,
                normalize=normalize,
            )
        else:
            raise ValueError(f"unsupported Native-Time Dynamic-FPDE mode: {mode}")
        per_sample_runtime = (time.perf_counter() - start) / len(group_items)
        for idx, item in enumerate(group_items):
            hyb_details = None
            if details is not None:
                hyb_details = {
                    "diff_evidence": float(details["diff_evidence"][idx]),
                    "cos_evidence": float(details["cos_evidence"][idx]),
                    "diff_scale": float(details["diff_scale"][idx]),
                    "cos_scale": float(details["cos_scale"][idx]),
                }
            out[item.sample.sample_id] = (
                _explanation_from_cuda(
                    api,
                    item,
                    mode=mode,
                    attributions=np.asarray(attrs[idx], dtype=float),
                    evidence=float(np.asarray(evidences)[idx]),
                    lambda_hyb=lambda_hyb,
                    normalize=normalize,
                    hyb_details=hyb_details,
                ),
                per_sample_runtime,
            )
    return out


def _explain_many_cuda(
    api: SimpleNamespace,
    items: list[_ResolvedNativeInput],
    *,
    mode: str,
    lambda_hyb: float,
    normalize: str,
) -> dict[str, tuple[Any, float]]:
    """Explain all items with CUDA, grouped only by naturally equal ``(T, F)``.

    This wrapper exists so the main runner and tests can assert that CUDA is
    launched over the full resolved test set rather than one sample at a time.
    The underlying batch helper still refuses to pad or resample; it stacks
    only items that already have identical shapes.
    """

    return _explain_batch_cuda(api, items, mode=mode, lambda_hyb=lambda_hyb, normalize=normalize)


def _baseline_explanation(api: SimpleNamespace, base: Any, scores: np.ndarray, *, method: str) -> Any:
    return api.NativeTimeDynamicFPDEExplanation(
        mode=method,
        evidence=float(base.evidence),
        attributions=np.zeros_like(base.attributions),
        time_importance=np.asarray(scores, dtype=float),
        feature_importance=np.zeros(base.attributions.shape[1], dtype=float),
        positive_score=float(base.positive_score),
        negative_score=float(base.negative_score),
        target_label=base.target_label,
        rival_label=base.rival_label,
        exactness_residual=float("nan"),
        details={"baseline": method, "ranking_only": True, "time_mode": "native", "prototype_kind": "feature_vector"},
    )


def _validate_native_output(X: np.ndarray, explanation: Any, item: _ResolvedNativeInput) -> None:
    phi = np.asarray(explanation.attributions, dtype=float)
    if phi.shape != X.shape:
        raise ValueError(f"Native-Time Phi shape must equal X shape, got Phi={phi.shape}, X={X.shape}")
    if item.p_target.shape != (X.shape[1],):
        raise ValueError("target prototype must have shape (F,)")
    if item.p_rival.shape != (X.shape[1],):
        raise ValueError("rival prototype must have shape (F,)")
    if item.anchor.shape != (X.shape[1],):
        raise ValueError("anchor must have shape (F,)")
    arrays = [X, phi, item.p_target, item.p_rival, item.anchor, np.asarray([explanation.evidence], dtype=float)]
    if not all(np.all(np.isfinite(arr)) for arr in arrays):
        raise ValueError("Native-Time result contains NaN or inf")
    details = getattr(explanation, "details", {})
    if "resampled_length" in details or "prototype_length" in details:
        raise ValueError("Native-Time results must not contain resampled_length or prototype_length fields")


def _row_from_result(
    *,
    fold: int,
    seed: int,
    item: _ResolvedNativeInput,
    method: str,
    lambda_hyb: float | str,
    normalize: str,
    anchor: str,
    explanation: Any,
    selection_explanation: Any,
    curves: dict[str, Any],
    native_fpde_runtime_sec: float,
    diagnostic_runtime_sec: float,
    random_repetition: int | str = "",
) -> dict[str, object]:
    _validate_native_output(item.X, explanation, item)
    residual_value = float(explanation.exactness_residual)
    residual: float | str = "" if not np.isfinite(residual_value) else residual_value
    prototype_margin = float(explanation.evidence)
    selection_margin = float(selection_explanation.evidence)
    evidence_role = "evaluation_margin" if _is_ranking_baseline(method) else "explanation_margin"
    target_meta = item.target_metadata
    rival_meta = item.rival_metadata
    total_runtime_sec = float(native_fpde_runtime_sec) + float(diagnostic_runtime_sec)
    row = {
        "dataset": "esc50",
        "fold": int(fold),
        "seed": int(seed),
        "sample_id": item.sample.sample_id,
        "true_label": item.sample.category,
        "target_label": explanation.target_label,
        "rival_label": explanation.rival_label,
        "common_rival_label": selection_explanation.rival_label,
        "method": method,
        "lambda_hyb": lambda_hyb,
        "normalize": normalize,
        "anchor": anchor,
        "evidence": prototype_margin,
        "evidence_role": evidence_role,
        "evaluation_evidence": prototype_margin,
        "evaluation_margin": prototype_margin,
        "prototype_margin": prototype_margin,
        "prototype_margin_positive": bool(prototype_margin > 0.0),
        "prototype_margin_sign": _margin_sign(prototype_margin),
        "selection_margin": selection_margin,
        "selection_margin_positive": bool(selection_margin > 0.0),
        "selection_margin_sign": _margin_sign(selection_margin),
        "selection_margin_source": "native_dynamic_diff",
        "exactness_residual": residual,
        "abs_exactness_residual": "" if residual == "" else abs(float(residual)),
        "deletion_drop_auc": float(curves["deletion_drop_auc"]),
        "insertion_gain_auc": float(curves["insertion_gain_auc"]),
        "combined_score": float(curves["combined_score"]),
        "runtime_sec": total_runtime_sec,
        "native_fpde_runtime_sec": float(native_fpde_runtime_sec),
        "diagnostic_runtime_sec": float(diagnostic_runtime_sec),
        "total_runtime_sec": total_runtime_sec,
        "T": int(item.X.shape[0]),
        "F": int(item.X.shape[1]),
        "phi_shape": str(tuple(np.asarray(explanation.attributions).shape)),
        "x_shape": str(tuple(item.X.shape)),
        "shape_preserved": bool(np.asarray(explanation.attributions).shape == item.X.shape),
        "target_prototype_source_sample_id": target_meta.get("source_sample_id", ""),
        "target_prototype_source_frame_index": target_meta.get("source_frame_index", ""),
        "target_prototype_source_time_sec": target_meta.get("source_time_sec", ""),
        "target_prototype_label": target_meta.get("label", ""),
        "rival_prototype_source_sample_id": rival_meta.get("source_sample_id", ""),
        "rival_prototype_source_frame_index": rival_meta.get("source_frame_index", ""),
        "rival_prototype_source_time_sec": rival_meta.get("source_time_sec", ""),
        "rival_prototype_label": rival_meta.get("label", ""),
        "prototype_mode": target_meta.get("prototype_mode", "selected_exemplar"),
        "prototype_selection_rule": target_meta.get("selection_rule", ""),
        "random_repetition": random_repetition,
        "aggregation_unit": "sample_repetition" if random_repetition != "" else "sample",
    }
    if "resampled_length" in row or "prototype_length" in row:
        raise ValueError("Native-Time CSV rows must not contain resampled_length or prototype_length")
    return row


def _select_lambda(
    api: SimpleNamespace,
    val_samples: list[ESCSample],
    standardized: dict[str, np.ndarray],
    prototypes: dict[str, NativePrototype],
    *,
    lambda_grid: list[float] | None,
    lambda_hyb: float,
    normalize: str,
    anchor: str,
    class_means: dict[str, np.ndarray],
    global_mean: np.ndarray,
    feature_names: list[str],
    feature_config: FeatureConfig,
    steps: int,
) -> tuple[float, list[dict[str, object]]]:
    candidates = list(lambda_grid) if lambda_grid else [float(lambda_hyb)]
    rows: list[dict[str, object]] = []
    best_lambda = float(candidates[0])
    best_score = -float("inf")
    for candidate in candidates:
        scores: list[float] = []
        deletion_scores: list[float] = []
        insertion_scores: list[float] = []
        for sample in val_samples:
            selection_item = _resolve_native_input(
                sample,
                standardized[sample.sample_id],
                prototypes,
                rival_label=None,
                anchor_mode=anchor,
                class_means=class_means,
                global_mean=global_mean,
            )
            selection = _explain_one_cpu(
                api,
                selection_item,
                mode="dynamic_diff",
                lambda_hyb=candidate,
                normalize=normalize,
                feature_names=feature_names,
                feature_config=feature_config,
            )
            item = _resolve_native_input(
                sample,
                standardized[sample.sample_id],
                prototypes,
                rival_label=selection.rival_label,
                anchor_mode=anchor,
                class_means=class_means,
                global_mean=global_mean,
            )
            exp = _explain_one_cpu(
                api,
                item,
                mode="dynamic_hyb",
                lambda_hyb=candidate,
                normalize=normalize,
                feature_names=feature_names,
                feature_config=feature_config,
            )
            curves = native_temporal_deletion_insertion_curves(exp, steps=steps)
            scores.append(float(curves["combined_score"]))
            deletion_scores.append(float(curves["deletion_drop_auc"]))
            insertion_scores.append(float(curves["insertion_gain_auc"]))
        mean_score = float(np.mean(scores)) if scores else float("nan")
        rows.append(
            {
                "lambda_hyb": float(candidate),
                "mean_combined_score": mean_score,
                "mean_deletion_drop_auc": float(np.mean(deletion_scores)) if deletion_scores else "",
                "mean_insertion_gain_auc": float(np.mean(insertion_scores)) if insertion_scores else "",
                "n_eval_samples": len(scores),
                "selection_mode": "fixed" if not lambda_grid else "validation_grid",
            }
        )
        if mean_score > best_score:
            best_score = mean_score
            best_lambda = float(candidate)
    return best_lambda, rows


def _maybe_write_example_figures(output_dir: Path, sample: ESCSample, explanation: Any, curves: dict[str, Any]) -> None:
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
    lambda_hyb: float,
    lambda_grid: list[float] | None,
    feature_config: FeatureConfig,
    prototype_mode: str,
    prototype_selection: str,
    anchor: str,
    normalize: str,
    skip_errors: bool,
    steps: int,
    random_repetitions: int,
    backend: str = "cpu",
    make_figures: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, str]]]:
    if prototype_mode != "exemplar":
        raise ValueError("Native-Time Dynamic-FPDE currently supports only --prototype-mode exemplar")
    api = _require_native_fpde()
    _ensure_backend(backend)
    mode_config = get_mode_config(mode)
    train_samples, val_samples, test_samples = split_esc50(samples, fold=fold, mode_config=mode_config, seed=seed)
    all_samples = train_samples + val_samples + test_samples
    feature_map, feature_names, feature_errors = _load_feature_map(
        all_samples,
        output_dir=output_dir,
        config=feature_config,
        skip_errors=skip_errors,
    )
    train_samples = [sample for sample in train_samples if sample.sample_id in feature_map]
    val_samples = [sample for sample in val_samples if sample.sample_id in feature_map]
    test_samples = [sample for sample in test_samples if sample.sample_id in feature_map]
    if not train_samples or not val_samples or not test_samples:
        raise RuntimeError("train, validation, and test splits must all contain loaded acoustic feature matrices")

    standardizer = fit_standardizer([feature_map[sample.sample_id] for sample in train_samples], feature_names)
    standardized = {
        sample.sample_id: transform_features(feature_map[sample.sample_id], standardizer)
        for sample in all_samples
        if sample.sample_id in feature_map
    }
    train_feature_map = {sample.sample_id: standardized[sample.sample_id] for sample in train_samples}
    train_label_map = {sample.sample_id: sample.category for sample in train_samples}
    prototypes = select_class_exemplar_prototypes(
        train_feature_map,
        train_label_map,
        selection_rule=prototype_selection,
        feature_config=feature_config,
        feature_names=feature_names,
    )
    class_means, global_mean = _class_means(train_feature_map, train_samples)
    selected_lambda, selection_rows = _select_lambda(
        api,
        val_samples,
        standardized,
        prototypes,
        lambda_grid=lambda_grid,
        lambda_hyb=lambda_hyb,
        normalize=normalize,
        anchor=anchor,
        class_means=class_means,
        global_mean=global_mean,
        feature_names=feature_names,
        feature_config=feature_config,
        steps=steps,
    )
    lambda_rows = [{"dataset": "esc50", "fold": int(fold), "seed": int(seed), **row} for row in selection_rows]

    sample_rows: list[dict[str, object]] = []
    first_plot_payload: tuple[ESCSample, Any, dict[str, Any]] | None = None
    selection_items = [
        _resolve_native_input(
            sample,
            standardized[sample.sample_id],
            prototypes,
            rival_label=None,
            anchor_mode=anchor,
            class_means=class_means,
            global_mean=global_mean,
        )
        for sample in test_samples
    ]
    if backend == "cuda":
        selection_results = _explain_many_cuda(
            api,
            selection_items,
            mode="dynamic_diff",
            lambda_hyb=selected_lambda,
            normalize=normalize,
        )
    else:
        selection_results = {}
        for item in selection_items:
            start = time.perf_counter()
            selection_results[item.sample.sample_id] = (
                _explain_one_cpu(
                    api,
                    item,
                    mode="dynamic_diff",
                    lambda_hyb=selected_lambda,
                    normalize=normalize,
                    feature_names=feature_names,
                    feature_config=feature_config,
                ),
                time.perf_counter() - start,
            )

    items = [
        _resolve_native_input(
            item.sample,
            standardized[item.sample.sample_id],
            prototypes,
            rival_label=selection_results[item.sample.sample_id][0].rival_label,
            anchor_mode=anchor,
            class_means=class_means,
            global_mean=global_mean,
        )
        for item in selection_items
    ]
    fpde_results: dict[tuple[str, str], tuple[Any, float]] = {}
    for method, fpde_mode in (("dynamic_diff", "dynamic_diff"), ("dynamic_cos", "dynamic_cos"), ("dynamic_hyb", "dynamic_hyb")):
        if backend == "cuda":
            batch_results = _explain_many_cuda(
                api,
                items,
                mode=fpde_mode,
                lambda_hyb=selected_lambda,
                normalize=normalize,
            )
            fpde_results.update({(sample_id, method): value for sample_id, value in batch_results.items()})
        else:
            for item in items:
                start = time.perf_counter()
                fpde_results[(item.sample.sample_id, method)] = (
                    _explain_one_cpu(
                        api,
                        item,
                        mode=fpde_mode,
                        lambda_hyb=selected_lambda,
                        normalize=normalize,
                        feature_names=feature_names,
                        feature_config=feature_config,
                    ),
                    time.perf_counter() - start,
                )

    for item in items:
        sample = item.sample
        selection_explanation = selection_results[sample.sample_id][0]

        for method, fpde_mode in (("dynamic_diff", "dynamic_diff"), ("dynamic_cos", "dynamic_cos"), ("dynamic_hyb", "dynamic_hyb")):
            explanation, native_runtime = fpde_results[(sample.sample_id, method)]
            diagnostic_start = time.perf_counter()
            curves = native_temporal_deletion_insertion_curves(explanation, steps=steps)
            diagnostic_runtime = time.perf_counter() - diagnostic_start
            if method == "dynamic_hyb" and first_plot_payload is None:
                first_plot_payload = (sample, explanation, curves)
            sample_rows.append(
                _row_from_result(
                    fold=fold,
                    seed=seed,
                    item=item,
                    method=method,
                    lambda_hyb=selected_lambda if method == "dynamic_hyb" else "",
                    normalize=normalize,
                    anchor=anchor,
                    explanation=explanation,
                    selection_explanation=selection_explanation,
                    curves=curves,
                    native_fpde_runtime_sec=native_runtime,
                    diagnostic_runtime_sec=diagnostic_runtime,
                )
            )

        for method, scores_iter in (
            ("energy_baseline_raw", [energy_frame_scores(feature_map[sample.sample_id], feature_names)]),
            ("feature_norm_baseline_standardized", [energy_frame_scores(item.X, feature_names)]),
            (
                "random_baseline",
                [
                    random_frame_scores(
                        item.X.shape[0],
                        seed=_stable_sample_seed(seed, sample.sample_id),
                        repetition=repetition,
                    )
                    for repetition in range(random_repetitions)
                ],
            ),
        ):
            for repetition, scores in enumerate(scores_iter):
                start = time.perf_counter()
                explanation = _baseline_explanation(api, selection_explanation, scores, method=method)
                native_runtime = time.perf_counter() - start
                diagnostic_start = time.perf_counter()
                curves = native_temporal_deletion_insertion_curves(
                    explanation,
                    ranking_scores=scores,
                    evidence_by_frame=selection_explanation.time_importance,
                    steps=steps,
                )
                diagnostic_runtime = time.perf_counter() - diagnostic_start
                sample_rows.append(
                    _row_from_result(
                        fold=fold,
                        seed=seed,
                        item=item,
                        method=method,
                        lambda_hyb="",
                        normalize=normalize,
                        anchor=anchor,
                        explanation=explanation,
                        selection_explanation=selection_explanation,
                        curves=curves,
                        native_fpde_runtime_sec=native_runtime,
                        diagnostic_runtime_sec=diagnostic_runtime,
                        random_repetition=repetition if method == "random_baseline" else "",
                    )
                )

    _write_json(
        output_dir / f"native_time_feature_config_fold_{fold}.json",
        {
            "feature_config": asdict(feature_config),
            "feature_names": feature_names,
            "standardizer": standardizer.to_json_dict(),
            "prototype_metadata": {label: proto.metadata for label, proto in prototypes.items()},
            "input_space": "frame-level acoustic feature matrices",
            "primary_output": "Phi with shape equal to X for each clip",
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
                "phi_shape": tuple(explanation.attributions.shape),
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


class _LegacyUnsupportedAction(argparse.Action):
    def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace, values: Any, option_string: str | None = None) -> None:
        parser.error(f"{option_string} is legacy resampled-time Dynamic-FPDE and is unsupported in Native-Time mode")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Native-Time Dynamic-FPDE on ESC-50 frame-level acoustic features.")
    parser.add_argument("--dataset", default="esc50", choices=["esc50"])
    parser.add_argument("--dataset-root", type=Path, default=Path("data/ESC-50"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/native_time_dynamic_fpde_esc50_smoke"))
    parser.add_argument("--mode", default="smoke", choices=["smoke", "pilot", "full"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--folds", default=None, help="Comma-separated ESC-50 folds. Overrides --fold.")
    parser.add_argument("--prototype-mode", default="exemplar", choices=["exemplar"])
    parser.add_argument("--prototype-selection", default="nearest_to_class_centroid_frame", choices=["medoid_frame", "nearest_to_class_centroid_frame"])
    parser.add_argument("--anchor", default="zero", choices=["zero", "class_mean", "global_mean"])
    parser.add_argument("--normalize", default="none", choices=["none", "l1"])
    parser.add_argument("--lambda-hyb", type=float, default=0.5)
    parser.add_argument("--lambda-grid", default=None, help="Optional comma-separated validation grid for lambda_hyb.")
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--target-sr", "--sr", type=int, default=16000)
    parser.add_argument("--frame-length", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--skip-errors", action="store_true")
    parser.add_argument("--make-figures", action="store_true")
    parser.add_argument("--prototype-length", nargs="?", action=_LegacyUnsupportedAction, help=argparse.SUPPRESS)
    parser.add_argument("--duration-sec", nargs="?", action=_LegacyUnsupportedAction, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0.0 <= float(args.lambda_hyb) <= 1.0:
        raise ValueError("--lambda-hyb must be in [0, 1]")
    _ensure_backend(args.backend)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = read_esc50_metadata(args.dataset_root)
    folds = parse_folds(args.fold, args.folds)
    lambda_grid = [float(value.strip()) for value in args.lambda_grid.split(",") if value.strip()] if args.lambda_grid else None
    if lambda_grid and any(value < 0.0 or value > 1.0 for value in lambda_grid):
        raise ValueError("--lambda-grid values must be in [0, 1]")
    feature_config = FeatureConfig(target_sr=args.target_sr, frame_length=args.frame_length, hop_length=args.hop_length)
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
            "prototype_mode": args.prototype_mode,
            "prototype_selection": args.prototype_selection,
            "anchor": args.anchor,
            "normalize": args.normalize,
            "lambda_hyb": args.lambda_hyb,
            "lambda_grid": lambda_grid,
            "backend": args.backend,
            "steps": args.steps,
            "random_repetitions": random_repetitions,
            "fpde_source": "git+https://github.com/fpde-xai/fpde.git@dynamic",
            "time_mode": "native",
            "temporal_resampling": False,
            "temporal_pooling": False,
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
            "soundfile": _package_version("soundfile"),
            "seed": args.seed,
        },
    )
    _write_json(
        output_dir / "native_time_feature_config.json",
        {
            "feature_config": asdict(feature_config),
            "input_space": "frame-level acoustic feature matrices, not raw waveform samples",
            "primary_output": "Phi_i in R^{T_i x F} with Phi_i.shape == X_i.shape",
            "allowed_preprocessing": "decode audio, resample audio to target_sr, convert to mono, convert to frame-level features, standardize features",
            "forbidden_preprocessing": "fixed-length temporal resampling, temporal pooling, padding variable-length samples into a dense tensor, DTW alignment",
            "sampling_rate_invariance": "Dynamic-FPDE itself is not sampling-rate invariant; audio is normalized to target_sr before feature extraction",
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
            lambda_hyb=args.lambda_hyb,
            lambda_grid=lambda_grid,
            feature_config=feature_config,
            prototype_mode=args.prototype_mode,
            prototype_selection=args.prototype_selection,
            anchor=args.anchor,
            normalize=args.normalize,
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
    print(f"Wrote Native-Time Dynamic-FPDE audio experiment outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
