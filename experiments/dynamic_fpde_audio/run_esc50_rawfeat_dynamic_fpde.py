"""Run RawFeat Dynamic-FPDE on variable-length ESC-50 audio."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.dynamic_fpde_audio.aggregate import (  # noqa: E402
    aggregate_rawfeat_generation,
    aggregate_rawfeat_samples,
    write_csv,
)
from experiments.dynamic_fpde_audio.datasets import (  # noqa: E402
    ESCSample,
    get_mode_config,
    parse_folds,
    read_esc50_metadata,
    split_esc50,
)
from experiments.dynamic_fpde_audio.features import FeatureConfig  # noqa: E402
from experiments.dynamic_fpde_audio.rawfeat_representation import (  # noqa: E402
    build_rawfeat_input,
    overlap_add_frames,
)


SAMPLE_FIELDS = [
    "dataset",
    "fold",
    "seed",
    "sample_id",
    "category",
    "method",
    "lambda_hyb",
    "normalize",
    "target_label",
    "rival_label",
    "evidence",
    "absolute_evidence",
    "attribution_sum",
    "exactness_residual",
    "abs_exactness_residual",
    "audit_passed",
    "raw_group_attribution",
    "feature_group_attribution",
    "dt_group_attribution",
    "raw_shape",
    "feature_shape",
    "attribution_shape",
    "time_importance_shape",
    "shape_match",
    "T",
    "C_raw",
    "F",
    "runtime_sec",
]

GENERATION_FIELDS = [
    "dataset",
    "fold",
    "seed",
    "sample_id",
    "target_label",
    "rival_label",
    "lambda_hyb",
    "noise_scale",
    "summary_scaling",
    "generated_wav",
    "generated_raw_shape",
    "reprocessed_raw_shape",
    "reprocessed_feature_shape",
    "shape_match",
    "generated_evidence",
    "generated_absolute_evidence",
    "generated_attribution_sum",
    "generated_exactness_residual",
    "generated_abs_exactness_residual",
    "generated_audit_passed",
    "selected_neighbor_index",
    "selected_neighbor_distance",
]


def _shape(value: np.ndarray) -> str:
    return "x".join(str(int(part)) for part in value.shape)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, slice):
        return {"start": value.start, "stop": value.stop, "step": value.step}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_sample_summary(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(row)


def _load_inputs(
    samples: Sequence[ESCSample], feature_config: FeatureConfig
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]], list[dict[str, Any]]]:
    inputs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]] = {}
    errors: list[dict[str, Any]] = []
    for sample in samples:
        try:
            inputs[sample.sample_id] = build_rawfeat_input(sample.audio_path, feature_config)
        except Exception as exc:
            errors.append(
                {
                    "sample_id": sample.sample_id,
                    "audio_path": str(sample.audio_path),
                    "stage": "build_rawfeat_input",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    return inputs, errors


def _pad_training(api: Any, items: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]]):
    raw, raw_mask = api.pad_sequences([item[0] for item in items])
    features, feature_mask = api.pad_sequences([item[1] for item in items])
    dt, dt_mask = api.pad_sequences([item[2][:, None] for item in items])
    if not (np.array_equal(raw_mask, feature_mask) and np.array_equal(raw_mask, dt_mask)):
        raise RuntimeError("raw/features/dt padding masks differ")
    return raw, features, dt, raw_mask


def _sample_metrics(
    *,
    sample: ESCSample,
    fold: int,
    seed: int,
    lambda_hyb: float,
    normalize: str,
    raw: np.ndarray,
    features: np.ndarray,
    result: Any,
    runtime_sec: float,
) -> dict[str, Any]:
    attribution_sum = float(np.sum(result.attributions))
    residual = float(result.evidence - attribution_sum)
    expected_channels = raw.shape[1] + features.shape[1] + 1
    shape_match = bool(
        result.attributions.shape == (raw.shape[0], expected_channels)
        and result.raw_attributions.shape == raw.shape
        and result.feature_attributions is not None
        and result.feature_attributions.shape == features.shape
        and result.time_attributions.shape == (raw.shape[0],)
    )
    groups = result.group_attributions
    return {
        "dataset": "ESC-50",
        "fold": fold,
        "seed": seed,
        "sample_id": sample.sample_id,
        "category": sample.category,
        "method": "rawfeat_hyb",
        "lambda_hyb": lambda_hyb,
        "normalize": normalize,
        "target_label": result.target_class,
        "rival_label": result.rival_class,
        "evidence": float(result.evidence),
        "absolute_evidence": abs(float(result.evidence)),
        "attribution_sum": attribution_sum,
        "exactness_residual": residual,
        "abs_exactness_residual": abs(residual),
        "audit_passed": bool(result.audit.get("passed", False)),
        "raw_group_attribution": float(groups.get("raw", 0.0)),
        "feature_group_attribution": float(groups.get("features", 0.0)),
        "dt_group_attribution": float(groups.get("dt", 0.0)),
        "raw_shape": _shape(raw),
        "feature_shape": _shape(features),
        "attribution_shape": _shape(result.attributions),
        "time_importance_shape": _shape(result.time_attributions),
        "shape_match": shape_match,
        "T": raw.shape[0],
        "C_raw": raw.shape[1],
        "F": features.shape[1],
        "runtime_sec": runtime_sec,
    }


def _save_figure(path: Path, time_importance: np.ndarray, normalize: str) -> None:
    import matplotlib.pyplot as plt

    values = np.asarray(time_importance, dtype=float)
    if normalize == "l1":
        scale = float(np.sum(np.abs(values)))
        if scale > 0.0:
            values = values / scale
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.plot(np.arange(values.size), values)
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.set(xlabel="Frame", ylabel="Attribution", title="RawFeat time importance")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _generate_and_audit(
    *,
    api: Any,
    engine: Any,
    generator: Any,
    sample: ESCSample,
    fold: int,
    seed: int,
    raw: np.ndarray,
    features: np.ndarray,
    mask: np.ndarray,
    original_result: Any,
    feature_config: FeatureConfig,
    lambda_hyb: float,
    noise_scale: float,
    summary_scaling: str,
    method_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import soundfile as sf

    generated_meta = generator.generate_with_metadata(
        label=original_result.target_class,
        length=raw.shape[0],
        condition_features=features,
        condition_mask=mask,
        noise_scale=noise_scale,
        random_state=seed,
    )
    generated_frames = np.asarray(generated_meta["raw"], dtype=np.float64)
    if generated_frames.shape != raw.shape:
        raise RuntimeError(f"generated raw shape mismatch: {generated_frames.shape} vs {raw.shape}")
    waveform = overlap_add_frames(generated_frames, feature_config.frame_length, feature_config.hop_length)
    wav_path = method_dir / "generated_target.wav"
    method_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(wav_path), waveform, feature_config.target_sr, subtype="FLOAT")

    re_raw, re_features, re_dt, re_mask, re_metadata = build_rawfeat_input(wav_path, feature_config)
    generated_result = engine.explain_one(
        raw=re_raw,
        features=re_features,
        dt=re_dt[:, None],
        mask=re_mask,
        method="hyb",
        target_class=original_result.target_class,
        rival_class=original_result.rival_class,
    )
    generated_sum = float(np.sum(generated_result.attributions))
    generated_residual = float(generated_result.evidence - generated_sum)
    shape_match = bool(
        re_raw.shape == generated_frames.shape
        and re_features.shape[0] == generated_frames.shape[0]
        and generated_result.attributions.shape[0] == generated_frames.shape[0]
    )
    row = {
        "dataset": "ESC-50",
        "fold": fold,
        "seed": seed,
        "sample_id": sample.sample_id,
        "target_label": original_result.target_class,
        "rival_label": original_result.rival_class,
        "lambda_hyb": lambda_hyb,
        "noise_scale": noise_scale,
        "summary_scaling": summary_scaling,
        "generated_wav": str(wav_path),
        "generated_raw_shape": _shape(generated_frames),
        "reprocessed_raw_shape": _shape(re_raw),
        "reprocessed_feature_shape": _shape(re_features),
        "shape_match": shape_match,
        "generated_evidence": float(generated_result.evidence),
        "generated_absolute_evidence": abs(float(generated_result.evidence)),
        "generated_attribution_sum": generated_sum,
        "generated_exactness_residual": generated_residual,
        "generated_abs_exactness_residual": abs(generated_residual),
        "generated_audit_passed": bool(generated_result.audit.get("passed", False)),
        "selected_neighbor_index": generated_meta.get("selected_neighbor_index"),
        "selected_neighbor_distance": generated_meta.get("selected_neighbor_distance"),
    }
    details = {
        "generator_metadata": generated_meta,
        "reprocessed_metadata": re_metadata,
        "generated_explanation": {
            "evidence": generated_result.evidence,
            "group_attributions": generated_result.group_attributions,
            "audit": generated_result.audit,
        },
    }
    return row, details


def run_fold(
    samples: Sequence[ESCSample],
    *,
    fold: int,
    output_dir: Path,
    mode: str,
    seed: int,
    feature_config: FeatureConfig,
    lambda_hyb: float,
    normalize: str,
    generation_scope: str,
    noise_scale: float,
    summary_scaling: str,
    skip_errors: bool,
    make_figures: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    from fpde import dynamic as api

    train_samples, _validation_samples, test_samples = split_esc50(
        samples, fold=fold, mode_config=get_mode_config(mode), seed=seed
    )
    all_needed = [*train_samples, *test_samples]
    inputs, errors = _load_inputs(all_needed, feature_config)
    if errors and not skip_errors:
        first = errors[0]
        raise RuntimeError(f"{first['sample_id']}: {first['error']}")
    train_samples = [sample for sample in train_samples if sample.sample_id in inputs]
    test_samples = [sample for sample in test_samples if sample.sample_id in inputs]
    if len({sample.category for sample in train_samples}) < 2:
        raise ValueError("RawFeat Dynamic-FPDE requires at least two training classes")

    training_items = [inputs[sample.sample_id] for sample in train_samples]
    train_raw, train_features, train_dt, train_mask = _pad_training(api, training_items)
    train_labels = [sample.category for sample in train_samples]
    engine = api.DynamicFPDEEngine(lambda_hyb=lambda_hyb).fit(
        raw=train_raw,
        features=train_features,
        dt=train_dt,
        mask=train_mask,
        y=train_labels,
    )
    generator = api.PrototypeRawGenerator(summary_scaling=summary_scaling).fit(
        raw=train_raw,
        y=train_labels,
        features=train_features,
        mask=train_mask,
    )

    sample_rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    for sample in test_samples:
        raw, features, dt, mask, metadata = inputs[sample.sample_id]
        method_dir = output_dir / "samples" / sample.sample_id / f"rawfeat_hyb_lambda_{lambda_hyb:g}"
        try:
            started = time.perf_counter()
            result = engine.explain_one(
                raw=raw,
                features=features,
                dt=dt[:, None],
                mask=mask,
                method="hyb",
                target_class=sample.category,
            )
            runtime_sec = time.perf_counter() - started
            row = _sample_metrics(
                sample=sample,
                fold=fold,
                seed=seed,
                lambda_hyb=lambda_hyb,
                normalize=normalize,
                raw=raw,
                features=features,
                result=result,
                runtime_sec=runtime_sec,
            )
            if not row["shape_match"]:
                raise RuntimeError("RawFeat output shape contract failed")
            if not np.isclose(row["evidence"], row["attribution_sum"], rtol=1e-9, atol=1e-9):
                raise RuntimeError("RawFeat evidence/additivity audit failed")
            sample_rows.append(row)
            _write_sample_summary(output_dir / "samples" / sample.sample_id / "summary.csv", row)
            metrics: dict[str, Any] = {
                "sample": row,
                "input_metadata": metadata,
                "group_attributions": result.group_attributions,
                "audit": result.audit,
                "feature_slices": result.feature_slices,
                "interpretation": "prototype evidence decomposition; not a causal or classifier-faithfulness claim",
            }
            if make_figures:
                _save_figure(method_dir / "time_importance.png", result.time_attributions, normalize)
            if generation_scope != "none":
                generated_row, generated_details = _generate_and_audit(
                    api=api,
                    engine=engine,
                    generator=generator,
                    sample=sample,
                    fold=fold,
                    seed=seed,
                    raw=raw,
                    features=features,
                    mask=mask,
                    original_result=result,
                    feature_config=feature_config,
                    lambda_hyb=lambda_hyb,
                    noise_scale=noise_scale,
                    summary_scaling=summary_scaling,
                    method_dir=method_dir,
                )
                generation_rows.append(generated_row)
                metrics["generation_audit"] = generated_row
                metrics["generation_details"] = generated_details
            _write_json(method_dir / "metrics.json", metrics)
        except Exception as exc:
            error = {
                "sample_id": sample.sample_id,
                "audio_path": str(sample.audio_path),
                "stage": "explain_or_generate",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            errors.append(error)
            if not skip_errors:
                raise
    return sample_rows, generation_rows, errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["smoke", "pilot", "full"], default="smoke")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--folds", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--frame-length", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--lambda-hyb", type=float, default=0.5)
    parser.add_argument("--normalize", choices=["none", "l1"], default="none")
    parser.add_argument("--make-figures", action="store_true")
    parser.add_argument("--generation-scope", choices=["none", "selected", "all"], default="none")
    parser.add_argument("--noise-scale", type=float, default=0.0)
    parser.add_argument("--summary-scaling", choices=["none", "standard", "robust"], default="standard")
    parser.add_argument("--skip-errors", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0.0 <= args.lambda_hyb <= 1.0:
        raise ValueError("--lambda-hyb must be between 0 and 1")
    if args.noise_scale < 0.0:
        raise ValueError("--noise-scale must be non-negative")
    config_path = args.output_dir / "rawfeat_config.json"
    if config_path.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {config_path}; pass --overwrite to replace outputs")
    samples = read_esc50_metadata(args.dataset_root)
    folds = parse_folds(args.fold, args.folds)
    feature_config = FeatureConfig(
        target_sr=args.target_sr,
        frame_length=args.frame_length,
        hop_length=args.hop_length,
    )
    config = {
        "workflow": "RawFeat Dynamic-FPDE",
        "dataset_root": str(args.dataset_root),
        "output_dir": str(args.output_dir),
        "mode": args.mode,
        "folds": folds,
        "seed": args.seed,
        "feature_config": asdict(feature_config),
        "lambda_hyb": args.lambda_hyb,
        "normalize": args.normalize,
        "generation_scope": args.generation_scope,
        "noise_scale": args.noise_scale,
        "summary_scaling": args.summary_scaling,
        "generated_raw_policy": "inspection/audit only; waveform is reprocessed by the feature extractor",
        "interpretation_limit": "prototype evidence decomposition only",
    }
    _write_json(config_path, config)
    sample_rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for fold in folds:
        fold_samples, fold_generation, fold_errors = run_fold(
            samples,
            fold=fold,
            output_dir=args.output_dir,
            mode=args.mode,
            seed=args.seed,
            feature_config=feature_config,
            lambda_hyb=args.lambda_hyb,
            normalize=args.normalize,
            generation_scope=args.generation_scope,
            noise_scale=args.noise_scale,
            summary_scaling=args.summary_scaling,
            skip_errors=args.skip_errors,
            make_figures=args.make_figures,
        )
        sample_rows.extend(fold_samples)
        generation_rows.extend(fold_generation)
        errors.extend(fold_errors)

    results_dir = args.output_dir / "results"
    write_csv(results_dir / "rawfeat_sample_metrics.csv", sample_rows, SAMPLE_FIELDS)
    write_csv(results_dir / "rawfeat_sample_summary.csv", aggregate_rawfeat_samples(sample_rows))
    write_csv(results_dir / "rawfeat_generation_metrics.csv", generation_rows, GENERATION_FIELDS)
    write_csv(results_dir / "rawfeat_generation_summary.csv", aggregate_rawfeat_generation(generation_rows))
    if errors:
        write_csv(results_dir / "rawfeat_errors.csv", errors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
