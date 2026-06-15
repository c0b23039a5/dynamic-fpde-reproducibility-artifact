"""Run ESC-50 Dynamic-FPDE experiments with fpde-xai/fpde@dynamic."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from fpde import (
    DynamicFPDEExplanation,
    dynamic_fpde_explain_one,
    prepare_dynamic_fpde_context,
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
    make_figures: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, str]]]:
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

    for sample in test_samples:
        X = standardized[sample.sample_id]
        raw_X = raw_features[sample.sample_id]
        selection_explanation = dynamic_fpde_explain_one(
            X,
            context,
            target_label=sample.category,
            rival_label=None,
            mode="dynamic_diff",
        )
        common_rival_label = selection_explanation.rival_label
        for method in ("dynamic_diff", "dynamic_cos"):
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
