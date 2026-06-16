"""Run Raw-Waveform Dynamic-FPDE on ESC-50 audio.

The raw runner keeps waveform samples as the explanation domain. It delegates
Raw-Diff, Raw-Cos, Raw-Hyb, masking, overlap-add, and lambda-wise artifact
saving to the fpde package's Raw-Waveform API.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from experiments.dynamic_fpde_audio.aggregate import write_csv
from experiments.dynamic_fpde_audio.datasets import (
    ESCSample,
    get_mode_config,
    parse_folds,
    read_esc50_metadata,
    split_esc50,
)


DEFAULT_LAMBDA_GRID = tuple(i / 10.0 for i in range(11))

RAW_SAMPLE_FIELDS = [
    "dataset",
    "fold",
    "seed",
    "sample_id",
    "filename",
    "target_label",
    "rival_label",
    "lambda_hyb",
    "evidence",
    "n_windows",
    "input_length",
    "sample_rate",
    "phi_shape",
    "shape_match",
    "generation_status",
    "target_generation_status",
    "rival_generation_status",
    "top_positive_rank",
    "top_positive_window_index",
    "top_positive_start_sample",
    "top_positive_end_sample",
    "top_positive_evidence",
    "top_negative_rank",
    "top_negative_window_index",
    "top_negative_start_sample",
    "top_negative_end_sample",
    "top_negative_evidence",
    "runtime_sec",
    "context_runtime_sec",
    "explain_runtime_sec",
    "save_runtime_sec",
    "device",
    "segment_length",
    "hop_length",
]


RawGenerator = Callable[[Any, float, np.ndarray, int, str, dict[str, Any]], np.ndarray]


def _require_raw_fpde() -> Any:
    import fpde

    required = [
        "prepare_raw_waveform_fpde_context",
        "raw_waveform_fpde_explain_one",
        "save_raw_waveform_fpde_results",
    ]
    missing = [name for name in required if not hasattr(fpde, name)]
    if missing:
        raise RuntimeError(
            "The installed fpde package does not expose Raw-Waveform Dynamic-FPDE APIs. "
            "Refresh the dependency with `python -m pip install --upgrade --force-reinstall "
            "\"fpde @ git+https://github.com/fpde-xai/fpde.git@dynamic\"`."
        )
    return fpde


def _read_waveform(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise ImportError("Raw-Waveform Dynamic-FPDE requires soundfile to read ESC-50 WAV files.") from exc

    waveform, sample_rate = sf.read(str(path), always_2d=False)
    arr = np.asarray(waveform, dtype=float)
    if arr.size == 0:
        raise ValueError(f"empty waveform: {path}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"waveform contains NaN or inf: {path}")
    return arr, int(sample_rate)


def _load_waveforms(samples: Sequence[ESCSample]) -> tuple[list[np.ndarray], list[int], list[str], list[dict[str, object]]]:
    waveforms: list[np.ndarray] = []
    sample_rates: list[int] = []
    labels: list[str] = []
    errors: list[dict[str, object]] = []
    for sample in samples:
        try:
            waveform, sample_rate = _read_waveform(sample.audio_path)
        except Exception as exc:
            errors.append(
                {
                    "sample_id": sample.sample_id,
                    "filename": sample.filename,
                    "error": str(exc),
                }
            )
            continue
        waveforms.append(waveform)
        sample_rates.append(sample_rate)
        labels.append(sample.category)
    return waveforms, sample_rates, labels, errors


def parse_lambda_grid(value: str | None) -> tuple[float, ...] | None:
    if value is None or value == "":
        return DEFAULT_LAMBDA_GRID
    parsed = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not parsed:
        raise ValueError("--lambda-grid did not contain any values")
    for item in parsed:
        if not math.isfinite(item) or item < 0.0 or item > 1.0:
            raise ValueError("lambda grid values must be finite numbers in [0, 1]")
    return parsed


def _load_raw_generator(spec: str | None) -> RawGenerator | None:
    if not spec:
        return None
    if ":" not in spec:
        raise ValueError("--raw-generator must use module:function syntax")
    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    generator = getattr(module, function_name)
    if not callable(generator):
        raise TypeError(f"raw generator is not callable: {spec}")
    return generator


def _compact_status(status: dict[str, Any]) -> str:
    return json.dumps(status, sort_keys=True)


def _segment_metadata(rows: list[dict[str, Any]]) -> dict[str, object]:
    if not rows:
        return {
            "rank": "",
            "window_index": "",
            "start_sample": "",
            "end_sample": "",
            "evidence": "",
        }
    row = rows[0]
    return {
        "rank": row.get("rank", ""),
        "window_index": row.get("window_index", ""),
        "start_sample": row.get("start_sample", ""),
        "end_sample": row.get("end_sample", ""),
        "evidence": row.get("evidence", ""),
    }


def _row_from_lambda_result(
    *,
    sample: ESCSample,
    fold: int,
    seed: int,
    explanation: Any,
    lambda_hyb: float,
    result: dict[str, Any],
    context_runtime_sec: float,
    explain_runtime_sec: float,
    save_runtime_sec: float,
) -> dict[str, object]:
    phi = np.asarray(result["phi"], dtype=float)
    top_positive = _segment_metadata(result.get("top_positive_segments", []))
    top_negative = _segment_metadata(result.get("top_negative_segments", []))
    generation_status = dict(result.get("generation_status", {}))
    runtime_sec = context_runtime_sec + explain_runtime_sec + save_runtime_sec
    return {
        "dataset": "esc50",
        "fold": fold,
        "seed": seed,
        "sample_id": sample.sample_id,
        "filename": sample.filename,
        "target_label": explanation.target_label,
        "rival_label": explanation.rival_label,
        "lambda_hyb": float(lambda_hyb),
        "evidence": float(result["evidence"]),
        "n_windows": int(np.asarray(result["window_evidence"]).shape[0]),
        "input_length": int(explanation.waveform.shape[0]),
        "sample_rate": int(explanation.sample_rate),
        "phi_shape": str(tuple(phi.shape)),
        "shape_match": bool(phi.shape == explanation.waveform.shape),
        "generation_status": _compact_status(generation_status),
        "target_generation_status": generation_status.get("target", ""),
        "rival_generation_status": generation_status.get("rival", ""),
        "top_positive_rank": top_positive["rank"],
        "top_positive_window_index": top_positive["window_index"],
        "top_positive_start_sample": top_positive["start_sample"],
        "top_positive_end_sample": top_positive["end_sample"],
        "top_positive_evidence": top_positive["evidence"],
        "top_negative_rank": top_negative["rank"],
        "top_negative_window_index": top_negative["window_index"],
        "top_negative_start_sample": top_negative["start_sample"],
        "top_negative_end_sample": top_negative["end_sample"],
        "top_negative_evidence": top_negative["evidence"],
        "runtime_sec": runtime_sec,
        "context_runtime_sec": context_runtime_sec,
        "explain_runtime_sec": explain_runtime_sec,
        "save_runtime_sec": save_runtime_sec,
        "device": result.get("details", {}).get("device", explanation.details.get("device", "")),
        "segment_length": explanation.details.get("segment_length", ""),
        "hop_length": explanation.details.get("hop_length", ""),
    }


def _stats(values: list[float]) -> dict[str, object]:
    if not values:
        return {"mean": "", "std": "", "min": "", "max": "", "n": 0}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "mean": mean,
        "std": variance**0.5,
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def summarize_by_lambda(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["fold"]), str(row["lambda_hyb"]))].append(row)
    out: list[dict[str, object]] = []
    for (fold, lambda_hyb), group_rows in sorted(groups.items(), key=lambda item: (int(item[0][0]), float(item[0][1]))):
        evidence = [float(row["evidence"]) for row in group_rows]
        runtime = [float(row["runtime_sec"]) for row in group_rows]
        evidence_stats = _stats(evidence)
        runtime_stats = _stats(runtime)
        shape_matches = [str(row["shape_match"]) == "True" or row["shape_match"] is True for row in group_rows]
        out.append(
            {
                "dataset": "esc50",
                "fold": fold,
                "lambda_hyb": float(lambda_hyb),
                "n": len(group_rows),
                "evidence_mean": evidence_stats["mean"],
                "evidence_std": evidence_stats["std"],
                "evidence_min": evidence_stats["min"],
                "evidence_max": evidence_stats["max"],
                "runtime_sec_mean": runtime_stats["mean"],
                "runtime_sec_std": runtime_stats["std"],
                "shape_match_rate": sum(1 for value in shape_matches if value) / len(shape_matches),
                "target_generation_ok": sum(1 for row in group_rows if row.get("target_generation_status") == "ok"),
                "rival_generation_ok": sum(1 for row in group_rows if row.get("rival_generation_status") == "ok"),
            }
        )
    return out


def run_fold(
    samples: Sequence[ESCSample],
    *,
    fold: int,
    output_dir: Path,
    mode: str,
    seed: int,
    target_sr: int,
    segment_sec: float,
    hop_sec: float,
    lambda_grid: Sequence[float] | None,
    top_k_segments: int,
    device: str,
    raw_generator: RawGenerator | None,
    skip_errors: bool,
    save_plots: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    fpde = _require_raw_fpde()
    mode_config = get_mode_config(mode)
    train_samples, _val_samples, test_samples = split_esc50(samples, fold=fold, mode_config=mode_config, seed=seed)

    train_waveforms, train_rates, train_labels, train_errors = _load_waveforms(train_samples)
    if train_errors and not skip_errors:
        raise RuntimeError(f"failed to load training waveforms: {train_errors[:3]}")
    if not train_waveforms:
        raise RuntimeError("no training waveforms were loaded")
    context_start = perf_counter()
    context = fpde.prepare_raw_waveform_fpde_context(
        train_waveforms,
        train_labels,
        sample_rates=train_rates,
        target_sr=target_sr,
        segment_sec=segment_sec,
        hop_sec=hop_sec,
    )
    context_runtime_sec = perf_counter() - context_start

    rows: list[dict[str, object]] = []
    errors = list(train_errors)
    lambdas = None if lambda_grid is None else tuple(lambda_grid)
    for sample in test_samples:
        try:
            waveform, sample_rate = _read_waveform(sample.audio_path)
            explain_start = perf_counter()
            explanation = fpde.raw_waveform_fpde_explain_one(
                waveform,
                context,
                sample_rate=sample_rate,
                target_label=sample.category,
                lambda_grid=lambdas,
                top_k_segments=top_k_segments,
                generator=raw_generator,
                device=device,
                details={"sample_id": sample.sample_id, "fold": fold, "seed": seed},
            )
            explain_runtime_sec = perf_counter() - explain_start
            sample_dir = output_dir / "samples" / sample.sample_id
            save_start = perf_counter()
            fpde.save_raw_waveform_fpde_results(explanation, sample_dir, save_plots=save_plots)
            save_runtime_sec = perf_counter() - save_start
        except Exception as exc:
            if not skip_errors:
                raise
            errors.append({"sample_id": sample.sample_id, "filename": sample.filename, "error": str(exc)})
            continue

        for lambda_hyb, result in sorted(explanation.lambda_results.items(), key=lambda item: item[0]):
            rows.append(
                _row_from_lambda_result(
                    sample=sample,
                    fold=fold,
                    seed=seed,
                    explanation=explanation,
                    lambda_hyb=float(lambda_hyb),
                    result=result,
                    context_runtime_sec=context_runtime_sec,
                    explain_runtime_sec=explain_runtime_sec,
                    save_runtime_sec=save_runtime_sec,
                )
            )

    return rows, errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Raw-Waveform Dynamic-FPDE on ESC-50 raw audio.")
    parser.add_argument("--dataset", default="esc50", choices=["esc50"])
    parser.add_argument("--dataset-root", type=Path, default=Path("data/ESC-50"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/raw_waveform_dynamic_fpde_esc50_smoke"))
    parser.add_argument("--mode", default="smoke", choices=["smoke", "pilot", "full"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--folds", default=None, help="Comma-separated ESC-50 folds. Overrides --fold.")
    parser.add_argument("--target-sr", "--sr", type=int, default=16000)
    parser.add_argument("--segment-sec", type=float, default=0.5)
    parser.add_argument("--hop-sec", type=float, default=0.1)
    parser.add_argument("--lambda-grid", default=None, help="Comma-separated lambda_hyb values. Defaults to 0.0..1.0 by 0.1.")
    parser.add_argument("--top-k-segments", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--raw-generator", default=None, help="Optional module:function hook for label-conditioned RAW generation.")
    parser.add_argument("--skip-errors", action="store_true")
    parser.add_argument("--no-plots", dest="save_plots", action="store_false", help="Skip PNG plot artifacts.")
    parser.set_defaults(save_plots=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    samples = read_esc50_metadata(args.dataset_root)
    folds = parse_folds(args.fold, args.folds)
    lambda_grid = parse_lambda_grid(args.lambda_grid)
    generator = _load_raw_generator(args.raw_generator)

    output_dir = args.output_dir
    results_dir = output_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "dataset": args.dataset,
        "dataset_root": str(args.dataset_root),
        "mode": args.mode,
        "seed": args.seed,
        "folds": folds,
        "target_sr": args.target_sr,
        "segment_sec": args.segment_sec,
        "hop_sec": args.hop_sec,
        "lambda_grid": list(lambda_grid) if lambda_grid is not None else list(DEFAULT_LAMBDA_GRID),
        "top_k_segments": args.top_k_segments,
        "device": args.device,
        "raw_generator": args.raw_generator,
        "input_space": "raw waveform + label",
        "uses_acoustic_features": False,
        "uses_spectrogram": False,
        "uses_mfcc": False,
        "waveform_normalization": False,
        "label_conditioned_raw_generation": "after important segment extraction; skipped when --raw-generator is omitted",
    }
    with (output_dir / "raw_waveform_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")

    all_rows: list[dict[str, object]] = []
    all_errors: list[dict[str, object]] = []
    for fold in folds:
        rows, errors = run_fold(
            samples,
            fold=fold,
            output_dir=output_dir,
            mode=args.mode,
            seed=args.seed,
            target_sr=args.target_sr,
            segment_sec=args.segment_sec,
            hop_sec=args.hop_sec,
            lambda_grid=lambda_grid,
            top_k_segments=args.top_k_segments,
            device=args.device,
            raw_generator=generator,
            skip_errors=args.skip_errors,
            save_plots=args.save_plots,
        )
        all_rows.extend(rows)
        all_errors.extend(errors)

    write_csv(results_dir / "raw_waveform_sample_metrics.csv", all_rows, RAW_SAMPLE_FIELDS)
    write_csv(results_dir / "raw_waveform_summary_by_lambda.csv", summarize_by_lambda(all_rows))
    if all_errors:
        write_csv(results_dir / "raw_waveform_errors.csv", all_errors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
