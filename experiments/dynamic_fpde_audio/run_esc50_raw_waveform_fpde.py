"""Run Raw-Waveform Dynamic-FPDE on ESC-50 audio.

The raw runner keeps waveform samples as the explanation domain. It delegates
Raw-Diff, Raw-Cos, Raw-Hyb, masking, overlap-add, and lambda-wise artifact
saving to the fpde package's Raw-Waveform API.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable, Sequence

from tqdm.auto import tqdm

import numpy as np

from experiments.dynamic_fpde_audio.aggregate import write_csv
from experiments.dynamic_fpde_audio.datasets import (
    ESCSample,
    get_mode_config,
    parse_folds,
    read_esc50_metadata,
    split_esc50,
)
from experiments.dynamic_fpde_audio.raw_waveform_context import (
    load_raw_context_cache,
    prepare_fast_raw_waveform_fpde_context,
    resample_raw_waveform_polyphase,
    save_raw_context_cache,
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
    "evidence_total",
    "evidence_per_window",
    "evidence_per_valid_sample",
    "positive_evidence",
    "negative_evidence",
    "absolute_evidence",
    "positive_window_rate",
    "negative_window_rate",
    "n_valid_samples",
    "coverage_rate",
    "raw_diff_unscaled_evidence",
    "raw_cos_unscaled_evidence",
    "raw_hyb_l1_evidence",
    "lambda_selection_mode",
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
    "fold_context_runtime_sec",
    "context_runtime_amortized_sec",
    "sample_audio_load_runtime_sec",
    "fold_audio_load_runtime_sec",
    "audio_load_runtime_sec",
    "sample_resample_runtime_sec",
    "resample_runtime_sec",
    "windowing_runtime_sec",
    "bank_build_runtime_sec",
    "medoid_runtime_sec",
    "rival_selection_runtime_sec",
    "diff_runtime_sec",
    "cos_runtime_sec",
    "hyb_runtime_sec",
    "overlap_add_runtime_sec",
    "generation_runtime_sec",
    "plot_runtime_sec",
    "explain_runtime_sec",
    "save_runtime_sec",
    "sample_explain_runtime_sec",
    "sample_save_runtime_sec",
    "sample_total_runtime_sec",
    "timings_overlap",
    "n_test_samples",
    "device",
    "segment_length",
    "hop_length",
    "prototype_selection",
    "medoid_block_size",
    "max_prototype_candidates",
    "context_device",
    "context_cache_hit",
    "resample_method",
    "resampler_version",
]

RAW_METHOD_FIELDS = [
    "dataset",
    "fold",
    "seed",
    "sample_id",
    "filename",
    "target_label",
    "rival_label",
    "method",
    "lambda_hyb",
    "evidence_total",
    "evidence_per_window",
    "evidence_per_valid_sample",
    "positive_evidence",
    "negative_evidence",
    "absolute_evidence",
    "positive_window_rate",
    "negative_window_rate",
    "n_windows",
    "n_valid_samples",
    "coverage_rate",
    "runtime_sec",
    "sample_total_runtime_sec",
    "device",
    "segment_length",
    "hop_length",
]


RawGenerator = Callable[[Any, float, np.ndarray, int, str, dict[str, Any]], np.ndarray]


def _require_raw_fpde() -> Any:
    import fpde

    required = [
        "RawWaveformFPDEContext",
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


def _load_waveforms(samples: Sequence[ESCSample]) -> tuple[list[np.ndarray], list[int], list[str], list[str], list[dict[str, object]], float]:
    waveforms: list[np.ndarray] = []
    sample_rates: list[int] = []
    labels: list[str] = []
    sample_ids: list[str] = []
    errors: list[dict[str, object]] = []
    runtime_sec = 0.0
    for sample in tqdm(
    samples,
    desc="Loading training WAV files",
    unit="file",
    dynamic_ncols=True,
    ):
        try:
            start = perf_counter()
            waveform, sample_rate = _read_waveform(sample.audio_path)
            runtime_sec += perf_counter() - start
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
        sample_ids.append(sample.sample_id)
    return waveforms, sample_rates, labels, sample_ids, errors, runtime_sec


def _none_if_nonpositive(value: int | None) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    return None if parsed <= 0 else parsed


def _package_version(package: str) -> str:
    try:
        import importlib.metadata

        return importlib.metadata.version(package)
    except Exception:
        return "unknown"


def _hash_train_split(train_samples: Sequence[ESCSample]) -> str:
    digest = hashlib.sha256()
    for sample in sorted(train_samples, key=lambda item: item.sample_id):
        stat = sample.audio_path.stat()
        digest.update(
            "|".join(
                [
                    sample.sample_id,
                    sample.filename,
                    sample.category,
                    str(sample.fold),
                    str(stat.st_size),
                    str(int(stat.st_mtime_ns)),
                ]
            ).encode("utf-8")
        )
    return digest.hexdigest()[:16]


def _raw_context_cache_key(
    train_samples: Sequence[ESCSample],
    *,
    fold: int,
    seed: int,
    target_sr: int,
    segment_sec: float,
    hop_sec: float,
    prototype_selection: str,
    medoid_block_size: int,
    max_prototype_candidates: int | None,
    context_device: str,
) -> str:
    payload = {
        "dataset_hash": _hash_train_split(train_samples),
        "fold": int(fold),
        "seed": int(seed),
        "target_sr": int(target_sr),
        "segment_sec": float(segment_sec),
        "hop_sec": float(hop_sec),
        "prototype_selection": prototype_selection,
        "medoid_block_size": int(medoid_block_size),
        "max_prototype_candidates": _none_if_nonpositive(max_prototype_candidates),
        "context_device": context_device,
        "fpde_version": _package_version("fpde"),
    }
    text = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def _atomic_write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    write_csv(tmp, rows, fieldnames)
    tmp.replace(path)


def _read_csv_if_exists(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _dedupe_rows(rows: Iterable[dict[str, object]], key_fields: Sequence[str]) -> list[dict[str, object]]:
    keyed: dict[tuple[str, ...], dict[str, object]] = {}
    for row in rows:
        key = tuple(str(row.get(field, "")) for field in key_fields)
        keyed[key] = dict(row)
    return list(keyed.values())


def _read_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _write_completed(path: Path, sample_ids: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(f"{sample_id}\n" for sample_id in sorted(sample_ids)), encoding="utf-8")
    tmp.replace(path)


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


def _window_evidence_metrics(
    window_evidence: np.ndarray,
    effective_masks: np.ndarray,
    input_length: int,
    window_starts: np.ndarray | None = None,
) -> dict[str, object]:
    evidence = np.asarray(window_evidence, dtype=float)
    masks = np.asarray(effective_masks, dtype=bool)
    n_windows = int(evidence.shape[0])
    covered = np.zeros(int(input_length), dtype=bool)
    if masks.ndim == 2 and covered.size:
        starts = np.zeros(masks.shape[0], dtype=np.intp) if window_starts is None else np.asarray(window_starts, dtype=np.intp)
        for mask, start in zip(masks, starts, strict=False):
            valid_idx = np.where(mask)[0] + int(start)
            valid_idx = valid_idx[(valid_idx >= 0) & (valid_idx < covered.size)]
            covered[valid_idx] = True
    n_valid_samples = int(np.sum(covered))
    total = float(np.sum(evidence))
    positive_values = evidence[evidence > 0.0]
    negative_values = evidence[evidence < 0.0]
    return {
        "evidence_total": total,
        "evidence_per_window": total / n_windows if n_windows else "",
        "evidence_per_valid_sample": total / n_valid_samples if n_valid_samples else "",
        "positive_evidence": float(np.sum(positive_values)) if positive_values.size else 0.0,
        "negative_evidence": float(np.sum(negative_values)) if negative_values.size else 0.0,
        "absolute_evidence": float(np.sum(np.abs(evidence))),
        "positive_window_rate": float(np.mean(evidence > 0.0)) if n_windows else "",
        "negative_window_rate": float(np.mean(evidence < 0.0)) if n_windows else "",
        "n_windows": n_windows,
        "n_valid_samples": n_valid_samples,
        "coverage_rate": n_valid_samples / int(input_length) if int(input_length) > 0 else "",
    }


def _detail_vector(details: dict[str, Any], key: str) -> np.ndarray:
    values = details.get(key, [])
    return np.asarray(values, dtype=float)


def _row_from_lambda_result(
    *,
    sample: ESCSample,
    fold: int,
    seed: int,
    explanation: Any,
    lambda_hyb: float,
    result: dict[str, Any],
    context_runtime_sec: float,
    context_timings: dict[str, float],
    explain_runtime_sec: float,
    save_runtime_sec: float,
    audio_load_runtime_sec: float,
    sample_resample_runtime_sec: float,
    n_test_samples: int,
    prototype_selection: str,
    medoid_block_size: int,
    max_prototype_candidates: int | None,
    context_device: str,
    context_cache_hit: bool,
) -> dict[str, object]:
    phi = np.asarray(result["phi"], dtype=float)
    top_positive = _segment_metadata(result.get("top_positive_segments", []))
    top_negative = _segment_metadata(result.get("top_negative_segments", []))
    generation_status = dict(result.get("generation_status", {}))
    details = dict(result.get("details", {}))
    window_metrics = _window_evidence_metrics(
        np.asarray(result["window_evidence"], dtype=float),
        np.asarray(result.get("effective_window_masks", []), dtype=bool),
        int(explanation.waveform.shape[0]),
        np.asarray(result.get("window_starts", []), dtype=np.intp),
    )
    raw_diff = _detail_vector(details, "diff_evidence")
    raw_cos = _detail_vector(details, "cos_evidence")
    sample_total_runtime_sec = float(audio_load_runtime_sec) + float(sample_resample_runtime_sec) + float(explain_runtime_sec) + float(save_runtime_sec)
    context_runtime_amortized_sec = float(context_runtime_sec) / int(n_test_samples) if int(n_test_samples) > 0 else ""
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
        **window_metrics,
        "raw_diff_unscaled_evidence": float(np.sum(raw_diff)) if raw_diff.size else "",
        "raw_cos_unscaled_evidence": float(np.sum(raw_cos)) if raw_cos.size else "",
        "raw_hyb_l1_evidence": float(result["evidence"]),
        "lambda_selection_mode": "all_lambdas_reported_no_test_best_selection",
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
        "runtime_sec": sample_total_runtime_sec,
        "context_runtime_sec": context_runtime_sec,
        "fold_context_runtime_sec": context_runtime_sec,
        "context_runtime_amortized_sec": context_runtime_amortized_sec,
        "sample_audio_load_runtime_sec": float(audio_load_runtime_sec),
        "fold_audio_load_runtime_sec": float(context_timings.get("audio_load_runtime_sec", 0.0)),
        "audio_load_runtime_sec": float(audio_load_runtime_sec),
        "sample_resample_runtime_sec": float(sample_resample_runtime_sec),
        "resample_runtime_sec": float(sample_resample_runtime_sec),
        "windowing_runtime_sec": context_timings.get("windowing_runtime_sec", ""),
        "bank_build_runtime_sec": context_timings.get("bank_build_runtime_sec", ""),
        "medoid_runtime_sec": context_timings.get("medoid_runtime_sec", ""),
        "rival_selection_runtime_sec": details.get("rival_selection_runtime_sec", ""),
        "diff_runtime_sec": details.get("diff_runtime_sec", ""),
        "cos_runtime_sec": details.get("cos_runtime_sec", ""),
        "hyb_runtime_sec": details.get("hyb_runtime_sec", ""),
        "overlap_add_runtime_sec": details.get("overlap_add_runtime_sec", ""),
        "generation_runtime_sec": details.get("generation_runtime_sec", ""),
        "plot_runtime_sec": details.get("plot_runtime_sec", ""),
        "explain_runtime_sec": explain_runtime_sec,
        "save_runtime_sec": save_runtime_sec,
        "sample_explain_runtime_sec": explain_runtime_sec,
        "sample_save_runtime_sec": save_runtime_sec,
        "sample_total_runtime_sec": sample_total_runtime_sec,
        "timings_overlap": False,
        "n_test_samples": int(n_test_samples),
        "device": details.get("device", explanation.details.get("device", "")),
        "segment_length": explanation.details.get("segment_length", ""),
        "hop_length": explanation.details.get("hop_length", ""),
        "prototype_selection": prototype_selection,
        "medoid_block_size": int(medoid_block_size),
        "max_prototype_candidates": "" if max_prototype_candidates is None else int(max_prototype_candidates),
        "context_device": context_device,
        "context_cache_hit": bool(context_cache_hit),
        "resample_method": explanation.details.get("resample_method", ""),
        "resampler_version": explanation.details.get("resampler_version", ""),
    }


def _method_rows_from_lambda_result(
    *,
    sample: ESCSample,
    fold: int,
    seed: int,
    explanation: Any,
    lambda_hyb: float,
    result: dict[str, Any],
    runtime_sec: float,
    include_unscaled_components: bool,
) -> list[dict[str, object]]:
    details = dict(result.get("details", {}))
    rows: list[dict[str, object]] = []
    base = {
        "dataset": "esc50",
        "fold": int(fold),
        "seed": int(seed),
        "sample_id": sample.sample_id,
        "filename": sample.filename,
        "target_label": explanation.target_label,
        "rival_label": explanation.rival_label,
        "runtime_sec": float(runtime_sec),
        "sample_total_runtime_sec": float(runtime_sec),
        "device": details.get("device", explanation.details.get("device", "")),
        "segment_length": explanation.details.get("segment_length", ""),
        "hop_length": explanation.details.get("hop_length", ""),
    }
    effective_masks = np.asarray(result.get("effective_window_masks", []), dtype=bool)
    window_starts = np.asarray(result.get("window_starts", []), dtype=np.intp)
    input_length = int(explanation.waveform.shape[0])
    method_specs: list[tuple[str, float | str, np.ndarray]] = []
    if include_unscaled_components:
        method_specs.extend(
            [
                ("raw_diff_unscaled", "", _detail_vector(details, "diff_evidence")),
                ("raw_cos_unscaled", "", _detail_vector(details, "cos_evidence")),
            ]
        )
    method_specs.append((f"raw_hyb_l1_lambda_{float(lambda_hyb):.1f}", float(lambda_hyb), np.asarray(result["window_evidence"], dtype=float)))
    for method, lambda_value, evidence_values in method_specs:
        metrics = _window_evidence_metrics(evidence_values, effective_masks, input_length, window_starts)
        rows.append({**base, "method": method, "lambda_hyb": lambda_value, **metrics})
    return rows


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


def _float_values(rows: Iterable[dict[str, object]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key, "")
        if value in ("", None):
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(parsed):
            values.append(parsed)
    return values


def summarize_by_lambda(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["fold"]), str(row["lambda_hyb"]))].append(row)
    out: list[dict[str, object]] = []
    for (fold, lambda_hyb), group_rows in sorted(groups.items(), key=lambda item: (int(item[0][0]), float(item[0][1]))):
        evidence = _float_values(group_rows, "evidence")
        runtime = _float_values(group_rows, "runtime_sec")
        evidence_stats = _stats(evidence)
        runtime_stats = _stats(runtime)
        shape_matches = [str(row["shape_match"]) == "True" or row["shape_match"] is True for row in group_rows]
        summary = {
            "dataset": "esc50",
            "fold": fold,
            "lambda_hyb": float(lambda_hyb),
            "n": len(group_rows),
            "n_unique_samples": len({str(row.get("sample_id", "")) for row in group_rows}),
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
        for metric in (
            "evidence_per_window",
            "evidence_per_valid_sample",
            "absolute_evidence",
            "positive_evidence",
            "negative_evidence",
            "positive_window_rate",
            "negative_window_rate",
            "coverage_rate",
            "sample_total_runtime_sec",
            "context_runtime_amortized_sec",
        ):
            stats = _stats(_float_values(group_rows, metric))
            summary[f"{metric}_mean"] = stats["mean"]
            summary[f"{metric}_std"] = stats["std"]
        out.append(summary)
    return out


def run_fold(
    samples: Sequence[ESCSample],
    *,
    fold: int,
    output_dir: Path,
    fold_output_dir: Path | None = None,
    mode: str,
    seed: int,
    target_sr: int,
    segment_sec: float,
    hop_sec: float,
    lambda_grid: Sequence[float] | None,
    top_k_segments: int,
    device: str,
    prototype_selection: str = "exact_medoid",
    medoid_block_size: int = 128,
    max_prototype_candidates: int | None = None,
    context_device: str = "cpu",
    retain_segment_banks: bool = False,
    context_cache_dir: Path | None = None,
    resume: bool = False,
    overwrite: bool = False,
    skip_completed_samples: bool = False,
    raw_generator: RawGenerator | None = None,
    skip_errors: bool = False,
    save_plots: bool = True,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    fpde = _require_raw_fpde()
    mode_config = get_mode_config(mode)
    train_samples, _val_samples, test_samples = split_esc50(samples, fold=fold, mode_config=mode_config, seed=seed)

    train_waveforms, train_rates, train_labels, train_sample_ids, train_errors, train_audio_runtime = _load_waveforms(train_samples)
    if train_errors and not skip_errors:
        raise RuntimeError(f"failed to load training waveforms: {train_errors[:3]}")
    if not train_waveforms:
        raise RuntimeError("no training waveforms were loaded")

    max_candidates = _none_if_nonpositive(max_prototype_candidates)
    cache_path: Path | None = None
    if context_cache_dir is not None and not retain_segment_banks:
        cache_key = _raw_context_cache_key(
            train_samples,
            fold=fold,
            seed=seed,
            target_sr=target_sr,
            segment_sec=segment_sec,
            hop_sec=hop_sec,
            prototype_selection=prototype_selection,
            medoid_block_size=medoid_block_size,
            max_prototype_candidates=max_candidates,
            context_device=context_device,
        )
        cache_path = context_cache_dir / f"esc50_fold{fold}_seed{seed}_sr{target_sr}_seg{segment_sec:g}_hop{hop_sec:g}_{cache_key}.npz"

    context_cache_hit = False
    cached_context: Any | None = None
    if cache_path is not None and cache_path.exists() and not overwrite:
        context_start = perf_counter()
        try:
            cached_context = load_raw_context_cache(cache_path, fpde)
        except Exception:
            cache_path.unlink(missing_ok=True)
            cached_context = None
        if cached_context is not None:
            context = cached_context
            context_timings = {
                "audio_load_runtime_sec": train_audio_runtime,
                "context_runtime_sec": perf_counter() - context_start,
                "resample_runtime_sec": 0.0,
                "windowing_runtime_sec": 0.0,
                "bank_build_runtime_sec": 0.0,
                "medoid_runtime_sec": 0.0,
            }
            context_cache_hit = True
    if cached_context is None:
        build_result = prepare_fast_raw_waveform_fpde_context(
            fpde,
            train_waveforms,
            train_labels,
            sample_rates=train_rates,
            sample_ids=train_sample_ids,
            target_sr=target_sr,
            segment_sec=segment_sec,
            hop_sec=hop_sec,
            prototype_selection=prototype_selection,
            medoid_block_size=medoid_block_size,
            max_candidates_per_label=max_candidates,
            context_device=context_device,
            retain_segment_banks=retain_segment_banks,
            seed=seed,
        )
        context = build_result.context
        context_timings = dict(build_result.timings)
        context_timings["audio_load_runtime_sec"] = train_audio_runtime
        if cache_path is not None:
            save_raw_context_cache(cache_path, context)
    context_runtime_sec = float(context_timings.get("context_runtime_sec", 0.0))

    active_output_dir = fold_output_dir or output_dir
    fold_results_dir = active_output_dir / "results"
    completed_path = active_output_dir / "completed_samples.txt"
    fold_sample_metrics_path = fold_results_dir / "raw_waveform_sample_metrics.csv"
    fold_method_metrics_path = fold_results_dir / "raw_waveform_method_metrics.csv"
    completed = _read_completed(completed_path) if resume and not overwrite else set()
    rows: list[dict[str, object]] = (
        _dedupe_rows(_read_csv_if_exists(fold_sample_metrics_path), ("fold", "seed", "sample_id", "lambda_hyb"))
        if resume and not overwrite
        else []
    )
    method_rows: list[dict[str, object]] = (
        _dedupe_rows(_read_csv_if_exists(fold_method_metrics_path), ("fold", "seed", "sample_id", "method"))
        if resume and not overwrite
        else []
    )
    errors = list(train_errors)
    lambdas = None if lambda_grid is None else tuple(lambda_grid)
    n_test_samples = len(test_samples)
    for sample in test_samples:
        if (resume or skip_completed_samples) and sample.sample_id in completed:
            continue
        try:
            audio_start = perf_counter()
            waveform, sample_rate = _read_waveform(sample.audio_path)
            sample_audio_runtime = perf_counter() - audio_start
            sample_resample_start = perf_counter()
            waveform = resample_raw_waveform_polyphase(waveform, source_sr=sample_rate, target_sr=target_sr)
            sample_resample_runtime = perf_counter() - sample_resample_start
            explain_start = perf_counter()
            explanation = fpde.raw_waveform_fpde_explain_one(
                waveform,
                context,
                sample_rate=target_sr,
                target_label=sample.category,
                lambda_grid=lambdas,
                top_k_segments=top_k_segments,
                generator=raw_generator,
                device=device,
                details={
                    "sample_id": sample.sample_id,
                    "fold": fold,
                    "seed": seed,
                    "resample_method": context.details.get("resample_method", ""),
                    "resampler_version": context.details.get("resampler_version", ""),
                },
            )
            explain_runtime_sec = perf_counter() - explain_start
            sample_dir = active_output_dir / "samples" / sample.sample_id
            save_start = perf_counter()
            fpde.save_raw_waveform_fpde_results(explanation, sample_dir, save_plots=save_plots)
            save_runtime_sec = perf_counter() - save_start
        except Exception as exc:
            if not skip_errors:
                raise
            errors.append({"sample_id": sample.sample_id, "filename": sample.filename, "error": str(exc)})
            continue

        for lambda_index, (lambda_hyb, result) in enumerate(sorted(explanation.lambda_results.items(), key=lambda item: item[0])):
            rows.append(
                _row_from_lambda_result(
                    sample=sample,
                    fold=fold,
                    seed=seed,
                    explanation=explanation,
                    lambda_hyb=float(lambda_hyb),
                    result=result,
                    context_runtime_sec=context_runtime_sec,
                    context_timings=context_timings,
                    explain_runtime_sec=explain_runtime_sec,
                    save_runtime_sec=save_runtime_sec,
                    audio_load_runtime_sec=sample_audio_runtime,
                    sample_resample_runtime_sec=sample_resample_runtime,
                    n_test_samples=n_test_samples,
                    prototype_selection=prototype_selection,
                    medoid_block_size=medoid_block_size,
                    max_prototype_candidates=max_candidates,
                    context_device=context_device,
                    context_cache_hit=context_cache_hit,
                )
            )
            method_rows.extend(
                _method_rows_from_lambda_result(
                    sample=sample,
                    fold=fold,
                    seed=seed,
                    explanation=explanation,
                    lambda_hyb=float(lambda_hyb),
                    result=result,
                    runtime_sec=sample_audio_runtime + sample_resample_runtime + explain_runtime_sec + save_runtime_sec,
                    include_unscaled_components=lambda_index == 0,
                )
            )
        rows = _dedupe_rows(rows, ("fold", "seed", "sample_id", "lambda_hyb"))
        method_rows = _dedupe_rows(method_rows, ("fold", "seed", "sample_id", "method"))
        completed.add(sample.sample_id)
        if fold_output_dir is not None or resume or skip_completed_samples:
            _atomic_write_csv(fold_sample_metrics_path, rows, RAW_SAMPLE_FIELDS)
            _atomic_write_csv(fold_method_metrics_path, method_rows, RAW_METHOD_FIELDS)
            _atomic_write_csv(fold_results_dir / "raw_waveform_summary_by_lambda.csv", summarize_by_lambda(rows))
            _write_completed(completed_path, completed)

    if fold_output_dir is not None or resume or skip_completed_samples:
        _atomic_write_csv(fold_sample_metrics_path, rows, RAW_SAMPLE_FIELDS)
        _atomic_write_csv(fold_method_metrics_path, method_rows, RAW_METHOD_FIELDS)
        _atomic_write_csv(fold_results_dir / "raw_waveform_summary_by_lambda.csv", summarize_by_lambda(rows))
        if errors:
            _atomic_write_csv(fold_results_dir / "raw_waveform_errors.csv", errors)
        _write_completed(completed_path, completed)
    return rows, method_rows, errors


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
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cuda")
    parser.add_argument("--context-device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--prototype-selection", choices=["exact_medoid", "sampled_medoid"], default="exact_medoid")
    parser.add_argument("--medoid-block-size", type=int, default=128)
    parser.add_argument(
        "--max-prototype-candidates",
        type=int,
        default=0,
        help="Maximum candidate medoid windows per label. Non-positive means all candidates.",
    )
    parser.add_argument("--retain-segment-banks", action="store_true", help="Keep full raw segment banks in the in-memory context.")
    parser.add_argument("--context-cache-dir", type=Path, default=None, help="Directory for compact raw context .npz caches.")
    parser.add_argument("--resume", action="store_true", help="Resume from fold-level checkpoint CSVs when present.")
    parser.add_argument("--overwrite", action="store_true", help="Ignore existing checkpoints and context caches.")
    parser.add_argument("--fold-output-dir", type=Path, default=None, help="Base directory for fold_<N> checkpoint outputs.")
    parser.add_argument("--skip-completed-samples", action="store_true", help="Skip samples listed in completed_samples.txt during resume.")
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
    context_cache_dir = args.context_cache_dir or (output_dir / "cache" / "raw_context")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    max_prototype_candidates = _none_if_nonpositive(args.max_prototype_candidates)

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
        "context_device": args.context_device,
        "prototype_selection": args.prototype_selection,
        "medoid_block_size": args.medoid_block_size,
        "max_prototype_candidates": max_prototype_candidates,
        "retain_segment_banks": args.retain_segment_banks,
        "context_cache_dir": str(context_cache_dir),
        "context_cache_enabled": not args.retain_segment_banks,
        "context_cache_disabled_reason": "retain_segment_banks requires full banks that compact cache does not store" if args.retain_segment_banks else "",
        "resume": args.resume,
        "overwrite": args.overwrite,
        "fold_output_dir": str(args.fold_output_dir) if args.fold_output_dir is not None else "",
        "skip_completed_samples": args.skip_completed_samples,
        "cuda_runtime": "CUDA 13 via cupy-cuda13x when --device cuda is used",
        "raw_generator": args.raw_generator,
        "input_space": "raw waveform + label",
        "uses_acoustic_features": False,
        "uses_spectrogram": False,
        "uses_mfcc": False,
        "waveform_normalization": False,
        "distance_metric": "masked_mean_squared_distance",
        "resample_method": "scipy.signal.resample_poly",
        "lambda_selection": "all lambda_hyb grid points are reported; test-sample best_lambda is not used for evaluation selection",
        "label_conditioned_raw_generation": "after important segment extraction; skipped when --raw-generator is omitted",
    }
    with (output_dir / "raw_waveform_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")

    all_rows: list[dict[str, object]] = []
    all_method_rows: list[dict[str, object]] = []
    all_errors: list[dict[str, object]] = []
    for fold in folds:
        fold_dir = None
        if args.fold_output_dir is not None:
            fold_dir = args.fold_output_dir / f"fold_{fold}"
        elif args.resume or args.skip_completed_samples:
            fold_dir = output_dir / f"fold_{fold}"
        rows, method_rows, errors = run_fold(
            samples,
            fold=fold,
            output_dir=output_dir,
            fold_output_dir=fold_dir,
            mode=args.mode,
            seed=args.seed,
            target_sr=args.target_sr,
            segment_sec=args.segment_sec,
            hop_sec=args.hop_sec,
            lambda_grid=lambda_grid,
            top_k_segments=args.top_k_segments,
            device=args.device,
            prototype_selection=args.prototype_selection,
            medoid_block_size=args.medoid_block_size,
            max_prototype_candidates=max_prototype_candidates,
            context_device=args.context_device,
            retain_segment_banks=args.retain_segment_banks,
            context_cache_dir=context_cache_dir,
            resume=args.resume,
            overwrite=args.overwrite,
            skip_completed_samples=args.skip_completed_samples,
            raw_generator=generator,
            skip_errors=args.skip_errors,
            save_plots=args.save_plots,
        )
        all_rows.extend(rows)
        all_method_rows.extend(method_rows)
        all_errors.extend(errors)

    _atomic_write_csv(results_dir / "raw_waveform_sample_metrics.csv", all_rows, RAW_SAMPLE_FIELDS)
    _atomic_write_csv(results_dir / "raw_waveform_method_metrics.csv", all_method_rows, RAW_METHOD_FIELDS)
    _atomic_write_csv(results_dir / "raw_waveform_summary_by_lambda.csv", summarize_by_lambda(all_rows))
    if all_errors:
        _atomic_write_csv(results_dir / "raw_waveform_errors.csv", all_errors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
