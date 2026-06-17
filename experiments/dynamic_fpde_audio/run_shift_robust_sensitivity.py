"""Synthetic sensitivity checks for Shift-Robust Raw-Waveform Dynamic-FPDE."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.dynamic_fpde_audio.raw_waveform_context import prepare_fast_raw_waveform_fpde_context
from experiments.dynamic_fpde_audio.shift_robust_raw_waveform import (
    LAMBDA_GRID,
    ShiftAlignmentConfig,
    explain_shift_robust_raw_waveform,
    masked_shift_mse,
    shift_with_mask,
)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    aa = a - float(np.mean(a))
    bb = b - float(np.mean(b))
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return float(np.dot(aa, bb) / denom) if denom > 0.0 else 0.0


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(values.shape[0], dtype=float)
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return _pearson(_rankdata(a), _rankdata(b))


def _topk_jaccard(a: np.ndarray, b: np.ndarray, k: int = 10) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    kk = min(k, a.size, b.size)
    sa = set(np.argsort(np.abs(a))[-kk:].tolist())
    sb = set(np.argsort(np.abs(b))[-kk:].tolist())
    return float(len(sa & sb) / max(1, len(sa | sb)))


def _zero_padded_shift(waveform: np.ndarray, lag_samples: int) -> tuple[np.ndarray, np.ndarray]:
    mask = np.ones(waveform.shape, dtype=bool)
    return shift_with_mask(waveform, mask, lag_samples)


def _synthetic_waveforms(sample_rate: int) -> tuple[list[np.ndarray], list[str]]:
    t = np.arange(int(sample_rate * 0.25), dtype=float) / float(sample_rate)
    a = 0.7 * np.sin(2.0 * np.pi * 90.0 * t) + 0.2 * np.sin(2.0 * np.pi * 210.0 * t)
    b = -0.6 * np.sin(2.0 * np.pi * 120.0 * t) + 0.15 * np.sin(2.0 * np.pi * 250.0 * t)
    a_shifted, _ = _zero_padded_shift(a, 3)
    b_shifted, _ = _zero_padded_shift(b, -3)
    return [a, a_shifted, b, b_shifted], ["class_a", "class_a", "class_b", "class_b"]


def run_sensitivity(output_dir: Path, *, sample_rate: int = 1000) -> Path:
    import fpde

    output_dir.mkdir(parents=True, exist_ok=True)
    waveforms, labels = _synthetic_waveforms(sample_rate)
    context = prepare_fast_raw_waveform_fpde_context(
        fpde,
        waveforms,
        labels,
        sample_rates=[sample_rate] * len(waveforms),
        target_sr=sample_rate,
        segment_sec=0.08,
        hop_sec=0.04,
        prototype_selection="exact_medoid",
    ).context
    base_waveform = waveforms[0]
    shifts_ms = [0, -20, -10, -5, -2, -1, 1, 2, 5, 10, 20]
    modes = ["none", "hard_bounded", "soft_bounded"]
    shift_max_values = [0, 5, 10, 20, 50]
    rows: list[dict[str, object]] = []
    baseline_phi: dict[tuple[str, int, float], np.ndarray] = {}
    baseline_evidence: dict[tuple[str, int, float], float] = {}
    baseline_window_evidence: dict[tuple[str, int, float], np.ndarray] = {}
    for mode in modes:
        for shift_max_ms in shift_max_values:
            for artificial_shift_ms in shifts_ms:
                lag_samples = int(round(sample_rate * artificial_shift_ms / 1000.0))
                shifted_waveform, shifted_mask = _zero_padded_shift(base_waveform, lag_samples)
                shifted_waveform = np.where(shifted_mask, shifted_waveform, 0.0)
                if mode == "none":
                    explanation = fpde.raw_waveform_fpde_explain_one(
                        shifted_waveform,
                        context,
                        sample_rate=sample_rate,
                        target_label="class_a",
                        lambda_grid=LAMBDA_GRID,
                        top_k_segments=1,
                        device="cpu",
                    )
                else:
                    explanation = explain_shift_robust_raw_waveform(
                        fpde,
                        shifted_waveform,
                        context,
                        sample_rate=sample_rate,
                        target_label="class_a",
                        lambda_grid=LAMBDA_GRID,
                        top_k_segments=1,
                        device="cpu",
                        config=ShiftAlignmentConfig(
                            alignment_mode=mode,
                            shift_max_ms=float(shift_max_ms),
                            coarse_step_ms=1.0,
                            fine_radius_ms=2.0,
                            coarse_top_k=3,
                            minimum_overlap_ratio=0.5,
                            alignment_temperature=0.05,
                        ),
                    )
                for lambda_hyb, result in explanation.lambda_results.items():
                    key = (mode, shift_max_ms, float(lambda_hyb))
                    phi = np.asarray(result["phi"], dtype=float)
                    evidence = float(result["evidence"])
                    if artificial_shift_ms == 0:
                        baseline_phi[key] = phi
                        baseline_evidence[key] = evidence
                        baseline_window_evidence[key] = np.asarray(result["window_evidence"], dtype=float)
                    ref_phi = baseline_phi.get(key, phi)
                    ref_evidence = baseline_evidence.get(key, evidence)
                    ref_window_evidence = baseline_window_evidence.get(key, np.asarray(result["window_evidence"], dtype=float))
                    details = dict(result.get("details", {}))
                    target_lags = np.asarray(details.get("target_lags_by_lambda", []), dtype=float)
                    selected_lag = float(np.median(target_lags)) if target_lags.size else 0.0
                    zero_mse, _ = masked_shift_mse(shifted_waveform[: context.segment_length], np.ones(context.segment_length, dtype=bool), context.prototypes["class_a"], context.prototype_masks["class_a"], 0)
                    rows.append(
                        {
                            "alignment_mode": mode,
                            "shift_max_ms": shift_max_ms,
                            "artificial_shift_ms": artificial_shift_ms,
                            "lambda_hyb": float(lambda_hyb),
                            "evidence_difference": evidence - ref_evidence,
                            "evidence_sign_agreement": bool(np.sign(evidence) == np.sign(ref_evidence)),
                            "window_evidence_spearman": _spearman(np.asarray(result["window_evidence"], dtype=float), ref_window_evidence),
                            "phi_pearson": _pearson(phi, ref_phi),
                            "phi_spearman": _spearman(phi, ref_phi),
                            "topk_jaccard": _topk_jaccard(phi, ref_phi),
                            "rival_label_agreement": bool(explanation.rival_label == "class_b"),
                            "selected_lag_error_ms": selected_lag - artificial_shift_ms,
                            "boundary_hit_rate": details.get("target_boundary_hit_rate", ""),
                            "alignment_confidence": details.get("alignment_confidence_mean", ""),
                            "zero_lag_mse_reference": zero_mse,
                        }
                    )
    path = output_dir / "shift_robust_sensitivity.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run synthetic shift-robust Raw-Waveform Dynamic-FPDE sensitivity checks.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/shift_robust_sensitivity"))
    parser.add_argument("--sample-rate", type=int, default=1000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = run_sensitivity(args.output_dir, sample_rate=args.sample_rate)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
