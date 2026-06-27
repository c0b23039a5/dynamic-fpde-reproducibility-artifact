"""Run and aggregate the IEEE Access RawFeat hybrid-lambda sensitivity grid."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.dynamic_fpde_audio.run_esc50_rawfeat_dynamic_fpde import main as rawfeat_main


LAMBDA_HYB_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
SUMMARY_FIELDS = (
    "lambda_hyb",
    "n",
    "shape_match_rate",
    "audit_pass_rate",
    "max_abs_exactness_residual",
    "mean_absolute_evidence",
    "positive_evidence_rate",
    "raw_group_attribution_mean",
    "feature_group_attribution_mean",
    "dt_group_attribution_mean",
)


def _is_true(value: object) -> bool:
    return str(value).strip().lower() == "true"


def _finite(rows: list[dict[str, object]], column: str) -> np.ndarray:
    values = np.asarray([float(row[column]) for row in rows], dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"column {column!r} must contain finite numeric values")
    return values


def aggregate_by_lambda(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[float, list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(float(row["lambda_hyb"]), []).append(row)
    output: list[dict[str, object]] = []
    for lambda_hyb, group in sorted(groups.items()):
        evidence = _finite(group, "evidence")
        output.append(
            {
                "lambda_hyb": lambda_hyb,
                "n": len(group),
                "shape_match_rate": float(np.mean([_is_true(row["shape_match"]) for row in group])),
                "audit_pass_rate": float(np.mean([_is_true(row["audit_passed"]) for row in group])),
                "max_abs_exactness_residual": float(np.max(_finite(group, "abs_exactness_residual"))),
                "mean_absolute_evidence": float(np.mean(_finite(group, "absolute_evidence"))),
                "positive_evidence_rate": float(np.mean(evidence > 0.0)),
                "raw_group_attribution_mean": float(np.mean(_finite(group, "raw_group_attribution"))),
                "feature_group_attribution_mean": float(np.mean(_finite(group, "feature_group_attribution"))),
                "dt_group_attribution_mean": float(np.mean(_finite(group, "dt_group_attribution"))),
            }
        )
    return output


def _lambda_dir(output_dir: Path, value: float) -> Path:
    return output_dir / "runs" / f"lambda_{value:.2f}".replace(".", "p")


def _read_csv(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"sensitivity input not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/ESC-50"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/rawfeat_dynamic_fpde_esc50_lambda_sensitivity"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--frame-length", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--normalize", choices=("none", "l1"), default="l1")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    all_rows: list[dict[str, object]] = []
    for lambda_hyb in LAMBDA_HYB_VALUES:
        run_dir = _lambda_dir(args.output_dir, lambda_hyb)
        metrics_path = run_dir / "results" / "rawfeat_sample_metrics.csv"
        if not args.aggregate_only and (args.overwrite or not metrics_path.is_file()):
            run_args = [
                "--dataset-root", str(args.dataset_root),
                "--output-dir", str(run_dir),
                "--mode", "full",
                "--folds", "1,2,3,4,5",
                "--seed", str(args.seed),
                "--target-sr", str(args.target_sr),
                "--frame-length", str(args.frame_length),
                "--hop-length", str(args.hop_length),
                "--lambda-hyb", str(lambda_hyb),
                "--normalize", args.normalize,
                "--generation-scope", "none",
            ]
            if args.overwrite or (run_dir / "rawfeat_config.json").exists():
                run_args.append("--overwrite")
            rawfeat_main(run_args)
        lambda_rows = _read_csv(metrics_path)
        for row in lambda_rows:
            if not np.isclose(float(row["lambda_hyb"]), lambda_hyb):
                raise ValueError(f"unexpected lambda_hyb in {metrics_path}: {row['lambda_hyb']}")
        all_rows.extend(lambda_rows)

    results_dir = args.output_dir / "results"
    combined_fields = list(all_rows[0]) if all_rows else []
    _write_csv(results_dir / "rawfeat_lambda_sensitivity_samples.csv", all_rows, combined_fields)
    summary = aggregate_by_lambda(all_rows)
    _write_csv(results_dir / "rawfeat_lambda_sensitivity_summary.csv", summary, SUMMARY_FIELDS)
    print(results_dir / "rawfeat_lambda_sensitivity_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
