from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.make_rawfeat_ieee_tables import generate
from scripts.run_rawfeat_lambda_sensitivity import aggregate_by_lambda


def _row(*, fold: int, evidence: float, lambda_hyb: float = 0.5) -> dict[str, object]:
    return {
        "fold": fold,
        "lambda_hyb": lambda_hyb,
        "evidence": evidence,
        "absolute_evidence": abs(evidence),
        "abs_exactness_residual": 0.0,
        "audit_passed": True,
        "shape_match": True,
        "raw_group_attribution": evidence * 0.25,
        "feature_group_attribution": evidence * 0.75,
        "dt_group_attribution": 0.0,
    }


def test_ieee_table_and_figure_generation(tmp_path: Path):
    pytest.importorskip("matplotlib")
    input_csv = tmp_path / "rawfeat_sample_metrics.csv"
    rows = [_row(fold=1, evidence=-2.0), _row(fold=2, evidence=1.0)]
    with input_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    written = generate(input_csv, tmp_path / "tables", tmp_path / "figures")

    assert len(written) == 6
    assert all(path.is_file() and path.stat().st_size > 0 for path in written)
    assert "Positive & 1 (0.5000)" in (tmp_path / "tables/rawfeat_evidence_sign.tex").read_text()
    assert "Maximum absolute residual & 0.000e+00" in (
        tmp_path / "tables/rawfeat_audit_summary.tex"
    ).read_text()


def test_lambda_sensitivity_aggregation_has_ieee_metrics():
    rows = [
        _row(fold=1, evidence=-2.0, lambda_hyb=0.0),
        _row(fold=2, evidence=1.0, lambda_hyb=0.0),
        _row(fold=1, evidence=3.0, lambda_hyb=1.0),
    ]

    summary = aggregate_by_lambda(rows)

    assert [row["lambda_hyb"] for row in summary] == [0.0, 1.0]
    assert summary[0]["n"] == 2
    assert summary[0]["shape_match_rate"] == 1.0
    assert summary[0]["audit_pass_rate"] == 1.0
    assert summary[0]["max_abs_exactness_residual"] == 0.0
    assert summary[0]["mean_absolute_evidence"] == 1.5
    assert summary[0]["positive_evidence_rate"] == 0.5
    assert summary[0]["raw_group_attribution_mean"] == -0.125
    assert summary[0]["feature_group_attribution_mean"] == -0.375
    assert summary[0]["dt_group_attribution_mean"] == 0.0
