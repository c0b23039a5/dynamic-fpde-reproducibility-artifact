from __future__ import annotations

from pathlib import Path

import pytest

from experiments.dynamic_fpde_audio.aggregate import aggregate_additivity, aggregate_by_method, write_csv
from experiments.dynamic_fpde_audio.tables import generate_tables


def test_metrics_aggregation_groups_by_dataset_fold_and_method():
    rows = [
        {
            "dataset": "esc50",
            "fold": 1,
            "method": "dynamic_hyb",
            "evidence": 1.0,
            "abs_exactness_residual": 0.01,
            "deletion_drop_auc": 0.4,
            "insertion_gain_auc": 0.6,
            "combined_score": 0.5,
            "runtime_sec": 0.1,
        },
        {
            "dataset": "esc50",
            "fold": 1,
            "method": "dynamic_hyb",
            "evidence": 3.0,
            "abs_exactness_residual": 0.03,
            "deletion_drop_auc": 0.6,
            "insertion_gain_auc": 0.8,
            "combined_score": 0.7,
            "runtime_sec": 0.3,
        },
        {
            "dataset": "esc50",
            "fold": 1,
            "method": "energy_baseline",
            "evidence": 2.0,
            "abs_exactness_residual": "",
            "deletion_drop_auc": 0.2,
            "insertion_gain_auc": 0.4,
            "combined_score": 0.3,
            "runtime_sec": 0.2,
        },
    ]

    summary = aggregate_by_method(rows)
    hyb = next(row for row in summary if row["method"] == "dynamic_hyb")
    energy = next(row for row in summary if row["method"] == "energy_baseline")
    additivity = aggregate_additivity(rows)

    assert hyb["combined_score_mean"] == pytest.approx(0.6)
    assert hyb["combined_score_median"] == pytest.approx(0.6)
    assert hyb["n"] == 2
    assert energy["abs_exactness_residual_n"] == 0
    assert additivity[0]["abs_exactness_residual_mean"] == pytest.approx(0.02)
    assert additivity[0]["abs_exactness_residual_std"] == pytest.approx(0.01)


def test_latex_table_generation_reads_csv_values(tmp_path: Path):
    results = tmp_path / "results"
    tables = tmp_path / "tables"
    write_csv(
        results / "dynamic_fpde_summary_by_method.csv",
        [
            {
                "dataset": "esc50",
                "fold": 1,
                "method": "dynamic_hyb",
                "combined_score_mean": 0.75,
                "deletion_drop_auc_mean": 0.7,
                "insertion_gain_auc_mean": 0.8,
                "n": 2,
            }
        ],
    )
    write_csv(
        results / "dynamic_fpde_additivity_summary.csv",
        [{"dataset": "esc50", "fold": 1, "method": "dynamic_hyb", "abs_exactness_residual_mean": 1e-12, "n": 2}],
    )
    write_csv(
        results / "dynamic_fpde_lambda_selection.csv",
        [
            {
                "dataset": "esc50",
                "fold": 1,
                "lambda_hyb": 0.5,
                "mean_combined_score": 0.75,
                "mean_deletion_drop_auc": 0.7,
                "mean_insertion_gain_auc": 0.8,
                "n_eval_samples": 2,
            }
        ],
    )

    written = generate_tables(results, tables)

    assert {path.name for path in written} == {
        "table_dynamic_fpde_main_results.tex",
        "table_dynamic_fpde_additivity.tex",
        "table_dynamic_fpde_lambda.tex",
    }
    main_text = (tables / "table_dynamic_fpde_main_results.tex").read_text(encoding="utf-8")
    assert "\\toprule" in main_text
    assert "dynamic\\_hyb" in main_text
    assert "0.7500" in main_text


def test_table_generation_fails_clearly_when_csv_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="required CSV"):
        generate_tables(tmp_path / "missing-results", tmp_path / "tables")

