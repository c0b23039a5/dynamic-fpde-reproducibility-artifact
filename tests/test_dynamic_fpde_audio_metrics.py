from __future__ import annotations

from pathlib import Path

import pytest

from experiments.dynamic_fpde_audio.aggregate import (
    aggregate_additivity,
    aggregate_by_method,
    average_random_repetitions,
    positive_margin_rows,
    write_csv,
)
from experiments.dynamic_fpde_audio.tables import generate_tables


def test_metrics_aggregation_groups_by_dataset_fold_and_method():
    rows = [
        {
            "dataset": "esc50",
            "fold": 1,
            "method": "dynamic_hyb",
            "evidence": 1.0,
            "prototype_margin": 1.0,
            "selection_margin": 4.0,
            "evaluation_evidence": 1.0,
            "evaluation_margin": 1.0,
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
            "prototype_margin": -3.0,
            "selection_margin": -4.0,
            "evaluation_evidence": 3.0,
            "evaluation_margin": 3.0,
            "abs_exactness_residual": 0.03,
            "deletion_drop_auc": 0.6,
            "insertion_gain_auc": 0.8,
            "combined_score": 0.7,
            "runtime_sec": 0.3,
        },
        {
            "dataset": "esc50",
            "fold": 1,
            "method": "energy_baseline_raw",
            "evidence": 2.0,
            "prototype_margin": 2.0,
            "selection_margin": 4.0,
            "evaluation_evidence": 2.0,
            "evaluation_margin": 2.0,
            "abs_exactness_residual": "",
            "deletion_drop_auc": 0.2,
            "insertion_gain_auc": 0.4,
            "combined_score": 0.3,
            "runtime_sec": 0.2,
        },
    ]

    summary = aggregate_by_method(rows)
    hyb = next(row for row in summary if row["method"] == "dynamic_hyb")
    energy = next(row for row in summary if row["method"] == "energy_baseline_raw")
    additivity = aggregate_additivity(rows)

    assert hyb["combined_score_mean"] == pytest.approx(0.6)
    assert hyb["combined_score_median"] == pytest.approx(0.6)
    assert hyb["n"] == 2
    assert hyb["n_unique_samples"] == 2
    assert hyb["n_rows"] == 2
    assert hyb["random_repetitions_mean"] == ""
    assert hyb["prototype_margin_mean"] == pytest.approx(-1.0)
    assert hyb["prototype_margin_median"] == pytest.approx(-1.0)
    assert hyb["prototype_margin_positive_rate"] == pytest.approx(0.5)
    assert hyb["n_positive_margin"] == 1
    assert hyb["n_negative_margin"] == 1
    assert hyb["selection_margin_mean"] == pytest.approx(0.0)
    assert hyb["selection_margin_median"] == pytest.approx(0.0)
    assert hyb["selection_margin_positive_rate"] == pytest.approx(0.5)
    assert hyb["n_selection_positive_margin"] == 1
    assert hyb["n_selection_negative_margin"] == 1
    assert energy["abs_exactness_residual_n"] == 0
    assert additivity[0]["abs_exactness_residual_mean"] == pytest.approx(0.02)
    assert additivity[0]["abs_exactness_residual_std"] == pytest.approx(0.01)


def test_positive_margin_summary_filters_on_selection_margin_not_method_margin():
    rows = [
        {"dataset": "esc50", "fold": 1, "method": "dynamic_hyb", "prototype_margin": -1.0, "selection_margin": 2.0, "combined_score": 0.7},
        {"dataset": "esc50", "fold": 1, "method": "dynamic_hyb", "prototype_margin": 3.0, "selection_margin": 0.0, "combined_score": 0.5},
        {"dataset": "esc50", "fold": 1, "method": "dynamic_hyb", "prototype_margin": 4.0, "selection_margin": -1.0, "combined_score": 0.3},
    ]

    positive_rows = positive_margin_rows(rows)
    summary = aggregate_by_method(positive_rows)

    assert len(positive_rows) == 1
    assert positive_rows[0]["combined_score"] == pytest.approx(0.7)
    assert summary[0]["n"] == 1
    assert summary[0]["prototype_margin_positive_rate"] == pytest.approx(0.0)
    assert summary[0]["selection_margin_positive_rate"] == pytest.approx(1.0)


def test_selection_positive_margin_summary_uses_same_sample_count_per_method():
    rows = []
    for sample_id, selection_margin in [("a", 1.0), ("b", -1.0), ("c", 2.0)]:
        for method, prototype_margin in [
            ("dynamic_diff", selection_margin),
            ("dynamic_cos", -selection_margin),
            ("dynamic_hyb", selection_margin / 2.0),
            ("energy_baseline_raw", -selection_margin),
            ("feature_norm_baseline_standardized", -selection_margin),
            ("random_baseline", selection_margin),
        ]:
            rows.append(
                {
                    "dataset": "esc50",
                    "fold": 1,
                    "sample_id": sample_id,
                    "method": method,
                    "prototype_margin": prototype_margin,
                    "selection_margin": selection_margin,
                    "combined_score": 0.5,
                }
            )

    summary = aggregate_by_method(positive_margin_rows(rows))

    assert {row["method"]: row["n"] for row in summary} == {
        "dynamic_diff": 2,
        "dynamic_cos": 2,
        "dynamic_hyb": 2,
        "energy_baseline_raw": 2,
        "feature_norm_baseline_standardized": 2,
        "random_baseline": 2,
    }


def test_random_baseline_repetitions_are_averaged_before_method_summary():
    rows = [
        {
            "dataset": "esc50",
            "fold": 1,
            "seed": 0,
            "sample_id": "a",
            "method": "dynamic_diff",
            "prototype_margin": 2.0,
            "selection_margin": 2.0,
            "combined_score": 0.6,
            "deletion_drop_auc": 0.5,
            "insertion_gain_auc": 0.7,
            "runtime_sec": 0.1,
        },
        {
            "dataset": "esc50",
            "fold": 1,
            "seed": 0,
            "sample_id": "a",
            "method": "random_baseline",
            "prototype_margin": 2.0,
            "selection_margin": 2.0,
            "combined_score": 0.2,
            "deletion_drop_auc": 0.1,
            "insertion_gain_auc": 0.3,
            "runtime_sec": 0.2,
            "random_repetition": 0,
            "aggregation_unit": "sample_repetition",
        },
        {
            "dataset": "esc50",
            "fold": 1,
            "seed": 0,
            "sample_id": "a",
            "method": "random_baseline",
            "prototype_margin": 4.0,
            "selection_margin": 2.0,
            "combined_score": 0.8,
            "deletion_drop_auc": 0.7,
            "insertion_gain_auc": 0.9,
            "runtime_sec": 0.4,
            "random_repetition": 1,
            "aggregation_unit": "sample_repetition",
        },
    ]

    averaged = average_random_repetitions(rows)
    summary = aggregate_by_method(averaged)
    random_row = next(row for row in averaged if row["method"] == "random_baseline")
    random_summary = next(row for row in summary if row["method"] == "random_baseline")

    assert random_row["aggregation_unit"] == "sample"
    assert random_row["combined_score"] == pytest.approx(0.5)
    assert random_row["prototype_margin"] == pytest.approx(3.0)
    assert random_summary["n"] == 1
    assert random_summary["n_unique_samples"] == 1
    assert random_summary["n_rows"] == 2
    assert random_summary["random_repetitions_mean"] == pytest.approx(2.0)
    assert random_summary["combined_score_mean"] == pytest.approx(0.5)


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
                "prototype_margin_mean": 0.25,
                "prototype_margin_median": 0.25,
                "prototype_margin_positive_rate": 1.0,
                "n_positive_margin": 2,
                "n_negative_margin": 0,
                "selection_margin_mean": 0.4,
                "selection_margin_median": 0.4,
                "selection_margin_positive_rate": 1.0,
                "n_selection_positive_margin": 2,
                "n_selection_negative_margin": 0,
                "combined_score_mean": 0.75,
                "deletion_drop_auc_mean": 0.7,
                "insertion_gain_auc_mean": 0.8,
                "n": 2,
                "n_unique_samples": 2,
                "n_rows": 2,
            }
        ],
    )
    write_csv(
        results / "dynamic_fpde_summary_positive_margin_by_method.csv",
        [
            {
                "dataset": "esc50",
                "fold": 1,
                "method": "dynamic_hyb",
                "prototype_margin_mean": 0.25,
                "prototype_margin_median": 0.25,
                "prototype_margin_positive_rate": 1.0,
                "n_positive_margin": 2,
                "n_negative_margin": 0,
                "selection_margin_mean": 0.4,
                "selection_margin_median": 0.4,
                "selection_margin_positive_rate": 1.0,
                "n_selection_positive_margin": 2,
                "n_selection_negative_margin": 0,
                "combined_score_mean": 0.75,
                "deletion_drop_auc_mean": 0.7,
                "insertion_gain_auc_mean": 0.8,
                "n": 2,
                "n_unique_samples": 2,
                "n_rows": 2,
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
    write_csv(
        results / "dynamic_fpde_sample_metrics.csv",
        [
            {
                "dataset": "esc50",
                "fold": 1,
                "method": "dynamic_hyb",
                "shape_preserved": True,
                "target_prototype_source_sample_id": "train_a",
                "rival_prototype_source_sample_id": "train_b",
                "abs_exactness_residual": 1e-12,
            }
        ],
    )

    written = generate_tables(results, tables)

    assert {path.name for path in written} == {
        "table_dynamic_fpde_main_results.tex",
        "table_dynamic_fpde_positive_margin_results.tex",
        "table_dynamic_fpde_margin_summary.tex",
        "table_dynamic_fpde_additivity.tex",
        "table_dynamic_fpde_lambda.tex",
        "table_dynamic_fpde_native_time_checks.tex",
    }
    main_text = (tables / "table_dynamic_fpde_main_results.tex").read_text(encoding="utf-8")
    assert "\\toprule" in main_text
    assert "Native-Time Dynamic-Hyb" in main_text
    assert "0.7500" in main_text
    assert "Unique N" in main_text
    assert "Rows" in main_text
    margin_text = (tables / "table_dynamic_fpde_margin_summary.tex").read_text(encoding="utf-8")
    assert "Selection Positive Rate" in margin_text
    assert "1.0000" in margin_text


def test_table_generation_fails_clearly_when_csv_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="required CSV"):
        generate_tables(tmp_path / "missing-results", tmp_path / "tables")
