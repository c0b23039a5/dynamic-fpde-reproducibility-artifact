# Bayesian-FPDE Reproducibility Artifact

This repository contains a reproducible experimental pipeline for **Bayesian-FPDE**, an uncertainty-aware prototype-contrast feature attribution method for tabular classification.

Repository URL:

```text
https://github.com/c0b23039a5/bayesian-fpde-reproducibility-artifact
```

## Scope

The main experiments use **public benchmark data**. The default public-data configuration uses **OpenML-CC18**, suite ID `99`.

Public benchmark datasets do not provide ground-truth feature attributions. Therefore, the public-data experiments do not report true attribution coverage or known-truth calibration as main results. Instead, they evaluate whether Bayesian-FPDE uncertainty estimates are consistent with empirical variation across random seeds, resampling runs, model-output perturbations, and training-size changes.

Use the term `empirical_reference_coverage_95` for coverage against a leave-one-seed empirical reference. Do not describe this as true coverage.

## Installation

```bash
python -m pip install --upgrade pip
python -m pip install -e .[dev]
python -m pytest
```

For OpenML and optional baselines:

```bash
python -m pip install -e .[dev,openml,baselines]
```

## Public-data configuration

```text
configs/openml_public_4experiments.yaml
```

The configuration defines OpenML suite ID `99`, task IDs, seeds, train/test row limits, explanation count, posterior/bootstrap sample counts, top-k, and training fractions.

The `smoke` mode is only an implementation check. It may use a local OpenML-shaped smoke dataset so CI can run without network access. Do not report smoke-mode numbers as paper results.

## Four public-data experiments

| Experiment | Purpose | Main evaluation |
|---|---|---|
| Public-data uncertainty validation | Check whether uncertainty estimates align with empirical seed/resampling variation | Empirical reference coverage, uncertainty-error correlation, sign agreement |
| Stability experiment | Check whether explanations, ranks, signs, and top-k sets are stable across seeds/splits/resampling | Spearman/Pearson correlation, top-k Jaccard, sign agreement |
| Faithfulness experiment | Check whether replacing important features with a baseline changes model output | Faithfulness correlation, deletion AUC, insertion AUC |
| Training-size uncertainty | Check whether uncertainty decreases and explanations approach full-training references as training data increases | CI width, posterior std, sign confidence, distance to full-training reference |

## Smoke workflow

```bash
python -m pip install -e .[dev]
python -m pytest

python -m experiments.run_public_uncertainty_validation --config configs/openml_public_4experiments.yaml --mode smoke
python -m experiments.run_stability --config configs/openml_public_4experiments.yaml --mode smoke
python -m experiments.run_faithfulness --config configs/openml_public_4experiments.yaml --mode smoke
python -m experiments.run_training_size_uncertainty --config configs/openml_public_4experiments.yaml --mode smoke
python -m experiments.aggregate_results --results-dir results --figures-dir figures
```

## Main outputs

```text
results/public_uncertainty_validation.csv
results/stability_metrics.csv
results/faithfulness_metrics.csv
results/training_size_uncertainty.csv
results/public_uncertainty_validation_summary.csv
results/stability_summary.csv
results/faithfulness_summary.csv
results/training_size_uncertainty_summary.csv
results/statistical_tests.csv
results/effect_sizes.csv
results/bootstrap_confidence_intervals.csv
```

## Metric interpretation

- `empirical_reference_coverage_95` is coverage against a leave-one-seed empirical reference, not true attribution coverage.
- `attribution_distance_to_full_train` is distance to a full-training empirical reference, not distance to a true attribution.
- Faithfulness metrics depend on the chosen baseline. The shared evaluation path currently uses the training-set mean as the baseline.
- Synthetic known-truth calibration may remain as auxiliary code, but it is not the main public-data experiment described here.

## Citation

If you use this artifact, cite the repository and the associated Bayesian-FPDE manuscript or article when available. Citation metadata is provided in `CITATION.cff`.

## License

Unless otherwise stated, this repository is distributed under the Apache License 2.0.
