# Feature Prototype Direction Explainer (FPDE) Reproducibility Artifact

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20225275.svg)](https://doi.org/10.5281/zenodo.20225275)

This package contains code, documentation, and precomputed outputs for the manuscript:

**Feature Prototype Direction Explainer: Auditable Prototype-Contrast Attributions for Black-Box Classification**

Author: Yuki Kato, School of Computer Science, Tokyo University of Technology.

This package is a public reproducibility artifact for the FPDE manuscript. It supports independent inspection of the experimental workflow and includes generated table artifacts for reproducibility.

## GitHub quick start

This repository is available as a public GitHub reproducibility artifact. The public release includes repository-level files for reviewer inspection and long-term access:

- `LICENSE` with the Apache License 2.0;
- `.gitignore` for Python caches, virtual environments, OpenML caches, and recomputed outputs;
- `.gitattributes` for line-ending normalization and generated-result metadata;
- `.github/workflows/artifact-check.yml` for lightweight CI;
- `docs/repository_metadata.md` with repository metadata.

Public repository URL:

```text
https://github.com/fpde-xai/fpde-reproducibility-artifact
```

Artifact DOI: [10.5281/zenodo.20225275](https://doi.org/10.5281/zenodo.20225275)


## 1. Artifact identification

The artifact supports the experiments for FPDE, a lightweight prototype-based post-hoc attribution method for black-box classification. The manuscript evaluates Diff-FPDE, Cos-FPDE, and Hyb-FPDE against AIME, LIME, and SHAP on OpenML-CC18 using LightGBM.

The artifact contributes to reproducibility by providing:

- the FPDE implementation as an installable GitHub dependency from `fpde-xai/fpde`;
- the OpenML-CC18 experimental runner;
- scripts for per-task/per-seed reruns;
- precomputed 72-task, 10-seed result folders included with this artifact;
- generated LaTeX tables and audit CSV files;
- a table-generation script that reconstructs the manuscript tables from the precomputed result archive.

## 2. Artifact contents

```text
code/
  run_fpde_openml_cc18_experimental_aime_clean.py
scripts/
  aggregate_precomputed_results.py
  build_reproducibility_tables.py
  run_full_seed.py
docs/
  reproducibility_checklist.md
  data_and_code_availability_statement.md
  repository_metadata.md
generated/
  generated_tex/
    main_results_subject_area_balanced.tex
    task_weighted_results.tex
    stat_tests.tex
    lambda_distribution.tex
    ...
  processed_csv/
    summary_by_task_seed_filtered.csv
    subject_balanced_seed_method.csv
    task_balanced_seed_method.csv
    ...
results_precomputed/
  fpde-openml-task-<task-id>-seed-<seed>-<run-id>/
    experimental_outputs_aime_full_n200_seed<seed>_task<task-id>/
      summary_by_task.csv
      summary_by_method.csv
      per_instance_results.csv
      lambda_distribution.csv
      run_config.json
requirements.txt
environment.yml
Dockerfile
CITATION.cff
LICENSE
.gitattributes
.gitignore
.github/workflows/artifact-check.yml
.github/workflows/full-seeds.yml
RELEASE_NOTES.md
```

## 3. Dependencies and requirements

### Hardware

A standard CPU environment is sufficient. GPU is not required. Full 10-seed reproduction can be time-consuming because all 72 OpenML-CC18 tasks and six explanation methods are evaluated.

### Operating system

The scripts are plain Python and should run on Linux, macOS, Windows, Google Colab, or a Linux container. The included Dockerfile targets Linux with Python 3.14.

### Software

Recommended Python version: **Python 3.14**.

Install dependencies with either:

```bash
python -m pip install -r requirements.txt
```

or:

```bash
conda env create -f environment.yml
conda activate fpde-repro
```

Main Python dependencies:

- fpde, installed from [https://github.com/fpde-xai/fpde](https://github.com/fpde-xai/fpde)
- numpy
- pandas
- tabulate
- jinja2
- scikit-learn
- lightgbm
- openml
- shap
- lime
- aime-xai

### Dataset

The experiments use **OpenML-CC18**, benchmark suite ID **99**. The runner downloads task data from OpenML at execution time. This artifact does not redistribute OpenML datasets.

## 4. Installation and deployment

From the artifact root:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/build_reproducibility_tables.py \
  --input results_precomputed \
  --output generated \
  --expected-tasks 72 \
  --expected-seeds 10
```

Docker option:

```bash
docker build -t fpde-repro .
docker run --rm fpde-repro
```

Approximate installation time depends on network and package compilation. On a typical cloud notebook or Linux container, Python package installation is expected to take several minutes. Full execution time depends strongly on CPU resources and optional baseline packages.

## 5. Reproducibility of experiments

### 5.0 Bayesian-FPDE uncertainty pipeline

This repository also includes a clean-room Bayesian-FPDE experimental pipeline
for evaluating posterior uncertainty in prototype-contrast attributions. The
pipeline does not contain fabricated numerical results: every CSV, Parquet
file, and figure is generated by running the scripts below.

Smoke runs are intentionally small and are suitable for checking the complete
pipeline locally:

```bash
python -m pip install -e .[dev]
python -m pytest
python -m experiments.run_synthetic_calibration --config configs/synthetic.yaml --mode smoke
python -m experiments.run_openml_benchmark --config configs/openml_cc18.yaml --mode smoke
python -m experiments.run_stability --config configs/openml_cc18.yaml --mode smoke
python -m experiments.run_faithfulness --config configs/openml_cc18.yaml --mode smoke
python -m experiments.run_training_size_uncertainty --config configs/synthetic.yaml --mode smoke
python -m experiments.run_ablation --config configs/ablation.yaml --mode smoke
python -m experiments.run_case_studies --config configs/case_study.yaml --mode smoke
python -m experiments.aggregate_results --results-dir results --figures-dir figures
```

The OpenML smoke mode uses a local OpenML-shaped tabular dataset so that CI and
local validation do not depend on network availability. Medium and full modes in
`configs/openml_cc18.yaml` use OpenML-CC18 task IDs from configuration.

Main generated outputs include:

- `results/synthetic_calibration.csv`
- `results/synthetic_calibration_summary.csv`
- `results/openml_local_explanations.parquet`
- `results/openml_global_summary.csv`
- `results/openml_metrics.csv`
- `results/openml_runtime.csv`
- `results/stability_metrics.csv`
- `results/faithfulness_metrics.csv`
- `results/training_size_uncertainty.csv`
- `results/ablation_metrics.csv`
- `results/statistical_tests.csv`
- `results/effect_sizes.csv`
- `results/bootstrap_confidence_intervals.csv`
- figures under `figures/`

Bayesian-FPDE reports posterior mean, posterior standard deviation, 95% credible
intervals, posterior sign probabilities, `P(|phi_j| > tau)`, mean/std ranks,
and top-k rank probabilities. Optional SHAP, LIME, and AIME baselines are
recorded as `skipped` when their dependencies or adapters are unavailable; they
are not silently ignored.

Synthetic calibration uses known true prototype-contrast attributions from the
data generator. `coverage_95` measures whether the 95% credible interval
contains the true attribution. Sign calibration treats exactly zero true
attributions as neutral and excludes them from sign-calibration denominators;
the ignored count is reported. The reported sign metrics are:

- `sign_brier_score`: Brier score for predicted sign confidence versus whether
  the predicted sign is correct.
- `sign_ece`: expected calibration error over confidence bins.
- `sign_accuracy_at_confidence_0_8` and
  `sign_accuracy_at_confidence_0_9`: empirical sign accuracy among features
  whose predicted sign confidence is at least the threshold.
- `synthetic_sign_calibration_bins.csv`: reliability-bin audit table.

Faithfulness uses model-output deltas, not distance from the baseline. For each
feature, the runner replaces only that feature with the baseline value, measures
the drop in black-box predicted probability for the target class, and reports
the Spearman correlation between `|attribution_j|` and `|delta_j|` as
`faithfulness_correlation`.

Full OpenML-CC18 Bayesian experiments are intended to run through the manual
GitHub Actions workflow **Bayesian OpenML experiment**:

```text
.github/workflows/bayesian-openml.yml
```

Start with a small manual run such as `max_tasks=1`, `seeds=[0]`,
`n_explain=5`, and `posterior_samples=20`; then increase to `medium` or `full`
once the workflow and optional baseline dependencies are behaving as expected.
Every push and pull request also runs **Bayesian pipeline CI**, which installs
the local package, runs `pytest`, executes smoke experiments, aggregates the
outputs, and uploads `results/`, `figures/`, and `logs/` for debugging.

### 5.1 Reconstruct manuscript tables

```bash
python scripts/build_reproducibility_tables.py \
  --input results_precomputed \
  --output generated \
  --expected-tasks 72 \
  --expected-seeds 10
```

The script verifies completeness for 72 OpenML-CC18 tasks, 10 seeds, and the four reported methods: Hyb-FPDE, AIME, SHAP, and LIME. It excludes Diff-FPDE and Cos-FPDE from the reported comparison because they are internal components of Hyb-FPDE in this manuscript.

Expected output folders:

```text
generated/generated_tex/
generated/processed_csv/
generated/manifest.json
```

The checked-in `generated/` directory contains the current generated tables and audit CSV files.

### 5.2 Full per-task/per-seed rerun

```bash
python scripts/run_full_seed.py --seed 1 --task-id 10093
```

The full run configuration used for the included results is:

- OpenML benchmark suite: 99
- tasks: 72 OpenML-CC18 tasks
- seeds: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
- fold/repeat/sample: 0/0/0
- explained test instances per task: 200
- validation instances for Hyb-FPDE lambda selection: 200
- LightGBM estimators: 100
- learning rate: 0.05
- num leaves: 31
- methods: diff_fpde, cos_fpde, hyb_fpde_grid, shap, lime, aime
- AIME local target vector: onehot

Recomputed outputs are written to:

```text
results_recomputed/experimental_outputs_aime_full_n200_seed<seed>_task<task-id>/
```

For GitHub Actions, use the manual workflow **Full seed OpenML run**. It accepts
`max_tasks`, `suite_id`, `task_ids`, `n_explain`, `n_val_select`, `methods`, and
`seeds` inputs. The default seed list is `[1,2,3,4,5,6,7,8,9,10]`.

## 6. Expected results

The precomputed result archive contains 720 task--seed folders. Each run folder contains:

- `per_instance_results.csv`
- `summary_by_task.csv`
- `summary_by_method.csv`
- `lambda_distribution.csv`
- `main_results_table.tex`
- `lambda_distribution_table.tex`
- `run_config.json`

The generated tables and audit CSVs are checked in under `generated/`.

## 7. Artifact status

- Code included: yes
- Precomputed outputs included: yes, 72 tasks x 10 seeds
- Public DOI included: yes, [10.5281/zenodo.20225275](https://doi.org/10.5281/zenodo.20225275)
- Public repository URL included: yes, [https://github.com/fpde-xai/fpde-reproducibility-artifact](https://github.com/fpde-xai/fpde-reproducibility-artifact)
- License finalized: yes, Apache License 2.0 included
- Manuscript file included: no


## 8. License

Unless otherwise stated, this repository is distributed under the Apache License 2.0. The precomputed result files are included as generated reproducibility artifacts. If you reuse the code or generated results in a publication, cite the associated manuscript/article and this repository using `CITATION.cff`.
