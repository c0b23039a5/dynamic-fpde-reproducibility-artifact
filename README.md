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
- `results/synthetic_sign_calibration_bins.csv`
- `results/openml_local_explanations.parquet`
- `results/openml_seed_summary.csv`
- `results/openml_global_summary.csv`
- `results/openml_method_summary.csv`
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

Full synthetic calibration for paper reporting is configured separately in
`configs/synthetic_full.yaml`:

```bash
python -m experiments.run_synthetic_calibration --config configs/synthetic_full.yaml --mode smoke
python -m experiments.run_synthetic_calibration --config configs/synthetic_full.yaml --mode pilot
python -m experiments.run_synthetic_calibration --config configs/synthetic_full.yaml --mode full
```

Use `smoke` for end-to-end checks, `pilot` for runtime estimation and output
validation, and `full` for paper-scale synthetic calibration. Pilot outputs are
engineering checks only and must not be treated as final paper results. The full run
covers `n_samples=[50,100,500,1000]`,
`n_features=[10,50,100]`, `n_informative=[3,5,10]`, three class-separation
levels, independent/correlated features, balanced/imbalanced classes, and five
seeds. The pilot grid is deliberately smaller and must not be reported as final
paper results. All modes write:

- `results/synthetic_calibration.csv`
- `results/synthetic_calibration_summary.csv`
- `results/synthetic_sign_calibration_bins.csv`
- `figures/synthetic_coverage_vs_n.png`
- `figures/synthetic_ci_width_vs_n.png`
- `figures/synthetic_sign_ece_vs_n.png`
- `figures/synthetic_topk_precision.png`

The full grid is intentionally large. Run the smoke mode first, then launch the
full command on a machine or CI runner with enough CPU time. Do not copy numbers
into the manuscript unless they were produced by an actual completed run.
For partial local runs, the synthetic runner accepts filters such as:

```bash
python -m experiments.run_synthetic_calibration \
  --config configs/synthetic_full.yaml \
  --mode full \
  --seed 0 \
  --n-samples 50,100 \
  --n-features 10 \
  --feature-correlation independent \
  --class-balance balanced
```

The optional GitHub Actions workflow `.github/workflows/synthetic-calibration.yml`
can run `smoke`, `pilot`, or `full` by manual dispatch. It supports input
overrides for `seeds`, `posterior_samples`, and `n_explain`. For full mode, the
workflow splits work across a matrix over seed, sample-size group, feature-size
group, feature correlation, and class balance. Each matrix job uploads partial
synthetic outputs; the aggregate job combines them into the same CSV files and
four figures listed above while preserving SHA and hash metadata as strings.
Failed synthetic conditions are recorded with `status="error"` and an
`error_message` in the summary instead of being silently dropped.

After a completed full synthetic run has produced
`results/synthetic_calibration_summary.csv`, create paper-ready analysis tables
and figures with:

```bash
python -m experiments.analyze_synthetic_full \
  --input results/synthetic_calibration_summary.csv \
  --results-dir results \
  --figures-dir figures
```

This command reads only the completed summary CSV and does not fabricate or
impute missing experiment outcomes. It writes:

- `results/synthetic_full_method_summary.csv`
- `results/synthetic_full_by_n_samples.csv`
- `results/synthetic_full_by_n_features.csv`
- `results/synthetic_full_by_class_separation.csv`
- `results/synthetic_full_by_effective_warning.csv`
- `results/synthetic_full_low_effective_warning_summary.csv`
- `figures/synthetic_full_coverage_by_method.png`
- `figures/synthetic_full_coverage_by_n_samples.png`
- `figures/synthetic_full_ci_width_by_n_samples.png`
- `figures/synthetic_full_sign_ece_by_n_samples.png`
- `figures/synthetic_full_topk_precision_by_n_samples.png`
- `figures/synthetic_full_effective_n_explain_by_n_samples.png`
- `figures/synthetic_full_warning_rate_heatmap.png`

For synthetic calibration, `coverage_95` is the empirical coverage of nominal
95% posterior credible intervals against the known true attribution values.
Values below 0.95 indicate undercoverage; the analysis tables therefore include
`coverage_gap_from_95`, `abs_coverage_gap_from_95`, and undercoverage counts and
rates. The `low_effective_n_explain_warning` flag indicates that too few
correctly classified test examples were available for the requested explanation
count in a condition. Rows and figures stratified by that warning should be
consulted before interpreting calibration, especially for small or difficult
synthetic settings. Pilot results remain runtime and output-shape checks only;
they should not be used as final paper results.

Bayesian-FPDE reports posterior mean, posterior standard deviation, 95% credible
intervals, posterior sign probabilities, `P(|phi_j| > tau)`, mean/std ranks,
and top-k rank probabilities. Optional SHAP, LIME, and AIME baselines are
recorded as `skipped` when their dependencies or adapters are unavailable; they
are not silently ignored.

Synthetic calibration uses known true prototype-contrast attributions from the
data generator. `coverage_95` measures whether the 95% credible interval
contains the true attribution. Sign calibration treats exactly zero true
attributions as neutral and excludes them from sign-calibration denominators;
the ignored count is reported. The reported synthetic calibration metrics
include `coverage_95`, `mean_ci_width`, `median_ci_width`, `sign_accuracy`,
`top_k_precision`, `spearman_rank_correlation`, and `kendall_tau`. The reported
sign calibration metrics are:

- `sign_brier_score`: Brier score for predicted sign confidence versus whether
  the predicted sign is correct.
- `sign_ece`: expected calibration error over confidence bins.
- `sign_accuracy_at_confidence_0_8` and
  `sign_accuracy_at_confidence_0_9`: empirical sign accuracy among features
  whose predicted sign confidence is at least the threshold.
- `synthetic_sign_calibration_bins.csv`: reliability-bin audit table. In this
  file, `n_features` remains the condition-level synthetic feature count
  (`10`, `50`, or `100` in the full grid). `bin_feature_count` is the number of
  non-neutral true-attribution features assigned to a reliability bin, and
  `bin_weight` is the fraction of non-neutral features in that bin.

Synthetic summaries also report explanation-count metadata:

- `requested_n_explain`: requested maximum number of explained test instances.
- `effective_n_explain`: actual number of explained instances selected for the
  condition.
- `n_explanation_rows`: attribution rows contributing to the condition summary;
  normally `effective_n_explain * n_features`.
- `n_unique_explanation_units`: unique explained instances contributing to the
  condition summary.
- `selection_policy`: currently `correctly_classified_only`, falling back to
  `all_test_samples` if no correctly classified test samples are available.
- `low_effective_n_explain_warning`: true when `effective_n_explain` is below
  the configured threshold, defaulting to half of `requested_n_explain`.

Inspect `low_effective_n_explain_warning` before interpreting pilot or full
calibration metrics, especially for small or difficult synthetic conditions.

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

OpenML benchmark summaries are split by aggregation level:

- `openml_seed_summary.csv`: one row per `dataset_name`, `task_id`, `seed`,
  and `method`.
- `openml_global_summary.csv`: one row per `dataset_name`, `task_id`, and
  `method`, averaged across explained instances and seeds.
- `openml_method_summary.csv`: one row per method, averaged across datasets and
  seeds.

Summary files report counts, status counts, and means of meaningful metrics
only; identifiers such as explained indices and labels are not averaged.
Explanation counts are explicit:

- `n_explanation_rows`: total explanation rows contributing to the summary.
- `n_unique_explanation_units`: unique explanation units defined by
  `dataset_name`, `task_id`, `seed`, `fold`, and `explained_index`.
- `mean_explanation_units_per_dataset_seed`,
  `min_explanation_units_per_dataset_seed`, and
  `max_explanation_units_per_dataset_seed`: explanation-unit counts per
  dataset/seed/fold unit. For example, 10 datasets x 3 seeds x 50 explained
  samples should produce 1500 unique explanation units per method.
- `n_unique_explained_indices`, `n_explain_instances`,
  `mean_explain_instances_per_seed`, `min_explain_instances_per_seed`, and
  `max_explain_instances_per_seed` are retained only as deprecated
  backward-compatible aliases for the explanation-unit fields.

Runtime summaries distinguish all attempted rows from comparable successful
rows:

- `mean_runtime_seconds_all_rows`: mean over `ok`, `skipped`, and `error` rows.
- `mean_runtime_seconds_ok_only`: mean over `ok` rows only. Method comparison
  summaries and runtime plots use this ok-only value; methods with only skipped
  rows have `NaN` for ok-only runtime.

Result rows separate model-call accounting:

- `explanation_model_calls`: approximate model calls needed to generate the
  explanation.
- `evaluation_model_calls`: model calls used for deletion/insertion,
  comprehensiveness, sufficiency, and per-feature faithfulness evaluation.
- `total_model_calls`: the sum of explanation and evaluation calls.
- `number_of_model_calls`: deprecated backward-compatible alias for
  `total_model_calls`.

Configuration hashes are also split:

- `experiment_config_hash`: paper-level experiment identity. It is computed
  from the selected experiment configuration and mode, or injected by the
  manual OpenML workflow. It must be constant across all task_id x seed matrix
  jobs that belong to one paper-level OpenML run.
- `workflow_run_id`: GitHub Actions run ID when available. It should be
  constant across all matrix jobs in one workflow run.
- `workflow_run_attempt`, `workflow_name`, `workflow_ref`, and `workflow_sha`:
  GitHub Actions provenance fields. `workflow_sha` is the exact commit SHA used
  by the workflow.
- `runner_invocation_hash`: hash of the actual per-runner config. In the
  OpenML matrix workflow this may vary across task_id x seed jobs because each
  job receives a narrowed generated config.
- `job_config_hash`: hash of the dataset/seed/fold/job-specific configuration;
  it is expected to vary across task IDs, datasets, seeds, folds, splits, and
  method/evaluation settings.
- `run_config_hash`: deprecated backward-compatible alias for
  `runner_invocation_hash`.
- `config_hash`: deprecated backward-compatible alias for
  `experiment_config_hash`, which is the preferred paper-level identity.

String-like metadata are read and written as strings during artifact
combination and aggregation. In particular, `git_commit` and `workflow_sha`
must remain exact SHA strings and must not be parsed into floats or scientific
notation, even when a SHA starts with digits and contains `e`.

Aggregate outputs expose hash consistency instead of hiding it:

- `n_experiment_config_hashes`: number of distinct paper-level experiment
  hashes in the aggregated input.
- `experiment_config_hash_consistent`: true when there is at most one distinct
  paper-level experiment hash.
- `experiment_config_hashes`: comma-separated hashes when there are 10 or
  fewer.
- `n_workflow_run_ids`, `n_runner_invocation_hashes`, and
  `n_job_config_hashes`: audit counts for workflow, runner, and job levels.
  Multiple `runner_invocation_hash` or `job_config_hash` values are expected in
  OpenML matrix aggregates and do not make the aggregate invalid. If multiple
  `experiment_config_hash` values are found, aggregate outputs mark
  `experiment_config_hash`/`config_hash` as `multiple` and write a warning to
  `logs/aggregate_results.log`. A fresh OpenML medium aggregate with
  `n_experiment_config_hashes > 1` is not a valid single paper-level aggregate;
  it indicates mixed inputs and must be rerun or separated before paper
  reporting.
  A matrix OpenML aggregate is valid when `experiment_config_hash` and
  `workflow_run_id` are consistent, even if `runner_invocation_hash`,
  deprecated `run_config_hash`, and `job_config_hash` have multiple values.

SHAP explanations use a training-background sample rather than the explained
instance alone. The runner samples up to `max_background=100` rows from
`X_train` with the configured seed and uses that background when constructing
`shap.Explainer`. The recorded SHAP model-call count is an approximation based
on the background size and one explained sample. SHAP rows report
`background_size` and `max_background`; LIME rows report the training rows used
to initialize the explainer.

`combined_score` is a convenience aggregate, not a primary theoretical metric:

```text
combined_score = (deletion_drop_auc + insertion_auc) / 2
```

`deletion_drop_auc = p0 - deletion_auc`, so higher values indicate a larger
probability drop under deletion. `insertion_auc` is also higher-is-better
because important inserted features should recover the target probability
earlier. Result rows record `metric_direction = "higher_is_better"`.

Optional baselines are always attempted and recorded as `ok`, `skipped`, or
`error`. AIME may be recorded as skipped even when the dependency is installed
because the AIME adapter is not implemented in this artifact; that case is
distinguished from a missing dependency in `error_message`. BayesSHAP,
BayesLIME, and Bayesian-AIME are not aliases for ordinary SHAP, LIME, or AIME:
they are recorded as skipped unless true Bayesian adapters are implemented.
Smoke tests explicitly check that these Bayesian baseline rows have exact
skipped messages, zero model-call counts, and no attribution/faithfulness
metrics.

Paper-level bootstrap confidence intervals are computed over dataset-seed units
by default: rows are first averaged within `dataset_name`, `task_id`, `seed`,
`fold`, and `method`, then bootstrap resampling is applied to those unit-level
values. Instance-level bootstrap is reserved for debugging because it can
overstate precision when many explanations come from the same fitted model.

OpenML-10/`medium` runs are preliminary engineering checks. They are not
equivalent to the full OpenML-CC18 experiment and should not be reported as full
benchmark results.

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
