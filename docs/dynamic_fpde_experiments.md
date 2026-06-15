# Dynamic-FPDE Audio Experiments

This artifact installs Dynamic-FPDE from the `dynamic` branch of the FPDE
repository:

```text
fpde @ git+https://github.com/fpde-xai/fpde.git@dynamic
```

The experiment suite evaluates Dynamic-FPDE as a time-resolved
prototype-directional explanation method for frame-level audio feature
sequences. Dynamic-FPDE explains prototype evidence. It does not explain raw
waveform samples, and the outputs are not causal explanations.

## Dataset

The first supported dataset is ESC-50. The runner expects a local dataset with
this structure:

```text
data/ESC-50/
  audio/
  meta/esc50.csv
```

The runner reads `meta/esc50.csv`, uses the provided `fold` column, and uses
`category` as the class label. It does not download ESC-50 automatically.

## Install

```bash
python -m pip install -e ".[dev,dynamic-audio]"
```

For the broad existing OpenML artifact dependencies, continue to use the
existing `openml` and `baselines` extras as needed.

## Run Smoke Mode

```bash
python experiments/dynamic_fpde_audio/run_esc50_dynamic_fpde.py \
  --dataset-root data/ESC-50 \
  --output-dir outputs/dynamic_fpde_esc50_smoke \
  --mode smoke \
  --fold 1 \
  --seed 0 \
  --prototype-length 64
```

Smoke mode uses a small deterministic subset intended for quick local sanity
checks.

## Run Pilot Mode

```bash
python experiments/dynamic_fpde_audio/run_esc50_dynamic_fpde.py \
  --dataset-root data/ESC-50 \
  --output-dir outputs/dynamic_fpde_esc50_pilot \
  --mode pilot \
  --fold 1 \
  --seed 0 \
  --prototype-length 128
```

## Outputs

The runner writes:

- `run_config.json`
- `feature_config.json`
- `environment_info.json`
- `results/dynamic_fpde_sample_metrics.csv`
- `results/dynamic_fpde_summary_by_method.csv`
- `results/dynamic_fpde_lambda_selection.csv`
- `results/dynamic_fpde_additivity_summary.csv`
- `tables/table_dynamic_fpde_main_results.tex`
- `tables/table_dynamic_fpde_additivity.tex`
- `tables/table_dynamic_fpde_lambda.tex`

When `--make-figures` is passed, optional figure files are written under
`figures/`.

## Metrics

The method operates on frame-level acoustic features, not raw waveform samples.
The feature extractor returns a matrix `X` with shape `(T, F)` containing RMS,
zero crossing rate, spectral features, and MFCCs. Features are standardized
with training-set mean and standard deviation before Dynamic-FPDE prototypes are
built.

For each explained sample, the runner records prototype evidence, the
auditable attribution sum residual, deletion AUC, insertion AUC, and a combined
score. The deletion/insertion metrics are prototype-driven and normalized. They
are computed from prototype-evidence curves rather than class probabilities.

`dynamic_hyb` selects `lambda_hyb` on a deterministic validation split inside
the ESC-50 training folds. The suite also reports `dynamic_diff`,
`dynamic_cos`, an RMS energy ranking baseline, and a seeded random ranking
baseline.

## LaTeX Tables

```bash
python scripts/make_dynamic_fpde_tables.py \
  --results-dir outputs/dynamic_fpde_esc50_smoke/results \
  --tables-dir outputs/dynamic_fpde_esc50_smoke/tables
```

The table script reads existing CSV summaries. It does not hardcode or invent
research values.

## Interpretation Limits

Dynamic-FPDE explains prototype evidence for a target prototype over a rival
prototype. The outputs are not causal explanations, ground-truth explanations,
human preference explanations, or black-box model faithfulness measurements.
This experiment suite intentionally does not implement Delta-Dynamic-FPDE,
original-cover difference explanations, raw waveform direct attribution, DTW
alignment, AIME, SHAP, LIME, recommender-system logic, or causal claims.
