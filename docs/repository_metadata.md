# Repository Metadata

This repository is a public reproducibility artifact for the FPDE manuscript.

## Public repository URL

```text
https://github.com/fpde-xai/fpde-reproducibility-artifact
```

## Files intentionally included

- `code/`: FPDE implementation, OpenML-CC18 experimental runner, and table builder.
- `scripts/`: compatibility table reproduction entry point, result aggregation, and per-task/per-seed rerun script.
- `results_precomputed/`: 72-task, 10-seed precomputed outputs.
- `generated/`: generated LaTeX tables and audit CSV files.
- `docs/`: reproducibility checklist, availability statement, and repository metadata.
- `LICENSE`: Apache License 2.0 for the repository.
- `CITATION.cff`: citation metadata for GitHub's citation interface.
- `.github/workflows/artifact-check.yml`: lightweight continuous-integration check.

## Repository description

```text
Reproducibility artifact for Feature Prototype Direction Explainer (FPDE): code, OpenML-CC18 runner, precomputed 72-task/10-seed outputs, and table reproduction scripts.
```

## Repository topics

```text
explainable-ai, xai, feature-attribution, prototype-based, openml, lightgbm, reproducibility, machine-learning
```

## Consistency checks

Run:

```bash
python scripts/build_reproducibility_tables.py --input results_precomputed --output generated --expected-tasks 72 --expected-seeds 10
```

The checks confirm:

- No private files or local paths are included.
- The selected license is included in `LICENSE`.
- The precomputed result files are intentionally included; each individual file is below GitHub's 100 MiB hard limit.
