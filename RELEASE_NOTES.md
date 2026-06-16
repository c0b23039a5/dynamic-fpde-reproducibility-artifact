# Release Notes

## Unreleased

- Make Raw-Waveform Dynamic-FPDE the primary ESC-50 formulation.
- Add `run_esc50_raw_waveform_fpde.py`, which uses raw waveform + label only,
  keeps variable-length clips, evaluates Raw-Hyb for `lambda_hyb=0.0..1.0`,
  saves lambda-wise raw artifacts, and records skipped generation when no
  external RAW generator hook is supplied.
- Make the Raw-Waveform runner default to `--device cuda` for CUDA 13
  processing through `cupy-cuda13x`; CPU execution remains available with
  explicit `--device cpu`.
- Preserve the Native-Time frame-level feature runner as a legacy/comparison
  path rather than the primary confirmed workflow.
- Replace fixed-length `prototype_length` time-series prototypes with real
  exemplar frame prototype vectors.
- Reject legacy `--prototype-length` in the primary runner.
- Report `Phi.shape == X.shape`, prototype metadata, Native-Time additivity,
  and prototype-evidence deletion/insertion diagnostics.
- Clarify that the Raw-Waveform path does not use acoustic feature extraction,
  spectrograms, MFCCs, or waveform normalization, and that label-conditioned
  RAW generation happens only after important segment extraction.
- Batch CUDA attribution over all resolved test samples that naturally share
  the same `(T, F)` while keeping diagnostics on CPU.
- Keep CUDA Dynamic-Hyb diff, cosine, and hybrid attribution arrays on GPU
  until final materialization.
- Split runtime reporting into shared common-rival selection, Native-Time FPDE,
  diagnostic, and total runtime columns.
- Rename norm ranking helpers around `frame_norm_scores` while retaining
  `energy_frame_scores` as a backward-compatible alias.
- Clarify that sub-frame zero padding is an intra-clip analysis-frame guard,
  not fixed-length temporal alignment or CUDA batch padding.

## v0.1.0

Initial public reproducibility artifact for FPDE.

Included:

- FPDE implementation.
- OpenML-CC18 experimental runner.
- Per-task/per-seed rerun script and GitHub Actions workflow.
- 72-task, 10-seed precomputed results.
- Generated LaTeX tables, audit CSV files, and a script to regenerate them from the included outputs.
- Apache License 2.0, citation metadata, GitHub Actions check, and repository metadata.
- Zenodo DOI `10.5281/zenodo.20225275`.

Known limitations:

- Public GitHub repository URL: `https://github.com/fpde-xai/fpde-reproducibility-artifact`.
- Full 10-seed reruns may take substantial CPU time and require network access to OpenML.
