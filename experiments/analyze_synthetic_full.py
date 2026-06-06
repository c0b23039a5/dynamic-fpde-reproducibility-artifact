from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from bayesian_fpde.utils import ensure_dirs, read_csv_preserve_metadata


METRICS = [
    "coverage_95",
    "mean_ci_width",
    "median_ci_width",
    "sign_accuracy",
    "top_k_precision",
    "spearman_rank_correlation",
    "kendall_tau",
    "sign_brier_score",
    "sign_ece",
    "sign_accuracy_at_confidence_0_8",
    "sign_accuracy_at_confidence_0_9",
    "effective_n_explain",
]

DERIVED_METRICS = [
    "coverage_gap_from_95",
    "abs_coverage_gap_from_95",
]

OUTPUTS = {
    "method": "synthetic_full_method_summary.csv",
    "n_samples": "synthetic_full_by_n_samples.csv",
    "n_features": "synthetic_full_by_n_features.csv",
    "class_separation": "synthetic_full_by_class_separation.csv",
    "effective_warning": "synthetic_full_by_effective_warning.csv",
    "warning_summary": "synthetic_full_low_effective_warning_summary.csv",
}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _coerce_bool(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.lower()
    return text.isin(["true", "1", "yes", "y"])


def _read_summary(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"synthetic calibration summary not found: {path}")
    df = read_csv_preserve_metadata(path)
    if df.empty:
        raise ValueError(f"synthetic calibration summary is empty: {path}")

    required = {"method", "status", "coverage_95", "low_effective_n_explain_warning"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    out = df.copy()
    for col in METRICS:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ["n_samples", "n_features", "seed", "n_informative", "posterior_samples", "n_explain", "requested_n_explain"]:
        if col in out.columns:
            numeric = pd.to_numeric(out[col], errors="coerce")
            if numeric.notna().any():
                out[col] = numeric
    out["low_effective_n_explain_warning"] = _coerce_bool(out["low_effective_n_explain_warning"])
    out["coverage_gap_from_95"] = out["coverage_95"] - 0.95
    out["abs_coverage_gap_from_95"] = out["coverage_gap_from_95"].abs()
    out["undercoverage_flag"] = out["coverage_95"] < 0.90
    return out


def _status_counts(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    grouped = df.groupby(list(group_cols), dropna=False)
    status = grouped.agg(
        n_rows=("method", "size"),
        n_ok=("status", lambda s: int((s.astype("string") == "ok").sum())),
        n_error=("status", lambda s: int((s.astype("string") == "error").sum())),
        n_skipped=("status", lambda s: int((s.astype("string") == "skipped").sum())),
        low_effective_warning_count=("low_effective_n_explain_warning", "sum"),
        undercoverage_count=("undercoverage_flag", "sum"),
    ).reset_index()
    status["low_effective_warning_rate"] = status["low_effective_warning_count"] / status["n_rows"].replace(0, np.nan)
    status["undercoverage_rate"] = status["undercoverage_count"] / status["n_rows"].replace(0, np.nan)
    status["undercoverage_flag_count"] = status["undercoverage_count"]
    status["undercoverage_flag_rate"] = status["undercoverage_rate"]
    return status


def _metric_summary(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    ok = df[df["status"].astype("string") == "ok"].copy()
    all_metrics = METRICS + DERIVED_METRICS
    grouped = ok.groupby(list(group_cols), dropna=False)
    pieces: list[pd.DataFrame] = []
    for metric in all_metrics:
        piece = grouped[metric].agg(["mean", "std", "median", "min", "max", "count"]).reset_index()
        piece = piece.rename(
            columns={
                "mean": f"{metric}_mean",
                "std": f"{metric}_std",
                "median": f"{metric}_median",
                "min": f"{metric}_min",
                "max": f"{metric}_max",
                "count": f"{metric}_n",
            }
        )
        pieces.append(piece)
    if not pieces:
        return pd.DataFrame(columns=list(group_cols))
    out = pieces[0]
    for piece in pieces[1:]:
        out = out.merge(piece, on=list(group_cols), how="outer")
    out["coverage_gap_from_95"] = out["coverage_gap_from_95_mean"]
    out["abs_coverage_gap_from_95"] = out["abs_coverage_gap_from_95_mean"]
    return out


def _summarize(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    status = _status_counts(df, group_cols)
    metrics = _metric_summary(df, group_cols)
    out = status.merge(metrics, on=list(group_cols), how="left")
    return out.sort_values(list(group_cols)).reset_index(drop=True)


def _warning_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for col in ["n_samples", "class_separation", "class_balance", "method"]:
        if col not in df.columns:
            continue
        summary = _status_counts(df, [col])
        summary = summary.rename(columns={col: "warning_level"})
        summary.insert(0, "warning_group", col)
        rows.append(summary)
    if not rows:
        return pd.DataFrame(
            columns=[
                "warning_group",
                "warning_level",
                "n_rows",
                "n_ok",
                "n_error",
                "n_skipped",
                "low_effective_warning_count",
                "low_effective_warning_rate",
            ]
        )
    return pd.concat(rows, ignore_index=True, sort=False)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dirs(path.parent)
    df.to_csv(path, index=False, lineterminator="\n")


def _line_by_method(df: pd.DataFrame, *, x: str, y: str, path: Path, title: str, ylabel: str, nominal_coverage: bool = False) -> None:
    plt = _plt()
    ensure_dirs(path.parent)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    if df.empty or x not in df.columns or y not in df.columns:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        ok = df[df["status"].astype("string") == "ok"].copy()
        if not ok.empty:
            grouped = ok.groupby([x, "method"], dropna=False)[y].mean().reset_index()
            for method, sub in grouped.groupby("method", dropna=False):
                sub = sub.sort_values(x)
                ax.plot(sub[x], sub[y], marker="o", linewidth=1.8, label=str(method))
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "No ok rows", ha="center", va="center")
    if nominal_coverage:
        ax.axhline(0.95, color="black", linestyle="--", linewidth=1.0, label="nominal 0.95")
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _coverage_by_method(df: pd.DataFrame, path: Path) -> None:
    plt = _plt()
    ensure_dirs(path.parent)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ok = df[df["status"].astype("string") == "ok"].copy()
    if ok.empty:
        ax.text(0.5, 0.5, "No ok rows", ha="center", va="center")
    else:
        methods = sorted(ok["method"].astype(str).unique())
        values = [ok.loc[ok["method"].astype(str) == method, "coverage_95"].dropna().to_numpy() for method in methods]
        try:
            ax.boxplot(values, tick_labels=methods, showmeans=True)
        except TypeError:  # pragma: no cover - compatibility with older matplotlib.
            ax.boxplot(values, labels=methods, showmeans=True)
        ax.axhline(0.95, color="black", linestyle="--", linewidth=1.0)
        ax.tick_params(axis="x", rotation=25)
    ax.set_title("Synthetic full coverage by method")
    ax.set_ylabel("coverage_95")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _warning_heatmap(df: pd.DataFrame, path: Path) -> None:
    plt = _plt()
    ensure_dirs(path.parent)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    if df.empty or "n_samples" not in df.columns or "class_separation" not in df.columns:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        heat = (
            df.groupby(["n_samples", "class_separation"], dropna=False)["low_effective_n_explain_warning"]
            .mean()
            .reset_index()
            .pivot(index="n_samples", columns="class_separation", values="low_effective_n_explain_warning")
        )
        heat = heat.sort_index()
        image = ax.imshow(heat.to_numpy(dtype=float), aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_xticks(np.arange(len(heat.columns)))
        ax.set_xticklabels([str(v) for v in heat.columns])
        ax.set_yticks(np.arange(len(heat.index)))
        ax.set_yticklabels([str(v) for v in heat.index])
        ax.set_xlabel("class_separation")
        ax.set_ylabel("n_samples")
        for i in range(len(heat.index)):
            for j in range(len(heat.columns)):
                value = heat.iloc[i, j]
                if pd.notna(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value > 0.45 else "black", fontsize=8)
        fig.colorbar(image, ax=ax, label="warning rate")
    ax.set_title("Low effective explanation warning rate")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_analysis_tables(df: pd.DataFrame, results_dir: Path) -> dict[str, pd.DataFrame]:
    ensure_dirs(results_dir)
    outputs = {
        "method": _summarize(df, ["method"]),
        "n_samples": _summarize(df, ["n_samples", "method"]) if "n_samples" in df.columns else _summarize(df, ["method"]),
        "n_features": _summarize(df, ["n_features", "method"]) if "n_features" in df.columns else _summarize(df, ["method"]),
        "class_separation": _summarize(df, ["class_separation", "method"]) if "class_separation" in df.columns else _summarize(df, ["method"]),
        "effective_warning": _summarize(df, ["low_effective_n_explain_warning", "method"]),
        "warning_summary": _warning_summary(df),
    }
    for key, frame in outputs.items():
        _write_csv(frame, results_dir / OUTPUTS[key])
    return outputs


def write_figures(df: pd.DataFrame, figures_dir: Path) -> None:
    ensure_dirs(figures_dir)
    _coverage_by_method(df, figures_dir / "synthetic_full_coverage_by_method.png")
    _line_by_method(
        df,
        x="n_samples",
        y="coverage_95",
        path=figures_dir / "synthetic_full_coverage_by_n_samples.png",
        title="Synthetic full coverage by n_samples",
        ylabel="coverage_95",
        nominal_coverage=True,
    )
    _line_by_method(
        df,
        x="n_samples",
        y="mean_ci_width",
        path=figures_dir / "synthetic_full_ci_width_by_n_samples.png",
        title="Synthetic full CI width by n_samples",
        ylabel="mean_ci_width",
    )
    _line_by_method(
        df,
        x="n_samples",
        y="sign_ece",
        path=figures_dir / "synthetic_full_sign_ece_by_n_samples.png",
        title="Synthetic full sign ECE by n_samples",
        ylabel="sign_ece",
    )
    _line_by_method(
        df,
        x="n_samples",
        y="top_k_precision",
        path=figures_dir / "synthetic_full_topk_precision_by_n_samples.png",
        title="Synthetic full top-k precision by n_samples",
        ylabel="top_k_precision",
    )
    _line_by_method(
        df,
        x="n_samples",
        y="effective_n_explain",
        path=figures_dir / "synthetic_full_effective_n_explain_by_n_samples.png",
        title="Synthetic full effective explanations by n_samples",
        ylabel="effective_n_explain",
    )
    _warning_heatmap(df, figures_dir / "synthetic_full_warning_rate_heatmap.png")


def analyze(input_path: str | Path, results_dir: str | Path, figures_dir: str | Path) -> dict[str, pd.DataFrame]:
    df = _read_summary(input_path)
    outputs = write_analysis_tables(df, Path(results_dir))
    write_figures(df, Path(figures_dir))
    return outputs


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create paper-ready analysis tables for the full synthetic calibration run.")
    parser.add_argument("--input", default="results/synthetic_calibration_summary.csv", help="Completed synthetic calibration summary CSV.")
    parser.add_argument("--results-dir", default="results", help="Directory for generated analysis CSV files.")
    parser.add_argument("--figures-dir", default="figures", help="Directory for generated analysis figures.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    analyze(args.input, args.results_dir, args.figures_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
