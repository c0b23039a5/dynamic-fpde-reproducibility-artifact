from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .utils import ensure_dirs


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_metric_boxplot(df: pd.DataFrame, *, metric: str, path: str | Path, title: str) -> None:
    plt = _plt()
    ensure_dirs(Path(path).parent)
    fig, ax = plt.subplots(figsize=(8, 4))
    if df.empty or metric not in df.columns:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        df.boxplot(column=metric, by="method", ax=ax, rot=30)
        fig.suptitle("")
    ax.set_title(title)
    ax.set_ylabel(metric)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_line_plot(df: pd.DataFrame, *, x: str, y: str, path: str | Path, title: str, group: Optional[str] = None) -> None:
    plt = _plt()
    ensure_dirs(Path(path).parent)
    fig, ax = plt.subplots(figsize=(7, 4))
    if df.empty or x not in df.columns or y not in df.columns:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    elif group and group in df.columns:
        for label, sub in df.groupby(group):
            sub = sub.sort_values(x)
            ax.plot(sub[x], sub[y], marker="o", label=str(label))
        ax.legend(fontsize=8)
    else:
        sub = df.sort_values(x)
        ax.plot(sub[x], sub[y], marker="o")
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_ci_bar(summary: pd.DataFrame, *, path: str | Path, title: str, top_n: int = 12) -> None:
    plt = _plt()
    ensure_dirs(Path(path).parent)
    df = summary.copy()
    if "posterior_mean" in df.columns:
        df = df.reindex(df["posterior_mean"].abs().sort_values(ascending=False).index).head(top_n)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(df))))
    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        y_pos = np.arange(len(df))
        mean = df["posterior_mean"].to_numpy(dtype=float)
        lower = df["ci_lower_95"].to_numpy(dtype=float)
        upper = df["ci_upper_95"].to_numpy(dtype=float)
        ax.barh(y_pos, mean, xerr=[mean - lower, upper - mean], color="#4c78a8", alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df["feature"].astype(str).tolist())
        ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("attribution")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_rank_probability(summary: pd.DataFrame, *, path: str | Path, title: str, top_n: int = 12) -> None:
    plt = _plt()
    ensure_dirs(Path(path).parent)
    df = summary.sort_values("rank_probability_top_k", ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(df))))
    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        ax.barh(np.arange(len(df)), df["rank_probability_top_k"], color="#59a14f")
        ax.set_yticks(np.arange(len(df)))
        ax.set_yticklabels(df["feature"].astype(str).tolist())
        ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.set_xlabel("P(top-k)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
