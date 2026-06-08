from __future__ import annotations

from pathlib import Path
from typing import Optional

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

