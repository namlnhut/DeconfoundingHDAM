"""
Shared utilities for all simulation scripts.

Provides the true additive functions, palette, parallel runner,
and a reusable violin-plot helper that mirrors the ggplot2 style
used in the original R scripts.
"""

from __future__ import annotations

import pickle
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# True component functions
# ---------------------------------------------------------------------------

def f1(x): return -np.sin(2 * x)
def f2(x): return 2 - 2 * np.tanh(x + 0.5)
def f3(x): return x
def f4(x): return 4 / (np.exp(x) + np.exp(-x))

def f_true(X: np.ndarray) -> np.ndarray:
    """True additive function f(X) = f1(X1) + f2(X2) + f3(X3) + f4(X4)."""
    return f1(X[:, 0]) + f2(X[:, 1]) + f3(X[:, 2]) + f4(X[:, 3])


# ---------------------------------------------------------------------------
# Colour palette (matching R's colour scheme)
# ---------------------------------------------------------------------------

PALETTE = {
    "deconfounded":      "#4472C4",   # blue
    "estimated factors": "#70AD47",   # green
    "naive":             "#FF0000",   # red
}
METHS = ["deconfounded", "estimated factors", "naive"]


# ---------------------------------------------------------------------------
# Parallel runner
# ---------------------------------------------------------------------------

def run_parallel(fn, args_list: list, n_cores: int = 4) -> list:
    """
    Run fn(*args) for every args in args_list using n_cores processes.
    Falls back to sequential execution when n_cores == 1.
    """
    if n_cores == 1:
        return [fn(*a) for a in args_list]
    with mp.Pool(n_cores) as pool:
        return pool.starmap(fn, args_list)


# ---------------------------------------------------------------------------
# Pickle I/O
# ---------------------------------------------------------------------------

def save_results(results: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(results, fh)


def load_results(path: Path) -> object:
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ---------------------------------------------------------------------------
# Violin plot helper
# ---------------------------------------------------------------------------

def violin_plot(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_order,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    hline: float | None = None,
) -> None:
    """
    Draw a grouped violin plot (one group per method) on *ax*.

    Parameters
    ----------
    ax        : matplotlib Axes
    df        : DataFrame with columns [x_col, y_col, 'meth']
    x_col     : column to use for the x-axis (categorical)
    x_order   : ordered list of x values
    hline     : draw a horizontal reference line at this y value
    """
    x_vals = list(x_order)
    n_x = len(x_vals)
    n_m = len(METHS)
    width = 0.22
    offsets = np.linspace(-(n_m - 1) / 2, (n_m - 1) / 2, n_m) * width * 1.5

    for mi, meth in enumerate(METHS):
        sub = df[df["meth"] == meth]
        data_per_x = [
            sub.loc[sub[x_col] == xv, y_col].dropna().values
            for xv in x_vals
        ]
        positions = np.arange(n_x) + offsets[mi]
        parts = ax.violinplot(
            data_per_x,
            positions=positions,
            widths=width,
            showmedians=False,
            showextrema=False,
        )
        colour = PALETTE[meth]
        for pc in parts["bodies"]:
            pc.set_facecolor(colour)
            pc.set_alpha(0.6)
            pc.set_edgecolor("black")
            pc.set_linewidth(0.5)
        # mean dot
        means = [d.mean() if len(d) > 0 else np.nan for d in data_per_x]
        ax.scatter(positions, means, color=colour, s=15, zorder=3)

    ax.set_xticks(range(n_x))
    ax.set_xticklabels([str(v) for v in x_vals], fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)

    if hline is not None:
        ax.axhline(hline, color="black", linewidth=0.8, linestyle="--")

    handles = [
        mpatches.Patch(facecolor=PALETTE[m], alpha=0.6, label=m)
        for m in METHS
    ]
    ax.legend(handles=handles, fontsize=8, frameon=False)
