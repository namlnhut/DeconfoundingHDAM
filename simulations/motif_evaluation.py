"""
Reproduce Figures 9, 10, 11 and 12.

Analysis of the motif regression data set.

The original motif data (Beer & Tavazoie, 2004; pre-processed by Guo et al.,
2019) is stored in MotifData/MotifData.RData.  Loading requires `pyreadr`:

    pip install pyreadr

Equivalent to R's SimulationScripts/MotifEvaluation.R.

Usage
-----
python motif_evaluation.py         # fit models + generate all plots
python motif_evaluation.py --plot-only   # load saved fit and plot
"""

from __future__ import annotations
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hdam import (
    fit_deconfounded_hdam,
    fit_hdam_with_est_factors,
    estimate_fj,
)

PLOT_DIR  = Path("PlotResults/Motif")
FIT_PATH  = Path("SimulationResults/Motif/MotifFits.pkl")
DATA_PATH = Path("MotifData/MotifData.RData")

COLOURS = {
    "deconfounded":      "red",
    "estimated factors": "darkgreen",
    "naive":             "blue",
}


# ---------------------------------------------------------------------------
# Load motif data
# ---------------------------------------------------------------------------

def load_motif_data():
    """Load genes and motifs matrices from the .RData file."""
    try:
        import pyreadr
    except ImportError:
        raise ImportError(
            "pyreadr is required to read MotifData.RData.\n"
            "Install it with:  pip install pyreadr"
        )
    rdata = pyreadr.read_r(str(DATA_PATH))
    genes  = rdata["genes"].values   # numpy array
    motifs = rdata["motifs"].values
    return genes, motifs


# ---------------------------------------------------------------------------
# Fit models
# ---------------------------------------------------------------------------

def fit_models(y: np.ndarray, X: np.ndarray, seed: int = 1443) -> dict:
    np.random.seed(seed)
    kw = dict(n_K=5, cv_method="1se", cv_k=5, n_lambda1=15, n_lambda2=50)

    print("Fitting naive (meth='none') ...")
    fit_null = fit_deconfounded_hdam(y, X.copy(), meth="none", **kw)
    print(f"  active: {len(fit_null['active'])}")

    print("Fitting deconfounded (meth='trim') ...")
    fit_trim = fit_deconfounded_hdam(y, X.copy(), meth="trim", **kw)
    print(f"  active: {len(fit_trim['active'])}")

    print("Fitting estimated factors ...")
    fit_ef = fit_hdam_with_est_factors(y, X.copy(), **kw)
    print(f"  active: {len(fit_ef['active'])}")

    return {"null": fit_null, "trim": fit_trim, "estFac": fit_ef}


# ---------------------------------------------------------------------------
# Figure 9 and 10: component functions
# ---------------------------------------------------------------------------

def _plot_components(X, fits, indices, title_suffix: str, out_path: Path) -> None:
    """
    3×3 grid of component function plots for the given predictor indices.
    Equivalent to the two for-loops in R that plot fj for 9 predictors.
    """
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    fig.suptitle(title_suffix, fontsize=11)

    for pos, j in enumerate(indices[:9]):
        ax = axes[pos // 3][pos % 3]
        xx = np.linspace(X[:, j].min(), X[:, j].max(), 50)

        ax.plot(xx, estimate_fj(xx, j, fits["trim"]),
                color=COLOURS["deconfounded"],      lw=1.5, label="deconfounded")
        ax.plot(xx, estimate_fj(xx, j, fits["null"]),
                color=COLOURS["naive"],              lw=1.5, label="naive")
        ax.plot(xx, estimate_fj(xx, j, fits["estFac"]),
                color=COLOURS["estimated factors"],  lw=1.5, label="est. factors")

        ax.axhline(0, color="grey", lw=0.6)
        # Rug plot
        ax.scatter(X[:, j], np.full(X.shape[0], ax.get_ylim()[0]),
                   color="black", alpha=0.1, s=4, marker="|")
        ax.set_title(f"j = {j}", fontsize=9)
        ax.set_xlabel(r"$X_j$", fontsize=8)
        ax.set_ylabel(r"$\hat{f}_j(X_j)$", fontsize=8)

        if pos == 0:
            ax.legend(fontsize=7, frameon=False)

    for pos in range(len(indices[:9]), 9):
        axes[pos // 3][pos % 3].axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"saved {out_path.name}")


def make_component_plots(X, fits: dict, plot_dir: Path) -> None:
    coef_sq = lambda f: np.array([np.sum(c ** 2) for c in f["coefs"]])

    cl_trim = coef_sq(fits["trim"])
    cl_null = coef_sq(fits["null"])

    # Figure 9 & 10: top-9 predictors by ||coef||^2 for the deconfounded fit
    ord_trim = np.argsort(cl_trim)[::-1]
    top9_trim = ord_trim[:9]
    _plot_components(X, fits, top9_trim,
                     "9 strongest components (deconfounded ordering)",
                     plot_dir / "MotifComponents_top9_trim.pdf")

    # Figures 11 & 12: top-9 predictors that are active in naive but NOT in trim
    active_diff = sorted(set(fits["null"]["active"]) - set(fits["trim"]["active"]))
    # order within active_diff by naive coefficient length
    cl_null_diff = [(j, cl_null[j]) for j in active_diff]
    cl_null_diff.sort(key=lambda x: -x[1])
    top9_diff = [j for j, _ in cl_null_diff[:9]]
    _plot_components(X, fits, top9_diff,
                     "9 strongest naive-only components (set to 0 by deconfounding)",
                     plot_dir / "MotifComponents_naiveOnly.pdf")


# ---------------------------------------------------------------------------
# Figure 11 (MotifCoefLength.pdf): coef norm scatter + Jaccard similarity
# ---------------------------------------------------------------------------

def make_summary_plots(X, fits: dict, plot_dir: Path) -> None:
    coef_sq = lambda f: np.array([np.sum(c ** 2) for c in f["coefs"]])
    cl_trim  = coef_sq(fits["trim"])
    cl_null  = coef_sq(fits["null"])
    cl_ef    = coef_sq(fits["estFac"])

    n_null = len(fits["null"]["active"])
    n_trim = len(fits["trim"]["active"])
    n_ef   = len(fits["estFac"]["active"])

    # Jaccard similarity as a function of top-l cutoff
    ord_null = np.argsort(cl_null)[::-1][:n_null]
    ord_trim = np.argsort(cl_trim)[::-1][:n_trim]
    ord_ef   = np.argsort(cl_ef)[::-1][:n_ef]

    max_l_null = n_null
    jac_null = np.zeros(max_l_null)
    for l in range(1, max_l_null + 1):
        top_null = set(ord_null[:l])
        top_trim = set(ord_trim[:l])
        union_  = top_null | top_trim
        inter_  = top_null & top_trim
        jac_null[l - 1] = len(inter_) / len(union_) if union_ else 0.0

    max_l_ef = n_ef
    jac_ef = np.zeros(max_l_ef)
    for l in range(1, max_l_ef + 1):
        top_ef   = set(ord_ef[:l])
        top_trim = set(ord_trim[:l])
        union_   = top_ef | top_trim
        inter_   = top_ef & top_trim
        jac_ef[l - 1] = len(inter_) / len(union_) if union_ else 0.0

    # Singular values of centred X (Figure 9 / top panel)
    Xc = X - X.mean(axis=0)
    sv = np.linalg.svd(Xc, compute_uv=False)

    # --- 3 panels in one figure ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel A: singular values
    axes[0].scatter(range(1, len(sv) + 1), sv, s=5, color="black")
    axes[0].set_xlabel("l", fontsize=10)
    axes[0].set_ylabel("singular value $d_l$", fontsize=10)
    axes[0].set_title("Singular values of centred motif data", fontsize=10)

    # Panel B: coefficient norm scatter
    axes[1].scatter(np.sqrt(cl_trim), np.sqrt(cl_null),
                    color=COLOURS["naive"], marker="o", s=12, alpha=0.7,
                    label="deconfounded vs. naive")
    axes[1].scatter(np.sqrt(cl_trim), np.sqrt(cl_ef),
                    color=COLOURS["estimated factors"], marker="x", s=12, alpha=0.7,
                    label="deconfounded vs. est. factors")
    lim = max(np.sqrt(cl_null).max(), np.sqrt(cl_ef).max(), np.sqrt(cl_trim).max())
    axes[1].plot([0, lim], [0, lim], color="black", lw=0.8)
    axes[1].set_xlabel("deconfounded", fontsize=10)
    axes[1].set_ylabel("naive / estimated factors", fontsize=10)
    axes[1].set_title("Norm of coefficient vectors", fontsize=10)
    axes[1].legend(fontsize=8, frameon=False)

    # Panel C: Jaccard similarity
    axes[2].plot(range(1, max_l_null + 1), jac_null,
                 color=COLOURS["naive"],             lw=1.5,
                 label="deconfounded vs. naive")
    axes[2].plot(range(1, max_l_ef + 1), jac_ef,
                 color=COLOURS["estimated factors"],  lw=1.5,
                 label="deconfounded vs. est. factors")
    axes[2].set_xlabel("l", fontsize=10)
    axes[2].set_ylabel("Jaccard similarity", fontsize=10)
    axes[2].set_title("Jaccard similarity of top-l index sets", fontsize=10)
    axes[2].legend(fontsize=8, frameon=False)

    fig.tight_layout()
    out = plot_dir / "MotifCoefLength.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading motif data from {DATA_PATH} ...")
    genes, motifs = load_motif_data()

    # Response: gene column 131 (0-indexed: 130)
    T_IDX = 130
    y = genes[:, T_IDX]
    X = motifs

    if args.plot_only:
        if not FIT_PATH.exists():
            raise FileNotFoundError(
                f"{FIT_PATH} not found; run without --plot-only first."
            )
        with open(FIT_PATH, "rb") as fh:
            fits = pickle.load(fh)
    else:
        fits = fit_models(y, X)
        FIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(FIT_PATH, "wb") as fh:
            pickle.dump(fits, fh)
        print(f"Fits saved to {FIT_PATH}")

    # Print active set sizes
    print(f"\nActive set sizes:")
    print(f"  naive (none):          {len(fits['null']['active'])}")
    print(f"  deconfounded (trim):   {len(fits['trim']['active'])}")
    print(f"  estimated factors:     {len(fits['estFac']['active'])}")
    print(f"  intersect(none, trim): "
          f"{len(set(fits['null']['active']) & set(fits['trim']['active']))}")
    print(f"  intersect(ef,   trim): "
          f"{len(set(fits['estFac']['active']) & set(fits['trim']['active']))}")

    make_component_plots(X, fits, PLOT_DIR)
    make_summary_plots(X, fits, PLOT_DIR)
