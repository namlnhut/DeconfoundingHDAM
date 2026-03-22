"""
Reproduce Figures 19 and 20  (nonlinear confounding effects).

alpha controls nonlinearity in H -> X:   X = nlX(alpha, H @ Gamma) + E
beta  controls nonlinearity in H -> Y:   Y = f(X) + nlY(beta,  H @ psi) + e

where nlX(a, t) = (1-a)*t + a*|t|  interpolates linearly between t and |t|.

Results are shown as 7×7 heatmaps of average MSE ratios over the (alpha, beta) grid.

Equivalent to R's SimulationScripts/Nonlinear.R.

Usage
-----
python nonlinear.py              # run simulations + plot
python nonlinear.py --plot-only  # load saved results and plot
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hdam import fit_deconfounded_hdam, fit_hdam_with_est_factors, estimate_function
from _sim_utils import (
    f_true,
    run_parallel, save_results, load_results,
)

# ---------------------------------------------------------------------------
# Fixed parameters
# ---------------------------------------------------------------------------

Q = 5
N = 400
P = 500
N_REP = 100

AL_VEC  = list(np.linspace(0, 1, 7))   # alpha values
BET_VEC = list(np.linspace(0, 1, 7))   # beta  values

CI_VEC = [
    ("equal",      False),
    ("decreasing", True),
]


# ---------------------------------------------------------------------------
# Nonlinearity helpers  (equivalent to R's nlX / nlY)
# ---------------------------------------------------------------------------

def nlX(al: float, t: np.ndarray) -> np.ndarray:
    """Linear interpolation between t and |t|."""
    return (1 - al) * t + al * np.abs(t)


def nlY(bet: float, t: np.ndarray) -> np.ndarray:
    """Linear interpolation between t and |t|."""
    return (1 - bet) * t + bet * np.abs(t)


# ---------------------------------------------------------------------------
# Single simulation run
# ---------------------------------------------------------------------------

def one_sim(al, bet, q, n, p, seed_val, decreasing_confounding=False):
    rng = np.random.default_rng(seed_val)

    Gamma = rng.uniform(-1, 1, size=(q, p))
    if decreasing_confounding:
        Gamma = (1.0 / np.arange(1, q + 1))[:, None] * Gamma

    psi = rng.uniform(0, 2, size=q)
    H   = rng.standard_normal((n, q))
    E   = rng.standard_normal((n, p))
    e   = 0.5 * rng.standard_normal(n)

    X = nlX(al, H @ Gamma) + E
    Y = f_true(X) + nlY(bet, H @ psi) + e

    kw = dict(n_K=5, cv_method="1se", cv_k=5, n_lambda1=10, n_lambda2=25)
    lres_trim    = fit_deconfounded_hdam(Y, X.copy(), meth="trim", **kw)
    lres_none    = fit_deconfounded_hdam(Y, X.copy(), meth="none", **kw)
    lres_est_fac = fit_hdam_with_est_factors(Y, X.copy(), **kw)

    n_test = 1000
    H_test = rng.standard_normal((n_test, q))
    E_test = rng.standard_normal((n_test, p))
    X_test = nlX(al, H_test @ Gamma) + E_test
    f_test = f_true(X_test)

    return {
        "MSE": [
            float(np.mean((f_test - estimate_function(X_test, lres_trim)) ** 2)),
            float(np.mean((f_test - estimate_function(X_test, lres_none)) ** 2)),
            float(np.mean((f_test - estimate_function(X_test, lres_est_fac)) ** 2)),
        ],
        "active": [
            len(lres_trim["active"]),
            len(lres_none["active"]),
            len(lres_est_fac["active"]),
        ],
    }


# ---------------------------------------------------------------------------
# Run simulations
# ---------------------------------------------------------------------------

def run_simulations(out_dir: Path, n_rep: int, n_cores: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    rng_master = np.random.default_rng(1432)
    seed_vec = rng_master.integers(1, 2_100_000_000, size=n_rep).tolist()

    for i, al in enumerate(AL_VEC, start=1):
        for j, bet in enumerate(BET_VEC, start=1):
            for ci_tag, dec in CI_VEC:
                fname = out_dir / f"{ci_tag}CI_al_{i}_bet_{j}.pkl"
                if fname.exists():
                    print(f"[skip] {fname.name}")
                    continue
                print(f"[run ] al={al:.3f}, bet={bet:.3f}, {ci_tag} CI ...")
                args = [(al, bet, Q, N, P, sv, dec) for sv in seed_vec]
                results = run_parallel(one_sim, args, n_cores)
                save_results(results, fname)
                print(f"       saved {fname.name}")


# ---------------------------------------------------------------------------
# Compute average MSE matrices
# ---------------------------------------------------------------------------

def _avg_mse_matrix(out_dir: Path, n_rep: int, ci_tag: str, meth_idx: int) -> np.ndarray:
    """Return 7x7 matrix of average MSE[meth_idx] for the given CI setting."""
    mat = np.full((len(AL_VEC), len(BET_VEC)), np.nan)
    for i, _ in enumerate(AL_VEC):
        for j, _ in enumerate(BET_VEC):
            fname = out_dir / f"{ci_tag}CI_al_{i+1}_bet_{j+1}.pkl"
            if not fname.exists():
                continue
            results = load_results(fname)
            mat[i, j] = np.mean([r["MSE"][meth_idx] for r in results[:n_rep]])
    return mat


# ---------------------------------------------------------------------------
# Plots (Figures 19 and 20)
# ---------------------------------------------------------------------------

def _heatmap(ax, ratio_mat, al_vec, bet_vec, title: str) -> None:
    """
    Draw a heatmap of ratio_mat with text annotations.
    Uses a reversed magma-like colormap (dark = high ratio).
    """
    im = ax.imshow(
        ratio_mat,
        origin="lower",
        aspect="auto",
        cmap="magma_r",
        vmin=ratio_mat[np.isfinite(ratio_mat)].min(),
        vmax=ratio_mat[np.isfinite(ratio_mat)].max(),
    )
    ax.set_xticks(range(len(bet_vec)))
    ax.set_yticks(range(len(al_vec)))
    ax.set_xticklabels([f"{v:.2f}" for v in bet_vec], fontsize=8)
    ax.set_yticklabels([f"{v:.2f}" for v in al_vec],  fontsize=8)
    ax.set_xlabel(r"$\beta$", fontsize=11)
    ax.set_ylabel(r"$\alpha$", fontsize=11)
    ax.set_title(title, fontsize=10)

    for i in range(ratio_mat.shape[0]):
        for j in range(ratio_mat.shape[1]):
            val = ratio_mat[i, j]
            if np.isfinite(val):
                # white text on dark cells, black on light cells
                color = "white" if val >= 0.995 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color)


def make_plots(out_dir: Path, plot_dir: Path, n_rep: int) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)

    # meth indices: 0=trim(deconfounded), 1=none(naive), 2=estFac
    meth_names = ["deconfounded", "naive", "estimated factors"]

    ci_labels = {
        "equal":      "equal confounding influence",
        "decreasing": "decreasing confounding influence",
    }

    for ci_tag, ci_lbl in ci_labels.items():
        mat_dec  = _avg_mse_matrix(out_dir, n_rep, ci_tag, 0)  # deconfounded
        mat_naive= _avg_mse_matrix(out_dir, n_rep, ci_tag, 1)  # naive
        mat_ef   = _avg_mse_matrix(out_dir, n_rep, ci_tag, 2)  # est. factors

        if np.all(np.isnan(mat_dec)):
            print(f"No results for {ci_tag}; skipping plots.")
            continue

        ratio_dec_vs_naive = mat_dec / mat_naive
        ratio_dec_vs_ef    = mat_dec / mat_ef

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            f"Ratio of average MSE \u2013 {ci_lbl}", fontsize=12,
        )

        _heatmap(axes[0], ratio_dec_vs_naive, AL_VEC, BET_VEC,
                 "deconfounded vs. naive")
        _heatmap(axes[1], ratio_dec_vs_ef,    AL_VEC, BET_VEC,
                 "deconfounded vs. estimated factors")

        fig.tight_layout()
        out = plot_dir / f"Nonlinear_{ci_tag}CI.pdf"
        fig.savefig(out)
        plt.close(fig)
        print(f"saved {out.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--n-rep",   type=int, default=N_REP)
    parser.add_argument("--n-cores", type=int, default=4)
    args = parser.parse_args()

    out_dir  = Path("SimulationResults/Nonlinear")
    plot_dir = Path("PlotResults/Nonlinear")

    if not args.plot_only:
        run_simulations(out_dir, args.n_rep, args.n_cores)

    make_plots(out_dir, plot_dir, args.n_rep)
