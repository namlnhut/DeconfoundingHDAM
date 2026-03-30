"""
Reproduce Figures 17 and 18  (vary the proportion of confounded covariates).

`prop` controls the sparsity of Gamma: each entry of Gamma is zeroed
independently with probability (1 - prop).  prop=1 means all predictors
are confounded; prop=0 means none are.

Equivalent to R's SimulationScripts/VarCP.R.

Usage
-----
python var_cp.py              # run simulations + plot
python var_cp.py --plot-only  # load saved results and plot
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hdam import fit_deconfounded_hdam, fit_hdam_with_est_factors, estimate_function
from hdam.fit_deconfounded_hdam import _svd_full
from _sim_utils import (
    f_true, METHS,
    run_parallel, make_pool, save_results, load_results, violin_plot,
)

# ---------------------------------------------------------------------------
# Fixed parameters
# ---------------------------------------------------------------------------

Q = 5
N = 400
P = 500
PROP_VEC = list(np.round(np.arange(0, 1.01, 0.05), 2))   # 0.00, 0.05, ..., 1.00
N_REP    = 100

CI_VEC = [
    ("equal",     False),
    ("decreasing", True),
]


# ---------------------------------------------------------------------------
# Single simulation run
# ---------------------------------------------------------------------------

def one_sim(prop, q, n, p, seed_val, decreasing_confounding=False):
    rng = np.random.default_rng(seed_val)

    Gamma = rng.uniform(-1, 1, size=(q, p))
    if decreasing_confounding:
        Gamma = (1.0 / np.arange(1, q + 1))[:, None] * Gamma

    # Sparse Gamma: keep each entry independently with probability prop
    Gamma0 = rng.binomial(1, prop, size=(q, p))
    Gamma  = Gamma * Gamma0

    psi = rng.uniform(0, 2, size=q)
    H   = rng.standard_normal((n, q))
    E   = rng.standard_normal((n, p))
    e   = 0.5 * rng.standard_normal(n)
    X   = H @ Gamma + E
    Y   = f_true(X) + H @ psi + e

    precomputed_svd = _svd_full(X - X.mean(axis=0))

    kw = dict(n_K=5, cv_method="1se", cv_k=5, n_lambda1=10, n_lambda2=25)
    lres_trim    = fit_deconfounded_hdam(Y, X.copy(), meth="trim",
                                         precomputed_svd=precomputed_svd, **kw)
    lres_none    = fit_deconfounded_hdam(Y, X.copy(), meth="none", **kw)
    lres_est_fac = fit_hdam_with_est_factors(Y, X.copy(),
                                              precomputed_svd=precomputed_svd, **kw)

    n_test = 1000
    H_test = rng.standard_normal((n_test, q))
    E_test = rng.standard_normal((n_test, p))
    X_test = H_test @ Gamma + E_test
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

    pool = make_pool(n_cores)
    try:
        for idx, prop in enumerate(PROP_VEC, start=1):
            for ci_tag, dec in CI_VEC:
                fname = out_dir / f"{ci_tag}CI_CP_{idx}.pkl"
                if fname.exists():
                    print(f"[skip] {fname.name}")
                    continue
                print(f"[run ] prop={prop:.2f}, {ci_tag} CI ...")
                args = [(prop, Q, N, P, sv, dec) for sv in seed_vec]
                results = run_parallel(one_sim, args, n_cores, pool=pool)
                save_results(results, fname)
                print(f"       saved {fname.name}")
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()


# ---------------------------------------------------------------------------
# Build DataFrame
# ---------------------------------------------------------------------------

def build_df(out_dir: Path, n_rep: int) -> pd.DataFrame:
    rows = []
    for idx, prop in enumerate(PROP_VEC, start=1):
        for ci_tag, _ in CI_VEC:
            fname = out_dir / f"{ci_tag}CI_CP_{idx}.pkl"
            if not fname.exists():
                continue
            results = load_results(fname)
            for res in results[:n_rep]:
                for mi, meth in enumerate(METHS):
                    rows.append({
                        "prop":     prop,
                        "CI":       ci_tag,
                        "meth":     meth,
                        "MSE":      res["MSE"][mi],
                        "s_active": res["active"][mi],
                    })

    df = pd.DataFrame(rows)
    df["prop"] = pd.Categorical(df["prop"], categories=PROP_VEC, ordered=True)
    df["meth"] = pd.Categorical(df["meth"], categories=METHS,    ordered=True)
    return df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(df: pd.DataFrame, plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    prop_labels = [f"{v:.2f}" for v in PROP_VEC]
    ci_labels = {"equal": "equal confounding influence",
                 "decreasing": "decreasing confounding influence"}

    for ci_tag, ci_lbl in ci_labels.items():
        sub = df[df["CI"] == ci_tag]
        if sub.empty:
            continue

        fig, axes = plt.subplots(2, 1, figsize=(16, 8))
        fig.suptitle(f"n=400, p=500, q=5, s=4  |  {ci_lbl}", fontsize=11)

        violin_plot(axes[0], sub, "prop", "MSE",      PROP_VEC,
                    title="MSE of f", xlabel="prop", ylabel="MSE")
        violin_plot(axes[1], sub, "prop", "s_active", PROP_VEC,
                    title="Size of estimated active set",
                    xlabel="prop", ylabel="Size of active set", hline=4)

        for ax in axes:
            ax.set_xticklabels(prop_labels, rotation=60, ha="right", fontsize=7)

        fig.tight_layout()
        out = plot_dir / f"VarCP_{ci_tag}CI.pdf"
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

    out_dir  = Path("SimulationResults/VarCP")
    plot_dir = Path("PlotResults/VarCP")

    if not args.plot_only:
        run_simulations(out_dir, args.n_rep, args.n_cores)

    df = build_df(out_dir, args.n_rep)
    if df.empty:
        print("No results found; run without --plot-only first.")
    else:
        make_plots(df, plot_dir)
