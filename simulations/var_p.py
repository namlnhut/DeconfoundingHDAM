"""
Reproduce Figures 5, 6, 15 and 16  (vary dimension p of X).

Equivalent to R's SimulationScripts/VarP.R.

Usage
-----
python var_p.py              # run simulations + plot
python var_p.py --plot-only  # load saved results and plot
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import toeplitz
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hdam import fit_deconfounded_hdam, fit_hdam_with_est_factors, estimate_function
from hdam.fit_deconfounded_hdam import _svd_full
from _sim_utils import (
    f_true, PALETTE, METHS,
    run_parallel, make_pool, save_results, load_results, violin_plot,
)

# ---------------------------------------------------------------------------
# Fixed parameters
# ---------------------------------------------------------------------------

Q = 5
N = 300
P_VEC = [50, 100, 200, 400, 800]
N_REP = 100

SETTINGS = [
    ("rhoNULL", None,  False),
    ("rho04",   0.4,   False),
    ("rho08",   0.8,   False),
    ("rhoNULL_dec", None, True),
    ("rho04_dec",   0.4,  True),
    ("rho08_dec",   0.8,  True),
]


# ---------------------------------------------------------------------------
# Single simulation run
# ---------------------------------------------------------------------------

def one_sim(p, q, n, seed_val, rho=None, decreasing_confounding=False):
    rng = np.random.default_rng(seed_val)

    Gamma = rng.uniform(-1, 1, size=(q, p))
    if decreasing_confounding:
        Gamma = (1.0 / np.arange(1, q + 1))[:, None] * Gamma

    psi = rng.uniform(0, 2, size=q)
    H   = rng.standard_normal((n, q))

    if rho is None:
        E = rng.standard_normal((n, p))
    else:
        cov_row = rho ** np.arange(p)
        R_toe = np.linalg.cholesky(toeplitz(cov_row)).T
        E = rng.standard_normal((n, p)) @ R_toe

    e = 0.5 * rng.standard_normal(n)
    X = H @ Gamma + E
    Y = f_true(X) + H @ psi + e

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
        for p in P_VEC:
            for tag, rho, dec in SETTINGS:
                fname = out_dir / f"P_{p}_{tag}.pkl"
                if fname.exists():
                    print(f"[skip] {fname.name}")
                    continue
                print(f"[run ] p={p}, {tag} ...")
                args = [(p, Q, N, sv, rho, dec) for sv in seed_vec]
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
    for p in P_VEC:
        for tag, rho_val, dec in SETTINGS:
            fname = out_dir / f"P_{p}_{tag}.pkl"
            if not fname.exists():
                continue
            rho_str = "00" if rho_val is None else str(int(rho_val * 10)).zfill(2)
            ci_str  = "decreasing" if dec else "equal"
            results = load_results(fname)
            for res in results[:n_rep]:
                for mi, meth in enumerate(METHS):
                    rows.append({
                        "p": p, "rho": rho_str, "CI": ci_str,
                        "meth": meth,
                        "MSE": res["MSE"][mi],
                        "s_active": res["active"][mi],
                    })

    df = pd.DataFrame(rows)
    df["p"] = pd.Categorical(df["p"], categories=P_VEC, ordered=True)
    df["meth"] = pd.Categorical(df["meth"], categories=METHS, ordered=True)
    return df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(df: pd.DataFrame, plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)

    rho_map = {"00": "independent", "04": "Toeplitz(0.4)", "08": "Toeplitz(0.8)"}
    ci_map  = {"equal": "equal confounding", "decreasing": "decreasing confounding"}

    for rho, rho_lbl in rho_map.items():
        for ci, ci_lbl in ci_map.items():
            sub = df[(df["rho"] == rho) & (df["CI"] == ci)]
            if sub.empty:
                continue
            fig, axes = plt.subplots(2, 1, figsize=(10, 8))
            fig.suptitle(
                f"n=300, q=5, s=4  |  E: {rho_lbl}  |  {ci_lbl}", fontsize=11,
            )

            violin_plot(axes[0], sub, "p", "MSE",      P_VEC,
                        title="MSE of f", xlabel="p", ylabel="MSE")
            violin_plot(axes[1], sub, "p", "s_active", P_VEC,
                        title="Size of estimated active set",
                        xlabel="p", ylabel="Size of active set", hline=4)

            fig.tight_layout()
            out = plot_dir / f"VarP_rho{rho}_{ci}CI.pdf"
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

    out_dir  = Path("SimulationResults/VarP")
    plot_dir = Path("PlotResults/VarP")

    if not args.plot_only:
        run_simulations(out_dir, args.n_rep, args.n_cores)

    df = build_df(out_dir, args.n_rep)
    if df.empty:
        print("No results found; run without --plot-only first.")
    else:
        make_plots(df, plot_dir)
