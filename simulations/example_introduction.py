"""
Reproduce Figure 1.

Introductory example: histograms of MSE and active set size for the
setting n=300, p=800 from the VarP simulation.

Requires var_p.py to have been run first (p=800, rho=NULL, equal CI).

Equivalent to R's SimulationScripts/ExampleIntroduction.R.

Usage
-----
python example_introduction.py
"""

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _sim_utils import load_results, PALETTE, METHS

# ---------------------------------------------------------------------------
# Load results for p=800, rho=NULL, equal confounding influence
# ---------------------------------------------------------------------------

results_path = Path("SimulationResults/VarP/P_800_rhoNULL.pkl")

if not results_path.exists():
    raise FileNotFoundError(f"{results_path} not found. Run var_p.py first.")

results = load_results(results_path)
n_rep = min(100, len(results))

mse_dec = [r["MSE"][0] for r in results[:n_rep]]  # deconfounded
mse_naive = [r["MSE"][1] for r in results[:n_rep]]  # naive
act_dec = [r["active"][0] for r in results[:n_rep]]
act_naive = [r["active"][1] for r in results[:n_rep]]

# ---------------------------------------------------------------------------
# Plot (Figure 1)
# ---------------------------------------------------------------------------

PLOT_DIR = Path("PlotResults/FinalPlots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(9, 4))

# --- Left: MSE histogram ---
ax = axes[0]
bins_mse = np.arange(0, 13.0, 0.25)
ax.hist(
    mse_dec,
    bins=bins_mse,
    color=PALETTE["deconfounded"],
    alpha=0.5,
    label="deconfounded",
)
ax.hist(mse_naive, bins=bins_mse, color=PALETTE["naive"], alpha=0.5, label="naive")
ax.set_xlim(0, 12.5)
ax.set_xlabel("MSE", fontsize=11)
ax.set_ylabel("Frequency", fontsize=11)
ax.set_title("Mean squared error of f", fontsize=11)
ax.legend(fontsize=9)

# --- Right: active set size histogram ---
ax = axes[1]
bins_act = np.arange(0, 146, 5)
ax.hist(
    act_dec,
    bins=bins_act,
    color=PALETTE["deconfounded"],
    alpha=0.5,
    label="deconfounded",
)
ax.hist(act_naive, bins=bins_act, color=PALETTE["naive"], alpha=0.5, label="naive")
ax.axvline(x=4, color="black", linewidth=1.0)  # true active set size
ax.set_xlim(0, 145)
ax.set_xlabel("Size of estimated active set", fontsize=11)
ax.set_ylabel("Frequency", fontsize=11)
ax.set_title("Size of estimated active set", fontsize=11)
ax.legend(fontsize=9)

fig.tight_layout()
out_path = PLOT_DIR / "ExampleIntroduction.pdf"
fig.savefig(out_path)
plt.close(fig)
print(f"Saved {out_path}")
