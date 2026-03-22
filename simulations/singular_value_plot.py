"""
Reproduce Figure 2.

Plots of the singular values of X for two simulation scenarios:
  - equal confounding influence
  - decreasing confounding influence

Equivalent to R's SimulationScripts/SingularValuePlot.R.

Usage
-----
python singular_value_plot.py
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

Q = 5
N = 100
P = 300
SEED = 1219
PLOT_DIR = Path("PlotResults/FinalPlots")

# ---------------------------------------------------------------------------
# Generate data
# ---------------------------------------------------------------------------

rng = np.random.default_rng(SEED)

Gamma_equal = rng.uniform(-1, 1, size=(Q, P))
gam_weight = 1.0 / np.arange(1, Q + 1)          # 1/1, 1/2, ..., 1/q
Gamma_decreasing = gam_weight[:, None] * Gamma_equal

H = rng.standard_normal((N, Q))
E = rng.standard_normal((N, P))

X_equal      = H @ Gamma_equal      + E
X_decreasing = H @ Gamma_decreasing + E

sv_equal      = np.linalg.svd(X_equal,      compute_uv=False)
sv_decreasing = np.linalg.svd(X_decreasing, compute_uv=False)

# ---------------------------------------------------------------------------
# Plot (Figure 2)
# ---------------------------------------------------------------------------

PLOT_DIR.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(9, 4))

axes[0].scatter(range(1, len(sv_equal) + 1), sv_equal, s=12, color="black")
axes[0].set_xlabel("l", fontsize=11)
axes[0].set_ylabel("singular value $d_l$", fontsize=11)
axes[0].set_title("equal confounding influence", fontsize=11)

axes[1].scatter(range(1, len(sv_decreasing) + 1), sv_decreasing, s=12, color="black")
axes[1].set_xlabel("l", fontsize=11)
axes[1].set_ylabel("singular value $d_l$", fontsize=11)
axes[1].set_title("decreasing confounding influence", fontsize=11)

fig.tight_layout()
out_path = PLOT_DIR / "SingularValuePlot.pdf"
fig.savefig(out_path)
plt.close(fig)
print(f"Saved {out_path}")
