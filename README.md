# Spectral Deconfounding for High-Dimensional Sparse Additive Models

This repository contains the code for reproducing the plots in the paper
<i>Cyrill Scheidegger, Zijian Guo and Peter Bühlmann (2023). Spectral deconfounding for high-dimensional sparse additive models, arXiv:2312.02860</i>.

## Repository Structure

- **`hdam/`** — Python package implementing the core statistical methods
- **`r/`** — R reference implementation
- **`simulations/`** — Scripts to reproduce all figures from the paper (Python and R)

## R Implementation

The file `FitDeconfoundedHDAM.R` in the folder `r` contains the function `FitDeconfoundedHDAM` to fit a deconfounded high-dimensional additive model and some helper functions. The file `EstimationFactors.R` in the folder `r` contains the function `FitHDAM.withEstFactors` that implements an ad hoc method to achieve the same goal. The file `AnalyzeFittedHDAM.R` in the folder `r` contains functions to analyze/predict high-dimensional additive models based on the output of the function `FitDeconfoundedHDAM`.

The R scripts in `simulations/` generate the plots from the paper. `ExampleIntroduction.R` generates Figure 1; `SingularValuePlot.R` generates Figure 2; `VarN.R` generates Figures 3, 4, 13 and 14; `VarP.R` generates Figures 5, 6, 15 and 16; `VarCS.R` generates Figures 7 and 8; `MotifEvaluation.R` generates Figures 9, 10, 11 and 12; `VarCP.R` generates Figures 17 and 18; `Nonlinear.R` generates Figures 19 and 20.

## Python Implementation

### Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

To run `motif_evaluation.py`, also install:

```bash
pip install pyreadr
```

### Running the simulations

All Python simulation scripts must be run from the **`simulations/`** directory. Each script supports a `--plot-only` flag to skip re-running simulations and load previously saved results instead.

**Figure 1** — Introductory example (requires `var_p.py` with `p=800` to have been run first):
```bash
python example_introduction.py
```

**Figure 2** — Singular value plot:
```bash
python singular_value_plot.py
```

**Figures 3, 4, 13, 14** — Vary sample size `n`:
```bash
python var_n.py              # run simulations + plot
python var_n.py --plot-only  # plot from saved results
```

**Figures 5, 6, 15, 16** — Vary predictor dimension `p`:
```bash
python var_p.py              # run simulations + plot
python var_p.py --plot-only  # plot from saved results
```

**Figures 7, 8** — Vary confounding strength:
```bash
python var_cs.py              # run simulations + plot
python var_cs.py --plot-only  # plot from saved results
```

**Figures 9, 10, 11, 12** — Motif regression (real data):
```bash
python motif_evaluation.py              # fit models + plot
python motif_evaluation.py --plot-only  # plot from saved results
```

**Figures 17, 18** — Vary confounding structure:
```bash
python var_cp.py              # run simulations + plot
python var_cp.py --plot-only  # plot from saved results
```

**Figures 19, 20** — Nonlinear confounding:
```bash
python nonlinear.py              # run simulations + plot
python nonlinear.py --plot-only  # plot from saved results
```

### Output

Simulation results are saved to `SimulationResults/` and plots to `PlotResults/`, both created automatically relative to the project root.
