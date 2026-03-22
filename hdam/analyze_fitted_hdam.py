"""
Prediction functions for fitted HDAM models.

Equivalent to R's FunctionsHDAM/AnalyzeFittedHDAM.R:
  - estimate.function.1d()  ->  estimate_function_1d()
  - estimate.function()     ->  estimate_function()
  - estimate.fj.1d()        ->  estimate_fj_1d()
  - estimate.fj()           ->  estimate_fj()

All functions accept the dict returned by fit_deconfounded_hdam() or
fit_hdam_with_est_factors().

Note: predictor indices are 0-based in Python (vs. 1-based in R).
"""

import numpy as np

from .bspline import bspline_basis


def estimate_function(X: np.ndarray, model: dict) -> np.ndarray:
    """
    Predict the additive function f at a matrix of observations X in R^{n x p}.

    Equivalent to R's estimate.function().

    Parameters
    ----------
    X : ndarray (n, p)
    model : dict

    Returns
    -------
    y : ndarray (n,)
    """
    X_c = X - model["Xmeans"]
    y = np.full(X_c.shape[0], model["intercept"])
    for j in model["active"]:
        # bspline_basis clips out-of-range values, so constant extrapolation
        # (cj[0] below the left knot, cj[-1] above the right knot) is handled
        # automatically — no per-row branching needed.
        Bj = bspline_basis(X_c[:, j], model["breaks"][j])   # (n, K)
        y += Bj @ model["coefs"][j]
    return y


def estimate_function_1d(x: np.ndarray, model: dict) -> float:
    """
    Predict the additive function f at a single observation x in R^p.

    Equivalent to R's estimate.function.1d().

    Parameters
    ----------
    x : ndarray (p,)
    model : dict  output of fit_deconfounded_hdam() or fit_hdam_with_est_factors()

    Returns
    -------
    float
    """
    return float(estimate_function(np.asarray(x).reshape(1, -1), model)[0])


def estimate_fj(x: np.ndarray, j: int, model: dict) -> np.ndarray:
    """
    Predict the j-th component function f_j at multiple scalar values.

    Equivalent to R's estimate.fj().

    Parameters
    ----------
    x : array-like  scalar values for predictor j
    j : int  predictor index (0-based)
    model : dict

    Returns
    -------
    ndarray (len(x),)
    """
    x = np.asarray(x, dtype=float) - float(model["Xmeans"][j])
    B = bspline_basis(x, model["breaks"][j])   # (n, K)
    return B @ model["coefs"][j]


def estimate_fj_1d(x: float, j: int, model: dict) -> float:
    """
    Predict the j-th component function f_j at a single scalar value x.

    Equivalent to R's estimate.fj.1d().

    Parameters
    ----------
    x : float
    j : int  predictor index (0-based)
    model : dict

    Returns
    -------
    float
    """
    return float(estimate_fj(np.array([float(x)]), j, model)[0])
