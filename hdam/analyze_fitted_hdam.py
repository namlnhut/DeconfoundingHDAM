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
    y = model["intercept"]
    x = x - model["Xmeans"]   # centre with training means

    for j in model["active"]:
        breaks_j = model["breaks"][j]
        xj = float(x[j])
        cj = model["coefs"][j]

        if xj < breaks_j[0]:
            y += cj[0]
        elif xj > breaks_j[-1]:
            y += cj[-1]
        else:
            Bj = bspline_basis(np.array([xj]), breaks_j)   # (1, K)
            y += float(Bj @ cj)

    return float(y)


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
    return np.array([estimate_function_1d(x, model) for x in X])


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
    x = float(x) - float(model["Xmeans"][j])
    breaks = model["breaks"][j]
    coefs = model["coefs"][j]

    if x < breaks[0]:
        return float(coefs[0])
    elif x > breaks[-1]:
        return float(coefs[-1])
    else:
        B = bspline_basis(np.array([x]), breaks)
        return float(B @ coefs)


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
    return np.array([estimate_fj_1d(xi, j, model) for xi in np.asarray(x)])
