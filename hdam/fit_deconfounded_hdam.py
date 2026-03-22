"""
Main deconfounded HDAM fitting functions.

Equivalent to R's FunctionsHDAM/FitDeconfoundedHDAM.R:
  - calcTrim()           ->  calc_trim()
  - cv.hdam()            ->  _cv_hdam()         (internal)
  - DeconfoundedHDAM()   ->  _deconfounded_hdam() (internal)
  - FitDeconfoundedHDAM()->  fit_deconfounded_hdam()

Cholesky note
-------------
R's chol(A) returns the *upper* triangular factor R s.t. R^T R = A.
numpy.linalg.cholesky(A) returns the *lower* triangular L s.t. L L^T = A.
So:  R_upper = np.linalg.cholesky(A).T
     R_inv   = np.linalg.inv(R_upper)   (equivalent to R's solve(chol(A)))
"""

import warnings
import numpy as np

from .bspline import bspline_basis
from .group_lasso import lambda_max_group, group_lasso_fista, group_lasso_path


# ---------------------------------------------------------------------------
# Spectral (trim) transformation
# ---------------------------------------------------------------------------

def calc_trim(X: np.ndarray) -> np.ndarray:
    """
    Compute the trim spectral transformation matrix Q.

    Q = I_n - U diag(1 - d_tilde / d) U^T

    where d_tilde = min(d, median(d)) element-wise, and U, d are the
    left singular vectors / values of X.

    Equivalent to R's calcTrim(X).

    Parameters
    ----------
    X : ndarray (n, p)

    Returns
    -------
    Q : ndarray (n, n)
    """
    n = X.shape[0]
    # full_matrices=False gives U of shape (n, min(n,p)), matching R's svd()
    U, d, _ = np.linalg.svd(X, full_matrices=False)
    d_tilde = np.minimum(d, np.median(d))
    # Diagonal weights: 1 - d_tilde/d
    w = 1.0 - d_tilde / d
    Q = np.eye(n) - (U * w) @ U.T  # equivalent to U @ diag(w) @ U.T
    return Q


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_basis(X: np.ndarray, K: int):
    """
    Build the orthonormalised B-spline design matrix for all predictors.

    Returns
    -------
    B_full : ndarray (n, 1 + p*K)   intercept prepended
    Rlist  : list of p arrays, each (K, K)  inv. upper-Cholesky factors
    lbreaks: list of p arrays   breakpoints per predictor
    groups : ndarray (1 + p*K,)  -1 for intercept, 0..p-1 for predictors
    unpen_mask : ndarray (1 + p*K,) bool
    """
    n, p = X.shape
    B = np.zeros((n, p * K))
    Rlist = []
    lbreaks = []

    for j in range(p):
        breaks = np.quantile(X[:, j], np.linspace(0, 1, K - 2))
        lbreaks.append(breaks)

        Bj = bspline_basis(X[:, j], breaks)          # (n, K)
        gram = Bj.T @ Bj / n                          # (K, K)
        R_upper = np.linalg.cholesky(gram).T          # upper triangular
        Rj_inv = np.linalg.inv(R_upper)               # (K, K)

        B[:, j * K:(j + 1) * K] = Bj @ Rj_inv
        Rlist.append(Rj_inv)

    # Prepend intercept column
    B_full = np.column_stack([np.ones(n), B])         # (n, 1 + p*K)

    # Group index: -1 = unpenalised intercept, 0..p-1 = predictors
    groups = np.concatenate([[-1], np.repeat(np.arange(p), K)])
    unpen_mask = groups < 0

    return B_full, Rlist, lbreaks, groups, unpen_mask


def _cv_hdam(
    QY: np.ndarray,
    QB: np.ndarray,
    groups: np.ndarray,
    unpen_mask: np.ndarray,
    lambdas: np.ndarray,
    k: int = 5,
) -> dict:
    """
    k-fold cross-validation for group lasso lambda selection.

    Equivalent to R's cv.hdam().

    Returns
    -------
    dict with keys: mse, se, lambda_min, lambda_1se
    """
    n = QB.shape[0]
    n_lam = len(lambdas)

    base = np.tile(np.arange(k), -(-n // k))[:n]   # ceiling division
    fold_ids = np.random.permutation(base)

    mses = np.zeros((n_lam, k))

    for fold in range(k):
        test = fold_ids == fold
        train = ~test
        QY_train, QY_test = QY[train], QY[test]
        QB_train, QB_test = QB[train], QB[test]

        coefs = group_lasso_path(QB_train, QY_train, groups, lambdas, unpen_mask)
        preds = QB_test @ coefs  # (n_test, n_lam)

        for li in range(n_lam):
            mses[li, fold] = np.mean((QY_test - preds[:, li]) ** 2)

    mse_agg = mses.mean(axis=1)
    se_agg = mses.std(axis=1) / np.sqrt(k)

    idx_min = int(np.argmin(mse_agg))
    lambda_min = lambdas[idx_min]

    # 1-SE rule: largest lambda whose mean CV error <= min + 1 SE
    # lambdas are *decreasing*, so "largest" = smallest index
    threshold = mse_agg[idx_min] + se_agg[idx_min]
    eligible = np.where(mse_agg <= threshold)[0]
    lambda_1se = lambdas[int(eligible[0])]

    return {
        "mse": mse_agg,
        "se": se_agg,
        "lambda_min": float(lambda_min),
        "lambda_1se": float(lambda_1se),
    }


def _deconfounded_hdam(
    Y: np.ndarray,
    X: np.ndarray,
    basis_k: int,
    meth: str = "trim",
    cv_method: str = "1se",
    cv_k: int = 5,
    n_lambda: int = 20,
) -> dict:
    """
    Fit deconfounded HDAM for a fixed number of basis functions K.

    Equivalent to R's DeconfoundedHDAM().
    """
    n, p = X.shape
    K = basis_k

    B, Rlist, lbreaks, groups, unpen_mask = _build_basis(X, K)

    if meth == "trim":
        Q = calc_trim(X)
        QY = Q @ Y
        QB = Q @ B
    elif meth == "none":
        QY = Y.copy()
        QB = B
    else:
        raise ValueError("meth must be 'trim' or 'none'")

    lam_max = lambda_max_group(QB, QY, groups, unpen_mask)
    lambdas = lam_max / (1000.0 ** (np.arange(n_lambda) / (n_lambda - 1)))

    res_cv = _cv_hdam(QY, QB, groups, unpen_mask, lambdas, k=cv_k)

    if cv_method == "1se":
        lambda_star = res_cv["lambda_1se"]
    elif cv_method == "min":
        lambda_star = res_cv["lambda_min"]
    else:
        warnings.warn("cv_method not recognised; using '1se'.")
        lambda_star = res_cv["lambda_1se"]

    coef = group_lasso_fista(QB, QY, groups, lambda_star, unpen_mask)

    lcoef = []
    active = []
    for j in range(p):
        cj = coef[1 + j * K: 1 + (j + 1) * K]
        lcoef.append(Rlist[j] @ cj)
        if np.sum(cj ** 2) > 0:
            active.append(j)

    return {
        "intercept": float(coef[0]),
        "breaks": lbreaks,
        "coefs": lcoef,
        "active": active,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_deconfounded_hdam(
    Y: np.ndarray,
    X: np.ndarray,
    n_K: int = 4,
    meth: str = "trim",
    cv_method: str = "1se",
    cv_k: int = 5,
    n_lambda1: int = 15,
    n_lambda2: int = 30,
) -> dict:
    """
    Fit a deconfounded high-dimensional additive model (HDAM).

    Equivalent to R's FitDeconfoundedHDAM().

    The function performs two-level cross-validation:
      1. Select the optimal number of B-spline basis functions K.
      2. Re-fit with K_opt and select the optimal regularisation lambda.

    Parameters
    ----------
    Y : ndarray (n,)
        Response variable.
    X : ndarray (n, p)
        Predictor matrix. Will be mean-centred internally.
    n_K : int
        Number of candidate K values to try (capped at 10).
    meth : {'trim', 'none'}
        'trim'  applies spectral deconfounding (recommended).
        'none'  fits a plain HDAM without deconfounding.
    cv_method : {'1se', 'min'}
        Lambda selection rule after cross-validation.
    cv_k : int
        Number of CV folds.
    n_lambda1 : int
        Number of lambdas for the K-selection CV.
    n_lambda2 : int
        Number of lambdas for the final lambda-selection CV.

    Returns
    -------
    dict with keys
        intercept : float
        breaks    : list of p arrays  (B-spline breakpoints per predictor)
        coefs     : list of p arrays  (B-spline coefficients per predictor)
        active    : list of int        (0-indexed active predictor indices)
        K_min     : int               (selected number of basis functions)
        Xmeans    : ndarray (p,)      (column means used for centring)
    """
    n, p = X.shape

    # Centre predictors (X is modified in-place copy)
    Xmeans = X.mean(axis=0)
    X = X - Xmeans

    # Candidate K values: between 4 and round(10 * n^0.2)
    if n_K > 10:
        n_K = 10
        warnings.warn("n_K capped at 10.")
    K_up = round(10 * n ** 0.2)
    vK = np.round(np.linspace(4, K_up, n_K)).astype(int)

    # Spectral transformation (computed once, shared across all K)
    if meth == "trim":
        Q = calc_trim(X)
        QY = Q @ Y
    elif meth == "none":
        Q = np.eye(n)
        QY = Y.copy()
    else:
        raise ValueError("meth must be 'trim' or 'none'")

    # Build basis matrices and lambda grids for each K
    lmodK = []
    for K in vK:
        B, Rlist, lbreaks, groups, unpen_mask = _build_basis(X, K)
        QB = Q @ B

        lam_max = lambda_max_group(QB, QY, groups, unpen_mask)
        lambdas = lam_max / (1000.0 ** (np.arange(n_lambda1) / (n_lambda1 - 1)))

        lmodK.append({
            "groups": groups,
            "unpen_mask": unpen_mask,
            "QB": QB,
            "lambdas": lambdas,
        })

    # Shared CV folds across all K values
    base = np.tile(np.arange(cv_k), -(-n // cv_k))[:n]
    fold_ids = np.random.permutation(base)

    MSES_agg = np.zeros((len(vK), n_lambda1))

    for fold in range(cv_k):
        test = fold_ids == fold
        train = ~test
        QY_train, QY_test = QY[train], QY[test]

        for i, modK in enumerate(lmodK):
            QB_train = modK["QB"][train]
            QB_test = modK["QB"][test]

            coefs = group_lasso_path(
                QB_train, QY_train,
                modK["groups"], modK["lambdas"], modK["unpen_mask"],
            )
            preds = QB_test @ coefs  # (n_test, n_lambda1)

            for li in range(n_lambda1):
                MSES_agg[i, li] += np.mean((QY_test - preds[:, li]) ** 2) / cv_k

    # Best (K, lambda) pair
    idx_min = np.unravel_index(np.argmin(MSES_agg), MSES_agg.shape)
    K_min = int(vK[idx_min[0]])

    # Re-fit with K_min using a finer lambda grid
    result = _deconfounded_hdam(
        Y, X,
        basis_k=K_min,
        meth=meth,
        cv_method=cv_method,
        cv_k=cv_k,
        n_lambda=n_lambda2,
    )
    result["K_min"] = K_min
    result["Xmeans"] = Xmeans
    return result
