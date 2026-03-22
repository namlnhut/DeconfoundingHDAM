"""
Ad-hoc comparative method: estimate hidden confounders from X,
then include them as unpenalised linear covariates.

Equivalent to R's FunctionsHDAM/EstimationFactors.R:
  - estimate.qhat()           ->  estimate_qhat()
  - estimate.Hhat()           ->  estimate_Hhat()
  - cv.hdam.withFactors()     ->  _cv_hdam_with_factors()   (internal)
  - HDAM.withFactors()        ->  _hdam_with_factors()      (internal)
  - FitHDAM.withFactors()     ->  fit_hdam_with_factors()
  - FitHDAM.withEstFactors()  ->  fit_hdam_with_est_factors()
"""

import warnings
import numpy as np

from .bspline import bspline_basis
from .group_lasso import lambda_max_group, group_lasso_fista, group_lasso_path


# ---------------------------------------------------------------------------
# Factor estimation
# ---------------------------------------------------------------------------

def estimate_qhat(X: np.ndarray) -> int:
    """
    Estimate the number of hidden confounders via the eigenvalue ratio method.

    Equivalent to R's estimate.qhat().

    Parameters
    ----------
    X : ndarray (n, p)

    Returns
    -------
    qhat : int  (1-indexed count, >= 1)
    """
    n, p = X.shape
    q_max = round(min(n, p) / 2)
    _, d, _ = np.linalg.svd(X, full_matrices=False)
    d = d[:q_max + 1]
    drat = d[:q_max] / d[1:q_max + 1]
    qhat = int(np.argmax(drat)) + 1   # +1 because argmax is 0-indexed
    return qhat


def estimate_Hhat(X: np.ndarray, qhat: int | None = None) -> np.ndarray:
    """
    Estimate the hidden factor matrix.

    Equivalent to R's estimate.Hhat().

    Parameters
    ----------
    X : ndarray (n, p)
    qhat : int or None  if None, estimated via estimate_qhat()

    Returns
    -------
    Hhat : ndarray (n, qhat)  scaled left singular vectors: sqrt(n) * U[:, :qhat]
    """
    n = X.shape[0]
    if qhat is None:
        qhat = estimate_qhat(X)
    U, _, _ = np.linalg.svd(X, full_matrices=False)
    return np.sqrt(n) * U[:, :qhat]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_basis_with_factors(X: np.ndarray, H: np.ndarray, K: int):
    """
    Build design matrix [1 | B-spline basis | H] and group labels.

    H columns enter as unpenalised linear covariates (like the intercept).

    Returns
    -------
    B_full : ndarray (n, 1 + p*K + q)
    Rlist  : list of p arrays (K, K)
    lbreaks: list of p arrays
    groups : ndarray (1 + p*K + q,)
    unpen_mask : ndarray (1 + p*K + q,) bool
    """
    n, p = X.shape
    q = H.shape[1]

    B = np.zeros((n, p * K))
    Rlist = []
    lbreaks = []

    for j in range(p):
        breaks = np.quantile(X[:, j], np.linspace(0, 1, K - 2))
        lbreaks.append(breaks)

        Bj = bspline_basis(X[:, j], breaks)
        gram = Bj.T @ Bj / n
        R_upper = np.linalg.cholesky(gram).T
        Rj_inv = np.linalg.inv(R_upper)

        B[:, j * K:(j + 1) * K] = Bj @ Rj_inv
        Rlist.append(Rj_inv)

    # [intercept | basis | H]  -- intercept and H are unpenalised
    B_full = np.column_stack([np.ones(n), B, H])

    groups = np.concatenate([
        [-1],                            # intercept -> unpenalised
        np.repeat(np.arange(p), K),     # B-spline groups
        np.full(q, -1),                  # H columns -> unpenalised
    ])
    unpen_mask = groups < 0

    return B_full, Rlist, lbreaks, groups, unpen_mask


def _cv_hdam_with_factors(
    Y: np.ndarray,
    B: np.ndarray,
    groups: np.ndarray,
    unpen_mask: np.ndarray,
    lambdas: np.ndarray,
    k: int = 5,
) -> dict:
    """
    k-fold CV for HDAM with factors.

    Equivalent to R's cv.hdam.withFactors().
    """
    n = B.shape[0]
    n_lam = len(lambdas)

    base = np.tile(np.arange(k), -(-n // k))[:n]
    fold_ids = np.random.permutation(base)

    mses = np.zeros((n_lam, k))

    for fold in range(k):
        test = fold_ids == fold
        train = ~test
        Y_train, Y_test = Y[train], Y[test]
        B_train, B_test = B[train], B[test]

        coefs = group_lasso_path(B_train, Y_train, groups, lambdas, unpen_mask)
        preds = B_test @ coefs

        for li in range(n_lam):
            mses[li, fold] = np.mean((Y_test - preds[:, li]) ** 2)

    mse_agg = mses.mean(axis=1)
    se_agg = mses.std(axis=1) / np.sqrt(k)

    idx_min = int(np.argmin(mse_agg))
    lambda_min = lambdas[idx_min]

    threshold = mse_agg[idx_min] + se_agg[idx_min]
    eligible = np.where(mse_agg <= threshold)[0]
    lambda_1se = lambdas[int(eligible[0])]

    return {
        "mse": mse_agg,
        "se": se_agg,
        "lambda_min": float(lambda_min),
        "lambda_1se": float(lambda_1se),
    }


def _hdam_with_factors(
    Y: np.ndarray,
    H: np.ndarray,
    X: np.ndarray,
    basis_k: int,
    cv_method: str = "1se",
    cv_k: int = 5,
    n_lambda: int = 20,
) -> dict:
    """
    Fit HDAM with factors for a fixed K.

    Equivalent to R's HDAM.withFactors().
    """
    n, p = X.shape
    K = basis_k

    B, Rlist, lbreaks, groups, unpen_mask = _build_basis_with_factors(X, H, K)

    lam_max = lambda_max_group(B, Y, groups, unpen_mask)
    lambdas = lam_max / (1000.0 ** (np.arange(n_lambda) / (n_lambda - 1)))

    res_cv = _cv_hdam_with_factors(Y, B, groups, unpen_mask, lambdas, k=cv_k)

    if cv_method == "1se":
        lambda_star = res_cv["lambda_1se"]
    elif cv_method == "min":
        lambda_star = res_cv["lambda_min"]
    else:
        warnings.warn("cv_method not recognised; using '1se'.")
        lambda_star = res_cv["lambda_1se"]

    coef = group_lasso_fista(B, Y, groups, lambda_star, unpen_mask)

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

def fit_hdam_with_factors(
    Y: np.ndarray,
    H: np.ndarray,
    X: np.ndarray,
    n_K: int = 4,
    cv_method: str = "1se",
    cv_k: int = 5,
    n_lambda1: int = 15,
    n_lambda2: int = 30,
) -> dict:
    """
    Fit HDAM with *known* hidden factors H entering linearly.

    Equivalent to R's FitHDAM.withFactors().

    Parameters
    ----------
    Y : ndarray (n,)
    H : ndarray (n, q)  known factor matrix
    X : ndarray (n, p)  predictors (should already be mean-centred)
    n_K, cv_method, cv_k, n_lambda1, n_lambda2 : see fit_deconfounded_hdam

    Returns
    -------
    dict with keys: intercept, breaks, coefs, active, K_min
    """
    n, p = X.shape

    if n_K > 10:
        n_K = 10
        warnings.warn("n_K capped at 10.")
    K_up = round(10 * n ** 0.2)
    vK = np.round(np.linspace(4, K_up, n_K)).astype(int)

    lmodK = []
    for K in vK:
        B, _, _, groups, unpen_mask = _build_basis_with_factors(X, H, K)
        lam_max = lambda_max_group(B, Y, groups, unpen_mask)
        lambdas = lam_max / (1000.0 ** (np.arange(n_lambda1) / (n_lambda1 - 1)))
        lmodK.append({
            "groups": groups,
            "unpen_mask": unpen_mask,
            "B": B,
            "lambdas": lambdas,
        })

    base = np.tile(np.arange(cv_k), -(-n // cv_k))[:n]
    fold_ids = np.random.permutation(base)

    MSES_agg = np.zeros((len(vK), n_lambda1))

    for fold in range(cv_k):
        test = fold_ids == fold
        train = ~test
        Y_train, Y_test = Y[train], Y[test]

        for i, modK in enumerate(lmodK):
            B_train = modK["B"][train]
            B_test = modK["B"][test]

            coefs = group_lasso_path(
                B_train, Y_train,
                modK["groups"], modK["lambdas"], modK["unpen_mask"],
            )
            preds = B_test @ coefs

            for li in range(n_lambda1):
                MSES_agg[i, li] += np.mean((Y_test - preds[:, li]) ** 2) / cv_k

    idx_min = np.unravel_index(np.argmin(MSES_agg), MSES_agg.shape)
    K_min = int(vK[idx_min[0]])

    result = _hdam_with_factors(
        Y, H, X,
        basis_k=K_min,
        cv_method=cv_method,
        cv_k=cv_k,
        n_lambda=n_lambda2,
    )
    result["K_min"] = K_min
    return result


def fit_hdam_with_est_factors(
    Y: np.ndarray,
    X: np.ndarray,
    n_K: int = 4,
    cv_method: str = "1se",
    cv_k: int = 5,
    n_lambda1: int = 15,
    n_lambda2: int = 30,
) -> dict:
    """
    Estimate hidden factors from X, then fit HDAM with those factors.

    Equivalent to R's FitHDAM.withEstFactors().

    Parameters
    ----------
    Y : ndarray (n,)
    X : ndarray (n, p)  will be mean-centred internally

    Returns
    -------
    dict with keys: intercept, breaks, coefs, active, K_min, Xmeans
    """
    Xmeans = X.mean(axis=0)
    X = X - Xmeans

    qhat = estimate_qhat(X)
    Hhat = estimate_Hhat(X, qhat=qhat)

    result = fit_hdam_with_factors(
        Y=Y, H=Hhat, X=X,
        n_K=n_K,
        cv_method=cv_method,
        cv_k=cv_k,
        n_lambda1=n_lambda1,
        n_lambda2=n_lambda2,
    )
    result["Xmeans"] = Xmeans
    return result
