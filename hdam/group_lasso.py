"""
Group lasso fitting via FISTA and supporting utilities.

Replaces R's grplasso package:
  - lambdamax()  ->  lambda_max_group()
  - grplasso()   ->  group_lasso_fista() / group_lasso_path()

Objective minimised:
    (1 / 2n) * ||y - X beta||^2  +  lambda * sum_j ||beta_j||_2

where the sum runs over penalised groups only.

Convention for `groups` array (length p):
  -  >= 0  : integer group label (penalised)
  -   < 0  : unpenalised (intercept, factor columns, …)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Lambda max
# ---------------------------------------------------------------------------

def lambda_max_group(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    unpen_mask: np.ndarray,
) -> float:
    """
    Compute the smallest lambda that zeros all penalised groups.

    At lambda_max the gradient condition ||X_j^T r||_2 / n <= lambda is
    tight for the most active group, where r is the residual after fitting
    the unpenalised columns by OLS.

    Parameters
    ----------
    X : ndarray (n, p)
    y : ndarray (n,)
    groups : ndarray (p,)  integer labels; values < 0 are unpenalised
    unpen_mask : ndarray (p,) bool  True where unpenalised

    Returns
    -------
    float
    """
    n = len(y)
    unpen_idx = np.where(unpen_mask)[0]

    if len(unpen_idx) > 0:
        X_u = X[:, unpen_idx]
        beta_u, _, _, _ = np.linalg.lstsq(X_u, y, rcond=None)
        r = y - X_u @ beta_u
    else:
        r = y.copy()

    lam_max = 0.0
    for g in np.unique(groups[~unpen_mask]):
        idx = groups == g
        score = np.linalg.norm(X[:, idx].T @ r) / n
        if score > lam_max:
            lam_max = score

    return float(lam_max)


# ---------------------------------------------------------------------------
# FISTA solver
# ---------------------------------------------------------------------------

def group_lasso_fista(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    lam: float,
    unpen_mask: np.ndarray,
    max_iter: int = 2000,
    tol: float = 1e-8,
    beta_init: np.ndarray | None = None,
) -> np.ndarray:
    """
    Fit the group lasso at a single lambda value using FISTA.

    Parameters
    ----------
    X : ndarray (n, p)
    y : ndarray (n,)
    groups : ndarray (p,)
    lam : float
    unpen_mask : ndarray (p,) bool
    max_iter : int
    tol : float  convergence tolerance (max |beta_new - beta_old|)
    beta_init : ndarray (p,) or None  warm-start coefficients

    Returns
    -------
    beta : ndarray (p,)
    """
    n, p = X.shape
    beta = np.zeros(p) if beta_init is None else beta_init.copy()
    z = beta.copy()
    t_val = 1.0

    # Lipschitz constant L = sigma_max(X)^2 / n
    L = float(np.linalg.norm(X, ord=2) ** 2) / n

    pen_groups = np.unique(groups[~unpen_mask])
    thresh = lam / L  # scalar threshold used in group soft-thresholding

    for _ in range(max_iter):
        beta_old = beta.copy()

        # Gradient of (1/2n)||y - Xz||^2  w.r.t. z
        grad = X.T @ (X @ z - y) / n

        # Gradient step
        u = z - grad / L

        # Proximal step: group soft-thresholding for penalised groups;
        # unpenalised variables keep the gradient-step value (u already set)
        beta_new = u.copy()
        for g in pen_groups:
            idx = groups == g
            norm_u = np.linalg.norm(u[idx])
            if norm_u > thresh:
                beta_new[idx] = (1.0 - thresh / norm_u) * u[idx]
            else:
                beta_new[idx] = 0.0

        # FISTA momentum
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_val ** 2)) / 2.0
        z = beta_new + ((t_val - 1.0) / t_new) * (beta_new - beta_old)
        beta = beta_new
        t_val = t_new

        if np.max(np.abs(beta - beta_old)) < tol:
            break

    return beta


# ---------------------------------------------------------------------------
# Regularisation path (warm-started)
# ---------------------------------------------------------------------------

def group_lasso_path(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    lambdas: np.ndarray,
    unpen_mask: np.ndarray,
    max_iter: int = 2000,
    tol: float = 1e-8,
) -> np.ndarray:
    """
    Fit the group lasso along a sequence of lambdas (warm-started).

    Equivalent to calling grplasso() in R with a vector of lambda values.

    Parameters
    ----------
    lambdas : array-like  decreasing sequence of lambda values

    Returns
    -------
    coefs : ndarray (p, len(lambdas))
        Column i corresponds to lambdas[i].
    """
    p = X.shape[1]
    lambdas = np.asarray(lambdas)
    coefs = np.zeros((p, len(lambdas)))
    beta = np.zeros(p)

    for i, lam in enumerate(lambdas):
        beta = group_lasso_fista(
            X, y, groups, lam, unpen_mask,
            max_iter=max_iter, tol=tol, beta_init=beta,
        )
        coefs[:, i] = beta

    return coefs
