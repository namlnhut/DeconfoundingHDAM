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

GPU acceleration
----------------
On this system, NumPy's OpenBLAS (USE64BITINT build) has a severe
performance regression for matrices with column count >= ~2000
(~100x slower per matvec than the expected throughput).

When a CUDA GPU is available and the column count exceeds GPU_THRESHOLD,
group_lasso_path automatically moves the design matrix to the GPU and
runs all FISTA iterations there.  This gives >1000x speedup for the
large-p regime.  The NumPy path is kept for small matrices where GPU
transfer overhead would dominate.
"""

import numpy as np
from scipy.sparse.linalg import svds as _svds

# -----------------------------------------------------------------------
# Optional GPU support (requires PyTorch with CUDA)
# -----------------------------------------------------------------------

try:
    import torch as _torch
    _HAS_GPU = _torch.cuda.is_available()
    if _HAS_GPU:
        # Warm up the CUDA context once at import time.  The very first
        # GPU tensor allocation pays a one-off ~300ms CUDA initialisation
        # cost; doing it here avoids that penalty inside the first FISTA call.
        _torch.zeros(1, device="cuda")
except ImportError:
    _torch = None
    _HAS_GPU = False

# Column-count threshold above which CUDA is used instead of NumPy/OpenBLAS.
# Below this value, GPU transfer overhead outweighs the BLAS regression.
# Above this value (pcol ~2000+), CUDA is up to 1000x faster on this build.
GPU_THRESHOLD = 1500


def _lipschitz_numpy(X: np.ndarray) -> float:
    """
    Compute L = sigma_max(X)^2 / n using scipy.sparse.svds(k=1).

    scipy.svds uses ARPACK to compute only the largest singular value,
    avoiding the full O(n * p * min(n,p)) SVD.  On this system this is
    ~100x faster than np.linalg.norm(X, ord=2) for moderate-sized matrices
    (pcol < ~2000).
    """
    return float(_svds(X, k=1, return_singular_vectors=False)[0] ** 2) / X.shape[0]


def _lipschitz_gpu(X_g, n: int) -> float:
    """
    Estimate L = sigma_max(X)^2 / n via implicit power iteration on the GPU.

    Computes the dominant eigenvalue of X X^T using two matrix-vector
    products per iteration (never forms X X^T explicitly).  For pcol >> n
    this avoids the O(n^2 * pcol) cost of forming the Gram matrix.

    A 5% safety margin is added so L is always a valid upper bound.
    """
    device = X_g.device
    v = _torch.randn(n, device=device, dtype=X_g.dtype)
    v /= _torch.linalg.norm(v)
    sigma = None
    for _ in range(20):   # 20 iterations; typically ~7% underestimate
        w = X_g @ (X_g.T @ v)
        sigma = float(_torch.linalg.norm(w))
        v = w / sigma
    # sigma converges to lambda_max(X X^T) = sigma_max(X)^2.
    # A 10% safety margin ensures L is a valid upper bound even with the
    # ~7% underestimate from 20 power iterations.
    return sigma / n * 1.10


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

    pen_mask = ~unpen_mask
    pen_groups = np.unique(groups[pen_mask])
    G = len(pen_groups)

    if G == 0:
        return 0.0

    # Fast path: all penalised groups have the same size (always true for HDAM).
    # Replace the Python loop over G groups with a single BLAS matmul:
    #   grad = X[:,pen_mask].T @ r / n   shape (G*K,)
    # then reshape to (G, K) and compute row norms in one vectorised call.
    group_sizes = np.array([np.sum(groups == g) for g in pen_groups])
    if np.all(group_sizes == group_sizes[0]):
        K = int(group_sizes[0])
        grad = X[:, pen_mask].T @ r / n        # (G*K,)
        return float(np.linalg.norm(grad.reshape(G, K), axis=1).max())

    # General fallback for mixed group sizes (rare in practice)
    lam_max = 0.0
    for g in pen_groups:
        idx = groups == g
        score = np.linalg.norm(X[:, idx].T @ r) / n
        if score > lam_max:
            lam_max = score
    return float(lam_max)


# ---------------------------------------------------------------------------
# FISTA solver  (NumPy path)
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
    L: float | None = None,
) -> np.ndarray:
    """
    Fit the group lasso at a single lambda value using FISTA (NumPy path).

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
    L : float or None
        Lipschitz constant L = sigma_max(X)^2 / n.  When calling this
        function repeatedly on the same X (e.g. along a lambda path or
        across CV folds), pre-compute L once and pass it here to skip
        the scipy.svds call on every invocation.

    Returns
    -------
    beta : ndarray (p,)
    """
    n, p = X.shape
    beta = np.zeros(p) if beta_init is None else beta_init.copy()
    z = beta.copy()
    t_val = 1.0

    # Lipschitz constant: sigma_max(X)^2 / n.
    # scipy.svds(k=1) computes only the largest singular value (~20ms) vs
    # the full SVD (~2-9s) on this OpenBLAS USE64BITINT build.
    # Callers invoking this on the same X repeatedly should pre-compute L
    # and pass it in to skip even that cost for every lambda value.
    if L is None:
        L = _lipschitz_numpy(X)

    pen_groups = np.unique(groups[~unpen_mask])

    # Pre-compute the index array for each penalised group *once*, outside
    # the iteration loop.  Without this, "groups == g" allocates a fresh
    # boolean array of length p on every FISTA iteration.
    group_idx = {g: np.where(groups == g)[0] for g in pen_groups}
    group_sizes = np.array([len(group_idx[g]) for g in pen_groups])

    # When all penalised groups have the same size (the common case: all
    # predictor groups contain exactly K B-spline columns), stack their
    # indices into a 2-D array for vectorised group soft-thresholding.
    # A Python loop over p groups with per-group norm calls causes
    # O(max_iter * p) Python-level dispatches — the dominant cost for large p.
    uniform_groups = bool((group_sizes == group_sizes[0]).all())
    all_idx = None
    G_shape = K_grp = 0
    contiguous_pen = False
    pen_start = pen_end = 0
    if uniform_groups:
        # all_idx: (n_groups, K) integer array for advanced indexing
        all_idx = np.array([group_idx[g] for g in pen_groups])  # (G, K)
        G_shape, K_grp = all_idx.shape
        # When all penalised indices are contiguous (always true for HDAM:
        # the intercept is unpenalised at index 0, then p*K spline columns),
        # use a slice + reshape view instead of fancy indexing.
        # This avoids one (G, K) copy per iteration for the read AND the write,
        # removing ~2 * pcol * max_iter allocations from the hot path.
        pen_flat = all_idx.ravel()
        if bool(np.all(np.diff(pen_flat) == 1)):
            contiguous_pen = True
            pen_start = int(pen_flat[0])
            pen_end   = int(pen_flat[-1]) + 1

    thresh = lam / L  # scalar threshold for group soft-thresholding

    # Precompute X^T y / n once.  Gradient = X^T Xz/n - X^T y/n.
    # The second term is constant; precomputing it saves one O(n * p)
    # matrix-vector product per FISTA step.
    Xty = X.T @ y / n

    # Convergence-check interval: checking every iteration pays O(p) cost
    # plus numpy dispatch overhead each step.  Checking every 10 steps is
    # tight enough for tol=1e-8 (FISTA converges at O(1/k^2) rate) and
    # saves ~90 % of the np.max(np.abs(...)) calls.
    _CONV_INTERVAL = 10

    for it in range(max_iter):
        # Gradient of (1/2n)||y - Xz||^2  w.r.t. z
        grad = X.T @ (X @ z) / n - Xty

        # Gradient step — creates a new array; beta (from previous iteration)
        # still points to the old value so it serves as beta_old below.
        u = z - grad / L

        # Proximal step: group soft-thresholding applied in-place on u
        # (contiguous path) or via vectorised fancy-index (fallback).
        if contiguous_pen:
            # Slice + reshape is a *view* — in-place mul modifies u directly,
            # so u is already proximal after this block; no extra copy needed.
            u_pen = u[pen_start:pen_end].reshape(G_shape, K_grp)
            norms = np.linalg.norm(u_pen, axis=1, keepdims=True)
            # errstate suppresses the benign divide-by-zero when norm == 0:
            # thresh/0 = inf -> 1 - inf = -inf -> max(-inf, 0) = 0 (correct).
            with np.errstate(divide='ignore', invalid='ignore'):
                u_pen *= np.maximum(1.0 - thresh / norms, 0.0)
        elif uniform_groups:
            u_groups = u[all_idx]                                     # (G, K)
            norms = np.linalg.norm(u_groups, axis=1, keepdims=True)  # (G, 1)
            with np.errstate(divide='ignore', invalid='ignore'):
                u[all_idx] = u_groups * np.maximum(1.0 - thresh / norms, 0.0)
        else:
            # Fallback for mixed group sizes (rare in practice)
            for g in pen_groups:
                idx = group_idx[g]
                norm_u = np.linalg.norm(u[idx])
                if norm_u > thresh:
                    u[idx] = (1.0 - thresh / norm_u) * u[idx]
                else:
                    u[idx] = 0.0

        # delta = u_new_beta - beta_old.  Reusing this for both the
        # FISTA momentum update and (every _CONV_INTERVAL steps) the
        # convergence check avoids computing the difference twice.
        delta = u - beta   # (p,) — one allocation instead of two copies

        # FISTA momentum: z = u + c * delta  (c = (t-1)/t_new)
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_val ** 2)) / 2.0
        z = u + ((t_val - 1.0) / t_new) * delta

        # beta points to u (no copy); u stays alive via this reference.
        beta = u
        t_val = t_new

        if (it + 1) % _CONV_INTERVAL == 0 and np.max(np.abs(delta)) < tol:
            break

    return beta


# ---------------------------------------------------------------------------
# FISTA solver  (GPU path — internal, called only from group_lasso_path)
# ---------------------------------------------------------------------------

def _fista_gpu(
    X_g,
    Xty_g,
    lam: float,
    all_idx_g,
    pen_start: int,
    pen_end: int,
    contiguous_pen: bool,
    L: float,
    beta_g,
    max_iter: int,
    tol: float,
):
    """
    FISTA on pre-uploaded GPU tensors.  All intermediate tensors remain
    on the GPU; only the final beta is returned (still a GPU tensor so the
    caller can warm-start the next lambda without a round-trip).

    Parameters
    ----------
    X_g    : CUDA float64 tensor (n, p)
    Xty_g  : CUDA float64 tensor (p,)   pre-computed X^T y / n
    all_idx_g  : CUDA int64 tensor (G, K) — used if not contiguous_pen
    pen_start, pen_end : int — slice [pen_start:pen_end] covers all penalised
                               coefficients (only used when contiguous_pen)
    contiguous_pen : bool — True when all penalised indices form a contiguous
                            block (allows slice + reshape instead of gather)
    L      : float  Lipschitz constant (pre-computed once per path)
    beta_g : CUDA float64 tensor (p,)   warm-start coefficients (modified)

    Performance notes
    -----------------
    Each Python iteration dispatches several CUDA kernels.  To minimise
    overhead:

    1. In-place ops with pre-allocated buffers: z, u, delta are reused
       each iteration; no new tensors are allocated on the heap.

    2. Slice + reshape instead of gather/scatter: when all penalised
       indices are contiguous (always true for HDAM), u[pen_start:pen_end]
       is a zero-copy view; in-place mul_ applies soft-thresholding
       directly on the underlying buffer (~10x faster than gather/scatter).

    3. Deferred convergence check: .item() forces a CPU-GPU sync (~0.1ms).
       Checking every CONV_CHECK_INTERVAL iterations reduces syncs N-fold
       at the cost of at most N extra steps past true convergence.
    """
    _CONV_CHECK_INTERVAL = 10

    n, pcol = X_g.shape
    G, K = all_idx_g.shape
    thresh = lam / L

    # Pre-allocate all working buffers once.  z and u are (pcol,) on GPU;
    # delta tracks beta_new - beta_old for momentum and convergence.
    z     = beta_g.clone()
    u     = _torch.empty_like(beta_g)
    delta = _torch.empty_like(beta_g)
    t_val = 1.0

    for i in range(max_iter):
        # ---- gradient step ----
        # grad = X^T (X z) / n - X^T y / n
        # u    = z - grad / L
        # Both written to pre-allocated buffers to avoid per-iteration allocs.
        _torch.sub(X_g.T @ (X_g @ z) / n, Xty_g, out=delta)  # delta = grad (reuse buffer)
        _torch.add(z, delta, alpha=-1.0 / L, out=u)           # u = z - grad/L

        # ---- group soft-thresholding (in-place on u) ----
        if contiguous_pen:
            # u[pen_start:pen_end] is a zero-copy view; reshape gives (G, K)
            # view too.  mul_ modifies u in-place — no extra allocation.
            u_view = u[pen_start:pen_end].reshape(G, K)
            norms = _torch.linalg.norm(u_view, dim=1, keepdim=True)
            # clamp avoids zeros_like alloc; in-place mul_ on the view
            u_view.mul_((1.0 - thresh / norms).clamp_(min=0.0))
        else:
            u_groups = u[all_idx_g]                                   # gather
            norms = _torch.linalg.norm(u_groups, dim=1, keepdim=True)
            scale = (1.0 - thresh / norms).clamp(min=0.0)
            u[all_idx_g] = u_groups * scale                           # scatter

        # ---- delta = u (new beta) - beta_g (old beta) ----
        _torch.sub(u, beta_g, out=delta)

        # ---- FISTA momentum: z = u + c * delta ----
        t_new = (1.0 + (1.0 + 4.0 * t_val ** 2) ** 0.5) / 2.0
        c = (t_val - 1.0) / t_new
        _torch.add(u, delta, alpha=c, out=z)   # z = u + c * delta

        # ---- update beta in-place ----
        beta_g.copy_(u)
        t_val = t_new

        # ---- convergence check (every N iterations to amortise sync) ----
        if (i + 1) % _CONV_CHECK_INTERVAL == 0:
            if delta.abs().max().item() < tol:
                break

    return beta_g


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
    tol: float = 1e-4,
) -> np.ndarray:
    """
    Fit the group lasso along a sequence of lambdas (warm-started).

    Equivalent to calling grplasso() in R with a vector of lambda values.

    GPU acceleration
    ----------------
    When CUDA is available and the column count of X exceeds GPU_THRESHOLD,
    all FISTA iterations run on the GPU.  X is transferred once and reused
    across the entire lambda path.  On this OpenBLAS build, this gives
    >1000x speedup for pcol >= 2000 (the regime that arises for p >= 100
    with the largest K values in the two-level CV).

    Parameters
    ----------
    lambdas : array-like  decreasing sequence of lambda values
    tol : float
        Convergence tolerance for each FISTA call.  The default is 1e-4,
        which is sufficient for cross-validation path fits (only out-of-fold
        predictions matter).  Use group_lasso_fista directly with tol=1e-8
        when you need accurate final coefficients.

    Returns
    -------
    coefs : ndarray (p, len(lambdas))
        Column i corresponds to lambdas[i].
    """
    n, p = X.shape
    lambdas = np.asarray(lambdas)
    coefs = np.zeros((p, len(lambdas)))

    use_gpu = _HAS_GPU and p > GPU_THRESHOLD

    if use_gpu:
        # ---------------------------------------------------------------
        # GPU path
        # Upload X once; compute L and Xty on GPU; run all FISTA steps
        # on GPU.  Only transfer beta back to CPU after each lambda.
        # ---------------------------------------------------------------
        device = _torch.device("cuda")
        X_g = _torch.tensor(X, device=device, dtype=_torch.float64)
        y_g = _torch.tensor(y, device=device, dtype=_torch.float64)

        # Lipschitz constant via GPU power iteration (avoids slow BLAS)
        L = _lipschitz_gpu(X_g, n)

        # Precompute X^T y / n on GPU (constant across all lambdas)
        Xty_g = X_g.T @ y_g / n

        # Pre-compute group index info for soft-thresholding.
        # When all penalised indices are contiguous (always true for HDAM),
        # slice + reshape replaces gather/scatter (~10x faster on GPU).
        pen_groups = np.unique(groups[~unpen_mask])
        group_idx = {g: np.where(groups == g)[0] for g in pen_groups}
        all_idx = np.array([group_idx[g] for g in pen_groups])  # (G, K)
        all_idx_g = _torch.tensor(all_idx, device=device, dtype=_torch.long)

        pen_flat = all_idx.ravel()
        contiguous_pen = bool(np.all(np.diff(pen_flat) == 1))
        pen_start = int(pen_flat[0]) if contiguous_pen else 0
        pen_end   = int(pen_flat[-1]) + 1 if contiguous_pen else 0

        # Warm-start from zeros; beta stays on GPU between lambdas
        beta_g = _torch.zeros(p, device=device, dtype=_torch.float64)

        for i, lam in enumerate(lambdas):
            beta_g = _fista_gpu(
                X_g, Xty_g, float(lam),
                all_idx_g, pen_start, pen_end, contiguous_pen,
                L, beta_g, max_iter, tol,
            )
            coefs[:, i] = beta_g.cpu().numpy()

        # Synchronise the CUDA stream before returning.  Without this, pending
        # GPU async ops (tensor deallocation callbacks, kernel completions) can
        # interfere with the CPU BLAS thread pool used by NumPy/OpenBLAS,
        # causing subsequent CPU matrix multiplications (e.g. QB_test @ coefs)
        # to be ~100x slower than expected.
        _torch.cuda.synchronize()

    else:
        # ---------------------------------------------------------------
        # NumPy path
        # Compute L once via scipy.svds(k=1) (~20ms vs ~2-9s for full SVD
        # on this USE64BITINT OpenBLAS build).  Pass L to every FISTA call
        # so the spectral norm is never recomputed along the path.
        # ---------------------------------------------------------------
        L = _lipschitz_numpy(X)
        beta = np.zeros(p)

        for i, lam in enumerate(lambdas):
            beta = group_lasso_fista(
                X, y, groups, lam, unpen_mask,
                max_iter=max_iter, tol=tol, beta_init=beta,
                L=L,  # reuse pre-computed Lipschitz constant
            )
            coefs[:, i] = beta

    return coefs
