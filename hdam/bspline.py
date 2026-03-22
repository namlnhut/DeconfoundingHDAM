"""
B-spline basis matrix construction.

Equivalent to fda::bsplineS() in R:
  bsplineS(x, breaks, norder=4)

R's fda uses `breaks` as the full breakpoint sequence (boundary + interior knots).
Number of basis functions = len(breaks) + norder - 2.

For cubic B-splines (norder=4, degree=3), the extended knot vector repeats
the boundary knots 4 times.
"""

import numpy as np
from scipy.interpolate import BSpline


def bspline_basis(x, breaks, order: int = 4) -> np.ndarray:
    """
    Compute a B-spline design matrix.

    Equivalent to R's fda::bsplineS(x, breaks, norder=order).

    Parameters
    ----------
    x : array-like, shape (n,)
        Evaluation points.
    breaks : array-like
        Breakpoints including both boundary knots (min and max).
    order : int, default 4
        B-spline order (4 = cubic). degree = order - 1.

    Returns
    -------
    B : ndarray, shape (n, nbasis)
        Design matrix.  nbasis = len(breaks) + order - 2.
    """
    x = np.asarray(x, dtype=float).ravel()
    breaks = np.asarray(breaks, dtype=float)
    degree = order - 1

    # Extended knot vector: repeat boundary knots `order` times
    t = np.concatenate([
        np.repeat(breaks[0], order),
        breaks[1:-1],
        np.repeat(breaks[-1], order),
    ])

    # Clip x to the support to avoid NaN at the boundaries
    x_clipped = np.clip(x, breaks[0], breaks[-1])

    B = BSpline.design_matrix(x_clipped, t, degree).toarray()
    return B
