"""Finds plateau region by segmented regression."""

import numpy as np

__all__ = [
    "plateau_type2",
    "plateau_type3",
]


def _ols(Xi, Y):
    params, _, _, _ = np.linalg.lstsq(Xi, Y, rcond=None)
    return params


def _segreg(x, Y, psi0, tol=1e-5, maxiter=30, max_backtracks=50):
    r"""Segmented regression with one breakpoint.

    Parameters
    ----------
    x, Y : (M,) ndarray
        Data points.
    psi0 : scalar
        Initial guess for breakpoint coordinate.
    tol : float, default=1e-5
        Convergence tolerance.
    maxiter : int, default=30
        Maximum number of segmented-regression iterations.
    max_backtracks : int, default=50
        Maximum number of step halvings in each iteration.

    Returns
    -------
    params : (4,) ndarray
        Estimated parameters: b0, b1, b2, psi.
    reached_max : bool
        Whether the iteration stopped without convergence because it reached an
        iteration limit or encountered a numerically degenerate fit.
    """
    x = np.asarray(x, dtype=float)
    Y = np.asarray(Y, dtype=float)
    psi = float(psi0)

    if x.ndim != 1 or Y.ndim != 1 or x.shape != Y.shape:
        raise ValueError("x and Y must be one-dimensional arrays of equal length.")
    if len(x) < 4:
        raise ValueError("Segmented regression requires at least four data points.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(Y)):
        raise ValueError("x and Y must contain only finite values.")
    if not np.all(np.diff(x) > 0):
        raise ValueError("x must be strictly increasing.")
    if not x[0] < psi < x[-1]:
        raise ValueError("psi0 must lie strictly inside the x domain.")
    if tol <= 0:
        raise ValueError("tol must be positive.")
    if maxiter < 1:
        raise ValueError("maxiter must be positive.")
    if max_backtracks < 1:
        raise ValueError("max_backtracks must be positive.")

    params = None
    for _ in range(maxiter):
        Xi = np.array(
            [
                np.ones_like(x),
                x,
                (x - psi) * np.heaviside(x - psi, 0),
                -np.heaviside(x - psi, 0),
            ]
        ).T

        b0, b1, b2, gamma = _ols(Xi, Y)
        params = np.array([b0, b1, b2, psi])
        RSS = np.sum((Y - _segreg_predict(x, b0, b1, b2, psi)) ** 2)

        if not np.all(np.isfinite([b0, b1, b2, gamma, RSS])):
            return params, True

        b2_scale = max(1.0, abs(b0), abs(b1), abs(gamma))
        if abs(b2) <= np.finfo(float).eps * b2_scale:
            return params, True

        full_step = gamma / b2
        if not np.isfinite(full_step):
            return params, True
        if abs(full_step) <= tol:
            return params, False

        accepted = False
        step_scale = 1.0
        for _ in range(max_backtracks):
            step = step_scale * full_step

            if abs(step) <= tol or psi + step == psi:
                return params, False

            psi_new = psi + step
            if x[0] < psi_new < x[-1]:
                RSS_new = np.sum((Y - _segreg_predict(x, b0, b1, b2, psi_new)) ** 2)
                if np.isfinite(RSS_new) and RSS_new < RSS:
                    accepted = True
                    break

            step_scale /= 2

        if not accepted:
            return params, True

        psi = psi_new

    Xi = np.array(
        [
            np.ones_like(x),
            x,
            (x - psi) * np.heaviside(x - psi, 0),
            -np.heaviside(x - psi, 0),
        ]
    ).T
    b0, b1, b2, _ = _ols(Xi, Y)
    return np.array([b0, b1, b2, psi]), True


def _segreg_predict(x, b0, b1, b2, psi):
    x = np.asarray(x)
    return b0 + b1 * x + b2 * (x - psi) * np.heaviside(x - psi, 0)


def plateau_type2(x, Ys, peaks, knees):
    """Find plateau for type 2 heavy edge profiles.

    Parameters
    ----------
    x : array of shape (M,)
        X grid of profiles.
    Ys : array of shape (N, M)
        Height data of N profiles.
    peaks, knees : arrays of shape (N,)
        X coordinates of peak point and knee point.

    Returns
    -------
    array of shape (N, 3)
        Plateau intercept, slope and boundary coordinates.

    Notes
    -----
    Plateau boundary is located by segmented regression.

    See Also
    --------
    landmarks_type2 : Detects *peaks* and *knees*.

    Examples
    --------
    >>> from heavyedge import get_sample_path, ProfileData
    >>> from heavyedge_features.landmarks import landmarks_type2
    >>> from heavyedge_features.plateau import plateau_type2
    >>> with ProfileData(get_sample_path("Prep-Type2.h5")) as data:
    ...     x = data.x()
    ...     Ys, Ls, _ = data[:]
    >>> lm = landmarks_type2(x, Ys, Ls, 32)
    >>> peaks, knees = lm[:, 0, 1:].T
    >>> plateau = plateau_type2(x, Ys, peaks, knees)
    >>> plateau.shape
    (22, 3)
    >>> plateau_x = np.stack([np.zeros(len(plateau)), plateau[:, 2]])
    >>> plateau_y = plateau[:, 0] + plateau_x * plateau[:, 1]
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... plt.plot(x, Ys.T, color="gray")
    ... plt.plot(plateau_x, plateau_y)
    """
    ret = []
    for Y, peak, knee in zip(Ys, peaks, knees):
        peak, knee = np.searchsorted(x, [peak, knee])
        (b0, b1, _, psi), _ = _segreg(x[:peak], Y[:peak], x[knee])
        ret.append([b0, b1, psi])
    return np.array(ret)


def plateau_type3(x, Ys, troughs, knees):
    """Find plateau for type 3 heavy edge profiles.

    Parameters
    ----------
    x : array of shape (M,)
        X grid of profiles.
    Ys : array of shape (N, M)
        Height data of N profiles.
    troughs, knees : arrays of shape (N,)
        X coordinates of trough point and knee point.

    Returns
    -------
    array of shape (N, 3)
        Plateau intercept, slope and boundary coordinates.

    Notes
    -----
    Plateau boundary is located by segmented regression.

    See Also
    --------
    landmarks_type3 : Detects *troughs* and *knees*.

    Examples
    --------
    >>> from heavyedge import get_sample_path, ProfileData
    >>> from heavyedge_features.landmarks import landmarks_type3
    >>> from heavyedge_features.plateau import plateau_type3
    >>> with ProfileData(get_sample_path("Prep-Type3.h5")) as data:
    ...     x = data.x()
    ...     Ys, Ls, _ = data[:]
    >>> lm = landmarks_type3(x, Ys, Ls, 32)
    >>> troughs, knees = lm[:, 0, 2:].T
    >>> plateau = plateau_type3(x, Ys, troughs, knees)
    >>> plateau.shape
    (35, 3)
    >>> plateau_x = np.stack([np.zeros(len(plateau)), plateau[:, 2]])
    >>> plateau_y = plateau[:, 0] + plateau_x * plateau[:, 1]
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... plt.plot(x, Ys.T, color="gray")
    ... plt.plot(plateau_x, plateau_y)
    """
    ret = []
    for Y, trough, knee in zip(Ys, troughs, knees):
        trough, knee = np.searchsorted(x, [trough, knee])
        (b0, b1, _, psi), _ = _segreg(x[:trough], Y[:trough], x[knee])
        ret.append([b0, b1, psi])
    return np.array(ret)
