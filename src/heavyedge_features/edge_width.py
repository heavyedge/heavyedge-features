"""Edge width by edge type."""

import numpy as np
from heavyedge_landmarks import (
    landmarks_type2,
    landmarks_type3,
    plateau_type2,
    plateau_type3,
)

__all__ = [
    "width_type0",
    "width_type1",
    "width_type2",
    "width_type3",
]


def width_type0(x, Y, L, wt):
    """Edge width for type 0 profiles.

    Parameters
    ----------
    x : array of shape (M,)
        X grid of profiles.
    Y : array of shape (M,)
        Height data of profile.
    L : int
        Length of profile before contact point in number of points.
    wt : scalar
        Wet thickness of the profile.

    Returns
    -------
    scalar
        Edge width of the profile.
    """
    (idxs,) = np.where(Y <= wt)
    if len(idxs) == 0:
        edge_idx = 0
    else:
        edge_idx = idxs[0]
    return x[L - 1] - x[edge_idx]


def width_type1(x, Y, L, wt):
    """Edge width for type 1 profiles.

    Parameters
    ----------
    x : array of shape (M,)
        X grid of profiles.
    Y : array of shape (M,)
        Height data of profile.
    L : int
        Length of profile before contact point in number of points.
    wt : scalar
        Wet thickness of the profile.

    Returns
    -------
    scalar
        Edge width of the profile.

    Examples
    --------
    >>> from heavyedge import get_sample_path, ProfileData
    >>> from heavyedge_features.edge_width import width_type1
    >>> with ProfileData(get_sample_path("Prep-Type1.h5")) as data:
    ...     x = data.x()
    ...     Y, L, _ = data[0]
    >>> wt = 0.25
    >>> b = width_type1(x, Y, L, wt)
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... plt.plot(x, Y, color="gray", alpha=0.5)
    ... plt.axhline(wt, ls="--", label="Wet thickness")
    ... plt.axvline(x[L - 1] - b, color="red", ls="--", label="Edge boundary")
    ... plt.legend()
    """
    (idxs,) = np.where(Y >= wt)
    if len(idxs) == 0:
        edge_idx = 0
    else:
        edge_idx = idxs[-1]
    return x[L - 1] - x[edge_idx]


def width_type2(x, Y, L, sigma):
    """Edge width for type 2 profiles.

    Parameters
    ----------
    x : array of shape (M,)
        X grid of profiles.
    Y : array of shape (M,)
        Height data of profile.
    L : int
        Length of profile before contact point in number of points.
    sigma : scalar
        Standard deviation of Gaussian filter for smoothing.

    Returns
    -------
    scalar
        Edge width of the profile.

    Examples
    --------
    >>> from heavyedge import get_sample_path, ProfileData
    >>> from heavyedge_features.edge_width import width_type2
    >>> with ProfileData(get_sample_path("Prep-Type2.h5")) as data:
    ...     x = data.x()
    ...     Y, L, _ = data[0]
    >>> sigma = 32.0
    >>> b = width_type2(x, Y, L, sigma)
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... plt.plot(x, Y, color="gray", alpha=0.5)
    ... plt.axvline(x[L - 1] - b, color="red", ls="--", label="Edge boundary")
    ... plt.legend()
    """
    lm = landmarks_type2(x, [Y], [L], sigma)
    peaks, knees = lm[:, 0, 1:].T
    plateau = plateau_type2(x, [Y], peaks, knees)
    return x[L - 1] - plateau[0, -1]


def width_type3(x, Y, L, sigma):
    """Edge width for type 3 profiles.

    Parameters
    ----------
    x : array of shape (M,)
        X grid of profiles.
    Y : array of shape (M,)
        Height data of profile.
    L : int
        Length of profile before contact point in number of points.
    sigma : scalar
        Standard deviation of Gaussian filter for smoothing.

    Returns
    -------
    scalar
        Edge width of the profile.

    Examples
    --------
    >>> from heavyedge import get_sample_path, ProfileData
    >>> from heavyedge_features.edge_width import width_type3
    >>> with ProfileData(get_sample_path("Prep-Type3.h5")) as data:
    ...     x = data.x()
    ...     Y, L, _ = data[0]
    >>> sigma = 32.0
    >>> b = width_type3(x, Y, L, sigma)
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... plt.plot(x, Y, color="gray", alpha=0.5)
    ... plt.axvline(x[L - 1] - b, color="red", ls="--", label="Edge boundary")
    ... plt.legend()
    """
    lm = landmarks_type3(x, [Y], [L], sigma)
    troughs, knees = lm[:, 0, 2:].T
    plateau = plateau_type3(x, [Y], troughs, knees)
    return x[L - 1] - plateau[0, -1]
