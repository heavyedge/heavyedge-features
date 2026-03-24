"""High-level Python runtime interface."""

import numpy as np

from ..edge_width import width_type0, width_type1, width_type2, width_type3
from ..iproj import signed_iproj

__all__ = [
    "global_deviation",
    "edge_width",
]


def global_deviation(
    soft_labels,
    target_indices,
    logger=lambda x: None,
):
    """Compute global shape deviations using probabilistic classification labels.

    Negative values indicaete profiles are within the desired classes.
    Larger values mean more deviation from the desired classes.

    Parameters
    ----------
    soft_labels : np.ndarray
        Probabilistic classification labels for the profiles.
    target_indices : list of int
        Indices of target classes to compute values for.
    logger : callable, optional
        Logger function which accepts a progress message string.

    Returns
    -------
    values : np.ndarray
        Array containing global shape deviations for each profile.

    Examples
    --------
    >>> import numpy as np
    >>> from heavyedge_classify.samples import get_sample_path
    >>> from heavyedge_features.api import global_deviation
    >>> soft_labels = np.load(get_sample_path("labels-pred.npy"))
    >>> global_deviation(soft_labels, [0]).shape
    (75,)
    """
    N, _ = soft_labels.shape
    values = []
    for i, p in enumerate(soft_labels):
        value, _ = signed_iproj(p, target_indices)
        values.append(value)
        logger(f"{i + 1}/{N}")
    return np.array(values)


def edge_width(
    profiles,
    hard_labels,
    wet_thicknesses,
    sigma,
    type1_indices,
    type2_indices,
    type3_indices,
    logger=lambda x: None,
):
    """Detect edge with of profiles using profile data and classification labels.

    Parameters
    ----------
    profiles : heavyedge.ProfileData
        Open h5 file of profiles.
    hard_labels : np.ndarray
        Hard classification labels for the profiles.
    wet_thicknesses : np.ndarray
        Wet thickness values for the profiles.
    sigma : scalar
        Standard deviation of Gaussian filter for smoothing.
        Using the same value as the one used for preprocessing is recommended.
    type1_indices, type2_indices, type3_indices : list of int
        Lists of indices of Type 1, 2, and 3 classes from trained labels, respectively.
    logger : callable, optional
        Logger function which accepts a progress message string.

    Returns
    -------
    widths : np.ndarray
        Array containing edge width values for each profile.

    Examples
    --------
    >>> from heavyedge import ProfileData
    >>> from heavyedge_classify.samples import get_sample_path as classify_sample
    >>> from heavyedge_features.samples import get_sample_path as features_sample
    >>> from heavyedge_features.api import edge_width
    >>> import numpy as np
    >>> profiles = ProfileData(features_sample("Profiles.h5"))
    >>> hard_labels = np.load(classify_sample("labels-pred.npy")).argmax(axis=1)
    >>> wet_thicknesses = np.full(hard_labels.shape, 0.25)
    >>> sigma = 32
    >>> edge_width(profiles, hard_labels, wet_thicknesses, sigma, [0], [1], [2]).shape
    (75,)
    """
    x = profiles.x()
    N, _ = profiles.shape()
    ret = []
    for i, ((Y, L, _), label, wt) in enumerate(
        zip(profiles, hard_labels, wet_thicknesses)
    ):
        if label in type1_indices:
            width = width_type1(x, Y, L, wt)
        elif label in type2_indices:
            width = width_type2(x, Y, L, sigma)
        elif label in type3_indices:
            width = width_type3(x, Y, L, sigma)
        else:
            width = width_type0(x, Y, L, wt)
        logger(f"{i + 1}/{N}")
        ret.append(width)
    return np.array(ret)
