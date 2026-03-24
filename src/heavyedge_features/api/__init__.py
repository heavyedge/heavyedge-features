"""High-level Python runtime interface."""

import numpy as np

from ..iproj import signed_iproj

__all__ = [
    "global_deviation",
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
