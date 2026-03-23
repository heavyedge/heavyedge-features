"""High-level Python runtime interface."""

import numpy as np
from heavyedge_classify.api import classify_predict

from ..iproj import signed_iproj

__all__ = [
    "global_deviation",
]


def global_deviation(
    profiles,
    target_indices,
    model,
    normalize=True,
    batch_size=None,
    logger=lambda x: None,
):
    """Compute global shape deviations for profiles using a classification model.

    Negative values indicaete profiles are within the desired classes.
    Larger values mean more deviation from the desired classes.

    Parameters
    ----------
    profiles : heavyedge.ProfileData
        Open h5 file of profiles.
    target_indices : list of int
        Indices of target classes to compute values for.
    model
        Trained classification model object.
    normalize : bool, default=True
        Whether to normalize profiles by area under curve.
        Set this to False if *profiles* are already normalized.
    batch_size : int, optional
        Batch size to load data.
        If not passed, all data are loaded at once.
    logger : callable, optional
        Logger function which accepts a progress message string.

    Yields
    ------
    values : np.ndarray
        Array of shape (batch_size,) containing global shape deviations
        for each profile.

    Examples
    --------
    >>> import pickle
    >>> from heavyedge import ProfileData
    >>> from heavyedge_classify.samples import get_sample_path
    >>> from heavyedge_features.api import global_deviation
    >>> with open(get_sample_path("model.pkl"), "rb") as f:
    ...     model = pickle.load(f)
    >>> profiles = ProfileData(get_sample_path("Profiles.h5"))
    >>> [v.shape for v in global_deviation(profiles, [1], model, batch_size=50)]
    [(50,), (25,)]
    """
    N, _ = profiles.shape()
    prob_gen = classify_predict(
        model, profiles, normalize=normalize, batch_size=batch_size
    )
    count = 0
    for probabilities in prob_gen:
        values = []
        for p in probabilities:
            value, _ = signed_iproj(p, target_indices)
            values.append(value)
        yield np.array(values)
        count += len(probabilities)
        logger(f"{count}/{N}")
