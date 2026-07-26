"""High-level Python runtime interface."""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from ..edge_width import width_type0, width_type1, width_type2, width_type3
from ..iproj import signed_iproj

__all__ = [
    "global_deviation",
    "edge_height",
    "edge_width",
]


def _compute_global_deviation(p, target_indices):
    value, _ = signed_iproj(p, target_indices)
    return value


def _compute_edge_width(
    x,
    Y,
    L,
    label,
    wet_thickness,
    sigma,
    type1_indices,
    type2_indices,
    type3_indices,
):
    if label in type1_indices:
        return width_type1(x, Y, L, wet_thickness)
    if label in type2_indices:
        return width_type2(x, Y, L, sigma)
    if label in type3_indices:
        return width_type3(x, Y, L, sigma)
    return width_type0(x, Y, L, wet_thickness)


def _log_progress(logger, completed, total, log_every):
    if completed == total or completed % log_every == 0:
        logger(f"{completed}/{total}")


def _resolve_max_workers(n_jobs, total):
    if not isinstance(n_jobs, (int, np.integer)) or isinstance(n_jobs, bool):
        raise TypeError("n_jobs must be an integer.")
    if n_jobs == 0 or n_jobs < -1:
        raise ValueError("n_jobs must be -1 or a positive integer.")

    requested = os.cpu_count() if n_jobs == -1 else int(n_jobs)
    return min(requested or 1, max(total, 1))


def global_deviation(
    soft_labels,
    target_indices,
    n_jobs=1,
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
    n_jobs : int, optional (default=1)
        Number of worker processes. ``1`` disables parallelization and ``-1``
        uses all available CPUs.
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
    max_workers = _resolve_max_workers(n_jobs, N)
    if N == 0:
        return np.empty(0, dtype=float)

    log_every = max(1, (N + 99) // 100)
    if max_workers == 1:
        values = []
        for completed, p in enumerate(soft_labels, start=1):
            values.append(_compute_global_deviation(p, target_indices))
            _log_progress(logger, completed, N, log_every)
        return np.asarray(values)

    values = np.empty(N, dtype=float)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_compute_global_deviation, p, target_indices): index
            for index, p in enumerate(soft_labels)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            values[futures[future]] = future.result()
            _log_progress(logger, completed, N, log_every)
    return values


def edge_height(profiles, logger=lambda x: None):
    """Dimensionless edge height of edge profiles.

    Parameters
    ----------
    profiles : heavyedge.ProfileData
        Open h5 file of profiles.
    logger : callable, optional
        Logger function which accepts a progress message string.

    Returns
    -------
    heights : np.ndarray
        Array containing edge height values for each profile.

    Examples
    --------
    >>> from heavyedge import ProfileData
    >>> from heavyedge_features.samples import get_sample_path as features_sample
    >>> from heavyedge_features.api import edge_height
    >>> edge_height(ProfileData(features_sample("Profiles.h5"))).shape
    (75,)
    """
    N, _ = profiles.shape()
    ret = []
    for i, (Y, L, _) in enumerate(profiles):
        ret.append(Y[:L].max() / Y[0])
        logger(f"edge height ({i + 1}/{N})")
    return np.array(ret)


def edge_width(
    profiles,
    hard_labels,
    wet_thicknesses,
    sigma,
    type1_indices,
    type2_indices,
    type3_indices,
    n_jobs=1,
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
    n_jobs : int, optional (default=1)
        Number of worker processes. ``1`` disables parallelization and ``-1``
        uses all available CPUs.
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
    max_workers = _resolve_max_workers(n_jobs, N)
    if len(hard_labels) != N or len(wet_thicknesses) != N:
        raise ValueError(
            "profiles, hard_labels, and wet_thicknesses must have equal lengths."
        )
    if N == 0:
        return np.empty(0, dtype=float)

    log_every = max(1, (N + 99) // 100)
    tasks = (
        (
            x,
            Y,
            L,
            label,
            wet_thickness,
            sigma,
            type1_indices,
            type2_indices,
            type3_indices,
        )
        for (Y, L, _), label, wet_thickness in zip(
            profiles, hard_labels, wet_thicknesses
        )
    )

    if max_workers == 1:
        values = []
        for completed, task in enumerate(tasks, start=1):
            values.append(_compute_edge_width(*task))
            _log_progress(logger, completed, N, log_every)
        return np.asarray(values)

    values = np.empty(N, dtype=float)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_compute_edge_width, *task): index
            for index, task in enumerate(tasks)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            values[futures[future]] = future.result()
            _log_progress(logger, completed, N, log_every)
    return values
