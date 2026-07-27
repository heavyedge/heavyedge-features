"""Information projection to allowed classes."""

import numpy as np

__all__ = [
    "signed_iproj",
]


def signed_iproj(p, target_indices):
    """Signed information projection distance to allowed classes.

    Parameters
    ----------
    p : array-like, shape (K,)
        Probability distribution over K classes.
    target_indices : list of int
        List of target class indices to project onto.

    Returns
    -------
    signed_distance : float
        Signed information projection distance to the allowed classes.
    projected_distribution : array-like, shape (K,)
        Probability distribution after projection.

    Examples
    --------
    >>> import numpy as np
    >>> from heavyedge_features.iproj import signed_iproj
    >>> p = np.array([0.1, 0.7, 0.2])
    >>> target_indices = [0, 2]
    >>> dist, q = signed_iproj(p, target_indices)
    >>> dist
     np.float64(0.164...)
    >>> q
    array([0.117..., 0.441..., 0.441...])
    """
    p = np.maximum(p, 1e-12)  # Avoid log(0)
    p = p / p.sum()

    dists = []
    qs = []
    for target_index in target_indices:
        dist, q = _class_dist(p, target_index)
        dists.append(dist)
        qs.append(q)
    idx = np.argmin(dists)
    min_dist = dists[idx]
    mindist_q = qs[idx]

    if np.argmax(p) in target_indices:
        sign = -1  # signed distance
    else:
        sign = 1
    signed_distance, projected_distribution = sign * min_dist, mindist_q
    return signed_distance, projected_distribution


def _class_dist(p, i):
    K = len(p)

    dists = []
    qs = []
    for j in range(K):
        if j == i:
            continue
        dist, q = _class_dist_ij(p, i, j)
        dists.append(dist)
        qs.append(q)
    idx = np.argmin(dists)
    return dists[idx], qs[idx]


def _class_dist_ij(p, i, j):
    """Project ``p`` onto ``q[i] == q[j] >= q[k]`` for every other ``k``.

    The KL projection preserves the relative probabilities of unconstrained
    classes.  Classes whose probabilities would exceed the tied maximum are
    pooled with ``i`` and ``j``.  Their common unnormalised probability is the
    geometric mean of the probabilities in the pool.
    """
    log_p = np.log(p)
    pooled = [i, j]

    # Add possible constraint violators from largest to smallest.  Adding them
    # all at once is incorrect: the geometric mean can rise above a smaller
    # candidate after a larger candidate has joined the pool.
    candidates = sorted(
        (k for k in range(len(p)) if k != i and k != j),
        key=log_p.__getitem__,
        reverse=True,
    )
    log_geometric_mean = np.mean(log_p[pooled])
    for k in candidates:
        if log_p[k] <= log_geometric_mean:
            break
        pooled.append(k)
        log_geometric_mean = np.mean(log_p[pooled])

    # Before normalisation, pooled classes have the same geometric-mean
    # probability and all other classes retain their original probabilities.
    log_q = log_p.copy()
    log_q[pooled] = log_geometric_mean
    log_normalizer = np.logaddexp.reduce(log_q)
    log_q -= log_normalizer
    q = np.exp(log_q)

    # The log-ratios inside the pool cancel, so KL(q || p) = -log(Z).
    distance = np.maximum(0.0, -log_normalizer)
    return distance, q
