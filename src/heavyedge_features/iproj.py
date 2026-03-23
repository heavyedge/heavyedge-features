"""Information projection to allowed classes."""

import cvxpy as cp
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
    K = len(p)
    q = cp.Variable(K, nonneg=True)
    constraints = [cp.sum(q) == 1]

    # Constraints: q[i] == q[j], q[i] >= q[other]
    for k in range(K):
        if k == i:
            continue
        elif k == j:
            constraints.append(q[i] == q[k])
        else:
            constraints.append(q[i] >= q[k])
    objective = cp.Minimize(cp.sum(cp.kl_div(q, p)))
    return cp.Problem(objective, constraints).solve(), q.value
