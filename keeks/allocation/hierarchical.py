"""
Hierarchical risk parity - the covariance-only family that never inverts.

:class:`HierarchicalRiskParity` builds a portfolio from a covariance matrix
alone (Lopez de Prado, 2016): options are clustered by correlation distance,
the cluster tree is read as a leaf permutation that concentrates similar
options together (quasi-diagonalization), and a top-down bisection allocates
each cluster's mass between its two halves in inverse-variance proportion.
The correlation structure only decides which options share a split; the
allocation itself is inverse-variance, so the matrix is never inverted and no
expected returns are needed.

Everything is hand-rolled and deterministic by construction: the linkage is a
single-linkage agglomeration whose ties break to the lexicographically
smallest cluster-id pair, the quasi-diagonalization expands cluster ids in a
fixed left-to-right order, and the bisection splits odd clusters with the
first half taking the floor. No solver, no randomness, no scipy - repeated
constructions from the same covariance replay bit-identically.
"""

import numpy as np

from keeks.allocation.base import (
    AllocationResult,
    BaseAllocationStrategy,
    _validate_covariance,
    _validate_weights,
)

__author__ = "willmcginnis"


def _correlation_distance(covariance: np.ndarray) -> np.ndarray:
    """
    Correlation-distance matrix: ``d_ij = sqrt((1 - rho_ij) / 2)``.

    The correlation is rescaled from the covariance through the diagonal
    volatilities, clipped to ``[-1, 1]`` (floating-point noise can push a
    perfect correlation one ulp past its bound), and mapped so that perfect
    correlation is distance zero and perfect anticorrelation is distance one.

    Parameters
    ----------
    covariance : numpy.ndarray
        A covariance matrix with strictly positive diagonal variances.

    Returns
    -------
    numpy.ndarray
        The symmetric distance matrix, zeros on the diagonal.

    Examples
    --------
    >>> covariance = np.array([[4.0, 2.0], [2.0, 4.0]])
    >>> _correlation_distance(covariance)
    array([[0. , 0.5],
           [0.5, 0. ]])
    """
    std = np.sqrt(np.diag(covariance))
    correlation = np.clip(covariance / np.outer(std, std), -1.0, 1.0)
    distance = np.sqrt((1.0 - correlation) / 2.0)
    np.fill_diagonal(distance, 0.0)
    return distance


def _agglomerative_linkage(distance: np.ndarray) -> np.ndarray:
    """
    Single-linkage agglomerative clustering in the scipy linkage layout.

    Starts from one cluster per option and repeatedly merges the two active
    clusters with the smallest linkage distance - for single linkage, the
    smallest pairwise option distance across the two clusters' members -
    until one cluster holds every option. Returns an ``(n - 1, 4)`` matrix,
    one row per merge: ``[first_id, second_id, distance, size]``. Leaves
    carry their option index as id; the merge recorded on row ``k`` forms
    cluster ``n + k``.

    Deterministic by construction: candidate pairs iterate in cluster-id
    order and only a strictly smaller distance replaces the incumbent, so
    equal distances break to the lexicographically smallest id pair.

    Parameters
    ----------
    distance : numpy.ndarray
        Symmetric distance matrix, zeros on the diagonal.

    Returns
    -------
    numpy.ndarray
        The ``(n - 1, 4)`` linkage matrix.

    Examples
    --------
    >>> distance = np.array([[0.0, 0.5, 1.0], [0.5, 0.0, 1.0], [1.0, 1.0, 0.0]])
    >>> _agglomerative_linkage(distance).tolist()
    [[0.0, 1.0, 0.5, 2.0], [2.0, 3.0, 1.0, 3.0]]

    An all-tie distance matrix still clusters deterministically - every
    candidate distance is equal, so the lexicographically smallest pair
    merges first:

    >>> distance = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    >>> _agglomerative_linkage(distance).tolist()
    [[0.0, 1.0, 0.5, 2.0], [2.0, 3.0, 0.5, 3.0]]
    """
    option_count = distance.shape[0]
    pair_distances = {}
    for i in range(option_count):
        for j in range(i + 1, option_count):
            pair_distances[(i, j)] = float(distance[i, j])
    sizes = dict.fromkeys(range(option_count), 1)
    active = list(range(option_count))
    linkage = []
    for cluster_id in range(option_count, 2 * option_count - 1):
        first, second = min(
            pair_distances, key=lambda pair: (pair_distances[pair], pair)
        )
        linkage.append(
            [
                first,
                second,
                pair_distances[(first, second)],
                sizes[first] + sizes[second],
            ]
        )
        del pair_distances[(first, second)]
        # Single linkage: the merged cluster's distance to another cluster is
        # the smaller of its parents' distances (Lance-Williams update).
        merged = {}
        for other in active:
            if other in (first, second):
                continue
            to_first = pair_distances.pop((min(first, other), max(first, other)))
            to_second = pair_distances.pop((min(second, other), max(second, other)))
            merged[(other, cluster_id)] = min(to_first, to_second)
        pair_distances.update(merged)
        active = [other for other in active if other not in (first, second)]
        active.append(cluster_id)
        sizes[cluster_id] = sizes[first] + sizes[second]
        del sizes[first]
        del sizes[second]
    return np.asarray(linkage, dtype=float).reshape(len(linkage), 4)


def _quasi_diagonal(linkage: np.ndarray) -> np.ndarray:
    """
    Leaf permutation that concentrates correlated options together.

    Reads the linkage tree left-to-right: the root's two children, then each
    child expanded the same way, until only leaves remain. The result is a
    permutation of the option indices - Lopez de Prado's quasi-diagonalization,
    which reorders the options so similar ones sit together and the bisection
    splits happen inside correlated groups.

    Parameters
    ----------
    linkage : numpy.ndarray
        The ``(n - 1, 4)`` linkage matrix from :func:`_agglomerative_linkage`.

    Returns
    -------
    list of int
        One option index per position, in quasi-diagonal order.

    Examples
    --------
    >>> distance = np.array([[0.0, 0.5, 1.0], [0.5, 0.0, 1.0], [1.0, 1.0, 0.0]])
    >>> _quasi_diagonal(_agglomerative_linkage(distance))
    [2, 0, 1]

    A single option has nothing to reorder:

    >>> _quasi_diagonal(np.empty((0, 4)))
    [0]
    """
    option_count = linkage.shape[0] + 1
    if option_count == 1:
        return [0]
    order = [int(linkage[-1, 0]), int(linkage[-1, 1])]
    while any(item >= option_count for item in order):
        expanded = []
        for item in order:
            if item >= option_count:
                row = linkage[item - option_count]
                expanded.append(int(row[0]))
                expanded.append(int(row[1]))
            else:
                expanded.append(item)
        order = expanded
    return order


def _cluster_variance(covariance: np.ndarray, indices: np.ndarray) -> float:
    """
    Inverse-variance-weighted variance of one cluster.

    The cluster's slice of the covariance is weighted by the inverse of its
    diagonal variances (normalized to sum to one) and read as a quadratic
    form - the risk the bisection balances.

    Parameters
    ----------
    covariance : numpy.ndarray
        The full covariance matrix.
    indices : list of int
        The option indices of the cluster, in quasi-diagonal order.

    Returns
    -------
    float
        The cluster's inverse-variance-weighted variance.

    Examples
    --------
    >>> covariance = np.array([[4.0, 0.0], [0.0, 1.0]])
    >>> round(_cluster_variance(covariance, [0, 1]), 12)
    0.8
    """
    slice_ = covariance[np.ix_(indices, indices)]
    weights = 1.0 / np.diag(slice_)
    weights = weights / weights.sum()
    return float(weights @ slice_ @ weights)


def _recursive_bisection(covariance: np.ndarray, order: np.ndarray) -> np.ndarray:
    """
    Top-down allocation over the quasi-diagonal order.

    Splits every cluster at the midpoint of its quasi-diagonal slice (odd
    clusters give the first half the floor), computes each half's
    inverse-variance cluster variance, and scales the two halves' weights so
    the split is inverse-variance proportional: the less risky half keeps the
    larger share. Singletons are leaves and keep their accumulated share, so
    the weights always sum to one.

    Parameters
    ----------
    covariance : numpy.ndarray
        The full covariance matrix.
    order : list of int
        The quasi-diagonal permutation from :func:`_quasi_diagonal`.

    Returns
    -------
    numpy.ndarray
        One weight per option, in option order, all positive and summing to
        one within floating-point tolerance.

    Examples
    --------
    >>> covariance = np.array([[0.125, 0.0], [0.0, 0.375]])
    >>> _recursive_bisection(covariance, [0, 1]).tolist()
    [0.75, 0.25]
    """
    option_count = covariance.shape[0]
    weights = np.ones(option_count)
    clusters = [list(order)]
    while clusters:
        next_clusters = []
        for cluster in clusters:
            if len(cluster) > 1:
                half = len(cluster) // 2
                first, second = cluster[:half], cluster[half:]
                first_variance = _cluster_variance(covariance, first)
                second_variance = _cluster_variance(covariance, second)
                total = first_variance + second_variance
                # Both halves carrying zero cluster variance leaves no risk
                # information to balance, so split evenly.
                alpha = 1.0 - first_variance / total if total > 0.0 else 0.5
                weights[first] *= alpha
                weights[second] *= 1.0 - alpha
                next_clusters.append(first)
                next_clusters.append(second)
        clusters = next_clusters
    return weights


class HierarchicalRiskParity(BaseAllocationStrategy):
    """
    Hierarchical risk parity over a covariance matrix.

    Sizes the portfolio from the covariance alone (Lopez de Prado, 2016):
    correlation distance clusters the options, the cluster tree is read as a
    quasi-diagonal leaf permutation, and a recursive bisection allocates each
    split in inverse-variance proportion. No expected returns, no solver, no
    matrix inversion - and no randomness, so repeated constructions from the
    same covariance replay bit-identically.

    The derivation is public and inspectable: ``distance`` (the
    correlation-distance matrix), ``linkage`` (the agglomeration), and
    ``order`` (the quasi-diagonal permutation) are exposed attributes.

    Weights are long-only and sum to one - the hierarchy fully invests the
    budget, leaving no cash residual - and are scale-free: all zeros for a
    nonpositive bankroll.

    Parameters
    ----------
    covariance : array-like
        The covariance matrix of the options' simple returns. Validated like
        every allocation descriptor (square, finite, symmetric, positive
        semidefinite), and additionally required to have strictly positive
        variances: the bisection's inverse-variance splits are undefined for
        a zero-variance option.

    Examples
    --------
    >>> covariance = np.array([[0.04, 0.004], [0.004, 0.04]])
    >>> strategy = HierarchicalRiskParity(covariance)
    >>> strategy.evaluate(1000.0)
    (0.5, 0.5)
    >>> strategy.evaluate(0.0)
    (0.0, 0.0)

    Two equally risky options split the budget evenly regardless of their
    correlation; an asymmetric covariance splits by inverse variance:

    >>> risky = np.array([[0.04, 0.002], [0.002, 0.01]])
    >>> result = HierarchicalRiskParity(risky).optimize()
    >>> [round(weight, 4) for weight in result.weights.tolist()]
    [0.2, 0.8]
    """

    def __init__(self, covariance: np.typing.ArrayLike) -> None:
        covariance = _validate_covariance(covariance)
        if np.any(np.diag(covariance) <= 0):
            raise ValueError(
                "Covariance must have strictly positive variances: the "
                "inverse-variance splits of hierarchical risk parity are "
                "undefined for a zero-variance option"
            )
        self.covariance = covariance
        self.distance = _correlation_distance(covariance)
        self.linkage = _agglomerative_linkage(self.distance)
        self.order = _quasi_diagonal(self.linkage)
        self.weights = _recursive_bisection(covariance, self.order)

    def optimize(self) -> AllocationResult:
        """
        Return the allocation with its diagnostics.

        Hierarchical risk parity runs no solver, so ``objective``,
        ``converged``, ``iterations``, and ``expected_growth`` stay ``None``;
        ``volatility`` is ``sqrt(w' Sigma w)`` under the construction
        covariance.

        Returns
        -------
        AllocationResult
            The weight vector and the portfolio volatility it implies.

        Examples
        --------
        >>> covariance = np.array([[0.04, 0.004], [0.004, 0.04]])
        >>> result = HierarchicalRiskParity(covariance).optimize()
        >>> result.weights.tolist()
        [0.5, 0.5]
        >>> round(result.volatility, 4)
        0.1483
        """
        volatility = float(np.sqrt(self.weights @ self.covariance @ self.weights))
        return AllocationResult(weights=self.weights.copy(), volatility=volatility)

    def evaluate(self, current_bankroll: float) -> tuple[float, ...]:
        """
        Return one long-only weight per option.

        Parameters
        ----------
        current_bankroll : float
            The current bankroll. Weights are scale-free fractions: the same
            vector at every positive bankroll, all zeros at zero or below.

        Returns
        -------
        tuple of float
            The validated weight tuple, one weight per option.

        Examples
        --------
        >>> covariance = np.array([[0.04, 0.004], [0.004, 0.04]])
        >>> HierarchicalRiskParity(covariance).evaluate(250.0)
        (0.5, 0.5)
        """
        option_count = self.covariance.shape[0]
        if current_bankroll <= 0:
            return _validate_weights([0.0] * option_count, option_count=option_count)
        return _validate_weights(self.weights, option_count=option_count)
