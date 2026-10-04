"""Hierarchical risk parity: cluster recovery, determinism, and contract.

``HierarchicalRiskParity`` is the covariance-only family: correlation
distance, hand-rolled single-linkage clustering with documented tie-breaking,
quasi-diagonalization, and recursive inverse-variance bisection. These tests
pin the derivation end to end - planted clusters are recovered, repeated
constructions replay bit-identically, and the weight contract (long-only,
budget-capped, zeros for a nonpositive bankroll) holds on every input.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from keeks.allocation import AllocationResult
from keeks.allocation.hierarchical import (
    HierarchicalRiskParity,
    _agglomerative_linkage,
    _cluster_variance,
    _correlation_distance,
    _quasi_diagonal,
    _recursive_bisection,
)
from keeks.utils import PROBABILITY_SUM_TOLERANCE

settings.register_profile("keeks", max_examples=50, deadline=None)
settings.load_profile("keeks")


@st.composite
def covariance_matrices(draw):
    """Random positive-semidefinite covariance, strictly positive diagonal.

    Drawn as a random signed factor times its transpose: symmetry and positive
    semidefiniteness hold by construction, and every row of the factor carries
    at least one entry of magnitude 0.1, so every diagonal variance is
    positive and the inverse-variance arithmetic is defined.
    """
    option_count = draw(st.integers(min_value=1, max_value=6))
    factor = np.array(
        [
            [
                draw(st.floats(min_value=0.1, max_value=2.0, allow_nan=False))
                * draw(st.sampled_from([1.0, -1.0]))
                for _ in range(option_count)
            ]
            for _ in range(option_count)
        ]
    )
    return factor @ factor.T


def test_correlation_distance_maps_correlation_to_distance():
    """Perfect correlation is distance zero; independence is sqrt(1/2)."""
    covariance = np.array(
        [
            [0.04, 0.02, 0.0],
            [0.02, 0.04, 0.0],
            [0.0, 0.0, 0.04],
        ]
    )
    distance = _correlation_distance(covariance)
    assert distance[0, 1] == pytest.approx(0.5)
    assert distance[0, 2] == pytest.approx(np.sqrt(0.5))
    assert np.allclose(np.diag(distance), 0.0)


def test_linkage_merges_smallest_distance_first():
    """Distinct distances cluster in ascending order, scipy-linkage layout."""
    distance = np.array([[0.0, 0.2, 0.8], [0.2, 0.0, 0.9], [0.8, 0.9, 0.0]])
    linkage = _agglomerative_linkage(distance)
    assert linkage.tolist() == [[0.0, 1.0, 0.2, 2.0], [2.0, 3.0, 0.8, 3.0]]


def test_linkage_ties_break_to_smallest_pair():
    """Every pair ties, so the documented lexicographic rule decides."""
    distance = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    linkage = _agglomerative_linkage(distance)
    assert linkage.tolist() == [[0.0, 1.0, 0.5, 2.0], [2.0, 3.0, 0.5, 3.0]]
    assert _quasi_diagonal(linkage) == [2, 0, 1]


def test_quasi_diagonal_orders_a_familiar_tree():
    """The permutation places the first-merged pair together at the front."""
    distance = np.array([[0.0, 0.5, 1.0], [0.5, 0.0, 1.0], [1.0, 1.0, 0.0]])
    linkage = _agglomerative_linkage(distance)
    assert _quasi_diagonal(linkage) == [2, 0, 1]


def test_cluster_variance_is_inverse_variance_weighted():
    """Diag(4, 1): inverse-variance weights [0.2, 0.8] give variance 0.8."""
    covariance = np.array([[4.0, 0.0], [0.0, 1.0]])
    assert _cluster_variance(covariance, [0, 1]) == pytest.approx(0.8)


def test_two_unrelated_options_split_by_inverse_variance():
    """No correlation: the bisection is plain inverse-variance allocation."""
    covariance = np.array([[0.04, 0.0], [0.0, 0.01]])
    strategy = HierarchicalRiskParity(covariance)
    weights = strategy.evaluate(100.0)
    assert weights[0] == pytest.approx(0.2)
    assert weights[1] == pytest.approx(0.8)
    assert sum(weights) == pytest.approx(1.0, abs=PROBABILITY_SUM_TOLERANCE)


def test_equidistant_options_weighted_by_cluster_variance():
    """All pairs tie: the lone option outweighs the diversified pair 3:4.

    Option 2 keeps its full variance (1.0) while the paired options share a
    cluster variance of 0.75, so the bisection hands the pair 4/7 and option
    2 keeps 3/7.
    """
    covariance = np.array(
        [
            [1.0, 0.5, 0.5],
            [0.5, 1.0, 0.5],
            [0.5, 0.5, 1.0],
        ]
    )
    strategy = HierarchicalRiskParity(covariance)
    weights = strategy.evaluate(100.0)
    assert weights[2] == pytest.approx(3.0 / 7.0)
    assert weights[0] == pytest.approx(2.0 / 7.0)
    assert weights[1] == pytest.approx(2.0 / 7.0)


def test_planted_block_clusters_recovered():
    """Two planted blocks: contiguity, exchangeability, and risk ordering.

    A block-diagonal covariance with within-block correlation 0.9 plants two
    clusters of two options each; the low-variance block should receive the
    larger total allocation, split evenly inside.
    """
    covariance = np.array(
        [
            [0.04, 0.036, 0.0, 0.0],
            [0.036, 0.04, 0.0, 0.0],
            [0.0, 0.0, 0.01, 0.009],
            [0.0, 0.0, 0.009, 0.01],
        ]
    )
    strategy = HierarchicalRiskParity(covariance)
    assert set(strategy.order[:2]) == {0, 1}
    assert set(strategy.order[2:]) == {2, 3}

    weights = strategy.evaluate(1000.0)
    assert weights[0] == pytest.approx(weights[1])
    assert weights[2] == pytest.approx(weights[3])
    assert sum(weights) == pytest.approx(1.0, abs=PROBABILITY_SUM_TOLERANCE)
    assert weights[2] + weights[3] > weights[0] + weights[1]
    assert weights[0] == pytest.approx(0.1)
    assert weights[2] == pytest.approx(0.4)


@given(covariance=covariance_matrices())
def test_repeated_construction_replays_bit_identically(covariance):
    """No randomness or solver: construction twice replays bit-identically."""
    first = HierarchicalRiskParity(covariance)
    second = HierarchicalRiskParity(covariance)
    assert first.linkage.tolist() == second.linkage.tolist()
    assert first.order == second.order
    assert first.weights.tolist() == second.weights.tolist()


@given(
    covariance=covariance_matrices(),
    bankroll=st.floats(min_value=0.01, max_value=1e6, allow_nan=False),
)
def test_weight_contract_random_covariances(covariance, bankroll):
    """The weight contract holds on every admissible covariance."""
    strategy = HierarchicalRiskParity(covariance)
    weights = strategy.evaluate(bankroll)
    assert len(weights) == covariance.shape[0]
    assert all(weight >= 0.0 for weight in weights)
    assert all(weight <= 1.0 for weight in weights)
    assert sum(weights) <= 1 + PROBABILITY_SUM_TOLERANCE
    zeros = strategy.evaluate(0.0)
    assert zeros == (0.0,) * covariance.shape[0]
    assert strategy.evaluate(-1.0) == zeros


def test_optimize_reports_weights_and_volatility_only():
    """No solver ran: diagnostics carry the weights and volatility alone."""
    covariance = np.array([[0.04, 0.004], [0.004, 0.04]])
    strategy = HierarchicalRiskParity(covariance)
    result = strategy.optimize()
    assert isinstance(result, AllocationResult)
    assert result.weights.tolist() == [0.5, 0.5]
    assert result.objective is None
    assert result.converged is None
    assert result.iterations is None
    assert result.expected_growth is None
    assert result.volatility == pytest.approx(np.sqrt(0.022))
    assert result.weights is not strategy.weights


def test_single_option_gets_the_whole_budget():
    """A one-option portfolio has no tree and allocates fully."""
    strategy = HierarchicalRiskParity([[0.04]])
    assert strategy.linkage.shape == (0, 4)
    assert strategy.order == [0]
    assert strategy.evaluate(100.0) == (1.0,)
    assert strategy.evaluate(0.0) == (0.0,)
    result = strategy.optimize()
    assert result.weights.tolist() == [1.0]
    assert result.volatility == pytest.approx(0.2)


def test_singular_covariance_with_positive_variances_is_accepted():
    """Rank-deficient but PSD descriptors pass: the derivation never inverts."""
    factor = np.array([[1.0, 0.0], [2.0, 0.0], [0.5, 1.0]])
    covariance = factor @ factor.T
    strategy = HierarchicalRiskParity(covariance)
    # Options 0 and 1 are perfectly correlated (distance zero) and merge
    # first, so they sit adjacent in the permutation.
    assert abs(strategy.order.index(0) - strategy.order.index(1)) == 1
    weights = strategy.evaluate(100.0)
    assert sum(weights) == pytest.approx(1.0, abs=PROBABILITY_SUM_TOLERANCE)


def test_zero_cluster_variance_on_both_halves_splits_evenly():
    """Both halves carrying zero variance leaves no risk signal: split evenly.

    Called directly with a hand-picked order so the perfectly anticorrelated
    pairs land as the two halves - unreachable through the linkage, which
    never merges a perfect anticorrelation (the largest distance) first.
    """
    covariance = np.array(
        [
            [1.0, -1.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -1.0],
            [0.0, 0.0, -1.0, 1.0],
        ]
    )
    weights = _recursive_bisection(covariance, [0, 1, 2, 3])
    # The even split happens at the top, then each pair splits evenly again:
    # every option lands at a quarter of the budget.
    assert weights.tolist() == [0.25, 0.25, 0.25, 0.25]


def test_constructor_rejects_zero_variance_option():
    """A zero-variance option has undefined inverse-variance weights."""
    with pytest.raises(ValueError, match="strictly positive variances"):
        HierarchicalRiskParity([[0.04, 0.0], [0.0, 0.0]])


def test_constructor_rejects_invalid_covariance():
    """Descriptor validation propagates: asymmetry, then indefiniteness."""
    with pytest.raises(ValueError, match="symmetric"):
        HierarchicalRiskParity([[0.04, 0.0], [0.001, 0.04]])
    with pytest.raises(ValueError, match="positive semidefinite"):
        HierarchicalRiskParity([[0.04, 0.04], [0.04, 0.0]])
