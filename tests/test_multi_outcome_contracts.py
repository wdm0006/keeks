"""Contract tests for the multi-outcome strategy surface.

``BaseMultiOutcomeStrategy`` is the vector analogue of the binary
``BaseStrategy`` contract: one stake fraction per mutually exclusive leg,
validated like the scalar stake gate, with the aggregate cap derived from the
worst leg's charge. The binary ABC and all nine binary strategies stay
untouched; these tests only read ``BaseStrategy.get_max_safe_bet`` to pin the
cap equivalence.
"""

import re

import numpy as np
import pytest

from keeks.binary_strategies.base import BaseStrategy
from keeks.multi_outcome import BaseMultiOutcomeStrategy
from keeks.multi_outcome.base import _validate_stake_fractions
from keeks.utils import PROBABILITY_SUM_TOLERANCE, normalize_probabilities


class _ConcreteStrategy(BaseMultiOutcomeStrategy):
    """Minimal concrete subclass so the ABC's shared methods can be called."""

    def evaluate(self, probabilities, _current_bankroll):
        return _validate_stake_fractions([0.0] * len(probabilities))


class _ConcreteBinaryStrategy(BaseStrategy):
    """Minimal concrete binary subclass for the cap cross-check."""

    def evaluate(self, _probability, _current_bankroll):
        return 0.0


class _FixedStakesStrategy(BaseMultiOutcomeStrategy):
    """Returns a fixed stake vector through the documented validator gate."""

    def __init__(self, payoffs, loss, stakes, transaction_cost=0):
        super().__init__(payoffs, loss, transaction_cost)
        self._stakes = stakes

    def evaluate(self, _probabilities, _current_bankroll):
        return _validate_stake_fractions(self._stakes)


class _EqualStakesStrategy(BaseMultiOutcomeStrategy):
    """Spreads the largest safe aggregate evenly across every leg."""

    def evaluate(self, probabilities, current_bankroll):
        probabilities = normalize_probabilities(probabilities)
        aggregate = self.get_max_safe_total_bet(current_bankroll)
        return _validate_stake_fractions(
            [aggregate / len(probabilities)] * len(probabilities)
        )


# ---
# The ABC contract: what it takes to become concrete, and what construction
# validates.
# ---


def test_base_is_abstract():
    """The ABC cannot be instantiated without implementing evaluate."""
    with pytest.raises(TypeError, match="abstract"):
        BaseMultiOutcomeStrategy(payoffs=[2.0], loss=1.0)


def test_abstract_surface_is_evaluate_only():
    """evaluate is the one method a concrete multi-outcome strategy must provide."""
    assert BaseMultiOutcomeStrategy.__abstractmethods__ == frozenset({"evaluate"})


@pytest.mark.parametrize(
    "payoffs",
    [
        [2.0, 3.0, 4.0],
        (2.0, 3.0, 4.0),
        np.array([2.0, 3.0, 4.0]),
    ],
)
def test_constructor_accepts_any_sequence(payoffs):
    """Any sequence input is accepted and stored as an immutable tuple."""
    strategy = _ConcreteStrategy(payoffs=payoffs, loss=1.0)

    assert strategy.payoffs == (2.0, 3.0, 4.0)
    assert strategy.loss == 1.0
    assert strategy.transaction_cost == 0


def test_constructor_payoffs_are_frozen_at_construction():
    """Later mutation of the caller's sequence cannot reprice the strategy."""
    payoffs = [2.0, 3.0]
    strategy = _ConcreteStrategy(payoffs=payoffs, loss=1.0)

    payoffs.append(99.0)

    assert strategy.payoffs == (2.0, 3.0)


@pytest.mark.parametrize(
    ("payoffs", "message"),
    [
        # None coerces to a 0-d nan array, mirroring normalize_probabilities.
        (None, "Payoffs must be one-dimensional"),
        (object(), "Payoffs must be a finite sequence"),
        (["a", "b"], "Payoffs must be a finite sequence"),
        (2.0, "Payoffs must be one-dimensional"),
        ([[2.0], [3.0]], "Payoffs must be one-dimensional"),
        ([], "Payoffs must be non-empty"),
        ([2.0, float("nan")], "Payoffs must contain only finite values"),
        ([2.0, float("inf")], "Payoffs must contain only finite values"),
        ([2.0, 0.0], "Payoffs must be greater than 0"),
        ([2.0, -1.0], "Payoffs must be greater than 0"),
    ],
)
def test_constructor_rejects_bad_payoffs(payoffs, message):
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        _ConcreteStrategy(payoffs=payoffs, loss=1.0)


@pytest.mark.parametrize(
    ("loss", "transaction_cost", "message"),
    [
        (float("nan"), 0.0, "Loss must be a finite number"),
        (float("inf"), 0.0, "Loss must be a finite number"),
        (-0.1, 0.0, "Loss must be non-negative"),
        (1.0, float("nan"), "Transaction cost must be a finite number"),
        (1.0, -0.1, "Transaction cost must be non-negative"),
        (0.0, 0.0, "Total cost (loss + transaction_cost) must be greater than 0"),
    ],
)
def test_constructor_rejects_bad_costs(loss, transaction_cost, message):
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        _ConcreteStrategy(payoffs=[2.0], loss=loss, transaction_cost=transaction_cost)


# ---
# The vector stake validator: every failure mode of the scalar stake gate,
# vectorized, plus the aggregate sum cap.
# ---


@pytest.mark.parametrize(
    "stakes",
    [
        [0.25, 0.25, 0.5],
        (0.25, 0.25, 0.5),
        np.array([0.25, 0.25, 0.5]),
        [1.0],
        [0.0, 0.0, 0.0],
        # Within PROBABILITY_SUM_TOLERANCE of one - accepted.
        [0.5, 0.5 + 5e-13],
    ],
)
def test_validator_accepts_valid_stake_vectors(stakes):
    """Any sequence input is accepted; the validated stakes come back as a tuple."""
    result = _validate_stake_fractions(stakes)

    assert isinstance(result, tuple)
    assert all(isinstance(value, float) for value in result)
    assert result == tuple(np.asarray(stakes, dtype=float).tolist())
    assert sum(result) <= 1 + PROBABILITY_SUM_TOLERANCE


@pytest.mark.parametrize(
    ("stakes", "message"),
    [
        # None coerces to a 0-d nan array, mirroring normalize_probabilities.
        (None, "Strategy stake fractions must be one-dimensional"),
        (object(), "Strategy stake fractions must be a finite sequence"),
        (0.25, "Strategy stake fractions must be one-dimensional"),
        (
            [[0.25, 0.25], [0.25, 0.25]],
            "Strategy stake fractions must be one-dimensional",
        ),
        ([], "Strategy stake fractions must be non-empty"),
        (
            [0.25, float("nan")],
            "Strategy stake fractions must contain only finite values",
        ),
        (
            [0.25, float("inf")],
            "Strategy stake fractions must contain only finite values",
        ),
        ([-0.25, 0.5], "Strategy stake fractions must be between 0 and 1"),
        ([1.5], "Strategy stake fractions must be between 0 and 1"),
        ([0.6, 0.6], "Strategy stake fractions must sum to no more than one"),
        # Above the tolerance by more than PROBABILITY_SUM_TOLERANCE - rejected.
        ([0.5, 0.5 + 1e-9], "Strategy stake fractions must sum to no more than one"),
    ],
)
def test_validator_rejects_bad_stake_vectors(stakes, message):
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        _validate_stake_fractions(stakes)


def test_overcommitted_stakes_are_rejected_at_the_validator():
    """A strategy promising more than the bankroll has cannot leave evaluate unchecked."""
    strategy = _FixedStakesStrategy(payoffs=[2.0, 3.0], loss=1.0, stakes=[0.8, 0.8])

    with pytest.raises(
        ValueError, match="Strategy stake fractions must sum to no more than one"
    ):
        strategy.evaluate([0.5, 0.3], 1000.0)


# ---
# The evaluate contract as documented on the ABC: one fraction per leg, any
# sequence in, tuple out, probabilities gated by P1's public normalizer.
# ---


@pytest.mark.parametrize(
    "probabilities",
    [
        [0.5, 0.3, 0.1],
        (0.5, 0.3, 0.1),
        np.array([0.5, 0.3, 0.1]),
    ],
)
def test_evaluate_returns_one_fraction_per_leg(probabilities):
    """len == len(ps), every element in [0, 1], sum at most 1 + tolerance."""
    strategy = _EqualStakesStrategy(payoffs=[2.0, 3.0, 4.0], loss=1.0)

    result = strategy.evaluate(probabilities, 1000.0)

    assert isinstance(result, tuple)
    assert len(result) == len(probabilities)
    assert all(0.0 <= value <= 1.0 for value in result)
    assert sum(result) <= 1 + PROBABILITY_SUM_TOLERANCE


def test_evaluate_spreads_the_safe_aggregate():
    """A concrete strategy caps its total at get_max_safe_total_bet."""
    strategy = _EqualStakesStrategy(
        payoffs=[2.0, 3.0, 4.0], loss=1.0, transaction_cost=0.01
    )

    result = strategy.evaluate([0.5, 0.3, 0.1], 1000.0)

    assert result == pytest.approx((1 / 1.01 / 3, 1 / 1.01 / 3, 1 / 1.01 / 3))


def test_evaluate_stakes_nothing_on_a_depleted_bankroll():
    """A non-positive bankroll leaves every leg at zero - no stake at all."""
    strategy = _EqualStakesStrategy(payoffs=[2.0, 3.0, 4.0], loss=1.0)

    assert strategy.evaluate([0.5, 0.3, 0.1], 0.0) == (0.0, 0.0, 0.0)


def test_evaluate_rejects_probability_sum_above_tolerance():
    """The probability gate is P1's public normalizer, not a validator rewrite."""
    strategy = _EqualStakesStrategy(payoffs=[2.0, 3.0, 4.0], loss=1.0)

    with pytest.raises(ValueError, match="Probabilities must sum to no more than one"):
        strategy.evaluate([0.6, 0.6, 0.1], 1000.0)


# ---
# get_max_safe_total_bet boundaries: the aggregate cap is the worst leg's
# min(1, 1 / (loss + transaction_cost)) bound.
# ---


@pytest.mark.parametrize("bankroll", [0.0, -0.01, -100.0])
def test_get_max_safe_total_bet_zero_at_non_positive_bankroll(bankroll):
    """The clamp returns 0.0 rather than dividing by the bankroll."""
    strategy = _ConcreteStrategy(payoffs=[2.0, 3.0], loss=1.0, transaction_cost=0.01)

    assert strategy.get_max_safe_total_bet(bankroll) == 0.0


@pytest.mark.parametrize(
    ("loss", "transaction_cost", "expected"),
    [
        # loss + transaction_cost above 1.0 - the reciprocal binds.
        (1.5, 0.1, 1 / 1.6),
        (4.0, 0.0, 0.25),
        # loss + transaction_cost below 1.0 - the full bankroll is the cap.
        (0.5, 0.0, 1.0),
        (0.0, 0.5, 1.0),
        # loss + transaction_cost exactly 1.0.
        (1.0, 0.0, 1.0),
    ],
)
def test_get_max_safe_total_bet_boundaries(loss, transaction_cost, expected):
    """The aggregate cap is min(1, 1 / (loss + transaction_cost))."""
    strategy = _ConcreteStrategy(
        payoffs=[2.0, 3.0, 4.0], loss=loss, transaction_cost=transaction_cost
    )

    assert strategy.get_max_safe_total_bet(1000.0) == pytest.approx(expected)


@pytest.mark.parametrize("n_legs", [1, 2, 3, 10])
def test_get_max_safe_total_bet_is_leg_count_independent(n_legs):
    """The cap bounds the total across N legs, which share one scalar charge."""
    strategy = _ConcreteStrategy(
        payoffs=[2.0] * n_legs, loss=1.0, transaction_cost=0.01
    )

    assert strategy.get_max_safe_total_bet(1000.0) == pytest.approx(1 / 1.01)


@pytest.mark.parametrize("bankroll", [1000.0, 42.0, 0.0, -5.0])
def test_aggregate_cap_equals_binary_single_bet_cap(bankroll):
    """With shared scalar charges the worst leg's bound is the binary bound."""
    binary = _ConcreteBinaryStrategy(payoff=2.0, loss=1.0, transaction_cost=0.01)
    multi = _ConcreteStrategy(payoffs=[2.0, 3.0, 4.0], loss=1.0, transaction_cost=0.01)

    assert multi.get_max_safe_total_bet(bankroll) == binary.get_max_safe_bet(bankroll)


def test_get_max_safe_total_bet_rejects_nonfinite_bankroll():
    strategy = _ConcreteStrategy(payoffs=[2.0], loss=1.0)

    with pytest.raises(ValueError, match="Current bankroll must be a finite number"):
        strategy.get_max_safe_total_bet(float("nan"))
