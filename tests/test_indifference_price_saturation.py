"""
Tests for the saturation signal on ``find_indifference_price``.

The binary search is bounded above by ``current_wealth * max_search_fraction``.
When the true indifference price lies at or above that bound the function
returns the bound itself, which is indistinguishable from a solved price unless
it says so. These tests pin the ``RuntimeWarning`` that says so, and pin that
non-saturating calls stay silent.
"""

import warnings
from contextlib import contextmanager

import numpy as np
import pytest

from keeks.binary_strategies import (
    CPPIStrategy,
    DrawdownAdjustedKelly,
    DynamicBankrollManagement,
    FixedFractionStrategy,
    FractionalKellyCriterion,
    KellyCriterion,
    MertonShare,
    NaiveStrategy,
    OptimalF,
)
from keeks.utils import find_indifference_price

# A gamble paying $1000 with certainty. A log-utility agent's indifference price
# for it is exactly $1000, so the default 0.5 search fraction binds below
# $2000 of wealth and does not bind above it.
CERTAIN_OUTCOMES = [1000.0]
CERTAIN_PROBABILITIES = [1.0]


@contextmanager
def _assert_no_runtime_warning():
    """Fail the test if the wrapped block emits any ``RuntimeWarning``."""
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        yield
    unexpected = [str(r.message) for r in records if r.category is RuntimeWarning]
    assert not unexpected, f"unexpected RuntimeWarning(s): {unexpected}"


class TestSaturationWarning:
    """``find_indifference_price`` warns when its search bound binds."""

    def test_warns_and_returns_the_bound(self):
        with pytest.warns(RuntimeWarning) as record:
            price = find_indifference_price(
                CERTAIN_OUTCOMES,
                CERTAIN_PROBABILITIES,
                current_wealth=1000.0,
                risk_aversion=1.0,
                tolerance=1e-6,
            )

        message = str(record[0].message)
        assert "500.0" in message
        assert "max_search_fraction=0.5" in message
        # Unchanged from before the warning existed: the bound, to tolerance.
        assert price == pytest.approx(500.0, abs=1e-6)

    def test_no_warning_when_the_bound_does_not_bind(self):
        with _assert_no_runtime_warning():
            price = find_indifference_price(
                CERTAIN_OUTCOMES,
                CERTAIN_PROBABILITIES,
                current_wealth=4000.0,
                risk_aversion=1.0,
                tolerance=1e-6,
            )

        # The honest answer: a certain $1000 is worth $1000 to a log-utility agent.
        assert price == pytest.approx(1000.0, abs=1e-6)

    def test_message_names_a_custom_bound(self):
        with pytest.warns(RuntimeWarning, match=r"max_search_fraction=0\.25") as record:
            find_indifference_price(
                CERTAIN_OUTCOMES,
                CERTAIN_PROBABILITIES,
                current_wealth=1000.0,
                max_search_fraction=0.25,
            )

        assert "250.0" in str(record[0].message)

    def test_zero_search_fraction_warns_and_still_returns_zero(self):
        """A zero bound genuinely understates a favourable gamble's price."""
        with pytest.warns(RuntimeWarning, match=r"max_search_fraction=0"):
            price = find_indifference_price(
                CERTAIN_OUTCOMES,
                CERTAIN_PROBABILITIES,
                current_wealth=1000.0,
                max_search_fraction=0.0,
            )

        assert price == 0.0

    @pytest.mark.parametrize(
        ("outcomes", "probabilities", "current_wealth", "risk_aversion"),
        [
            # Fair coin flip: worth almost nothing, nowhere near the bound.
            ([100, -100], [0.5, 0.5], 1000, 1.0),
            # Positive-EV bet, priced well inside the bound.
            ([200, -50], [0.5, 0.5], 1000, 2.0),
            # St. Petersburg: infinite EV, but a modest utility price.
            (
                [2**n for n in range(1, 21)],
                [(0.5) ** n for n in range(1, 21)],
                10000,
                2.0,
            ),
        ],
    )
    def test_existing_fixtures_stay_warning_free(
        self, outcomes, probabilities, current_wealth, risk_aversion
    ):
        with _assert_no_runtime_warning():
            find_indifference_price(
                outcomes, probabilities, current_wealth, risk_aversion
            )


class TestStrategyDelegation:
    """The utility-backed overrides inherit the signal; the others cannot."""

    @pytest.mark.parametrize(
        "strategy_factory",
        [
            pytest.param(
                lambda: KellyCriterion(payoff=1.0, loss=1.0, transaction_cost=0.0),
                id="KellyCriterion",
            ),
            pytest.param(
                lambda: FractionalKellyCriterion(
                    payoff=1.0, loss=1.0, transaction_cost=0.0, fraction=0.5
                ),
                id="FractionalKellyCriterion",
            ),
            pytest.param(
                lambda: DrawdownAdjustedKelly(
                    payoff=1.0, loss=1.0, transaction_cost=0.0
                ),
                id="DrawdownAdjustedKelly",
            ),
            pytest.param(
                lambda: OptimalF(
                    payoff=1.0, loss=1.0, transaction_cost=0.0, win_rate=0.6
                ),
                id="OptimalF",
            ),
            pytest.param(
                lambda: MertonShare(payoff=1.0, loss=1.0, transaction_cost=0.01),
                id="MertonShare",
            ),
        ],
    )
    def test_utility_backed_overrides_warn(self, strategy_factory):
        with pytest.warns(RuntimeWarning, match="saturated at its search bound"):
            strategy_factory().calculate_max_entry_price(
                CERTAIN_OUTCOMES, CERTAIN_PROBABILITIES, 1000.0
            )

    @pytest.mark.parametrize(
        "strategy_factory",
        [
            pytest.param(
                lambda: NaiveStrategy(payoff=1.0, loss=1.0, transaction_cost=0.0),
                id="NaiveStrategy",
            ),
            pytest.param(
                lambda: FixedFractionStrategy(
                    fraction=0.1, payoff=1.0, loss=1.0, transaction_cost=0.0
                ),
                id="FixedFractionStrategy",
            ),
            pytest.param(
                lambda: CPPIStrategy(
                    floor_fraction=0.5,
                    multiplier=2.0,
                    initial_bankroll=1000.0,
                    payoff=1.0,
                    loss=1.0,
                ),
                id="CPPIStrategy",
            ),
            pytest.param(
                lambda: DynamicBankrollManagement(
                    base_fraction=0.1, payoff=1.0, loss=1.0, transaction_cost=0.0
                ),
                id="DynamicBankrollManagement",
            ),
        ],
    )
    def test_mechanical_overrides_do_not_warn(self, strategy_factory):
        """Their fraction-of-wealth rule is deliberate, not a clamped search."""
        with _assert_no_runtime_warning():
            strategy_factory().calculate_max_entry_price(
                CERTAIN_OUTCOMES, CERTAIN_PROBABILITIES, 1000.0
            )


class TestZeroProbabilityOutcomes:
    """Explicit p=0 rows must not poison the expectation with NaN.

    A zero-probability outcome whose final wealth is nonpositive carries a
    -inf utility; 0 * -inf used to be NaN, which made every EU comparison
    false, collapsed the bisection onto current wealth, and suppressed the
    saturation warning for the same gamble. Regression for the accuracy
    audit's finding F2.
    """

    OUTCOMES = [100000.0, 99000.0, 0.0]
    PROBABILITIES = [0.5, 0.5, 0.0]

    def test_zero_probability_row_does_not_truncate_the_price(self):
        """The search reaches the true price instead of stopping at wealth."""
        # Exact solution: with x = current_wealth - price, the indifference
        # condition is 0.5*ln(x+100000) + 0.5*ln(x+99000) = ln(1000), i.e.
        # (x+100000)(x+99000) = 1000**2, giving price ~= 99381.97.
        with (
            np.errstate(invalid="raise", divide="raise"),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("error", RuntimeWarning)
            price = find_indifference_price(
                self.OUTCOMES,
                self.PROBABILITIES,
                current_wealth=1000.0,
                risk_aversion=1.0,
                tolerance=0.01,
                max_search_fraction=100,
            )

        assert price == pytest.approx(99381.9684, rel=1e-4)

    def test_saturation_warning_is_no_longer_suppressed(self):
        """A bound below the true price warns again; NaN disabled it before."""
        with (
            np.errstate(invalid="raise", divide="raise"),
            pytest.warns(RuntimeWarning, match="saturated") as record,
        ):
            find_indifference_price(
                self.OUTCOMES,
                self.PROBABILITIES,
                current_wealth=1000.0,
                risk_aversion=1.0,
                tolerance=0.01,
                max_search_fraction=50,
            )

        # Exactly one warning: the saturation signal, no numpy leak alongside.
        assert len(record) == 1

    def test_matches_the_same_gamble_without_the_zero_row(self):
        """Masking the p=0 row leaves the price equal to the control gamble."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with_row = find_indifference_price(
                self.OUTCOMES,
                self.PROBABILITIES,
                current_wealth=1000.0,
                risk_aversion=1.0,
                tolerance=0.01,
                max_search_fraction=100,
            )
            without_row = find_indifference_price(
                [100000.0, 99000.0],
                [0.5, 0.5],
                current_wealth=1000.0,
                risk_aversion=1.0,
                tolerance=0.01,
                max_search_fraction=100,
            )

        assert with_row == pytest.approx(without_row, abs=0.02)
