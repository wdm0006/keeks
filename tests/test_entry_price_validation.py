"""Scalar-input validation for the entry-price API.

``find_indifference_price`` and every strategy's ``calculate_max_entry_price``
share one contract for their scalar controls: finite positive wealth, finite
positive tolerance, finite positive risk aversion (where applicable) and a
finite non-negative search fraction. Some invalid values used to loop forever
(``tolerance=0``, infinite wealth); others silently returned negative or NaN
prices.

``expected_utility`` is the fourth caller of that contract: it shares
``current_wealth`` and ``risk_aversion`` with its sibling and adds
``entry_price``, which must be finite but may be negative.
"""

import math
import threading

import pytest

from keeks.binary_strategies import CPPIStrategy
from keeks.utils import expected_utility, find_indifference_price

from .test_max_safe_bet_boundaries import STRATEGY_FACTORIES

OUTCOMES = [100, -50]
PROBABILITIES = [0.6, 0.4]
WEALTH = 5000.0

INVALID_WEALTH = [0.0, -1.0, -5000.0, math.nan, math.inf, -math.inf]
INVALID_TOLERANCE = [0.0, -0.01, -1.0, math.nan, math.inf, -math.inf]
INVALID_SEARCH_FRACTION = [-0.01, -1.0, math.nan, math.inf, -math.inf]
INVALID_RISK_AVERSION = [0.0, -1.0, math.nan, math.inf, -math.inf]
INVALID_ENTRY_PRICE = [math.nan, math.inf, -math.inf]

# Pinned pre-change value: validating the scalars must not move any valid call.
UNCHANGED_CALL = ([200, -100], [0.6, 0.4], 1000, 0, 2.0)
UNCHANGED_RESULT = -0.0009444444444444445


def _call_with_deadline(call, timeout=5.0):
    """Run ``call`` on a worker thread, failing the test if it never returns.

    ``tolerance=0`` and a non-finite ``current_wealth`` both used to spin the
    binary search forever, so a plain ``pytest.raises`` would hang instead of
    reporting a regression.
    """
    outcome = {}

    def run():
        try:
            outcome["value"] = call()
        except BaseException as exc:  # re-raised on the calling thread below
            outcome["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout)

    if worker.is_alive():
        pytest.fail(f"call did not return within {timeout} seconds")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


class TestFindIndifferencePriceScalars:
    """find_indifference_price rejects out-of-range scalar controls."""

    @pytest.mark.parametrize("tolerance", INVALID_TOLERANCE)
    def test_invalid_tolerance(self, tolerance):
        with pytest.raises(ValueError, match="Tolerance"):
            _call_with_deadline(
                lambda: find_indifference_price(
                    OUTCOMES, PROBABILITIES, WEALTH, tolerance=tolerance
                )
            )

    @pytest.mark.parametrize("current_wealth", INVALID_WEALTH)
    def test_invalid_current_wealth(self, current_wealth):
        with pytest.raises(ValueError, match="Current wealth"):
            _call_with_deadline(
                lambda: find_indifference_price(OUTCOMES, PROBABILITIES, current_wealth)
            )

    @pytest.mark.parametrize("risk_aversion", INVALID_RISK_AVERSION)
    def test_invalid_risk_aversion(self, risk_aversion):
        with pytest.raises(ValueError, match="Risk aversion"):
            _call_with_deadline(
                lambda: find_indifference_price(
                    OUTCOMES, PROBABILITIES, WEALTH, risk_aversion=risk_aversion
                )
            )

    @pytest.mark.parametrize("max_search_fraction", INVALID_SEARCH_FRACTION)
    def test_invalid_max_search_fraction(self, max_search_fraction):
        with pytest.raises(ValueError, match="Maximum search fraction"):
            _call_with_deadline(
                lambda: find_indifference_price(
                    OUTCOMES,
                    PROBABILITIES,
                    WEALTH,
                    max_search_fraction=max_search_fraction,
                )
            )

    def test_non_numeric_scalar(self):
        with pytest.raises(ValueError, match="Current wealth"):
            find_indifference_price(OUTCOMES, PROBABILITIES, "a lot")

    def test_zero_search_fraction_still_returns_zero(self):
        with pytest.warns(RuntimeWarning, match="saturated at its search bound"):
            price = find_indifference_price(
                OUTCOMES, PROBABILITIES, WEALTH, max_search_fraction=0.0
            )

        assert price == 0.0

    def test_search_fraction_above_one_is_accepted(self):
        """The documented ability to search beyond current wealth is preserved."""
        # A sure payout of ten times wealth: the true price is above the bound,
        # so the search saturates at current_wealth * max_search_fraction.
        with pytest.warns(RuntimeWarning, match="saturated at its search bound"):
            price = find_indifference_price(
                [10 * WEALTH], [1.0], WEALTH, tolerance=0.001
            )
        assert price == pytest.approx(0.5 * WEALTH, abs=0.01)

        with pytest.warns(RuntimeWarning, match="saturated at its search bound"):
            wide = find_indifference_price(
                [10 * WEALTH], [1.0], WEALTH, tolerance=0.001, max_search_fraction=2.0
            )
        assert wide == pytest.approx(2.0 * WEALTH, abs=0.01)
        assert wide > WEALTH


class TestExpectedUtilityScalars:
    """expected_utility enforces the same scalar contract as its sibling.

    ``INVALID_WEALTH`` straddles two guards, so this is not one rule tested six
    ways: ``nan``/``±inf`` are rejected by the finiteness coercion while ``0``
    and the negative values are rejected by the ``> 0`` comparison after it.
    """

    @pytest.mark.parametrize("current_wealth", INVALID_WEALTH)
    def test_invalid_current_wealth(self, current_wealth):
        with pytest.raises(ValueError, match="Current wealth"):
            expected_utility(OUTCOMES, PROBABILITIES, current_wealth, 0.0)

    @pytest.mark.parametrize("risk_aversion", INVALID_RISK_AVERSION)
    def test_invalid_risk_aversion(self, risk_aversion):
        with pytest.raises(ValueError, match="Risk aversion"):
            expected_utility(
                OUTCOMES, PROBABILITIES, WEALTH, 0.0, risk_aversion=risk_aversion
            )

    @pytest.mark.parametrize("entry_price", INVALID_ENTRY_PRICE)
    def test_invalid_entry_price(self, entry_price):
        with pytest.raises(ValueError, match="Entry price must be a finite number"):
            expected_utility(OUTCOMES, PROBABILITIES, WEALTH, entry_price)

    def test_non_numeric_scalar(self):
        with pytest.raises(ValueError, match="Current wealth"):
            expected_utility(OUTCOMES, PROBABILITIES, "a lot", 0.0)

    def test_negative_entry_price_is_accepted(self):
        """Being paid to take the gamble is meaningful, so only finiteness is required."""
        paid = expected_utility(OUTCOMES, PROBABILITIES, WEALTH, -50.0)
        free = expected_utility(OUTCOMES, PROBABILITIES, WEALTH, 0.0)

        assert math.isfinite(paid)
        assert paid > free

    def test_valid_call_is_numerically_unchanged(self):
        assert expected_utility(*UNCHANGED_CALL) == pytest.approx(
            UNCHANGED_RESULT, rel=0, abs=0
        )


@pytest.mark.parametrize("name", sorted(STRATEGY_FACTORIES))
class TestStrategyEntryPriceScalars:
    """All nine shipped strategies enforce the same scalar contract."""

    @pytest.mark.parametrize("current_wealth", INVALID_WEALTH)
    def test_invalid_current_wealth(self, name, current_wealth):
        strategy = STRATEGY_FACTORIES[name]()

        with pytest.raises(ValueError, match="Current wealth"):
            _call_with_deadline(
                lambda: strategy.calculate_max_entry_price(
                    OUTCOMES, PROBABILITIES, current_wealth
                )
            )

    @pytest.mark.parametrize("tolerance", INVALID_TOLERANCE)
    def test_invalid_tolerance(self, name, tolerance):
        strategy = STRATEGY_FACTORIES[name]()

        with pytest.raises(ValueError, match="Tolerance"):
            _call_with_deadline(
                lambda: strategy.calculate_max_entry_price(
                    OUTCOMES, PROBABILITIES, WEALTH, tolerance=tolerance
                )
            )

    @pytest.mark.parametrize("max_search_fraction", INVALID_SEARCH_FRACTION)
    def test_invalid_max_search_fraction(self, name, max_search_fraction):
        strategy = STRATEGY_FACTORIES[name]()

        with pytest.raises(ValueError, match="Maximum search fraction"):
            _call_with_deadline(
                lambda: strategy.calculate_max_entry_price(
                    OUTCOMES,
                    PROBABILITIES,
                    WEALTH,
                    max_search_fraction=max_search_fraction,
                )
            )

    def test_zero_search_fraction_returns_zero(self, name):
        """max_search_fraction=0 stays valid and prices the gamble at nothing."""
        strategy = STRATEGY_FACTORIES[name]()

        price = strategy.calculate_max_entry_price(
            OUTCOMES, PROBABILITIES, WEALTH, max_search_fraction=0.0
        )

        assert price == pytest.approx(0.0)

    def test_search_fraction_above_one_is_accepted(self, name):
        """A search fraction above 1.0 remains legal and never shrinks the price."""
        strategy = STRATEGY_FACTORIES[name]()
        baseline = strategy.calculate_max_entry_price(OUTCOMES, PROBABILITIES, WEALTH)

        strategy = STRATEGY_FACTORIES[name]()
        wide = strategy.calculate_max_entry_price(
            OUTCOMES, PROBABILITIES, WEALTH, max_search_fraction=2.0
        )

        assert wide >= baseline - 1e-9
        assert math.isfinite(wide)


def _cppi_state(strategy):
    return (strategy.current_bankroll, strategy.peak_bankroll, strategy.floor)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"current_wealth": -1.0},
        {"current_wealth": math.inf},
        {"current_wealth": WEALTH, "tolerance": 0.0},
        {"current_wealth": WEALTH, "max_search_fraction": -0.5},
        {"current_wealth": WEALTH, "max_search_fraction": math.nan},
    ],
)
def test_invalid_call_does_not_mutate_cppi_state(kwargs):
    """CPPI ratchets its floor from the wealth it is handed - not on a bad call."""
    strategy = CPPIStrategy(
        floor_fraction=0.5,
        multiplier=2.0,
        initial_bankroll=1000.0,
        payoff=1.0,
        loss=1.0,
        transaction_cost=0.01,
    )
    before = _cppi_state(strategy)

    with pytest.raises(ValueError):
        _call_with_deadline(
            lambda: strategy.calculate_max_entry_price(
                OUTCOMES, PROBABILITIES, **kwargs
            )
        )

    assert _cppi_state(strategy) == before
