"""
Tests that ``crra_utility`` never evaluates its utility over nonpositive wealth.

Wealth at or below zero has no CRRA utility, so the function reports ``-inf``
for it. Computing the log/power over the whole array and masking afterwards
returns those same values, but applies the undefined operation to every
excluded entry first, and NumPy warns once per one. That noise is
indistinguishable from the ``RuntimeWarning`` ``find_indifference_price``
raises on purpose, so a consumer escalating ``RuntimeWarning`` to an error
cannot act on the signal.

These tests pin both halves: that the calls stay silent, and that the values
they return are unchanged. The warning assertions are deliberately scoped to
their own fixtures — the suite as a whole triggers roughly ten legitimate
saturation warnings, so it cannot run under a blanket ``-W error``.
"""

import math
import warnings
from contextlib import contextmanager

import numpy as np
import pytest

from keeks.binary_strategies import KellyCriterion, MertonShare
from keeks.utils import crra_utility, expected_utility

# Wealth levels spanning both undefined entries and a defined one.
MIXED_WEALTH = np.array([100.0, 0.0, -50.0])

# A gamble that can wipe out the caller's whole wealth — the case CRRA utility
# exists to model, and the one that drives the binary search into nonpositive
# final wealth on every iteration.
WIPEOUT_OUTCOMES = [200.0, -1000.0]
WIPEOUT_PROBABILITIES = [0.6, 0.4]
WIPEOUT_WEALTH = 1000.0

# Utility of $100 of wealth, per risk aversion. gamma=1 is the log branch;
# the others are the power branch, one on each side of 1.
UTILITY_OF_100 = {
    0.5: 20.0,  # 100 ** 0.5 / 0.5
    1.0: math.log(100.0),
    2.0: -0.01,  # 100 ** -1 / -1
    3.0: -5e-05,  # 100 ** -2 / -2
}


@contextmanager
def _no_warnings():
    """Turn any warning raised inside the block into a test failure."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


class TestNonpositiveWealthIsSilent:
    """Undefined entries are excluded rather than computed and masked."""

    @pytest.mark.parametrize("risk_aversion", sorted(UTILITY_OF_100))
    def test_mixed_sign_array_emits_no_warning(self, risk_aversion):
        with _no_warnings():
            crra_utility(MIXED_WEALTH, risk_aversion)

    @pytest.mark.parametrize("risk_aversion", sorted(UTILITY_OF_100))
    def test_all_positive_array_emits_no_warning(self, risk_aversion):
        with _no_warnings():
            crra_utility(np.array([100.0, 1.0, 3.5]), risk_aversion)

    @pytest.mark.parametrize("risk_aversion", sorted(UTILITY_OF_100))
    def test_nonpositive_scalar_emits_no_warning(self, risk_aversion):
        with _no_warnings():
            crra_utility(-50.0, risk_aversion)

    @pytest.mark.parametrize("risk_aversion", sorted(UTILITY_OF_100))
    def test_expected_utility_of_a_wipeout_gamble_emits_no_warning(self, risk_aversion):
        with _no_warnings():
            expected_utility(
                WIPEOUT_OUTCOMES,
                WIPEOUT_PROBABILITIES,
                current_wealth=WIPEOUT_WEALTH,
                entry_price=100.0,
                risk_aversion=risk_aversion,
            )


class TestEntryPricingIsSilent:
    """Pricing a gamble that can wipe out the caller's wealth stays quiet."""

    def test_kelly_max_entry_price(self):
        # Kelly prices from log utility, so this exercises the log branch.
        strategy = KellyCriterion(payoff=1.0, loss=1.0, transaction_cost=0.0)
        with _no_warnings():
            price = strategy.calculate_max_entry_price(
                WIPEOUT_OUTCOMES,
                WIPEOUT_PROBABILITIES,
                current_wealth=WIPEOUT_WEALTH,
            )
        assert price == pytest.approx(0.003814697265625)

    @pytest.mark.parametrize("risk_aversion", [0.5, 1.0, 2.0])
    def test_merton_share_max_entry_price(self, risk_aversion):
        strategy = MertonShare(
            payoff=1.0,
            loss=1.0,
            transaction_cost=0.0,
            risk_aversion=risk_aversion,
        )
        with _no_warnings():
            price = strategy.calculate_max_entry_price(
                WIPEOUT_OUTCOMES,
                WIPEOUT_PROBABILITIES,
                current_wealth=WIPEOUT_WEALTH,
            )
        assert price == pytest.approx(0.003814697265625)


class TestValuesAreUnchanged:
    """Excluding the undefined entries returns exactly what masking returned."""

    @pytest.mark.parametrize("risk_aversion", sorted(UTILITY_OF_100))
    def test_mixed_sign_array(self, risk_aversion):
        result = crra_utility(MIXED_WEALTH, risk_aversion)
        expected = np.array([UTILITY_OF_100[risk_aversion], -np.inf, -np.inf])
        assert np.array_equal(result, expected)

    @pytest.mark.parametrize("risk_aversion", sorted(UTILITY_OF_100))
    def test_all_positive_array_takes_the_fast_path(self, risk_aversion):
        # No entry is nonpositive, so the guarded branch is never entered.
        result = crra_utility(np.array([100.0, 100.0]), risk_aversion)
        expected = np.array([UTILITY_OF_100[risk_aversion]] * 2)
        assert np.array_equal(result, expected)

    @pytest.mark.parametrize("risk_aversion", sorted(UTILITY_OF_100))
    def test_positive_scalar(self, risk_aversion):
        assert crra_utility(100.0, risk_aversion) == UTILITY_OF_100[risk_aversion]

    @pytest.mark.parametrize("risk_aversion", sorted(UTILITY_OF_100))
    @pytest.mark.parametrize("wealth", [0.0, -50.0])
    def test_nonpositive_scalar_returns_negative_infinity(self, wealth, risk_aversion):
        # The scalar path returns early and computes nothing.
        assert crra_utility(wealth, risk_aversion) == -np.inf

    @pytest.mark.parametrize("risk_aversion", sorted(UTILITY_OF_100))
    def test_zero_dimensional_array(self, risk_aversion):
        result = crra_utility(np.array(-50.0), risk_aversion)
        assert result.shape == ()
        assert result == -np.inf

    @pytest.mark.parametrize("risk_aversion", sorted(UTILITY_OF_100))
    def test_not_a_number_is_preserved(self, risk_aversion):
        # NaN is not "wealth <= 0", so it is evaluated and stays NaN rather
        # than collapsing to -inf.
        result = crra_utility(np.array([100.0, 0.0, np.nan]), risk_aversion)
        assert result[0] == UTILITY_OF_100[risk_aversion]
        assert result[1] == -np.inf
        assert np.isnan(result[2])
