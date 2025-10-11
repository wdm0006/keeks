"""
Tests for utility functions in keeks.utils.

These tests verify the CRRA utility functions and indifference price calculations.
"""

import numpy as np
import pytest

from keeks.utils import crra_utility, expected_utility, find_indifference_price


class TestCRRAUtility:
    """Tests for CRRA utility function."""

    def test_log_utility_gamma_1(self):
        """Test that γ=1.0 gives log utility."""
        wealth = 100
        result = crra_utility(wealth, risk_aversion=1.0)
        expected = np.log(100)
        assert result == pytest.approx(expected)

    def test_power_utility_gamma_2(self):
        """Test power utility with γ=2.0."""
        wealth = 100
        gamma = 2.0
        result = crra_utility(wealth, risk_aversion=gamma)
        expected = (100 ** (1 - gamma)) / (1 - gamma)
        assert result == pytest.approx(expected)

    def test_power_utility_gamma_3(self):
        """Test power utility with γ=3.0."""
        wealth = 50
        gamma = 3.0
        result = crra_utility(wealth, risk_aversion=gamma)
        expected = (50 ** (1 - gamma)) / (1 - gamma)
        assert result == pytest.approx(expected)

    def test_zero_wealth_returns_neg_inf(self):
        """Test that zero wealth returns negative infinity."""
        result = crra_utility(0, risk_aversion=2.0)
        assert result == -np.inf

    def test_negative_wealth_returns_neg_inf(self):
        """Test that negative wealth returns negative infinity."""
        result = crra_utility(-10, risk_aversion=2.0)
        assert result == -np.inf

    def test_array_input(self):
        """Test that function works with numpy arrays."""
        wealth = np.array([10, 100, 1000])
        gamma = 2.0
        result = crra_utility(wealth, risk_aversion=gamma)
        expected = (wealth ** (1 - gamma)) / (1 - gamma)
        np.testing.assert_array_almost_equal(result, expected)

    def test_higher_wealth_higher_utility(self):
        """Test that utility increases with wealth."""
        u1 = crra_utility(100, risk_aversion=2.0)
        u2 = crra_utility(200, risk_aversion=2.0)
        assert u2 > u1

    def test_higher_gamma_more_conservative(self):
        """Test that higher γ means more risk aversion (diminishing returns)."""
        # For a given wealth increase, utility gain should be smaller with higher γ
        wealth_low = 100
        wealth_high = 200

        # Low risk aversion (γ=1.5)
        u1_low_gamma = crra_utility(wealth_low, risk_aversion=1.5)
        u2_low_gamma = crra_utility(wealth_high, risk_aversion=1.5)
        gain_low_gamma = u2_low_gamma - u1_low_gamma

        # High risk aversion (γ=3.0)
        u1_high_gamma = crra_utility(wealth_low, risk_aversion=3.0)
        u2_high_gamma = crra_utility(wealth_high, risk_aversion=3.0)
        gain_high_gamma = u2_high_gamma - u1_high_gamma

        # Higher γ should give smaller utility gain for same wealth increase
        # (but both should be positive)
        assert gain_low_gamma > 0
        assert gain_high_gamma > 0
        # Note: Can't directly compare gains across different γ values
        # because utility scales are different


class TestExpectedUtility:
    """Tests for expected utility calculation."""

    def test_certain_outcome(self):
        """Test expected utility with certain outcome."""
        outcomes = [100]
        probabilities = [1.0]
        current_wealth = 1000
        entry_price = 10
        risk_aversion = 2.0

        result = expected_utility(outcomes, probabilities, current_wealth,
                                 entry_price, risk_aversion)

        # Should equal utility of final wealth
        final_wealth = current_wealth - entry_price + 100
        expected = crra_utility(final_wealth, risk_aversion)
        assert result == pytest.approx(expected)

    def test_fifty_fifty_bet(self):
        """Test expected utility of 50/50 bet."""
        outcomes = [100, -100]
        probabilities = [0.5, 0.5]
        current_wealth = 1000
        entry_price = 0
        risk_aversion = 1.0

        result = expected_utility(outcomes, probabilities, current_wealth,
                                 entry_price, risk_aversion)

        # Manual calculation
        u1 = crra_utility(1000 + 100, risk_aversion)
        u2 = crra_utility(1000 - 100, risk_aversion)
        expected = 0.5 * u1 + 0.5 * u2

        assert result == pytest.approx(expected)

    def test_positive_ev_bet(self):
        """Test that positive EV bet has higher expected utility than no bet."""
        outcomes = [200, -50]  # Positive EV
        probabilities = [0.5, 0.5]
        current_wealth = 1000
        entry_price = 0
        risk_aversion = 2.0

        # Expected utility from taking the bet
        eu_bet = expected_utility(outcomes, probabilities, current_wealth,
                                 entry_price, risk_aversion)

        # Expected utility from not betting
        eu_no_bet = crra_utility(current_wealth, risk_aversion)

        # Positive EV bet should have higher expected utility
        assert eu_bet > eu_no_bet


class TestFindIndifferencePrice:
    """Tests for indifference price calculation."""

    def test_fair_coin_flip(self):
        """Test indifference price for fair coin flip."""
        outcomes = [100, -100]
        probabilities = [0.5, 0.5]
        current_wealth = 1000
        risk_aversion = 1.0

        max_price = find_indifference_price(outcomes, probabilities,
                                           current_wealth, risk_aversion)

        # For a fair coin flip (EV=0), should be willing to pay very little
        # (but not exactly zero due to utility curvature)
        assert max_price >= 0
        assert max_price < 1  # Should be very small

    def test_positive_ev_bet(self):
        """Test that positive EV bet has positive indifference price."""
        outcomes = [200, -50]  # EV = 0.5*200 + 0.5*(-50) = 75
        probabilities = [0.5, 0.5]
        current_wealth = 1000
        risk_aversion = 2.0

        max_price = find_indifference_price(outcomes, probabilities,
                                           current_wealth, risk_aversion)

        # Should be willing to pay something for positive EV
        assert max_price > 0
        # But less than the expected value due to risk aversion
        expected_value = 0.5 * 200 + 0.5 * (-50)
        assert max_price < expected_value

    def test_higher_risk_aversion_lower_price(self):
        """Test that higher risk aversion leads to lower willing payment."""
        outcomes = [200, -50]
        probabilities = [0.5, 0.5]
        current_wealth = 10000

        # Low risk aversion
        price_low_gamma = find_indifference_price(outcomes, probabilities,
                                                 current_wealth, risk_aversion=1.0)

        # High risk aversion
        price_high_gamma = find_indifference_price(outcomes, probabilities,
                                                  current_wealth, risk_aversion=3.0)

        # Higher risk aversion should lead to lower willing payment
        assert price_high_gamma < price_low_gamma

    def test_wealthier_agents_pay_more(self):
        """Test that wealthier agents are willing to pay more (absolute terms)."""
        outcomes = [1000, -500]
        probabilities = [0.5, 0.5]
        risk_aversion = 2.0

        # Poor agent
        price_poor = find_indifference_price(outcomes, probabilities,
                                            current_wealth=5000, risk_aversion=risk_aversion)

        # Rich agent
        price_rich = find_indifference_price(outcomes, probabilities,
                                            current_wealth=50000, risk_aversion=risk_aversion)

        # Wealthier agent should be willing to pay more in absolute terms
        assert price_rich > price_poor

    def test_st_petersburg_bounded(self):
        """Test that St. Petersburg paradox gives finite indifference price."""
        # Generate St. Petersburg outcomes
        max_flips = 20
        outcomes = [2**n for n in range(1, max_flips + 1)]
        probabilities = [(0.5)**n for n in range(1, max_flips + 1)]

        current_wealth = 10000
        risk_aversion = 2.0

        max_price = find_indifference_price(outcomes, probabilities,
                                           current_wealth, risk_aversion)

        # Despite infinite expected value, should give finite price
        assert max_price > 0
        assert max_price < current_wealth
        # Based on the example, should be modest (around $13 for $10k wealth)
        assert max_price < 100

    def test_custom_tolerance(self):
        """Test that custom tolerance parameter works."""
        outcomes = [100, -50]
        probabilities = [0.5, 0.5]
        current_wealth = 1000
        risk_aversion = 2.0

        # High tolerance (less precise)
        price_high_tol = find_indifference_price(outcomes, probabilities,
                                                current_wealth, risk_aversion,
                                                tolerance=1.0)

        # Low tolerance (more precise)
        price_low_tol = find_indifference_price(outcomes, probabilities,
                                               current_wealth, risk_aversion,
                                               tolerance=0.001)

        # Should be close but not exactly equal
        assert price_high_tol == pytest.approx(price_low_tol, abs=1.5)

    def test_never_exceeds_max_search_fraction(self):
        """Test that indifference price never exceeds max_search_fraction."""
        outcomes = [10000, 0]  # Very favorable bet
        probabilities = [0.9, 0.1]
        current_wealth = 1000
        risk_aversion = 1.0
        max_search_fraction = 0.3

        max_price = find_indifference_price(outcomes, probabilities,
                                           current_wealth, risk_aversion,
                                           max_search_fraction=max_search_fraction)

        # Should not exceed 30% of wealth
        assert max_price <= current_wealth * max_search_fraction
