"""
Tests for calculate_max_entry_price() method on strategies.

These tests verify that all strategies can calculate maximum
entry prices for one-time gambles using reasonable approaches.
"""

import pytest

from keeks.binary_strategies import KellyCriterion
from keeks.binary_strategies.kelly import DrawdownAdjustedKelly, FractionalKellyCriterion
from keeks.binary_strategies.simple import (
    CPPIStrategy,
    DynamicBankrollManagement,
    FixedFractionStrategy,
    MertonShare,
    NaiveStrategy,
    OptimalF,
)


class TestKellyCriterionEntryPrice:
    """Tests for KellyCriterion calculate_max_entry_price method."""

    def test_basic_functionality(self):
        """Test basic entry price calculation for Kelly."""
        strategy = KellyCriterion(payoff=1.0, loss=1.0, transaction_cost=0.0)

        # Simple 50/50 bet: win $100 or lose $100
        outcomes = [100, -100]
        probabilities = [0.5, 0.5]

        max_price = strategy.calculate_max_entry_price(
            outcomes=outcomes,
            probabilities=probabilities,
            current_wealth=1000
        )

        # For fair bet, should pay very little
        assert max_price >= 0
        assert max_price < 1.0

    def test_positive_ev_bet(self):
        """Test that Kelly is willing to pay for positive EV bet."""
        strategy = KellyCriterion(payoff=1.0, loss=1.0, transaction_cost=0.0)

        # Positive EV: 60% chance to win $100, 40% chance to lose $50
        outcomes = [100, -50]
        probabilities = [0.6, 0.4]

        max_price = strategy.calculate_max_entry_price(
            outcomes=outcomes,
            probabilities=probabilities,
            current_wealth=5000
        )

        # Should be willing to pay something positive
        assert max_price > 0
        # But less than expected value (60*100 - 40*50 = 40)
        assert max_price < 40

    def test_st_petersburg_bounded(self):
        """Test that Kelly gives finite price for St. Petersburg paradox."""
        strategy = KellyCriterion(payoff=1.0, loss=1.0, transaction_cost=0.0)

        # St. Petersburg outcomes
        max_flips = 20
        outcomes = [2**n for n in range(1, max_flips + 1)]
        probabilities = [(0.5)**n for n in range(1, max_flips + 1)]

        max_price = strategy.calculate_max_entry_price(
            outcomes=outcomes,
            probabilities=probabilities,
            current_wealth=10000
        )

        # Should be finite despite infinite EV
        assert max_price > 0
        assert max_price < 100  # Should be modest (around $14)

    def test_wealth_effect(self):
        """Test that wealthier agents pay more (absolute terms)."""
        strategy = KellyCriterion(payoff=1.0, loss=1.0, transaction_cost=0.0)

        outcomes = [1000, -500]
        probabilities = [0.5, 0.5]

        price_poor = strategy.calculate_max_entry_price(
            outcomes=outcomes,
            probabilities=probabilities,
            current_wealth=5000
        )

        price_rich = strategy.calculate_max_entry_price(
            outcomes=outcomes,
            probabilities=probabilities,
            current_wealth=50000
        )

        # Wealthier agent should pay more in absolute terms
        assert price_rich > price_poor


class TestMertonShareEntryPrice:
    """Tests for MertonShare calculate_max_entry_price method."""

    def test_basic_functionality(self):
        """Test basic entry price calculation for Merton."""
        strategy = MertonShare(
            payoff=1.0, loss=1.0, transaction_cost=0.0, risk_aversion=2.0
        )

        outcomes = [100, -50]
        probabilities = [0.6, 0.4]

        max_price = strategy.calculate_max_entry_price(
            outcomes=outcomes,
            probabilities=probabilities,
            current_wealth=5000
        )

        # Should be willing to pay something positive
        assert max_price > 0

    def test_risk_aversion_effect(self):
        """Test that higher risk aversion leads to lower entry price."""
        outcomes = [200, -100]
        probabilities = [0.55, 0.45]
        current_wealth = 10000

        # Low risk aversion (γ=1.5)
        strategy_low = MertonShare(
            payoff=1.0, loss=1.0, transaction_cost=0.0, risk_aversion=1.5
        )
        price_low = strategy_low.calculate_max_entry_price(
            outcomes, probabilities, current_wealth
        )

        # High risk aversion (γ=5.0)
        strategy_high = MertonShare(
            payoff=1.0, loss=1.0, transaction_cost=0.0, risk_aversion=5.0
        )
        price_high = strategy_high.calculate_max_entry_price(
            outcomes, probabilities, current_wealth
        )

        # Higher risk aversion should pay less
        assert price_high < price_low

    def test_kelly_equivalence(self):
        """Test that Merton with γ=1.0 matches Kelly Criterion."""
        outcomes = [100, -50]
        probabilities = [0.6, 0.4]
        current_wealth = 5000

        kelly = KellyCriterion(payoff=1.0, loss=1.0, transaction_cost=0.0)
        price_kelly = kelly.calculate_max_entry_price(
            outcomes, probabilities, current_wealth
        )

        merton = MertonShare(
            payoff=1.0, loss=1.0, transaction_cost=0.0, risk_aversion=1.0
        )
        price_merton = merton.calculate_max_entry_price(
            outcomes, probabilities, current_wealth
        )

        # Should be very close (within tolerance)
        assert price_kelly == pytest.approx(price_merton, abs=0.02)

    def test_st_petersburg_with_different_gammas(self):
        """Test St. Petersburg paradox with different risk aversions."""
        max_flips = 20
        outcomes = [2**n for n in range(1, max_flips + 1)]
        probabilities = [(0.5)**n for n in range(1, max_flips + 1)]
        current_wealth = 10000

        prices = {}
        for gamma in [1.0, 2.0, 3.0, 5.0]:
            strategy = MertonShare(
                payoff=1.0, loss=1.0, transaction_cost=0.0, risk_aversion=gamma
            )
            prices[gamma] = strategy.calculate_max_entry_price(
                outcomes, probabilities, current_wealth
            )

        # Prices should decrease with increasing risk aversion
        assert prices[1.0] > prices[2.0] > prices[3.0] > prices[5.0]
        # All should be finite and modest
        for price in prices.values():
            assert 0 < price < 50


class TestOtherStrategiesEntryPrice:
    """Test entry price calculations for other strategies."""

    def test_fractional_kelly_scales_price(self):
        """Test that FractionalKelly scales down Kelly's price."""
        outcomes = [100, -50]
        probabilities = [0.6, 0.4]
        current_wealth = 5000

        kelly = KellyCriterion(payoff=1.0, loss=1.0, transaction_cost=0.0)
        kelly_price = kelly.calculate_max_entry_price(outcomes, probabilities, current_wealth)

        half_kelly = FractionalKellyCriterion(
            payoff=1.0, loss=1.0, transaction_cost=0.0, fraction=0.5
        )
        half_kelly_price = half_kelly.calculate_max_entry_price(outcomes, probabilities, current_wealth)

        # Half Kelly should pay approximately half of Kelly
        assert half_kelly_price == pytest.approx(0.5 * kelly_price, rel=0.01)

    def test_drawdown_adjusted_kelly_reduces_price(self):
        """Test that DrawdownAdjustedKelly reduces entry price."""
        outcomes = [100, -50]
        probabilities = [0.6, 0.4]
        current_wealth = 5000

        kelly = KellyCriterion(payoff=1.0, loss=1.0, transaction_cost=0.0)
        kelly_price = kelly.calculate_max_entry_price(outcomes, probabilities, current_wealth)

        drawdown_kelly = DrawdownAdjustedKelly(
            payoff=1.0, loss=1.0, transaction_cost=0.0, max_acceptable_drawdown=0.2
        )
        drawdown_price = drawdown_kelly.calculate_max_entry_price(outcomes, probabilities, current_wealth)

        # Drawdown adjusted should pay less than full Kelly
        assert drawdown_price < kelly_price
        assert drawdown_price > 0

    def test_optimal_f_uses_log_utility(self):
        """Test that OptimalF uses log utility (like Kelly)."""
        strategy = OptimalF(
            payoff=1.0, loss=1.0, transaction_cost=0.0, win_rate=0.55
        )

        outcomes = [100, -50]
        probabilities = [0.6, 0.4]
        current_wealth = 5000

        max_price = strategy.calculate_max_entry_price(
            outcomes, probabilities, current_wealth
        )

        # Should be positive and reasonable
        assert max_price > 0
        assert max_price < current_wealth * 0.5

    def test_naive_pays_expected_value(self):
        """Test that NaiveStrategy pays expected value."""
        strategy = NaiveStrategy(payoff=1.0, loss=1.0, transaction_cost=0.0)

        outcomes = [100, -50]
        probabilities = [0.6, 0.4]
        current_wealth = 5000

        max_price = strategy.calculate_max_entry_price(
            outcomes, probabilities, current_wealth
        )

        # Expected value = 0.6 * 100 + 0.4 * (-50) = 60 - 20 = 40
        expected_value = 40.0
        assert max_price == pytest.approx(expected_value)

    def test_fixed_fraction_uses_fraction(self):
        """Test that FixedFractionStrategy pays a fixed fraction of wealth."""
        strategy = FixedFractionStrategy(
            payoff=1.0, loss=1.0, transaction_cost=0.0, fraction=0.05
        )

        outcomes = [100, -50]
        probabilities = [0.6, 0.4]
        current_wealth = 5000

        max_price = strategy.calculate_max_entry_price(
            outcomes, probabilities, current_wealth
        )

        # Should pay 5% of wealth = $250
        assert max_price == pytest.approx(0.05 * current_wealth)

    def test_cppi_uses_cushion(self):
        """Test that CPPIStrategy pays based on cushion above floor."""
        strategy = CPPIStrategy(
            payoff=1.0,
            loss=1.0,
            transaction_cost=0.0,
            floor_fraction=0.5,
            multiplier=2.0,
            initial_bankroll=1000
        )

        outcomes = [100, -50]
        probabilities = [0.6, 0.4]
        current_wealth = 5000

        max_price = strategy.calculate_max_entry_price(
            outcomes, probabilities, current_wealth
        )

        # Floor is 50% of initial ($500), cushion is $4500
        # But max_search_fraction limits to 50% of current wealth
        expected = min(4500, 0.5 * current_wealth)
        assert max_price == pytest.approx(expected)

    def test_dynamic_uses_base_fraction(self):
        """Test that DynamicBankrollManagement uses base fraction."""
        strategy = DynamicBankrollManagement(
            payoff=1.0,
            loss=1.0,
            transaction_cost=0.0,
            base_fraction=0.1,
            window_size=10
        )

        outcomes = [100, -50]
        probabilities = [0.6, 0.4]
        current_wealth = 5000

        max_price = strategy.calculate_max_entry_price(
            outcomes, probabilities, current_wealth
        )

        # Should use base fraction (10%) since no history
        assert max_price == pytest.approx(0.1 * current_wealth)


class TestEntryPriceParameters:
    """Test parameter handling for calculate_max_entry_price."""

    def test_custom_tolerance(self):
        """Test that custom tolerance parameter works."""
        strategy = KellyCriterion(payoff=1.0, loss=1.0, transaction_cost=0.0)

        outcomes = [100, -50]
        probabilities = [0.6, 0.4]
        current_wealth = 5000

        # High tolerance (less precise)
        price_high_tol = strategy.calculate_max_entry_price(
            outcomes, probabilities, current_wealth, tolerance=1.0
        )

        # Low tolerance (more precise)
        price_low_tol = strategy.calculate_max_entry_price(
            outcomes, probabilities, current_wealth, tolerance=0.001
        )

        # Should be close but not exactly equal
        assert price_high_tol == pytest.approx(price_low_tol, abs=1.5)

    def test_custom_max_search_fraction(self):
        """Test that max_search_fraction parameter is respected."""
        strategy = KellyCriterion(payoff=1.0, loss=1.0, transaction_cost=0.0)

        outcomes = [10000, 0]  # Very favorable bet
        probabilities = [0.9, 0.1]
        current_wealth = 1000

        max_price = strategy.calculate_max_entry_price(
            outcomes, probabilities, current_wealth, max_search_fraction=0.3
        )

        # Should not exceed 30% of wealth
        assert max_price <= current_wealth * 0.3
