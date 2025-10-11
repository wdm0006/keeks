import random

import pytest

from keeks.bankroll import BankRoll
from keeks.binary_strategies.simple import MertonShare
from keeks.simulators.repeated_binary import RepeatedBinarySimulator

random.seed(42)


def test_basic_functionality():
    """Test that MertonShare correctly calculates optimal bet sizes."""
    strategy = MertonShare(payoff=1.0, loss=1.0, transaction_cost=0, risk_aversion=2.0)

    # With 60% probability and 1:1 payoff ratio
    # Expected return = 0.6 * 1.0 - 0.4 * 1.0 = 0.2
    # Variance = 0.6 * 1.0^2 + 0.4 * 1.0^2 - (0.6 - 0.4)^2 = 1.0 - 0.04 = 0.96
    # Merton fraction = 0.2 / (2.0 * 0.96) = 0.2 / 1.92 ≈ 0.1042
    result = strategy.evaluate(0.6, 1000)
    assert result == pytest.approx(0.1042, abs=0.001)


def test_risk_aversion_effect():
    """Test how different risk aversion levels affect the bet size."""
    # Low risk aversion (aggressive)
    strategy_low = MertonShare(
        payoff=1.0, loss=1.0, transaction_cost=0, risk_aversion=1.0
    )

    # Moderate risk aversion
    strategy_moderate = MertonShare(
        payoff=1.0, loss=1.0, transaction_cost=0, risk_aversion=2.0
    )

    # High risk aversion (conservative)
    strategy_high = MertonShare(
        payoff=1.0, loss=1.0, transaction_cost=0, risk_aversion=5.0
    )

    # Higher risk aversion should result in smaller bet sizes
    bet_low = strategy_low.evaluate(0.6, 1000)
    bet_moderate = strategy_moderate.evaluate(0.6, 1000)
    bet_high = strategy_high.evaluate(0.6, 1000)

    assert bet_low > bet_moderate > bet_high
    assert bet_low == pytest.approx(2 * bet_moderate)  # Linear relationship with gamma
    assert bet_low == pytest.approx(5 * bet_high)


def test_min_probability_threshold():
    """Test that the strategy respects the minimum probability threshold."""
    strategy = MertonShare(
        payoff=1.0, loss=1.0, transaction_cost=0, min_probability=0.55
    )

    # Should bet when probability >= min_probability
    assert strategy.evaluate(0.6, 1000) > 0

    # Should not bet when probability < min_probability
    assert strategy.evaluate(0.54, 1000) == pytest.approx(0.0)


def test_transaction_costs():
    """Test how transaction costs affect the Merton Share."""
    # No transaction costs
    strategy_no_cost = MertonShare(payoff=1.0, loss=1.0, transaction_cost=0)

    # With transaction costs
    strategy_with_cost = MertonShare(payoff=1.0, loss=1.0, transaction_cost=0.05)

    # Transaction costs should reduce bet size
    assert strategy_with_cost.evaluate(0.6, 1000) < strategy_no_cost.evaluate(
        0.6, 1000
    )

    # With large transaction costs that make betting unprofitable
    strategy_large_cost = MertonShare(payoff=1.0, loss=1.0, transaction_cost=0.5)

    # Should not bet when transaction costs make betting unprofitable
    assert strategy_large_cost.evaluate(0.6, 1000) == pytest.approx(0.0)


def test_max_fraction_constraint():
    """Test that the max_fraction parameter caps bet sizes."""
    strategy = MertonShare(
        payoff=2.0,
        loss=1.0,
        transaction_cost=0,
        risk_aversion=0.5,
        max_fraction=0.25,
    )

    # Even with very favorable odds, should not exceed max_fraction
    assert strategy.evaluate(0.9, 1000) <= 0.25


def test_invalid_parameters():
    """Test that invalid parameters raise appropriate exceptions."""
    # Risk aversion must be positive
    with pytest.raises(ValueError, match="Risk aversion must be greater than 0"):
        MertonShare(payoff=1.0, loss=1.0, transaction_cost=0, risk_aversion=0)

    with pytest.raises(ValueError, match="Risk aversion must be greater than 0"):
        MertonShare(payoff=1.0, loss=1.0, transaction_cost=0, risk_aversion=-1.0)

    # Min probability must be between 0 and 1
    with pytest.raises(ValueError, match="Minimum probability must be between 0 and 1"):
        MertonShare(
            payoff=1.0, loss=1.0, transaction_cost=0, min_probability=1.5
        )

    # Max fraction must be between 0 and 1
    with pytest.raises(ValueError, match="Maximum fraction must be between 0 and 1"):
        MertonShare(payoff=1.0, loss=1.0, transaction_cost=0, max_fraction=1.5)

    with pytest.raises(ValueError, match="Maximum fraction must be between 0 and 1"):
        MertonShare(payoff=1.0, loss=1.0, transaction_cost=0, max_fraction=0)

    # Payoff must be positive (inherited from BaseStrategy)
    with pytest.raises(ValueError, match="Payoff must be greater than 0"):
        MertonShare(payoff=0, loss=1.0, transaction_cost=0)

    # Loss must be non-negative (inherited from BaseStrategy)
    with pytest.raises(ValueError, match="Loss must be non-negative"):
        MertonShare(payoff=1.0, loss=-1.0, transaction_cost=0)


def test_zero_probability():
    """Test behavior with 0% probability of winning."""
    strategy = MertonShare(
        payoff=2.0, loss=1.0, transaction_cost=0, min_probability=0.0
    )

    # 0% chance of winning should result in no bet
    assert strategy.evaluate(0.0, 1000) == pytest.approx(0.0)


def test_one_probability():
    """Test behavior with 100% probability of winning."""
    strategy = MertonShare(
        payoff=2.0, loss=1.0, transaction_cost=0, risk_aversion=1.0, max_fraction=1.0
    )

    # 100% chance of winning has zero variance (no uncertainty)
    # The strategy correctly returns 0 since variance = 0 creates division issues
    # In practice, this edge case is handled by the variance <= 0 check
    result = strategy.evaluate(1.0, 1000)
    assert result == pytest.approx(0.0)

    # With probability very close to 100% (but not exactly 1.0), we get a bet
    result_high = strategy.evaluate(0.999, 1000)
    assert result_high > 0


def test_even_odds():
    """Test with even odds (1:1 payoff)."""
    strategy = MertonShare(
        payoff=1.0, loss=1.0, transaction_cost=0, risk_aversion=2.0
    )
    current_bankroll = 1000

    # 60% chance of winning
    portion = strategy.evaluate(0.6, current_bankroll)
    assert portion > 0
    assert portion < 0.2  # Should be less than Kelly (which would be 0.2)

    # 50% chance of winning (break-even)
    portion = strategy.evaluate(0.5, current_bankroll)
    assert portion == pytest.approx(0.0)

    # 40% chance of winning (negative EV)
    portion = strategy.evaluate(0.4, current_bankroll)
    assert portion == pytest.approx(0.0)


def test_favorable_odds():
    """Test with favorable payoff ratio."""
    strategy = MertonShare(
        payoff=2.0, loss=1.0, transaction_cost=0, risk_aversion=2.0
    )
    current_bankroll = 1000

    # 60% chance of winning with 2:1 payoff
    # Expected return = 0.6 * 2.0 - 0.4 * 1.0 = 0.8
    # Variance = 0.6 * 4 + 0.4 * 1 - (1.2 - 0.4)^2 = 2.4 + 0.4 - 0.64 = 2.16
    # Merton fraction = 0.8 / (2.0 * 2.16) = 0.8 / 4.32 ≈ 0.185
    result = strategy.evaluate(0.6, current_bankroll)
    assert result == pytest.approx(0.185, abs=0.001)


def test_simulation_with_positive_edge():
    """Test the strategy in a simulation with positive expected value."""
    payoff = 1.0
    loss = 1.0
    transaction_cost = 0.01
    probability = 0.55  # Slight edge
    trials = 1000
    initial_bankroll = 1000

    # Initialize bankroll and strategy
    bankroll = BankRoll(
        initial_funds=initial_bankroll, percent_bettable=1.0, max_draw_down=None
    )
    strategy = MertonShare(
        payoff=payoff, loss=loss, transaction_cost=transaction_cost, risk_aversion=2.0
    )

    # Set up simulator
    simulator = RepeatedBinarySimulator(
        payoff=payoff,
        loss=loss,
        transaction_costs=transaction_cost,
        probability=probability,
        trials=trials,
    )

    # Run simulation
    simulator.evaluate_strategy(strategy, bankroll)

    # With positive edge and many trials, bankroll should grow
    assert bankroll.total_funds > initial_bankroll


def test_comparison_with_different_risk_aversions():
    """Compare performance across different risk aversion levels."""
    payoff = 1.0
    loss = 1.0
    transaction_cost = 0.01
    probability = 0.6
    trials = 500

    results = {}
    for gamma in [1.0, 2.0, 3.0, 5.0]:
        random.seed(42)  # Reset seed for fair comparison
        bankroll = BankRoll(initial_funds=1000, percent_bettable=1.0, max_draw_down=None)
        strategy = MertonShare(
            payoff=payoff,
            loss=loss,
            transaction_cost=transaction_cost,
            risk_aversion=gamma,
        )
        simulator = RepeatedBinarySimulator(
            payoff=payoff,
            loss=loss,
            transaction_costs=transaction_cost,
            probability=probability,
            trials=trials,
        )
        simulator.evaluate_strategy(strategy, bankroll)
        results[gamma] = bankroll.total_funds

    # All should grow with positive edge
    for gamma, final_bankroll in results.items():
        assert final_bankroll > 1000, f"Risk aversion {gamma} failed to grow bankroll"


def test_variance_calculation():
    """Test that variance is calculated correctly for binary outcomes."""
    strategy = MertonShare(payoff=2.0, loss=1.0, transaction_cost=0, risk_aversion=1.0)

    # For p=0.6, payoff=2, loss=1:
    # Mean = 0.6 * 2 - 0.4 * 1 = 0.8
    # Mean squared = 0.6 * 4 + 0.4 * 1 = 2.8
    # Variance = 2.8 - 0.8^2 = 2.8 - 0.64 = 2.16
    # Expected return = 0.8
    # Merton fraction = 0.8 / (1.0 * 2.16) = 0.37037...

    result = strategy.evaluate(0.6, 1000)
    expected = 0.8 / 2.16
    assert result == pytest.approx(expected, abs=0.001)


def test_conservative_behavior():
    """Test that MertonShare is more conservative than Kelly for typical risk aversion."""
    from keeks.binary_strategies.kelly import KellyCriterion

    payoff = 1.0
    loss = 1.0
    transaction_cost = 0.0
    probability = 0.6

    kelly = KellyCriterion(payoff, loss, transaction_cost)
    merton = MertonShare(payoff, loss, transaction_cost, risk_aversion=2.0)

    kelly_bet = kelly.evaluate(probability, 1000)
    merton_bet = merton.evaluate(probability, 1000)

    # With risk aversion of 2.0, Merton should be more conservative than Kelly
    assert merton_bet < kelly_bet


def test_extreme_probabilities():
    """Test behavior at extreme probability values."""
    strategy = MertonShare(
        payoff=1.0, loss=1.0, transaction_cost=0, risk_aversion=2.0, min_probability=0.0
    )

    # Very low probability
    result = strategy.evaluate(0.01, 1000)
    assert result == pytest.approx(0.0)  # Negative expected value

    # Very high probability (but not 100%)
    result = strategy.evaluate(0.99, 1000)
    assert result > 0
    assert result <= 1.0


def test_get_max_safe_bet():
    """Test that max safe bet constraint is respected."""
    strategy = MertonShare(
        payoff=1.0, loss=1.0, transaction_cost=0, risk_aversion=0.1, max_fraction=1.0
    )

    # With very low risk aversion, we might calculate a large bet
    # But it should still respect max_safe_bet from BaseStrategy
    current_bankroll = 100
    result = strategy.evaluate(0.9, current_bankroll)

    # Should not bet more than what would avoid negative bankroll
    max_safe = strategy.get_max_safe_bet(current_bankroll)
    assert result <= max_safe
