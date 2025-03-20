import random

import pytest

from keeks.bankroll import BankRoll
from keeks.binary_strategies import DynamicBankrollManagement

random.seed(42)


def test_basic_functionality():
    """Test that DynamicBankrollManagement correctly adjusts bet sizes based on performance."""
    strategy = DynamicBankrollManagement(
        base_fraction=0.1,
        payoff=1,
        loss=1,
        transaction_cost=0,
        window_size=10,
        max_fraction=0.2,
        min_fraction=0.05,
    )

    # Initial bet should be base_fraction
    assert strategy.evaluate(0.6, 1000) == pytest.approx(0.1)

    # Record some wins
    for _ in range(5):
        strategy.record_result(True)

    # After wins, bet size should increase but stay below max_fraction
    assert 0.1 < strategy.evaluate(0.6, 1000) <= 0.2

    # Record some losses
    for _ in range(10):
        strategy.record_result(False)

    # After many losses, bet size should decrease but stay above min_fraction
    assert 0.05 <= strategy.evaluate(0.6, 1000) < 0.1


def test_probability_effect():
    """Test that the strategy adjusts bet sizes based on probability."""
    strategy = DynamicBankrollManagement(
        base_fraction=0.1,
        payoff=1,
        loss=1,
        transaction_cost=0,
        window_size=10,
        max_fraction=0.2,
        min_fraction=0.05,
    )

    # Record some mixed results first
    strategy.record_result(True)
    strategy.record_result(False)
    strategy.record_result(True)

    # Should bet more with higher probability
    high_prob_bet = strategy.evaluate(0.8, 1000)  # 80% probability
    low_prob_bet = strategy.evaluate(0.6, 1000)  # 60% probability
    assert high_prob_bet > low_prob_bet


def test_window_size_effect():
    """Test how different window sizes affect the adjustment speed."""
    # Small window size (quick adjustments)
    strategy_small = DynamicBankrollManagement(
        base_fraction=0.1,
        payoff=1,
        loss=1,
        transaction_cost=0,
        window_size=5,
        max_fraction=0.2,
        min_fraction=0.05,
    )

    # Large window size (slower adjustments)
    strategy_large = DynamicBankrollManagement(
        base_fraction=0.1,
        payoff=1,
        loss=1,
        transaction_cost=0,
        window_size=20,
        max_fraction=0.2,
        min_fraction=0.05,
    )

    # Record some wins for both
    for _ in range(3):
        strategy_small.record_result(True)
        strategy_large.record_result(True)

    # Small window should adjust more quickly
    small_bet = strategy_small.evaluate(0.6, 1000)
    large_bet = strategy_large.evaluate(0.6, 1000)
    assert small_bet > large_bet


def test_invalid_parameters():
    """Test that invalid parameters raise appropriate exceptions."""
    # Base fraction must be between 0 and 1
    with pytest.raises(ValueError, match="Base fraction must be between 0 and 1"):
        DynamicBankrollManagement(
            base_fraction=-0.1,
            payoff=1,
            loss=1,
            transaction_cost=0,
            window_size=10,
            max_fraction=0.2,
            min_fraction=0.05,
        )

    with pytest.raises(ValueError, match="Base fraction must be between 0 and 1"):
        DynamicBankrollManagement(
            base_fraction=1.1,
            payoff=1,
            loss=1,
            transaction_cost=0,
            window_size=10,
            max_fraction=0.2,
            min_fraction=0.05,
        )

    # Window size must be positive
    with pytest.raises(ValueError, match="Window size must be positive"):
        DynamicBankrollManagement(
            base_fraction=0.1,
            payoff=1,
            loss=1,
            transaction_cost=0,
            window_size=0,
            max_fraction=0.2,
            min_fraction=0.05,
        )


def test_simulation():
    """Test the strategy in a simulation with varying performance."""
    payoff = 1
    loss = 1
    transaction_cost = 0.01
    probability = 0.55  # Slight edge
    trials = 300
    initial_bankroll = 1000

    # Initialize bankroll and strategy
    bankroll = BankRoll(
        initial_funds=initial_bankroll, percent_bettable=0.5, max_draw_down=None
    )
    strategy = DynamicBankrollManagement(
        base_fraction=0.1,
        payoff=payoff,
        loss=loss,
        transaction_cost=transaction_cost,
        window_size=10,
        max_fraction=0.2,
        min_fraction=0.05,
    )

    # Track bankroll history
    bankroll_history = [initial_bankroll]

    for _ in range(trials):
        # Get current capital
        current_bankroll = bankroll.total_funds

        # Calculate bet size as a proportion
        bet_proportion = strategy.evaluate(probability, current_bankroll)
        bet_amount = bet_proportion * current_bankroll

        # Simulate the bet outcome
        if random.random() < probability:
            # Win
            bankroll.add_funds(bet_amount * (payoff - transaction_cost))
            strategy.record_result(
                True, bet_amount * (payoff - transaction_cost) / current_bankroll
            )
        else:
            # Loss
            bankroll.remove_funds(bet_amount * (loss + transaction_cost))
            strategy.record_result(
                False, -bet_amount * (loss + transaction_cost) / current_bankroll
            )

        # Record bankroll history
        bankroll_history.append(bankroll.total_funds)

    # Should have grown with positive edge over many trials
    assert bankroll.total_funds > initial_bankroll * 0.9  # Allow for some variance

    # Check that we have history to compare
    assert len(bankroll_history) > 1


def test_streak_factor():
    """Test that streak factor increases with winning streak and decreases with losing streak."""
    strategy = DynamicBankrollManagement(
        base_fraction=0.1, payoff=1, loss=1, transaction_cost=0
    )

    # Initial streak factor should be 1.0
    assert strategy.get_streak_factor() == pytest.approx(1.0)

    # Record wins
    for _ in range(5):
        strategy.record_result(True)

    # Streak factor should increase
    assert strategy.get_streak_factor() > 1.0

    # Record losses
    for _ in range(10):
        strategy.record_result(False)

    # Streak factor should decrease
    assert strategy.get_streak_factor() < 1.0


def test_volatility_factor():
    """Test that volatility factor decreases with increasing volatility."""
    strategy = DynamicBankrollManagement(
        base_fraction=0.1, payoff=1, loss=1, transaction_cost=0
    )

    # Initial volatility factor should be 1.0 (no history)
    assert strategy.get_volatility_factor() == pytest.approx(1.0)

    # Record alternating wins and losses (high volatility)
    for i in range(10):
        strategy.record_result(i % 2 == 0)

    # Volatility factor should decrease
    assert strategy.get_volatility_factor() < 1.0

    # Reset strategy
    strategy = DynamicBankrollManagement(
        base_fraction=0.1, payoff=1, loss=1, transaction_cost=0
    )

    # Record all wins (low volatility)
    for _ in range(10):
        strategy.record_result(True)

    # Volatility factor should be higher
    assert strategy.get_volatility_factor() > 0.8


def test_drawdown_factor():
    """Test that drawdown factor decreases as drawdown increases."""
    strategy = DynamicBankrollManagement(
        base_fraction=0.1, payoff=1, loss=1, transaction_cost=0
    )

    # Initial drawdown factor should be 1.0
    assert strategy.get_drawdown_factor() == pytest.approx(1.0)

    # Record some losses to create drawdown
    initial_bankroll = 1000
    current_bankroll = initial_bankroll

    # First evaluation to set initial bankroll
    strategy.evaluate(0.6, current_bankroll)

    # Simulate 20% drawdown
    current_bankroll = 800
    strategy.evaluate(0.6, current_bankroll)

    # Drawdown factor should be reduced but above 0.5
    assert 0.5 < strategy.get_drawdown_factor() < 1.0


def test_combined_adjustments():
    """Test how multiple factors combine to adjust the base fraction."""
    strategy = DynamicBankrollManagement(
        base_fraction=0.1,
        payoff=1,
        loss=1,
        transaction_cost=0,
        max_fraction=0.3,  # Cap at 30%
        min_fraction=0.01,  # Floor at 1%
    )

    # Record a winning streak with low volatility
    initial_bankroll = 1000
    current_bankroll = initial_bankroll

    # First evaluation to set initial bankroll
    strategy.evaluate(0.6, current_bankroll)

    for _ in range(5):
        strategy.record_result(True, 0.1)  # 10% gain each time
        current_bankroll *= 1.1  # Simulate bankroll increase
        strategy.evaluate(0.6, current_bankroll)

    # With winning streak, low volatility, no drawdown, and decent probability
    # bet size should increase but stay within bounds
    final_bet = strategy.evaluate(0.7, current_bankroll)
    assert 0.1 < final_bet <= 0.3


def test_min_max_bounds():
    """Test that bet size is properly bounded by min and max fractions."""
    strategy = DynamicBankrollManagement(
        base_fraction=0.1,
        payoff=1,
        loss=1,
        transaction_cost=0,
        max_fraction=0.15,
        min_fraction=0.05,
    )

    # Record many wins to try to push above max_fraction
    for _ in range(10):
        strategy.record_result(True, 0.2)

    # Should be capped at max_fraction
    assert strategy.evaluate(0.6, 1000) <= strategy.max_fraction

    # Record many losses to try to push below min_fraction
    for _ in range(10):
        strategy.record_result(False, -0.2)

    # Should be floored at min_fraction
    assert strategy.evaluate(0.6, 1000) >= strategy.min_fraction


def test_window_size_limiting():
    """Test that history is limited to the window size."""
    window_size = 3
    strategy = DynamicBankrollManagement(
        base_fraction=0.1, payoff=1, loss=1, transaction_cost=0, window_size=window_size
    )

    # Record more results than window size
    for _i in range(5):
        strategy.record_result(True)

    # Should only keep last window_size results
    assert len(strategy.results) == window_size
