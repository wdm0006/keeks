import operator
import random

import numpy as np
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
        transaction_cost_rate=0,
        window_size=10,
        max_fraction=0.2,
        min_fraction=0.05,
    )

    # Initial bet should be base_fraction
    assert strategy.evaluate(0.6, 1000) == pytest.approx(0.1)

    # Record some wins
    for _ in range(5):
        strategy.record_settlement((True,))

    # After wins, bet size should increase but stay below max_fraction
    assert 0.1 < strategy.evaluate(0.6, 1000) <= 0.2

    # Record some losses
    for _ in range(10):
        strategy.record_settlement((False,))

    # After many losses, bet size should decrease but stay above min_fraction
    assert 0.05 <= strategy.evaluate(0.6, 1000) < 0.1


def test_probability_effect():
    """Test that the strategy adjusts bet sizes based on probability."""
    strategy = DynamicBankrollManagement(
        base_fraction=0.1,
        payoff=1,
        loss=1,
        transaction_cost_rate=0,
        window_size=10,
        max_fraction=0.2,
        min_fraction=0.05,
    )

    # Record some mixed results first
    strategy.record_settlement((True,))
    strategy.record_settlement((False,))
    strategy.record_settlement((True,))

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
        transaction_cost_rate=0,
        window_size=5,
        max_fraction=0.2,
        min_fraction=0.05,
    )

    # Large window size (slower adjustments)
    strategy_large = DynamicBankrollManagement(
        base_fraction=0.1,
        payoff=1,
        loss=1,
        transaction_cost_rate=0,
        window_size=20,
        max_fraction=0.2,
        min_fraction=0.05,
    )

    # Record some wins for both
    for _ in range(3):
        strategy_small.record_settlement((True,))
        strategy_large.record_settlement((True,))

    # Small window should adjust more quickly
    small_bet = strategy_small.evaluate(0.6, 1000)
    large_bet = strategy_large.evaluate(0.6, 1000)
    assert small_bet > large_bet


@pytest.mark.parametrize("window_size", [1, np.int64(3)])
def test_window_size_accepts_positive_integer_like_values(window_size):
    strategy = DynamicBankrollManagement(
        base_fraction=0.1,
        payoff=1,
        loss=1,
        transaction_cost_rate=0,
        window_size=window_size,
    )

    assert strategy.window_size == operator.index(window_size)


@pytest.mark.parametrize(
    "window_size",
    [True, False, 1.5, 10.0, "10", 0, -1, float("nan"), float("inf"), float("-inf")],
)
def test_window_size_rejects_non_positive_integers_and_other_types(window_size):
    with pytest.raises(ValueError, match="Window size"):
        DynamicBankrollManagement(
            base_fraction=0.1,
            payoff=1,
            loss=1,
            transaction_cost_rate=0,
            window_size=window_size,
        )


def test_invalid_parameters():
    """Test that invalid parameters raise appropriate exceptions."""
    # Base fraction must be between 0 and 1
    with pytest.raises(ValueError, match="Base fraction must be between 0 and 1"):
        DynamicBankrollManagement(
            base_fraction=-0.1,
            payoff=1,
            loss=1,
            transaction_cost_rate=0,
            window_size=10,
            max_fraction=0.2,
            min_fraction=0.05,
        )

    with pytest.raises(ValueError, match="Base fraction must be between 0 and 1"):
        DynamicBankrollManagement(
            base_fraction=1.1,
            payoff=1,
            loss=1,
            transaction_cost_rate=0,
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
            transaction_cost_rate=0,
            window_size=0,
            max_fraction=0.2,
            min_fraction=0.05,
        )


def test_simulation():
    """Test the strategy in a simulation with varying performance."""
    payoff = 1
    loss = 1
    transaction_cost_rate = 0.01
    probability = 0.55  # Slight edge
    trials = 300
    initial_bankroll = 1000

    # Initialize bankroll and strategy
    bankroll = BankRoll(
        initial_funds=initial_bankroll, percent_bettable=0.5, max_transaction_loss=None
    )
    strategy = DynamicBankrollManagement(
        base_fraction=0.1,
        payoff=payoff,
        loss=loss,
        transaction_cost_rate=transaction_cost_rate,
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
            bankroll.deposit(bet_amount * (payoff - transaction_cost_rate))
            strategy.record_settlement(
                (True,),
                (bet_amount * (payoff - transaction_cost_rate) / current_bankroll,),
            )
        else:
            # Loss
            bankroll.withdraw(bet_amount * (loss + transaction_cost_rate))
            strategy.record_settlement(
                (False,),
                (-bet_amount * (loss + transaction_cost_rate) / current_bankroll,),
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
        base_fraction=0.1, payoff=1, loss=1, transaction_cost_rate=0
    )

    # Initial streak factor should be 1.0
    assert strategy.get_streak_factor() == pytest.approx(1.0)

    # Record wins
    for _ in range(5):
        strategy.record_settlement((True,))

    # Streak factor should increase
    assert strategy.get_streak_factor() > 1.0

    # Record losses
    for _ in range(10):
        strategy.record_settlement((False,))

    # Streak factor should decrease
    assert strategy.get_streak_factor() < 1.0


def test_volatility_factor():
    """Test that volatility factor decreases with increasing volatility."""
    strategy = DynamicBankrollManagement(
        base_fraction=0.1, payoff=1, loss=1, transaction_cost_rate=0
    )

    # Initial volatility factor should be 1.0 (no history)
    assert strategy.get_volatility_factor() == pytest.approx(1.0)

    # Record alternating wins and losses (high volatility)
    for i in range(10):
        strategy.record_settlement((True,) if i % 2 == 0 else (False,))

    # Volatility factor should decrease
    assert strategy.get_volatility_factor() < 1.0

    # Reset strategy
    strategy = DynamicBankrollManagement(
        base_fraction=0.1, payoff=1, loss=1, transaction_cost_rate=0
    )

    # Record all wins (low volatility)
    for _ in range(10):
        strategy.record_settlement((True,))

    # Volatility factor should be higher
    assert strategy.get_volatility_factor() > 0.8


def test_drawdown_factor():
    """Test that drawdown factor decreases as drawdown increases."""
    strategy = DynamicBankrollManagement(
        base_fraction=0.1, payoff=1, loss=1, transaction_cost_rate=0
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
        transaction_cost_rate=0,
        max_fraction=0.3,  # Cap at 30%
        min_fraction=0.01,  # Floor at 1%
    )

    # Record a winning streak with low volatility
    initial_bankroll = 1000
    current_bankroll = initial_bankroll

    # First evaluation to set initial bankroll
    strategy.evaluate(0.6, current_bankroll)

    for _ in range(5):
        strategy.record_settlement((True,), (0.1,))  # 10% gain each time
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
        transaction_cost_rate=0,
        max_fraction=0.15,
        min_fraction=0.05,
    )

    # Record many wins to try to push above max_fraction
    for _ in range(10):
        strategy.record_settlement((True,))

    # Should be capped at max_fraction
    assert strategy.evaluate(0.6, 1000) <= strategy.max_fraction

    # Record many losses to try to push below min_fraction
    for _ in range(10):
        strategy.record_settlement((False,))

    # Should be floored at min_fraction
    assert strategy.evaluate(0.6, 1000) >= strategy.min_fraction


def test_below_min_probability_returns_zero():
    """Test that no bet is placed when probability is below min_probability."""
    strategy = DynamicBankrollManagement(
        base_fraction=0.1,
        payoff=1,
        loss=1,
        transaction_cost_rate=0,
    )

    # Default min_probability is 0.5; a losing 0.2 probability must not bet.
    assert strategy.evaluate(0.2, 1000) == 0.0

    # A custom threshold is respected.
    strategy = DynamicBankrollManagement(
        base_fraction=0.1,
        payoff=1,
        loss=1,
        transaction_cost_rate=0,
        min_probability=0.7,
    )
    assert strategy.evaluate(0.6, 1000) == 0.0
    assert strategy.evaluate(0.75, 1000) > 0.0


def test_invalid_min_probability():
    """Test that an out-of-range min_probability raises a ValueError."""
    with pytest.raises(ValueError, match="Minimum probability must be between 0 and 1"):
        DynamicBankrollManagement(
            base_fraction=0.1,
            payoff=1,
            loss=1,
            transaction_cost_rate=0,
            min_probability=1.5,
        )


def test_respects_max_safe_bet():
    """Test that the bet never exceeds the ruin-safe fraction with a high loss."""
    # With a large loss multiplier the safe bet is small (1 / (loss + cost)).
    strategy = DynamicBankrollManagement(
        base_fraction=0.5,
        payoff=1,
        loss=10,
        transaction_cost_rate=0,
        max_fraction=1.0,
        min_fraction=0.05,
    )

    max_safe = strategy.get_max_safe_bet(1000)
    # min_fraction (0.05) exceeds max_safe (1/10 = 0.1)? No; ensure clamp holds.
    assert strategy.evaluate(0.6, 1000) <= max_safe


def test_window_size_limiting():
    """Test that history is limited to the window size."""
    window_size = 3
    strategy = DynamicBankrollManagement(
        base_fraction=0.1,
        payoff=1,
        loss=1,
        transaction_cost_rate=0,
        window_size=window_size,
    )

    # Record more results than window size
    for _i in range(5):
        strategy.record_settlement((True,))

    # Should only keep last window_size results
    assert len(strategy.results) == window_size
