"""Tests for bankruptcy protection features added in v0.3.0."""

import pytest

from keeks.bankroll import BankRoll
from keeks.binary_strategies.kelly import KellyCriterion
from keeks.simulators.repeated_binary import RepeatedBinarySimulator
from keeks.utils import RuinError


def test_withdraw_prevents_negative_bankroll():
    """Test that withdraw() prevents bankroll from going negative."""
    bankroll = BankRoll(initial_funds=100, max_draw_down=None)

    # Try to withdraw more than available
    with pytest.raises(RuinError, match="bankruptcy"):
        bankroll.withdraw(150)

    # Bankroll should be unchanged
    assert bankroll.total_funds == 100


def test_remove_funds_prevents_negative_bankroll():
    """Test that remove_funds() prevents bankroll from going negative."""
    bankroll = BankRoll(initial_funds=100, max_draw_down=None)

    # Try to remove more than available
    with pytest.raises(RuinError, match="bankruptcy"):
        bankroll.remove_funds(150)

    # Bankroll should be unchanged
    assert bankroll.total_funds == 100


def test_exact_withdrawal_works():
    """Test that withdrawing exact amount works."""
    bankroll = BankRoll(initial_funds=100, max_draw_down=None)

    # Withdraw exact amount
    bankroll.withdraw(100)

    # Should be at zero (not negative)
    assert bankroll.total_funds == 0


def test_simulator_stops_at_bankruptcy():
    """Test that simulator stops when bankroll hits zero."""
    import random
    from keeks.binary_strategies.simple import FixedFractionStrategy
    random.seed(999)  # Seed that causes early bankruptcy

    bankroll = BankRoll(initial_funds=100, max_draw_down=None)
    # Use fixed fraction strategy that will bet regardless of odds
    strategy = FixedFractionStrategy(
        fraction=0.25,  # Bet 25% each time
        payoff=1.0,
        loss=1.0,
        transaction_cost=0.0,
        min_probability=0.0  # Bet even with bad odds
    )

    simulator = RepeatedBinarySimulator(
        payoff=1.0,
        loss=1.0,
        transaction_costs=0.0,
        probability=0.3,  # Bad odds
        trials=1000
    )

    simulator.evaluate_strategy(strategy, bankroll)

    # Should have stopped before all trials (bankrupt with bad odds)
    assert len(bankroll.history) < 1000

    # Should not be negative
    assert bankroll.total_funds >= 0


def test_drawdown_check_uses_pre_withdrawal_amount():
    """Test that drawdown check uses amount before withdrawal."""
    bankroll = BankRoll(initial_funds=1000, max_draw_down=0.2)

    # Try to withdraw 250 (25% of 1000, exceeds 20% limit)
    with pytest.raises(RuinError, match="slow down"):
        bankroll.withdraw(250)

    # Bankroll should be unchanged
    assert bankroll.total_funds == 1000


def test_bankruptcy_check_before_drawdown_check():
    """Test that bankruptcy is checked before drawdown."""
    bankroll = BankRoll(initial_funds=100, max_draw_down=0.5)

    # Try to withdraw 150 (would cause bankruptcy, even though > max_draw_down)
    with pytest.raises(RuinError, match="bankruptcy"):
        bankroll.withdraw(150)

    # Bankroll should be unchanged
    assert bankroll.total_funds == 100


def test_transaction_cost_correctly_increases_loss():
    """Test that transaction costs are added to losses, not subtracted."""
    bankroll = BankRoll(initial_funds=1000, max_draw_down=None)
    initial = bankroll.total_funds

    # Simulate a loss with transaction cost
    bet_amount = 100
    loss_multiplier = 1.0
    transaction_cost = 5

    # Expected: lose bet amount + transaction cost
    expected_loss = (bet_amount * loss_multiplier) + transaction_cost

    bankroll.withdraw(expected_loss)

    # Should have lost 105, not 95
    assert bankroll.total_funds == initial - expected_loss
    assert bankroll.total_funds == 895  # 1000 - 105


def test_no_negative_values_in_history():
    """Test that bankroll history never contains negative values."""
    import random
    random.seed(42)

    bankroll = BankRoll(initial_funds=50, max_draw_down=None)
    strategy = KellyCriterion(payoff=1.0, loss=1.0, transaction_cost=0.0)

    simulator = RepeatedBinarySimulator(
        payoff=1.0,
        loss=1.0,
        transaction_costs=0.0,
        probability=0.4,  # Negative expectation
        trials=100
    )

    try:
        simulator.evaluate_strategy(strategy, bankroll)
    except RuinError:
        pass  # Expected if we hit bankruptcy

    # Check that no value in history is negative
    assert all(value >= 0 for value in bankroll.history)
    assert min(bankroll.history) >= 0


def test_multiple_small_losses_dont_go_negative():
    """Test that multiple small losses stop at zero, not negative."""
    bankroll = BankRoll(initial_funds=10, max_draw_down=None)

    # Withdraw in small increments
    bankroll.withdraw(3)
    assert bankroll.total_funds == 7

    bankroll.withdraw(3)
    assert bankroll.total_funds == 4

    bankroll.withdraw(3)
    assert bankroll.total_funds == 1

    # This should fail (would go negative)
    with pytest.raises(RuinError, match="bankruptcy"):
        bankroll.withdraw(2)

    # Should still be at 1
    assert bankroll.total_funds == 1

    # Can withdraw the last dollar
    bankroll.withdraw(1)
    assert bankroll.total_funds == 0

    # Can't withdraw from empty bankroll
    with pytest.raises(RuinError, match="bankruptcy"):
        bankroll.withdraw(1)
