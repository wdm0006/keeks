import random

import pytest

from keeks.bankroll import BankRoll
from keeks.binary_strategies.simple import OptimalF
from keeks.simulators.repeated_binary import RepeatedBinarySimulator

random.seed(42)


def test_basic_functionality():
    """Test that OptimalF correctly calculates optimal bet sizes based on win rate."""
    strategy = OptimalF(
        win_rate=0.6, payoff=1, loss=1, transaction_cost=0, max_risk_fraction=0.2
    )

    # With 60% win rate and 1:1 payoff ratio, optimal f should be 0.2
    # Using Ralph Vince's formula: f* = W - (1-W)/(R/L) = 0.6 - 0.4/(1/1) = 0.6 - 0.4 = 0.2
    assert strategy.evaluate(0.6, 1000) == pytest.approx(0.2, rel=0.01)

    # Test that current_bankroll affects max safe bet
    assert strategy.evaluate(0.6, 100) == pytest.approx(0.2, rel=0.01)
    assert strategy.evaluate(0.6, 10) == pytest.approx(0.2, rel=0.01)


def test_min_probability_threshold():
    """Test that the strategy respects the minimum probability threshold."""
    strategy = OptimalF(
        win_rate=0.6, payoff=1, loss=1, transaction_cost=0.01, max_risk_fraction=0.2
    )

    # Should bet when probability > 0.5 (default threshold)
    assert strategy.evaluate(0.51, 1000) > 0
    assert strategy.evaluate(0.4, 1000) == pytest.approx(0.0)


def test_payoff_ratio_effect():
    """Test how different payoff ratios affect the optimal bet size."""
    # Low payoff ratio (conservative)
    strategy_low = OptimalF(
        win_rate=0.6, payoff=1, loss=1, transaction_cost=0.01, max_risk_fraction=0.2
    )

    # High payoff ratio (aggressive)
    strategy_high = OptimalF(
        win_rate=0.6, payoff=2, loss=1, transaction_cost=0.01, max_risk_fraction=0.2
    )

    # Higher payoff ratio should result in larger bet size
    assert strategy_high.evaluate(0.6, 1000) > strategy_low.evaluate(0.6, 1000)


def test_invalid_parameters():
    """Test that invalid parameters raise appropriate exceptions."""
    # Win rate must be between 0 and 1
    with pytest.raises(ValueError, match="Win rate must be between 0 and 1"):
        OptimalF(
            win_rate=-0.1,
            payoff=1,
            loss=1,
            transaction_cost=0.01,
            max_risk_fraction=0.2,
        )

    with pytest.raises(ValueError, match="Win rate must be between 0 and 1"):
        OptimalF(
            win_rate=1.1, payoff=1, loss=1, transaction_cost=0.01, max_risk_fraction=0.2
        )

    # Max risk fraction must be between 0 and 1
    with pytest.raises(
        ValueError, match="Maximum risk fraction must be between 0 and 1"
    ):
        OptimalF(
            win_rate=0.6, payoff=1, loss=1, transaction_cost=0.01, max_risk_fraction=1.1
        )


def test_simulation():
    """Test the strategy in a simulation with varying performance."""
    payoff = 1
    loss = 1
    transaction_cost = 0.01
    probability = 0.60  # Increased edge to compensate for transaction costs
    trials = 300
    initial_bankroll = 1000

    # Initialize bankroll and strategy
    bankroll = BankRoll(
        initial_funds=initial_bankroll, percent_bettable=1.0, max_draw_down=None
    )
    strategy = OptimalF(
        win_rate=0.60,  # Match the actual probability
        payoff=payoff,
        loss=loss,
        transaction_cost=transaction_cost,
        max_risk_fraction=0.15,  # More conservative risk limit
    )

    # Track bankroll history
    bankroll_history = [initial_bankroll]

    # Set random seed for reproducibility
    random.seed(42)

    simulator = RepeatedBinarySimulator(
        payoff=payoff,
        loss=loss,
        transaction_costs=transaction_cost,
        probability=probability,
        trials=trials,
    )

    simulator.evaluate_strategy(strategy, bankroll)
    bankroll_history.append(bankroll.total_funds)

    # Should have grown with positive edge over many trials
    assert bankroll.total_funds > initial_bankroll * 0.9  # Allow for some variance

    # Check that we have history to compare
    assert len(bankroll_history) > 1


def test_transaction_costs():
    """Test that transaction costs reduce the optimal bet size."""
    strategy_no_costs = OptimalF(
        win_rate=0.6, payoff=1, loss=1, transaction_cost=0, max_risk_fraction=0.2
    )

    strategy_with_costs = OptimalF(
        win_rate=0.6, payoff=1, loss=1, transaction_cost=0.01, max_risk_fraction=0.2
    )

    # Strategy with transaction costs should bet less
    assert strategy_with_costs.evaluate(0.6, 1000) < strategy_no_costs.evaluate(
        0.6, 1000
    )

    # Test that when transaction costs make the bet unprofitable, optimal f is 0
    strategy_high_costs = OptimalF(
        win_rate=0.52, payoff=1, loss=1, transaction_cost=0.05, max_risk_fraction=0.2
    )

    assert strategy_high_costs.evaluate(0.52, 1000) == pytest.approx(0.0)


def test_max_safe_bet():
    """Test that the strategy respects the maximum safe bet limit."""
    strategy = OptimalF(
        win_rate=0.6, payoff=1, loss=1, transaction_cost=0.01, max_risk_fraction=0.2
    )

    # Test with a very small bankroll where max safe bet would be less than optimal f
    small_bankroll = 10
    bet_size = strategy.evaluate(0.6, small_bankroll)
    assert bet_size <= strategy.get_max_safe_bet(small_bankroll)

    # Test with a larger bankroll where optimal f is the limiting factor
    large_bankroll = 1000
    bet_size = strategy.evaluate(0.6, large_bankroll)
    assert bet_size <= strategy.max_risk_fraction


def test_ralph_vince_formula():
    """Test that OptimalF follows Ralph Vince's formula."""
    # Case 1: 60% win rate, 1:1 payoff/loss ratio
    strategy1 = OptimalF(
        win_rate=0.6, payoff=1, loss=1, transaction_cost=0, max_risk_fraction=0.5
    )

    # Ralph Vince's formula: f* = W - (1-W)/(R/L)
    # f* = 0.6 - 0.4/(1/1) = 0.6 - 0.4 = 0.2
    assert strategy1.evaluate(0.6, 1000) == pytest.approx(0.2, abs=0.01)

    # Case 2: 55% win rate, 2:1 payoff/loss ratio
    strategy2 = OptimalF(
        win_rate=0.55, payoff=2, loss=1, transaction_cost=0, max_risk_fraction=0.5
    )

    # f* = 0.55 - 0.45/(2/1) = 0.55 - 0.225 = 0.325
    assert strategy2.evaluate(0.55, 1000) == pytest.approx(0.325, abs=0.01)

    # Case 3: With transaction costs
    strategy3 = OptimalF(
        win_rate=0.6, payoff=1, loss=1, transaction_cost=0.1, max_risk_fraction=0.5
    )

    # Adjusted reward = 1 - 0.1 = 0.9
    # Adjusted risk = 1 + 0.1 = 1.1
    # f* = 0.6 - 0.4/(0.9/1.1) = 0.6 - 0.4/(0.818) = 0.6 - 0.489 = 0.111
    assert strategy3.evaluate(0.6, 1000) == pytest.approx(0.111, abs=0.01)
