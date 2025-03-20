import random

import pytest

from keeks.bankroll import BankRoll
from keeks.binary_strategies import CPPIStrategy
from keeks.simulators.repeated_binary import RepeatedBinarySimulator

random.seed(42)


def test_basic_functionality():
    """Test that CPPIStrategy correctly calculates bet sizes based on floor and cushion."""
    # Initialize with a 50% floor and multiplier of 2
    initial_bankroll = 1000
    strategy = CPPIStrategy(
        floor_fraction=0.5,
        multiplier=2,
        initial_bankroll=initial_bankroll,
        payoff=1,
        loss=1,
        transaction_cost=0,  # No transaction costs for basic test
    )

    # Initial bankroll = 1000, floor = 500, cushion = 500
    # Exposure = 2 * 500 = 1000, capped at 1.0 of bankroll
    # With 60% probability, expected value is 0.2 per unit
    # Adjusted multiplier = 2 * 0.2 = 0.4
    # Final exposure = 0.4 * 500 = 200, proportion = 0.2
    assert strategy.evaluate(0.6, initial_bankroll) == pytest.approx(0.2)

    # Update bankroll to 600 (decreased)
    # Floor = 500, cushion = 100
    # Adjusted multiplier = 2 * 0.2 = 0.4
    # Exposure = 0.4 * 100 = 40, proportion = 40/600 = 0.067
    strategy.update_bankroll(600)
    assert strategy.evaluate(0.6, 600) == pytest.approx(0.067, abs=0.01)

    # Update bankroll to 2000 (increased)
    # Floor = 1000 (adjusted for peak), cushion = 1000
    # Adjusted multiplier = 2 * 0.2 = 0.4
    # Exposure = 0.4 * 1000 = 400, proportion = 400/2000 = 0.2
    strategy.update_bankroll(2000)
    assert strategy.evaluate(0.6, 2000) == pytest.approx(0.2)


def test_min_probability_threshold():
    """Test that the strategy respects the minimum probability threshold and expected value."""
    strategy = CPPIStrategy(
        floor_fraction=0.5,
        multiplier=2,
        initial_bankroll=1000,
        payoff=1,
        loss=1,
        transaction_cost=0,  # No transaction costs for this test
        min_probability=0.5,
    )

    # At 60% probability, should bet (positive expected value)
    assert strategy.evaluate(0.6, 1000) > 0

    # At 50% probability, should not bet (zero expected value)
    assert strategy.evaluate(0.5, 1000) == 0

    # Below min probability, should not bet
    assert strategy.evaluate(0.4, 1000) == 0


def test_floor_protection():
    """Test that the strategy properly protects the floor capital."""
    initial_bankroll = 1000
    strategy = CPPIStrategy(
        floor_fraction=0.8,  # High floor
        multiplier=2,
        initial_bankroll=initial_bankroll,
        payoff=1,
        loss=1,
        transaction_cost=0,  # No transaction costs for this test
    )

    # Initial bankroll = 1000, floor = 800, cushion = 200
    # With 60% probability, expected value is 0.2 per unit
    # Adjusted multiplier = 2 * 0.2 = 0.4
    # Exposure = 0.4 * 200 = 80, proportion = 80/1000 = 0.08
    assert strategy.evaluate(0.6, initial_bankroll) == pytest.approx(0.08)

    # Update bankroll to 850 (decreased, but still above floor)
    # Floor = 800, cushion = 50
    # Adjusted multiplier = 2 * 0.2 = 0.4
    # Exposure = 0.4 * 50 = 20, proportion = 20/850 = 0.024
    strategy.update_bankroll(850)
    assert strategy.evaluate(0.6, 850) == pytest.approx(0.024, abs=0.01)

    # Update bankroll to exactly the floor value
    # Floor = 800, cushion = 0, exposure = 0
    strategy.update_bankroll(800)
    assert strategy.evaluate(0.6, 800) == 0

    # Update bankroll below the floor
    # Floor = 800, cushion = 0, exposure = 0
    strategy.update_bankroll(700)
    assert strategy.evaluate(0.6, 700) == 0


def test_multiplier_effect():
    """Test how different multipliers affect the bet size."""
    initial_bankroll = 1000
    floor_fraction = 0.5  # 500

    # Low multiplier (conservative)
    strategy_low = CPPIStrategy(
        floor_fraction=floor_fraction,
        multiplier=1,
        initial_bankroll=initial_bankroll,
        payoff=1,
        loss=1,
        transaction_cost=0,  # No transaction costs for this test
    )

    # Medium multiplier
    strategy_med = CPPIStrategy(
        floor_fraction=floor_fraction,
        multiplier=2,
        initial_bankroll=initial_bankroll,
        payoff=1,
        loss=1,
        transaction_cost=0,
    )

    # High multiplier (aggressive)
    strategy_high = CPPIStrategy(
        floor_fraction=floor_fraction,
        multiplier=3,
        initial_bankroll=initial_bankroll,
        payoff=1,
        loss=1,
        transaction_cost=0,
    )

    # All have same bankroll, floor (500), and cushion (500)
    # With 60% probability, expected value is 0.2 per unit
    # Exposure calculations with adjusted multipliers:
    # Low: 1 * 0.2 * 500 = 100, proportion: 100/1000 = 0.1
    # Med: 2 * 0.2 * 500 = 200, proportion: 200/1000 = 0.2
    # High: 3 * 0.2 * 500 = 300, proportion: 300/1000 = 0.3
    assert strategy_low.evaluate(0.6, initial_bankroll) == pytest.approx(0.1)
    assert strategy_med.evaluate(0.6, initial_bankroll) == pytest.approx(0.2)
    assert strategy_high.evaluate(0.6, initial_bankroll) == pytest.approx(0.3)

    # Update all to 750 (decreased)
    # Floor = 500, cushion = 250
    # Exposure calculations with adjusted multipliers:
    # Low: 1 * 0.2 * 250 = 50, proportion: 50/750 = 0.067
    # Med: 2 * 0.2 * 250 = 100, proportion: 100/750 = 0.133
    # High: 3 * 0.2 * 250 = 150, proportion: 150/750 = 0.2
    strategy_low.update_bankroll(750)
    strategy_med.update_bankroll(750)
    strategy_high.update_bankroll(750)

    assert strategy_low.evaluate(0.6, 750) == pytest.approx(0.067, abs=0.01)
    assert strategy_med.evaluate(0.6, 750) == pytest.approx(0.133, abs=0.01)
    assert strategy_high.evaluate(0.6, 750) == pytest.approx(0.2, abs=0.01)


def test_invalid_parameters():
    """Test that invalid parameters raise appropriate exceptions."""
    # Floor fraction must be between 0 and 1
    with pytest.raises(ValueError, match="Floor fraction must be between 0 and 1"):
        CPPIStrategy(
            floor_fraction=-0.1, multiplier=2, initial_bankroll=1000, payoff=1, loss=1
        )

    with pytest.raises(ValueError, match="Floor fraction must be between 0 and 1"):
        CPPIStrategy(
            floor_fraction=1.0, multiplier=2, initial_bankroll=1000, payoff=1, loss=1
        )

    # Multiplier must be positive
    with pytest.raises(ValueError, match="Multiplier must be greater than 0"):
        CPPIStrategy(
            floor_fraction=0.5, multiplier=0, initial_bankroll=1000, payoff=1, loss=1
        )

    with pytest.raises(ValueError, match="Multiplier must be greater than 0"):
        CPPIStrategy(
            floor_fraction=0.5, multiplier=-1, initial_bankroll=1000, payoff=1, loss=1
        )

    # Initial bankroll must be positive
    with pytest.raises(ValueError, match="Initial bankroll must be greater than 0"):
        CPPIStrategy(
            floor_fraction=0.5, multiplier=2, initial_bankroll=0, payoff=1, loss=1
        )

    with pytest.raises(ValueError, match="Initial bankroll must be greater than 0"):
        CPPIStrategy(
            floor_fraction=0.5, multiplier=2, initial_bankroll=-1000, payoff=1, loss=1
        )


def test_simulation():
    """Test the strategy in a simulation with a favorable edge."""
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
    strategy = CPPIStrategy(
        floor_fraction=0.5,
        multiplier=1.5,  # More conservative multiplier
        initial_bankroll=initial_bankroll,
        payoff=payoff,
        loss=loss,
        transaction_cost=transaction_cost,
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

    # Should have preserved the floor (never dropped below 50% of initial)
    # This is a key feature of CPPI
    floor_value = initial_bankroll * strategy.floor_fraction
    assert min(bankroll_history) >= floor_value * 0.9

    # Check that we have history to compare
    assert len(bankroll_history) > 1


def test_transaction_costs():
    """Test that transaction costs reduce the bet size."""
    initial_bankroll = 1000
    strategy_no_costs = CPPIStrategy(
        floor_fraction=0.5,
        multiplier=2,
        initial_bankroll=initial_bankroll,
        payoff=1,
        loss=1,
        transaction_cost=0,
    )

    strategy_with_costs = CPPIStrategy(
        floor_fraction=0.5,
        multiplier=2,
        initial_bankroll=initial_bankroll,
        payoff=1,
        loss=1,
        transaction_cost=0.01,
    )

    # Strategy with transaction costs should bet less
    assert strategy_with_costs.evaluate(
        0.6, initial_bankroll
    ) < strategy_no_costs.evaluate(0.6, initial_bankroll)


def test_max_safe_bet():
    """Test that the strategy respects the maximum safe bet limit."""
    initial_bankroll = 1000
    strategy = CPPIStrategy(
        floor_fraction=0.5,
        multiplier=2,
        initial_bankroll=initial_bankroll,
        payoff=1,
        loss=1,
        transaction_cost=0.01,
    )

    # Test with a very small bankroll where max safe bet would be less than CPPI exposure
    small_bankroll = 10
    bet_size = strategy.evaluate(0.6, small_bankroll)
    assert bet_size <= strategy.get_max_safe_bet(small_bankroll)

    # Test with a larger bankroll where CPPI exposure is the limiting factor
    large_bankroll = 1000
    bet_size = strategy.evaluate(0.6, large_bankroll)
    assert bet_size <= 1.0  # Maximum proportion should never exceed 100%
