import random

import pytest

from keeks.bankroll import BankRoll
from keeks.binary_strategies import FixedFractionStrategy
from keeks.simulators.repeated_binary import RepeatedBinarySimulator

random.seed(42)


def test_basic_functionality():
    """Test that the FixedFractionStrategy returns the correct fixed fraction."""
    # Initialize with a 0.1 (10%) fraction
    strategy = FixedFractionStrategy(fraction=0.1, payoff=1, loss=1)
    current_bankroll = 1000

    # Should return the fixed fraction when probability is >= min_probability
    assert strategy.evaluate(0.6, current_bankroll) == pytest.approx(0.1)
    assert strategy.evaluate(0.5, current_bankroll) == pytest.approx(0.1)

    # Should return 0 when probability is < min_probability
    assert strategy.evaluate(0.4, current_bankroll) == pytest.approx(0)


def test_min_probability_threshold():
    """Test that the strategy respects the minimum probability threshold."""
    # Default min_probability is 0.5
    strategy = FixedFractionStrategy(fraction=0.2, payoff=1, loss=1)
    current_bankroll = 1000

    assert strategy.evaluate(0.5, current_bankroll) == pytest.approx(0.2)
    assert strategy.evaluate(0.4, current_bankroll) == pytest.approx(0)

    # Custom min_probability of 0.3
    strategy = FixedFractionStrategy(
        fraction=0.2, payoff=1, loss=1, min_probability=0.3
    )

    assert strategy.evaluate(0.3, current_bankroll) == pytest.approx(0.2)
    assert strategy.evaluate(0.29, current_bankroll) == pytest.approx(0)


def test_invalid_parameters():
    """Test that invalid parameters raise appropriate exceptions."""
    # Fraction must be between 0 and 1
    with pytest.raises(ValueError, match="Fraction must be between 0 and 1"):
        FixedFractionStrategy(fraction=-0.1, payoff=1, loss=1)

    with pytest.raises(ValueError, match="Fraction must be between 0 and 1"):
        FixedFractionStrategy(fraction=1.1, payoff=1, loss=1)

    # Min probability must be between 0 and 1
    with pytest.raises(ValueError, match="Minimum probability must be between 0 and 1"):
        FixedFractionStrategy(fraction=0.1, payoff=1, loss=1, min_probability=-0.1)

    with pytest.raises(ValueError, match="Minimum probability must be between 0 and 1"):
        FixedFractionStrategy(fraction=0.1, payoff=1, loss=1, min_probability=1.1)


def test_simulation():
    """Test the strategy in a simulation with a favorable edge."""
    payoff = 1
    loss = 1
    transaction_cost = 0.01
    probability = 0.55  # Slight edge
    trials = 500

    # Initialize bankroll and strategy
    bankroll = BankRoll(initial_funds=1000, percent_bettable=1, max_draw_down=1)
    strategy = FixedFractionStrategy(
        fraction=0.05, payoff=payoff, loss=loss
    )  # 5% fixed fraction

    # Set up and run simulation
    simulator = RepeatedBinarySimulator(
        payoff=payoff,
        loss=loss,
        transaction_costs=transaction_cost,
        probability=probability,
        trials=trials,
    )
    simulator.evaluate_strategy(strategy, bankroll)

    # Should increase funds with a positive edge
    assert bankroll.total_funds > 1000


def test_simulation_with_different_fractions():
    """Compare performance with different fixed fractions."""
    payoff = 1
    loss = 1
    transaction_cost = 0.01
    probability = 0.55  # Slight edge
    trials = 500

    # Strategy with higher fraction (more aggressive)
    bankroll_high = BankRoll(initial_funds=1000, percent_bettable=1, max_draw_down=1)
    strategy_high = FixedFractionStrategy(
        fraction=0.1, payoff=payoff, loss=loss
    )  # 10% fixed fraction

    # Strategy with lower fraction (more conservative)
    bankroll_low = BankRoll(initial_funds=1000, percent_bettable=1, max_draw_down=1)
    strategy_low = FixedFractionStrategy(
        fraction=0.01, payoff=payoff, loss=loss
    )  # 1% fixed fraction

    # Use the same random seed for fair comparison
    simulator_high = RepeatedBinarySimulator(
        payoff=payoff,
        loss=loss,
        transaction_costs=transaction_cost,
        probability=probability,
        trials=trials,
    )

    simulator_low = RepeatedBinarySimulator(
        payoff=payoff,
        loss=loss,
        transaction_costs=transaction_cost,
        probability=probability,
        trials=trials,
    )

    # Run simulations with the same seed
    random.seed(42)
    simulator_high.evaluate_strategy(strategy_high, bankroll_high)

    random.seed(42)
    simulator_low.evaluate_strategy(strategy_low, bankroll_low)

    # Both should grow, but they'll have different growth rates and volatility
    assert bankroll_high.total_funds > 1000
    assert bankroll_low.total_funds > 1000

    # Check that we have history to compare
    assert len(bankroll_high.history) > 1
    assert len(bankroll_low.history) > 1


def test_custom_min_probability():
    """Test that custom minimum probability threshold works."""
    strategy = FixedFractionStrategy(
        fraction=0.3, payoff=1, loss=1, min_probability=0.6
    )
    current_bankroll = 1000

    assert strategy.evaluate(0.6, current_bankroll) == pytest.approx(0.3)
    assert strategy.evaluate(0.5, current_bankroll) == pytest.approx(
        0.0
    )  # Below threshold


def test_zero_fraction():
    """Test that zero fraction always returns zero."""
    strategy = FixedFractionStrategy(fraction=0.0, payoff=1, loss=1)
    current_bankroll = 1000

    assert strategy.evaluate(0.6, current_bankroll) == pytest.approx(0.0)
    assert strategy.evaluate(0.5, current_bankroll) == pytest.approx(0.0)
    assert strategy.evaluate(0.4, current_bankroll) == pytest.approx(0.0)


def test_one_fraction():
    """Test that fraction of 1.0 returns 1.0 for valid probabilities."""
    strategy = FixedFractionStrategy(fraction=1.0, payoff=1, loss=1)
    current_bankroll = 1000

    assert strategy.evaluate(0.6, current_bankroll) == pytest.approx(1.0)
    assert strategy.evaluate(0.5, current_bankroll) == pytest.approx(1.0)
    assert strategy.evaluate(0.4, current_bankroll) == pytest.approx(
        0.0
    )  # Below threshold
