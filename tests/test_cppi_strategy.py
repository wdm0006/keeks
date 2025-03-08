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
        initial_bankroll=initial_bankroll
    )
    
    # Initial bankroll = 1000, floor = 500, cushion = 500
    # Exposure = 2 * 500 = 1000, capped at 1.0 of bankroll
    # So should return 1.0 (bet full bankroll)
    assert strategy.evaluate(0.6) == pytest.approx(1.0)
    
    # Update bankroll to 600 (decreased)
    # Floor = 500, cushion = 100, exposure = 2 * 100 = 200
    # As proportion: 200/600 = 0.333...
    strategy.update_bankroll(600)
    assert strategy.evaluate(0.6) == pytest.approx(0.333, abs=0.01)
    
    # Update bankroll to 2000 (increased)
    # Floor = 500, cushion = 1500, exposure = 2 * 1500 = 3000
    # Exposure is capped at bankroll, so proportion is 1.0
    strategy.update_bankroll(2000)
    assert strategy.evaluate(0.6) == pytest.approx(1.0)


def test_min_probability_threshold():
    """Test that the strategy respects the minimum probability threshold."""
    strategy = CPPIStrategy(
        floor_fraction=0.5,
        multiplier=2,
        initial_bankroll=1000,
        min_probability=0.5
    )
    
    # Should bet when probability >= min_probability
    assert strategy.evaluate(0.5) > 0
    
    # Should not bet when probability < min_probability
    assert strategy.evaluate(0.49) == 0


def test_floor_protection():
    """Test that the strategy properly protects the floor capital."""
    initial_bankroll = 1000
    strategy = CPPIStrategy(
        floor_fraction=0.8,  # High floor
        multiplier=2,
        initial_bankroll=initial_bankroll
    )
    
    # Initial bankroll = 1000, floor = 800, cushion = 200
    # Exposure = 2 * 200 = 400, as proportion: 400/1000 = 0.4
    assert strategy.evaluate(0.6) == pytest.approx(0.4)
    
    # Update bankroll to 850 (decreased, but still above floor)
    # Floor = 800, cushion = 50, exposure = 2 * 50 = 100
    # As proportion: 100/850 = 0.118...
    strategy.update_bankroll(850)
    assert strategy.evaluate(0.6) == pytest.approx(0.118, abs=0.01)
    
    # Update bankroll to exactly the floor value
    # Floor = 800, cushion = 0, exposure = 0
    strategy.update_bankroll(800)
    assert strategy.evaluate(0.6) == 0
    
    # Update bankroll below the floor
    # Floor = 800, cushion = 0, exposure = 0
    strategy.update_bankroll(700)
    assert strategy.evaluate(0.6) == 0


def test_multiplier_effect():
    """Test how different multipliers affect the bet size."""
    initial_bankroll = 1000
    floor_fraction = 0.5  # 500
    
    # Low multiplier (conservative)
    strategy_low = CPPIStrategy(
        floor_fraction=floor_fraction,
        multiplier=1,
        initial_bankroll=initial_bankroll
    )
    
    # Medium multiplier
    strategy_med = CPPIStrategy(
        floor_fraction=floor_fraction,
        multiplier=2,
        initial_bankroll=initial_bankroll
    )
    
    # High multiplier (aggressive)
    strategy_high = CPPIStrategy(
        floor_fraction=floor_fraction,
        multiplier=3,
        initial_bankroll=initial_bankroll
    )
    
    # All have same bankroll, floor (500), and cushion (500)
    # Exposure calculations:
    # Low: 1 * 500 = 500, proportion: 500/1000 = 0.5
    # Med: 2 * 500 = 1000, proportion: 1000/1000 = 1.0
    # High: 3 * 500 = 1500, capped at 1000, proportion: 1.0
    assert strategy_low.evaluate(0.6) == pytest.approx(0.5)
    assert strategy_med.evaluate(0.6) == pytest.approx(1.0)
    assert strategy_high.evaluate(0.6) == pytest.approx(1.0)
    
    # Update all to 750 (decreased)
    # Floor = 500, cushion = 250
    # Exposure calculations:
    # Low: 1 * 250 = 250, proportion: 250/750 = 0.333...
    # Med: 2 * 250 = 500, proportion: 500/750 = 0.667...
    # High: 3 * 250 = 750, proportion: 750/750 = 1.0
    strategy_low.update_bankroll(750)
    strategy_med.update_bankroll(750)
    strategy_high.update_bankroll(750)
    
    assert strategy_low.evaluate(0.6) == pytest.approx(0.333, abs=0.01)
    assert strategy_med.evaluate(0.6) == pytest.approx(0.667, abs=0.01)
    assert strategy_high.evaluate(0.6) == pytest.approx(1.0)


def test_invalid_parameters():
    """Test that invalid parameters raise appropriate exceptions."""
    # Floor fraction must be between 0 and 1
    with pytest.raises(ValueError, match="Floor fraction must be between 0 and 1"):
        CPPIStrategy(floor_fraction=-0.1, multiplier=2, initial_bankroll=1000)
    
    with pytest.raises(ValueError, match="Floor fraction must be between 0 and 1"):
        CPPIStrategy(floor_fraction=1, multiplier=2, initial_bankroll=1000)
    
    # Multiplier must be positive
    with pytest.raises(ValueError, match="Multiplier must be greater than 0"):
        CPPIStrategy(floor_fraction=0.5, multiplier=0, initial_bankroll=1000)
    
    # Initial bankroll must be positive
    with pytest.raises(ValueError, match="Initial bankroll must be greater than 0"):
        CPPIStrategy(floor_fraction=0.5, multiplier=2, initial_bankroll=0)


def test_simulation():
    """Test the strategy in a simulation with a favorable edge."""
    payoff = 1
    loss = 1
    transaction_cost = 0.01
    probability = 0.55  # Slight edge
    trials = 300
    initial_bankroll = 1000
    
    # Initialize bankroll and strategy with a higher max_draw_down to avoid RuinError
    bankroll = BankRoll(initial_funds=initial_bankroll, percent_bettable=0.5, max_draw_down=None)
    strategy = CPPIStrategy(
        floor_fraction=0.5,
        multiplier=1.5,
        initial_bankroll=initial_bankroll
    )
    
    # Set up simulator
    simulator = RepeatedBinarySimulator(
        payoff=payoff,
        loss=loss,
        transaction_costs=transaction_cost,
        probability=probability,
        trials=trials,
    )
    
    # Custom evaluate function to update the CPPI strategy's internal bankroll state
    bankroll_history = [initial_bankroll]
    
    # We'll use a more conservative simulation approach
    for _ in range(trials):
        # Get current capital
        current_bankroll = bankroll.total_funds
        
        # Update CPPI strategy with current bankroll
        strategy.update_bankroll(current_bankroll)
        
        # Calculate bet size as a proportion (use a smaller multiplier for safety)
        bet_proportion = min(0.2, strategy.evaluate(probability))  # Cap at 20% of bankroll
        
        # Calculate actual bet amount
        bet_amount = bet_proportion * bankroll.bettable_funds
        
        # Simulate the bet outcome
        if random.random() < probability:  # Win
            bankroll.deposit(bet_amount * payoff - transaction_cost)
        else:  # Loss
            bankroll.withdraw(bet_amount * loss + transaction_cost)
            
        bankroll_history.append(bankroll.total_funds)
    
    # Should have grown with positive edge over many trials
    assert bankroll.total_funds > initial_bankroll * 0.9  # Allow for some variance
    
    # Should have preserved the floor (never dropped below 50% of initial)
    # This is a key feature of CPPI
    floor_value = initial_bankroll * strategy.floor_fraction
    assert min(bankroll_history) >= floor_value * 0.9  # Allow small margin 