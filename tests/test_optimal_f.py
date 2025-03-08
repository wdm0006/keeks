import random

import pytest

from keeks.bankroll import BankRoll
from keeks.binary_strategies import OptimalF, KellyCriterion
from keeks.simulators.repeated_binary import RepeatedBinarySimulator

random.seed(42)


def test_basic_functionality():
    """Test that OptimalF properly calculates bet sizes based on win rate and payoff."""
    payoff = 2
    loss = 1
    transaction_cost = 0
    win_rate = 0.6
    
    # Initialize OptimalF with a 60% win rate and a higher max_risk_fraction
    strategy = OptimalF(payoff, loss, transaction_cost, win_rate, max_risk_fraction=0.5)
    
    # Calculate expected optimal f for a 60% win rate with 2:1 payoff
    # Using the formula (W*R - L)/R where W=win rate, L=loss rate, R=reward/risk
    # (0.6*2 - 0.4)/2 = 0.4
    expected_bet = 0.4
    
    # Test with the provided win rate
    actual_bet = strategy.evaluate(win_rate)
    assert actual_bet == pytest.approx(expected_bet)
    
    # Test with a different probability (should override the win_rate)
    higher_prob = 0.7
    higher_expected_bet = (0.7*2 - 0.3)/2  # 0.55
    # But this would be capped at max_risk_fraction of 0.5
    assert strategy.evaluate(higher_prob) == pytest.approx(0.5)
    
    # Test with default max_risk_fraction (0.2)
    default_strategy = OptimalF(payoff, loss, transaction_cost, win_rate)
    # This should be capped at 0.2 even though the calculated optimal f is 0.4
    assert default_strategy.evaluate(win_rate) == pytest.approx(0.2)


def test_transaction_costs():
    """Test that OptimalF correctly accounts for transaction costs."""
    payoff = 2
    loss = 1
    win_rate = 0.6
    max_risk_fraction = 0.5  # Set higher than the expected optimal f
    
    # Without transaction costs - uncapped optimal f would be 0.4
    strategy_no_cost = OptimalF(payoff, loss, 0, win_rate, max_risk_fraction=max_risk_fraction)
    bet_no_cost = strategy_no_cost.evaluate(win_rate)
    assert bet_no_cost == pytest.approx(0.4)
    
    # With transaction costs - should reduce the bet size
    transaction_cost = 0.05
    strategy_with_cost = OptimalF(payoff, loss, transaction_cost, win_rate, max_risk_fraction=max_risk_fraction)
    bet_with_cost = strategy_with_cost.evaluate(win_rate)
    
    # The bet with transaction costs should be smaller
    assert bet_with_cost < bet_no_cost
    
    # The difference should be approximately transaction_cost/payoff
    assert bet_no_cost - bet_with_cost == pytest.approx(transaction_cost/payoff)


def test_max_risk_fraction():
    """Test that OptimalF respects the maximum risk fraction."""
    payoff = 10  # High payoff to get a large optimal f
    loss = 1
    transaction_cost = 0
    win_rate = 0.6
    
    # Normal OptimalF calculation for these parameters would yield:
    # (0.6*10 - 0.4)/10 = 0.56
    
    # Test with default max_risk_fraction (0.2)
    strategy_default = OptimalF(payoff, loss, transaction_cost, win_rate)
    assert strategy_default.evaluate(win_rate) == pytest.approx(0.2)  # Capped at 0.2
    
    # Test with custom max_risk_fraction higher than calculated optimal f
    strategy_high = OptimalF(payoff, loss, transaction_cost, win_rate, max_risk_fraction=0.6)
    assert strategy_high.evaluate(win_rate) == pytest.approx(0.56)  # Not capped
    
    # Test with custom max_risk_fraction lower than calculated optimal f
    strategy_low = OptimalF(payoff, loss, transaction_cost, win_rate, max_risk_fraction=0.1)
    assert strategy_low.evaluate(win_rate) == pytest.approx(0.1)  # Capped at 0.1


def test_negative_ev_cases():
    """Test behavior when expected value is negative."""
    payoff = 1
    loss = 1
    transaction_cost = 0
    win_rate = 0.4  # Negative EV
    
    strategy = OptimalF(payoff, loss, transaction_cost, win_rate)
    
    # Optimal f with negative expectation should be 0
    assert strategy.evaluate(win_rate) == 0
    assert strategy.evaluate(0.3) == 0  # Even worse probability


def test_comparison_with_kelly():
    """Compare OptimalF to Kelly Criterion for various scenarios."""
    test_cases = [
        # payoff, loss, win_rate
        (2, 1, 0.6),    # Favorable odds, good win rate
        (1, 1, 0.55),   # Even odds, slight edge
        (3, 1, 0.4),    # High payoff, low win rate
    ]
    
    for payoff, loss, win_rate in test_cases:
        # No transaction costs for simpler comparison
        transaction_cost = 0
        
        kelly = KellyCriterion(payoff, loss, transaction_cost)
        optimal_f = OptimalF(payoff, loss, transaction_cost, win_rate, max_risk_fraction=1.0)
        
        kelly_bet = kelly.evaluate(win_rate)
        optimal_f_bet = optimal_f.evaluate(win_rate)
        
        # For a binary outcome, Kelly and OptimalF should be similar
        # when using the same parameters and no constraints
        assert kelly_bet == pytest.approx(optimal_f_bet, abs=0.01)


def test_invalid_parameters():
    """Test that invalid parameters raise appropriate exceptions."""
    payoff = 2
    loss = 1
    transaction_cost = 0
    
    # Win rate must be between 0 and 1
    with pytest.raises(ValueError, match="Win rate must be between 0 and 1"):
        OptimalF(payoff, loss, transaction_cost, win_rate=-0.1)
    
    with pytest.raises(ValueError, match="Win rate must be between 0 and 1"):
        OptimalF(payoff, loss, transaction_cost, win_rate=1.1)
    
    # Max risk fraction must be between 0 and 1
    with pytest.raises(ValueError, match="Maximum risk fraction must be between 0 and 1"):
        OptimalF(payoff, loss, transaction_cost, win_rate=0.6, max_risk_fraction=0)
    
    with pytest.raises(ValueError, match="Maximum risk fraction must be between 0 and 1"):
        OptimalF(payoff, loss, transaction_cost, win_rate=0.6, max_risk_fraction=1.1)


def test_simulation():
    """Test the strategy in a simulation with a favorable edge."""
    payoff = 1
    loss = 1
    transaction_cost = 0.01
    win_rate = 0.55  # Slight edge
    trials = 1000
    
    # Initialize bankroll and strategy
    bankroll = BankRoll(initial_funds=1000, percent_bettable=1, max_draw_down=1)
    strategy = OptimalF(payoff, loss, transaction_cost, win_rate)
    
    # Set up and run simulation
    simulator = RepeatedBinarySimulator(
        payoff=payoff,
        loss=loss,
        transaction_costs=transaction_cost,
        probability=win_rate,  # Use the same probability as win_rate for simplicity
        trials=trials,
    )
    simulator.evaluate_strategy(strategy, bankroll)
    
    # Should increase funds with a positive edge
    assert bankroll.total_funds > 1000
    assert len(bankroll.history) > 1 