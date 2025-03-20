import random

import pytest

from keeks.bankroll import BankRoll
from keeks.binary_strategies import KellyCriterion
from keeks.binary_strategies.kelly import FractionalKellyCriterion
from keeks.binary_strategies.simple import NaiveStrategy
from keeks.simulators.random_binary import RandomBinarySimulator
from keeks.simulators.random_uncertain_binary import RandomUncertainBinarySimulator
from keeks.simulators.repeated_binary import RepeatedBinarySimulator

random.seed(42)


def test_basic_functionality():
    """Test that KellyCriterion correctly calculates optimal bet sizes."""
    strategy = KellyCriterion(payoff=1, loss=1, transaction_cost=0)

    # With 60% probability and 1:1 payoff ratio, Kelly fraction should be 0.2
    assert strategy.evaluate(0.6, 1000) == pytest.approx(0.2)


def test_min_probability_threshold():
    """Test that the strategy respects the minimum probability threshold."""
    strategy = KellyCriterion(payoff=1, loss=1, transaction_cost=0, min_probability=0.5)

    # Should bet when probability > min_probability
    assert strategy.evaluate(0.51, 1000) > 0
    assert strategy.evaluate(0.4, 1000) == pytest.approx(0.0)


def test_payoff_ratio_effect():
    """Test how different payoff ratios affect the Kelly fraction."""
    # Low payoff ratio (conservative)
    strategy_low = KellyCriterion(payoff=1, loss=1, transaction_cost=0)

    # High payoff ratio (aggressive)
    strategy_high = KellyCriterion(payoff=2, loss=1, transaction_cost=0)

    # Higher payoff ratio should result in larger bet size
    assert strategy_high.evaluate(0.6, 1000) > strategy_low.evaluate(0.6, 1000)


def test_transaction_costs():
    """Test how transaction costs affect the Kelly fraction."""
    # No transaction costs
    strategy_no_cost = KellyCriterion(payoff=1, loss=1, transaction_cost=0)

    # With transaction costs
    strategy_with_cost = KellyCriterion(payoff=1, loss=1, transaction_cost=0.01)

    # Transaction costs should reduce bet size
    assert strategy_with_cost.evaluate(0.6, 1000) < strategy_no_cost.evaluate(0.6, 1000)

    # With large transaction costs that make betting unprofitable
    strategy_large_cost = KellyCriterion(payoff=1, loss=1, transaction_cost=0.5)

    # Should not bet when transaction costs make betting unprofitable
    assert strategy_large_cost.evaluate(0.6, 1000) == pytest.approx(0.0)


def test_invalid_parameters():
    """Test that invalid parameters raise appropriate exceptions."""
    # Payoff must be positive
    with pytest.raises(ValueError, match="Payoff must be greater than 0"):
        KellyCriterion(payoff=0, loss=1, transaction_cost=0)

    # Loss must be non-negative
    with pytest.raises(ValueError, match="Loss must be non-negative"):
        KellyCriterion(payoff=1, loss=-1, transaction_cost=0)

    # Transaction cost must be non-negative
    with pytest.raises(ValueError, match="Transaction cost must be non-negative"):
        KellyCriterion(payoff=1, loss=1, transaction_cost=-0.01)

    # Total cost must be positive
    with pytest.raises(ValueError, match="Total cost .* must be greater than 0"):
        KellyCriterion(payoff=1, loss=0, transaction_cost=0)


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
    strategy = KellyCriterion(
        payoff=payoff, loss=loss, transaction_cost=transaction_cost
    )

    # Set up simulator
    RepeatedBinarySimulator(
        payoff=payoff,
        loss=loss,
        transaction_costs=transaction_cost,
        probability=probability,
        trials=trials,
    )

    # Track bankroll history
    bankroll_history = [initial_bankroll]

    for _ in range(trials):
        # Get current capital
        current_bankroll = bankroll.total_funds

        # Calculate bet size as a proportion
        bet_proportion = strategy.evaluate(probability, current_bankroll)

        # Record the bet size
        bankroll_history.append(current_bankroll)

        # Update bankroll based on the bet
        if random.random() < probability:
            # Win
            bankroll.add_funds(
                current_bankroll * bet_proportion * (payoff - transaction_cost)
            )
        else:
            # Loss
            bankroll.remove_funds(
                current_bankroll * bet_proportion * (loss + transaction_cost)
            )

    # Verify that the bankroll never went below 50% of initial
    assert all(b >= initial_bankroll * 0.5 for b in bankroll_history)


def test_even_odds():
    strategy = KellyCriterion(payoff=1, loss=1, transaction_cost=0)
    current_bankroll = 1000

    # 60% chance of winning
    portion = strategy.evaluate(0.6, current_bankroll)
    assert portion == pytest.approx(0.2)  # (0.6 - 0.4) = 0.2

    # 40% chance of winning (negative EV)
    portion = strategy.evaluate(0.4, current_bankroll)
    assert portion == pytest.approx(0.0)  # Should return 0 for negative EV


def test_known_cases():
    # No transaction costs
    strategy = KellyCriterion(payoff=2, loss=1, transaction_cost=0)
    current_bankroll = 1000

    # 50% chance of winning with 2:1 payoff
    assert strategy.evaluate(0.5, current_bankroll) == pytest.approx(
        0.25
    )  # (0.5 * 2 - 0.5) / 2 = 0.25

    # 60% chance of winning with 2:1 payoff
    assert strategy.evaluate(0.6, current_bankroll) == pytest.approx(
        0.4
    )  # (0.6 * 2 - 0.4) / 2 = 0.4

    # 40% chance of winning with 2:1 payoff (negative EV)
    assert strategy.evaluate(0.4, current_bankroll) == pytest.approx(
        0.0
    )  # Should return 0 for negative EV

    # With transaction costs
    strategy_with_cost = KellyCriterion(payoff=2, loss=1, transaction_cost=0.1)

    # 60% chance of winning with 2:1 payoff and transaction costs
    # Expected formula: adjusted_payoff = 1.9, adjusted_loss = 1.1
    # b = 1.9/1.1 = 1.727, f* = (1.727*0.6 - 0.4)/1.727 = (1.036 - 0.4)/1.727 = 0.368
    assert strategy_with_cost.evaluate(0.6, current_bankroll) == pytest.approx(
        0.368, abs=0.001
    )


def test_zero_probability():
    strategy = KellyCriterion(payoff=2, loss=1, transaction_cost=0)
    current_bankroll = 1000

    # 0% chance of winning
    assert strategy.evaluate(0.0, current_bankroll) == pytest.approx(
        0.0
    )  # Should return 0 for 0% probability


def test_one_probability():
    strategy = KellyCriterion(payoff=2, loss=1, transaction_cost=0)
    current_bankroll = 1000

    # 100% chance of winning
    assert strategy.evaluate(1.0, current_bankroll) == pytest.approx(
        1.0
    )  # Should bet entire bankroll


def test_simulation_repeated():
    payoff = 1
    loss = 1
    transaction_cost = 0.01
    probability = 0.55
    trials = 1_000

    bankroll = BankRoll(initial_funds=1000, percent_bettable=1, max_draw_down=1)
    strategy = KellyCriterion(payoff, loss, transaction_cost)
    simulator = RepeatedBinarySimulator(
        payoff=payoff,
        loss=loss,
        transaction_costs=transaction_cost,
        probability=probability,
        trials=trials,
    )
    simulator.evaluate_strategy(strategy, bankroll)
    assert bankroll.total_funds > 1000


def test_simulation_random():
    payoff = 1
    loss = 1
    transaction_cost = 0.01
    trials = 100
    stdev = 0.05

    bankroll_kelly = BankRoll(
        initial_funds=1000, percent_bettable=0.1, max_draw_down=0.5
    )
    bankroll_naive = BankRoll(
        initial_funds=1000, percent_bettable=0.1, max_draw_down=0.5
    )

    strategy_kelly = KellyCriterion(payoff, loss, transaction_cost)
    strategy_naive = NaiveStrategy(payoff, loss, transaction_cost)

    simulator_kelly = RandomBinarySimulator(
        payoff=payoff,
        loss=loss,
        transaction_costs=transaction_cost,
        trials=trials,
        stdev=stdev,
    )

    simulator_naive = RandomBinarySimulator(
        payoff=payoff,
        loss=loss,
        transaction_costs=transaction_cost,
        trials=trials,
        stdev=stdev,
    )

    random.seed(42)
    simulator_kelly.evaluate_strategy(strategy_kelly, bankroll_kelly)

    random.seed(42)
    simulator_naive.evaluate_strategy(strategy_naive, bankroll_naive)

    assert len(bankroll_kelly.history) > 1
    assert len(bankroll_naive.history) > 1


def test_simulation_random_uncertain():
    payoff = 0.15
    loss = 0.15
    transaction_cost = 0.01
    trials = 100
    stdev = 0.05
    uncertainty_stdev = 0.02

    bankroll_kelly = BankRoll(
        initial_funds=1000, percent_bettable=0.1, max_draw_down=0.5
    )
    bankroll_naive = BankRoll(
        initial_funds=1000, percent_bettable=0.1, max_draw_down=0.5
    )
    bankroll_fractional_kelly = BankRoll(
        initial_funds=1000, percent_bettable=0.1, max_draw_down=0.5
    )

    strategy_kelly = KellyCriterion(payoff, loss, transaction_cost)
    strategy_naive = NaiveStrategy(payoff, loss, transaction_cost)
    strategy_fractional_kelly = FractionalKellyCriterion(
        payoff, loss, transaction_cost, 0.5
    )

    simulator_kelly = RandomUncertainBinarySimulator(
        payoff=payoff,
        loss=loss,
        transaction_costs=transaction_cost,
        trials=trials,
        stdev=stdev,
        uncertainty_stdev=uncertainty_stdev,
    )

    simulator_naive = RandomUncertainBinarySimulator(
        payoff=payoff,
        loss=loss,
        transaction_costs=transaction_cost,
        trials=trials,
        stdev=stdev,
        uncertainty_stdev=uncertainty_stdev,
    )

    simulator_fractional_kelly = RandomUncertainBinarySimulator(
        payoff=payoff,
        loss=loss,
        transaction_costs=transaction_cost,
        trials=trials,
        stdev=stdev,
        uncertainty_stdev=uncertainty_stdev,
    )

    random.seed(42)
    simulator_kelly.evaluate_strategy(strategy_kelly, bankroll_kelly)

    random.seed(42)
    simulator_naive.evaluate_strategy(strategy_naive, bankroll_naive)

    random.seed(42)
    simulator_fractional_kelly.evaluate_strategy(
        strategy_fractional_kelly, bankroll_fractional_kelly
    )

    assert len(bankroll_kelly.history) > 1
    assert len(bankroll_naive.history) > 1
    assert len(bankroll_fractional_kelly.history) > 1


def test_fractional_kelly_validation():
    """Test that FractionalKellyCriterion validates parameters correctly."""
    # Valid parameters
    valid_strategy = FractionalKellyCriterion(
        payoff=1, loss=1, transaction_cost=0, fraction=0.5
    )
    assert valid_strategy.fraction == 0.5

    # Fraction must be between 0 and 1
    with pytest.raises(ValueError, match="Fraction must be between 0 and 1"):
        FractionalKellyCriterion(payoff=1, loss=1, transaction_cost=0, fraction=-0.1)

    with pytest.raises(ValueError, match="Fraction must be between 0 and 1"):
        FractionalKellyCriterion(payoff=1, loss=1, transaction_cost=0, fraction=1.1)

    # Inherits validation from BaseStrategy
    with pytest.raises(ValueError, match="Payoff must be greater than 0"):
        FractionalKellyCriterion(payoff=0, loss=1, transaction_cost=0, fraction=0.5)
