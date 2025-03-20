import random

import pytest

from keeks.bankroll import BankRoll
from keeks.binary_strategies import NaiveStrategy
from keeks.utils import RuinError

random.seed(42)


def test_even_odds():
    strategy = NaiveStrategy(payoff=1, loss=1, transaction_cost=0)
    current_bankroll = 1000

    # 60% chance of winning
    portion = strategy.evaluate(0.6, current_bankroll)
    assert portion == pytest.approx(0.2)  # (0.6 - 0.4) = 0.2

    # 40% chance of winning (negative EV)
    portion = strategy.evaluate(0.4, current_bankroll)
    assert portion == pytest.approx(0.0)  # Should return 0 for negative EV


def test_simulation():
    payoff = 1
    loss = 1
    transaction_cost = 0.01
    probability = 0.55  # Positive EV but with high risk
    trials = 1_000
    initial_bankroll = 1000

    bankroll = BankRoll(
        initial_funds=initial_bankroll, percent_bettable=1, max_draw_down=0.3
    )
    strategy = NaiveStrategy(payoff, loss, transaction_cost)

    # Track bankroll history
    bankroll_history = [initial_bankroll]

    # Set random seed for reproducibility
    random.seed(42)

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
        else:
            # Loss
            bankroll.remove_funds(bet_amount * (loss + transaction_cost))

        # Record bankroll history
        bankroll_history.append(bankroll.total_funds)

        # Check for ruin
        if bankroll.total_funds < initial_bankroll * 0.3:  # 70% drawdown
            raise RuinError("Bankroll dropped below maximum drawdown")

    # Should have grown with positive edge over many trials
    assert bankroll.total_funds > initial_bankroll * 0.9  # Allow for some variance


def test_known_cases():
    strategy = NaiveStrategy(payoff=2, loss=1, transaction_cost=0)
    current_bankroll = 1000

    # 50% chance of winning with 2:1 payoff
    # EV = (0.5 * 2 - 0.5) = 0.5, bet size = 0.5/2 = 0.25
    assert strategy.evaluate(0.5, current_bankroll) == pytest.approx(0.25)

    # 60% chance of winning with 2:1 payoff
    # EV = (0.6 * 2 - 0.4) = 0.8, bet size = 0.8/2 = 0.4
    assert strategy.evaluate(0.6, current_bankroll) == pytest.approx(0.4)

    # 40% chance of winning with 2:1 payoff
    # EV = (0.4 * 2 - 0.6) = 0.2, bet size = 0.2/2 = 0.1
    assert strategy.evaluate(0.4, current_bankroll) == pytest.approx(0.1)


def test_transaction_costs():
    strategy = NaiveStrategy(payoff=2, loss=1, transaction_cost=0.01)
    current_bankroll = 1000

    # 60% chance of winning with 2:1 payoff and 1% transaction cost
    # EV = (0.6 * (2 - 0.01) - 0.4 * (1 + 0.01)) = 0.79
    # Bet size = 0.79/2 = 0.395
    assert strategy.evaluate(0.6, current_bankroll) == pytest.approx(0.395)


def test_zero_probability():
    strategy = NaiveStrategy(payoff=2, loss=1, transaction_cost=0)
    current_bankroll = 1000

    # 0% chance of winning should return 0 bet size
    assert strategy.evaluate(0.0, current_bankroll) == pytest.approx(0.0)


def test_one_probability():
    strategy = NaiveStrategy(payoff=2, loss=1, transaction_cost=0)
    current_bankroll = 1000

    # 100% chance of winning should bet maximum safe amount
    assert strategy.evaluate(1.0, current_bankroll) == pytest.approx(1.0)
