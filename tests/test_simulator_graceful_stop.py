"""Tests that simulators stop gracefully on RuinError instead of crashing.

A default ``BankRoll`` uses ``max_draw_down=0.3``. A strategy that stakes a
large fraction (here 50%) trips that limit on the first losing bet, which
``BankRoll.withdraw`` reports by raising ``RuinError``. The simulators should
catch it and stop the run gracefully rather than letting it propagate out of
``evaluate_strategy``.
"""

import random

import numpy as np
import pytest

from keeks.bankroll import BankRoll
from keeks.binary_strategies.simple import FixedFractionStrategy
from keeks.simulators.random_binary import RandomBinarySimulator
from keeks.simulators.random_uncertain_binary import RandomUncertainBinarySimulator
from keeks.simulators.repeated_binary import RepeatedBinarySimulator


@pytest.fixture(autouse=True)
def _seeded():
    """Seed both RNGs so a losing bet reliably occurs within the trials."""
    random.seed(42)
    np.random.seed(42)


def _aggressive_strategy():
    # Stakes 50% of the bankroll whenever probability >= 0.5, so a single loss
    # against a default BankRoll (max_draw_down=0.3) trips the drawdown limit.
    return FixedFractionStrategy(
        fraction=0.5, payoff=1.0, loss=1.0, transaction_cost=0.0
    )


def test_repeated_binary_stops_gracefully():
    bankroll = BankRoll(initial_funds=1000.0)  # default max_draw_down=0.3
    simulator = RepeatedBinarySimulator(
        payoff=1.0, loss=1.0, transaction_costs=0.0, probability=0.7, trials=1000
    )

    # Must not raise RuinError.
    simulator.evaluate_strategy(_aggressive_strategy(), bankroll)

    # The run stopped early on the first losing bet rather than completing.
    assert len(bankroll.history) < simulator.trials
    assert all(value >= 0 for value in bankroll.history)


def test_random_binary_stops_gracefully():
    bankroll = BankRoll(initial_funds=1000.0)
    simulator = RandomBinarySimulator(
        payoff=1.0, loss=1.0, transaction_costs=0.0, trials=1000
    )

    simulator.evaluate_strategy(_aggressive_strategy(), bankroll)

    assert all(value >= 0 for value in bankroll.history)


def test_random_uncertain_binary_stops_gracefully():
    bankroll = BankRoll(initial_funds=1000.0)
    simulator = RandomUncertainBinarySimulator(
        payoff=1.0, loss=1.0, transaction_costs=0.0, trials=1000
    )

    simulator.evaluate_strategy(_aggressive_strategy(), bankroll)

    assert all(value >= 0 for value in bankroll.history)
