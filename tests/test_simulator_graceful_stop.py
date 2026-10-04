"""Tests that simulators stop gracefully on RuinError instead of crashing.

A ``BankRoll`` configured with an explicit ``max_draw_down=0.3`` vetoes any
single settlement removing more than 30% of current funds by raising
``RuinError``. A strategy that stakes a large fraction (here 50%) trips that
limit on the first losing bet. The simulators should catch it, re-report the
refusal as a ``UserWarning`` (naming the attempted amount, the configured
limit, and current funds), and stop the run gracefully rather than letting
the error propagate out of ``evaluate_strategy`` — or stopping silently.
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
    # against a 0.3 drawdown cap trips the limit.
    return FixedFractionStrategy(
        fraction=0.5, payoff=1.0, loss=1.0, transaction_cost=0.0
    )


def test_repeated_binary_stops_gracefully():
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=0.3)
    simulator = RepeatedBinarySimulator(
        payoff=1.0, loss=1.0, transaction_costs=0.0, probability=0.7, trials=1000
    )

    # Must not raise RuinError, and the refusal is loud: the warning carries
    # the refused amount, the configured limit, and current funds.
    with pytest.warns(UserWarning, match=r"Simulation stopped early: Refused"):
        simulator.evaluate_strategy(_aggressive_strategy(), bankroll)

    # The run stopped early on the first losing bet rather than completing.
    assert len(bankroll.history) < simulator.trials
    assert all(value >= 0 for value in bankroll.history)


def test_random_binary_stops_gracefully():
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=0.3)
    simulator = RandomBinarySimulator(
        payoff=1.0, loss=1.0, transaction_costs=0.0, trials=1000
    )

    with pytest.warns(UserWarning, match=r"Simulation stopped early: Refused"):
        simulator.evaluate_strategy(_aggressive_strategy(), bankroll)

    assert all(value >= 0 for value in bankroll.history)


def test_random_uncertain_binary_stops_gracefully():
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=0.3)
    simulator = RandomUncertainBinarySimulator(
        payoff=1.0, loss=1.0, transaction_costs=0.0, trials=1000
    )

    with pytest.warns(UserWarning, match=r"Simulation stopped early: Refused"):
        simulator.evaluate_strategy(_aggressive_strategy(), bankroll)

    assert all(value >= 0 for value in bankroll.history)
