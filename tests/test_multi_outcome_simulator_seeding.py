"""Seeding-replay tests for the multi-outcome simulator.

Direct ports of the ``tests/test_simulator_seeding.py`` guarantees to the
multi-outcome settlement stream: the same seed replays byte-identically, a
different seed diverges, and a seeded run consumes nothing from the
process-global generators.
"""

import random

import numpy as np

from keeks.bankroll import BankRoll
from keeks.multi_outcome import RepeatedMultiOutcomeSimulator
from keeks.multi_outcome.base import _validate_stake_fractions


class _FixedFractionMultiOutcome:
    """Duck-typed fixed-fraction strategy: the same stakes every trial."""

    def __init__(self, stakes):
        self._stakes = stakes

    def evaluate(self, _probabilities, _current_bankroll):
        return _validate_stake_fractions(self._stakes)


def run_simulation(seed):
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
    strategy = _FixedFractionMultiOutcome(stakes=(0.1, 0.05, 0.0))
    simulator = RepeatedMultiOutcomeSimulator(
        payoffs=(2.0, 3.0, 2.5),
        loss=1.0,
        transaction_costs=0.0,
        probabilities=(0.4, 0.35, 0.2),
        trials=50,
        seed=seed,
    )
    simulator.evaluate_strategy(strategy, bankroll)
    return bankroll.history


def test_same_seed_replays_identical_history():
    assert run_simulation(42) == run_simulation(42)


def test_different_seeds_produce_different_histories():
    assert run_simulation(42) != run_simulation(43)


def test_seeded_simulation_does_not_consume_global_generators():
    random.seed(12345)
    np.random.seed(12345)
    expected_random = random.random()
    expected_numpy = np.random.random()

    random.seed(12345)
    np.random.seed(12345)
    run_simulation(42)

    assert random.random() == expected_random
    assert np.random.random() == expected_numpy
