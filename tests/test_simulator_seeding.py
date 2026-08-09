import random

import numpy as np
import pytest

from keeks.bankroll import BankRoll
from keeks.binary_strategies.simple import FixedFractionStrategy

from .test_simulator_configuration_validation import SIMULATORS, build


def run_simulation(simulator_cls, seed):
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
    strategy = FixedFractionStrategy(
        fraction=0.1, payoff=1.0, loss=1.0, transaction_cost=0.0
    )
    simulator = build(
        simulator_cls,
        seed=seed,
        trials=50,
        transaction_costs=0.0,
    )
    simulator.evaluate_strategy(strategy, bankroll)
    return bankroll.history


@pytest.mark.parametrize("simulator_cls", SIMULATORS)
def test_same_seed_replays_identical_history(simulator_cls):
    assert run_simulation(simulator_cls, 42) == run_simulation(simulator_cls, 42)


@pytest.mark.parametrize("simulator_cls", SIMULATORS)
def test_different_seeds_produce_different_histories(simulator_cls):
    assert run_simulation(simulator_cls, 42) != run_simulation(simulator_cls, 43)


@pytest.mark.parametrize("simulator_cls", SIMULATORS)
def test_seeded_simulation_does_not_consume_global_generators(simulator_cls):
    random.seed(12345)
    np.random.seed(12345)
    expected_random = random.random()
    expected_numpy = np.random.random()

    random.seed(12345)
    np.random.seed(12345)
    run_simulation(simulator_cls, 42)

    assert random.random() == expected_random
    assert np.random.random() == expected_numpy
