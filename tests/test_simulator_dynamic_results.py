import random

import numpy as np
import pytest

from keeks.bankroll import BankRoll
from keeks.binary_strategies.simple import DynamicBankrollManagement
from keeks.simulators.random_binary import RandomBinarySimulator
from keeks.simulators.random_uncertain_binary import RandomUncertainBinarySimulator
from keeks.simulators.repeated_binary import RepeatedBinarySimulator


def _strategy(**overrides):
    parameters = {
        "base_fraction": 0.1,
        "payoff": 1.0,
        "loss": 1.0,
        "transaction_cost": 0.0,
        "window_size": 2,
        "max_fraction": 0.5,
        "min_fraction": 0.0,
    }
    parameters.update(overrides)
    return DynamicBankrollManagement(**parameters)


@pytest.fixture(params=["repeated", "random", "uncertain"])
def simulator(request, monkeypatch):
    if request.param == "repeated":
        return RepeatedBinarySimulator(1.0, 1.0, 0.0, 0.75, trials=2)

    monkeypatch.setattr(
        np.random,
        "normal",
        lambda loc, _scale, _size: [0.75 if loc == 0.5 else 0.0],
    )
    if request.param == "random":
        return RandomBinarySimulator(1.0, 1.0, 0.0, trials=2)
    return RandomUncertainBinarySimulator(
        1.0, 1.0, 0.0, trials=2, uncertainty_stdev=0.0
    )


def test_settled_results_change_dynamic_bet_sizing(simulator, monkeypatch):
    outcomes = iter([0.5, 0.9])
    monkeypatch.setattr(random, "random", lambda: next(outcomes))
    strategy = _strategy()
    bankroll = BankRoll(initial_funds=100.0, max_draw_down=None)

    initial_bet = strategy.evaluate(0.75, bankroll.total_funds)
    simulator.evaluate_strategy(strategy, bankroll)

    assert initial_bet == pytest.approx(0.1)
    assert strategy.results == pytest.approx([0.1, -0.15625])
    assert strategy.evaluate(0.75, bankroll.total_funds) != pytest.approx(initial_bet)


def test_skipped_bet_does_not_record_result(simulator):
    strategy = _strategy(min_probability=0.8)

    simulator.evaluate_strategy(strategy, BankRoll(initial_funds=100.0))

    assert strategy.results == []


def test_rejected_settlement_does_not_record_result(simulator, monkeypatch):
    monkeypatch.setattr(random, "random", lambda: 0.9)
    strategy = _strategy(base_fraction=0.5)

    simulator.evaluate_strategy(strategy, BankRoll(initial_funds=100.0))

    assert strategy.results == []
