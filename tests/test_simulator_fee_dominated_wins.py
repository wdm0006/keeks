import pytest

from keeks.bankroll import BankRoll
from keeks.binary_strategies.simple import FixedFractionStrategy
from keeks.simulators.random_binary import RandomBinarySimulator
from keeks.simulators.random_uncertain_binary import RandomUncertainBinarySimulator
from keeks.simulators.repeated_binary import RepeatedBinarySimulator


@pytest.fixture(params=["repeated", "random", "uncertain"])
def simulator(request, monkeypatch):
    monkeypatch.setattr("random.random", lambda: 0.0)

    if request.param == "repeated":
        return RepeatedBinarySimulator(1.0, 1.0, 0.0, probability=0.9, trials=1)

    monkeypatch.setattr("numpy.random.normal", lambda *_args: [0.9])
    if request.param == "random":
        return RandomBinarySimulator(1.0, 1.0, 0.0, trials=1)

    return RandomUncertainBinarySimulator(1.0, 1.0, 0.0, trials=1)


def _strategy():
    return FixedFractionStrategy(
        fraction=0.1, payoff=1.0, loss=1.0, transaction_cost=0.0
    )


def test_nonnegative_win_is_deposited(simulator):
    simulator.transaction_costs = 0.5
    bankroll = BankRoll(initial_funds=10.0, max_draw_down=0.3)

    simulator.evaluate_strategy(_strategy(), bankroll)

    assert bankroll.total_funds == 10.5
    assert bankroll.history == [10.0, 10.5]


def test_affordable_fee_dominated_win_is_withdrawn(simulator):
    simulator.transaction_costs = 2.0
    bankroll = BankRoll(initial_funds=10.0, max_draw_down=0.3)

    simulator.evaluate_strategy(_strategy(), bankroll)

    assert bankroll.total_funds == 9.0
    assert bankroll.history == [10.0, 9.0]


def test_fee_dominated_win_exceeding_drawdown_stops_without_mutation(simulator):
    simulator.transaction_costs = 5.0
    bankroll = BankRoll(initial_funds=10.0, max_draw_down=0.3)

    simulator.evaluate_strategy(_strategy(), bankroll)

    assert bankroll.total_funds == 10.0
    assert bankroll.history == [10.0]


def test_fee_dominated_win_exceeding_funds_stops_without_mutation(simulator):
    simulator.transaction_costs = 12.0
    bankroll = BankRoll(initial_funds=10.0, max_draw_down=None)

    simulator.evaluate_strategy(_strategy(), bankroll)

    assert bankroll.total_funds == 10.0
    assert bankroll.history == [10.0]
