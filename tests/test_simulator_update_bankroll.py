import pytest

from keeks.bankroll import BankRoll
from keeks.binary_strategies import CPPIStrategy
from keeks.simulators import (
    RandomBinarySimulator,
    RandomUncertainBinarySimulator,
    RepeatedBinarySimulator,
)


def build_simulator(simulator_cls):
    common = {"payoff": 1.0, "loss": 1.0, "transaction_costs": 0.0, "seed": 11}
    if simulator_cls is RepeatedBinarySimulator:
        return simulator_cls(**common, probability=0.6, trials=5)
    return simulator_cls(**common, trials=5)


@pytest.mark.parametrize(
    ("simulator_cls", "expected_updates"),
    [
        (RepeatedBinarySimulator, [1000, 1100.0, 1210.0, 1089.0, 1197.9]),
        (RandomBinarySimulator, [1000, 1100.0, 1210.0, 1089.0, 980.1]),
        (RandomUncertainBinarySimulator, [1000, 1100.0, 1210.0, 1089.0, 1197.9]),
    ],
)
def test_update_bankroll_precedes_evaluate_each_trial(simulator_cls, expected_updates):
    class SpyStrategy:
        def __init__(self):
            self.events = []

        def update_bankroll(self, current_bankroll):
            self.events.append(("update_bankroll", current_bankroll))

        def evaluate(self, _probability, current_bankroll):
            self.events.append(("evaluate", current_bankroll))
            return 0.1

        def record_result(self, _won, _return_pct):
            self.events.append(("record_result", None))

    strategy = SpyStrategy()
    build_simulator(simulator_cls).evaluate_strategy(
        strategy, BankRoll(initial_funds=1000.0, max_draw_down=None)
    )

    assert strategy.events == [
        event
        for bankroll in expected_updates
        for event in (
            ("update_bankroll", bankroll),
            ("evaluate", bankroll),
            ("record_result", None),
        )
    ]


@pytest.mark.parametrize(
    "simulator_cls",
    [RepeatedBinarySimulator, RandomBinarySimulator, RandomUncertainBinarySimulator],
)
def test_noncallable_update_bankroll_is_ignored(simulator_cls):
    class Strategy:
        update_bankroll = "not callable"

        def evaluate(self, _probability, _current_bankroll):
            return 0.0

    build_simulator(simulator_cls).evaluate_strategy(
        Strategy(), BankRoll(initial_funds=1000.0)
    )


@pytest.mark.parametrize(
    ("simulator_cls", "expected_history"),
    [
        (
            RepeatedBinarySimulator,
            [1000, 1200.0, 1440.0, 1152.0, 1324.8, 1566.72],
        ),
        (RandomBinarySimulator, [1000, 1006.84, 1280.65, 966.96]),
        (RandomUncertainBinarySimulator, [1000, 1006.84, 1253.46, 1110.63]),
    ],
)
def test_cppi_seeded_histories_remain_unchanged(simulator_cls, expected_history):
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
    strategy = CPPIStrategy(
        floor_fraction=0.5,
        multiplier=2.0,
        initial_bankroll=1000.0,
        payoff=1.0,
        loss=1.0,
    )

    build_simulator(simulator_cls).evaluate_strategy(strategy, bankroll)

    assert bankroll.history == expected_history
