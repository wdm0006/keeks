"""Strategy stake-fraction validation shared by every simulator."""

import math

import pytest

from keeks.bankroll import BankRoll
from keeks.simulators.random_binary import RandomBinarySimulator
from keeks.simulators.random_uncertain_binary import RandomUncertainBinarySimulator
from keeks.simulators.repeated_binary import RepeatedBinarySimulator

BASE_KWARGS = {
    RepeatedBinarySimulator: {
        "payoff": 1.0,
        "loss": 1.0,
        "transaction_costs": 0.0,
        "probability": 1.0,
        "trials": 1,
        "seed": 1,
    },
    RandomBinarySimulator: {
        "payoff": 1.0,
        "loss": 1.0,
        "transaction_costs": 0.0,
        "trials": 1,
        "stdev": 0.0,
        "seed": 1,
    },
    RandomUncertainBinarySimulator: {
        "payoff": 1.0,
        "loss": 1.0,
        "transaction_costs": 0.0,
        "trials": 1,
        "stdev": 0.0,
        "uncertainty_stdev": 0.0,
        "seed": 1,
    },
}

SIMULATORS = list(BASE_KWARGS)
INVALID_FRACTIONS = [-0.01, 1.01, math.nan, math.inf, -math.inf, "not numeric"]


def build(simulator_cls, **overrides):
    return simulator_cls(**{**BASE_KWARGS[simulator_cls], **overrides})


class Strategy:
    def __init__(self, fraction):
        self.fraction = fraction
        self.probabilities = []
        self.results = []

    def evaluate(self, probability, _current_bankroll):
        self.probabilities.append(probability)
        return self.fraction

    def record_result(self, won, return_pct):
        self.results.append((won, return_pct))


@pytest.mark.parametrize("simulator_cls", SIMULATORS)
@pytest.mark.parametrize("fraction", INVALID_FRACTIONS)
def test_invalid_fraction_is_rejected_without_side_effects(simulator_cls, fraction):
    bankroll = BankRoll(initial_funds=100.0, max_draw_down=None)
    strategy = Strategy(fraction)

    with pytest.raises(ValueError, match="Strategy stake fraction"):
        build(simulator_cls).evaluate_strategy(strategy, bankroll)

    assert bankroll.total_funds == 100.0
    assert bankroll.history == [100.0]
    assert strategy.results == []


@pytest.mark.parametrize("simulator_cls", SIMULATORS)
def test_rejected_fraction_does_not_shift_seeded_run(simulator_cls):
    overrides = {"trials": 5}
    if simulator_cls is not RepeatedBinarySimulator:
        overrides["stdev"] = 0.1
    if simulator_cls is RandomUncertainBinarySimulator:
        overrides["uncertainty_stdev"] = 0.05

    reused = build(simulator_cls, **overrides)
    with pytest.raises(ValueError):
        reused.evaluate_strategy(Strategy(2.0), BankRoll(initial_funds=100.0))

    reused_bankroll = BankRoll(initial_funds=100.0, max_draw_down=None)
    fresh_bankroll = BankRoll(initial_funds=100.0, max_draw_down=None)
    reused_strategy = Strategy(0.1)
    fresh_strategy = Strategy(0.1)
    reused.evaluate_strategy(reused_strategy, reused_bankroll)
    build(simulator_cls, **overrides).evaluate_strategy(fresh_strategy, fresh_bankroll)

    assert reused_strategy.probabilities == fresh_strategy.probabilities
    assert reused_bankroll.history == fresh_bankroll.history


@pytest.mark.parametrize("simulator_cls", SIMULATORS)
@pytest.mark.parametrize(
    ("fraction", "expected_history"),
    [(0.0, [100.0]), (1.0, [100.0, 200.0])],
)
def test_fraction_boundaries(simulator_cls, fraction, expected_history):
    bankroll = BankRoll(initial_funds=100.0, max_draw_down=None)

    build(simulator_cls).evaluate_strategy(Strategy(fraction), bankroll)

    assert bankroll.history == expected_history
