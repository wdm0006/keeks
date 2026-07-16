"""Tests that the random simulators keep sampled probabilities within [0, 1].

Both random simulators draw their probabilities from an unbounded normal
distribution, so a large ``stdev`` produces samples below 0 or above 1. Those
are not probabilities, and passing them to ``strategy.evaluate`` feeds the
formulas an invalid input. The simulators clamp the samples instead.

The draws are stubbed here so each trial is fully deterministic; the strategy,
simulator, and bankroll are the real implementations.

Note that only the strategy-facing clamp is observable. ``random.random()``
returns a value in [0.0, 1.0), so an outcome threshold of 1.2 and a clamped 1.0
both always win, and -0.9 and a clamped 0.0 both always lose; the outcome clamp
states the invariant but cannot change a settlement.
"""

import random

import numpy as np
import pytest

from keeks.bankroll import BankRoll
from keeks.binary_strategies.kelly import FractionalKellyCriterion
from keeks.simulators.random_binary import RandomBinarySimulator
from keeks.simulators.random_uncertain_binary import RandomUncertainBinarySimulator


class RecordingStrategy:
    """Delegates to a real strategy while recording the probabilities it sees."""

    def __init__(self, inner):
        self.inner = inner
        self.probabilities = []

    def evaluate(self, probability, current_bankroll):
        self.probabilities.append(probability)
        return self.inner.evaluate(probability, current_bankroll)


def _strategy():
    # Half Kelly at even odds with no transaction cost: evaluate(p) is
    # 0.5 * min(2p - 1, 1.0), so 0.6 -> 0.1 and 1.0 -> 0.5.
    return RecordingStrategy(
        FractionalKellyCriterion(
            payoff=1.0, loss=1.0, transaction_cost=0.0, fraction=0.5
        )
    )


@pytest.fixture
def stub_draws(monkeypatch):
    """Return a helper that pins the normal samples and the outcome draw.

    ``probability_sample`` replaces the ``N(0.5, stdev)`` estimate draw and
    ``uncertainty_sample`` the ``N(0, uncertainty_stdev)`` draw, dispatched on
    the distribution mean the simulator asks for.
    """

    def _stub(probability_sample, uncertainty_sample=0.0, outcome_draw=0.5):
        def fake_normal(loc, _scale, _size):
            value = probability_sample if loc == 0.5 else uncertainty_sample
            return np.array([value])

        monkeypatch.setattr(np.random, "normal", fake_normal)
        monkeypatch.setattr(random, "random", lambda: outcome_draw)

    return _stub


def test_random_binary_clamps_sample_above_one(stub_draws):
    stub_draws(probability_sample=1.2)
    bankroll = BankRoll(initial_funds=1000.0)
    strategy = _strategy()

    RandomBinarySimulator(
        payoff=1.0, loss=1.0, transaction_costs=0.0, trials=1
    ).evaluate_strategy(strategy, bankroll)

    assert strategy.probabilities == [1.0]
    # Stakes 0.5 of 1000 bettable funds and wins (0.5 < 1.0).
    assert bankroll.total_funds == 1500.0
    assert bankroll.history == [1000.0, 1500.0]


def test_random_binary_clamps_sample_below_zero(stub_draws):
    stub_draws(probability_sample=-0.5)
    bankroll = BankRoll(initial_funds=1000.0)
    strategy = _strategy()

    RandomBinarySimulator(
        payoff=1.0, loss=1.0, transaction_costs=0.0, trials=1
    ).evaluate_strategy(strategy, bankroll)

    assert strategy.probabilities == [0.0]
    # Below Kelly's min_probability, so no bet is placed at all.
    assert bankroll.total_funds == 1000.0
    assert bankroll.history == [1000.0]


def test_random_binary_leaves_in_range_sample_alone(stub_draws):
    stub_draws(probability_sample=0.6)
    bankroll = BankRoll(initial_funds=1000.0)
    strategy = _strategy()

    RandomBinarySimulator(
        payoff=1.0, loss=1.0, transaction_costs=0.0, trials=1
    ).evaluate_strategy(strategy, bankroll)

    assert strategy.probabilities == [0.6]
    # Stakes 0.1 of 1000 bettable funds and wins (0.5 < 0.6).
    assert bankroll.total_funds == 1100.0
    assert bankroll.history == [1000.0, 1100.0]


def test_random_uncertain_binary_clamps_sample_above_one(stub_draws):
    stub_draws(probability_sample=1.2, uncertainty_sample=0.4)
    bankroll = BankRoll(initial_funds=1000.0)
    strategy = _strategy()

    RandomUncertainBinarySimulator(
        payoff=1.0, loss=1.0, transaction_costs=0.0, trials=1
    ).evaluate_strategy(strategy, bankroll)

    assert strategy.probabilities == [1.0]
    # Outcome threshold 1.0 + 0.4 clamps to 1.0, so the bet wins.
    assert bankroll.total_funds == 1500.0
    assert bankroll.history == [1000.0, 1500.0]


def test_random_uncertain_binary_clamps_sample_below_zero(stub_draws):
    stub_draws(probability_sample=-0.5, uncertainty_sample=0.4)
    bankroll = BankRoll(initial_funds=1000.0)
    strategy = _strategy()

    RandomUncertainBinarySimulator(
        payoff=1.0, loss=1.0, transaction_costs=0.0, trials=1
    ).evaluate_strategy(strategy, bankroll)

    assert strategy.probabilities == [0.0]
    assert bankroll.total_funds == 1000.0
    assert bankroll.history == [1000.0]


def test_random_uncertain_binary_outcome_at_negative_tail_loses(stub_draws):
    # An in-range estimate whose uncertainty pushes the outcome threshold
    # below zero: the strategy still sees 0.6 and the bet loses.
    stub_draws(probability_sample=0.6, uncertainty_sample=-1.5)
    bankroll = BankRoll(initial_funds=1000.0)
    strategy = _strategy()

    RandomUncertainBinarySimulator(
        payoff=1.0, loss=1.0, transaction_costs=0.0, trials=1
    ).evaluate_strategy(strategy, bankroll)

    assert strategy.probabilities == [0.6]
    # Stakes 0.1 of 1000 bettable funds against an outcome probability of 0.0.
    assert bankroll.total_funds == 900.0
    assert bankroll.history == [1000.0, 900.0]


def test_random_uncertain_binary_leaves_in_range_samples_alone(stub_draws):
    stub_draws(probability_sample=0.6, uncertainty_sample=0.1, outcome_draw=0.65)
    bankroll = BankRoll(initial_funds=1000.0)
    strategy = _strategy()

    RandomUncertainBinarySimulator(
        payoff=1.0, loss=1.0, transaction_costs=0.0, trials=1
    ).evaluate_strategy(strategy, bankroll)

    assert strategy.probabilities == [0.6]
    # Outcome probability 0.7 is untouched, so 0.65 still wins.
    assert bankroll.total_funds == 1100.0
    assert bankroll.history == [1000.0, 1100.0]
