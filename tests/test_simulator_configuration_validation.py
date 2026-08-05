"""Constructor validation for the three simulators.

Simulator constructors used to assign their controls unchecked, so a nonsensical
configuration either distorted results silently or failed much later than
configuration: a negative flat ``transaction_costs`` turned the per-bet fee into
a subsidy, a fixed ``probability`` above 1 made every outcome a win, and a
negative ``loss`` reached ``BankRoll.withdraw`` as a negative settlement amount.
They now fail fast with ``ValueError``, matching ``BaseStrategy.__init__``.

The flat per-bet fee semantics are unchanged; only rejection at construction is
new.
"""

import math

import pytest

from keeks.bankroll import BankRoll
from keeks.binary_strategies.simple import FixedFractionStrategy
from keeks.simulators.random_binary import RandomBinarySimulator
from keeks.simulators.random_uncertain_binary import RandomUncertainBinarySimulator
from keeks.simulators.repeated_binary import RepeatedBinarySimulator

BASE_KWARGS = {
    RepeatedBinarySimulator: {
        "payoff": 1.0,
        "loss": 1.0,
        "transaction_costs": 0.01,
        "probability": 0.55,
        "trials": 10,
    },
    RandomBinarySimulator: {
        "payoff": 1.0,
        "loss": 1.0,
        "transaction_costs": 0.01,
        "trials": 10,
        "stdev": 0.1,
    },
    RandomUncertainBinarySimulator: {
        "payoff": 1.0,
        "loss": 1.0,
        "transaction_costs": 0.01,
        "trials": 10,
        "stdev": 0.1,
        "uncertainty_stdev": 0.05,
    },
}

SIMULATORS = list(BASE_KWARGS)
RANDOM_SIMULATORS = [RandomBinarySimulator, RandomUncertainBinarySimulator]

INVALID_PAYOFF = [0.0, -1.0, math.nan, math.inf, -math.inf, None, "abc"]
INVALID_LOSS = [-0.01, -1.0, math.nan, math.inf, -math.inf, None, "abc"]
INVALID_TRANSACTION_COSTS = [-0.01, -10, math.nan, math.inf, -math.inf, None, "abc"]
INVALID_TRIALS = [-1, -1000, 1.5, 10.0, math.nan, math.inf, None, "10"]
INVALID_PROBABILITY = [-0.01, 1.01, 2, math.nan, math.inf, -math.inf, None, "abc"]
INVALID_STDEV = [-0.01, -1.0, math.nan, math.inf, -math.inf, None, "abc"]


def build(simulator_cls, **overrides):
    """Construct ``simulator_cls`` from its valid base kwargs plus overrides."""
    return simulator_cls(**{**BASE_KWARGS[simulator_cls], **overrides})


def test_base_kwargs_cover_every_simulator():
    """Guard: every simulator this suite validates is exercised here."""
    assert set(SIMULATORS) == {
        RepeatedBinarySimulator,
        RandomBinarySimulator,
        RandomUncertainBinarySimulator,
    }


@pytest.mark.parametrize("simulator_cls", SIMULATORS)
@pytest.mark.parametrize(
    ("field", "invalid_values"),
    [
        ("payoff", INVALID_PAYOFF),
        ("loss", INVALID_LOSS),
        ("transaction_costs", INVALID_TRANSACTION_COSTS),
        ("trials", INVALID_TRIALS),
    ],
)
def test_shared_controls_rejected(simulator_cls, field, invalid_values):
    for value in invalid_values:
        with pytest.raises(ValueError):
            build(simulator_cls, **{field: value})


@pytest.mark.parametrize("simulator_cls", SIMULATORS)
def test_shared_control_boundaries_accepted(simulator_cls):
    """``loss=0``, a zero fee and ``trials=0`` are meaningful configurations."""
    simulator = build(simulator_cls, loss=0.0, transaction_costs=0.0, trials=0)
    assert simulator.loss == 0.0
    assert simulator.transaction_costs == 0.0
    assert simulator.trials == 0
    assert simulator.payoff == 1.0


@pytest.mark.parametrize("probability", INVALID_PROBABILITY)
def test_fixed_probability_rejected(probability):
    with pytest.raises(ValueError):
        build(RepeatedBinarySimulator, probability=probability)


@pytest.mark.parametrize("probability", [0.0, 0.5, 1.0])
def test_fixed_probability_boundaries_accepted(probability):
    simulator = build(RepeatedBinarySimulator, probability=probability)
    assert simulator.probability == probability


@pytest.mark.parametrize("simulator_cls", RANDOM_SIMULATORS)
@pytest.mark.parametrize("stdev", INVALID_STDEV)
def test_stdev_rejected(simulator_cls, stdev):
    with pytest.raises(ValueError):
        build(simulator_cls, stdev=stdev)


@pytest.mark.parametrize("uncertainty_stdev", INVALID_STDEV)
def test_uncertainty_stdev_rejected(uncertainty_stdev):
    with pytest.raises(ValueError):
        build(RandomUncertainBinarySimulator, uncertainty_stdev=uncertainty_stdev)


@pytest.mark.parametrize("simulator_cls", RANDOM_SIMULATORS)
def test_zero_stdev_accepted(simulator_cls):
    simulator = build(simulator_cls, stdev=0.0)
    assert simulator.stdev == 0.0


def test_zero_uncertainty_stdev_accepted():
    simulator = build(RandomUncertainBinarySimulator, uncertainty_stdev=0.0)
    assert simulator.uncertainty_stdev == 0.0


def test_negative_fee_no_longer_manufactures_gains():
    """A negative flat fee used to pay the bettor on every settled bet."""
    with pytest.raises(ValueError):
        RepeatedBinarySimulator(
            payoff=1.0,
            loss=1.0,
            transaction_costs=-10,
            probability=0.55,
            trials=1,
        )


def test_valid_configuration_still_simulates():
    """Validation is the only change: a valid configuration runs as before."""
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
    strategy = FixedFractionStrategy(
        fraction=0.1, payoff=1.0, loss=1.0, transaction_cost=0.01
    )
    simulator = build(RepeatedBinarySimulator, trials=25)

    simulator.evaluate_strategy(strategy, bankroll)

    assert len(bankroll.history) > 1
    assert bankroll.total_funds > 0
