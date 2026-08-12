"""Constructor validation of the economic controls shared by every strategy.

``BaseStrategy.__init__`` used to check only signs, and IEEE NaN makes every
comparison false, so ``payoff=nan``, ``loss=nan`` or ``transaction_cost=nan``
reached the sizing formulas of all nine shipped strategies; positive infinity
passed for ``payoff`` and ``loss`` too. Those configurations now fail fast with
``ValueError``, matching the simulator constructors.

The accepted ranges are unchanged: ``payoff > 0``, ``loss >= 0``,
``transaction_cost >= 0``, and ``loss + transaction_cost > 0``.
"""

import math

import pytest

import keeks.binary_strategies as binary_strategies
from keeks.binary_strategies import (
    CPPIStrategy,
    DrawdownAdjustedKelly,
    DynamicBankrollManagement,
    FixedFractionStrategy,
    FractionalKellyCriterion,
    KellyCriterion,
    MertonShare,
    NaiveStrategy,
    OptimalF,
)

# One valid kwargs set per strategy exported from keeks.binary_strategies.__all__,
# the strategy-side analogue of the simulator suite's BASE_KWARGS.
BASE_KWARGS = {
    KellyCriterion: {"payoff": 1.0, "loss": 1.0, "transaction_cost": 0.01},
    FractionalKellyCriterion: {
        "payoff": 1.0,
        "loss": 1.0,
        "transaction_cost": 0.01,
        "fraction": 0.5,
    },
    DrawdownAdjustedKelly: {
        "payoff": 1.0,
        "loss": 1.0,
        "transaction_cost": 0.01,
        "max_acceptable_drawdown": 0.2,
    },
    OptimalF: {
        "payoff": 1.0,
        "loss": 1.0,
        "transaction_cost": 0.01,
        "win_rate": 0.6,
    },
    NaiveStrategy: {"payoff": 1.0, "loss": 1.0, "transaction_cost": 0.01},
    FixedFractionStrategy: {
        "fraction": 0.1,
        "payoff": 1.0,
        "loss": 1.0,
        "transaction_cost": 0.01,
    },
    CPPIStrategy: {
        "floor_fraction": 0.5,
        "multiplier": 2.0,
        "initial_bankroll": 1000.0,
        "payoff": 1.0,
        "loss": 1.0,
        "transaction_cost": 0.01,
    },
    DynamicBankrollManagement: {
        "base_fraction": 0.1,
        "payoff": 1.0,
        "loss": 1.0,
        "transaction_cost": 0.01,
    },
    MertonShare: {"payoff": 1.0, "loss": 1.0, "transaction_cost": 0.01},
}

STRATEGIES = list(BASE_KWARGS)

NON_FINITE = [math.nan, math.inf, -math.inf]

INVALID_PAYOFF = [*NON_FINITE, 0.0, -1.0, None, "abc"]
INVALID_LOSS = [*NON_FINITE, -0.01, -1.0, None, "abc"]
INVALID_TRANSACTION_COST = [*NON_FINITE, -0.01, -10, None, "abc"]


def build(strategy_cls, **overrides):
    """Construct ``strategy_cls`` from its valid base kwargs plus overrides."""
    return strategy_cls(**{**BASE_KWARGS[strategy_cls], **overrides})


def test_base_kwargs_cover_every_exported_strategy():
    """Guard: a newly exported strategy must be added to these cases."""
    assert {cls.__name__ for cls in STRATEGIES} == set(binary_strategies.__all__)


@pytest.mark.parametrize("strategy_cls", STRATEGIES)
@pytest.mark.parametrize("field", ["payoff", "loss", "transaction_cost"])
@pytest.mark.parametrize("value", NON_FINITE)
def test_non_finite_economics_rejected(strategy_cls, field, value):
    """NaN and both infinities are rejected for every shared economic control."""
    with pytest.raises(ValueError):
        build(strategy_cls, **{field: value})


@pytest.mark.parametrize("strategy_cls", STRATEGIES)
@pytest.mark.parametrize(
    ("field", "invalid_values"),
    [
        ("payoff", INVALID_PAYOFF),
        ("loss", INVALID_LOSS),
        ("transaction_cost", INVALID_TRANSACTION_COST),
    ],
)
def test_shared_economics_rejected(strategy_cls, field, invalid_values):
    """The pre-existing range checks still reject their own invalid values."""
    for value in invalid_values:
        with pytest.raises(ValueError):
            build(strategy_cls, **{field: value})


@pytest.mark.parametrize("strategy_cls", STRATEGIES)
def test_zero_total_cost_still_rejected(strategy_cls):
    """``loss + transaction_cost`` must remain strictly positive."""
    with pytest.raises(ValueError):
        build(strategy_cls, loss=0.0, transaction_cost=0.0)


@pytest.mark.parametrize("strategy_cls", STRATEGIES)
def test_boundary_economics_accepted(strategy_cls):
    """A zero loss and a zero fee stay valid as long as their sum is positive."""
    zero_loss = build(strategy_cls, loss=0.0, transaction_cost=0.01)
    assert zero_loss.loss == 0.0
    assert zero_loss.transaction_cost == 0.01

    zero_cost = build(strategy_cls, loss=1.0, transaction_cost=0.0)
    assert zero_cost.loss == 1.0
    assert zero_cost.transaction_cost == 0.0


@pytest.mark.parametrize("strategy_cls", STRATEGIES)
def test_accepted_economics_stored_as_floats(strategy_cls):
    """Integer inputs are coerced, matching the simulator validation convention."""
    strategy = build(strategy_cls, payoff=2, loss=1, transaction_cost=0)

    assert isinstance(strategy.payoff, float)
    assert isinstance(strategy.loss, float)
    assert isinstance(strategy.transaction_cost, float)
    assert (strategy.payoff, strategy.loss, strategy.transaction_cost) == (
        2.0,
        1.0,
        0.0,
    )


@pytest.mark.parametrize("strategy_cls", STRATEGIES)
def test_valid_configuration_still_evaluates(strategy_cls):
    """Validation is the only change: a valid configuration sizes bets as before."""
    strategy = build(strategy_cls)

    proportion = strategy.evaluate(0.6, 1000.0)

    assert 0.0 <= proportion <= 1.0
