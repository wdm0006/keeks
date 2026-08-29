"""Shared argument validation for every exported strategy's evaluate method."""

import math

import pytest

import keeks.binary_strategies as binary_strategies
from tests.test_strategy_configuration_validation import BASE_KWARGS

STRATEGIES = list(BASE_KWARGS)
INVALID_PROBABILITIES = [2.0, 1.5, -0.5, math.nan, math.inf]


def build(strategy_cls):
    """Construct ``strategy_cls`` from its valid base kwargs."""
    return strategy_cls(**BASE_KWARGS[strategy_cls])


def test_base_kwargs_cover_every_exported_strategy():
    """Guard: a newly exported strategy must be added to these cases."""
    assert {cls.__name__ for cls in STRATEGIES} == set(binary_strategies.__all__)


@pytest.mark.parametrize("strategy_cls", STRATEGIES)
@pytest.mark.parametrize("probability", INVALID_PROBABILITIES)
def test_invalid_probability_rejected(strategy_cls, probability):
    with pytest.raises(ValueError):
        build(strategy_cls).evaluate(probability, 1000.0)


@pytest.mark.parametrize("strategy_cls", STRATEGIES)
@pytest.mark.parametrize("probability", [0.4, 0.6])
@pytest.mark.parametrize("current_bankroll", [math.nan, math.inf, -math.inf])
def test_non_finite_bankroll_rejected(strategy_cls, probability, current_bankroll):
    with pytest.raises(ValueError):
        build(strategy_cls).evaluate(probability, current_bankroll)


@pytest.mark.parametrize("strategy_cls", STRATEGIES)
@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_boundary_probability_accepted(strategy_cls, probability):
    proportion = build(strategy_cls).evaluate(probability, 1000.0)

    assert 0.0 <= proportion <= 1.0


def test_get_max_safe_bet_rejects_nan_bankroll():
    with pytest.raises(ValueError):
        build(STRATEGIES[0]).get_max_safe_bet(math.nan)


@pytest.mark.parametrize("current_bankroll", [0.0, -100.0])
def test_get_max_safe_bet_keeps_nonpositive_bankroll_semantics(current_bankroll):
    assert build(STRATEGIES[0]).get_max_safe_bet(current_bankroll) == 0.0
