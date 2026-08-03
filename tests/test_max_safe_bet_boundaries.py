"""Boundary behaviour of the shared maximum-safe-bet clamp.

Every shipped strategy ends ``evaluate`` with ``get_max_safe_bet``, so the answer
at a depleted or negative bankroll is a property of the whole public API.
"""

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
from keeks.binary_strategies.base import BaseStrategy

# One factory per strategy exported from keeks.binary_strategies.__all__. Factories
# rather than instances because CPPIStrategy and DynamicBankrollManagement carry
# bankroll state between evaluate() calls.
STRATEGY_FACTORIES = {
    "KellyCriterion": lambda: KellyCriterion(
        payoff=1.0, loss=1.0, transaction_cost=0.01
    ),
    "FractionalKellyCriterion": lambda: FractionalKellyCriterion(
        payoff=1.0, loss=1.0, transaction_cost=0.01, fraction=0.5
    ),
    "DrawdownAdjustedKelly": lambda: DrawdownAdjustedKelly(
        payoff=1.0, loss=1.0, transaction_cost=0.01, max_acceptable_drawdown=0.2
    ),
    "OptimalF": lambda: OptimalF(
        payoff=1.0, loss=1.0, transaction_cost=0.01, win_rate=0.6
    ),
    "NaiveStrategy": lambda: NaiveStrategy(payoff=1.0, loss=1.0, transaction_cost=0.01),
    "FixedFractionStrategy": lambda: FixedFractionStrategy(
        fraction=0.1, payoff=1.0, loss=1.0, transaction_cost=0.01
    ),
    "CPPIStrategy": lambda: CPPIStrategy(
        floor_fraction=0.5,
        multiplier=2.0,
        initial_bankroll=1000.0,
        payoff=1.0,
        loss=1.0,
        transaction_cost=0.01,
    ),
    "DynamicBankrollManagement": lambda: DynamicBankrollManagement(
        base_fraction=0.1, payoff=1.0, loss=1.0, transaction_cost=0.01
    ),
    "MertonShare": lambda: MertonShare(payoff=1.0, loss=1.0, transaction_cost=0.01),
}


class _ConcreteStrategy(BaseStrategy):
    """Minimal concrete subclass so BaseStrategy.get_max_safe_bet can be called."""

    def evaluate(self, _probability, current_bankroll):
        return self.get_max_safe_bet(current_bankroll)


def test_factories_cover_every_exported_strategy():
    """Guard: a newly exported strategy must be added to the boundary cases."""
    assert set(STRATEGY_FACTORIES) == set(binary_strategies.__all__)


@pytest.mark.parametrize("name", sorted(STRATEGY_FACTORIES))
@pytest.mark.parametrize("bankroll", [0.0, -100.0])
def test_evaluate_returns_zero_at_non_positive_bankroll(name, bankroll):
    """No strategy stakes anything - or raises - on an empty or negative bankroll."""
    strategy = STRATEGY_FACTORIES[name]()

    assert strategy.evaluate(0.7, bankroll) == 0.0


@pytest.mark.parametrize("bankroll", [0.0, -0.01, -100.0])
def test_get_max_safe_bet_zero_at_non_positive_bankroll(bankroll):
    """The clamp itself returns 0.0 rather than dividing by the bankroll."""
    strategy = _ConcreteStrategy(payoff=1.0, loss=1.0, transaction_cost=0.01)

    assert strategy.get_max_safe_bet(bankroll) == 0.0


@pytest.mark.parametrize(
    ("loss", "transaction_cost", "expected"),
    [
        # loss + transaction_cost above 1.0 - the reciprocal binds.
        (1.0, 0.01, 1.0 / 1.01),
        (2.0, 0.5, 0.4),
        # loss + transaction_cost at or below 1.0 - the 1.0 cap binds.
        (1.0, 0.0, 1.0),
        (0.5, 0.0, 1.0),
        (0.25, 0.25, 1.0),
    ],
)
@pytest.mark.parametrize("bankroll", [0.01, 1.0, 1000.0, 1e9])
def test_get_max_safe_bet_unchanged_for_positive_bankroll(
    loss, transaction_cost, expected, bankroll
):
    """Positive bankrolls keep the pre-guard value: min(1, 1/(loss + cost))."""
    strategy = _ConcreteStrategy(
        payoff=1.0, loss=loss, transaction_cost=transaction_cost
    )

    assert strategy.get_max_safe_bet(bankroll) == pytest.approx(expected)


def test_dynamic_strategy_zero_bankroll_does_not_divide_by_zero_peak():
    """DynamicBankrollManagement's drawdown factor tolerates a zero peak bankroll."""
    strategy = DynamicBankrollManagement(
        base_fraction=0.1, payoff=1.0, loss=1.0, transaction_cost=0.01
    )
    strategy.current_bankroll = 0.0
    strategy.peak_bankroll = 0.0

    assert strategy.get_drawdown_factor() == 1.0
