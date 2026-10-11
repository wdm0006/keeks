"""Value contracts for compare_strategies against manual replay runs."""

import math

import pytest

from keeks import (
    BankRoll,
    FixedFractionStrategy,
    HistoricalBinarySimulator,
    KellyCriterion,
    StrategyComparison,
    compare_strategies,
    summarize_history,
)
from keeks.metrics import HistorySummary

ODDS = {"payoff": 1.0, "loss": 1.0, "transaction_cost_rate": 0.0}
PROBS = [0.6, 0.55, 0.7, 0.6, 0.65, 0.6]
OUTCOMES = [1, 0, 1, 1, 0, 1]
RUN = {"payoff": 1.0, "loss": 1.0, "fee_per_bet": 0.0, "initial_funds": 1000.0}
FACTORIES = {
    "Kelly": lambda: KellyCriterion(**ODDS),
    "5%": lambda: FixedFractionStrategy(**ODDS, fraction=0.05),
    "30%": lambda: FixedFractionStrategy(**ODDS, fraction=0.3),
}


def manual(factory, probs=PROBS, outcomes=OUTCOMES, **run):
    run = {**RUN, **run}
    sim = HistoricalBinarySimulator(
        run["payoff"], run["loss"], run["fee_per_bet"], probs, outcomes
    )
    bankroll = BankRoll(run["initial_funds"])
    sim.evaluate_strategy(factory(), bankroll)
    return summarize_history(bankroll.history)


def test_rows_equal_manual_runs_and_are_sorted_by_growth():
    rows = compare_strategies(FACTORIES, PROBS, OUTCOMES, **RUN)
    assert all(isinstance(row, StrategyComparison) for row in rows)
    for row in rows:
        assert row.summary == manual(FACTORIES[row.name])
    growth = [row.summary.geometric_growth_per_period for row in rows]
    assert growth == sorted(growth, reverse=True)
    assert len({row.name for row in rows}) == 3
    assert len(set(growth)) == 3


def test_sort_by_drawdown_is_ascending():
    rows = compare_strategies(FACTORIES, PROBS, OUTCOMES, sort_by="max_drawdown", **RUN)
    drawdowns = [row.summary.max_drawdown for row in rows]
    assert drawdowns == sorted(drawdowns)
    assert drawdowns[0] != drawdowns[-1]


def test_ruined_strategy_does_not_affect_next_run():
    ruin = lambda: FixedFractionStrategy(**ODDS, fraction=1.0)  # noqa: E731
    probs, outcomes = [0.9, 0.9, 0.9], [0, 0, 1]
    rows = compare_strategies(
        {"ruin": ruin, "after": FACTORIES["5%"]}, probs, outcomes, **RUN
    )
    by_name = {row.name: row.summary for row in rows}
    assert by_name["ruin"].ruined
    assert by_name["ruin"].geometric_growth_per_period is None
    assert by_name["after"] == manual(FACTORIES["5%"], probs, outcomes)
    assert not by_name["after"].ruined
    assert by_name["after"].start == 1000.0


def test_none_values_sort_last_for_both_directions():
    ruin = lambda: FixedFractionStrategy(**ODDS, fraction=1.0)  # noqa: E731
    probs, outcomes = [0.9, 0.9, 0.9], [0, 0, 1]
    for key in ("geometric_growth_per_period", "max_drawdown"):
        rows = compare_strategies(
            {"ruin": ruin, "ok": FACTORIES["5%"]}, probs, outcomes, sort_by=key, **RUN
        )
        none_names = [r.name for r in rows if getattr(r.summary, key) is None]
        defined_names = [r.name for r in rows if getattr(r.summary, key) is not None]
        assert [r.name for r in rows] == defined_names + none_names
    rows = compare_strategies(
        {"ruin": ruin, "ok": FACTORIES["5%"]}, probs, outcomes, **RUN
    )
    assert [r.name for r in rows] == ["ok", "ruin"]


def test_factory_called_once_per_run_and_fresh():
    built = []

    def factory():
        built.append(1)
        return FixedFractionStrategy(**ODDS, fraction=0.1)

    compare_strategies({"a": factory, "b": factory}, PROBS, OUTCOMES, **RUN)
    assert len(built) == 2


@pytest.mark.parametrize(
    "strategies",
    [{}, [], {"a": 3}, {1: FACTORIES["5%"]}, None],
)
def test_invalid_strategies_raise(strategies):
    with pytest.raises(ValueError):
        compare_strategies(strategies, PROBS, OUTCOMES, **RUN)


def test_invalid_arguments_raise_before_any_factory_runs():
    def boom():
        raise AssertionError("factory ran")

    for bad in (
        {"initial_funds": math.nan},
        {"initial_funds": -1.0},
        {"percent_bettable": 2.0},
        {"max_transaction_loss": 1.5},
        {"fee_per_bet": math.inf},
        {"sort_by": "bets"},
        {"sort_by": "nope"},
    ):
        with pytest.raises(ValueError):
            compare_strategies({"a": boom}, PROBS, OUTCOMES, **{**RUN, **bad})
    with pytest.raises(ValueError):
        compare_strategies({"a": boom}, PROBS, OUTCOMES[:-1], **RUN)
    with pytest.raises(ValueError):
        compare_strategies({"a": boom}, [1.5], [1], **RUN)


def test_odds_mismatch_raises_simulator_error():
    wrong = lambda: KellyCriterion(payoff=2.0, loss=1.0, transaction_cost_rate=0.0)  # noqa: E731
    with pytest.raises(ValueError, match="payoff"):
        compare_strategies({"w": wrong}, PROBS, OUTCOMES, **RUN)


def test_bankroll_options_are_passed_through():
    rows = compare_strategies(
        {"30%": FACTORIES["30%"]}, PROBS, OUTCOMES, percent_bettable=0.5, **RUN
    )
    expected = manual(FACTORIES["30%"])
    assert rows[0].summary != expected
    assert isinstance(rows[0].summary, HistorySummary)
