"""Value contracts for compare_multi_outcome_strategies on recorded markets."""

import doctest
from pathlib import Path

import pytest

import keeks
from keeks import (
    BankRoll,
    BaseMultiOutcomeStrategy,
    HistoricalMultiOutcomeSimulator,
    MultiOutcomeKellyCriterion,
    StrategyComparison,
    compare_multi_outcome_strategies,
    summarize_history,
)

PAYOFFS = (2.0, 3.0, 4.0)
PROBS = [[0.4, 0.3, 0.3]] * 4
OUTCOMES = [0, 1, None, 2]
RUN = {"payoffs": PAYOFFS, "loss": 1.0, "fee_per_bet": 1.0, "initial_funds": 1000.0}


class Fixed(BaseMultiOutcomeStrategy):
    def __init__(self, stakes, payoffs=PAYOFFS, loss=1.0):
        super().__init__(payoffs, loss)
        self.stakes = tuple(stakes)

    def evaluate(self, probabilities, current_bankroll):  # noqa: ARG002
        return self.stakes


FACTORIES = {
    "one": lambda: Fixed((0.2, 0.0, 0.0)),
    "spread": lambda: Fixed((0.1, 0.1, 0.1)),
}


def manual(factory, probs=PROBS, outcomes=OUTCOMES, **run):
    run = {**RUN, **run}
    sim = HistoricalMultiOutcomeSimulator(
        run["payoffs"], run["loss"], run["fee_per_bet"], probs, outcomes
    )
    bankroll = BankRoll(run["initial_funds"])
    sim.evaluate_strategy(factory(), bankroll)
    return summarize_history(bankroll.history)


def test_known_log_gives_hand_computed_values():
    rows = {
        r.name: r.summary
        for r in compare_multi_outcome_strategies(FACTORIES, PROBS, OUTCOMES, **RUN)
    }
    one, spread = rows["one"], rows["spread"]
    assert one.end == pytest.approx(765.56)
    assert one.bets == 3
    assert one.max_drawdown == pytest.approx(433.44 / 1199.0)
    assert spread.end == pytest.approx(980.4)
    assert spread.bets == 9
    assert spread.max_drawdown == pytest.approx(385.8 / 1099.0)


def test_rows_equal_independent_replay():
    rows = compare_multi_outcome_strategies(FACTORIES, PROBS, OUTCOMES, **RUN)
    assert all(isinstance(r, StrategyComparison) for r in rows)
    for row in rows:
        assert row.summary == manual(FACTORIES[row.name])


def test_void_rows_charge_no_fee():
    rows = compare_multi_outcome_strategies(
        {"one": FACTORIES["one"]}, PROBS[:2], [None, None], **RUN
    )
    summary = rows[0].summary
    assert summary.bets == 0
    assert summary.end == 1000.0


@pytest.mark.parametrize(
    ("sort_by", "expected"),
    [
        ("end", ["spread", "one"]),
        ("total_return", ["spread", "one"]),
        ("geometric_growth_per_period", ["spread", "one"]),
        ("max_drawdown", ["spread", "one"]),
    ],
)
def test_ranking_order(sort_by, expected):
    rows = compare_multi_outcome_strategies(
        FACTORIES, PROBS, OUTCOMES, sort_by=sort_by, **RUN
    )
    assert [r.name for r in rows] == expected


def test_ascending_metric_ranks_smaller_first_and_larger_metrics_larger_first():
    # "one" has the larger drawdown but spread ends higher: directions differ.
    by_dd = compare_multi_outcome_strategies(
        FACTORIES, PROBS, OUTCOMES, sort_by="max_drawdown", **RUN
    )
    dd = [r.summary.max_drawdown for r in by_dd]
    assert dd == sorted(dd)
    by_end = compare_multi_outcome_strategies(
        FACTORIES, PROBS, OUTCOMES, sort_by="end", **RUN
    )
    ends = [r.summary.end for r in by_end]
    assert ends == sorted(ends, reverse=True)


def test_ties_keep_mapping_order_and_undefined_last():
    factories = {
        "idle": lambda: Fixed((0.0, 0.0, 0.0)),
        "b": FACTORIES["one"],
        "a": FACTORIES["one"],
    }
    rows = compare_multi_outcome_strategies(factories, PROBS, OUTCOMES, **RUN)
    assert [r.name for r in rows] == ["b", "a", "idle"]
    assert rows[-1].summary.geometric_growth_per_period is None
    tied = compare_multi_outcome_strategies(
        {"x": factories["idle"], "y": factories["idle"]},
        PROBS,
        OUTCOMES,
        sort_by="end",
        **RUN,
    )
    assert [r.name for r in tied] == ["x", "y"]


def test_factories_called_once_per_policy_without_state_leakage():
    calls = []

    class Counting(Fixed):
        def __init__(self):
            super().__init__((0.1, 0.0, 0.0))
            self.seen = 0

        def evaluate(self, probabilities, current_bankroll):  # noqa: ARG002
            self.seen += 1
            return self.stakes if self.seen <= 2 else (0.0, 0.0, 0.0)

    def factory():
        calls.append(1)
        return Counting()

    first = compare_multi_outcome_strategies({"c": factory}, PROBS, OUTCOMES, **RUN)
    second = compare_multi_outcome_strategies({"c": factory}, PROBS, OUTCOMES, **RUN)
    assert len(calls) == 2
    assert first == second


def boom():
    raise AssertionError("factory ran")


@pytest.mark.parametrize(
    ("strategies", "bad"),
    [
        ({}, {}),
        ([("a", boom)], {}),
        ({"a": boom, 1: boom}, {}),
        ({"a": boom, "b": 3}, {}),
        ({"a": boom}, {"sort_by": "bets"}),
        ({"a": boom}, {"sort_by": "nope"}),
        ({"a": boom}, {"initial_funds": -1.0}),
        ({"a": boom}, {"percent_bettable": 2.0}),
        ({"a": boom}, {"max_transaction_loss": 5.0}),
        ({"a": boom}, {"payoffs": (2.0, -1.0, 4.0)}),
        ({"a": boom}, {"loss": -1.0}),
        ({"a": boom}, {"fee_per_bet": float("nan")}),
        ({"a": boom}, {"probabilities": [[0.5, 0.5]] * 4}),
        ({"a": boom}, {"probabilities": [[0.9, 0.9, 0.9]] * 4}),
        ({"a": boom}, {"probabilities": PROBS[:3]}),
        ({"a": boom}, {"outcomes": [0, 1, 3, 2]}),
        ({"a": boom}, {"outcomes": [0, 1, True, 2]}),
        ({"a": boom}, {"outcomes": [0, 1, 1.0, 2]}),
    ],
)
def test_invalid_arguments_fail_before_any_factory_runs(strategies, bad):
    kwargs = {"probabilities": PROBS, "outcomes": OUTCOMES, **RUN, **bad}
    with pytest.raises(ValueError):
        compare_multi_outcome_strategies(
            strategies,
            kwargs.pop("probabilities"),
            kwargs.pop("outcomes"),
            **kwargs,
        )


def test_odds_mismatch_keeps_simulator_error():
    with pytest.raises(ValueError, match="payoff"):
        compare_multi_outcome_strategies(
            {"bad": lambda: Fixed((0.1, 0.0, 0.0), payoffs=(2.0, 3.0, 5.0))},
            PROBS,
            OUTCOMES,
            **RUN,
        )


def test_kelly_runs_through_public_function():
    rows = compare_multi_outcome_strategies(
        {"kelly": lambda: MultiOutcomeKellyCriterion(PAYOFFS, 1.0)},
        PROBS,
        OUTCOMES,
        **RUN,
    )
    assert rows[0].summary == manual(lambda: MultiOutcomeKellyCriterion(PAYOFFS, 1.0))


def test_public_export_and_docs():
    assert "compare_multi_outcome_strategies" in keeks.__all__
    root = Path(__file__).parent.parent / "docs" / "source"
    text = (root / "bankroll.rst").read_text() + (
        root / "multi_outcome.rst"
    ).read_text()
    assert "keeks.comparison.compare_multi_outcome_strategies" in text


def test_docs_example_runs():
    path = Path(__file__).parent.parent / "docs" / "source" / "multi_outcome.rst"
    result = doctest.testfile(str(path), module_relative=False, verbose=False)
    assert result.attempted > 0
    assert result.failed == 0
