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


# --- bootstrap_comparison -------------------------------------------------

import doctest  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from keeks import BaseStrategy, BootstrapRow, bootstrap_comparison  # noqa: E402

BOOT = {**RUN, "n_resamples": 30, "seed": 3}
BOOT_PROBS = [0.5 + 0.01 * i for i in range(20)]
BOOT_OUTCOMES = [1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1]


def boot(**over):
    return bootstrap_comparison(
        FACTORIES, BOOT_PROBS, BOOT_OUTCOMES, **{**BOOT, **over}
    )


def test_same_seed_identical_and_different_seed_differs():
    assert boot() == boot()
    a, b = boot(seed=1), boot(seed=2)
    assert [(r.lower, r.upper) for r in a] != [(r.lower, r.upper) for r in b]


def test_point_matches_full_log_and_rows_follow_mapping_order():
    rows = boot()
    full = {
        r.name: r.summary.geometric_growth_per_period
        for r in compare_strategies(FACTORIES, BOOT_PROBS, BOOT_OUTCOMES, **RUN)
    }
    assert [r.name for r in rows] == list(FACTORIES)
    assert all(isinstance(r, BootstrapRow) for r in rows)
    for r in rows:
        assert r.point == pytest.approx(full[r.name])
        assert sum(r.rank_counts) == 30
        assert r.undefined == 0
    for k in range(len(FACTORIES)):
        assert sum(r.rank_counts[k] for r in rows) == 30


def test_dominated_strategy_ranks_last_every_time_and_intervals_ordered():
    odds = {**ODDS}
    rows = bootstrap_comparison(
        {
            "zero": lambda: FixedFractionStrategy(**odds, fraction=0.0),
            "ten": lambda: FixedFractionStrategy(**odds, fraction=0.1),
        },
        [0.7] * 10,
        [1] * 10,
        **{**BOOT, "n_resamples": 25},
    )
    zero, ten = rows
    assert zero.rank_counts == (0, 25)
    assert ten.rank_counts == (25, 0)
    # a zero stake settles nothing, so its growth is undefined everywhere
    assert (zero.lower, zero.point, zero.upper) == (None, None, None)
    assert zero.undefined == 25
    assert ten.undefined == 0
    assert ten.lower <= ten.point <= ten.upper


def test_interval_is_the_percentile_of_resample_values():
    seen = []

    class Spy(BaseStrategy):
        def evaluate(self, probability, current_bankroll):  # noqa: ARG002
            return 0.1

    def factory():
        return Spy(**ODDS)

    rows = bootstrap_comparison({"s": factory}, PROBS, OUTCOMES, **BOOT, confidence=0.5)
    rng = np.random.default_rng(3)
    n = len(PROBS)
    for _ in range(30):
        idx = np.sort(rng.integers(0, n, size=n))
        seen.append(
            manual(
                factory, [PROBS[i] for i in idx], [OUTCOMES[i] for i in idx]
            ).geometric_growth_per_period
        )
    assert rows[0].lower == pytest.approx(np.percentile(seen, 25))
    assert rows[0].upper == pytest.approx(np.percentile(seen, 75))


def test_every_strategy_sees_the_same_resample():
    seen: dict[str, list[tuple[float, ...]]] = {"a": [], "b": []}

    def make(name):
        class Rec(BaseStrategy):
            def __init__(self):
                super().__init__(**ODDS)
                self.log = []
                seen[name].append(self.log)

            def evaluate(self, probability, current_bankroll):  # noqa: ARG002
                self.log.append(probability)
                return 0.1

        return Rec

    bootstrap_comparison(
        {"a": make("a"), "b": make("b")}, BOOT_PROBS, BOOT_OUTCOMES, **BOOT
    )
    # first factory call per name is the full-log point run, then one per resample
    assert len(seen["a"]) == len(seen["b"]) == 31
    assert seen["a"] == seen["b"]
    assert seen["a"][1] != seen["a"][2]
    assert all(log == sorted(log) for log in seen["a"])


def test_block_size_resamples_contiguous_chronological_blocks():
    logs = []

    class Rec(BaseStrategy):
        def __init__(self):
            super().__init__(**ODDS)
            self.log = []
            logs.append(self.log)

        def evaluate(self, probability, current_bankroll):  # noqa: ARG002
            self.log.append(probability)
            return 0.1

    bootstrap_comparison(
        {"r": Rec}, BOOT_PROBS, BOOT_OUTCOMES, **{**BOOT, "block_size": 5}
    )
    assert all(len(log) == 20 for log in logs)
    assert all(log == sorted(log) for log in logs)
    assert any(log != BOOT_PROBS for log in logs[1:])


def test_dominated_small_stake_ranks_second_with_defined_bounds():
    rows = bootstrap_comparison(
        {
            "1%": lambda: FixedFractionStrategy(**ODDS, fraction=0.01),
            "10%": lambda: FixedFractionStrategy(**ODDS, fraction=0.1),
        },
        [0.7] * 10,
        [1] * 10,
        **BOOT,
    )
    assert [r.rank_counts for r in rows] == [(0, 30), (30, 0)]
    assert (rows[0].lower, rows[0].point, rows[0].upper) == pytest.approx(
        (0.01, 0.01, 0.01), abs=1e-4
    )


@pytest.mark.parametrize(
    "bad",
    [
        {"n_resamples": 0},
        {"n_resamples": 1.5},
        {"n_resamples": True},
        {"block_size": 0},
        {"block_size": 21},
        {"confidence": 0.0},
        {"confidence": 1.0},
        {"confidence": float("nan")},
        {"confidence": "x"},
        {"seed": None},
        {"seed": -1},
        {"seed": 1.5},
    ],
)
def test_invalid_controls_raise_before_any_replay(bad):
    def boom():
        raise AssertionError("replay started")

    with pytest.raises(ValueError):
        bootstrap_comparison({"a": boom}, BOOT_PROBS, BOOT_OUTCOMES, **{**BOOT, **bad})


def test_missing_seed_raises_value_error():
    with pytest.raises(ValueError, match="seed"):
        bootstrap_comparison(FACTORIES, PROBS, OUTCOMES, **RUN)


def test_docs_example_runs():
    path = Path(__file__).parent.parent / "docs" / "source" / "bankroll.rst"
    result = doctest.testfile(str(path), module_relative=False, verbose=False)
    assert result.attempted > 0
    assert result.failed == 0


def test_sort_by_ruined_ranks_survivors_first_and_keeps_factory_order_on_ties():
    ruin = lambda: FixedFractionStrategy(**ODDS, fraction=1.0)  # noqa: E731
    probs, outcomes = [0.9, 0.9, 0.9], [0, 0, 1]
    factories = {
        "ruin": ruin,
        "a": FACTORIES["5%"],
        "ruin2": ruin,
        "b": FACTORIES["30%"],
    }
    rows = compare_strategies(factories, probs, outcomes, sort_by="ruined", **RUN)
    assert [r.name for r in rows] == ["a", "b", "ruin", "ruin2"]
    assert [r.summary.ruined for r in rows] == [False, False, True, True]
