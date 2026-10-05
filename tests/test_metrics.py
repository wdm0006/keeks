import dataclasses
import math

import numpy as np
import pytest

import keeks
from keeks import BankRoll, RepeatedBinarySimulator, summarize_history
from keeks.binary_strategies.simple import FixedFractionStrategy
from keeks.metrics import HistorySummary


def test_known_peak_and_trough_uses_running_peak():
    # Peak 100 -> trough 50 (0.5) happens before the later 200 peak; the
    # later 200 -> 150 drop is only 0.25, so a global-max peak would say 0.75.
    summary = summarize_history([100.0, 50.0, 200.0, 150.0])
    assert summary.max_drawdown == 0.5
    assert summary.bets == 3
    assert summary.start == 100.0
    assert summary.end == 150.0
    assert summary.total_return == 0.5
    assert summary.geometric_growth_per_period == pytest.approx(1.5 ** (1 / 3) - 1)
    assert summary.ruined is False


def test_later_deeper_drawdown_wins():
    assert summarize_history([100.0, 90.0, 200.0, 100.0]).max_drawdown == 0.5


def test_ruin_path():
    summary = summarize_history([1000.0, 500.0, 0.0])
    assert summary == HistorySummary(
        bets=2,
        start=1000.0,
        end=0.0,
        total_return=-1.0,
        geometric_growth_per_period=None,
        max_drawdown=1.0,
        ruined=True,
    )


def test_flat_path():
    summary = summarize_history([500.0, 500.0, 500.0])
    assert summary.max_drawdown == 0.0
    assert summary.total_return == 0.0
    assert summary.geometric_growth_per_period == 0.0
    assert summary.ruined is False


def test_single_entry_history():
    summary = summarize_history([1000.0])
    assert summary.bets == 0
    assert summary.total_return == 0.0
    assert summary.geometric_growth_per_period is None
    assert summary.max_drawdown == 0.0
    assert summary.ruined is False


def test_unfunded_history_has_undefined_ratios():
    summary = summarize_history([0.0, 0.0])
    assert summary.total_return is None
    assert summary.geometric_growth_per_period is None
    assert summary.max_drawdown is None
    assert summary.ruined is True


def test_unfunded_start_then_funded_deposit():
    summary = summarize_history([0.0, 100.0, 50.0])
    assert summary.total_return is None
    assert summary.geometric_growth_per_period is None
    assert summary.max_drawdown == 0.5


@pytest.mark.parametrize(
    "history",
    [
        [],
        (),
        [1.0, math.nan],
        [math.inf, 1.0],
        [1.0, -math.inf],
        [1.0, -5.0],
    ],
)
def test_invalid_history_raises(history):
    with pytest.raises(ValueError):
        summarize_history(history)


def test_result_is_frozen():
    summary = summarize_history([1.0, 2.0])
    with pytest.raises(dataclasses.FrozenInstanceError):
        summary.end = 3.0


def test_exported_from_package_root():
    assert keeks.summarize_history is summarize_history
    assert keeks.HistorySummary is HistorySummary
    assert "summarize_history" in keeks.__all__
    assert "HistorySummary" in keeks.__all__


def test_seeded_simulation_end_to_end():
    bankroll = BankRoll(initial_funds=1000.0)
    strategy = FixedFractionStrategy(
        fraction=0.1, payoff=1.0, loss=1.0, transaction_cost_rate=0.0
    )
    RepeatedBinarySimulator(
        payoff=1.0,
        loss=1.0,
        fee_per_bet=0.0,
        probability=0.55,
        trials=40,
        seed=7,
    ).evaluate_strategy(strategy, bankroll)

    # Independent reference computed straight from the recorded path.
    path = np.array(bankroll.history)
    assert len(path) == 41
    drawdowns = [1 - path[i] / path[: i + 1].max() for i in range(len(path))]
    summary = summarize_history(bankroll.history)

    assert summary.bets == 40
    assert summary.start == 1000.0
    assert summary.end == path[-1]
    assert summary.total_return == pytest.approx(path[-1] / 1000.0 - 1)
    assert summary.geometric_growth_per_period == pytest.approx(
        math.exp(np.log(path[1:] / path[:-1]).mean()) - 1
    )
    assert summary.max_drawdown == pytest.approx(max(drawdowns))
    assert summary.max_drawdown > 0
    assert summary.ruined is False
