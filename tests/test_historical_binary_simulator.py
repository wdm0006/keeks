import random

import numpy as np
import pytest

from keeks import (
    BankRoll,
    HistoricalBinarySimulator,
    KellyCriterion,
    RepeatedBinarySimulator,
)

ODDS = {"payoff": 1.0, "loss": 1.0, "fee_per_bet": 0.5}


def _strategy():
    return KellyCriterion(payoff=1.0, loss=1.0, transaction_cost_rate=0.0)


def test_matches_repeated_simulator_on_the_outcomes_it_draws():
    probability, trials, seed = 0.6, 40, 7
    drawn = random.Random(seed)
    outcomes = [drawn.random() < probability for _ in range(trials)]
    assert any(outcomes) and not all(outcomes)

    expected = BankRoll(initial_funds=1000.0)
    RepeatedBinarySimulator(
        **ODDS, probability=probability, trials=trials, seed=seed
    ).evaluate_strategy(_strategy(), expected)

    actual = BankRoll(initial_funds=1000.0)
    HistoricalBinarySimulator(
        **ODDS, probabilities=[probability] * trials, outcomes=outcomes
    ).evaluate_strategy(_strategy(), actual)

    assert actual.history == expected.history
    assert actual.total_funds == expected.total_funds
    assert len(actual.history) > 1


def test_settlement_values_win_then_loss():
    bankroll = BankRoll(initial_funds=1000.0)
    HistoricalBinarySimulator(
        payoff=1.0,
        loss=1.0,
        fee_per_bet=0.0,
        probabilities=[0.6, 0.6],
        outcomes=[1, 0],
    ).evaluate_strategy(_strategy(), bankroll)
    # Kelly stakes 20%: win +200, then lose 20% of 1200.
    assert bankroll.total_funds == pytest.approx(1200.0 - 240.0)


def test_loss_only_and_win_only_move_funds_in_opposite_directions():
    kwargs = {"payoff": 1.0, "loss": 1.0, "fee_per_bet": 0.0}
    won = BankRoll(initial_funds=1000.0)
    HistoricalBinarySimulator(
        **kwargs, probabilities=[0.6], outcomes=[True]
    ).evaluate_strategy(_strategy(), won)
    lost = BankRoll(initial_funds=1000.0)
    HistoricalBinarySimulator(
        **kwargs, probabilities=[0.6], outcomes=[False]
    ).evaluate_strategy(_strategy(), lost)
    assert won.total_funds == pytest.approx(1200.0)
    assert lost.total_funds == pytest.approx(800.0)


def test_trials_is_log_length_and_no_rng_is_consumed():
    sim = HistoricalBinarySimulator(
        **ODDS, probabilities=[0.6, 0.6, 0.6], outcomes=[True, False, True]
    )
    assert sim.trials == 3
    py_state, np_state = random.getstate(), np.random.get_state()[1].copy()
    sim.evaluate_strategy(_strategy(), BankRoll(initial_funds=1000.0))
    assert random.getstate() == py_state
    assert (np.random.get_state()[1] == np_state).all()


def test_empty_log_is_a_no_op():
    bankroll = BankRoll(initial_funds=1000.0)
    HistoricalBinarySimulator(**ODDS, probabilities=[], outcomes=[]).evaluate_strategy(
        _strategy(), bankroll
    )
    assert bankroll.total_funds == 1000.0


def test_accepts_numpy_inputs():
    sim = HistoricalBinarySimulator(
        **ODDS,
        probabilities=np.array([0.6, 0.5]),
        outcomes=np.array([True, False]),
    )
    assert sim.outcomes == (True, False)
    sim = HistoricalBinarySimulator(
        **ODDS, probabilities=[0.6, 0.5], outcomes=np.array([1, 0])
    )
    assert sim.outcomes == (True, False)


@pytest.mark.parametrize(
    ("probabilities", "outcomes"),
    [
        ([0.6, 0.6], [True]),
        ([0.6], [True, False]),
        ([float("nan")], [True]),
        ([float("inf")], [True]),
        ([-0.1], [True]),
        ([1.1], [True]),
        ([0.6], [2]),
        ([0.6], [-1]),
        ([0.6], [0.5]),
        ([0.6], [1.0]),
        ([0.6], ["yes"]),
        ([0.6], [None]),
        ("0.6", [True]),
        ([0.6], True),
    ],
)
def test_invalid_log_raises_value_error(probabilities, outcomes):
    with pytest.raises(ValueError):
        HistoricalBinarySimulator(
            **ODDS, probabilities=probabilities, outcomes=outcomes
        )


def test_invalid_odds_controls_raise():
    with pytest.raises(ValueError):
        HistoricalBinarySimulator(
            payoff=0.0, loss=1.0, fee_per_bet=0.0, probabilities=[], outcomes=[]
        )


def test_odds_mismatched_strategy_raises():
    sim = HistoricalBinarySimulator(**ODDS, probabilities=[0.6], outcomes=[True])
    with pytest.raises(ValueError, match="payoff"):
        sim.evaluate_strategy(
            KellyCriterion(payoff=2.0, loss=1.0, transaction_cost_rate=0.0),
            BankRoll(initial_funds=1000.0),
        )


def test_invalid_stake_fraction_raises():
    class Bad:
        def evaluate(self, _probability, _current_bankroll):
            return 1.5

    sim = HistoricalBinarySimulator(**ODDS, probabilities=[0.6], outcomes=[True])
    with pytest.raises(ValueError, match="stake fraction"):
        sim.evaluate_strategy(Bad(), BankRoll(initial_funds=1000.0))


def test_ruin_error_stops_the_run_with_a_warning():
    class Half:
        def evaluate(self, _probability, _current_bankroll):
            return 0.5

    bankroll = BankRoll(initial_funds=1000.0, max_transaction_loss=0.3)
    sim = HistoricalBinarySimulator(
        payoff=1.0,
        loss=1.0,
        fee_per_bet=0.0,
        probabilities=[0.6] * 5,
        outcomes=[False] * 5,
    )
    with pytest.warns(UserWarning, match="Simulation stopped early"):
        sim.evaluate_strategy(Half(), bankroll)
    assert bankroll.total_funds == 1000.0


def test_no_bet_trials_charge_nothing_and_fire_hooks():
    class Spy:
        def __init__(self):
            self.updates, self.settlements, self.seen = [], [], []

        def evaluate(self, probability, _current_bankroll):
            self.seen.append(probability)
            return 0.0 if probability < 0.5 else 0.1

        def update_bankroll(self, new_bankroll):
            self.updates.append(new_bankroll)

        def record_settlement(self, won, realized_returns):
            self.settlements.append((won, realized_returns))

    spy = Spy()
    bankroll = BankRoll(initial_funds=1000.0)
    HistoricalBinarySimulator(
        payoff=1.0,
        loss=1.0,
        fee_per_bet=0.0,
        probabilities=[0.4, 0.6, 0.7],
        outcomes=[True, True, False],
    ).evaluate_strategy(spy, bankroll)

    assert spy.seen == [0.4, 0.6, 0.7]
    assert spy.updates == [1000.0, 1000.0, pytest.approx(1100.0)]
    assert spy.settlements == [
        ((True,), (pytest.approx(0.1),)),
        ((False,), (pytest.approx(-110.0 / 1100.0),)),
    ]


def test_hooks_stop_firing_after_depletion():
    class Everything:
        def __init__(self):
            self.updates = 0

        def evaluate(self, _probability, _current_bankroll):
            return 1.0

        def update_bankroll(self, _new_bankroll):
            self.updates += 1

    spy = Everything()
    bankroll = BankRoll(initial_funds=100.0)
    HistoricalBinarySimulator(
        payoff=1.0,
        loss=1.0,
        fee_per_bet=0.0,
        probabilities=[0.6] * 4,
        outcomes=[False] * 4,
    ).evaluate_strategy(spy, bankroll)
    assert spy.updates == 1
    assert bankroll.total_funds == 0.0


def test_fee_larger_than_win_withdraws():
    bankroll = BankRoll(initial_funds=1000.0)
    HistoricalBinarySimulator(
        payoff=1.0,
        loss=1.0,
        fee_per_bet=50.0,
        probabilities=[0.51],
        outcomes=[True],
    ).evaluate_strategy(_strategy(), bankroll)
    # Kelly stakes 2% = 20; the win of 20 is smaller than the 50 fee.
    assert bankroll.total_funds == pytest.approx(1000.0 + 20.0 - 50.0)
