from unittest.mock import Mock

import numpy as np
import pytest

from keeks import BankRoll, HistoricalMultiOutcomeSimulator, MultiOutcomeKellyCriterion
from keeks.multi_outcome import HistoricalMultiOutcomeSimulator as PublicSimulator
from keeks.multi_outcome.simulators import RepeatedMultiOutcomeSimulator

PAYOFFS = (3.0, 4.0, 5.0)
ROWS = ((0.5, 0.25, 0.125), (0.25, 0.5, 0.125), (0.125, 0.25, 0.5), (0.25, 0.25, 0.5))
OUTCOMES = (0, None, 1, 2)


def replay(probabilities=ROWS, outcomes=OUTCOMES, **kwargs):
    return HistoricalMultiOutcomeSimulator(
        PAYOFFS, 0.5, 1.0, probabilities, outcomes, **kwargs
    )


def fixed_stakes():
    strategy = Mock()
    strategy.evaluate.side_effect = lambda _p, funds: np.array((8, 16, 8)) / funds
    return strategy


def test_exact_history_and_hooks():
    assert PublicSimulator is HistoricalMultiOutcomeSimulator
    strategy = fixed_stakes()
    bank = BankRoll(128)
    simulator = replay()
    assert simulator.trials == 4
    simulator.evaluate_strategy(strategy, bank)
    assert bank.history == [128, 143, 134, 129, 124, 171, 166, 161, 152, 183]
    assert [call.args[0] for call in strategy.update_bankroll.call_args_list] == [
        128,
        129,
        129,
        166,
    ]
    for call, row in zip(strategy.evaluate.call_args_list, ROWS, strict=True):
        np.testing.assert_array_equal(call.args[0], row)
    assert strategy.record_settlement.call_args_list[1].args == (
        (None, None, None),
        (0.0, 0.0, 0.0),
    )
    assert strategy.record_settlement.call_args_list[0].args == (
        (True, False, False),
        (15 / 128, -9 / 128, -5 / 128),
    )


@pytest.mark.parametrize(
    "outcomes",
    [
        (0,),
        (0, None, 1, -1),
        (0, None, 1, 3),
        (0, None, 1, 1.0),
        (0, None, 1, True),
        (0, None, 1, np.bool_(False)),
        (0, None, 1, "1"),
    ],
)
def test_invalid_outcomes(outcomes):
    with pytest.raises(ValueError):
        replay(outcomes=outcomes)


@pytest.mark.parametrize(
    "row",
    [
        (0.5, 0.5),
        (0.5, 0.25, 0.25, 0),
        (-0.1, 0.5, 0.5),
        (float("nan"), 0, 0),
        (0.5, 0.5, 0.5),
        [],
        [[0.5, 0.25, 0.25]],
    ],
)
def test_invalid_rows_even_after_valid_rounds(row):
    with pytest.raises(ValueError):
        replay(probabilities=[ROWS[0], row], outcomes=[0, None])


@pytest.mark.parametrize("name", ["probabilities", "outcomes"])
@pytest.mark.parametrize("value", [None, "abc", b"abc", iter([])])
def test_requires_sequences(name, value):
    with pytest.raises(ValueError, match="sequence"):
        replay(**{name: value})


def test_empty_and_numpy_indices():
    simulator = replay([], [])
    strategy = fixed_stakes()
    bank = BankRoll(128)
    simulator.evaluate_strategy(strategy, bank)
    strategy.evaluate.assert_not_called()
    assert bank.history == [128]
    assert replay([ROWS[0]], [np.int64(2)]).outcomes == (2,)


def test_deterministic_kelly_and_no_rng(monkeypatch):
    monkeypatch.setattr(
        np.random, "random", Mock(side_effect=AssertionError("RNG used"))
    )
    monkeypatch.setattr(
        np.random, "default_rng", Mock(side_effect=AssertionError("RNG created"))
    )
    simulator = replay()
    histories = []
    for _ in range(2):
        bank = BankRoll(128)
        simulator.evaluate_strategy(MultiOutcomeKellyCriterion(PAYOFFS, 0.5, 0.0), bank)
        histories.append(bank.history)
    assert histories[0] == histories[1]
    assert histories[0][-1] != 128


def test_repeated_settlement_matches_recorded_legs(monkeypatch):
    monkeypatch.setattr(np.random, "random", Mock(side_effect=[0.1, 0.95, 0.6, 0.8]))
    repeated = RepeatedMultiOutcomeSimulator(PAYOFFS, 0.5, 1, ROWS[0], trials=4)
    historical = replay([ROWS[0]] * 4)
    banks = [BankRoll(128), BankRoll(128)]
    strategies = [fixed_stakes(), fixed_stakes()]
    repeated.evaluate_strategy(strategies[0], banks[0])
    historical.evaluate_strategy(strategies[1], banks[1])
    assert (
        banks[0].history
        == banks[1].history
        == [128, 143, 134, 129, 124, 171, 166, 161, 152, 183]
    )
    assert (
        strategies[0].record_settlement.call_args_list
        == strategies[1].record_settlement.call_args_list
    )


@pytest.mark.parametrize("payoffs,loss", [((2, 4, 5), 0.5), (PAYOFFS, 1)])
def test_strategy_odds_mismatch(payoffs, loss):
    bank = BankRoll(128)
    with pytest.raises(ValueError, match="does not match"):
        replay().evaluate_strategy(MultiOutcomeKellyCriterion(payoffs, loss, 0), bank)
    assert bank.history == [128]


@pytest.mark.parametrize(
    "fractions", [(0.1, 0.2), (-0.1, 0.2, 0.1), (0.5, 0.5, 0.5), (float("nan"), 0, 0)]
)
def test_invalid_strategy_stakes(fractions):
    strategy = Mock()
    strategy.evaluate.return_value = fractions
    bank = BankRoll(128)
    with pytest.raises(ValueError):
        replay().evaluate_strategy(strategy, bank)
    assert bank.history == [128]
    strategy.record_settlement.assert_not_called()


def test_skipped_round_advances_log_and_noncallable_hooks():
    strategy = Mock(update_bankroll=None, record_settlement=False)
    strategy.evaluate.side_effect = [(0, 0, 0), (0.25, 0, 0)]
    bank = BankRoll(128)
    replay(ROWS[:2], (0, 1)).evaluate_strategy(strategy, bank)
    assert bank.history == [128, 111]


def test_depleted_bankroll_and_ruin_finishes_batch():
    strategy = fixed_stakes()
    replay().evaluate_strategy(strategy, BankRoll(0))
    strategy.evaluate.assert_not_called()
    bank = BankRoll(128, max_transaction_loss=0.01)
    with pytest.warns(UserWarning, match="Settlement refused"):
        replay().evaluate_strategy(strategy, bank)
    assert bank.history == [128, 143]
    assert strategy.evaluate.call_count == 1
    assert strategy.record_settlement.call_args.args == (
        (True, False, False),
        (15 / 128, 0, 0),
    )


def test_fee_dominates_win_and_declined_leg():
    strategy = Mock()
    strategy.evaluate.return_value = (0.001, 0, 0)
    bank = BankRoll(128)
    replay(ROWS[:1], (0,)).evaluate_strategy(strategy, bank)
    assert bank.history == [128, 127.26]


def test_recorded_inputs_are_copied():
    rows = np.array([ROWS[0]])
    outcomes = [0]
    simulator = replay(rows, outcomes)
    rows[:] = 0
    outcomes[0] = None
    bank = BankRoll(128)
    simulator.evaluate_strategy(fixed_stakes(), bank)
    assert bank.history == [128, 143, 134, 129]


@pytest.mark.parametrize(
    "payoffs,loss,fee",
    [
        ((), 1, 0),
        ((float("nan"),), 1, 0),
        (PAYOFFS, -1, 0),
        (PAYOFFS, 1, -1),
        (PAYOFFS, 1, float("inf")),
    ],
)
def test_invalid_controls_on_empty_log(payoffs, loss, fee):
    with pytest.raises(ValueError):
        HistoricalMultiOutcomeSimulator(payoffs, loss, fee, [], [])
