"""Seeding, replay, and stream-purity tests for the portfolio simulator.

The portfolio spends no draws from the process-global generator: a seeded
simulator draws exclusively from one child of ``SeedSequence(seed).spawn()``
per bet. Adding a bet appends a new child stream and leaves every earlier
bet's stream untouched, so bet m's outcomes are a function of the seed
alone. Validation failures are side-effect free: a rejected run consumes no
draws, so the next run replays a fresh simulation exactly.
"""

import numpy as np
import pytest

from keeks.bankroll import BankRoll
from keeks.multi_outcome import PortfolioSimulator
from keeks.multi_outcome.base import _validate_stake_fractions


class _FixedFractionMultiOutcome:
    """Duck-typed strategy staking fixed fractions of the bettable funds."""

    def __init__(self, stakes):
        self._stakes = stakes

    def evaluate(self, _probabilities, _current_bankroll):
        return _validate_stake_fractions(self._stakes)


class _WinOneBetStrategy:
    """Stakes only bet ``index`` and records that bet's drawn outcomes."""

    def __init__(self, m, index, fraction=0.25):
        self._stakes = [0.0] * m
        self._stakes[index] = fraction
        self.won_flags = []

    def evaluate(self, _probabilities, _current_bankroll):
        return _validate_stake_fractions(tuple(self._stakes))

    def update_bankroll(self, _current_bankroll):
        pass

    def record_settlement(self, won_bets, _return_pcts):
        self.won_flags.append(won_bets)


class _RejectingStrategy:
    """Returns stakes that break the aggregate-exposure bound."""

    def __init__(self, stakes):
        self._stakes = stakes

    def evaluate(self, _probabilities, _current_bankroll):
        return tuple(self._stakes)


def run_simulation(seed, m=3, trials=50):
    simulator = PortfolioSimulator(
        bets=[(0.55, 2.0, 1.0), (0.45, 3.0, 1.0), (0.30, 2.4, 1.0)][:m],
        transaction_costs=0.01,
        trials=trials,
        seed=seed,
    )
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
    strategy = _FixedFractionMultiOutcome((0.05,) * m)
    simulator.evaluate_strategy(strategy, bankroll)
    return tuple(bankroll.history)


def test_same_seed_replays_identical_history():
    assert run_simulation(seed=42) == run_simulation(seed=42)


def test_different_seed_produces_different_history():
    assert run_simulation(seed=42) != run_simulation(seed=43)


def test_seeded_simulation_does_not_consume_global_generators():
    before = np.random.get_state()
    run_simulation(seed=20260803)
    run_simulation(seed=7)
    after = np.random.get_state()
    assert np.array_equal(before[1], after[1])
    assert before[2] == after[2]


def test_bet_streams_are_independent_of_bet_count():
    # Bet 0's stream must be a function of the seed alone: a two-bet and a
    # four-bet portfolio staking only bet 0 see the same win/loss sequence
    # and settle the same ledger. A shifted stream would surface as
    # diverging histories here.
    histories = []
    won_sequences = []
    for m in (2, 4):
        simulator = PortfolioSimulator(
            bets=[(0.5, 2.0, 1.0)] * m,
            transaction_costs=0.0,
            trials=60,
            seed=42,
        )
        strategy = _WinOneBetStrategy(m, index=0)
        bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
        simulator.evaluate_strategy(strategy, bankroll)
        histories.append(tuple(bankroll.history))
        won_sequences.append([won[0] for won in strategy.won_flags])

    assert won_sequences[0] == won_sequences[1]
    # Guard against a trivially empty comparison: a fair coin must actually
    # come up both ways across 60 staked trials.
    assert any(won_sequences[0]) and not all(won_sequences[0])
    assert histories[0] == histories[1]


def test_validation_failure_consumes_no_draws():
    simulator = PortfolioSimulator(
        bets=[(0.55, 2.0, 1.0), (0.45, 3.0, 1.0)],
        transaction_costs=0.0,
        trials=25,
        seed=99,
    )
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
    with pytest.raises(ValueError, match="sum to no more than one"):
        simulator.evaluate_strategy(_RejectingStrategy((0.6, 0.6)), bankroll)
    # The refused run left nothing behind: the same simulator still replays
    # a fresh same-seed run exactly.
    pristine = PortfolioSimulator(
        bets=[(0.55, 2.0, 1.0), (0.45, 3.0, 1.0)],
        transaction_costs=0.0,
        trials=25,
        seed=99,
    )
    other = BankRoll(initial_funds=1000.0, max_draw_down=None)
    pristine.evaluate_strategy(_FixedFractionMultiOutcome((0.1, 0.1)), other)
    simulator.evaluate_strategy(_FixedFractionMultiOutcome((0.1, 0.1)), bankroll)
    assert bankroll.history == other.history
