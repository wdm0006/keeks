"""Seeding, replay, and stream-purity tests for the portfolio simulator.

The portfolio spends no draws from the process-global generator. A seeded
simulator keys each bet's stream by its validated values, so heterogeneous
survivors keep their outcomes across portfolio edits. Identical duplicates
use deterministic occurrence ordinals. Validation failures are side-effect
free: a rejected run consumes no draws, so the next run replays a fresh
simulation exactly.
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
        self.return_pcts = []

    def evaluate(self, _probabilities, _current_bankroll):
        return _validate_stake_fractions(tuple(self._stakes))

    def update_bankroll(self, _current_bankroll):
        pass

    def record_settlement(self, won_bets, return_pcts):
        self.won_flags.append(won_bets)
        self.return_pcts.append(return_pcts)


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


SURVIVOR = (0.47, 2.3, 1.0)
OTHER_A = (0.71, 1.4, 0.8)
OTHER_B = (0.22, 4.1, 1.3)
INSERTED = (0.63, 1.8, 0.6)


def survivor_hook_values(bets, survivor=SURVIVOR):
    index = bets.index(survivor)
    simulator = PortfolioSimulator(bets=bets, trials=80, seed=42)
    strategy = _WinOneBetStrategy(len(bets), index=index)
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
    simulator.evaluate_strategy(strategy, bankroll)
    values = [
        (won[index], returns[index])
        for won, returns in zip(strategy.won_flags, strategy.return_pcts, strict=True)
    ]
    assert {won for won, _return_pct in values} == {False, True}
    return values


def test_surviving_bet_stream_is_stable_when_bet_is_inserted_before_it():
    assert survivor_hook_values([OTHER_A, SURVIVOR, OTHER_B]) == survivor_hook_values(
        [INSERTED, OTHER_A, SURVIVOR, OTHER_B]
    )


def test_surviving_bet_stream_is_stable_when_bet_is_removed_before_it():
    assert survivor_hook_values([INSERTED, OTHER_A, SURVIVOR]) == survivor_hook_values(
        [OTHER_A, SURVIVOR]
    )


def test_surviving_bet_stream_is_stable_when_portfolio_is_reordered():
    assert survivor_hook_values([OTHER_A, SURVIVOR, OTHER_B]) == survivor_hook_values(
        [OTHER_B, OTHER_A, SURVIVOR]
    )


def test_distinct_bets_receive_independent_streams():
    bets = [(0.5, 2.0, 1.0), (0.5, 3.0, 1.0)]
    simulator = PortfolioSimulator(bets=bets, trials=80, seed=42)
    strategy = _WinOneBetStrategy(2, index=0)
    strategy._stakes[1] = 0.25
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
    simulator.evaluate_strategy(strategy, bankroll)
    first = [won[0] for won in strategy.won_flags]
    second = [won[1] for won in strategy.won_flags]
    assert first != second
    assert set(first) == set(second) == {False, True}


def test_stream_identity_distinguishes_exact_validated_float_bits():
    bets = [(0.5, 2.0, 0.0), (0.5, 2.0, -0.0)]
    simulator = PortfolioSimulator(bets=bets, trials=80, seed=42)
    strategy = _WinOneBetStrategy(2, index=0)
    strategy._stakes[1] = 0.25
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
    simulator.evaluate_strategy(strategy, bankroll)
    first = [won[0] for won in strategy.won_flags]
    second = [won[1] for won in strategy.won_flags]
    assert first != second


def test_duplicate_bets_use_replayable_independent_occurrence_streams():
    bets = [(0.5, 2.0, 1.0)] * 2

    def duplicate_sequences():
        simulator = PortfolioSimulator(bets=bets, trials=80, seed=42)
        strategy = _WinOneBetStrategy(2, index=0)
        strategy._stakes[1] = 0.25
        bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
        simulator.evaluate_strategy(strategy, bankroll)
        return tuple(zip(*(won for won in strategy.won_flags), strict=True))

    first_run = duplicate_sequences()
    assert first_run == duplicate_sequences()
    assert first_run[0] != first_run[1]


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
