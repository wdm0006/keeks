"""Tests for the portfolio simulator.

The simulator settles M independent binary bets per trial: every staked bet
wins or loses on its own draw, the batch's signed settlements net into one
bankroll transaction, and the bankroll's safeguards evaluate that net total
once - the batch-level drawdown check. The scenarios below pin the
known-ledger accounting, the net-settlement batch shape, the
aggregate-exposure reject semantics, the batch refusal policy on
``RuinError``, the N-ary hook contract, and the constructor's validation
discipline.
"""

import numpy as np
import pytest

from keeks.bankroll import BankRoll
from keeks.multi_outcome import BaseMultiOutcomeStrategy, PortfolioSimulator
from keeks.multi_outcome.base import _validate_stake_fractions


class _FixedStakesStrategy(BaseMultiOutcomeStrategy):
    """Returns a fixed stake vector through the documented validator gate."""

    def __init__(self, payoffs, loss, stakes, transaction_cost=0):
        super().__init__(payoffs, loss, transaction_cost)
        self._stakes = stakes

    def evaluate(self, _probabilities, _current_bankroll):
        return _validate_stake_fractions(self._stakes)


class _RecordingStrategy(BaseMultiOutcomeStrategy):
    """Fixed stakes plus a log of every hook call the simulator makes."""

    def __init__(self, payoffs, loss, stakes, transaction_cost=0):
        super().__init__(payoffs, loss, transaction_cost)
        self._stakes = stakes
        self.events = []

    def update_bankroll(self, current_bankroll):
        self.events.append(("update_bankroll", current_bankroll))

    def evaluate(self, probabilities, current_bankroll):
        self.events.append(("evaluate", tuple(probabilities), current_bankroll))
        return _validate_stake_fractions(self._stakes)

    def record_settlement(self, won_bets, return_pcts):
        self.events.append(("record_settlement", won_bets, return_pcts))


class _DuckTypedStrategy:
    """A strategy outside the ABC hierarchy: no payoffs or loss to check."""

    def __init__(self, stakes):
        self._stakes = stakes

    def evaluate(self, _probabilities, _current_bankroll):
        return _validate_stake_fractions(self._stakes)


def build_simulator(**overrides):
    """A three-bet portfolio with deterministic outcome knobs."""
    common = {
        "bets": ((0.55, 2.0, 1.0), (0.45, 3.0, 1.0), (0.30, 2.4, 1.0)),
        "transaction_costs": 0.0,
        "trials": 3,
        "seed": 42,
    }
    common.update(overrides)
    return PortfolioSimulator(**common)


def _expected_history(initial_funds, fractions, bets, fee, trials):
    """Replicate the documented settlement model for deterministic bets.

    Bets must carry a probability of exactly 1.0 (always wins) or 0.0 (always
    loses). The bankroll's internal accumulator is unrounded and every
    history entry rounds it to cents, exactly as ``BankRoll`` reports it;
    stakes are fractions of the bettable funds as each trial began.
    """
    bank = float(initial_funds)
    history = [bank]
    for _ in range(trials):
        bettable = round(bank, 2)
        net = 0.0
        for fraction, bet in zip(fractions, bets, strict=True):
            if fraction <= 0:
                continue
            probability, payoff, loss = bet
            stake = bettable * fraction
            if probability == 1.0:
                net += (payoff * stake) - fee
            else:
                net -= (loss * stake) + fee
        if net >= 0:
            bank += net
        else:
            bank -= -net
        history.append(round(bank, 2))
    return history


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------
def test_bets_and_probabilities_are_stored():
    simulator = build_simulator()
    assert simulator.bets == ((0.55, 2.0, 1.0), (0.45, 3.0, 1.0), (0.30, 2.4, 1.0))
    assert simulator.probabilities.tolist() == [0.55, 0.45, 0.30]
    assert simulator.transaction_costs == 0.0
    assert simulator.trials == 3
    assert simulator.seed == 42


def test_probabilities_carry_no_sum_constraint():
    # Independent bets: three 50/50 markets sum to 1.5 and stay valid.
    simulator = build_simulator(
        bets=((0.5, 2.0, 1.0), (0.5, 2.0, 1.0), (0.5, 2.0, 1.0))
    )
    assert simulator.probabilities.tolist() == [0.5, 0.5, 0.5]


@pytest.mark.parametrize("bets", [(), 5])
def test_bets_must_be_a_nonempty_sequence_of_triples(bets):
    with pytest.raises(ValueError, match="non-empty sequence"):
        PortfolioSimulator(bets=bets)


@pytest.mark.parametrize("bet", [(0.5, 2.0), (0.5, 2.0, 1.0, 1.0), 3])
def test_each_bet_must_be_a_triple(bet):
    with pytest.raises(ValueError, match="triple"):
        PortfolioSimulator(bets=[bet])


@pytest.mark.parametrize("probability", [-0.1, 1.5, float("inf"), float("nan"), "x"])
def test_bet_probability_must_be_in_unit_interval(probability):
    with pytest.raises(ValueError, match="probability"):
        PortfolioSimulator(bets=[(probability, 2.0, 1.0)])


@pytest.mark.parametrize("payoff", [0.0, -1.0, float("inf"), float("nan"), "x"])
def test_bet_payoff_must_be_positive(payoff):
    with pytest.raises(ValueError, match="payoff"):
        PortfolioSimulator(bets=[(0.5, payoff, 1.0)])


@pytest.mark.parametrize("loss", [-0.1, float("inf"), float("nan"), "x"])
def test_bet_loss_must_be_nonnegative(loss):
    with pytest.raises(ValueError, match="loss"):
        PortfolioSimulator(bets=[(0.5, 2.0, loss)])


def test_transaction_costs_must_be_nonnegative():
    with pytest.raises(ValueError, match="Transaction costs"):
        build_simulator(transaction_costs=-0.01)


def test_transaction_costs_must_be_finite():
    with pytest.raises(ValueError, match="Transaction costs"):
        build_simulator(transaction_costs=float("nan"))


def test_trials_must_be_nonnegative_integer():
    with pytest.raises(ValueError, match="Trials"):
        build_simulator(trials=-1)
    with pytest.raises(ValueError, match="Trials"):
        build_simulator(trials="ten")


def test_seed_must_be_nonnegative_integer_or_none():
    with pytest.raises(ValueError, match="Seed"):
        build_simulator(seed=-1)
    with pytest.raises(ValueError, match="Seed"):
        build_simulator(seed="42")


# ---------------------------------------------------------------------------
# Strategy odds consistency
# ---------------------------------------------------------------------------
def test_strategy_payoffs_must_match_bet_payoffs():
    strategy = _FixedStakesStrategy(
        payoffs=(2.0, 3.0, 2.5), loss=1.0, stakes=(0.1, 0.1, 0.1)
    )
    simulator = build_simulator()
    with pytest.raises(ValueError, match="does not match portfolio bet payoffs"):
        simulator.evaluate_strategy(strategy, BankRoll(initial_funds=1000.0))


def test_strategy_loss_must_match_every_bets_loss():
    strategy = _FixedStakesStrategy(
        payoffs=(2.0, 3.0, 2.4), loss=1.0, stakes=(0.1, 0.1, 0.1)
    )
    simulator = build_simulator(
        bets=((0.55, 2.0, 1.0), (0.45, 3.0, 0.5), (0.30, 2.4, 1.0))
    )
    with pytest.raises(ValueError, match="does not match bet 1's loss"):
        simulator.evaluate_strategy(strategy, BankRoll(initial_funds=1000.0))


def test_duck_typed_strategy_skips_the_odds_check():
    strategy = _DuckTypedStrategy(stakes=(0.1, 0.1, 0.1))
    simulator = build_simulator()
    simulator.evaluate_strategy(strategy, BankRoll(initial_funds=1000.0))


def test_stake_vector_length_must_match_bet_count():
    # Regression: a strategy returning more fractions than the portfolio has
    # bets used to crash with IndexError deep in the settlement loop, and a
    # shorter one silently left bets unstaked.
    simulator = build_simulator(bets=((0.55, 2.0, 1.0),))
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
    with pytest.raises(ValueError, match="exactly 1 stake fractions, got 3"):
        simulator.evaluate_strategy(_DuckTypedStrategy((0.1, 0.1, 0.1)), bankroll)
    with pytest.raises(ValueError, match="exactly 1 stake fractions, got 2"):
        simulator.evaluate_strategy(_DuckTypedStrategy((0.1, 0.1)), bankroll)
    assert bankroll.history == [1000.0]


# ---------------------------------------------------------------------------
# Settlement accounting
# ---------------------------------------------------------------------------
def test_always_win_bets_settle_through_the_payoff_leg():
    bets = ((1.0, 2.0, 1.0),)
    fractions = (0.5,)
    simulator = build_simulator(bets=bets, transaction_costs=10.0, trials=2, seed=7)
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
    simulator.evaluate_strategy(_DuckTypedStrategy(fractions), bankroll)
    assert bankroll.history == _expected_history(
        1000.0, fractions, bets, fee=10.0, trials=2
    )


def test_always_lose_bets_settle_through_the_loss_leg():
    bets = ((0.0, 2.0, 1.0),)
    fractions = (0.5,)
    simulator = build_simulator(bets=bets, transaction_costs=10.0, trials=2, seed=7)
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
    simulator.evaluate_strategy(_DuckTypedStrategy(fractions), bankroll)
    assert bankroll.history == _expected_history(
        1000.0, fractions, bets, fee=10.0, trials=2
    )


def test_fee_dominated_win_settles_as_a_net_withdrawal():
    # payoff * stake = 50 < fee = 60: the winning bet still costs the
    # portfolio 10, and the net batch settles through one withdrawal.
    bets = ((1.0, 0.5, 1.0),)
    simulator = build_simulator(bets=bets, transaction_costs=60.0, trials=1, seed=7)
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
    simulator.evaluate_strategy(_DuckTypedStrategy((0.1,)), bankroll)
    assert bankroll.history == [1000.0, 990.0]


def test_batch_nets_into_exactly_one_transaction():
    # One winning and one losing bet: the batch's signed amounts (+595 and
    # -205) net to +390 and cross the bankroll as a single deposit - one
    # history entry for the whole portfolio, not one per bet.
    bets = ((1.0, 2.0, 1.0), (0.0, 3.0, 1.0))
    simulator = build_simulator(bets=bets, transaction_costs=5.0, trials=1, seed=7)
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
    simulator.evaluate_strategy(_DuckTypedStrategy((0.3, 0.2)), bankroll)
    assert bankroll.history == [1000.0, 1390.0]
    assert len(bankroll.history) == 2


def test_stakes_come_from_the_trial_start_bankroll():
    # Two back-to-back batches: the second batch's stakes are fractions of
    # the bankroll as it stood when that trial began.
    bets = ((1.0, 2.0, 1.0), (0.0, 3.0, 1.0))
    fractions = (0.3, 0.2)
    simulator = build_simulator(bets=bets, trials=2, seed=7)
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
    simulator.evaluate_strategy(_DuckTypedStrategy(fractions), bankroll)
    assert bankroll.history == _expected_history(
        1000.0, fractions, bets, fee=0.0, trials=2
    )


def test_declined_bets_pay_no_fee_and_draw_nothing():
    bets = ((1.0, 2.0, 1.0), (1.0, 2.0, 1.0))
    simulator = build_simulator(bets=bets, transaction_costs=25.0, trials=1, seed=7)
    strategy = _RecordingStrategy(payoffs=(2.0, 2.0), loss=1.0, stakes=(0.0, 0.25))
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
    simulator.evaluate_strategy(strategy, bankroll)
    # Bet 0 declined: no fee, no draw, won flag None, return 0.0. Bet 1 wins:
    # 2 * 250 - 25 = 475.
    assert bankroll.history == [1000.0, 1475.0]
    assert strategy.events[-1] == (
        "record_settlement",
        (None, True),
        (0.0, 475 / 1000),
    )


def test_fully_declined_trial_is_skipped_entirely():
    bets = ((0.5, 2.0, 1.0), (0.5, 3.0, 1.0))
    simulator = build_simulator(bets=bets, trials=3, seed=7)
    strategy = _RecordingStrategy(payoffs=(2.0, 3.0), loss=1.0, stakes=(0.0, 0.0))
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
    simulator.evaluate_strategy(strategy, bankroll)
    # No settlement, no fee, no history growth - but update_bankroll still
    # fires before evaluate on every non-bankrupt trial.
    assert bankroll.history == [1000.0]
    assert bankroll.total_funds == 1000.0
    assert [event[0] for event in strategy.events].count("record_settlement") == 0
    assert [event[0] for event in strategy.events].count("update_bankroll") == 3


def test_zero_trials_is_a_no_op():
    simulator = build_simulator(trials=0)
    strategy = _RecordingStrategy(
        payoffs=(2.0, 3.0, 2.4), loss=1.0, stakes=(0.1, 0.0, 0.0)
    )
    bankroll = BankRoll(initial_funds=1000.0)
    simulator.evaluate_strategy(strategy, bankroll)
    assert bankroll.history == [1000.0]
    assert strategy.events == []


def test_depleted_bankroll_stops_before_any_hook():
    simulator = build_simulator()
    strategy = _RecordingStrategy(
        payoffs=(2.0, 3.0, 2.4), loss=1.0, stakes=(0.1, 0.1, 0.1)
    )
    bankroll = BankRoll(initial_funds=0.0, max_draw_down=None)
    simulator.evaluate_strategy(strategy, bankroll)
    assert bankroll.history == [0.0]
    assert strategy.events == []


def test_one_bet_portfolio_shares_the_one_leg_market_stream():
    # A one-bet portfolio is the market simulator's M=1 seam: the same seed
    # hands both the first spawned child, and with a probability-1 bet both
    # settle every staked trial identically.
    from keeks.multi_outcome import RepeatedMultiOutcomeSimulator

    strategy = _FixedStakesStrategy(payoffs=(2.0,), loss=1.0, stakes=(0.25,))
    portfolio_bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
    portfolio = PortfolioSimulator(
        bets=[(1.0, 2.0, 1.0)], transaction_costs=0.05, trials=30, seed=42
    )
    portfolio.evaluate_strategy(strategy, portfolio_bankroll)

    market_bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
    market = RepeatedMultiOutcomeSimulator(
        payoffs=(2.0,),
        loss=1.0,
        transaction_costs=0.05,
        probabilities=(1.0,),
        trials=30,
        seed=42,
    )
    market.evaluate_strategy(strategy, market_bankroll)
    assert portfolio_bankroll.history == market_bankroll.history


# ---------------------------------------------------------------------------
# Aggregate exposure
# ---------------------------------------------------------------------------
def test_exposure_over_bound_is_rejected_not_reduced():
    simulator = build_simulator()
    strategy = _DuckTypedStrategy(stakes=(0.5, 0.4, 0.2))
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
    with pytest.raises(ValueError, match="sum to no more than one"):
        simulator.evaluate_strategy(strategy, bankroll)
    # Reject semantics: the run aborts before any settlement, the stakes are
    # never silently reduced to fit the bettable funds.
    assert bankroll.history == [1000.0]


def test_exposure_bound_holds_in_the_placement_arithmetic():
    # With no fee the settled return determines each stake exactly: a won
    # bet returned payoff * stake / before, a lost bet -loss * stake /
    # before. The bound is asserted against that recovered placement
    # arithmetic, not the strategy's intentions.
    bets = ((1.0, 2.0, 1.0), (0.0, 3.0, 1.0), (1.0, 2.4, 1.0))
    fractions = (0.2, 0.3, 0.1)
    simulator = build_simulator(bets=bets, trials=2, seed=7)
    strategy = _RecordingStrategy(payoffs=(2.0, 3.0, 2.4), loss=1.0, stakes=fractions)
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
    simulator.evaluate_strategy(strategy, bankroll)

    settlements = [
        event for event in strategy.events if event[0] == "record_settlement"
    ]
    befores = [event[2] for event in strategy.events if event[0] == "evaluate"]
    assert len(settlements) == len(befores) == 2
    for event, before in zip(settlements, befores, strict=True):
        won_bets, returns = event[1:]
        stakes = [
            abs(ret) * before / (bets[index][1] if won else bets[index][2])
            for index, (won, ret) in enumerate(zip(won_bets, returns, strict=True))
            if won is not None
        ]
        assert all(stake >= 0 for stake in stakes)
        assert sum(stakes) <= before * (1 + 1e-9)


# ---------------------------------------------------------------------------
# Batch-level drawdown
# ---------------------------------------------------------------------------
def test_batch_drawdown_evaluates_the_net_total_once():
    # max_draw_down = 0.07 caps one withdrawal at 70. Each leg alone (a 500
    # loss) would breach it, but the batch nets to -40, so the single
    # batch-level check lets the portfolio settle.
    bets = ((0.0, 1.1, 1.0), (1.0, 1.15, 1.0))
    simulator = build_simulator(bets=bets, trials=1, seed=7)
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=0.07)
    strategy = _RecordingStrategy(payoffs=(1.1, 1.15), loss=1.0, stakes=(0.5, 0.4))
    simulator.evaluate_strategy(strategy, bankroll)
    assert bankroll.history == [1000.0, 960.0]
    won_bets, returns = strategy.events[-1][1:]
    assert won_bets == (False, True)
    assert returns[0] == -0.5
    assert returns[1] == pytest.approx(0.46)


def test_batch_over_drawdown_is_refused_and_stops_the_run():
    # Both bets lose: the batch's net withdrawal (1000) breaches the
    # drawdown cap (50). The whole batch is refused, the bankroll is
    # untouched, and the simulation stops after the batch - never
    # mid-portfolio.
    bets = ((0.0, 1.1, 1.0), (0.0, 1.1, 1.0))
    simulator = build_simulator(bets=bets, trials=5, seed=7)
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=0.05)
    strategy = _RecordingStrategy(payoffs=(1.1, 1.1), loss=1.0, stakes=(0.5, 0.5))
    simulator.evaluate_strategy(strategy, bankroll)
    assert bankroll.history == [1000.0]
    assert bankroll.total_funds == 1000.0
    settlements = [
        event for event in strategy.events if event[0] == "record_settlement"
    ]
    assert len(settlements) == 1
    _, won_bets, returns = settlements[0]
    assert won_bets == (False, False)
    assert returns == (0.0, 0.0)


def test_refused_batch_reports_drawn_outcomes_with_zero_returns():
    # The draws happen before the settlement, so the hook still reports the
    # won/lost flags - only the returns are zeroed by the refusal.
    bets = ((0.0, 1.1, 1.0), (1.0, 1.1, 1.0))
    simulator = build_simulator(bets=bets, trials=1, seed=7)
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=0.05)
    strategy = _RecordingStrategy(payoffs=(1.1, 1.1), loss=1.0, stakes=(0.5, 0.1))
    simulator.evaluate_strategy(strategy, bankroll)
    # Net: -500 + 110 - 0 = -390, withdrawn past the 50 cap -> refused.
    _, won_bets, returns = strategy.events[-1]
    assert won_bets == (False, True)
    assert returns == (0.0, 0.0)
    assert bankroll.total_funds == 1000.0


# ---------------------------------------------------------------------------
# Hook contract
# ---------------------------------------------------------------------------
def test_hooks_fire_in_documented_order_per_trial():
    bets = ((1.0, 2.0, 1.0), (0.0, 3.0, 1.0))
    simulator = build_simulator(bets=bets, trials=2, seed=7)
    strategy = _RecordingStrategy(payoffs=(2.0, 3.0), loss=1.0, stakes=(0.1, 0.1))
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
    simulator.evaluate_strategy(strategy, bankroll)

    kinds = [event[0] for event in strategy.events]
    assert kinds == [
        "update_bankroll",
        "evaluate",
        "record_settlement",
        "update_bankroll",
        "evaluate",
        "record_settlement",
    ]
    first_update, first_eval, first_settlement = strategy.events[:3]
    assert first_update[1] == 1000.0
    assert first_eval[1] == (1.0, 0.0)
    assert first_eval[2] == 1000.0
    won_bets, returns = first_settlement[1:]
    assert won_bets == (True, False)
    assert returns == (0.2, -0.1)


def test_strategy_without_hooks_runs_quietly():
    strategy = _FixedStakesStrategy(
        payoffs=(2.0, 3.0, 2.4), loss=1.0, stakes=(0.1, 0.1, 0.1)
    )
    simulator = build_simulator()
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
    simulator.evaluate_strategy(strategy, bankroll)
    assert len(bankroll.history) == 4


# ---------------------------------------------------------------------------
# Unseeded path
# ---------------------------------------------------------------------------
def test_unseeded_path_draws_from_numpy_global_generator(monkeypatch):
    monkeypatch.setattr(np.random, "random", lambda: 0.99)
    simulator = build_simulator(bets=((0.5, 2.0, 1.0),), trials=2, seed=None)
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
    simulator.evaluate_strategy(_DuckTypedStrategy((0.1,)), bankroll)
    # Every draw is 0.99 >= 0.5, so every staked bet loses; each trial risks
    # a tenth of the bankroll as it stood when that trial began.
    assert bankroll.history == [1000.0, 900.0, 810.0]
