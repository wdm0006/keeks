"""Property-based invariants for the portfolio simulator (Hypothesis).

The invariants follow the multi-outcome research (art_bNZenx37 §3.4/§5.5)
and the Phase 3 spec's P5 row: no negative stakes, exactly-once accounting
(each settled batch updates the bankroll exactly once), and the
aggregate-exposure bound under random portfolios. Everything here is
hermetic: simulators are always seeded so the process-global RNG is never
touched.

The reject semantics for a broken exposure bound are pinned separately in
``tests/test_portfolio_simulator.py`` (ValueError before any settlement) —
these properties exercise portfolios that pass validation.
"""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from keeks.bankroll import BankRoll
from keeks.multi_outcome import PortfolioSimulator
from keeks.multi_outcome.base import _validate_stake_fractions


@st.composite
def portfolios(draw):
    """Random portfolios: a bet table and a matched-length stake vector.

    Stakes are one fraction per bet; uniform draws can sum past one, so
    rescaling keeps them valid (the bound is inclusive) without changing
    their shape. Exact zeros stay zero so the declined-bet path stays
    exercised; positive stakes are bounded away from zero so settlement
    returns cannot underflow to 0.0.
    """
    m = draw(st.integers(min_value=1, max_value=5))
    bets = []
    for _ in range(m):
        bets.append(
            (
                draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
                draw(st.floats(min_value=0.5, max_value=8.0, allow_nan=False)),
                draw(st.floats(min_value=0.1, max_value=2.0, allow_nan=False)),
            )
        )
    raw = draw(
        st.lists(
            st.one_of(
                st.just(0.0),
                st.floats(min_value=0.001, max_value=1.0, allow_nan=False),
            ),
            min_size=m,
            max_size=m,
        )
    )
    total = sum(raw)
    if total > 1.0:
        raw = [value / total for value in raw]
    return tuple(bets), tuple(raw)


class _RecordingStrategy:
    """Duck-typed strategy that logs what the simulator asked and told it."""

    def __init__(self, stakes):
        self._stakes = stakes
        self.evaluations = []
        self.settlements = []

    def evaluate(self, probabilities, current_bankroll):
        self.evaluations.append((tuple(probabilities), current_bankroll))
        return _validate_stake_fractions(self._stakes)

    def update_bankroll(self, _current_bankroll):
        pass

    def record_settlement(self, won_bets, return_pcts):
        self.settlements.append((won_bets, return_pcts))


def run_portfolio(bets, stakes, seed, trials=25):
    simulator = PortfolioSimulator(
        bets=bets, transaction_costs=0.0, trials=trials, seed=seed
    )
    strategy = _RecordingStrategy(stakes)
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
    simulator.evaluate_strategy(strategy, bankroll)
    return strategy, bankroll


def _staked_evaluations(strategy, stakes):
    """Evaluations from staked trials, paired in order with settlements.

    Declined trials (all-zero stakes) produce no settlement, so the i-th
    settlement belongs to the i-th staked evaluation.
    """
    staked = [evaluation for evaluation in strategy.evaluations if any(stakes)]
    assert len(staked) == len(strategy.settlements)
    return staked


@given(portfolio=portfolios(), seed=st.integers(0, 2**32))
def test_no_negative_stakes_and_exposure_bound_hold(portfolio, seed):
    """Recovered stakes are nonnegative and fit inside the bettable funds.

    With no fee, each settled return determines its stake exactly: a won bet
    returned ``payoff * stake / before``, a lost bet ``-loss * stake /
    before``. Both directions recover a nonnegative stake, and the recovered
    vector can never exceed the funds the trial had bettable.
    """
    bets, stakes = portfolio
    strategy, _bankroll = run_portfolio(bets, stakes, seed)
    staked = _staked_evaluations(strategy, stakes)

    for (_probabilities, before), (won_bets, returns) in zip(
        staked, strategy.settlements, strict=True
    ):
        recovered = [
            abs(ret) * before / (bets[index][1] if won else bets[index][2])
            for index, (won, ret) in enumerate(zip(won_bets, returns, strict=True))
            if won is not None
        ]
        assert all(stake >= 0 for stake in recovered)
        assert sum(recovered) <= before * (1 + 1e-9)


@given(portfolio=portfolios(), seed=st.integers(0, 2**32))
def test_each_batch_updates_the_bankroll_exactly_once(portfolio, seed):
    """Every settled batch adds exactly one history entry - no more, no less.

    Declined trials and refused batches add nothing; a refused batch also
    stops the run. The per-entry identity is the net settlement: the history
    moves by the batch's signed amounts valued against the bankroll as the
    trial began, up to the cent rounding the history reports.
    """
    bets, stakes = portfolio
    strategy, bankroll = run_portfolio(bets, stakes, seed)
    assert bankroll.history[0] == 1000.0

    staked = _staked_evaluations(strategy, stakes)
    settled = 0
    history_index = 1
    for (_probabilities, before), (_won_bets, returns) in zip(
        staked, strategy.settlements, strict=True
    ):
        # The bankroll the trial saw is the last history entry.
        assert before == bankroll.history[history_index - 1]
        if not any(returns):
            # Refused batch: bankroll untouched and the run stops, so the
            # history must already be fully accounted for.
            assert history_index == len(bankroll.history)
            break
        net = sum(ret * before for ret in returns)
        # The history reports cents, but at compounding magnitudes a double's
        # spacing exceeds a cent (one ULP at ~8e13 is 0.0156), and the
        # reconstruction round-trips each return through a division. Allow a
        # cent plus 100 ULPs of the bankroll - a dropped or duplicated
        # settlement would shift the entry by a stake, ~1e10x larger.
        assert abs(bankroll.history[history_index] - (before + net)) <= (
            0.01 + 1e-14 * abs(before)
        )
        history_index += 1
        settled += 1

    assert len(bankroll.history) == 1 + settled


@given(portfolio=portfolios(), seed=st.integers(0, 2**32))
def test_same_seed_replays_identically_under_random_portfolios(portfolio, seed):
    """Same seed, same history - regardless of the drawn portfolio shape."""
    bets, stakes = portfolio
    _, first = run_portfolio(bets, stakes, seed)
    _, second = run_portfolio(bets, stakes, seed)
    assert first.history == second.history


@given(portfolio=portfolios(), seed=st.integers(0, 2**32))
def test_zero_stake_portfolio_never_settles_or_draws(portfolio, seed):
    """A portfolio that stakes nothing is skipped wholesale, every trial."""
    bets, _stakes = portfolio
    m = len(bets)
    simulator = PortfolioSimulator(
        bets=bets, transaction_costs=0.01, trials=30, seed=seed
    )
    strategy = _RecordingStrategy((0.0,) * m)
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
    simulator.evaluate_strategy(strategy, bankroll)
    assert bankroll.history == [1000.0]
    assert strategy.settlements == []
    assert len(strategy.evaluations) == 30
    assert all(
        probabilities == tuple(np.asarray(bets)[:, 0])
        for probabilities, _ in strategy.evaluations
    )
