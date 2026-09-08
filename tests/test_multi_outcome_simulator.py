"""Tests for the repeated multi-outcome simulator.

The simulator settles one mutually exclusive market per trial: exactly one
leg realizes, the probability mass below one is a void or push, and every
staked leg settles through the binary simulators' net-settlement flow. The
scenarios below pin the known-ledger accounting, the void semantics, the
batch settlement policy on ``RuinError``, the N-ary hook contract, and the
constructor's validation discipline.
"""

import numpy as np
import pytest

from keeks.bankroll import BankRoll
from keeks.multi_outcome import BaseMultiOutcomeStrategy, RepeatedMultiOutcomeSimulator
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

    def record_settlement(self, won_leg, return_pcts):
        self.events.append(("record_settlement", won_leg, return_pcts))


class _DuckTypedStrategy:
    """A strategy outside the ABC hierarchy: no payoffs or loss to check."""

    def __init__(self, stakes):
        self._stakes = stakes

    def evaluate(self, _probabilities, _current_bankroll):
        return _validate_stake_fractions(self._stakes)


def build_simulator(**overrides):
    """A three-leg market with a deterministic outcome knob."""
    common = {
        "payoffs": (2.0, 3.0, 2.4),
        "loss": 1.0,
        "transaction_costs": 0.0,
        "probabilities": (0.4, 0.35, 0.2),
        "trials": 3,
        "seed": 42,
    }
    common.update(overrides)
    return RepeatedMultiOutcomeSimulator(**common)


def _expected_history(initial_funds, fractions, payoff, loss, fee, trials, won_leg):
    """Replicate the documented settlement model for a fixed realized leg.

    Stakes are fractions of the bankroll as it stood when each trial began
    (rounded to cents, as BankRoll reports it); the realized leg pays
    ``payoff * stake - fee`` and every other staked leg is charged
    ``loss * stake + fee``. One history entry lands per settled leg, in leg
    order, and a ``won_leg`` of ``None`` refunds every stake.
    """
    bank = float(initial_funds)
    history = [bank]
    for _ in range(trials):
        bettable = round(bank, 2)
        if won_leg is None:
            continue
        for leg, fraction in enumerate(fractions):
            if fraction <= 0:
                continue
            stake = bettable * fraction
            if leg == won_leg:
                bank += payoff * stake - fee
            else:
                bank -= loss * stake + fee
            history.append(round(bank, 2))
    return history


# ---
# Constructor validation.
# ---


@pytest.mark.parametrize(
    "payoffs",
    [
        [],
        [[2.0, 3.0], [2.4, 2.2]],
        [2.0, float("nan")],
        [2.0, float("inf")],
        [0.0, 3.0],
        [-1.0, 3.0],
    ],
)
def test_constructor_rejects_invalid_payoffs(payoffs):
    with pytest.raises(ValueError):
        build_simulator(payoffs=payoffs)


def test_constructor_rejects_scalar_payoffs():
    with pytest.raises(ValueError, match="one-dimensional"):
        build_simulator(payoffs=5.0)


def test_constructor_rejects_non_sequence_payoffs():
    with pytest.raises(ValueError, match="finite sequence"):
        build_simulator(payoffs="not numbers")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("loss", -0.5),
        ("loss", float("nan")),
        ("transaction_costs", -1.0),
        ("transaction_costs", float("inf")),
    ],
)
def test_constructor_rejects_invalid_scalar_controls(field, value):
    with pytest.raises(ValueError):
        build_simulator(**{field: value})


@pytest.mark.parametrize(
    "probabilities",
    [
        [],
        [[0.4, 0.35], [0.2, 0.05]],
        [0.4, -0.1],
        [0.4, float("nan")],
        [0.6, 0.6],
    ],
)
def test_constructor_rejects_invalid_probabilities(probabilities):
    with pytest.raises(ValueError):
        build_simulator(probabilities=probabilities)


def test_constructor_accepts_probabilities_within_tolerance():
    simulator = build_simulator(probabilities=(0.5, 0.5 + 1e-13))
    assert simulator.probabilities.sum() > 1


@pytest.mark.parametrize("trials", [-1, 3.5, "many"])
def test_constructor_rejects_invalid_trials(trials):
    with pytest.raises(ValueError, match="Trials must be a nonnegative integer"):
        build_simulator(trials=trials)


@pytest.mark.parametrize("seed", [-1, "deterministic", 4.2])
def test_constructor_rejects_invalid_seed(seed):
    with pytest.raises(ValueError, match="Seed must be a nonnegative integer or None"):
        build_simulator(seed=seed)


def test_constructor_stores_validated_configuration():
    payoffs = [2.0, 3.0, 2.4]
    simulator = build_simulator(payoffs=payoffs, trials=7, seed=5)

    payoffs.append(99.0)  # Later mutation of the caller's sequence is ignored.

    assert simulator.payoffs == (2.0, 3.0, 2.4)
    assert simulator.loss == 1.0
    assert simulator.transaction_costs == 0.0
    assert isinstance(simulator.probabilities, np.ndarray)
    assert simulator.trials == 7
    assert simulator.seed == 5


def test_constructor_accepts_zero_trials_and_zero_seed():
    simulator = build_simulator(trials=0, seed=0)
    assert simulator.trials == 0
    assert simulator.seed == 0


# ---
# Odds validation: the simulator and the strategy must agree on the market.
# ---


def test_stake_vector_length_must_match_leg_count():
    # Regression: a strategy returning more fractions than the market has
    # legs used to settle phantom losing legs (charging ``loss`` on stakes
    # for legs that do not exist), and a shorter one silently left legs
    # unstaked.
    simulator = build_simulator()
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
    with pytest.raises(ValueError, match="exactly 3 stake fractions, got 4"):
        simulator.evaluate_strategy(_DuckTypedStrategy((0.1, 0.1, 0.0, 0.1)), bankroll)
    with pytest.raises(ValueError, match="exactly 3 stake fractions, got 2"):
        simulator.evaluate_strategy(_DuckTypedStrategy((0.1, 0.1)), bankroll)
    assert bankroll.history == [1000.0]


def test_rejects_strategy_payoffs_that_disagree():
    strategy = _FixedStakesStrategy(
        payoffs=(2.0, 3.5, 2.4), loss=1.0, stakes=(0.1, 0.1, 0.0)
    )
    simulator = build_simulator()
    with pytest.raises(ValueError, match="does not match simulator payoffs"):
        simulator.evaluate_strategy(
            strategy, BankRoll(initial_funds=1000.0, max_draw_down=None)
        )


def test_rejects_strategy_loss_that_disagrees():
    strategy = _FixedStakesStrategy(
        payoffs=(2.0, 3.0, 2.4), loss=0.5, stakes=(0.1, 0.1, 0.0)
    )
    simulator = build_simulator()
    with pytest.raises(ValueError, match="does not match simulator loss"):
        simulator.evaluate_strategy(
            strategy, BankRoll(initial_funds=1000.0, max_draw_down=None)
        )


def test_accepts_matching_strategy_odds():
    strategy = _FixedStakesStrategy(
        payoffs=(2.0, 3.0, 2.4), loss=1.0, stakes=(0.0, 0.0, 0.0)
    )
    simulator = build_simulator()
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)

    simulator.evaluate_strategy(strategy, bankroll)

    assert bankroll.history == [1000.0]


def test_duck_typed_strategy_needs_no_odds_check():
    strategy = _DuckTypedStrategy(stakes=(0.1, 0.0, 0.0))
    simulator = build_simulator(probabilities=(1.0, 0.0, 0.0))
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)

    simulator.evaluate_strategy(strategy, bankroll)

    # Leg 0 always realizes; the stake is a fraction of the current bankroll.
    assert bankroll.history == _expected_history(
        1000.0, (0.1, 0.0, 0.0), payoff=2.0, loss=1.0, fee=0.0, trials=3, won_leg=0
    )


# ---
# Known-ledger settlement accounting.
# ---


def test_winning_leg_settles_against_a_known_ledger():
    strategy = _FixedStakesStrategy(
        payoffs=(2.0, 3.0, 2.4), loss=1.0, stakes=(0.1, 0.0, 0.0)
    )
    simulator = build_simulator(probabilities=(1.0, 0.0, 0.0), transaction_costs=0.5)
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)

    # Leg 0 always realizes. The stake grows with the bankroll; the fee is
    # flat, so it is subtracted once per settled leg.
    simulator.evaluate_strategy(strategy, bankroll)

    assert bankroll.history == _expected_history(
        1000.0, (0.1, 0.0, 0.0), payoff=2.0, loss=1.0, fee=0.5, trials=3, won_leg=0
    )


def test_fee_dominated_win_is_withdrawn():
    strategy = _FixedStakesStrategy(
        payoffs=(1.0, 3.0, 2.4), loss=1.0, stakes=(0.1, 0.0, 0.0)
    )
    simulator = build_simulator(
        payoffs=(1.0, 3.0, 2.4), probabilities=(1.0, 0.0, 0.0), transaction_costs=150.0
    )
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)

    # Stake = 10% of the current bankroll wins 1 * stake - 150 < 0: the fee
    # dominates, so the net is withdrawn instead of deposited.
    simulator.evaluate_strategy(strategy, bankroll)

    assert bankroll.history == _expected_history(
        1000.0, (0.1, 0.0, 0.0), payoff=1.0, loss=1.0, fee=150.0, trials=3, won_leg=0
    )


def test_exactly_one_leg_settles_per_trial():
    strategy = _FixedStakesStrategy(
        payoffs=(2.0, 3.0, 2.4), loss=1.0, stakes=(0.1, 0.2, 0.0)
    )
    simulator = build_simulator(probabilities=(0.0, 1.0, 0.0))
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)

    # Leg 1 realizes every trial. Leg 0's stake is lost and leg 1's pays
    # 3x: exactly one settlement per leg, in leg order, with the stake
    # frozen at the bankroll the trial began with.
    simulator.evaluate_strategy(strategy, bankroll)

    assert bankroll.history == _expected_history(
        1000.0, (0.1, 0.2, 0.0), payoff=3.0, loss=1.0, fee=0.0, trials=3, won_leg=1
    )


def test_flat_fee_is_charged_per_settled_leg():
    strategy = _FixedStakesStrategy(
        payoffs=(2.0, 3.0, 2.4), loss=1.0, stakes=(0.1, 0.1, 0.0)
    )
    simulator = build_simulator(probabilities=(0.0, 1.0, 0.0), transaction_costs=1.0)
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)

    # Leg 1 realizes: leg 0 loses stake + fee, leg 1 wins payoff * stake -
    # fee. Two settled legs, two fees, one history entry each.
    simulator.evaluate_strategy(strategy, bankroll)

    assert bankroll.history == _expected_history(
        1000.0, (0.1, 0.1, 0.0), payoff=3.0, loss=1.0, fee=1.0, trials=3, won_leg=1
    )


def test_zero_stake_leg_pays_no_fee_even_when_it_wins():
    strategy = _FixedStakesStrategy(
        payoffs=(2.0, 3.0, 2.4), loss=1.0, stakes=(0.1, 0.0, 0.0)
    )
    simulator = build_simulator(probabilities=(0.0, 1.0, 0.0), transaction_costs=1.0)
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)

    # Leg 1 realizes with no stake on it: no settlement for it, so only
    # leg 0's loss plus fee is charged, once per trial.
    simulator.evaluate_strategy(strategy, bankroll)

    assert bankroll.history == _expected_history(
        1000.0, (0.1, 0.0, 0.0), payoff=3.0, loss=1.0, fee=1.0, trials=3, won_leg=1
    )


def test_void_trial_settles_nothing_and_charges_no_fee():
    strategy = _FixedStakesStrategy(
        payoffs=(2.0, 3.0, 2.4), loss=1.0, stakes=(0.1, 0.1, 0.1)
    )
    simulator = build_simulator(probabilities=(0.0, 0.0, 0.0), transaction_costs=1.0)
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)

    # Every draw lands in the residual mass: no leg settles, no fee.
    simulator.evaluate_strategy(strategy, bankroll)

    assert bankroll.history == [1000.0]


def test_all_zero_stakes_skip_the_trial():
    strategy = _FixedStakesStrategy(
        payoffs=(2.0, 3.0, 2.4), loss=1.0, stakes=(0.0, 0.0, 0.0)
    )
    simulator = build_simulator(probabilities=(0.3, 0.3, 0.3), transaction_costs=1.0)
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
    recorder = _RecordingStrategy(
        payoffs=(2.0, 3.0, 2.4), loss=1.0, stakes=(0.0, 0.0, 0.0)
    )

    simulator.evaluate_strategy(strategy, bankroll)
    simulator.evaluate_strategy(
        recorder, BankRoll(initial_funds=1000.0, max_draw_down=None)
    )

    assert bankroll.history == [1000.0]
    # The zero-stake trials still run the update/evaluate traffic; only the
    # draw and the settlement hook are skipped.
    assert (
        recorder.events
        == [
            ("update_bankroll", 1000.0),
            ("evaluate", (0.3, 0.3, 0.3), 1000.0),
        ]
        * 3
    )


def test_skipped_trials_consume_no_draws():
    strategy = _FixedStakesStrategy(
        payoffs=(2.0, 3.0, 2.4), loss=1.0, stakes=(0.0, 0.0, 0.0)
    )
    runner = build_simulator(trials=5, seed=42)
    fresh = build_simulator(trials=0, seed=42)

    runner.evaluate_strategy(
        strategy, BankRoll(initial_funds=1000.0, max_draw_down=None)
    )

    # The stream's state is untouched by a run that never staked, so its next
    # draw matches a fresh generator built from the same seed.
    assert runner._outcome_rng.random() == fresh._outcome_rng.random()


# ---
# Hook traffic: the N-ary record_settlement contract.
# ---


def test_hook_traffic_follows_the_documented_order():
    strategy = _RecordingStrategy(
        payoffs=(2.0, 3.0, 2.4), loss=1.0, stakes=(0.1, 0.0, 0.0)
    )
    simulator = build_simulator(probabilities=(1.0, 0.0, 0.0), trials=2)
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)

    simulator.evaluate_strategy(strategy, bankroll)

    assert strategy.events == [
        ("update_bankroll", 1000),
        ("evaluate", (1.0, 0.0, 0.0), 1000),
        ("record_settlement", 0, (0.2, 0.0, 0.0)),
        ("update_bankroll", 1200.0),
        ("evaluate", (1.0, 0.0, 0.0), 1200.0),
        ("record_settlement", 0, (0.2, 0.0, 0.0)),
    ]


def test_record_settlement_reports_every_leg():
    strategy = _RecordingStrategy(
        payoffs=(2.0, 3.0, 2.4), loss=1.0, stakes=(0.1, 0.1, 0.0)
    )
    simulator = build_simulator(
        probabilities=(0.0, 1.0, 0.0), transaction_costs=1.0, trials=1
    )
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)

    simulator.evaluate_strategy(strategy, bankroll)

    # One settlement event for the single trial: one signed net return per
    # leg, as a fraction of the bankroll before the trial, with the declined
    # leg reporting 0.0.
    stake0 = round(1000.0, 2) * 0.1
    stake1 = round(1000.0, 2) * 0.1
    expected_returns = (
        -(1.0 * stake0 + 1.0) / 1000.0,
        (3.0 * stake1 - 1.0) / 1000.0,
        0.0,
    )
    assert strategy.events == [
        ("update_bankroll", 1000.0),
        ("evaluate", (0.0, 1.0, 0.0), 1000.0),
        ("record_settlement", 1, expected_returns),
    ]


def test_record_settlement_reports_voids():
    strategy = _RecordingStrategy(
        payoffs=(2.0, 3.0, 2.4), loss=1.0, stakes=(0.1, 0.1, 0.1)
    )
    simulator = build_simulator(probabilities=(0.0, 0.0, 0.0))
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)

    simulator.evaluate_strategy(strategy, bankroll)

    # Every draw lands in the residual mass: three void settlements, each
    # reporting no realized leg and zero net change on every leg.
    assert (
        strategy.events
        == [
            ("update_bankroll", 1000.0),
            ("evaluate", (0.0, 0.0, 0.0), 1000.0),
            ("record_settlement", None, (0.0, 0.0, 0.0)),
        ]
        * 3
    )


def test_strategies_without_hooks_run_untouched():
    class _SilentStrategy(BaseMultiOutcomeStrategy):
        def evaluate(self, _probabilities, _current_bankroll):
            return _validate_stake_fractions((0.1, 0.0, 0.0))

    strategy = _SilentStrategy(payoffs=(2.0, 3.0, 2.4), loss=1.0)
    simulator = build_simulator(probabilities=(1.0, 0.0, 0.0), trials=2)
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)

    simulator.evaluate_strategy(strategy, bankroll)

    assert bankroll.history == _expected_history(
        1000.0, (0.1, 0.0, 0.0), payoff=2.0, loss=1.0, fee=0.0, trials=2, won_leg=0
    )


# ---
# Graceful stops: bankruptcy and the settle-full-batch-then-stop policy.
# ---


def test_bankruptcy_stops_the_run():
    recorder = _RecordingStrategy(
        payoffs=(2.0, 3.0, 2.4), loss=1.0, stakes=(1.0, 0.0, 0.0)
    )
    simulator = build_simulator(probabilities=(0.0, 1.0, 0.0), trials=100)
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)

    # Trial 1 loses the entire bankroll (a full withdrawal leaves 0, which is
    # not bankruptcy); trial 2 sees total_funds == 0 and never starts.
    simulator.evaluate_strategy(recorder, bankroll)

    assert bankroll.history == [1000.0, 0.0]
    assert recorder.events == [
        ("update_bankroll", 1000),
        ("evaluate", (0.0, 1.0, 0.0), 1000),
        ("record_settlement", 1, (-1.0, 0.0, 0.0)),
    ]


def test_depleted_bankroll_never_starts_a_trial():
    class _CountingStrategy(BaseMultiOutcomeStrategy):
        def __init__(self):
            super().__init__(payoffs=(2.0, 3.0, 2.4), loss=1.0)
            self.calls = 0

        def evaluate(self, _probabilities, _current_bankroll):
            self.calls += 1
            return _validate_stake_fractions((0.1, 0.0, 0.0))

    strategy = _CountingStrategy()
    simulator = build_simulator(trials=5)
    bankroll = BankRoll(initial_funds=0.0, max_draw_down=None)

    simulator.evaluate_strategy(strategy, bankroll)

    assert strategy.calls == 0
    assert bankroll.history == [0.0]


def test_ruin_error_settles_the_rest_of_the_batch_then_stops():
    strategy = _RecordingStrategy(
        payoffs=(2.0, 3.0, 2.4), loss=1.0, stakes=(0.5, 0.01, 0.0)
    )
    simulator = build_simulator(probabilities=(0.0, 0.0, 1.0), trials=100)
    bankroll = BankRoll(initial_funds=1000.0)  # default max_draw_down=0.3

    # Leg 2 always realizes. Leg 0's 500 loss trips the drawdown limit and is
    # refused; leg 1's 10 loss is still settled; then the run stops.
    simulator.evaluate_strategy(strategy, bankroll)

    assert bankroll.history == [1000.0, 990.0]
    assert strategy.events == [
        ("update_bankroll", 1000),
        ("evaluate", (0.0, 0.0, 1.0), 1000),
        ("record_settlement", 2, (0.0, -0.01, 0.0)),
    ]


def test_ruin_error_on_every_leg_still_completes_the_batch():
    strategy = _FixedStakesStrategy(
        payoffs=(2.0, 3.0, 2.4), loss=1.0, stakes=(0.5, 0.5, 0.0)
    )
    simulator = build_simulator(probabilities=(0.0, 0.0, 1.0), trials=100)
    bankroll = BankRoll(initial_funds=1000.0)  # default max_draw_down=0.3

    # Both losing legs are refused by the drawdown limit; the batch still
    # runs to completion before the simulation stops.
    simulator.evaluate_strategy(strategy, bankroll)

    assert bankroll.history == [1000.0]


def test_ruin_error_after_a_win_still_reports_the_win():
    strategy = _RecordingStrategy(
        payoffs=(2.0, 3.0, 2.4), loss=1.0, stakes=(0.01, 0.5, 0.0)
    )
    simulator = build_simulator(probabilities=(1.0, 0.0, 0.0), trials=100)
    bankroll = BankRoll(initial_funds=1000.0)  # default max_draw_down=0.3

    # Leg 0 realizes and wins first (stake 10 pays 20); leg 1's 500 loss is
    # then refused by the drawdown limit; the batch reports both and stops.
    simulator.evaluate_strategy(strategy, bankroll)

    assert bankroll.history == [1000.0, 1020.0]
    assert strategy.events == [
        ("update_bankroll", 1000),
        ("evaluate", (1.0, 0.0, 0.0), 1000),
        ("record_settlement", 0, (0.02, 0.0, 0.0)),
    ]


# ---
# The unseeded path: numpy's global generator, no replay promise.
# ---


def test_unseeded_runs_replay_when_the_global_generator_is_pinned():
    np.random.seed(20260907)

    def run():
        bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
        strategy = _FixedStakesStrategy(
            payoffs=(2.0, 3.0, 2.4), loss=1.0, stakes=(0.1, 0.05, 0.0)
        )
        simulator = build_simulator(seed=None, trials=50)
        simulator.evaluate_strategy(strategy, bankroll)
        return bankroll.history

    first = run()
    np.random.seed(20260907)
    assert run() == first
