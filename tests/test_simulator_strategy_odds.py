"""Simulators refuse a shipped strategy whose odds contradict their own.

Every simulator sizes a bet through ``strategy.evaluate`` and then settles it
with its own ``payoff``/``loss``. Nothing used to check the two agreed, so a
Kelly strategy sized for even money could be settled at ten-to-one and report a
confident, meaningless bankroll path. ``evaluate_strategy`` now rejects that
configuration before touching the strategy, the bankroll, or any generator.

Duck-typed strategies are deliberately left alone, and the fractional
``transaction_cost`` is never compared against the flat ``transaction_costs``
fee: those units differ on purpose.
"""

import random

import numpy as np
import pytest

from keeks.bankroll import BankRoll
from keeks.binary_strategies.simple import (
    CPPIStrategy,
    DynamicBankrollManagement,
    FixedFractionStrategy,
)
from keeks.simulators.random_binary import RandomBinarySimulator
from keeks.simulators.random_uncertain_binary import RandomUncertainBinarySimulator
from keeks.simulators.repeated_binary import RepeatedBinarySimulator

from .test_simulator_configuration_validation import SIMULATORS, build

SEED = 7
TRIALS = 40


def strategy(**overrides):
    parameters = {
        "fraction": 0.1,
        "payoff": 1.0,
        "loss": 1.0,
        "transaction_cost": 0.0,
    }
    parameters.update(overrides)
    return FixedFractionStrategy(**parameters)


def simulator(simulator_cls, **overrides):
    parameters = {"seed": SEED, "trials": TRIALS, "transaction_costs": 0.0}
    parameters.update(overrides)
    return build(simulator_cls, **parameters)


def run(sim, **strategy_overrides):
    bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
    sim.evaluate_strategy(strategy(**strategy_overrides), bankroll)
    return bankroll


class TestMismatchRejected:
    @pytest.mark.parametrize("simulator_cls", SIMULATORS)
    @pytest.mark.parametrize("field", ["payoff", "loss"])
    def test_mismatch_raises_naming_both_values(self, simulator_cls, field):
        sim = simulator(simulator_cls)

        with pytest.raises(ValueError) as excinfo:
            sim.evaluate_strategy(
                strategy(**{field: 3.5}), BankRoll(initial_funds=1000.0)
            )

        message = str(excinfo.value)
        assert field in message
        assert "3.5" in message
        assert repr(getattr(sim, field)) in message

    @pytest.mark.parametrize("simulator_cls", SIMULATORS)
    @pytest.mark.parametrize("field", ["payoff", "loss"])
    def test_mismatch_leaves_funds_and_history_untouched(self, simulator_cls, field):
        sim = simulator(simulator_cls)
        bankroll = BankRoll(initial_funds=1000.0)

        with pytest.raises(ValueError):
            sim.evaluate_strategy(strategy(**{field: 3.5}), bankroll)

        assert bankroll.total_funds == 1000.0
        assert bankroll.history == [1000.0]

    @pytest.mark.parametrize("simulator_cls", SIMULATORS)
    @pytest.mark.parametrize("field", ["payoff", "loss"])
    def test_mismatch_consumes_no_private_randomness(self, simulator_cls, field):
        """A seeded simulator replays identically after a refused strategy."""
        sim = simulator(simulator_cls)
        with pytest.raises(ValueError):
            sim.evaluate_strategy(
                strategy(**{field: 3.5}), BankRoll(initial_funds=1000.0)
            )

        assert run(sim).history == run(simulator(simulator_cls)).history

    @pytest.mark.parametrize("simulator_cls", SIMULATORS)
    @pytest.mark.parametrize("field", ["payoff", "loss"])
    def test_mismatch_consumes_no_global_randomness(self, simulator_cls, field):
        """The unseeded path draws from the process-global generators."""
        sim = build(simulator_cls, trials=TRIALS, transaction_costs=0.0)
        random.seed(12345)
        np.random.seed(12345)
        expected_random = random.getstate()
        expected_numpy = np.random.get_state()

        with pytest.raises(ValueError):
            sim.evaluate_strategy(
                strategy(**{field: 3.5}), BankRoll(initial_funds=1000.0)
            )

        assert random.getstate() == expected_random
        assert np.array_equal(np.random.get_state()[1], expected_numpy[1])

    @pytest.mark.parametrize("simulator_cls", SIMULATORS)
    def test_mismatch_leaves_stateful_strategy_untouched(self, simulator_cls):
        sim = simulator(simulator_cls)
        dynamic = DynamicBankrollManagement(
            base_fraction=0.1, payoff=2.0, loss=1.0, transaction_cost=0.0
        )

        with pytest.raises(ValueError):
            sim.evaluate_strategy(dynamic, BankRoll(initial_funds=1000.0))

        assert dynamic.results == []
        assert dynamic.current_bankroll is None
        assert dynamic.peak_bankroll is None

    def test_mismatch_precedes_the_update_bankroll_hook(self):
        """The repeated simulator pushes bankroll into CPPI before sizing."""
        sim = simulator(RepeatedBinarySimulator)
        cppi = CPPIStrategy(
            floor_fraction=0.5,
            multiplier=2.0,
            initial_bankroll=500.0,
            payoff=1.0,
            loss=2.0,
        )

        with pytest.raises(ValueError):
            sim.evaluate_strategy(cppi, BankRoll(initial_funds=1000.0))

        assert cppi.current_bankroll == 500.0
        assert cppi.peak_bankroll == 500.0
        assert cppi.floor == 250.0

    def test_ten_to_one_settlement_of_an_even_money_strategy_is_refused(self):
        """The reported defect: even-money sizing settled at ten-to-one."""
        bankroll = BankRoll(initial_funds=100.0, max_draw_down=None)
        sim = RepeatedBinarySimulator(
            payoff=10.0, loss=0.1, transaction_costs=0.0, probability=1.0, trials=1
        )

        with pytest.raises(ValueError):
            sim.evaluate_strategy(strategy(fraction=1.0), bankroll)

        assert bankroll.total_funds == 100.0


class TestMatchingRunsUnchanged:
    @pytest.mark.parametrize(
        ("simulator_cls", "expected_funds", "expected_settled_bets"),
        [
            (RepeatedBinarySimulator, 4072.92, 40),
            (RandomBinarySimulator, 2824.59, 13),
            (RandomUncertainBinarySimulator, 2079.92, 14),
        ],
    )
    def test_seeded_matching_run_reaches_exact_bankroll(
        self, simulator_cls, expected_funds, expected_settled_bets
    ):
        """These values are the ones this configuration produced before the check."""
        bankroll = run(simulator(simulator_cls))

        assert bankroll.total_funds == expected_funds
        assert len(bankroll.history) == expected_settled_bets + 1

    @pytest.mark.parametrize("simulator_cls", SIMULATORS)
    def test_stateful_strategies_still_run_and_record(self, simulator_cls):
        bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
        dynamic = DynamicBankrollManagement(
            base_fraction=0.1, payoff=1.0, loss=1.0, transaction_cost=0.0
        )

        simulator(simulator_cls).evaluate_strategy(dynamic, bankroll)

        assert dynamic.results
        assert len(bankroll.history) > 1

    def test_cppi_still_runs(self):
        bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
        cppi = CPPIStrategy(
            floor_fraction=0.5,
            multiplier=2.0,
            initial_bankroll=1000.0,
            payoff=1.0,
            loss=1.0,
        )

        simulator(RepeatedBinarySimulator).evaluate_strategy(cppi, bankroll)

        assert len(bankroll.history) > 1
        assert cppi.current_bankroll in bankroll.history
        assert cppi.peak_bankroll == max(bankroll.history[:-1])
        assert cppi.floor == 0.5 * cppi.peak_bankroll

    @pytest.mark.parametrize("simulator_cls", SIMULATORS)
    def test_integer_and_float_odds_are_treated_as_equal(self, simulator_cls):
        bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)

        simulator(simulator_cls).evaluate_strategy(strategy(payoff=1, loss=1), bankroll)

        assert len(bankroll.history) > 1


class TestUncomparedConfiguration:
    @pytest.mark.parametrize("simulator_cls", SIMULATORS)
    def test_transaction_cost_units_are_not_compared(self, simulator_cls):
        """The fractional per-unit cost and the flat per-bet fee differ on purpose."""
        bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)
        sim = simulator(simulator_cls, transaction_costs=5.0)

        sim.evaluate_strategy(strategy(transaction_cost=0.01), bankroll)

        assert len(bankroll.history) > 1

    @pytest.mark.parametrize("simulator_cls", SIMULATORS)
    def test_duck_typed_strategy_needs_no_odds(self, simulator_cls):
        """A non-BaseStrategy stays the caller's responsibility, odds or not."""

        class DuckStrategy:
            def evaluate(self, _probability, _current_bankroll):
                return 0.05

        bankroll = BankRoll(initial_funds=1000.0, max_draw_down=None)

        simulator(simulator_cls).evaluate_strategy(DuckStrategy(), bankroll)

        assert len(bankroll.history) > 1
