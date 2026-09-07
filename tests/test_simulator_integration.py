"""Simulator integration contracts: stream layout, settlement, and early exits.

The seeded tests pin the exact composition of each simulator's random streams —
the numpy ``Generator`` probability/uncertainty stream against the legacy
``random.Random`` outcome stream — with golden values computed independently
from fresh generators. The unseeded tests pin the state-restore behaviour of
the legacy global stream, the exact-0.0 bankruptcy stop, the ``record_result``
payload, and the ``None`` return value of an early-exited run.
"""


import numpy as np
import pytest

from keeks.bankroll import BankRoll
from keeks.simulators.random_binary import RandomBinarySimulator
from keeks.simulators.random_uncertain_binary import RandomUncertainBinarySimulator
from keeks.simulators.repeated_binary import RepeatedBinarySimulator


class RecordingStrategy:
    """Minimal strategy that stakes a fixed fraction and records every call."""

    def __init__(self, fraction):
        self.fraction = fraction
        self.probabilities = []
        self.results = []

    def evaluate(self, probability, _current_bankroll):
        self.probabilities.append(probability)
        return self.fraction

    def record_result(self, won, return_pct):
        self.results.append((won, return_pct))


class TestSeededStreamLayout:
    """A seeded simulator replays its streams exactly, one generator each."""

    def test_uncertain_simulator_two_stream_golden_values(self):
        """Probability+uncertainty come from numpy Generator, outcomes from random.Random.

        Golden values for seed=7, computed independently:
        default_rng(7).normal draws 0.5001230153357482 then 0.47258621446377824
        for probabilities, 0.014937276875423495 then -0.04452959193786371 for
        uncertainty noise; random.Random(7) draws 0.32383276483316237 then
        0.15084917392450192 for outcomes. Any cross-wiring of the streams
        changes these exact values.
        """
        simulator = RandomUncertainBinarySimulator(
            payoff=2.0,
            loss=1.0,
            transaction_costs=0.0,
            trials=2,
            stdev=0.1,
            uncertainty_stdev=0.05,
            seed=7,
        )
        strategy = RecordingStrategy(0.1)
        bankroll = BankRoll(initial_funds=100.0, max_draw_down=None)

        simulator.evaluate_strategy(strategy, bankroll)

        assert strategy.probabilities == [
            0.5001230153357482,
            0.47258621446377824,
        ]
        # Both outcomes win: outcome_probability for trial one is
        # 0.5001230153357482 + 0.014937276875423495 = 0.5150602922111717, and
        # both outcome draws (0.3238..., 0.1508...) fall below their targets.
        # Each settled bet risks 10% of the current funds for a 2x payoff.
        assert strategy.results == [(True, 0.2), (True, 0.2)]
        assert bankroll.history == [100.0, 120.0, 144.0]

    def test_repeated_simulator_uses_only_the_outcome_stream(self):
        """The repeated simulator draws from random.Random(seed) alone."""
        simulator = RepeatedBinarySimulator(
            payoff=2.0,
            loss=1.0,
            transaction_costs=0.0,
            probability=0.5,
            trials=2,
            seed=7,
        )
        strategy = RecordingStrategy(0.1)
        bankroll = BankRoll(initial_funds=100.0, max_draw_down=None)

        numpy_state_before = np.random.get_state()
        simulator.evaluate_strategy(strategy, bankroll)

        # random.Random(7) draws 0.32383276483316237 then 0.15084917392450192;
        # both beat the fixed 0.5 probability.
        assert strategy.probabilities == [0.5, 0.5]
        assert strategy.results == [(True, 0.2), (True, 0.2)]
        assert bankroll.history == [100.0, 120.0, 144.0]

        after = np.random.get_state()
        # The 624-word Mersenne-Twister bit stream and the draw position are
        # both untouched by the seeded run.
        assert np.array_equal(after[1], numpy_state_before[1])
        assert after[2] == numpy_state_before[2]


class TestRecordResultPayload:
    """record_result receives (won: bool, return_pct: float) per settled bet."""

    def test_winning_settlement_payload(self):
        simulator = RepeatedBinarySimulator(
            payoff=2.0,
            loss=1.0,
            transaction_costs=0.0,
            probability=1.0,  # every outcome wins
            trials=1,
            seed=1,
        )
        strategy = RecordingStrategy(0.1)
        bankroll = BankRoll(initial_funds=100.0, max_draw_down=None)

        simulator.evaluate_strategy(strategy, bankroll)

        # Won 2x on a 10-unit bet of a 100-unit bankroll.
        assert strategy.results == [(True, 0.2)]
        assert isinstance(strategy.results[0][0], bool)
        assert isinstance(strategy.results[0][1], float)

    def test_losing_settlement_payload(self):
        simulator = RepeatedBinarySimulator(
            payoff=2.0,
            loss=1.0,
            transaction_costs=0.0,
            probability=0.0,  # every outcome loses
            trials=1,
            seed=1,
        )
        strategy = RecordingStrategy(0.1)
        bankroll = BankRoll(initial_funds=100.0, max_draw_down=None)

        simulator.evaluate_strategy(strategy, bankroll)

        # Lost 1x on a 10-unit bet of a 100-unit bankroll.
        assert strategy.results == [(False, -0.1)]

    def test_fee_shifts_return_pct_on_both_sides(self):
        winning = RepeatedBinarySimulator(
            payoff=2.0,
            loss=1.0,
            transaction_costs=5.0,
            probability=1.0,
            trials=1,
            seed=1,
        )
        losing = RepeatedBinarySimulator(
            payoff=2.0,
            loss=1.0,
            transaction_costs=5.0,
            probability=0.0,
            trials=1,
            seed=1,
        )
        win_strategy = RecordingStrategy(0.1)
        lose_strategy = RecordingStrategy(0.1)

        winning.evaluate_strategy(
            win_strategy, BankRoll(initial_funds=100.0, max_draw_down=None)
        )
        losing.evaluate_strategy(
            lose_strategy, BankRoll(initial_funds=100.0, max_draw_down=None)
        )

        # Fee is an absolute amount: (2*10 - 5)/100 and -(1*10 + 5)/100.
        assert win_strategy.results == [(True, 0.15)]
        assert lose_strategy.results == [(False, -0.15)]


class TestExactZeroRuinStop:
    """A bankroll drained to exactly 0.0 stops the simulation immediately."""

    def test_exact_zero_stops_before_next_trial(self):
        simulator = RepeatedBinarySimulator(
            payoff=1.0,
            loss=1.0,
            transaction_costs=0.0,
            probability=0.1,  # random.Random(1).random() < 0.1 is False: a loss
            trials=10,
            seed=1,
        )
        strategy = RecordingStrategy(1.0)  # stake the entire bettable funds
        bankroll = BankRoll(initial_funds=100.0, max_draw_down=None)

        result = simulator.evaluate_strategy(strategy, bankroll)

        # The full stake is lost: exactly 0.0 remains, and the `<= 0` stop
        # fires before any further evaluation or settlement.
        assert bankroll.total_funds == 0.0
        assert bankroll.history == [100.0, 0.0]
        assert len(strategy.probabilities) == 1
        assert strategy.results == [(False, -1.0)]
        assert result is None

    def test_normal_completion_also_returns_none(self):
        simulator = RepeatedBinarySimulator(
            payoff=1.0,
            loss=1.0,
            transaction_costs=0.0,
            probability=1.0,
            trials=3,
            seed=1,
        )
        strategy = RecordingStrategy(0.1)
        bankroll = BankRoll(initial_funds=100.0, max_draw_down=None)

        result = simulator.evaluate_strategy(strategy, bankroll)

        assert bankroll.history == [100.0, 110.0, 121.0, 133.1]
        assert result is None


class TestUnseededInvalidFractionRestore:
    """Without a seed, the legacy numpy global stream is rolled back on rejection."""

    @pytest.mark.parametrize(
        "simulator_cls", [RandomBinarySimulator, RandomUncertainBinarySimulator]
    )
    def test_global_numpy_state_restored_after_invalid_fraction(self, simulator_cls):
        np.random.seed(123)
        reference_draw = np.random.normal(0.5, 0.1)

        np.random.seed(123)
        kwargs = {
            "payoff": 2.0,
            "loss": 1.0,
            "transaction_costs": 0.0,
            "trials": 1,
            "stdev": 0.1,
        }
        if simulator_cls is RandomUncertainBinarySimulator:
            kwargs["uncertainty_stdev"] = 0.05
        simulator = simulator_cls(**{**kwargs, "seed": None})

        strategy = RecordingStrategy(2.0)  # invalid: outside [0, 1]
        bankroll = BankRoll(initial_funds=100.0, max_draw_down=None)

        with pytest.raises(ValueError, match="Strategy stake fraction"):
            simulator.evaluate_strategy(strategy, bankroll)

        # The one consumed probability draw was rolled back, so the next
        # global draw replays the reference value exactly.
        assert np.random.normal(0.5, 0.1) == reference_draw
        assert bankroll.history == [100.0]
        assert strategy.results == []

    def test_unseeded_probability_draw_reaches_the_strategy(self):
        """The unseeded path draws probabilities from the global numpy stream."""
        np.random.seed(123)
        expected_probability = min(1.0, max(0.0, np.random.normal(0.5, 0.1)))

        np.random.seed(123)
        simulator = RandomUncertainBinarySimulator(
            payoff=2.0,
            loss=1.0,
            transaction_costs=0.0,
            trials=1,
            stdev=0.1,
            uncertainty_stdev=0.05,
            seed=None,
        )
        strategy = RecordingStrategy(0.0)  # valid, and no bet is settled
        bankroll = BankRoll(initial_funds=100.0, max_draw_down=None)

        simulator.evaluate_strategy(strategy, bankroll)

        assert strategy.probabilities == [expected_probability]
        assert strategy.results == []
