"""Contract pins that close the remaining mutation-survivor gaps.

Every test here was written against a surviving mutant from the mutmut 3.7.0
campaign on this branch: a default argument, an error-message fragment, a
dtype coercion, an exact float boundary, or a settlement detail that the
existing suites never observed. Each test states the contract it pins so the
assertion reads as a specification, not as noise.
"""

import math
import os
import warnings
from fractions import Fraction

import numpy as np
import pytest

import keeks.utils
from keeks.bankroll import BankRoll
from keeks.binary_strategies.base import BaseStrategy
from keeks.binary_strategies.kelly import DrawdownAdjustedKelly, KellyCriterion
from keeks.binary_strategies.simple import (
    CPPIStrategy,
    DynamicBankrollManagement,
    FixedFractionStrategy,
    MertonShare,
    NaiveStrategy,
    OptimalF,
)
from keeks.simulators.random_binary import RandomBinarySimulator
from keeks.simulators.random_uncertain_binary import RandomUncertainBinarySimulator
from keeks.simulators.repeated_binary import RepeatedBinarySimulator
from keeks.utils import (
    _normalize_gamble,
    _update_strategy_bankroll,
    _validate_entry_price_scalars,
    _validate_stake_fraction,
    crra_utility,
    find_indifference_price,
)

# Closed form for the log-utility indifference price of a 50/50 gamble paying
# +80 / -20 out of wealth 100: p^2 - 260p + 4400 = 0.
LOG_INDIFFERENCE_PRICE = 18.196601125010465


class EntryPriceStub(BaseStrategy):
    """Minimal concrete subclass used to pin the base-class contract."""

    def evaluate(self, probability, current_bankroll):
        del probability, current_bankroll
        return 0.0


class RecordingStrategy(FixedFractionStrategy):
    """Fixed-fraction strategy that records every settlement payload."""

    def __init__(self):
        super().__init__(0.5, 1.0, 1.0, 0.0, min_probability=0.0)
        self.payloads = []

    def record_result(self, won, return_pct=None):
        self.payloads.append((won, return_pct))


class HookCounter(FixedFractionStrategy):
    """Zero-bet strategy that counts per-trial bankroll hook calls."""

    def __init__(self):
        super().__init__(0.0, 2.0, 1.0, 0.0, min_probability=0.0)
        self.calls = 0

    def update_bankroll(self, funds):
        del funds
        self.calls += 1


class TestCRRAUtilityContracts:
    def test_positive_scalar_returns_a_python_numpy_scalar(self):
        result = crra_utility(4.0, 2.0)
        assert np.isscalar(result)
        assert result == pytest.approx(-0.25)

    def test_nonpositive_scalar_returns_scalar_negative_infinity(self):
        result = crra_utility(-1.0, 2.0)
        assert np.isscalar(result)
        assert result == -math.inf

    def test_object_array_is_coerced_to_float64_in_nonpositive_branch(self):
        result = crra_utility(np.array([Fraction(-1, 2), Fraction(1, 2)]), 2.0)
        assert np.asarray(result).dtype == np.float64
        assert result[0] == -math.inf
        assert result[1] == pytest.approx(-2.0)


class TestNormalizeGambleContracts:
    def test_fraction_probabilities_are_coerced_to_float64(self):
        outcomes, probabilities = _normalize_gamble(
            [1, 0], [Fraction(1, 2), Fraction(1, 2)]
        )
        assert probabilities.dtype == np.float64
        assert list(probabilities) == [0.5, 0.5]

    def test_tolerance_window_output_sums_back_to_one(self):
        # total = 1 + 5e-13 sits inside PROBABILITY_SUM_TOLERANCE; the
        # normalization must still divide by it, so the sum returns to 1.0
        # to within float rounding rather than staying at 1 + 5e-13.
        _, probabilities = _normalize_gamble([1, 0], [0.5, 0.5 + 5e-13])
        total = float(probabilities.sum())
        assert 1.0 - 1e-15 <= total < 1.0 + 1e-15


class TestEntryPriceScalarMessages:
    def test_non_numeric_tolerance_names_the_control(self):
        with pytest.raises(ValueError, match=r"^Tolerance must be a finite number$"):
            _validate_entry_price_scalars(100.0, tolerance="x")

    def test_non_numeric_risk_aversion_names_the_control(self):
        with pytest.raises(
            ValueError, match=r"^Risk aversion must be a finite number$"
        ):
            _validate_entry_price_scalars(100.0, risk_aversion="x")

    def test_nonpositive_risk_aversion_message_is_anchored(self):
        with pytest.raises(ValueError, match=r"^Risk aversion must be greater than 0$"):
            _validate_entry_price_scalars(100.0, risk_aversion=0)


class TestStakeFractionMessage:
    def test_non_numeric_stake_names_the_control(self):
        with pytest.raises(
            ValueError, match=r"^Strategy stake fraction must be a finite number$"
        ):
            _validate_stake_fraction("x")


class TestOddsMismatchMessage:
    def test_message_ends_with_the_agreement_clause(self):
        strategy = KellyCriterion(2.0, 1.0, 0.0)
        simulator = RepeatedBinarySimulator(3.0, 1.0, 0.0, 0.6, trials=1, seed=1)
        with pytest.raises(
            ValueError,
            match=r"its own odds while the simulator settles with the "
            r"simulator's, so the two must agree\.$",
        ):
            simulator.evaluate_strategy(strategy, BankRoll(100.0, max_draw_down=None))


class TestUpdateBankrollHook:
    def test_object_without_hook_is_left_alone(self):
        # getattr without a default would raise AttributeError on plain objects.
        _update_strategy_bankroll(object(), 100.0)


class TestFindIndifferencePriceContracts:
    def test_default_arguments_solve_the_log_utility_problem(self):
        result = find_indifference_price([80, -20], [0.5, 0.5], 100.0)
        assert abs(result - LOG_INDIFFERENCE_PRICE) < 0.011

    def test_saturation_message_and_caller_stacklevel(self):
        with pytest.warns(
            RuntimeWarning,
            match=r"^find_indifference_price saturated at its search bound "
            r"\(50\.0 = current_wealth \* max_search_fraction=0\.5\); the "
            r"true indifference price is at or above this value\. Raise "
            r"max_search_fraction to search further\.$",
        ) as record:
            result = find_indifference_price([1000], [1.0], 100.0)
        assert result == pytest.approx(49.9969482421875)
        # stacklevel=2 attributes the warning one frame above utils.py. Under
        # pytest that frame is this test module; under mutmut the trampoline
        # sits between the test and the module under test, so the same
        # contract surfaces as trampoline.py there. Detect the harness from
        # where the module under test is imported from, not from whether
        # mutmut happens to be installed.
        utils_module = keeks.utils
        if "mutants" in utils_module.__file__.split(os.sep):
            assert record[0].filename.endswith(
                os.path.join("mutmut", "mutation", "trampoline.py")
            )
        else:
            assert os.path.abspath(record[0].filename) == os.path.abspath(__file__)

    def test_bisection_stops_at_the_exact_tolerance_boundary(self):
        # All bracket values are exact binary fractions, so the final
        # high - low == tolerance step is representable: the loop must stop
        # on strict inequality and return (7.875 + 8.0) / 2.
        with pytest.warns(RuntimeWarning):
            result = find_indifference_price(
                [1000], [1.0], 16.0, tolerance=0.125, max_search_fraction=0.5
            )
        assert result == 7.9375

    def test_exact_equality_is_not_worth_paying_more_for(self):
        # Paying exactly 25 to receive exactly 25 leaves wealth unchanged, so
        # every probe at a bisection mid that equals the current utility must
        # move the bracket DOWN; the converged price stays below 25.
        result = find_indifference_price([25, 25], [0.5, 0.5], 100.0)
        assert result < 25.0

    def test_zero_search_cap_returns_zero_without_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = find_indifference_price([0], [1.0], 100.0, max_search_fraction=0.0)
        assert result == 0.0


class TestBankRollContracts:
    def test_constructor_defaults(self):
        bankroll = BankRoll()
        assert bankroll.total_funds == 0.0
        assert bankroll.verbose == 0
        assert bankroll.history == [0.0]

    def test_withdraw_rejects_non_numeric_amounts_by_name(self):
        bankroll = BankRoll(100.0)
        with pytest.raises(
            ValueError, match=r"^amt must be a finite, nonnegative number$"
        ):
            bankroll.withdraw("x")

    def test_withdraw_reports_insufficient_funds(self):
        bankroll = BankRoll(100.0)
        with pytest.raises(
            Exception,
            match=r"^Insufficient funds for withdrawal \(would cause bankruptcy\)$",
        ):
            bankroll.withdraw(1000.0)

    def test_bet_reports_insufficient_funds(self):
        # 0.6 + 0.3 accumulates to 0.8999999999999999, whose 2-decimal
        # bettable cap (0.9) is one float ulp ABOVE the true bank: betting the
        # full bettable amount trips the bankruptcy guard with dust left over.
        bankroll = BankRoll(initial_funds=0.6)
        bankroll.add_funds(0.3)
        with pytest.raises(
            Exception, match=r"^Insufficient funds for bet \(would cause bankruptcy\)$"
        ):
            bankroll.bet(0.9)

    def test_add_funds_rejects_non_numeric_amounts_by_name(self):
        bankroll = BankRoll(100.0)
        with pytest.raises(
            ValueError, match=r"^amount must be a finite, nonnegative number$"
        ):
            bankroll.add_funds("x")

    def test_remove_funds_rejects_non_numeric_amounts_by_name(self):
        bankroll = BankRoll(100.0)
        with pytest.raises(
            ValueError, match=r"^amount must be a finite, nonnegative number$"
        ):
            bankroll.remove_funds("x")

    def test_remove_funds_reports_insufficient_funds(self):
        bankroll = BankRoll(100.0)
        with pytest.raises(
            Exception,
            match=r"^Insufficient funds for removal \(would cause bankruptcy\)$",
        ):
            bankroll.remove_funds(1000.0)

    def test_plot_history_draws_history_against_the_trial_index(self, tmp_path):
        bankroll = BankRoll(100.0, max_draw_down=None)
        bankroll.add_funds(10.0)
        bankroll.remove_funds(20.0)
        fname = tmp_path / "history.png"

        bankroll.plot_history(fname=str(fname))

        from matplotlib import pyplot as plt

        try:
            line = plt.gcf().gca().get_lines()[0]
            assert list(line.get_ydata()) == [100.0, 110.0, 90.0]
            assert list(line.get_xdata()) == [0, 1, 2]
            assert line.get_marker() == "o"
        finally:
            plt.close("all")


class TestSimulatorSettlementContracts:
    @pytest.mark.parametrize(
        "simulator_factory",
        [
            lambda: RepeatedBinarySimulator(2.0, 1.0, 0.0, 0.6, trials=5, seed=7),
            lambda: RandomBinarySimulator(2.0, 1.0, 0.0, trials=5, stdev=0.1, seed=7),
            lambda: RandomUncertainBinarySimulator(
                2.0, 1.0, 0.0, trials=5, stdev=0.1, uncertainty_stdev=0.05, seed=7
            ),
        ],
        ids=["repeated", "random", "uncertain"],
    )
    def test_funds_exactly_one_keeps_simulating(self, simulator_factory):
        # The bankruptcy stop is total_funds <= 0, so a bankroll starting at
        # exactly 1.0 must settle its first trial and append history.
        bankroll = BankRoll(initial_funds=1.0, max_draw_down=None)
        strategy = FixedFractionStrategy(0.5, 2.0, 1.0, 0.0, min_probability=0.0)
        simulator_factory().evaluate_strategy(strategy, bankroll)
        assert len(bankroll.history) > 1

    @pytest.mark.parametrize(
        "simulator_factory",
        [
            lambda: RepeatedBinarySimulator(1.0, 1.0, 5.0, 0.99, trials=3, seed=1),
            lambda: RandomBinarySimulator(1.0, 1.0, 5.0, trials=3, stdev=0.1, seed=1),
            lambda: RandomUncertainBinarySimulator(
                1.0, 1.0, 5.0, trials=3, stdev=0.1, uncertainty_stdev=0.05, seed=1
            ),
        ],
        ids=["repeated", "random", "uncertain"],
    )
    def test_break_even_settlement_records_positive_zero(self, simulator_factory):
        # payoff * bet == transaction_costs makes the winning settlement
        # amount exactly 0.0; break-even wins settle through the deposit arm
        # and record a nonnegative signed zero as the win return.
        strategy = RecordingStrategy()
        bankroll = BankRoll(initial_funds=10.0, max_draw_down=None)
        simulator_factory().evaluate_strategy(strategy, bankroll)
        wins = [pct for won, pct in strategy.payloads if won is True]
        assert wins, strategy.payloads
        assert math.copysign(1.0, wins[0]) == 1.0

    @pytest.mark.parametrize(
        "simulator_factory",
        [
            lambda: RandomBinarySimulator(2.0, 1.0, 0.0, stdev=0.1, seed=3),
            lambda: RandomUncertainBinarySimulator(
                2.0, 1.0, 0.0, stdev=0.1, uncertainty_stdev=0.05, seed=3
            ),
        ],
        ids=["random", "uncertain"],
    )
    def test_default_trials_run_exactly_one_thousand_times(self, simulator_factory):
        strategy = HookCounter()
        bankroll = BankRoll(initial_funds=100.0, max_draw_down=None)
        simulator_factory().evaluate_strategy(strategy, bankroll)
        assert strategy.calls == 1000


class TestStrategyBoundaryAndDefaultPins:
    def test_naive_zero_expected_value_returns_float_zero(self):
        # EV is exactly 0.0 at p=0.5 with symmetric 2/2 odds; the early
        # return is a float, not the int 0 the sizing formula would produce.
        result = NaiveStrategy(2.0, 2.0, 0.0).evaluate(0.5, 100.0)
        assert result == 0.0
        assert isinstance(result, float)

    def test_cppi_zero_expected_value_returns_float_zero(self):
        strategy = CPPIStrategy(0.2, 3.0, 100.0, 2.0, 2.0, 0.0)
        result = strategy.evaluate(0.5, 100.0)
        assert result == 0.0
        assert isinstance(result, float)

    def test_cppi_evaluates_at_exact_minimum_probability(self):
        strategy = CPPIStrategy(0.2, 3.0, 100.0, 2.0, 1.0, 0.0)
        assert strategy.evaluate(0.5, 100.0) == pytest.approx(0.8)

    def test_cppi_evaluates_at_bankroll_exactly_one(self):
        strategy = CPPIStrategy(0.05, 5.0, 10.0, 2.0, 1.0, 0.0)
        assert strategy.evaluate(0.9, 1.0) == pytest.approx(0.5)

    def test_cppi_sizes_sub_unit_cushions_by_the_cushion_itself(self):
        # bankroll - floor == 0.5 < 1: a max(1, cushion) mutant would double
        # the exposure and return 0.5/100.5 instead of 0.25/100.5.
        strategy = CPPIStrategy(0.5, 0.5, 200.0, 2.0, 1.0, 0.0)
        assert strategy.evaluate(0.9, 100.5) == pytest.approx(0.25 / 100.5, rel=1e-12)

    def test_cppi_entry_price_is_zero_at_the_floor(self):
        strategy = CPPIStrategy(0.5, 0.5, 200.0, 2.0, 1.0, 0.0)
        assert strategy.calculate_max_entry_price([0], [1], 100.0) == 0.0

    def test_cppi_entry_price_respects_the_default_search_cap(self):
        strategy = CPPIStrategy(0.05, 5.0, 200.0, 2.0, 1.0, 0.0)
        assert strategy.calculate_max_entry_price([0], [1], 100.0) == pytest.approx(
            50.0
        )

    def test_cppi_rejects_non_numeric_multiplier_by_name(self):
        with pytest.raises(ValueError, match=r"^Multiplier must be a finite number$"):
            CPPIStrategy(0.5, "x", 100.0, 2.0, 1.0, 0.0)

    def test_cppi_rejects_non_numeric_initial_bankroll_by_name(self):
        with pytest.raises(
            ValueError, match=r"^Initial bankroll must be a finite number$"
        ):
            CPPIStrategy(0.5, 3.0, "x", 2.0, 1.0, 0.0)

    def test_fixed_fraction_rejects_out_of_range_minimum_probability(self):
        with pytest.raises(
            ValueError, match=r"^Minimum probability must be between 0 and 1$"
        ):
            FixedFractionStrategy(0.1, 2.0, 1.0, 0.0, min_probability=1.5)

    def test_fixed_fraction_entry_price_respects_the_default_search_cap(self):
        strategy = FixedFractionStrategy(0.8, 2.0, 1.0, 0.0)
        assert strategy.calculate_max_entry_price([0], [1], 100.0) == pytest.approx(
            50.0
        )

    def test_dynamic_accepts_minimum_probability_endpoints(self):
        DynamicBankrollManagement(0.1, 2.0, 1.0, 0.0, min_probability=0)
        DynamicBankrollManagement(0.1, 2.0, 1.0, 0.0, min_probability=1)

    def test_dynamic_evaluates_at_exact_minimum_probability(self):
        strategy = DynamicBankrollManagement(0.1, 2.0, 1.0, 0.0, min_probability=0.5)
        assert strategy.evaluate(0.5, 100.0) == pytest.approx(0.1)

    def test_dynamic_volatility_cache_is_a_float_not_a_sentinel_string(self):
        strategy = DynamicBankrollManagement(0.1, 2.0, 1.0, 0.0)
        strategy.record_result(True, 0.1)
        strategy.record_result(False, -0.2)
        assert strategy.get_volatility_factor() == pytest.approx(0.97)

    def test_dynamic_streak_treats_zero_results_as_neither_win_nor_loss(self):
        strategy = DynamicBankrollManagement(0.1, 2.0, 1.0, 0.0)
        for _ in range(3):
            strategy.record_result(True, 0.0)
        assert strategy.get_streak_factor() == pytest.approx(1.15)

    def test_dynamic_streak_with_zero_and_loss_has_no_wins(self):
        strategy = DynamicBankrollManagement(0.1, 2.0, 1.0, 0.0)
        strategy.record_result(True, 0.0)
        strategy.record_result(False, -0.5)
        assert strategy.get_streak_factor() == pytest.approx(0.9)

    def test_dynamic_drawdown_uses_peak_of_one(self):
        strategy = DynamicBankrollManagement(0.1, 2.0, 1.0, 0.0)
        strategy.current_bankroll = 0.5
        strategy.peak_bankroll = 1.0
        assert strategy.get_drawdown_factor() == pytest.approx(0.5)

    def test_dynamic_drawdown_is_neutral_when_only_one_tracker_is_set(self):
        strategy = DynamicBankrollManagement(0.1, 2.0, 1.0, 0.0)
        strategy.current_bankroll = None
        strategy.peak_bankroll = 100.0
        assert strategy.get_drawdown_factor() == pytest.approx(1.0)

    def test_dynamic_entry_price_respects_the_default_search_cap(self):
        strategy = DynamicBankrollManagement(0.8, 2.0, 1.0, 0.0)
        assert strategy.calculate_max_entry_price([0], [1], 100.0) == pytest.approx(
            50.0
        )

    def test_optimal_f_default_entry_price_uses_log_utility(self):
        strategy = OptimalF(2.0, 1.0, 0.0, 0.6)
        result = strategy.calculate_max_entry_price([80, -20], [0.5, 0.5], 100.0)
        assert abs(result - LOG_INDIFFERENCE_PRICE) < 0.011

    def test_kelly_default_entry_price_uses_log_utility(self):
        strategy = KellyCriterion(2.0, 1.0, 0.0)
        result = strategy.calculate_max_entry_price([80, -20], [0.5, 0.5], 100.0)
        assert abs(result - LOG_INDIFFERENCE_PRICE) < 0.011

    def test_drawdown_adjusted_kelly_stores_the_requested_tolerance(self):
        strategy = DrawdownAdjustedKelly(2.0, 1.0, 0.0, max_acceptable_drawdown=0.25)
        assert strategy.max_acceptable_drawdown == pytest.approx(0.25)

    def test_merton_default_risk_aversion_is_two(self):
        assert MertonShare(2.0, 1.0, 0.0).risk_aversion == 2.0

    def test_merton_rejects_non_numeric_risk_aversion_by_name(self):
        with pytest.raises(
            ValueError, match=r"^Risk aversion must be a finite number$"
        ):
            MertonShare(2.0, 1.0, 0.0, risk_aversion="x")

    def test_merton_evaluates_at_exact_minimum_probability(self):
        strategy = MertonShare(2.0, 1.0, 0.0, min_probability=0.5)
        assert strategy.evaluate(0.5, 100.0) == pytest.approx(0.1111111111111111)

    def test_base_strategy_default_transaction_cost_is_zero(self):
        assert EntryPriceStub(2.0, 1.0).transaction_cost == 0.0

    def test_base_strategy_entry_price_message_is_pinned_in_full(self):
        expected = (
            "EntryPriceStub does not support one-time entry price calculation. "
            "This method is only available for utility-based strategies like "
            "KellyCriterion and MertonShare.\n\n"
            "Heuristic strategies (FixedFraction, CPPI, Dynamic, etc.) don't have "
            "underlying utility functions to derive indifference prices from."
        )
        with pytest.raises(NotImplementedError) as excinfo:
            EntryPriceStub(2.0, 1.0).calculate_max_entry_price([0], [1], 100.0)
        assert str(excinfo.value) == expected
