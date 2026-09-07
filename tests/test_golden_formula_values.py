"""Golden-value tests for the strategy sizing formulas.

Every value here is computed by hand from the documented formula in this file
(or plain closed-form algebra), never by calling the library, so a corrupted
formula cannot pass by agreeing with itself. These tests kill the mutation
campaign's highest-risk survivors: three independent corruptions of the
MertonShare variance formula, the untested CPPI floor/clamp behaviour, the
streak/volatility/probability factors of DynamicBankrollManagement, the
DrawdownAdjustedKelly scaling constant, and the unpinned entry-price search
configuration.
"""

import math

import pytest

from keeks.binary_strategies.kelly import (
    DrawdownAdjustedKelly,
    KellyCriterion,
)
from keeks.binary_strategies.simple import (
    CPPIStrategy,
    DynamicBankrollManagement,
    MertonShare,
    NaiveStrategy,
    OptimalF,
)

# MertonShare: f* = mu / (gamma * sigma^2) on gross binary returns.
# Case 1 uses loss=0.5 deliberately: at loss=1 squaring and cubing the loss leg
# coincide, which is exactly why the variance-corruption mutants survived.
MERTON_CASE_1 = {
    "payoff": 2.0,
    "loss": 0.5,
    "transaction_cost": 0.0,
    "risk_aversion": 2.0,
}
MERTON_CASE_2 = {
    "payoff": 2.0,
    "loss": 0.5,
    "transaction_cost": 0.1,
    "risk_aversion": 3.0,
}


def merton_closed_form(p, payoff, loss, transaction_cost, risk_aversion):
    """Independent closed form: mu / (gamma * Var[R]) for the binary return."""
    mu = p * (payoff - transaction_cost) - (1 - p) * (loss + transaction_cost)
    mean = p * payoff - (1 - p) * loss
    variance = p * payoff**2 + (1 - p) * loss**2 - mean**2
    return mu / (risk_aversion * variance)


class TestMertonShareGolden:
    def test_case1_matches_closed_form(self):
        strategy = MertonShare(**MERTON_CASE_1)
        expected = merton_closed_form(0.6, **MERTON_CASE_1)

        assert expected == pytest.approx(1 / 3, abs=1e-12)
        assert strategy.evaluate(0.6, 1000.0) == pytest.approx(expected, abs=1e-12)

    def test_case2_with_transaction_costs_matches_closed_form(self):
        strategy = MertonShare(**MERTON_CASE_2)
        expected = merton_closed_form(0.7, **MERTON_CASE_2)

        assert expected == pytest.approx(0.292063492063492, abs=1e-12)
        assert strategy.evaluate(0.7, 1000.0) == pytest.approx(expected, abs=1e-12)

    def test_max_fraction_clamps_an_aggressive_share(self):
        strategy = MertonShare(
            payoff=2.0,
            loss=0.5,
            transaction_cost=0.0,
            risk_aversion=0.5,
            max_fraction=0.4,
        )
        # Raw share is 1.0 / (0.5 * 1.5) = 1.333...; the cap must bind.
        raw = merton_closed_form(0.6, 2.0, 0.5, 0.0, 0.5)
        assert raw > 0.4
        assert strategy.evaluate(0.6, 1000.0) == 0.4

    def test_degenerate_edges_do_not_bet(self):
        never_edge = MertonShare(payoff=1.0, loss=1.0, transaction_cost=0.3)
        # Negative expected return.
        assert never_edge.evaluate(0.5, 1000.0) == 0.0
        # Zero variance: p=1 leaves no spread in the gross return.
        degenerate = MertonShare(payoff=2.0, loss=0.5, transaction_cost=0.0)
        assert degenerate.evaluate(1.0, 1000.0) == 0.0


class TestCPPIGolden:
    """CPPIStrategy floor protection and exposure clamps, hand-computed."""

    def make(self, multiplier, floor_fraction=0.2, loss=1.0, payoff=2.0):
        return {
            "floor_fraction": floor_fraction,
            "multiplier": multiplier,
            "initial_bankroll": 1000.0,
            "payoff": payoff,
            "loss": loss,
            "transaction_cost": 0.0,
        }

    def test_bankroll_below_floor_bets_nothing(self):
        strategy = CPPIStrategy(**self.make(2.0))
        # Floor is 200; at 100 the cushion is exactly 0 and the bet must be 0.0.
        assert strategy.evaluate(0.6, 100.0) == 0.0

    def test_bankroll_at_floor_bets_nothing(self):
        strategy = CPPIStrategy(**self.make(2.0))
        assert strategy.evaluate(0.6, 200.0) == 0.0

    def test_cushion_sizing_mid_range(self):
        strategy = CPPIStrategy(**self.make(0.5))
        # EV = 0.6*2 - 0.4*1 = 0.8; cushion = 800.
        # exposure = 0.5 * min(1, 0.8) * 800 = 320 -> proportion 0.32.
        assert strategy.evaluate(0.6, 1000.0) == pytest.approx(0.32, abs=1e-12)

    def test_expected_value_cap_at_one(self):
        strategy = CPPIStrategy(**self.make(0.5))
        # EV = 0.9*2 - 0.1*1 = 1.7 > 1: the multiplier must be scaled by
        # min(1.0, EV) = 1.0, not 1.7. exposure = 0.5 * 1.0 * 800 = 400.
        assert strategy.evaluate(0.9, 1000.0) == pytest.approx(0.4, abs=1e-12)

    def test_floor_protection_clamps_before_ruin(self):
        strategy = CPPIStrategy(**self.make(2.0))
        # exposure = 2 * 0.8 * 800 = 1280 -> raw proportion 1.28, but the
        # worst-case floor constraint is (1000-200)/(1000*1) = 0.8.
        assert strategy.evaluate(0.6, 1000.0) == pytest.approx(0.8, abs=1e-12)

    def test_exposure_proportion_before_clamps(self):
        # Small loss and floor: neither the floor constraint (9.0) nor the
        # max-safe bet (1.0) binds; the exposure term sets the proportion.
        strategy = CPPIStrategy(**self.make(0.5, floor_fraction=0.1, loss=0.1))
        # EV = 1.2 - 0.04 = 1.16 -> capped multiplier 0.5; cushion 900.
        assert strategy.evaluate(0.6, 1000.0) == pytest.approx(0.45, abs=1e-12)


class TestDynamicBankrollGolden:
    """Streak / volatility / probability factors and the combined bet size."""

    def make(self, **overrides):
        defaults = {
            "base_fraction": 0.1,
            "payoff": 2.0,
            "loss": 1.0,
            "transaction_cost": 0.0,
            "window_size": 10,
        }
        defaults.update(overrides)
        return DynamicBankrollManagement(**defaults)

    def record(self, strategy, *return_pcts):
        for pct in return_pcts:
            strategy.record_result(pct > 0, pct)

    def test_streak_factor_no_results(self):
        assert self.make().get_streak_factor() == 1.0

    def test_streak_factor_mixed_record(self):
        strategy = self.make()
        self.record(strategy, 2.0, 1.0, -0.5)
        # scale = 3/10; win_ratio = 2/3 -> 1 + (2/3 - 1/2) * 0.3 = 1.05
        assert strategy.get_streak_factor() == pytest.approx(1.05, abs=1e-12)

    def test_streak_factor_all_wins(self):
        strategy = self.make()
        self.record(strategy, 1.0, 1.0, 1.0)
        assert strategy.get_streak_factor() == pytest.approx(1.15, abs=1e-12)

    def test_streak_factor_all_losses(self):
        strategy = self.make()
        self.record(strategy, -1.0, -1.0, -1.0)
        assert strategy.get_streak_factor() == pytest.approx(0.85, abs=1e-12)

    def test_streak_factor_scales_with_window_fill(self):
        full = self.make(window_size=4)
        self.record(full, 1.0, 1.0, 1.0, 1.0)
        # scale = 4/4 = 1 -> maximum boost 1.5.
        assert full.get_streak_factor() == pytest.approx(1.5, abs=1e-12)

        partial = self.make(window_size=10)
        self.record(partial, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        # scale = 6/10 -> 1 + 0.5 * 0.6 = 1.3.
        assert partial.get_streak_factor() == pytest.approx(1.3, abs=1e-12)

    def test_volatility_factor_known_window(self):
        strategy = self.make()
        self.record(strategy, 1.0, -1.0)
        # Population std of [1, -1] is 1.0; scale 0.2 -> 1 - 0.2 = 0.8.
        assert strategy.get_volatility_factor() == pytest.approx(0.8, abs=1e-12)

    def test_volatility_factor_floor_at_half(self):
        strategy = self.make()
        self.record(strategy, 4.0, -4.0)
        # std = 4 -> 1 - 0.8 = 0.2, floored at 0.5.
        assert strategy.get_volatility_factor() == 0.5

    def test_volatility_factor_window_scale(self):
        strategy = self.make(window_size=2)
        self.record(strategy, 1.0, -1.0)
        # scale = 2/2 = 1 -> 1 - 1 = 0, floored at 0.5.
        assert strategy.get_volatility_factor() == 0.5

    def test_volatility_factor_zero_volatility(self):
        strategy = self.make()
        self.record(strategy, 1.0, 1.0)
        assert strategy.get_volatility_factor() == 1.0

    def test_probability_factor_no_results_is_neutral(self):
        assert self.make().get_probability_factor(0.9) == 1.0

    @pytest.mark.parametrize(
        ("probability", "expected"),
        [
            (1.0, 1.5),  # cap
            (0.9, 1.4),
            (0.75, 1.25),
            (0.5, 1.0),
            (0.1, 0.6),
            (0.0, 0.5),  # floor
        ],
    )
    def test_probability_factor_linear_with_cap_and_floor(self, probability, expected):
        strategy = self.make()
        self.record(strategy, 1.0)
        assert strategy.get_probability_factor(probability) == pytest.approx(
            expected, abs=1e-12
        )

    def test_drawdown_factor_tracks_peak(self):
        strategy = self.make()
        strategy.current_bankroll = 800.0
        strategy.peak_bankroll = 1000.0
        assert strategy.get_drawdown_factor() == pytest.approx(0.8, abs=1e-12)

    def test_combined_evaluate_golden(self):
        strategy = self.make()
        # First call: no results yet -> all factors neutral (probability factor
        # only activates once results exist) -> exactly the base fraction.
        assert strategy.evaluate(0.6, 1000.0) == pytest.approx(0.1, abs=1e-12)

        # A win at the peak, then an 800 bankroll: streak 1.05 (1 win, scale 0.1),
        # volatility 1.0 (single sample), drawdown 0.8, probability 1.1.
        strategy.record_result(True)
        assert strategy.evaluate(0.6, 800.0) == pytest.approx(0.0924, abs=1e-12)

    def test_min_and_max_fraction_clamps(self):
        small = self.make(base_fraction=0.01)
        assert small.evaluate(0.6, 1000.0) == pytest.approx(0.05, abs=1e-12)

        large = self.make(base_fraction=0.5)
        assert large.evaluate(0.6, 1000.0) == pytest.approx(0.2, abs=1e-12)

    def test_result_window_drops_oldest(self):
        strategy = self.make(window_size=10)
        for i in range(11):
            strategy.record_result(True, float(i + 1))
        assert len(strategy.results) == 10
        assert strategy.results[0] == 2.0
        assert strategy.results[-1] == 11.0


class TestDrawdownAdjustedKellyGolden:
    """drawdown_factor = min(1, mdd / 0.5), pinned through evaluate."""

    PAYOFF, LOSS, COST, PROBABILITY = 2.0, 1.0, 0.0, 0.7

    def kelly_fraction(self):
        return KellyCriterion(self.PAYOFF, self.LOSS, self.COST).evaluate(
            self.PROBABILITY, 1000.0
        )

    @pytest.mark.parametrize(
        ("drawdown", "factor"),
        [(0.25, 0.5), (0.5, 1.0), (0.75, 1.0)],
    )
    def test_drawdown_factor_scaling(self, drawdown, factor):
        kelly = self.kelly_fraction()
        strategy = DrawdownAdjustedKelly(
            self.PAYOFF, self.LOSS, self.COST, max_acceptable_drawdown=drawdown
        )
        assert strategy.evaluate(self.PROBABILITY, 1000.0) == pytest.approx(
            factor * kelly, rel=1e-12
        )

    def test_scaling_never_exceeds_full_kelly(self):
        kelly = self.kelly_fraction()
        wide = DrawdownAdjustedKelly(
            self.PAYOFF, self.LOSS, self.COST, max_acceptable_drawdown=0.75
        )
        assert wide.evaluate(self.PROBABILITY, 1000.0) <= kelly + 1e-12


class TestEntryPriceGolden:
    """Entry-price precision and search configuration, pinned to 1e-3."""

    GAMBLE_OUTCOMES = [100.0, -20.0]
    GAMBLE_PROBABILITIES = [0.5, 0.5]
    WEALTH = 100.0

    def closed_form_log_price(self):
        """Solve 0.5*ln(x+100) + 0.5*ln(x-20) = ln(100) by quadratic formula."""
        a, b, w = 100.0, 20.0, self.WEALTH
        x = (-(a - b) + math.sqrt((a - b) ** 2 + 4 * (a * b + w * w))) / 2
        return w - x

    def test_kelly_log_utility_entry_price_closed_form(self):
        strategy = KellyCriterion(payoff=1.0, loss=1.0, transaction_cost=0.0)
        expected = self.closed_form_log_price()
        assert expected == pytest.approx(23.380962103093992, abs=1e-9)

        price = strategy.calculate_max_entry_price(
            self.GAMBLE_OUTCOMES,
            self.GAMBLE_PROBABILITIES,
            self.WEALTH,
            tolerance=0.0005,
        )
        assert price == pytest.approx(expected, abs=1e-3)

    def test_optimal_f_log_utility_entry_price_closed_form(self):
        strategy = OptimalF(payoff=1.0, loss=1.0, transaction_cost=0.0, win_rate=0.6)
        price = strategy.calculate_max_entry_price(
            self.GAMBLE_OUTCOMES,
            self.GAMBLE_PROBABILITIES,
            self.WEALTH,
            tolerance=0.0005,
        )
        assert price == pytest.approx(self.closed_form_log_price(), abs=1e-3)

    def test_default_tolerance_and_search_fraction_are_pinned(self):
        """Defaults tolerance=0.01 / max_search_fraction=0.5 must hold jointly."""
        strategy = KellyCriterion(payoff=1.0, loss=1.0, transaction_cost=0.0)
        # A sure 10x payout saturates the default bound: 1000 * 0.5 = 500.
        with pytest.warns(RuntimeWarning, match="saturated at its search bound"):
            price = strategy.calculate_max_entry_price([10000.0], [1.0], 1000.0)
        assert price == pytest.approx(500.0, abs=0.01)

    def test_naive_entry_price_is_capped_at_search_bound(self):
        strategy = NaiveStrategy(payoff=1.0, loss=1.0, transaction_cost=0.0)
        assert strategy.calculate_max_entry_price([1e9], [1.0], 1000.0) == 500.0

    def test_naive_negative_ev_gamble_prices_at_zero(self):
        strategy = NaiveStrategy(payoff=1.0, loss=1.0, transaction_cost=0.0)
        # EV = 0.25*1 - 0.75*1 = -0.5 -> unwilling to pay anything.
        assert (
            strategy.calculate_max_entry_price([1.0, -1.0], [0.25, 0.75], 1000.0) == 0.0
        )

    def test_naive_zero_expected_value_bets_nothing(self):
        strategy = NaiveStrategy(payoff=1.0, loss=1.0, transaction_cost=0.0)
        assert strategy.evaluate(0.5, 1000.0) == 0.0


class TestOptimalFGolden:
    def test_probability_gate_at_half(self):
        strategy = OptimalF(payoff=2.0, loss=1.0, transaction_cost=0.0, win_rate=0.6)
        # optimal f = 0.6 - 0.4/(2/1) = 0.4, capped at max_risk_fraction 0.2,
        # converted to a stake fraction: 0.2 / (1 + 0) = 0.2.
        assert strategy.evaluate(0.5, 1000.0) == pytest.approx(0.2, abs=1e-12)
        assert strategy.evaluate(0.49, 1000.0) == 0.0

    def test_unprofitable_costs_do_not_bet(self):
        strategy = OptimalF(payoff=0.5, loss=1.0, transaction_cost=0.5, win_rate=0.6)
        assert strategy.evaluate(0.7, 1000.0) == 0.0
