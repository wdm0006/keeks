"""Boundary parametrization for every validator in the library.

Each 0/1 endpoint is exercised with its documented inclusivity or exclusivity,
gamble inputs are probed across shapes and tolerance edges, and the CRRA
utility laws are pinned at wealth values inside (0, 1]. These tests kill the
boundary-shift mutants (`<` vs `<=`, exclusive vs inclusive range checks) that
the mutation campaign found surviving across the validators.
"""

import math

import numpy as np
import pytest

from keeks.bankroll import BankRoll, RuinError
from keeks.binary_strategies.kelly import (
    DrawdownAdjustedKelly,
    FractionalKellyCriterion,
    KellyCriterion,
)
from keeks.binary_strategies.simple import (
    CPPIStrategy,
    DynamicBankrollManagement,
    FixedFractionStrategy,
    MertonShare,
    OptimalF,
)
from keeks.utils import _normalize_gamble, crra_utility, expected_utility


class TestValidatorEndpoints:
    """0 and 1 are legal values; just outside them is not."""

    def test_kelly_min_probability_endpoints(self):
        for probability in (0.0, 1.0):
            strategy = KellyCriterion(
                payoff=2.0, loss=1.0, transaction_cost=0.0, min_probability=probability
            )
            assert strategy.min_probability == probability

    @pytest.mark.parametrize("bad", [-0.01, 1.1])
    def test_kelly_min_probability_rejects_outside(self, bad):
        with pytest.raises(
            ValueError, match=r"^Minimum probability must be between 0 and 1$"
        ):
            KellyCriterion(
                payoff=2.0, loss=1.0, transaction_cost=0.0, min_probability=bad
            )

    @pytest.mark.parametrize("fraction", (0.0, 1.0))
    def test_fractional_kelly_endpoints(self, fraction):
        strategy = FractionalKellyCriterion(
            payoff=2.0, loss=1.0, transaction_cost=0.0, fraction=fraction
        )
        assert strategy.fraction == fraction

    @pytest.mark.parametrize("bad", (-0.1, 1.1))
    def test_fractional_kelly_rejects_outside(self, bad):
        with pytest.raises(ValueError, match=r"^Fraction must be between 0 and 1$"):
            FractionalKellyCriterion(
                payoff=2.0, loss=1.0, transaction_cost=0.0, fraction=bad
            )

    @pytest.mark.parametrize("fraction", (0.0, 1.0))
    @pytest.mark.parametrize("min_probability", (0.0, 1.0))
    def test_fixed_fraction_endpoints(self, fraction, min_probability):
        strategy = FixedFractionStrategy(
            payoff=2.0,
            loss=1.0,
            transaction_cost=0.0,
            fraction=fraction,
            min_probability=min_probability,
        )
        assert strategy.fraction == fraction

    def test_fixed_fraction_rejects_outside(self):
        with pytest.raises(ValueError, match=r"^Fraction must be between 0 and 1$"):
            FixedFractionStrategy(
                payoff=2.0, loss=1.0, transaction_cost=0.0, fraction=1.01
            )

    def test_dynamic_endpoints_and_clamps(self):
        # 0 and 1 are legal for the base fraction, and min == max is legal.
        strategy = DynamicBankrollManagement(
            base_fraction=0.0,
            payoff=2.0,
            loss=1.0,
            transaction_cost=0.0,
            min_fraction=0.1,
            max_fraction=0.1,
        )
        assert strategy.base_fraction == 0.0

        full = DynamicBankrollManagement(
            base_fraction=1.0, payoff=2.0, loss=1.0, transaction_cost=0.0
        )
        assert full.base_fraction == 1.0

        one_bound = DynamicBankrollManagement(
            base_fraction=0.1,
            payoff=2.0,
            loss=1.0,
            transaction_cost=0.0,
            max_fraction=1.0,
        )
        assert one_bound.max_fraction == 1.0

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"base_fraction": 1.1}, r"^Base fraction must be between 0 and 1$"),
            (
                {"max_fraction": 1.5},
                r"^Min fraction must be between 0 and max fraction",
            ),
            (
                {"min_fraction": 0.3, "max_fraction": 0.2},
                r"^Min fraction must be between 0 and max fraction",
            ),
            ({"window_size": 0}, r"^Window size must be positive integer$"),
            ({"window_size": -3}, r"^Window size must be positive integer$"),
            ({"window_size": "5"}, r"^Window size must be positive integer$"),
            ({"window_size": True}, r"^Window size must be positive integer$"),
            (
                {"min_probability": -0.01},
                r"^Minimum probability must be between 0 and 1$",
            ),
            (
                {"min_probability": 1.1},
                r"^Minimum probability must be between 0 and 1$",
            ),
        ],
    )
    def test_dynamic_rejects_outside(self, kwargs, message):
        parameters = {
            "base_fraction": 0.1,
            "payoff": 2.0,
            "loss": 1.0,
            "transaction_cost": 0.0,
        }
        parameters.update(kwargs)
        with pytest.raises(ValueError, match=message):
            DynamicBankrollManagement(**parameters)

    def test_optimal_f_endpoints(self):
        zero_rate = OptimalF(payoff=2.0, loss=1.0, transaction_cost=0.0, win_rate=0.0)
        assert zero_rate.win_rate == 0.0

        full_risk = OptimalF(
            payoff=2.0,
            loss=1.0,
            transaction_cost=0.0,
            win_rate=1.0,
            max_risk_fraction=1.0,
        )
        assert full_risk.max_risk_fraction == 1.0

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"win_rate": -0.01}, r"^Win rate must be between 0 and 1$"),
            ({"win_rate": 1.1}, r"^Win rate must be between 0 and 1$"),
            (
                {"max_risk_fraction": 0.0},
                r"^Maximum risk fraction must be between 0 and 1$",
            ),
            (
                {"max_risk_fraction": 1.1},
                r"^Maximum risk fraction must be between 0 and 1$",
            ),
        ],
    )
    def test_optimal_f_rejects_outside(self, kwargs, message):
        parameters = {
            "payoff": 2.0,
            "loss": 1.0,
            "transaction_cost": 0.0,
            "win_rate": 0.6,
        }
        parameters.update(kwargs)
        with pytest.raises(ValueError, match=message):
            OptimalF(**parameters)

    def test_cppi_endpoints(self):
        # Fractional wealth is legal.
        small = CPPIStrategy(
            floor_fraction=0.2,
            multiplier=1.0,
            initial_bankroll=0.5,
            payoff=2.0,
            loss=1.0,
            transaction_cost=0.0,
        )
        assert small.floor == 0.5 * 0.2

        # min_probability endpoints are inclusive.
        for probability in (0.0, 1.0):
            strategy = CPPIStrategy(
                floor_fraction=0.2,
                multiplier=1.0,
                initial_bankroll=1000.0,
                payoff=2.0,
                loss=1.0,
                transaction_cost=0.0,
                min_probability=probability,
            )
            assert strategy.min_probability == probability

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"floor_fraction": 0.0}, r"^Floor fraction must be between 0 and 1$"),
            ({"floor_fraction": 1.0}, r"^Floor fraction must be between 0 and 1$"),
            ({"multiplier": 0.0}, r"^Multiplier must be greater than 0$"),
            ({"initial_bankroll": 0.0}, r"^Initial bankroll must be greater than 0$"),
            ({"initial_bankroll": -1.0}, r"^Initial bankroll must be greater than 0$"),
            (
                {"min_probability": -0.01},
                r"^Minimum probability must be between 0 and 1$",
            ),
            (
                {"min_probability": 1.1},
                r"^Minimum probability must be between 0 and 1$",
            ),
        ],
    )
    def test_cppi_rejects_outside(self, kwargs, message):
        parameters = {
            "floor_fraction": 0.2,
            "multiplier": 2.0,
            "initial_bankroll": 1000.0,
            "payoff": 2.0,
            "loss": 1.0,
            "transaction_cost": 0.0,
        }
        parameters.update(kwargs)
        with pytest.raises(ValueError, match=message):
            CPPIStrategy(**parameters)

    def test_merton_endpoints(self):
        tiny = MertonShare(
            payoff=2.0, loss=1.0, transaction_cost=0.0, risk_aversion=1e-9
        )
        assert tiny.risk_aversion == 1e-9

        capped = MertonShare(
            payoff=2.0, loss=1.0, transaction_cost=0.0, max_fraction=1.0
        )
        assert capped.max_fraction == 1.0

        for probability in (0.0, 1.0):
            strategy = MertonShare(
                payoff=2.0, loss=1.0, transaction_cost=0.0, min_probability=probability
            )
            assert strategy.min_probability == probability

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"risk_aversion": 0.0}, r"^Risk aversion must be greater than 0$"),
            ({"risk_aversion": -1.0}, r"^Risk aversion must be greater than 0$"),
            (
                {"min_probability": 1.1},
                r"^Minimum probability must be between 0 and 1$",
            ),
            ({"max_fraction": 0.0}, r"^Maximum fraction must be between 0 and 1$"),
            ({"max_fraction": 1.01}, r"^Maximum fraction must be between 0 and 1$"),
        ],
    )
    def test_merton_rejects_outside(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            MertonShare(payoff=2.0, loss=1.0, transaction_cost=0.0, **kwargs)

    @pytest.mark.parametrize("drawdown", (0.0, 1.0, 1.1))
    def test_drawdown_adjusted_kelly_rejects_outside(self, drawdown):
        with pytest.raises(
            ValueError,
            match=r"^Maximum acceptable drawdown must be between 0 and 1 \(exclusive\)$",
        ):
            DrawdownAdjustedKelly(
                payoff=2.0,
                loss=1.0,
                transaction_cost=0.0,
                max_acceptable_drawdown=drawdown,
            )


class TestCRRALaws:
    """crra_utility: log branch at gamma=1, power branch otherwise, -inf guard."""

    def test_default_risk_aversion_is_log_utility(self):
        assert crra_utility(0.5) == pytest.approx(math.log(0.5), rel=1e-15)

    def test_power_utility_known_values(self):
        assert crra_utility(0.5, 2.0) == pytest.approx(-2.0, rel=1e-15)
        assert crra_utility(2.0, 2.0) == pytest.approx(-0.5, rel=1e-15)
        np.testing.assert_allclose(
            crra_utility(np.array([0.5, 2.0]), 2.0), [-2.0, -0.5], rtol=1e-15
        )

    def test_scalar_matches_single_element_array(self):
        for risk_aversion in (1.0, 1.5, 3.0):
            scalar = crra_utility(0.5, risk_aversion)
            array = crra_utility(np.array([0.5]), risk_aversion)
            assert scalar == array[0]

    def test_nonpositive_wealth_is_minus_infinity(self):
        assert crra_utility(0.0) == -np.inf
        assert crra_utility(-1.0, 2.0) == -np.inf
        np.testing.assert_array_equal(
            crra_utility(np.array([-1.0, 2.0]), 2.0), [-np.inf, -0.5]
        )

    def test_utility_increases_in_wealth(self):
        for risk_aversion in (1.0, 2.0, 5.0):
            assert crra_utility(2.0, risk_aversion) > crra_utility(0.5, risk_aversion)


class TestFractionalWealthEntryPricing:
    """The entry-price and utility APIs accept wealth anywhere in (0, 1]."""

    OUTCOMES = [2.0, -0.4]
    PROBABILITIES = [0.9, 0.1]

    def test_expected_utility_at_fractional_wealth(self):
        value = expected_utility(self.OUTCOMES, self.PROBABILITIES, 0.5, 0.0)
        assert value == pytest.approx(
            0.9 * math.log(0.5 + 2.0) + 0.1 * math.log(0.5 - 0.4), rel=1e-12
        )

    def test_find_indifference_price_at_fractional_wealth(self):
        from keeks.utils import find_indifference_price

        # A sure payout of four times wealth is worth more than the whole
        # search bound, so the price saturates at wealth * max_search_fraction
        # (within the bisection tolerance of the bound).
        with pytest.warns(RuntimeWarning, match="saturated at its search bound"):
            price = find_indifference_price(
                [2.0], [1.0], 0.5, tolerance=0.001, max_search_fraction=1.0
            )
        assert price == pytest.approx(0.5, abs=1e-3)

        # expected_utility stays well-defined and strictly decreasing in the
        # price at points far from any log-domain singularity.
        outcomes, probabilities = [2.0, 0.5], [0.9, 0.1]
        free = expected_utility(outcomes, probabilities, 0.5, 0.0, risk_aversion=1.0)
        priced = expected_utility(outcomes, probabilities, 0.5, 0.25, risk_aversion=1.0)
        assert free > priced > -np.inf

    @pytest.mark.parametrize("bad_wealth", (0.0, -0.5))
    def test_zero_or_negative_wealth_rejected(self, bad_wealth):
        from keeks.utils import find_indifference_price

        with pytest.raises(
            ValueError, match=r"^Current wealth must be greater than 0$"
        ):
            find_indifference_price(self.OUTCOMES, self.PROBABILITIES, bad_wealth)

        with pytest.raises(
            ValueError, match=r"^Current wealth must be greater than 0$"
        ):
            expected_utility(self.OUTCOMES, self.PROBABILITIES, bad_wealth, 0.0)


class TestGambleInputBoundaries:
    """_normalize_gamble shape, emptiness, and tolerance-window edges."""

    def test_cross_shape_outcomes_rejected(self):
        with pytest.raises(ValueError, match=r"must be one-dimensional$"):
            _normalize_gamble([[1.0, 2.0], [3.0, 4.0]], [0.5, 0.5])

    def test_cross_shape_probabilities_rejected(self):
        with pytest.raises(ValueError, match=r"must be one-dimensional$"):
            _normalize_gamble([1.0, 2.0], [[0.5, 0.5]])

    def test_empty_outcomes_rejected(self):
        with pytest.raises(ValueError, match=r"must be non-empty$"):
            _normalize_gamble([], [1.0])

    def test_empty_probabilities_rejected(self):
        with pytest.raises(ValueError, match=r"must be non-empty$"):
            _normalize_gamble([1.0], [])

    def test_mismatched_lengths_rejected(self):
        with pytest.raises(ValueError, match=r"must have equal length$"):
            _normalize_gamble([1.0, 2.0], [1.0])

    def test_non_finite_values_rejected(self):
        with pytest.raises(ValueError, match=r"must contain only finite values$"):
            _normalize_gamble([1.0, math.nan], [0.5, 0.5])

        with pytest.raises(ValueError, match=r"must contain only finite values$"):
            _normalize_gamble([1.0, 2.0], [0.5, math.inf])

    def test_negative_probability_rejected(self):
        with pytest.raises(ValueError, match=r"must be nonnegative$"):
            _normalize_gamble([1.0, -1.0], [1.5, -0.5])

    def test_sum_exactly_one_has_no_implicit_leg(self):
        outcomes, probabilities = _normalize_gamble([2.0, -1.0], [0.6, 0.4])
        assert len(outcomes) == 2
        assert len(probabilities) == 2
        assert probabilities.sum() == pytest.approx(1.0, abs=1e-15)

    def test_zero_probability_leg_is_accepted(self):
        outcomes, probabilities = _normalize_gamble([1.0, 2.0, 3.0], [0.5, 0.3, 0.0])
        assert len(outcomes) == 4
        assert outcomes[-1] == 0.0
        assert probabilities[-1] == pytest.approx(0.2, abs=1e-15)
        assert probabilities.sum() == pytest.approx(1.0, abs=1e-12)

    def test_zero_probability_leg_priced_without_error(self):
        value = expected_utility([1.0, 2.0, 3.0], [0.5, 0.3, 0.0], 1000.0, 0.0)
        assert math.isfinite(value)

    def test_subunity_sum_gets_implicit_zero_leg(self):
        """A sub-unity sum appends an implicit zero-outcome leg (no rescale)."""
        outcomes, probabilities = _normalize_gamble([1.0, 2.0], [0.25, 0.25])
        assert len(outcomes) == 3
        assert outcomes[-1] == 0.0
        np.testing.assert_allclose(probabilities, [0.25, 0.25, 0.5], rtol=1e-15)
        assert probabilities.sum() == pytest.approx(1.0, abs=1e-15)

    def test_probability_tolerance_window(self):
        # 1 + 5e-13 sits inside PROBABILITY_SUM_TOLERANCE: accepted, normalized.
        outcomes, probabilities = _normalize_gamble([1.0], [1.0 + 5e-13])
        assert len(outcomes) == 1
        assert probabilities[0] == pytest.approx(1.0, abs=1e-12)

        # 1 + 5e-12 is an order of magnitude outside the window: rejected.
        with pytest.raises(ValueError, match=r"must sum to no more than one$"):
            _normalize_gamble([1.0], [1.0 + 5e-12])


class TestBankrollBoundaries:
    """BankRoll numeric coercion and the max_draw_down=0 corner."""

    def test_non_numeric_amounts_rejected(self):
        with pytest.raises(
            ValueError, match=r"^initial_funds must be a finite, nonnegative number$"
        ):
            BankRoll(initial_funds=None)

        with pytest.raises(
            ValueError, match=r"^initial_funds must be a finite, nonnegative number$"
        ):
            BankRoll(initial_funds="1000")

        bankroll = BankRoll(initial_funds=100.0)
        with pytest.raises(
            ValueError, match=r"^amt must be a finite, nonnegative number$"
        ):
            bankroll.deposit(None)
        with pytest.raises(
            ValueError, match=r"^amount must be a finite, nonnegative number$"
        ):
            bankroll.bet(None)
        assert bankroll.total_funds == 100.0

    def test_zero_drawdown_rejects_any_positive_withdrawal(self):
        bankroll = BankRoll(initial_funds=100.0, max_draw_down=0)
        bankroll.deposit(50.0)

        with pytest.raises(RuinError, match=r"^You lost too much"):
            bankroll.withdraw(1.0)

        # A zero-amount withdrawal is still a no-op, not ruin.
        bankroll.withdraw(0.0)
        assert bankroll.total_funds == 150.0
