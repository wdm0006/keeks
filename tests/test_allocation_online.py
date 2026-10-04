"""Online allocator tests: update semantics, hooks, and oracle comparisons.

The spec verification row for the online family: ``ExponentialGradient``
beats ``FixedWeights`` on a mean-reverting synthetic series,
``OnlineNewtonStep`` beats ``ExponentialGradient`` on the same series,
state flows only through the ``record_settlement`` hook, and every period's
weights honor the house contract. Around that row sit the validation and
edge matrix: constructor gates, settlement-vector rejections, extreme
returns, single-option portfolios, default hyperparameters, and
run-to-run determinism.
"""

import numpy as np
import pytest

from keeks.allocation import (
    BaseAllocationStrategy,
    ExponentialGradient,
    FixedWeights,
    OnlineNewtonStep,
)
from keeks.allocation.online import (
    _project_to_simplex,
    _validate_option_count,
    _validate_positive_finite,
    _validate_realized_returns,
)
from keeks.utils import PROBABILITY_SUM_TOLERANCE

MEAN_REVERTING_PERIODS = 40
MEAN_REVERTING_AMPLITUDE = 0.1


def _mean_reverting_scenarios(
    periods=MEAN_REVERTING_PERIODS, amplitude=MEAN_REVERTING_AMPLITUDE
):
    """Two anti-correlated mean-reverting options: (+a, -a), (-a, +a), ..."""
    signs = np.where(np.arange(periods) % 2 == 0, 1.0, -1.0)
    first = amplitude * signs
    return np.column_stack([first, -first])


def _run_online(allocator, scenarios, option_count=2):
    """Stake one period at a time, checking the weight contract each period."""
    growth = 1.0
    for realized in scenarios:
        weights = np.asarray(allocator.evaluate(1.0), dtype=float)
        assert weights.shape == (option_count,)
        assert np.all(weights >= 0.0)
        assert np.all(weights <= 1.0)
        assert weights.sum() <= 1.0 + PROBABILITY_SUM_TOLERANCE
        growth *= 1.0 + float(weights @ realized)
        allocator.record_settlement(realized)
    return growth


class TestFixedWeights:
    def test_returns_bound_weights(self):
        allocator = FixedWeights([0.25, 0.25, 0.5])
        assert allocator.evaluate(1000.0) == (0.25, 0.25, 0.5)

    @pytest.mark.parametrize("bankroll", [0.0, -100.0])
    def test_nonpositive_bankroll_returns_zeros(self, bankroll):
        allocator = FixedWeights([0.25, 0.75])
        assert allocator.evaluate(bankroll) == (0.0, 0.0)

    def test_settlement_is_stateless(self):
        allocator = FixedWeights([0.5, 0.5])
        allocator.record_settlement([0.2, -0.2])
        assert allocator.evaluate(1000.0) == (0.5, 0.5)

    @pytest.mark.parametrize(
        "weights",
        [
            [-0.1, 0.5],
            [0.6, 0.6],
            [0.5, float("nan")],
            [0.5, float("inf")],
            [[0.5], [0.5]],
            [],
            0.5,
            "0.5",
        ],
    )
    def test_rejects_invalid_weights(self, weights):
        with pytest.raises(ValueError):
            FixedWeights(weights)

    def test_settlement_validates_shape_and_finiteness(self):
        allocator = FixedWeights([0.5, 0.5])
        with pytest.raises(ValueError, match="exactly 2"):
            allocator.record_settlement([0.1, 0.1, 0.1])
        with pytest.raises(ValueError, match="finite"):
            allocator.record_settlement([0.1, float("nan")])
        with pytest.raises(ValueError, match="one-dimensional"):
            allocator.record_settlement([[0.1, 0.1]])
        with pytest.raises(ValueError, match="non-empty"):
            allocator.record_settlement([])


class TestExponentialGradient:
    def test_uniform_fully_invested_start(self):
        allocator = ExponentialGradient(option_count=2)
        assert allocator.evaluate(1000.0) == (0.5, 0.5)

    @pytest.mark.parametrize("bankroll", [0.0, -100.0])
    def test_nonpositive_bankroll_returns_zeros(self, bankroll):
        allocator = ExponentialGradient(option_count=2, learning_rate=0.5)
        assert allocator.evaluate(bankroll) == (0.0, 0.0)

    def test_follows_the_loser(self):
        allocator = ExponentialGradient(option_count=2, learning_rate=0.5)
        allocator.record_settlement([0.2, -0.1])
        weights = np.asarray(allocator.evaluate(1000.0))
        assert weights[1] > weights[0]

        flipped = ExponentialGradient(option_count=2, learning_rate=0.5)
        flipped.record_settlement([-0.1, 0.2])
        flipped_weights = np.asarray(flipped.evaluate(1000.0))
        assert flipped_weights[0] > flipped_weights[1]

    def test_state_accumulates_across_settlements(self):
        once = ExponentialGradient(option_count=2, learning_rate=0.5)
        once.record_settlement([0.1, -0.1])
        after_one = np.asarray(once.evaluate(1000.0))

        twice = ExponentialGradient(option_count=2, learning_rate=0.5)
        twice.record_settlement([0.1, -0.1])
        twice.record_settlement([0.1, -0.1])
        after_two = np.asarray(twice.evaluate(1000.0))

        assert after_two[1] > after_one[1]

    def test_evaluate_does_not_mutate_state(self):
        allocator = ExponentialGradient(option_count=2, learning_rate=0.5)
        allocator.record_settlement([0.1, -0.1])
        first = allocator.evaluate(1000.0)
        assert allocator.evaluate(1000.0) == first
        assert allocator.evaluate(0.0) == (0.0, 0.0)
        assert allocator.evaluate(1000.0) == first

    def test_zero_return_settlement_leaves_weights_unchanged(self):
        allocator = ExponentialGradient(option_count=3)
        before = allocator.evaluate(1000.0)
        allocator.record_settlement([0.0, 0.0, 0.0])
        assert allocator.evaluate(1000.0) == before

    def test_extreme_returns_stay_finite(self):
        allocator = ExponentialGradient(option_count=2, learning_rate=0.05)
        allocator.record_settlement([-2000.0, 0.0])
        weights = np.asarray(allocator.evaluate(1000.0))
        assert np.all(np.isfinite(weights))
        assert weights[0] > 0.99
        assert weights[1] < weights[0]

    def test_single_option(self):
        allocator = ExponentialGradient(option_count=1)
        assert allocator.evaluate(1000.0) == (1.0,)
        allocator.record_settlement([0.3])
        assert allocator.evaluate(1000.0) == (1.0,)

    def test_rejects_invalid_option_count(self):
        with pytest.raises(ValueError, match="integer"):
            ExponentialGradient(option_count=2.5)
        with pytest.raises(ValueError, match="integer"):
            ExponentialGradient(option_count=True)
        with pytest.raises(ValueError, match="at least one"):
            ExponentialGradient(option_count=0)

    @pytest.mark.parametrize("learning_rate", [0.0, -0.1, float("nan"), float("inf")])
    def test_rejects_invalid_learning_rate(self, learning_rate):
        with pytest.raises(ValueError, match="Learning rate"):
            ExponentialGradient(option_count=2, learning_rate=learning_rate)

    def test_record_settlement_validates(self):
        allocator = ExponentialGradient(option_count=2)
        for bad in ([0.1, 0.1, 0.1], [0.1, float("nan")], [[0.1, 0.1]], [], "nope"):
            with pytest.raises(ValueError):
                allocator.record_settlement(bad)

    def test_default_learning_rate_is_the_spec_value(self):
        default = ExponentialGradient(option_count=2)
        explicit = ExponentialGradient(option_count=2, learning_rate=0.05)
        for realized in _mean_reverting_scenarios(periods=10, amplitude=0.03):
            default.record_settlement(realized)
            explicit.record_settlement(realized)
        assert default.evaluate(1000.0) == explicit.evaluate(1000.0)


class TestOnlineNewtonStep:
    def test_uniform_fully_invested_start(self):
        allocator = OnlineNewtonStep(option_count=2)
        assert allocator.evaluate(1000.0) == (0.5, 0.5)

    @pytest.mark.parametrize("bankroll", [0.0, -100.0])
    def test_nonpositive_bankroll_returns_zeros(self, bankroll):
        allocator = OnlineNewtonStep(option_count=2)
        assert allocator.evaluate(bankroll) == (0.0, 0.0)

    def test_follows_the_loser(self):
        allocator = OnlineNewtonStep(option_count=2)
        allocator.record_settlement([0.05, -0.05])
        weights = np.asarray(allocator.evaluate(1000.0))
        assert weights[1] > weights[0]

        flipped = OnlineNewtonStep(option_count=2)
        flipped.record_settlement([-0.05, 0.05])
        flipped_weights = np.asarray(flipped.evaluate(1000.0))
        assert flipped_weights[0] > flipped_weights[1]

    def test_evaluate_does_not_mutate_state(self):
        allocator = OnlineNewtonStep(option_count=2)
        allocator.record_settlement([0.05, -0.05])
        first = allocator.evaluate(1000.0)
        assert allocator.evaluate(1000.0) == first
        assert allocator.evaluate(0.0) == (0.0, 0.0)
        assert allocator.evaluate(1000.0) == first

    def test_extreme_returns_stay_feasible(self):
        # A huge settlement absorbs the ridge into its own outer product;
        # the pseudo-inverse degrades to the ridge's intended effect instead
        # of raising, and the projection keeps the weights inside the
        # contract with a small follow-the-loser tilt.
        allocator = OnlineNewtonStep(option_count=2)
        allocator.record_settlement([-1e6, 1e6])
        weights = np.asarray(allocator.evaluate(1000.0))
        assert np.all(np.isfinite(weights))
        assert np.all(weights >= 0.0)
        assert np.all(weights <= 1.0)
        assert weights.sum() <= 1.0 + PROBABILITY_SUM_TOLERANCE
        assert weights[0] > weights[1]

    def test_single_option(self):
        allocator = OnlineNewtonStep(option_count=1)
        assert allocator.evaluate(1000.0) == (1.0,)
        allocator.record_settlement([0.1])
        assert allocator.evaluate(1000.0) == (1.0,)

    @pytest.mark.parametrize(
        ("learning_rate", "epsilon"),
        [(0.0, 1e-6), (-0.5, 1e-6), (0.5, 0.0), (0.5, -1e-6), (0.5, float("nan"))],
    )
    def test_rejects_invalid_hyperparameters(self, learning_rate, epsilon):
        with pytest.raises(ValueError):
            OnlineNewtonStep(
                option_count=2, learning_rate=learning_rate, epsilon=epsilon
            )

    def test_default_hyperparameters_are_the_spec_values(self):
        default = OnlineNewtonStep(option_count=2)
        explicit = OnlineNewtonStep(option_count=2, learning_rate=0.5, epsilon=1e-6)
        for realized in _mean_reverting_scenarios(periods=6, amplitude=0.02):
            default.record_settlement(realized)
            explicit.record_settlement(realized)
        assert default.evaluate(1000.0) == explicit.evaluate(1000.0)


class TestOnlineOracles:
    def test_exponential_gradient_beats_fixed_weights_on_mean_reversion(self):
        scenarios = _mean_reverting_scenarios()
        fixed_growth = _run_online(FixedWeights([0.5, 0.5]), scenarios)
        adaptive_growth = _run_online(
            ExponentialGradient(option_count=2, learning_rate=0.5), scenarios
        )
        # The anti-correlated alternating series earns the fixed benchmark
        # exactly nothing.
        assert fixed_growth == pytest.approx(1.0)
        assert adaptive_growth > fixed_growth

    def test_online_newton_step_beats_exponential_gradient(self):
        scenarios = _mean_reverting_scenarios()
        gradient_growth = _run_online(
            ExponentialGradient(option_count=2, learning_rate=0.5), scenarios
        )
        newton_growth = _run_online(OnlineNewtonStep(option_count=2), scenarios)
        assert newton_growth > gradient_growth

    def test_every_member_is_an_allocation_strategy_with_the_hook(self):
        for allocator in (
            FixedWeights([0.5, 0.5]),
            ExponentialGradient(option_count=2),
            OnlineNewtonStep(option_count=2),
        ):
            assert isinstance(allocator, BaseAllocationStrategy)
            assert callable(allocator.record_settlement)


class TestSettlementHistory:
    @staticmethod
    def _replay(allocator, history):
        for realized in history:
            allocator.record_settlement(realized)
        return allocator.evaluate(1000.0)

    def test_exponential_gradient_replays_identically(self):
        history = [
            [0.01, -0.02, 0.005],
            [-0.01, 0.02, 0.0],
            [0.0, 0.0, 0.0],
            [0.02, 0.0, -0.01],
        ]

        def run():
            return self._replay(
                ExponentialGradient(option_count=3, learning_rate=0.3), history
            )

        assert run() == run()

    def test_online_newton_step_replays_identically(self):
        history = [
            [0.01, -0.02, 0.005],
            [-0.01, 0.02, 0.0],
            [0.0, 0.0, 0.0],
            [0.02, 0.0, -0.01],
        ]

        def run():
            return self._replay(
                OnlineNewtonStep(option_count=3, learning_rate=0.3), history
            )

        assert run() == run()


class TestProjectToSimplex:
    def test_feasible_interior_point_unchanged(self):
        weights = _project_to_simplex([0.25, 0.25, 0.5])
        assert np.allclose(weights, [0.25, 0.25, 0.5])
        assert abs(weights.sum() - 1.0) <= PROBABILITY_SUM_TOLERANCE

    def test_overshooting_vector_is_clipped_and_renormalized(self):
        weights = _project_to_simplex([0.7, 0.7])
        assert np.allclose(weights, [0.5, 0.5])
        assert abs(weights.sum() - 1.0) <= PROBABILITY_SUM_TOLERANCE

    def test_negative_coordinates_are_lifted(self):
        weights = _project_to_simplex([-1.0, 2.0])
        assert np.allclose(weights, [0.0, 1.0])
        assert abs(weights.sum() - 1.0) <= PROBABILITY_SUM_TOLERANCE

    def test_input_is_never_modified(self):
        source = np.array([0.25, 0.75])
        weights = _project_to_simplex(source)
        source[0] = 99.0
        assert weights[0] == pytest.approx(0.25)
        assert weights[1] == pytest.approx(0.75)


class TestValidateRealizedReturns:
    def test_returns_float_array(self):
        returns = _validate_realized_returns([0.01, -0.02])
        assert returns.shape == (2,)
        assert returns.dtype == np.float64

    def test_option_count_gate(self):
        assert _validate_realized_returns([0.1, 0.1, 0.1], option_count=3).shape == (3,)
        with pytest.raises(ValueError, match="exactly 3"):
            _validate_realized_returns([0.1, 0.1], option_count=3)

    @pytest.mark.parametrize(
        "bad",
        [
            None,
            "nope",
            0.5,
            [[0.1, 0.1]],
            [],
            [0.1, float("nan")],
            [float("inf"), 0.0],
        ],
    )
    def test_rejects_invalid_returns(self, bad):
        with pytest.raises(ValueError):
            _validate_realized_returns(bad)


class TestSharedValidators:
    def test_option_count(self):
        assert _validate_option_count(3) == 3
        for bad in (0, -1, 2.0, True, "3", None):
            with pytest.raises(ValueError):
                _validate_option_count(bad)

    def test_positive_finite(self):
        assert _validate_positive_finite(0.05, "Learning rate") == 0.05
        for bad in (0.0, -1.0, float("nan"), float("inf"), "x", None):
            with pytest.raises(ValueError, match="positive finite"):
                _validate_positive_finite(bad, "Learning rate")
