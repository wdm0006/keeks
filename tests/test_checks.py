"""Contract tests for the ``keeks.checks`` harness.

The checks are the contributor-facing contract harness on the
``check_estimator`` pattern: each probe in this file pins a check's verdict
against the shipped implementations (which must pass silently) and against
deliberately broken implementations (which must fail with the expectation
and the received value named).
"""

import re

import numpy as np
import pytest

from keeks import (
    BaseAllocationStrategy,
    CPPIStrategy,
    DrawdownAdjustedKelly,
    DynamicBankrollManagement,
    FixedFractionStrategy,
    FixedWeights,
    FractionalKellyCriterion,
    KellyCriterion,
    MertonShare,
    NaiveStrategy,
    OptimalF,
    binary_bets_model,
    check_allocation_strategy,
    check_model,
    check_strategy,
    scenario_model,
)
from keeks.allocation.models import JointReturnModel


def _all_shipped_binary_strategies():
    return [
        KellyCriterion(payoff=2.0, loss=1.0, transaction_cost=0.01),
        FractionalKellyCriterion(
            payoff=2.0, loss=1.0, transaction_cost=0.01, fraction=0.5
        ),
        DrawdownAdjustedKelly(payoff=2.0, loss=1.0, transaction_cost=0.01),
        NaiveStrategy(payoff=2.0, loss=1.0, transaction_cost=0.01),
        FixedFractionStrategy(fraction=0.1, payoff=2.0, loss=1.0),
        CPPIStrategy(
            floor_fraction=0.1,
            multiplier=3.0,
            initial_bankroll=1000.0,
            payoff=2.0,
            loss=1.0,
        ),
        DynamicBankrollManagement(
            base_fraction=0.1, payoff=2.0, loss=1.0, transaction_cost=0.01
        ),
        OptimalF(payoff=2.0, loss=1.0, transaction_cost=0.01, win_rate=0.55),
        MertonShare(payoff=2.0, loss=1.0, transaction_cost=0.01),
    ]


class _RecklessStrategy(KellyCriterion):
    """Sizes beyond the bankroll - every contract's favorite villain."""

    def evaluate(self, _probability, _current_bankroll):
        return 1.5


class _NanStrategy(KellyCriterion):
    """Returns a non-finite fraction."""

    def evaluate(self, _probability, _current_bankroll):
        return float("nan")


class _TupleStrategy(KellyCriterion):
    """Returns a vector where the binary contract wants one float."""

    def evaluate(self, _probability, _current_bankroll):
        return (0.25, 0.25)


class _DeadbeatStrategy(KellyCriterion):
    """Stakes at a depleted bankroll."""

    def evaluate(self, _probability, _current_bankroll):
        return 0.25


class _UnsafeCapStrategy(KellyCriterion):
    """Overrides the safe-stake cap to answer more than the bankroll."""

    def get_max_safe_bet(self, current_bankroll):
        return 2.0 if current_bankroll > 0 else 0.0


class _NoCapAtRuinStrategy(KellyCriterion):
    """Overrides the safe-stake cap to stake into a depleted bankroll."""

    def evaluate(self, probability, current_bankroll):
        if current_bankroll <= 0:
            return 0.0
        return super().evaluate(probability, current_bankroll)

    def get_max_safe_bet(self, _current_bankroll):
        return 0.5  # ignores the bankroll - even at ruin


class _UncallableHookStrategy(KellyCriterion):
    """Defines a hook the simulators would resolve but could not call."""


class _OverBudgetAllocator(BaseAllocationStrategy):
    """Splits the bankroll twice over."""

    def evaluate(self, _current_bankroll):
        return (0.6, 0.6)


class _RuinStakerAllocator(BaseAllocationStrategy):
    """Allocates into a depleted bankroll."""

    def evaluate(self, _current_bankroll):
        return (0.5, 0.5)


class _ShrinkingAllocator(BaseAllocationStrategy):
    """Returns a different option count as the bankroll falls."""

    def evaluate(self, current_bankroll):
        if current_bankroll <= 0:
            return (0.0,)
        return (0.5, 0.5)


class _GoodAllocator(BaseAllocationStrategy):
    """A contract-honest reference allocator."""

    def evaluate(self, current_bankroll):
        return (0.25, 0.25) if current_bankroll > 0 else (0.0, 0.0)


class _DriftingModel(JointReturnModel):
    """Ignores the generator it is handed, breaking replay."""

    def sample(self, n_samples, _rng):
        return np.random.default_rng().uniform(size=(n_samples, 2))


class _FlatModel(JointReturnModel):
    """Returns a 1-D draw vector instead of an (n_samples, N) matrix."""

    def sample(self, n_samples, _rng):
        return np.zeros(n_samples)


class _VoidModel(JointReturnModel):
    """Draws zero option columns."""

    def sample(self, n_samples, _rng):
        return np.zeros((n_samples, 0))


class _NanModel(JointReturnModel):
    """Draws non-finite returns."""

    def sample(self, n_samples, _rng):
        return np.full((n_samples, 2), np.nan)


class _ArityDriftModel(JointReturnModel):
    """Changes the option count with the sample count."""

    def sample(self, n_samples, _rng):
        columns = 2 if n_samples == 8 else 3
        return np.zeros((n_samples, columns))


class _BadMomentsModel(JointReturnModel):
    """Knows moments, wrongly shaped."""

    def sample(self, n_samples, _rng):
        return np.zeros((n_samples, 2))

    def moments(self):
        return np.zeros(3), np.zeros((3, 3))


class _PairlessMomentsModel(JointReturnModel):
    """Knows moments, unpackable as a (mean, covariance) pair."""

    def sample(self, n_samples, _rng):
        return np.zeros((n_samples, 2))

    def moments(self):
        return 0.0


# --- check_strategy ---------------------------------------------------------


@pytest.mark.parametrize("strategy", _all_shipped_binary_strategies())
def test_check_strategy_passes_for_shipped_strategies(strategy):
    check_strategy(strategy)


def test_check_strategy_rejects_non_strategy():
    with pytest.raises(TypeError, match="check_strategy expects a keeks binary"):
        check_strategy(object())


def test_check_strategy_flags_fraction_out_of_range():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "evaluate(0.0, 1000.0) must return a bankroll fraction between 0 "
            "and 1, got 1.5"
        ),
    ):
        check_strategy(_RecklessStrategy(payoff=2.0, loss=1.0, transaction_cost=0.01))


def test_check_strategy_flags_nonfinite_fraction():
    with pytest.raises(
        ValueError, match="must return a finite bankroll fraction, got nan"
    ):
        check_strategy(_NanStrategy(payoff=2.0, loss=1.0, transaction_cost=0.01))


def test_check_strategy_flags_non_scalar_fraction():
    with pytest.raises(
        ValueError,
        match=re.escape("must return a single bankroll fraction, got (0.25, 0.25)"),
    ):
        check_strategy(_TupleStrategy(payoff=2.0, loss=1.0, transaction_cost=0.01))


def test_check_strategy_flags_staking_at_ruin():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "evaluate(0.6, 0.0) must return 0.0 - a nonpositive bankroll has "
            "nothing left to stake, got 0.25"
        ),
    ):
        check_strategy(_DeadbeatStrategy(payoff=2.0, loss=1.0, transaction_cost=0.01))


def test_check_strategy_flags_cap_out_of_range():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "get_max_safe_bet(1000.0) must return a bankroll fraction between "
            "0 and 1, got 2.0"
        ),
    ):
        check_strategy(_UnsafeCapStrategy(payoff=2.0, loss=1.0, transaction_cost=0.01))


def test_check_strategy_flags_staking_into_ruin_cap():
    with pytest.raises(ValueError, match="get_max_safe_bet\\(0.0\\) must return 0.0"):
        check_strategy(
            _NoCapAtRuinStrategy(payoff=2.0, loss=1.0, transaction_cost=0.01)
        )


def test_check_strategy_flags_uncallable_hook():
    strategy = _UncallableHookStrategy(payoff=2.0, loss=1.0, transaction_cost=0.01)
    strategy.update_bankroll = 5
    with pytest.raises(ValueError, match="update_bankroll must be callable"):
        check_strategy(strategy)


# --- check_allocation_strategy ----------------------------------------------


def test_check_allocation_strategy_passes_for_shipped_and_reference():
    check_allocation_strategy(FixedWeights([0.25, 0.75]))
    check_allocation_strategy(_GoodAllocator())


def test_check_allocation_strategy_rejects_non_strategy():
    with pytest.raises(
        TypeError, match="check_allocation_strategy expects a keeks allocation"
    ):
        check_allocation_strategy(object())


def test_check_allocation_strategy_flags_over_budget_weights():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Strategy weights must sum to no more than one; got (0.6, 0.6)"
        ),
    ):
        check_allocation_strategy(_OverBudgetAllocator())


def test_check_allocation_strategy_flags_staking_at_ruin():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "evaluate(0.0) must return all zeros - a nonpositive bankroll has "
            "nothing left to allocate, got (0.5, 0.5)"
        ),
    ):
        check_allocation_strategy(_RuinStakerAllocator())


def test_check_allocation_strategy_flags_arity_drift():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "evaluate must return one weight per option at every bankroll: "
            "evaluate(1000.0) returned 2, evaluate(0.0) returned 1"
        ),
    ):
        check_allocation_strategy(_ShrinkingAllocator())


# --- check_model ------------------------------------------------------------


def test_check_model_passes_for_shipped_models():
    check_model(binary_bets_model([(0.55, 2.0, 1.0), (0.30, 2.5, 1.0)]))
    check_model(scenario_model([[0.02, -0.01], [-0.01, 0.02], [0.01, 0.005]]))


def test_check_model_rejects_non_model():
    with pytest.raises(TypeError, match="check_model expects a keeks joint-return"):
        check_model(object())


def test_check_model_flags_nondeterministic_sampling():
    with pytest.raises(ValueError, match="sample must be deterministic"):
        check_model(_DriftingModel())


def test_check_model_flags_nondraw_matrix():
    with pytest.raises(
        ValueError,
        match=re.escape("sample(8, rng) must return an (8, N) matrix"),
    ):
        check_model(_FlatModel())


def test_check_model_flags_empty_option_count():
    with pytest.raises(ValueError, match="at least one option column"):
        check_model(_VoidModel())


def test_check_model_flags_nonfinite_draws():
    with pytest.raises(ValueError, match="sample must return finite simple returns"):
        check_model(_NanModel())


def test_check_model_flags_option_count_drift():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "sample(3, rng) must return shape (3, 2) - one column per option"
        ),
    ):
        check_model(_ArityDriftModel())


def test_check_model_flags_bad_moment_shapes():
    with pytest.raises(ValueError, match="moments\\(\\) must return a shape \\(2,\\)"):
        check_model(_BadMomentsModel())


def test_check_model_flags_unpairable_moments():
    with pytest.raises(
        ValueError, match="moments\\(\\) must return a \\(mean, covariance\\) pair"
    ):
        check_model(_PairlessMomentsModel())


def test_check_model_accepts_moments_none():
    class _NoMomentsModel(JointReturnModel):
        def sample(self, n_samples, _rng):
            return np.zeros((n_samples, 2))

    check_model(_NoMomentsModel())
