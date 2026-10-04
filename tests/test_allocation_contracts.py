"""Contract tests for the allocation strategy surface.

``BaseAllocationStrategy`` is the portfolio analogue of the multi-outcome
strategy contract: one long-only weight per distribution-valued option,
validated like the stake-fraction gates, with descriptors bound at
construction. No allocator or simulator exists yet - these tests pin the
contract itself through reference implementations, so the method families
that land later are held to it from birth.
"""

import dataclasses
import re

import numpy as np
import pytest

from keeks.allocation import AllocationResult, BaseAllocationStrategy
from keeks.allocation.base import (
    _validate_covariance,
    _validate_mean,
    _validate_scenarios,
    _validate_strategy_scenarios,
    _validate_weights,
)
from keeks.utils import PROBABILITY_SUM_TOLERANCE


class _EqualWeightAllocator(BaseAllocationStrategy):
    """Reference allocator: equal weights, zeros for a nonpositive bankroll."""

    def __init__(self, option_count):
        self.option_count = option_count

    def evaluate(self, current_bankroll):
        if current_bankroll <= 0:
            return _validate_weights(
                [0.0] * self.option_count, option_count=self.option_count
            )
        return _validate_weights(
            [1.0 / self.option_count] * self.option_count,
            option_count=self.option_count,
        )


class _HookAllocator(_EqualWeightAllocator):
    """Reference allocator implementing both optional simulator hooks."""

    def __init__(self, option_count):
        super().__init__(option_count)
        self.recorded_bankrolls = []
        self.recorded_settlements = []

    def update_bankroll(self, current_bankroll):
        self.recorded_bankrolls.append(current_bankroll)

    def record_settlement(self, won, realized_returns):
        self.recorded_settlements.append((won, realized_returns))


class _ScenarioBoundAllocator(BaseAllocationStrategy):
    """Reference allocator carrying a scenario descriptor, MeanCVaR-shaped."""

    def __init__(self, scenarios):
        self.scenarios = _validate_scenarios(scenarios)[0]

    def evaluate(self, current_bankroll):
        if current_bankroll <= 0:
            return _validate_weights([0.0] * self.scenarios.shape[1])
        return _validate_weights([0.5] * self.scenarios.shape[1])


class _NoneBoundAllocator(BaseAllocationStrategy):
    """Moment-bound allocator whose scenario descriptor is explicitly None."""

    scenarios = None

    def evaluate(self, _current_bankroll):
        return _validate_weights([0.5, 0.5])


class _GarbageDescriptorAllocator(BaseAllocationStrategy):
    """Allocator whose scenario descriptor cannot be read as a matrix."""

    scenarios = object()

    def evaluate(self, _current_bankroll):
        return _validate_weights([0.5, 0.5])


SIMULATOR_SCENARIOS = [[0.01, -0.02], [0.03, 0.01], [0.0, 0.005]]


# ---
# The ABC contract: what it takes to become concrete, and the evaluate
# convention every allocator shares.
# ---


def test_base_is_abstract():
    """The ABC cannot be instantiated without implementing evaluate."""
    with pytest.raises(TypeError, match="abstract"):
        BaseAllocationStrategy()


def test_abstract_surface_is_evaluate_only():
    """evaluate is the one method a concrete allocator must provide."""
    assert BaseAllocationStrategy.__abstractmethods__ == frozenset({"evaluate"})


def test_reference_allocator_returns_validated_weights():
    """A concrete allocator returns one validated weight tuple per option."""
    weights = _EqualWeightAllocator(3).evaluate(100.0)

    assert weights == (1 / 3, 1 / 3, 1 / 3)
    assert all(isinstance(weight, float) for weight in weights)
    assert sum(weights) <= 1 + PROBABILITY_SUM_TOLERANCE


@pytest.mark.parametrize("current_bankroll", [0.0, -100.0])
def test_reference_allocator_returns_zeros_for_nonpositive_bankroll(
    current_bankroll,
):
    """The zeros-for-bankroll-<=-0 convention: nothing left to allocate."""
    assert _EqualWeightAllocator(2).evaluate(current_bankroll) == (0.0, 0.0)


def test_optional_hooks_resolve_getattr_style():
    """Simulators resolve update_bankroll/record_settlement via getattr."""
    allocator = _HookAllocator(2)

    update_bankroll = getattr(allocator, "update_bankroll", None)
    record_settlement = getattr(allocator, "record_settlement", None)
    assert callable(update_bankroll)
    assert callable(record_settlement)

    update_bankroll(150.0)
    record_settlement((True, False), np.asarray([0.01, -0.02]))

    assert allocator.recorded_bankrolls == [150.0]
    assert len(allocator.recorded_settlements) == 1


def test_static_allocators_have_no_hooks():
    """A static allocator exposes neither hook, so resolution is a no-op."""
    allocator = _EqualWeightAllocator(2)

    assert getattr(allocator, "update_bankroll", None) is None
    assert getattr(allocator, "record_settlement", None) is None


# ---
# _validate_weights: the long-only budget gate.
# ---


@pytest.mark.parametrize(
    "weights",
    [
        [0.25, 0.25, 0.5],
        (0.25, 0.25, 0.5),
        np.array([0.25, 0.25, 0.5]),
    ],
)
def test_validate_weights_accepts_any_sequence(weights):
    """Any sequence input is accepted and returned as an immutable tuple."""
    assert _validate_weights(weights) == (0.25, 0.25, 0.5)


def test_validate_weights_coerces_to_floats():
    """Integer inputs come back as floats."""
    weights = _validate_weights([1, 0])

    assert weights == (1.0, 0.0)
    assert all(isinstance(weight, float) for weight in weights)


@pytest.mark.parametrize(
    "weights",
    [
        [1.0],
        [0.5, 0.5],
        [0.5, 0.5 + PROBABILITY_SUM_TOLERANCE / 2],
    ],
)
def test_validate_weights_accepts_full_allocation(weights):
    """A total of exactly one (within tolerance) spends the whole bankroll."""
    _validate_weights(weights)


def test_validate_weights_option_count_gate():
    """An option_count mismatch is rejected, not silently misaligned."""
    assert _validate_weights([0.25, 0.75], option_count=2) == (0.25, 0.75)

    with pytest.raises(
        ValueError, match="Strategy must return exactly 3 weights, got 2"
    ):
        _validate_weights([0.25, 0.75], option_count=3)


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        # None coerces to a 0-d nan array, mirroring validate_probabilities.
        (None, "Strategy weights must be one-dimensional"),
        (object(), "Strategy weights must be a finite sequence"),
        (["a", "b"], "Strategy weights must be a finite sequence"),
        (2.0, "Strategy weights must be one-dimensional"),
        ([[0.25], [0.75]], "Strategy weights must be one-dimensional"),
        ([], "Strategy weights must be non-empty"),
        ([0.25, float("nan")], "Strategy weights must contain only finite values"),
        ([0.25, float("inf")], "Strategy weights must contain only finite values"),
        ([-0.1, 0.5], "Strategy weights must be between 0 and 1"),
        ([1.2, 0.5], "Strategy weights must be between 0 and 1"),
        ([0.6, 0.5], "Strategy weights must sum to no more than one"),
        (
            [0.5, 0.5 + PROBABILITY_SUM_TOLERANCE * 2],
            "Strategy weights must sum to no more than one",
        ),
    ],
)
def test_validate_weights_rejects_bad_input(weights, message):
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        _validate_weights(weights)


# ---
# _validate_mean: the expected-return descriptor gate.
# ---


@pytest.mark.parametrize(
    "mean",
    [
        [0.01, 0.02],
        (0.01, 0.02),
        np.array([0.01, 0.02]),
    ],
)
def test_validate_mean_accepts_any_sequence(mean):
    """Any sequence input is accepted and returned as a float array."""
    validated = _validate_mean(mean)

    assert np.array_equal(validated, np.array([0.01, 0.02]))
    assert validated.dtype == np.float64


@pytest.mark.parametrize(
    ("mean", "message"),
    [
        (None, "Mean must be one-dimensional"),
        (object(), "Mean must be a finite sequence"),
        (0.01, "Mean must be one-dimensional"),
        ([[0.01], [0.02]], "Mean must be one-dimensional"),
        ([], "Mean must be non-empty"),
        ([0.01, float("nan")], "Mean must contain only finite values"),
        ([0.01, float("-inf")], "Mean must contain only finite values"),
    ],
)
def test_validate_mean_rejects_bad_input(mean, message):
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        _validate_mean(mean)


# ---
# _validate_covariance: the square, symmetric, PSD descriptor gate.
# ---


def test_validate_covariance_accepts_valid_matrix():
    """A positive definite matrix passes and comes back as a float array."""
    validated = _validate_covariance([[0.01, 0.002], [0.002, 0.02]])

    assert np.array_equal(validated, np.array([[0.01, 0.002], [0.002, 0.02]]))
    assert validated.dtype == np.float64


def test_validate_covariance_accepts_symmetry_within_tolerance():
    """Floating-point-level asymmetry is noise, not a malformed matrix."""
    _validate_covariance([[1.0, 0.5], [0.5 + 1e-9, 1.0]])


@pytest.mark.parametrize(
    "covariance",
    [
        np.diag([1.0, -1e-11]),
        # The floor is relative to the spectral magnitude, so a
        # large-magnitude matrix gets proportionally more slack.
        np.diag([1e6, -1e-5]),
    ],
)
def test_validate_covariance_accepts_eigenvalues_to_the_floor(covariance):
    """Eigenvalues barely below zero are floating-point noise, accepted."""
    _validate_covariance(covariance)


@pytest.mark.parametrize(
    ("covariance", "message"),
    [
        (None, "Covariance must be two-dimensional"),
        (object(), "Covariance must be a finite sequence"),
        ([0.01, 0.002], "Covariance must be two-dimensional"),
        (np.ones((2, 2, 2)), "Covariance must be two-dimensional"),
        (np.empty((0, 2)), "Covariance must be non-empty"),
        ([[0.01, 0.002, 0.001], [0.002, 0.02, 0.001]], "Covariance must be square"),
        (
            [[0.01, 0.002], [0.002, float("nan")]],
            "Covariance must contain only finite values",
        ),
        (
            [[0.01, 0.002], [0.002, float("inf")]],
            "Covariance must contain only finite values",
        ),
        ([[1.0, 0.5], [0.4, 1.0]], "Covariance must be symmetric"),
    ],
)
def test_validate_covariance_rejects_bad_input(covariance, message):
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        _validate_covariance(covariance)


@pytest.mark.parametrize(
    "covariance",
    [
        np.diag([1.0, -1.0]),
        # Genuinely indefinite matrices are rejected at every scale - the
        # relative floor widens the noise band, not the indefiniteness band.
        np.diag([1e6, -1e-3]),
        np.diag([0.01, -1e-9]),
    ],
)
def test_validate_covariance_rejects_indefinite(covariance):
    with pytest.raises(ValueError, match="Covariance must be positive semidefinite"):
        _validate_covariance(covariance)


def test_validate_covariance_option_count_gate():
    """An option_count mismatch is rejected, not silently misaligned."""
    _validate_covariance(np.eye(2), option_count=2)

    with pytest.raises(
        ValueError,
        match=re.escape("Covariance must be exactly 3x3, got shape (2, 2)"),
    ):
        _validate_covariance(np.eye(2), option_count=3)


# ---
# _validate_scenarios: the joint simple-return observation gate.
# ---


def test_validate_scenarios_rows_are_observations():
    """A (observations, options) matrix passes and keeps its shape."""
    validated, _ = _validate_scenarios(SIMULATOR_SCENARIOS)

    assert validated.shape == (3, 2)
    assert validated.dtype == np.float64


def test_validate_scenarios_without_probabilities_returns_none():
    """No probabilities means equally likely rows, signalled by None."""
    assert _validate_scenarios(SIMULATOR_SCENARIOS)[1] is None


def test_validate_scenarios_never_renormalizes_probabilities():
    """Probability mass below one is all-cash periods, left as-is."""
    _, probabilities = _validate_scenarios(SIMULATOR_SCENARIOS, [0.5, 0.2, 0.1])

    assert np.array_equal(probabilities, np.array([0.5, 0.2, 0.1]))


def test_validate_scenarios_probabilities_must_match_rows():
    """One probability entry per scenario row, no more and no fewer."""
    with pytest.raises(
        ValueError, match="Probabilities must have one entry per scenario row"
    ):
        _validate_scenarios(SIMULATOR_SCENARIOS, [0.5, 0.5])


@pytest.mark.parametrize(
    ("probabilities", "message"),
    [
        ([0.6, 0.5, -0.1], "Probabilities must be nonnegative"),
        ([0.6, 0.5, 0.1], "Probabilities must sum to no more than one"),
    ],
)
def test_validate_scenarios_probabilities_validation_propagates(probabilities, message):
    """Row probabilities go through validate_probabilities unchanged."""
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        _validate_scenarios(SIMULATOR_SCENARIOS, probabilities)


@pytest.mark.parametrize(
    ("scenarios", "message"),
    [
        (None, "Scenarios must be two-dimensional"),
        (object(), "Scenarios must be a finite sequence"),
        ([0.01, -0.02], "Scenarios must be two-dimensional"),
        (np.ones((2, 2, 2)), "Scenarios must be two-dimensional"),
        (np.empty((0, 2)), "Scenarios must be non-empty"),
        (np.empty((2, 0)), "Scenarios must be non-empty"),
        ([[0.01, float("nan")]], "Scenarios must contain only finite values"),
        ([[0.01, float("-inf")]], "Scenarios must contain only finite values"),
    ],
)
def test_validate_scenarios_rejects_bad_input(scenarios, message):
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        _validate_scenarios(scenarios)


# ---
# _validate_strategy_scenarios: the descriptor-equality gate.
# ---


def test_descriptor_gate_accepts_matching_scenarios():
    """An allocator bound to the simulator's scenarios settles that simulator."""
    allocator = _ScenarioBoundAllocator(SIMULATOR_SCENARIOS)

    _validate_strategy_scenarios(
        allocator, np.asarray(SIMULATOR_SCENARIOS, dtype=float)
    )


def test_descriptor_gate_rejects_mismatched_values():
    """Same shape, different returns: the run would describe nothing."""
    allocator = _ScenarioBoundAllocator(SIMULATOR_SCENARIOS)

    with pytest.raises(ValueError, match="do not match the simulator's scenarios"):
        _validate_strategy_scenarios(
            allocator, np.asarray([[0.01, -0.02], [0.03, 0.02], [0.0, 0.005]])
        )


def test_descriptor_gate_rejects_mismatched_shapes():
    """A different scenario set is a mismatch even before values."""
    allocator = _ScenarioBoundAllocator(SIMULATOR_SCENARIOS)

    with pytest.raises(ValueError, match="do not match the simulator's scenarios"):
        _validate_strategy_scenarios(allocator, np.asarray([[0.01, -0.02]]))


def test_descriptor_gate_passes_moment_bound_allocators():
    """Allocators with no scenario descriptor are not the gate's business."""
    mean_covariance_allocator = _EqualWeightAllocator(2)
    none_bound_allocator = _NoneBoundAllocator()

    _validate_strategy_scenarios(
        mean_covariance_allocator, np.asarray(SIMULATOR_SCENARIOS)
    )
    _validate_strategy_scenarios(none_bound_allocator, np.asarray(SIMULATOR_SCENARIOS))


def test_descriptor_gate_rejects_undescribable_descriptor():
    """A garbage scenario descriptor fails loudly, not silently."""
    garbage_allocator = _GarbageDescriptorAllocator()

    with pytest.raises(
        ValueError, match="Strategy scenarios descriptor must be a finite sequence"
    ):
        _validate_strategy_scenarios(
            garbage_allocator, np.asarray(SIMULATOR_SCENARIOS, dtype=float)
        )


# ---
# AllocationResult: frozen diagnostics with identity comparison.
# ---


def test_result_defaults_are_none():
    """Only weights is required; diagnostics default to None."""
    result = AllocationResult(weights=np.array([0.25, 0.75]))

    assert np.array_equal(result.weights, np.array([0.25, 0.75]))
    assert result.objective is None
    assert result.converged is None
    assert result.iterations is None
    assert result.expected_growth is None
    assert result.volatility is None


def test_result_is_frozen():
    """Diagnostics are immutable once produced."""
    result = AllocationResult(weights=np.array([0.25, 0.75]))

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.weights = np.array([1.0, 0.0])


def test_result_comparison_is_identity():
    """eq=False: ndarray fields make generated equality a landmine."""
    first = AllocationResult(weights=np.array([0.25, 0.75]))
    second = AllocationResult(weights=np.array([0.25, 0.75]))

    assert first == first
    assert first != second


def test_result_repr_and_hash():
    """The generated repr names the class; identity hashing works."""
    result = AllocationResult(weights=np.array([0.25, 0.75]), converged=True)

    assert "AllocationResult" in repr(result)
    assert isinstance(hash(result), int)


class _OverBudgetAllocator(BaseAllocationStrategy):
    """Returns a weight vector that splits the bankroll twice over."""

    def evaluate(self, _current_bankroll):
        return (0.6, 0.6)


class _ListReturningAllocator(BaseAllocationStrategy):
    """Returns a list; the base validates and normalizes to a tuple."""

    def evaluate(self, _current_bankroll):
        return [0.5, 0.5]


class _DuckTypedAllocator:
    """Outside the ABC: the simulator's gate is its only contract check."""

    def evaluate(self, _current_bankroll):
        return (0.6, 0.6)


def test_base_enforces_weight_contract():
    """A concrete evaluate returning over-budget weights fails its own call."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Strategy weights must sum to no more than one; got (0.6, 0.6)"
        ),
    ):
        _OverBudgetAllocator().evaluate(1000.0)


def test_base_enforcement_normalizes_valid_vectors():
    """The wrapper's validator is the contract: a list comes back a tuple."""
    assert _ListReturningAllocator().evaluate(1000.0) == (0.5, 0.5)


def test_duck_typed_allocators_are_not_wrapped():
    """Outside the ABC the base adds nothing; the simulator's gate still holds."""
    assert _DuckTypedAllocator().evaluate(1000.0) == (0.6, 0.6)


def test_result_all_cash_reason_defaults_to_none():
    result = AllocationResult(weights=np.array([0.25, 0.75]))

    assert result.all_cash_reason is None
