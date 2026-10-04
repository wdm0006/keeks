"""
Tests for the estimator-introspection surface: sklearn conventions.

Covers the ``ParameterMixin`` ``get_params``/``set_params`` pair across every
strategy and allocator family, the ``weights_`` fitted-state convention, the
structural ``option_count`` introspection of the online family, the package
exports the API assessment flagged (``BaseStrategy`` in the binary-strategies
subpackage, the tolerance constants at the root, ``PALETTE``/``COLORMAP`` one
level down), and the annotation coverage of the public surface.

Everything here is introspection only: no behavior changes, and the
bit-exactness goldens must pass untouched.
"""

import inspect

import numpy as np
import pytest

import keeks
from keeks.allocation import COLORMAP, PALETTE
from keeks.allocation.base import (
    COVARIANCE_SYMMETRY_TOLERANCE,
    EIGENVALUE_FLOOR,
)
from keeks.allocation.online import ExponentialGradient, FixedWeights, OnlineNewtonStep
from keeks.allocation.scenarios import MeanCVaR
from keeks.binary_strategies import BaseStrategy
from keeks.binary_strategies.kelly import KellyCriterion
from keeks.multi_outcome.kelly import MultiOutcomeKellyCriterion
from keeks.simulators import (
    RandomBinarySimulator,
    RandomUncertainBinarySimulator,
    RepeatedBinarySimulator,
)

COVARIANCE = [[0.04, 0.001], [0.001, 0.01]]
MEAN = [0.02, 0.01]
SCENARIOS = [[0.03, 0.01], [-0.01, 0.02], [0.01, -0.01]]


def _moment_allocators():
    from keeks import (
        GlobalMinimumVariance,
        MaximumDiversification,
        MaximumSharpe,
        MeanVariance,
        RiskAversionScaling,
        RiskBudgeting,
    )

    return [
        (MeanVariance, (MEAN, COVARIANCE)),
        (GlobalMinimumVariance, (COVARIANCE,)),
        (MaximumSharpe, (MEAN, COVARIANCE)),
        (MaximumDiversification, (COVARIANCE,)),
        (RiskBudgeting, (COVARIANCE,)),
        (RiskAversionScaling, (MeanVariance(MEAN, COVARIANCE),)),
    ]


def _params_equal(left, right):
    """Dict equality that compares ndarrays elementwise."""
    if set(left) != set(right):
        return False
    for key in left:
        first, second = left[key], right[key]
        if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
            if not np.array_equal(np.asarray(first), np.asarray(second)):
                return False
        elif first != second:
            return False
    return True


def _online_allocators():
    return [
        FixedWeights([0.25, 0.75]),
        ExponentialGradient(2),
        OnlineNewtonStep(2),
    ]


# ---------------------------------------------------------------------------
# get_params / set_params round-trips
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls,args",
    _moment_allocators(),
    ids=lambda item: item.__name__ if inspect.isclass(item) else str(item),
)
def test_moment_allocator_get_params_round_trips(cls, args):
    allocator = cls(*args)
    params = allocator.get_params(deep=False)
    assert set(params) == {
        name for name in inspect.signature(cls.__init__).parameters if name != "self"
    }
    # set_params returns self (sklearn convention) and leaves the values equal
    assert allocator.set_params(**params) is allocator
    assert _params_equal(allocator.get_params(deep=False), params)


def test_get_params_deep_false_has_no_nested_entries():
    from keeks import MeanVariance, RiskAversionScaling

    allocator = RiskAversionScaling(MeanVariance(MEAN, COVARIANCE), factor=0.5)
    params = allocator.get_params(deep=False)
    assert set(params) == {"inner", "factor"}


def test_get_params_deep_expands_nested_estimator():
    from keeks import MeanVariance, RiskAversionScaling

    inner = MeanVariance(MEAN, COVARIANCE, risk_aversion=2.0)
    allocator = RiskAversionScaling(inner, factor=0.5)
    params = allocator.get_params(deep=True)
    assert params["inner__risk_aversion"] == 2.0
    assert params["factor"] == 0.5


def test_set_params_nested_path_updates_the_inner_estimator():
    from keeks import RiskAversionScaling

    inner = ExponentialGradient(2, learning_rate=0.05)
    allocator = RiskAversionScaling(inner, factor=0.5)
    allocator.set_params(inner__learning_rate=0.2)
    assert inner.learning_rate == 0.2


def test_set_params_rejects_unknown_names():
    allocator = KellyCriterion(payoff=2.0, loss=1.0, transaction_cost_rate=0.01)
    with pytest.raises(ValueError, match="Invalid parameter 'nonexistent'"):
        allocator.set_params(nonexistent=1.0)


def test_set_params_rejects_unknown_nested_names():
    from keeks import MeanVariance, RiskAversionScaling

    allocator = RiskAversionScaling(MeanVariance(MEAN, COVARIANCE), factor=0.5)
    with pytest.raises(ValueError, match="not an introspectable parameter"):
        allocator.set_params(factor__nonexistent=1.0)


@pytest.mark.parametrize(
    "allocator", _online_allocators(), ids=lambda a: type(a).__name__
)
def test_online_allocator_get_params_round_trips(allocator):
    params = allocator.get_params()
    assert allocator.set_params(**params) is allocator
    assert _params_equal(allocator.get_params(), params)


def test_mean_cvar_get_params_round_trips():
    allocator = MeanCVaR(SCENARIOS, tail_alpha=0.1)
    params = allocator.get_params()
    assert params["tail_alpha"] == 0.1
    assert allocator.set_params(**params) is allocator
    assert _params_equal(allocator.get_params(), params)


def test_fixed_weights_params_expose_the_bound_weights():
    allocator = FixedWeights([0.25, 0.75])
    assert list(allocator.get_params()["weights"]) == [0.25, 0.75]


def test_online_params_are_public_attributes():
    gradient = ExponentialGradient(3, learning_rate=0.1)
    newton = OnlineNewtonStep(3, learning_rate=0.4, epsilon=1e-5)
    assert gradient.learning_rate == 0.1
    assert newton.learning_rate == 0.4
    assert newton.epsilon == 1e-5
    assert not hasattr(gradient, "_learning_rate")
    assert not hasattr(newton, "_learning_rate")
    assert not hasattr(newton, "_epsilon")


def test_binary_and_multi_outcome_strategies_round_trip():
    kelly = KellyCriterion(payoff=2.0, loss=1.0, transaction_cost_rate=0.01)
    assert kelly.set_params(**kelly.get_params()) is kelly
    multi = MultiOutcomeKellyCriterion(
        payoffs=(3.2, 2.4), loss=1.0, transaction_cost_rate=0.01
    )
    assert multi.set_params(**multi.get_params()) is multi


def test_cppi_retains_initial_bankroll_for_introspection():
    from keeks import CPPIStrategy

    strategy = CPPIStrategy(
        floor_fraction=0.9,
        multiplier=3.0,
        initial_bankroll=1000.0,
        payoff=2.0,
        loss=1.0,
    )
    assert strategy.get_params()["initial_bankroll"] == 1000.0
    assert strategy.initial_bankroll == 1000.0


# ---------------------------------------------------------------------------
# option_count: structural, introspectable, guarded
# ---------------------------------------------------------------------------


def test_option_count_reads_back_and_reassigns_the_current_count():
    for allocator in _online_allocators():
        assert allocator.option_count == 2
        allocator.option_count = 2  # equal reassignment is a no-op
        assert allocator.option_count == 2


@pytest.mark.parametrize("count", [1, 5])
def test_option_count_change_is_rejected(count):
    for allocator in _online_allocators():
        with pytest.raises(ValueError, match="option_count is structural"):
            allocator.option_count = count
        assert allocator.option_count == 2


# ---------------------------------------------------------------------------
# weights_: the sklearn fitted-state convention
# ---------------------------------------------------------------------------


def test_weights_aliases_weights_after_optimize():
    from keeks import MeanVariance

    allocator = MeanVariance(MEAN, COVARIANCE)
    allocator.optimize()
    assert isinstance(allocator.weights_, np.ndarray)
    np.testing.assert_allclose(allocator.weights_, allocator.weights)


@pytest.mark.parametrize(
    "allocator", _online_allocators(), ids=lambda a: type(a).__name__
)
def test_online_weights_aliases_the_adaptation_state(allocator):
    assert isinstance(allocator.weights_, np.ndarray)
    np.testing.assert_allclose(allocator.weights_, allocator.weights)


def test_weights_setter_validates_like_construction():
    allocator = FixedWeights([0.25, 0.75])
    with pytest.raises(ValueError, match="must sum to no more than one"):
        allocator.weights = [0.6, 0.6]
    with pytest.raises(ValueError, match="exactly 2"):
        allocator.weights = [0.5]
    allocator.weights = [0.1, 0.5]
    np.testing.assert_allclose(allocator.weights_, [0.1, 0.5])


def test_wrapper_without_fitted_weights_raises_with_guidance():
    from keeks import MeanVariance, RiskAversionScaling

    allocator = RiskAversionScaling(MeanVariance(MEAN, COVARIANCE), factor=0.5)
    with pytest.raises(AttributeError, match="RiskAversionScaling"):
        _ = allocator.weights_


# ---------------------------------------------------------------------------
# Exports the API assessment flagged
# ---------------------------------------------------------------------------


def test_base_strategy_is_subpackage_exported():
    assert "BaseStrategy" in keeks.binary_strategies.__all__
    assert keeks.binary_strategies.BaseStrategy is BaseStrategy
    assert keeks.BaseStrategy is BaseStrategy


def test_tolerance_constants_are_root_exported():
    assert keeks.PROBABILITY_SUM_TOLERANCE == 1e-12
    assert keeks.COVARIANCE_SYMMETRY_TOLERANCE == COVARIANCE_SYMMETRY_TOLERANCE
    assert keeks.EIGENVALUE_FLOOR == EIGENVALUE_FLOOR
    for name in (
        "PROBABILITY_SUM_TOLERANCE",
        "COVARIANCE_SYMMETRY_TOLERANCE",
        "EIGENVALUE_FLOOR",
    ):
        assert name in keeks.__all__


def test_palette_and_colormap_are_one_level_down():
    assert "PALETTE" in keeks.allocation.__all__
    assert "COLORMAP" in keeks.allocation.__all__
    assert keeks.allocation.PALETTE is PALETTE
    assert keeks.allocation.COLORMAP is COLORMAP


# ---------------------------------------------------------------------------
# help()-visible constructor docs
# ---------------------------------------------------------------------------


def test_bankroll_init_docstring_documents_parameters_and_units():
    from keeks import BankRoll

    doc = inspect.getdoc(BankRoll.__init__)
    assert doc is not None
    for parameter in (
        "initial_funds",
        "percent_bettable",
        "max_transaction_loss",
        "verbose",
    ):
        assert parameter in doc
    assert "currency" in doc.lower()
    assert "fraction" in doc.lower()


@pytest.mark.parametrize(
    "simulator",
    [RandomBinarySimulator, RandomUncertainBinarySimulator, RepeatedBinarySimulator],
    ids=lambda s: s.__name__,
)
def test_binary_simulator_init_docstrings_documents_parameters(simulator):
    doc = inspect.getdoc(simulator.__init__)
    assert doc is not None
    for parameter in ("payoff", "loss", "fee_per_bet", "trials", "seed"):
        assert parameter in doc, f"{simulator.__name__}.__init__ misses {parameter}"
    assert "fee" in doc.lower()


# ---------------------------------------------------------------------------
# Annotation coverage of the public surface
# ---------------------------------------------------------------------------


def _public_callables():
    for name in keeks.__all__:
        if name == "__version__":
            continue
        obj = getattr(keeks, name)
        if inspect.isclass(obj):
            if not obj.__module__.startswith("keeks"):
                continue
            if issubclass(obj, Exception):
                continue  # exceptions inherit object.__init__'s signature
            for method_name, method in inspect.getmembers(
                obj, predicate=inspect.isfunction
            ):
                if not method_name.startswith("_"):
                    yield f"{name}.{method_name}", method
            init = obj.__init__
            if init is not object.__init__:
                yield f"{name}.__init__", init
        elif inspect.isfunction(obj):
            yield name, obj


def test_every_public_callable_is_annotated():
    unannotated = {
        qualified
        for qualified, callable_ in _public_callables()
        if not getattr(callable_, "__annotations__", None)
    }
    assert unannotated == set()


def test_every_public_callable_fully_annotates_its_parameters():
    missing = {}
    for qualified, callable_ in _public_callables():
        signature = inspect.signature(callable_)
        gaps = [
            name
            for name, parameter in signature.parameters.items()
            if name not in ("self", "cls")
            and parameter.annotation is inspect.Parameter.empty
        ]
        if gaps:
            missing[qualified] = gaps
    assert missing == {}
