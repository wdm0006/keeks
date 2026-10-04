"""
Contract checks for strategies and joint-return models, on the
``check_estimator`` pattern.

The strategy contracts are enforced at the base classes: a concrete
``evaluate`` returning a contract-violating vector fails loudly at its own
call site. A contributor implementing
:class:`keeks.binary_strategies.BaseStrategy`,
:class:`keeks.allocation.BaseAllocationStrategy`, or
:class:`keeks.allocation.models.JointReturnModel` has nonetheless had no way
to verify an implementation against the documented contract without running
a simulator. The checks here are that harness: each runs the contract's
probes and raises - ``ValueError`` naming the expectation and the received
value for a contract violation, ``TypeError`` for a subject outside the
check's ABC - and passes silently when everything holds.

Examples
--------
>>> from keeks import FixedWeights, check_allocation_strategy
>>> check_allocation_strategy(FixedWeights([0.25, 0.75]))
"""

import numpy as np

from keeks.allocation.base import BaseAllocationStrategy, _validate_weights
from keeks.allocation.models import JointReturnModel
from keeks.binary_strategies.base import BaseStrategy

__all__ = ["check_allocation_strategy", "check_model", "check_strategy"]

_PROBE_BANKROLL = 1000.0
_PROBE_PROBABILITIES = (0.0, 0.3, 0.6, 1.0)


def check_strategy(strategy: BaseStrategy) -> None:
    """
    Verify a binary strategy against the documented ``BaseStrategy`` contract.

    Runs the contract probes: ``evaluate`` returns a single finite bankroll
    fraction within ``[0, 1]`` across a probability grid; a nonpositive
    bankroll returns exactly ``0.0`` - there is nothing left to stake;
    ``get_max_safe_bet`` stays a fraction within ``[0, 1]`` and answers
    ``0.0`` for a nonpositive bankroll; and the optional simulator hooks
    ``update_bankroll`` and ``record_settlement``, when defined, are callable -
    the shapes the simulators resolve ``getattr``-style.

    Parameters
    ----------
    strategy : BaseStrategy
        The binary strategy to check.

    Raises
    ------
    TypeError
        If ``strategy`` is not a :class:`keeks.binary_strategies.BaseStrategy`
        subclass.
    ValueError
        If any probe violates the contract.

    Examples
    --------
    >>> from keeks import KellyCriterion, check_strategy
    >>> check_strategy(KellyCriterion(payoff=2.0, loss=1.0, transaction_cost_rate=0.01))

    A contract violation raises with the expectation and the received value:

    >>> class _Reckless(KellyCriterion):
    ...     def evaluate(self, probability, current_bankroll):
    ...         return 1.5
    >>> check_strategy(_Reckless(payoff=2.0, loss=1.0, transaction_cost_rate=0.01))
    Traceback (most recent call last):
        ...
    ValueError: evaluate(0.0, 1000.0) must return a bankroll fraction between 0 and 1, got 1.5
    """
    if not isinstance(strategy, BaseStrategy):
        raise TypeError(
            "check_strategy expects a keeks binary strategy (a BaseStrategy "
            f"subclass), got {type(strategy).__name__}"
        )

    for probability in _PROBE_PROBABILITIES:
        fraction = strategy.evaluate(probability, _PROBE_BANKROLL)
        try:
            fraction = float(fraction)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"evaluate({probability}, {_PROBE_BANKROLL}) must return a "
                f"single bankroll fraction, got {fraction!r}"
            ) from exc
        if not np.isfinite(fraction):
            raise ValueError(
                f"evaluate({probability}, {_PROBE_BANKROLL}) must return a "
                f"finite bankroll fraction, got {fraction!r}"
            )
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(
                f"evaluate({probability}, {_PROBE_BANKROLL}) must return a "
                f"bankroll fraction between 0 and 1, got {fraction!r}"
            )

    for bankroll in (0.0, -_PROBE_BANKROLL):
        fraction = float(strategy.evaluate(0.6, bankroll))
        if fraction != 0.0:
            raise ValueError(
                f"evaluate(0.6, {bankroll}) must return 0.0 - a nonpositive "
                f"bankroll has nothing left to stake, got {fraction!r}"
            )

    safe = float(strategy.get_max_safe_bet(_PROBE_BANKROLL))
    if not 0.0 <= safe <= 1.0:
        raise ValueError(
            f"get_max_safe_bet({_PROBE_BANKROLL}) must return a bankroll "
            f"fraction between 0 and 1, got {safe!r}"
        )
    if float(strategy.get_max_safe_bet(0.0)) != 0.0:
        raise ValueError(
            "get_max_safe_bet(0.0) must return 0.0 - a nonpositive bankroll "
            "has no safe stake"
        )

    for hook in ("update_bankroll", "record_settlement"):
        resolved = getattr(strategy, hook, None)
        if resolved is not None and not callable(resolved):
            raise ValueError(
                f"{hook} must be callable when defined - the simulators "
                "resolve it getattr-style and fire it around each settled "
                f"bet, got {resolved!r}"
            )


def check_allocation_strategy(strategy: BaseAllocationStrategy) -> None:
    """
    Verify an allocator against the documented weight contract.

    Runs the contract probes: ``evaluate`` at a positive bankroll returns a
    valid long-only weight vector (one finite weight per option, each within
    ``[0, 1]``, summing to no more than one within
    ``PROBABILITY_SUM_TOLERANCE``); a nonpositive bankroll returns all zeros
    - there is nothing left to allocate - with one weight per option at
    every bankroll.

    Parameters
    ----------
    strategy : BaseAllocationStrategy
        The allocation strategy to check.

    Raises
    ------
    TypeError
        If ``strategy`` is not a
        :class:`keeks.allocation.BaseAllocationStrategy` subclass.
    ValueError
        If any probe violates the contract.

    Examples
    --------
    >>> from keeks import BaseAllocationStrategy, check_allocation_strategy
    >>> class _OverBudget(BaseAllocationStrategy):
    ...     def evaluate(self, current_bankroll):
    ...         return (0.6, 0.6)
    >>> check_allocation_strategy(_OverBudget())
    Traceback (most recent call last):
        ...
    ValueError: Strategy weights must sum to no more than one; got (0.6, 0.6)
    """
    if not isinstance(strategy, BaseAllocationStrategy):
        raise TypeError(
            "check_allocation_strategy expects a keeks allocation strategy "
            f"(a BaseAllocationStrategy subclass), got {type(strategy).__name__}"
        )

    # Defense in depth: the base class wraps every concrete evaluate with
    # this same validator, so a violation raises at the evaluate call itself.
    weights = _validate_weights(strategy.evaluate(_PROBE_BANKROLL))

    for bankroll in (0.0, -_PROBE_BANKROLL):
        zeros = _validate_weights(strategy.evaluate(bankroll))
        if any(weight != 0.0 for weight in zeros):
            raise ValueError(
                f"evaluate({bankroll}) must return all zeros - a nonpositive "
                f"bankroll has nothing left to allocate, got {zeros!r}"
            )
        if len(zeros) != len(weights):
            raise ValueError(
                "evaluate must return one weight per option at every "
                f"bankroll: evaluate({_PROBE_BANKROLL}) returned "
                f"{len(weights)}, evaluate({bankroll}) returned {len(zeros)}"
            )


def check_model(model: JointReturnModel) -> None:
    """
    Verify a joint-return model against the documented sampling contract.

    Runs the contract probes: ``sample(n_samples, rng)`` returns a finite
    ``(n_samples, N)`` matrix of joint simple returns with ``N >= 1``; the
    same generator state produces the same draws (the house reproducibility
    contract); every sample count carries one column per option; and
    ``moments()``, when the model knows them, returns a shape ``(N,)`` mean
    and ``(N, N)`` covariance - ``None`` is the documented answer for a
    model without closed-form moments.

    Parameters
    ----------
    model : JointReturnModel
        The joint-return model to check.

    Raises
    ------
    TypeError
        If ``model`` is not a :class:`keeks.allocation.models.JointReturnModel`
        subclass.
    ValueError
        If any probe violates the contract.

    Examples
    --------
    >>> from keeks import binary_bets_model, check_model
    >>> check_model(binary_bets_model([(0.55, 2.0, 1.0), (0.30, 2.5, 1.0)]))

    A model that ignores its generator breaks reproducibility:

    >>> import numpy as np
    >>> from keeks.allocation.models import JointReturnModel
    >>> class _DriftingModel(JointReturnModel):
    ...     def sample(self, n_samples, rng):
    ...         return np.random.default_rng().uniform(size=(n_samples, 2))
    >>> check_model(_DriftingModel())
    Traceback (most recent call last):
        ...
    ValueError: sample must be deterministic given a generator's state - the same generator state must produce the same draws (the house reproducibility contract), but two identically-seeded generators diverged
    """
    if not isinstance(model, JointReturnModel):
        raise TypeError(
            "check_model expects a keeks joint-return model (a "
            f"JointReturnModel subclass), got {type(model).__name__}"
        )

    first = np.asarray(model.sample(8, np.random.default_rng(0)))
    if first.ndim != 2 or first.shape[0] != 8:
        raise ValueError(
            "sample(8, rng) must return an (8, N) matrix of joint simple "
            f"returns, got shape {first.shape}"
        )
    option_count = first.shape[1]
    if option_count < 1:
        raise ValueError(
            f"sample must return at least one option column, got {first.shape}"
        )
    if not np.all(np.isfinite(first)):
        raise ValueError("sample must return finite simple returns")

    replay = np.asarray(model.sample(8, np.random.default_rng(0)))
    if not np.array_equal(first, replay):
        raise ValueError(
            "sample must be deterministic given a generator's state - the "
            "same generator state must produce the same draws (the house "
            "reproducibility contract), but two identically-seeded "
            "generators diverged"
        )

    second = np.asarray(model.sample(3, np.random.default_rng(1)))
    if second.shape != (3, option_count):
        raise ValueError(
            f"sample(3, rng) must return shape (3, {option_count}) - one "
            f"column per option at every sample count, got {second.shape}"
        )

    moments = model.moments()
    if moments is None:
        return
    try:
        mean, covariance = moments
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "moments() must return a (mean, covariance) pair when the model "
            f"knows them, got {moments!r}"
        ) from exc
    expected_shapes = ((option_count,), (option_count, option_count))
    if (
        tuple(np.asarray(mean).shape) != expected_shapes[0]
        or tuple(np.asarray(covariance).shape) != expected_shapes[1]
    ):
        raise ValueError(
            f"moments() must return a shape ({option_count},) mean and a "
            f"({option_count}, {option_count}) covariance when it knows "
            f"them, got shapes {np.asarray(mean).shape} and "
            f"{np.asarray(covariance).shape}"
        )
