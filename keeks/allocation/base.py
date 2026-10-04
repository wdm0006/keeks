import abc
import dataclasses

import numpy as np

from keeks.utils import PROBABILITY_SUM_TOLERANCE, normalize_probabilities

__author__ = "willmcginnis"

# Descriptor numeric discipline: a covariance may deviate from symmetry by at
# most this fraction of its own scale, and an eigenvalue may sit at most
# EIGENVALUE_FLOOR (scaled by the matrix's spectral magnitude, floored at an
# absolute 1.0 scale) below zero before the matrix counts as indefinite. The
# scaled floors keep legitimate small- or large-magnitude matrices from
# failing on floating-point noise while still rejecting genuinely indefinite
# inputs.
COVARIANCE_SYMMETRY_TOLERANCE = 1e-8
EIGENVALUE_FLOOR = -1e-10


class BaseAllocationStrategy(abc.ABC):
    """
    Abstract base class for portfolio allocation strategies.

    This class defines the interface that all allocation strategies must
    implement. An allocation strategy sizes a portfolio across N
    distribution-valued options - options described by a mean vector and
    covariance, a scenario matrix, or any other joint-return description - in
    a single decision: every option settles every period, and the weight
    vector decides how much of the bankroll rides on each option's simple
    return, with any unallocated residual held as cash at zero return.

    Weights are long-only budget fractions, not stake fractions: each weight
    is in ``[0, 1]`` and the weights sum to no more than one, so the portfolio
    never promises more of the bankroll than it holds. This is the one
    contract every static method family (mean-variance, minimum variance,
    risk budgeting, hierarchical risk parity, ...) and every online method
    shares.

    Descriptive inputs bind at construction and are immutable thereafter - an
    allocator reprices by fresh construction, matching the multi-outcome
    convention. Weights are scale-free: they ignore the bankroll level and
    are all zeros for a nonpositive bankroll, like every keeks strategy.

    Concrete strategy implementations should inherit from this class and
    implement the evaluate method. Simulators resolve two optional hooks
    ``getattr``-style: ``update_bankroll(current_bankroll)`` carries the
    bankroll path for plumbing, and online allocators additionally implement
    ``record_settlement(realized_returns)`` - called once per staked period
    with the realized joint simple-return vector - as their only sanctioned
    stateful channel.
    """

    @abc.abstractmethod
    def evaluate(self, current_bankroll: float) -> tuple[float, ...]:
        """
        Evaluate the strategy for the current bankroll.

        Every descriptive input - a mean vector and covariance, a scenario
        matrix, or whatever else the concrete strategy optimizes over - binds
        at construction, so the only call-time input is the bankroll.
        Implementations return one weight per option and validate the vector
        through :func:`_validate_weights`.

        Parameters
        ----------
        current_bankroll : float
            The current bankroll to use for calculations.

        Returns
        -------
        tuple of float
            One long-only weight per option: ``len(result) == N``, every
            element finite and within ``[0, 1]``, and
            ``sum(result) <= 1 + PROBABILITY_SUM_TOLERANCE``. The shortfall
            below one is cash held at zero return. All zeros when
            ``current_bankroll <= 0``: there is nothing left to allocate.

        Raises
        ------
        ValueError
            If the weight vector leaving this strategy violates the contract
            above.
        """
        pass


def _validate_weights(weights, option_count=None):
    """
    Coerce an allocator's weight vector to a tuple of finite floats in ``[0, 1]``.

    One long-only weight per option: every element finite in ``[0, 1]``, and
    the total at most one within ``PROBABILITY_SUM_TOLERANCE`` - the
    portfolio can never promise more of the bankroll than it holds, and the
    shortfall below one is cash held at zero return. This is the weight-space
    sibling of :func:`keeks.multi_outcome.base._validate_stake_fractions`.

    Parameters
    ----------
    weights : sequence of float
        One weight per option. Any sequence is accepted; a bare scalar is not
        a one-dimensional sequence and is rejected.
    option_count : int, optional
        When given, the vector must carry exactly that many weights - one per
        option of the portfolio it settles. A length mismatch would otherwise
        silently skip options or weight options that do not exist, so
        simulators pass their option count and reject the vector.

    Returns
    -------
    tuple of float
        The validated weights, one per option, in option order.

    Raises
    ------
    ValueError
        If the weights are not a non-empty one-dimensional sequence of finite
        numbers, if any element falls outside ``[0, 1]``, if they sum to more
        than ``1 + PROBABILITY_SUM_TOLERANCE``, or if ``option_count`` is
        given and the vector's length differs from it.

    Examples
    --------
    >>> _validate_weights([0.25, 0.25, 0.5])
    (0.25, 0.25, 0.5)
    """
    try:
        weights = np.asarray(weights, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Strategy weights must be a finite sequence") from exc
    if weights.ndim != 1:
        raise ValueError("Strategy weights must be one-dimensional")
    if weights.size == 0:
        raise ValueError("Strategy weights must be non-empty")
    if option_count is not None and weights.size != option_count:
        raise ValueError(
            f"Strategy must return exactly {option_count} weights, got {weights.size}"
        )
    if not np.all(np.isfinite(weights)):
        raise ValueError("Strategy weights must contain only finite values")
    if np.any((weights < 0) | (weights > 1)):
        raise ValueError("Strategy weights must be between 0 and 1")
    if weights.sum() > 1 + PROBABILITY_SUM_TOLERANCE:
        raise ValueError("Strategy weights must sum to no more than one")
    return tuple(weights.tolist())


def _validate_mean(mean):
    """
    Validate an expected simple-return vector and return it as a float array.

    Parameters
    ----------
    mean : array-like
        The expected simple return of each option. Must be a non-empty
        one-dimensional sequence of finite numbers.

    Returns
    -------
    numpy.ndarray
        The validated means as a one-dimensional float array.

    Raises
    ------
    ValueError
        If the means are not a non-empty one-dimensional sequence of finite
        numbers.

    Examples
    --------
    >>> _validate_mean([0.01, 0.02])
    array([0.01, 0.02])
    """
    try:
        mean = np.asarray(mean, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Mean must be a finite sequence") from exc
    if mean.ndim != 1:
        raise ValueError("Mean must be one-dimensional")
    if mean.size == 0:
        raise ValueError("Mean must be non-empty")
    if not np.all(np.isfinite(mean)):
        raise ValueError("Mean must contain only finite values")
    return mean


def _validate_covariance(covariance, option_count=None):
    """
    Validate a covariance matrix and return it as a float array.

    The matrix must be square, finite, symmetric within
    ``COVARIANCE_SYMMETRY_TOLERANCE`` of its own scale, and positive
    semidefinite down to ``EIGENVALUE_FLOOR`` of its spectral magnitude -
    floating-point noise passes, a genuinely indefinite matrix does not.
    Eigenvalues are read with :func:`numpy.linalg.eigvalsh`, which uses the
    lower triangle; the symmetry gate bounds how far that triangle can drift
    from the upper one before the eigenvalues stop describing the matrix.

    Parameters
    ----------
    covariance : array-like
        The covariance matrix of the options' simple returns. Must be a
        non-empty, square, two-dimensional sequence of finite numbers.
    option_count : int, optional
        When given, the matrix must be exactly ``option_count`` x
        ``option_count`` - one row and column per option of the portfolio it
        sizes. A shape mismatch would otherwise silently misalign means,
        weights, and covariance rows, so callers that know their option count
        pass it and reject the matrix.

    Returns
    -------
    numpy.ndarray
        The validated covariance as a two-dimensional float array.

    Raises
    ------
    ValueError
        If the matrix is not a non-empty square two-dimensional sequence of
        finite numbers, is materially asymmetric, is indefinite, or does not
        match ``option_count``.

    Examples
    --------
    >>> covariance = _validate_covariance([[0.01, 0.002], [0.002, 0.02]])
    >>> covariance.shape
    (2, 2)
    """
    try:
        covariance = np.asarray(covariance, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Covariance must be a finite sequence") from exc
    if covariance.ndim != 2:
        raise ValueError("Covariance must be two-dimensional")
    if covariance.size == 0:
        raise ValueError("Covariance must be non-empty")
    rows, cols = covariance.shape
    if rows != cols:
        raise ValueError("Covariance must be square")
    if option_count is not None and rows != option_count:
        raise ValueError(
            f"Covariance must be exactly {option_count}x{option_count}, "
            f"got shape {covariance.shape}"
        )
    if not np.all(np.isfinite(covariance)):
        raise ValueError("Covariance must contain only finite values")

    scale = max(1.0, float(np.max(np.abs(covariance))))
    asymmetry = float(np.max(np.abs(covariance - covariance.T)))
    if asymmetry > COVARIANCE_SYMMETRY_TOLERANCE * scale:
        raise ValueError("Covariance must be symmetric")

    eigenvalues = np.linalg.eigvalsh(covariance)
    floor = EIGENVALUE_FLOOR * max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(np.min(eigenvalues)) < floor:
        raise ValueError("Covariance must be positive semidefinite")
    return covariance


def _validate_scenarios(scenarios, probabilities=None):
    """
    Validate a scenario matrix (and optional row probabilities).

    Scenarios are joint simple-return observations: one row per observation,
    one column per option. Rows are equally likely unless ``probabilities``
    says otherwise, and any probability mass below one is left as-is - it
    models all-cash periods at zero return, so probabilities are validated
    through :func:`keeks.utils.normalize_probabilities` and never
    renormalized.

    Parameters
    ----------
    scenarios : array-like
        The (observations, options) matrix of joint simple returns. Must be a
        non-empty two-dimensional sequence of finite numbers.
    probabilities : array-like, optional
        The probability of each scenario row. Must satisfy
        :func:`keeks.utils.normalize_probabilities` and carry exactly one
        entry per row.

    Returns
    -------
    tuple of numpy.ndarray
        The validated scenarios as a two-dimensional float array, and the
        validated probabilities as a one-dimensional float array (or
        ``None`` when none were given).

    Raises
    ------
    ValueError
        If the scenarios are not a non-empty two-dimensional sequence of
        finite numbers, or if the probabilities are invalid or do not match
        the row count.
    """
    try:
        scenarios = np.asarray(scenarios, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Scenarios must be a finite sequence") from exc
    if scenarios.ndim != 2:
        raise ValueError("Scenarios must be two-dimensional")
    if scenarios.size == 0:
        raise ValueError("Scenarios must be non-empty")
    if not np.all(np.isfinite(scenarios)):
        raise ValueError("Scenarios must contain only finite values")

    if probabilities is None:
        return scenarios, None

    probabilities = normalize_probabilities(probabilities)
    if probabilities.size != scenarios.shape[0]:
        raise ValueError("Probabilities must have one entry per scenario row")
    return scenarios, probabilities


def _validate_strategy_scenarios(strategy, scenarios):
    """
    Reject an allocator whose scenario descriptor contradicts the simulator's.

    A scenario-bound allocator sizes every period through the scenarios it
    bound at construction, while a simulator settles draws from its own
    scenario matrix - so the two have to agree for the run to describe
    anything. This is the allocation analogue of
    :func:`keeks.utils._validate_strategy_odds`: the gate reads the
    strategy's ``scenarios`` descriptor when it exposes one and compares it
    to the simulator's. Allocators bound to other descriptors (means and
    covariances) carry no ``scenarios`` attribute - or carry ``None`` - and
    pass untouched, their compatibility staying the caller's responsibility.
    """
    bound = getattr(strategy, "scenarios", None)
    if bound is None:
        return

    try:
        bound = np.asarray(bound, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Strategy scenarios descriptor must be a finite sequence"
        ) from exc

    if not np.array_equal(bound, scenarios):
        raise ValueError(
            f"Strategy scenarios (shape {bound.shape}) do not match the "
            f"simulator's scenarios (shape {np.asarray(scenarios).shape}); "
            "the strategy sizes each period with its own scenarios while the "
            "simulator settles draws from the simulator's, so the two must "
            "agree."
        )


@dataclasses.dataclass(frozen=True, eq=False)
class AllocationResult:
    """
    Diagnostics from one allocation optimization.

    ``optimize()`` returns one of these alongside the simulator-facing
    ``evaluate`` tuple, so solver diagnostics are first-class rather than a
    side dict. Every field except ``weights`` is optional: an allocator fills
    the fields its formulation produces and leaves the rest ``None``.

    Comparison is identity-based (``eq=False``): ``weights`` is a
    ``numpy.ndarray``, so generated field-wise equality would raise on its
    ambiguous truth value. Compare the fields you care about instead.

    Attributes
    ----------
    weights : numpy.ndarray
        The optimal weight vector, shape ``(N,)``, one long-only weight per
        option.
    objective : float, optional
        The optimal objective value, when a solver ran.
    converged : bool, optional
        Solver convergence, when a solver ran.
    iterations : int, optional
        Solver iterations, when a solver ran.
    expected_growth : float, optional
        The estimated ``E[log(1 + w'R)]`` under the strategy's inputs.
    volatility : float, optional
        ``sqrt(w' Σ w)``, when a covariance is available.
    """

    weights: np.ndarray
    objective: float | None = None
    converged: bool | None = None
    iterations: int | None = None
    expected_growth: float | None = None
    volatility: float | None = None
