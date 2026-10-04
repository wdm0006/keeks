"""
Moment-based allocation methods: the mean-variance family.

The static optimizers here size a portfolio from a ``(mean, covariance)``
descriptor - or, for the covariance-only members, from the covariance alone -
and return one long-only weight per option through the shared
:class:`keeks.allocation.base.BaseAllocationStrategy` contract: weights in
``[0, 1]`` summing to no more than one, the residual held as cash at zero
return, and all zeros for a nonpositive bankroll.

The family:

- :class:`MeanVariance` - maximize ``w'mu - (l / 2) w'Sigma w``. At unit risk
  aversion the objective is the second-order approximation of expected log
  growth, ``E[log(1 + w'R)] ~ w'mu - w'Sigma w / 2``, so the optimum is the
  second-order Kelly allocation - the bridge between keeks' bet sizing and
  portfolio allocation.
- :class:`GlobalMinimumVariance` - minimize ``w'Sigma w``; no expected
  returns needed.
- :class:`MaximumSharpe` - the tangency portfolio, through the
  Cornuejols-Tutuncu convex reformulation.
- :class:`MaximumDiversification` - maximize the diversification ratio
  ``w'sigma / sqrt(w'Sigma w)``, through the same reformulation.
- :class:`RiskBudgeting` - the Spinu convex risk-budgeting objective, solved
  by cyclical coordinate descent; equal budgets give the equal risk
  contribution portfolio. Numpy-only, no scipy.
- :class:`RiskAversionScaling` - wrap any allocator and shrink its weights
  toward cash, ``w -> factor * w``: the fractional-Kelly dial (a ``kappa``
  fraction of the Kelly portfolio corresponds to CRRA risk aversion
  ``1 / kappa``).

The four quadratic programs are convex and solved with scipy's SLSQP behind
the ``keeks[allocation]`` optional extra (spec decision D3): constructing
any of them without scipy raises an ImportError that names the extra.
Everything else - validation, the risk-budgeting solver, the wrapper - is
numpy-only. Mean-variance methods assume finite second moments (a covariance
that exists); books with infinite-variance marginals are the territory of
the scenario-based methods, where sample-CVaR stays well defined where
variance does not.

Every allocator also accepts joint-return models through
:class:`keeks.allocation.models.ModelInputMixin`: ``from_model`` prefers the
model's exact ``moments()`` and falls back to Monte Carlo estimation
(:func:`keeks.allocation.models.estimate_moments`). The covariance-only
members drop the mean a model reports - it never enters their formulation.

Descriptors bind at construction and the solve runs there too, so a solver
pathology surfaces where the inputs do; an allocator reprices by fresh
construction, matching the multi-outcome convention.
"""

import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np

from keeks.allocation.base import (
    AllocationResult,
    BaseAllocationStrategy,
    _validate_covariance,
    _validate_mean,
    _validate_weights,
)
from keeks.allocation.models import (
    JointReturnModel,
    ModelInputMixin,
    _require_scipy,
    estimate_moments,
)
from keeks.utils import _require_finite

if TYPE_CHECKING:
    from scipy.optimize import OptimizeResult

__author__ = "willmcginnis"

# Solver discipline for the scipy-gated quadratic programs: a tight function
# tolerance (the programs are convex quadratics SLSQP solves exactly at
# convergence) and a generous iteration cap; a non-converged solve raises
# rather than shipping an approximate allocation.
_SLSQP_FUNCTION_TOLERANCE = 1e-12
_SLSQP_MAX_ITERATIONS = 1000

# Cyclical coordinate descent for the Spinu risk-budgeting objective: sweep
# until the largest weight move in a full sweep is this small relative to the
# weights' own scale, or give up loudly.
_CCD_TOLERANCE = 1e-12
_CCD_MAX_SWEEPS = 10_000


def _scale_free_weights(
    weights: np.ndarray, option_count: int, current_bankroll: float
) -> tuple[float, ...]:
    """
    The house scale-free ``evaluate``: the weights at any positive bankroll,
    all zeros at zero or below.

    Parameters
    ----------
    weights : numpy.ndarray
        The bankroll-independent weight vector the allocator solved for.
    option_count : int
        The number of options; the zero vector must match its length.
    current_bankroll : float
        The bankroll the strategy is evaluated at.

    Returns
    -------
    tuple of float
        The validated weight tuple, one weight per option.
    """
    if current_bankroll <= 0:
        return _validate_weights([0.0] * option_count, option_count=option_count)
    return _validate_weights(weights, option_count=option_count)


def _validated_solver_weights(
    raw: np.ndarray, option_count: int, upper: float = 1.0
) -> np.ndarray:
    """
    Clean a solver's raw weight vector and enforce the house weight contract.

    Clips at zero (SLSQP satisfies bounds only to solver tolerance, so a
    weight can come back ``-1e-17``), absorbs feasibility slack - a total
    above one by floating-point noise scales into the budget - and then runs
    the loud validator: solver output that cannot pass the same gate a
    hand-written vector would is a bug, not a rounding note.

    Parameters
    ----------
    raw : array-like
        The solver's raw solution vector.
    option_count : int
        The number of options the vector must carry.
    upper : float or None, default=1.0
        The per-coordinate upper bound to clip at. The simplex programs use
        ``1.0``; unbounded reformulation solutions are normalized separately
        by :func:`_normalized_direction_weights`.

    Returns
    -------
    numpy.ndarray
        The cleaned, validated weights.

    Raises
    ------
    ValueError
        If the cleaned vector still violates the weight contract (a
        non-finite solution, for instance).
    """
    weights = np.clip(np.asarray(raw, dtype=float), 0.0, upper)
    total = float(weights.sum())
    if total > 1.0:
        weights = weights / total
    return np.asarray(_validate_weights(weights, option_count=option_count))


def _normalized_direction_weights(raw: np.ndarray, option_count: int) -> np.ndarray:
    """
    Read a reformulation's unbounded solution as long-only weights.

    The Cornuejols-Tutuncu substitution's optimum ``y`` is a direction, not
    a budget: clip at zero and normalize by its strictly positive sum (the
    equality constraint guarantees one) to land the fully invested
    representative of the same ratio.

    Parameters
    ----------
    raw : array-like
        The reformulation's raw solution vector.
    option_count : int
        The number of options the vector must carry.

    Returns
    -------
    numpy.ndarray
        The normalized, validated weights, summing to one.
    """
    solution = np.clip(np.asarray(raw, dtype=float), 0.0, None)
    total = float(solution.sum())
    return np.asarray(_validate_weights(solution / total, option_count=option_count))


def _model_covariance(
    model: JointReturnModel, n_samples: int, seed: int | None
) -> np.ndarray:
    """
    The covariance a joint-return model implies, dropping its mean.

    Exact when the model knows its moments, a Monte Carlo estimate otherwise
    - the same extraction :meth:`ModelInputMixin.from_model` performs, minus
    the mean, which covariance-only allocators never consume.

    Parameters
    ----------
    model : JointReturnModel
        Any object implementing the sampling contract, optionally with an
        exact ``moments()`` hook.
    n_samples : int
        The draw count for the estimation fallback.
    seed : int or None
        Seed for the estimation fallback's private stream.

    Returns
    -------
    numpy.ndarray
        The (exact or estimated) covariance matrix.

    Examples
    --------
    >>> from keeks.allocation.models import scenario_model
    >>> model = scenario_model([[0.01, 0.02], [-0.01, 0.0]])
    >>> np.round(_model_covariance(model, 8, seed=0), 12).tolist()
    [[0.0001, 0.0001], [0.0001, 0.0001]]
    """
    moments = getattr(model, "moments", None)
    exact = moments() if callable(moments) else None
    if exact is None:
        exact = estimate_moments(model, n_samples, seed=seed)
    return exact[1]


class _CovarianceOnlyModelInput(ModelInputMixin):
    """
    :class:`ModelInputMixin` variant for covariance-only allocators.

    The shared ``from_model`` constructor builds ``(mean, covariance)``
    descriptors and passes both; these allocators' formulations consume the
    covariance alone, so their ``from_model`` drops the mean and constructs
    with the covariance positional.
    """

    @classmethod
    def from_model(
        cls,
        model: JointReturnModel,
        n_samples: int = 10_000,
        seed: int | None = None,
        **kwargs,
    ) -> BaseAllocationStrategy:
        """
        Build ``cls`` from a joint-return model's covariance.

        Exact model moments win when the model carries them; ``n_samples``
        and ``seed`` are only consumed on the estimation fallback. Extra
        keyword arguments pass through to the constructor.

        Parameters
        ----------
        model : JointReturnModel
            The model to build the covariance from.
        n_samples : int, default=10000
            The draw count for the estimation fallback.
        seed : int, optional
            Seed for the estimation fallback's private stream.
        **kwargs
            Extra constructor keyword arguments for ``cls``.

        Returns
        -------
        BaseAllocationStrategy
            The allocator built from the model's covariance.
        """
        covariance = _model_covariance(model, n_samples, seed)
        return cls(covariance, **kwargs)


def _run_slsqp(
    objective: Callable[[np.ndarray], float],
    gradient: Callable[[np.ndarray], np.ndarray],
    initial: np.ndarray,
    constraints: Sequence[dict],
    label: str,
    unbounded: bool,
) -> "OptimizeResult":
    """
    Run SLSQP with analytic gradients under the house solver discipline.

    The function tolerance is tight and a non-converged solve is a
    RuntimeError, never an approximate allocation: a vector that looks like
    weights but is not the optimum is worse than a loud failure.

    Parameters
    ----------
    objective : callable
        The scalar objective, minimized.
    gradient : callable
        The objective's analytic gradient.
    initial : numpy.ndarray
        A feasible starting point.
    constraints : sequence of dict
        SciPy constraint specifications.
    label : str
        The solver's owner, named in the failure message.
    unbounded : bool
        Whether coordinates may exceed one (the reformulated programs' ``y``
        unbounded above); otherwise coordinates are bounded at one.

    Returns
    -------
    scipy.optimize.OptimizeResult
        The converged result.

    Raises
    ------
    RuntimeError
        When SLSQP does not converge.
    """
    import scipy.optimize

    upper = None if unbounded else 1.0
    result = scipy.optimize.minimize(
        objective,
        initial,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, upper)] * initial.size,
        constraints=list(constraints),
        options={
            "maxiter": _SLSQP_MAX_ITERATIONS,
            "ftol": _SLSQP_FUNCTION_TOLERANCE,
        },
    )
    if not result.success:
        raise RuntimeError(f"The {label} solver did not converge: {result.message}")
    return result


def _solve_simplex_qp(
    matrix: np.ndarray,
    linear: np.ndarray,
    label: str,
    fully_invested: bool = False,
) -> "OptimizeResult":
    """
    Minimize ``0.5 * w' matrix w - linear' w`` over the long-only budget
    simplex ``{w : w >= 0, sum(w) <= 1}``, or - with ``fully_invested`` -
    over the invested simplex ``{w : w >= 0, sum(w) = 1}``.

    Every long-only quadratic program in this module - mean-variance and
    global minimum variance alike - is this problem with different
    coefficients, solved from the uniform feasible start with analytic
    gradients. The invested variant exists for variance-only objectives:
    with a cash option available, minimizing variance over the budget
    simplex just holds all cash (the origin), so those programs pin the
    budget with an equality instead.

    Parameters
    ----------
    matrix : numpy.ndarray
        The quadratic coefficient matrix (symmetric positive semidefinite).
    linear : numpy.ndarray
        The linear coefficient vector.
    label : str
        The solver's owner, named in the failure message.
    fully_invested : bool, default=False
        Replace the budget inequality with the equality ``sum(w) = 1``.

    Returns
    -------
    scipy.optimize.OptimizeResult
        The converged result.
    """

    def objective(weights):
        return 0.5 * float(weights @ matrix @ weights) - float(linear @ weights)

    def gradient(weights):
        return matrix @ weights - linear

    constraints = [
        {
            "type": "ineq",
            "fun": lambda weights: 1.0 - weights.sum(),
            "jac": lambda weights: -np.ones(weights.size),
        }
    ]
    if fully_invested:
        constraints = [
            {
                "type": "eq",
                "fun": lambda weights: weights.sum() - 1.0,
                "jac": lambda weights: np.ones(weights.size),
            }
        ]

    return _run_slsqp(
        objective,
        gradient,
        np.full(matrix.shape[0], 1.0 / matrix.shape[0]),
        tuple(constraints),
        label,
        unbounded=False,
    )


def _solve_ratio_qp(
    matrix: np.ndarray, direction: np.ndarray, label: str
) -> "OptimizeResult":
    """
    Maximize ``direction'w / sqrt(w' matrix w)`` over the budget simplex.

    The Cornuejols-Tutuncu substitution: a ratio that is invariant to
    scaling the portfolio (``direction'w`` positive somewhere, so only such
    portfolios can be optimal) turns into the convex quadratic program
    ``min y'matrix y`` subject to ``direction'y = 1`` and ``y >= 0`` via
    ``y = w / (direction'w)``. The optimum reads back as ``w = y / sum(y)``,
    and the maximized ratio is ``1 / sqrt(y'matrix y)``; the caller
    normalizes and reads the ratio off the weights.

    Parameters
    ----------
    matrix : numpy.ndarray
        The quadratic coefficient matrix (symmetric positive semidefinite).
    direction : numpy.ndarray
        The ratio's numerator vector, with at least one strictly positive
        entry - the substitution needs a positive excess return or a
        positive volatility to normalize by.
    label : str
        The solver's owner, named in the failure message.

    Returns
    -------
    scipy.optimize.OptimizeResult
        The converged result.
    """

    def objective(y):
        return float(y @ matrix @ y)

    def gradient(y):
        return 2.0 * (matrix @ y)

    # A feasible equality start: everything on the direction's largest
    # entry, scaled to satisfy the constraint exactly.
    pivot = int(np.argmax(direction))
    initial = np.zeros(direction.size)
    initial[pivot] = 1.0 / direction[pivot]
    return _run_slsqp(
        objective,
        gradient,
        initial,
        (
            {
                "type": "eq",
                "fun": lambda y: float(direction @ y) - 1.0,
                "jac": lambda _y: direction,
            },
        ),
        label,
        unbounded=True,
    )


def _risk_budget_weights(
    covariance: np.ndarray, budgets: np.ndarray
) -> tuple[np.ndarray, int]:
    """
    Solve the Spinu risk-budgeting objective by cyclical coordinate descent.

    Minimizing ``0.5 x'Sigma x - sum_i b_i log x_i`` over strictly positive
    ``x`` is the convex program whose first-order condition ties each
    option's risk contribution to its budget: ``x_i (Sigma x)_i = b_i``.
    One coordinate update solves the per-coordinate minimization exactly,

        ``x_i <- (-s_i + sqrt(s_i^2 + 4 * Sigma_ii * b_i)) / (2 * Sigma_ii)``

    with ``s_i`` the off-diagonal part of ``(Sigma x)_i``, so a sweep is N
    closed-form updates in place; sweeps repeat until the largest move is
    within :data:`_CCD_TOLERANCE` of the weights' own scale. The returned
    ``x`` is normalized once, by the caller, into weights.

    Parameters
    ----------
    covariance : numpy.ndarray
        The validated covariance matrix, strictly positive variances.
    budgets : numpy.ndarray
        The strictly positive risk budgets, one per option.

    Returns
    -------
    tuple of (numpy.ndarray, int)
        The optimal solution ``x`` (positive, satisfying
        ``x_i (Sigma x)_i = b_i``) and the number of sweeps it took.

    Raises
    ------
    RuntimeError
        When the sweeps do not converge.
    """
    variances = np.diag(covariance)
    # The diagonal program's exact solution - each coordinate alone would
    # sit at sqrt(b_i / Sigma_ii) - is the warm start.
    solution = np.sqrt(budgets / variances)
    converged = False
    sweeps = 0
    while not converged and sweeps < _CCD_MAX_SWEEPS:
        sweeps += 1
        previous = solution.copy()
        for index in range(solution.size):
            off_diagonal = (
                float(covariance[index] @ solution) - variances[index] * solution[index]
            )
            solution[index] = (
                -off_diagonal
                + math.sqrt(
                    off_diagonal * off_diagonal
                    + 4.0 * variances[index] * budgets[index]
                )
            ) / (2.0 * variances[index])
        change = float(np.max(np.abs(solution - previous)))
        if change <= _CCD_TOLERANCE * max(1.0, float(np.max(np.abs(previous)))):
            converged = True
            break
    if not converged:
        raise RuntimeError(
            f"The risk-budgeting coordinate descent did not converge within "
            f"{_CCD_MAX_SWEEPS} sweeps"
        )
    return solution, sweeps


class MeanVariance(ModelInputMixin, BaseAllocationStrategy):
    """
    Long-only mean-variance allocation: maximize ``w'mu - (l / 2) w'Sigma w``.

    The classic Markowitz problem over the long-only budget simplex: every
    weight in ``[0, 1]``, the weights summing to no more than one, the
    residual held as cash at zero return. The risk aversion ``l`` prices
    variance against expected return; at ``l = 1`` the objective is the
    second-order approximation of expected log growth, so the optimum is
    the second-order Kelly allocation - :meth:`optimize` reports the
    weights' implied growth as ``expected_growth``, and at unit risk
    aversion it is the maximized objective itself. Larger lambdas shrink the
    optimum toward all cash - with a cash option available, zero exposure
    minimizes variance - so :class:`RiskAversionScaling` is the post-hoc way
    to dial any solved allocation toward cash while the risk aversion itself
    prices the growth-variance trade.

    The quadratic program is convex and solved with scipy's SLSQP behind
    the ``keeks[allocation]`` optional extra. Mean-variance methods assume
    finite second moments - a covariance that exists - so books with
    infinite-variance marginals belong to the scenario-based methods.

    Weights are scale-free: they ignore the bankroll level and are all
    zeros for a nonpositive bankroll.

    Parameters
    ----------
    mean : array-like
        The expected simple return of each option.
    covariance : array-like
        The covariance matrix of the options' simple returns, matching the
        mean's length.
    risk_aversion : float, default=1.0
        The positive variance price ``l`` in the objective above.

    Raises
    ------
    ImportError
        When scipy is not installed; install the ``keeks[allocation]``
        extra.
    ValueError
        If the mean, the covariance, or the risk aversion is invalid.

    Examples
    --------
    One option with expected return 2% and variance 4%: the unit
    risk-aversion optimum is ``mean / variance = 0.5``, the rest cash:

    >>> strategy = MeanVariance([0.02], [[0.04]], risk_aversion=1.0)
    >>> [round(weight, 4) for weight in strategy.evaluate(1000.0)]
    [0.5]
    >>> strategy.evaluate(0.0)
    (0.0,)

    And keeks-native binary bets enter through the model door - the exact
    closed-form moments of a thin-edge even-money bet give the second-order
    Kelly stake, near Thorp's ``f* = (2p - 1) / a`` with ``a`` the net win
    per unit staked (at odds 2.0, ``a = 1``):

    >>> from keeks.allocation.models import binary_bets_model
    >>> model = binary_bets_model([(0.505, 2.0, 1.0)])
    >>> kelly = MeanVariance.from_model(model)
    >>> [round(weight, 4) for weight in kelly.evaluate(1000.0)]
    [0.01]
    """

    def __init__(
        self,
        mean: np.typing.ArrayLike,
        covariance: np.typing.ArrayLike,
        risk_aversion: float = 1.0,
    ) -> None:
        _require_scipy("MeanVariance")
        self.mean = _validate_mean(mean)
        self.covariance = _validate_covariance(covariance, option_count=self.mean.size)
        risk_aversion = _require_finite(risk_aversion, "Risk aversion")
        if risk_aversion <= 0:
            raise ValueError("Risk aversion must be positive")
        self.risk_aversion = risk_aversion
        # Minimizing the negated utility is the same problem, and dividing
        # the whole objective by the risk aversion (the same argmin) keeps
        # the solver's numbers on the covariance's scale for every lambda.
        result = _solve_simplex_qp(
            self.covariance, self.mean / self.risk_aversion, "MeanVariance"
        )
        self.weights = _validated_solver_weights(result.x, self.covariance.shape[0])
        self.converged = True
        self.iterations = int(result.nit)

    def optimize(self) -> AllocationResult:
        """
        Return the allocation with its solver diagnostics.

        ``objective`` is the maximized utility ``w'mu - (l / 2) w'Sigma w``
        at the returned weights; ``expected_growth`` is the second-order
        estimate ``w'mu - w'Sigma w / 2`` of ``E[log(1 + w'R)]`` - the same
        number at unit risk aversion, the Kelly bridge; ``volatility`` is
        the portfolio standard deviation ``sqrt(w'Sigma w)``.

        Returns
        -------
        AllocationResult
            The optimal weights and the diagnostics above.

        Examples
        --------
        The unconstrained optimum of this two-option book sits inside the
        simplex (it holds about 68% cash), which is the budget constraint
        doing its job:

        >>> result = MeanVariance(
        ...     [0.004, 0.005], [[0.02, 0.004], [0.004, 0.03]]
        ... ).optimize()
        >>> result.converged
        True
        >>> [round(weight, 3) for weight in result.weights.tolist()]
        [0.171, 0.144]
        """
        linear = float(self.weights @ self.mean)
        quadratic = float(self.weights @ self.covariance @ self.weights)
        return AllocationResult(
            weights=self.weights.copy(),
            objective=linear - 0.5 * self.risk_aversion * quadratic,
            converged=self.converged,
            iterations=self.iterations,
            expected_growth=linear - 0.5 * quadratic,
            volatility=float(np.sqrt(quadratic)),
        )

    def evaluate(self, current_bankroll: float) -> tuple[float, ...]:
        """
        Return one long-only weight per option.

        Parameters
        ----------
        current_bankroll : float
            The current bankroll. Weights are scale-free fractions: the
            same vector at every positive bankroll, all zeros at zero or
            below.

        Returns
        -------
        tuple of float
            The validated weight tuple, one weight per option.
        """
        return _scale_free_weights(
            self.weights, self.covariance.shape[0], current_bankroll
        )


class GlobalMinimumVariance(_CovarianceOnlyModelInput, BaseAllocationStrategy):
    """
    The long-only global minimum variance portfolio.

    Minimizes ``w'Sigma w`` over the fully invested simplex ``{w : w >= 0,
    sum(w) = 1}`` - no expected returns, the one Markowitz corner that
    ignores ``mu`` entirely, which makes it the reference point when
    expected returns are untrustworthy. The equality is the formulation,
    not a restriction: under keeks' budget contract a variance-only
    objective with a cash option available would simply hold all cash, so
    the classical fully invested anchor portfolio is the object worth
    naming. The unconstrained closed form ``w = Sigma^-1 1 / (1' Sigma^-1
    1)`` solves the problem when it happens to land inside the simplex;
    the convex program handles the general long-only case.

    Solved with scipy's SLSQP behind the ``keeks[allocation]`` extra.

    Weights are scale-free: they ignore the bankroll level and are all
    zeros for a nonpositive bankroll.

    Parameters
    ----------
    covariance : array-like
        The covariance matrix of the options' simple returns.

    Raises
    ------
    ImportError
        When scipy is not installed; install the ``keeks[allocation]``
        extra.
    ValueError
        If the covariance is invalid.

    Examples
    --------
    Two uncorrelated options with variances 4% and 1%: inverse-variance
    weights, one fifth and four fifths:

    >>> result = GlobalMinimumVariance([[0.04, 0.0], [0.0, 0.01]]).optimize()
    >>> [round(weight, 4) for weight in result.weights.tolist()]
    [0.2, 0.8]
    >>> round(result.objective, 6)
    0.008
    """

    def __init__(self, covariance: np.typing.ArrayLike) -> None:
        _require_scipy("GlobalMinimumVariance")
        self.covariance = _validate_covariance(covariance)
        option_count = self.covariance.shape[0]
        result = _solve_simplex_qp(
            2.0 * self.covariance,
            np.zeros(option_count),
            "GlobalMinimumVariance",
            fully_invested=True,
        )
        self.weights = _validated_solver_weights(result.x, option_count)
        self.converged = True
        self.iterations = int(result.nit)

    def optimize(self) -> AllocationResult:
        """
        Return the allocation with its solver diagnostics.

        ``objective`` is the minimized portfolio variance ``w'Sigma w``,
        and ``volatility`` its square root; there is no expected growth -
        the formulation never sees a mean.

        Returns
        -------
        AllocationResult
            The optimal weights and the diagnostics above.

        Examples
        --------
        >>> result = GlobalMinimumVariance([[0.04, 0.0], [0.0, 0.01]]).optimize()
        >>> round(result.volatility, 4)
        0.0894
        """
        quadratic = float(self.weights @ self.covariance @ self.weights)
        return AllocationResult(
            weights=self.weights.copy(),
            objective=quadratic,
            converged=self.converged,
            iterations=self.iterations,
            volatility=math.sqrt(quadratic),
        )

    def evaluate(self, current_bankroll: float) -> tuple[float, ...]:
        """
        Return one long-only weight per option.

        Parameters
        ----------
        current_bankroll : float
            The current bankroll. Weights are scale-free fractions: the
            same vector at every positive bankroll, all zeros at zero or
            below.

        Returns
        -------
        tuple of float
            The validated weight tuple, one weight per option.
        """
        return _scale_free_weights(
            self.weights, self.covariance.shape[0], current_bankroll
        )


class MaximumSharpe(ModelInputMixin, BaseAllocationStrategy):
    """
    The tangency portfolio: maximize the Sharpe ratio over the budget simplex.

    The Sharpe ratio ``(w'mu - r) / sqrt(w'Sigma w)`` is invariant to
    scaling a portfolio (cash at the risk-free rate scales both numerator
    and denominator), so the budget constraint never binds and the answer
    is a direction. The Cornuejols-Tutuncu reformulation turns "maximize
    the ratio" into the convex quadratic ``min y'Sigma y`` subject to
    ``(mu - r)'y = 1`` and ``y >= 0`` through ``y = w / ((mu - r)'w)``, and
    the optimum reads back normalized to the fully invested representative
    ``w = y / sum(y)``. The reformulation needs at least one option with
    expected return above the risk-free rate - with none, cash dominates
    every option and no tangency exists.

    Solved with scipy's SLSQP behind the ``keeks[allocation]`` extra.
    Mean-variance methods assume finite second moments; see
    :class:`MeanVariance` for the note.

    Weights are scale-free: they ignore the bankroll level and are all
    zeros for a nonpositive bankroll.

    Parameters
    ----------
    mean : array-like
        The expected simple return of each option.
    covariance : array-like
        The covariance matrix of the options' simple returns, matching the
        mean's length.
    risk_free : float, default=0.0
        The per-period risk-free (cash) rate the excess return is measured
        against.

    Raises
    ------
    ImportError
        When scipy is not installed; install the ``keeks[allocation]``
        extra.
    ValueError
        If the mean, the covariance, or the risk-free rate is invalid, or
        no option's expected return is above the risk-free rate.

    Examples
    --------
    Two uncorrelated equal-variance options with returns 1% and 2%: the
    tangency holds them in proportion to their excess returns, one third
    and two thirds:

    >>> result = MaximumSharpe(
    ...     [0.01, 0.02], [[0.01, 0.0], [0.0, 0.01]]
    ... ).optimize()
    >>> [round(weight, 4) for weight in result.weights.tolist()]
    [0.3333, 0.6667]
    >>> round(result.objective, 4)
    0.2236
    """

    def __init__(
        self,
        mean: np.typing.ArrayLike,
        covariance: np.typing.ArrayLike,
        risk_free: float = 0.0,
    ) -> None:
        _require_scipy("MaximumSharpe")
        self.mean = _validate_mean(mean)
        self.covariance = _validate_covariance(covariance, option_count=self.mean.size)
        self.risk_free = _require_finite(risk_free, "Risk-free rate")
        excess = self.mean - self.risk_free
        if float(np.max(excess)) <= 0.0:
            raise ValueError(
                "MaximumSharpe needs at least one option with expected return "
                "above the risk-free rate: cash dominates every option, and "
                "no tangency portfolio exists"
            )
        result = _solve_ratio_qp(self.covariance, excess, "MaximumSharpe")
        self.weights = _normalized_direction_weights(result.x, self.covariance.shape[0])
        self.converged = True
        self.iterations = int(result.nit)

    def optimize(self) -> AllocationResult:
        """
        Return the allocation with its solver diagnostics.

        ``objective`` is the maximized Sharpe ratio of the returned
        weights; ``expected_growth`` is the second-order estimate
        ``w'mu - w'Sigma w / 2`` of ``E[log(1 + w'R)]``; ``volatility`` is
        the portfolio standard deviation ``sqrt(w'Sigma w)``.

        Returns
        -------
        AllocationResult
            The optimal weights and the diagnostics above.

        Examples
        --------
        >>> result = MaximumSharpe(
        ...     [0.01, 0.02], [[0.01, 0.0], [0.0, 0.01]]
        ... ).optimize()
        >>> round(result.expected_growth, 4)
        0.0139
        """
        excess = float(self.weights @ (self.mean - self.risk_free))
        quadratic = float(self.weights @ self.covariance @ self.weights)
        linear = float(self.weights @ self.mean)
        return AllocationResult(
            weights=self.weights.copy(),
            objective=excess / math.sqrt(quadratic),
            converged=self.converged,
            iterations=self.iterations,
            expected_growth=linear - 0.5 * quadratic,
            volatility=float(np.sqrt(quadratic)),
        )

    def evaluate(self, current_bankroll: float) -> tuple[float, ...]:
        """
        Return one long-only weight per option.

        Parameters
        ----------
        current_bankroll : float
            The current bankroll. Weights are scale-free fractions: the
            same vector at every positive bankroll, all zeros at zero or
            below.

        Returns
        -------
        tuple of float
            The validated weight tuple, one weight per option.
        """
        return _scale_free_weights(
            self.weights, self.covariance.shape[0], current_bankroll
        )


class MaximumDiversification(_CovarianceOnlyModelInput, BaseAllocationStrategy):
    """
    Maximize the diversification ratio ``w'sigma / sqrt(w'Sigma w)``.

    Choueifaty-Coignard's ratio: the weighted average volatility over the
    portfolio volatility. The ratio is scale-invariant and the numerator is
    positive for any nonzero long-only ``w`` (volatilities are), so the
    same Cornuejols-Tutuncu substitution :class:`MaximumSharpe` uses
    applies with the volatilities in the direction's place - and like the
    tangency, the formulation never sees a mean. The answer is the fully
    invested representative of the most diversified direction.

    Solved with scipy's SLSQP behind the ``keeks[allocation]`` extra.

    Weights are scale-free: they ignore the bankroll level and are all
    zeros for a nonpositive bankroll.

    Parameters
    ----------
    covariance : array-like
        The covariance matrix of the options' simple returns; strictly
        positive variances, since the ratio divides by them.

    Raises
    ------
    ImportError
        When scipy is not installed; install the ``keeks[allocation]``
        extra.
    ValueError
        If the covariance is invalid or carries a zero variance.

    Examples
    --------
    Two correlated options with volatilities 20% and 10%: the most
    diversified long-only book holds a third and two thirds:

    >>> result = MaximumDiversification(
    ...     [[0.04, 0.004], [0.004, 0.01]]
    ... ).optimize()
    >>> [round(weight, 4) for weight in result.weights.tolist()]
    [0.3333, 0.6667]
    >>> round(result.objective, 3)
    1.291
    """

    def __init__(self, covariance: np.typing.ArrayLike) -> None:
        _require_scipy("MaximumDiversification")
        self.covariance = _validate_covariance(covariance)
        option_count = self.covariance.shape[0]
        if float(np.min(np.diag(self.covariance))) <= 0.0:
            raise ValueError(
                "Covariance must have strictly positive variances: the "
                "diversification ratio divides by each option's volatility"
            )
        volatilities = np.sqrt(np.diag(self.covariance))
        result = _solve_ratio_qp(
            self.covariance, volatilities, "MaximumDiversification"
        )
        self.weights = _normalized_direction_weights(result.x, option_count)
        self.converged = True
        self.iterations = int(result.nit)

    def optimize(self) -> AllocationResult:
        """
        Return the allocation with its solver diagnostics.

        ``objective`` is the maximized diversification ratio of the
        returned weights and ``volatility`` the portfolio standard
        deviation ``sqrt(w'Sigma w)``; there is no expected growth - the
        formulation never sees a mean.

        Returns
        -------
        AllocationResult
            The optimal weights and the diagnostics above.

        Examples
        --------
        >>> result = MaximumDiversification(
        ...     [[0.04, 0.004], [0.004, 0.01]]
        ... ).optimize()
        >>> round(result.volatility, 4)
        0.1033
        """
        weighted_volatility = float(self.weights @ np.sqrt(np.diag(self.covariance)))
        quadratic = float(self.weights @ self.covariance @ self.weights)
        return AllocationResult(
            weights=self.weights.copy(),
            objective=weighted_volatility / math.sqrt(quadratic),
            converged=self.converged,
            iterations=self.iterations,
            volatility=math.sqrt(quadratic),
        )

    def evaluate(self, current_bankroll: float) -> tuple[float, ...]:
        """
        Return one long-only weight per option.

        Parameters
        ----------
        current_bankroll : float
            The current bankroll. Weights are scale-free fractions: the
            same vector at every positive bankroll, all zeros at zero or
            below.

        Returns
        -------
        tuple of float
            The validated weight tuple, one weight per option.
        """
        return _scale_free_weights(
            self.weights, self.covariance.shape[0], current_bankroll
        )


class RiskBudgeting(_CovarianceOnlyModelInput, BaseAllocationStrategy):
    """
    Spinu's risk budgeting: give each option its targeted share of the risk.

    Minimizes the convex objective ``0.5 x'Sigma x - sum_i b_i log x_i``
    over strictly positive ``x``; the first-order condition
    ``x_i (Sigma x)_i = b_i`` ties every option's risk contribution
    ``w_i (Sigma w)_i`` to its budget ``b_i``. Normalized budgets are
    therefore the risk contribution shares, and equal budgets - the
    default - give the equal risk contribution portfolio. Budgets are
    scale-free: rescaling them rescales ``x`` but not the returned weights.

    Solved by cyclical coordinate descent - one closed-form coordinate
    update at a time, numpy-only, no scipy, deterministic. The solve runs
    at construction and normalizes once into weights summing to one.

    Weights are scale-free: they ignore the bankroll level and are all
    zeros for a nonpositive bankroll.

    Parameters
    ----------
    covariance : array-like
        The covariance matrix of the options' simple returns; strictly
        positive variances, since the coordinate updates divide by them.
    risk_budgets : array-like, optional
        The strictly positive risk budget of each option, scale-free.
        Defaults to equal budgets (the ERC portfolio).

    Raises
    ------
    ValueError
        If the covariance or the budgets are invalid.
    RuntimeError
        When the coordinate descent does not converge.

    Examples
    --------
    Two uncorrelated options with volatilities 20% and 10%: equal risk
    budgets weight the quieter option twice as heavy, one third and two
    thirds:

    >>> result = RiskBudgeting([[0.04, 0.0], [0.0, 0.01]]).optimize()
    >>> [round(weight, 4) for weight in result.weights.tolist()]
    [0.3333, 0.6667]

    And the budgets are the risk contribution shares - here 75/25:

    >>> budgeted = RiskBudgeting(
    ...     [[0.04, 0.0], [0.0, 0.01]], risk_budgets=[0.75, 0.25]
    ... )
    >>> [round(weight, 4) for weight in budgeted.weights.tolist()]
    [0.4641, 0.5359]
    """

    def __init__(
        self,
        covariance: np.typing.ArrayLike,
        risk_budgets: np.typing.ArrayLike | None = None,
    ) -> None:
        self.covariance = _validate_covariance(covariance)
        option_count = self.covariance.shape[0]
        variances = np.diag(self.covariance)
        if float(np.min(variances)) <= 0.0:
            raise ValueError(
                "Covariance must have strictly positive variances: the "
                "risk-budgeting coordinate updates divide by each option's "
                "variance"
            )
        if risk_budgets is None:
            budgets = np.full(option_count, 1.0 / option_count)
        else:
            try:
                budgets = np.asarray(risk_budgets, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError("Risk budgets must be a finite sequence") from exc
            if budgets.ndim != 1 or budgets.size != option_count:
                raise ValueError(
                    f"Risk budgets must carry exactly {option_count} entries, "
                    f"one per option, got shape {budgets.shape}"
                )
            if not np.all(np.isfinite(budgets)) or np.any(budgets <= 0):
                raise ValueError("Risk budgets must be positive finite numbers")
        self.risk_budgets = budgets
        solution, sweeps = _risk_budget_weights(self.covariance, budgets)
        self.weights = np.asarray(
            _validate_weights(solution / solution.sum(), option_count=option_count)
        )
        self.objective = 0.5 * float(solution @ self.covariance @ solution) - float(
            budgets @ np.log(solution)
        )
        self.converged = True
        self.iterations = sweeps

    def optimize(self) -> AllocationResult:
        """
        Return the allocation with its solver diagnostics.

        ``objective`` is the minimized Spinu objective
        ``0.5 x'Sigma x - sum_i b_i log x_i`` at the solver's solution
        (before the single normalization into weights), ``converged`` and
        ``iterations`` the coordinate descent's outcome and sweep count,
        and ``volatility`` the portfolio standard deviation
        ``sqrt(w'Sigma w)``. There is no expected growth - the formulation
        never sees a mean.

        Returns
        -------
        AllocationResult
            The optimal weights and the diagnostics above.

        Examples
        --------
        >>> result = RiskBudgeting([[0.04, 0.0], [0.0, 0.01]]).optimize()
        >>> result.converged, result.iterations > 0
        (True, True)
        """
        return AllocationResult(
            weights=self.weights.copy(),
            objective=self.objective,
            converged=self.converged,
            iterations=self.iterations,
            volatility=float(np.sqrt(self.weights @ self.covariance @ self.weights)),
        )

    def evaluate(self, current_bankroll: float) -> tuple[float, ...]:
        """
        Return one long-only weight per option.

        Parameters
        ----------
        current_bankroll : float
            The current bankroll. Weights are scale-free fractions: the
            same vector at every positive bankroll, all zeros at zero or
            below.

        Returns
        -------
        tuple of float
            The validated weight tuple, one weight per option.
        """
        return _scale_free_weights(
            self.weights, self.covariance.shape[0], current_bankroll
        )


class RiskAversionScaling(BaseAllocationStrategy):
    """
    Shrink any allocator's weights toward cash: ``w -> factor * w``.

    The fractional-Kelly dial for the whole layer: a ``factor`` of
    ``kappa`` stakes ``kappa`` of the wrapped allocator's every weight and
    holds the rest as cash - the fractional-Kelly construction (a
    ``kappa`` fraction of the Kelly portfolio corresponds to CRRA risk
    aversion ``1 / kappa``), applicable to any allocator the layer can
    produce without a new solver. For an unconstrained interior
    mean-variance optimum, shrinking by ``kappa`` is exactly re-solving
    with ``kappa`` times the risk aversion; once the budget or bounds
    bind, the shrunk portfolio is the honest "same portfolio, less of it"
    answer.

    The wrapped allocator stays in charge of its own state: settlements
    flow to it (an online allocator inside the wrapper keeps adapting),
    and ``evaluate`` is a pure read of the wrapped weights, scaled.

    Parameters
    ----------
    inner : BaseAllocationStrategy
        The allocator whose weights are scaled.
    factor : float, default=1.0
        The scale ``kappa`` in ``[0, 1]``: 1 leaves the allocator
        untouched, 0 is the all-cash portfolio.

    Raises
    ------
    ValueError
        If ``inner`` is not a :class:`BaseAllocationStrategy` or the
        factor is not a finite number in ``[0, 1]``.

    Examples
    --------
    Half-Kelly over the unit-risk-aversion mean-variance stake:

    >>> inner = MeanVariance([0.02], [[0.04]], risk_aversion=1.0)
    >>> half_kelly = RiskAversionScaling(inner, factor=0.5)
    >>> [round(weight, 4) for weight in half_kelly.evaluate(1000.0)]
    [0.25]
    >>> [round(weight, 4) for weight in half_kelly.optimize().weights.tolist()]
    [0.25]
    """

    def __init__(self, inner: BaseAllocationStrategy, factor: float = 1.0) -> None:
        if not isinstance(inner, BaseAllocationStrategy):
            raise ValueError("RiskAversionScaling must wrap a BaseAllocationStrategy")
        factor = _require_finite(factor, "Scaling factor")
        if not 0.0 <= factor <= 1.0:
            raise ValueError("Scaling factor must be between 0 and 1")
        self.inner = inner
        self.factor = factor

    def optimize(self) -> AllocationResult:
        """
        Return the shrunk allocation with the inner solver's diagnostics.

        The weights are the inner allocator's optimum scaled by the
        factor; ``converged`` and ``iterations`` are the inner solve's,
        and ``volatility`` scales exactly (``sqrt((k w)'Sigma (k w)) ==
        k sqrt(w'Sigma w)``). ``objective`` and ``expected_growth``
        describe the inner allocator's unshrunk problem, so they stay
        ``None`` here. An inner allocator without an ``optimize`` - the
        online family, whose state is its settlement history - returns
        the shrunk current weights with no diagnostics.

        Returns
        -------
        AllocationResult
            The scaled weights and whatever diagnostics transfer.
        """
        inner_optimize = getattr(self.inner, "optimize", None)
        if inner_optimize is None:
            # Weights are scale-free, so any positive bankroll reads the
            # online allocator's current weights.
            weights = np.asarray(
                [self.factor * weight for weight in self.inner.evaluate(1.0)],
                dtype=float,
            )
            return AllocationResult(weights=np.asarray(_validate_weights(weights)))
        result = inner_optimize()
        volatility = (
            self.factor * result.volatility if result.volatility is not None else None
        )
        return AllocationResult(
            weights=self.factor * result.weights,
            converged=result.converged,
            iterations=result.iterations,
            volatility=volatility,
        )

    def evaluate(self, current_bankroll: float) -> tuple[float, ...]:
        """
        Return one scaled long-only weight per option.

        Parameters
        ----------
        current_bankroll : float
            The current bankroll, passed through to the wrapped allocator.
            All zeros when it is nonpositive, like every keeks strategy.

        Returns
        -------
        tuple of float
            The validated weight tuple: the wrapped allocator's weights,
            each scaled by the factor.
        """
        inner_weights = self.inner.evaluate(current_bankroll)
        return _validate_weights([self.factor * weight for weight in inner_weights])
