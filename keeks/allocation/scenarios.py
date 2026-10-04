"""
Mean-CVaR: the scenario-based tail-risk family.

:class:`MeanCVaR` sizes a portfolio against a scenario matrix - a sample of
joint simple-return vectors that can come from anywhere (empirical history,
:class:`keeks.allocation.models.JointReturnModel` draws, a user sampler) -
and that is the point: sample-CVaR is defined for any finite sample, so
books with fat-tailed marginals, mixture jumps, or no finite variance have
a working optimizer here, where the moment-based families' second-moment
assumption would already be meaningless.

The objective is the unit-risk-aversion mean-CVaR tradeoff: maximize the
scenarios' expected return minus the expected loss in the worst
``tail_alpha`` fraction of scenarios. Through the Rockafellar-Uryasev
auxiliary-variable trick it is exactly a linear program, solved with
``scipy.optimize.linprog`` under the HiGHS backend - which is why the class
gates on the ``keeks[allocation]`` optional extra.

Also here: :func:`scenarios_to_moments`, the bridge that turns a scenario
matrix (with optional row probabilities) into the ``(mean, covariance)``
descriptor the moment-based methods consume - the two halves of the layer
share one scenario vocabulary.
"""

import numpy as np

from keeks.allocation.base import (
    AllocationResult,
    BaseAllocationStrategy,
    _validate_scenarios,
    _validate_weights,
)
from keeks.allocation.models import (
    JointReturnModel,
    ModelInputMixin,
    ScenarioModel,
    _require_scipy,
    _validate_draws,
    _validate_n_samples,
)
from keeks.utils import PROBABILITY_SUM_TOLERANCE, _validate_simulator_seed

__author__ = "willmcginnis"


def _validate_tail_alpha(tail_alpha: float) -> float:
    """
    Validate the tail fraction, which must lie strictly between 0 and 1.

    The tail fraction is the share of scenarios whose average loss the
    optimizer minimizes: 0 would divide by zero in the Rockafellar-Uryasev
    objective and 1 would make the "tail" the whole distribution, so both
    ends are rejected along with anything non-finite.

    Parameters
    ----------
    tail_alpha : float
        The requested tail fraction.

    Returns
    -------
    float
        The validated tail fraction.

    Raises
    ------
    ValueError
        If the fraction is not a finite number strictly between 0 and 1.

    Examples
    --------
    >>> _validate_tail_alpha(0.05)
    0.05
    """
    try:
        tail_alpha = float(tail_alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Tail alpha must be a finite number strictly between 0 and 1"
        ) from exc
    if not np.isfinite(tail_alpha) or not 0.0 < tail_alpha < 1.0:
        raise ValueError("Tail alpha must be a finite number strictly between 0 and 1")
    return tail_alpha


def _mean_cvar_lp(
    scenarios: np.ndarray, tail_alpha: float
) -> tuple[np.ndarray, float, float, bool, int]:
    """
    Solve the unit-risk-aversion mean-CVaR program over ``scenarios``.

    The Rockafellar-Uryasev reformulation turns ``maximize w'mu - CVaR(w)``
    into the linear program

        minimize    alpha + (1 / (tail_alpha * T)) * sum_t u_t - w' mu
        subject to  u_t >= -r_t . w - alpha    (one tail surplus per scenario)
                    u_t >= 0
                    w >= 0,  sum(w) <= 1       (the house long-only budget)

    over the variables ``(w, alpha, u)``: ``r_t`` is scenario row ``t``,
    ``mu`` the scenarios' mean vector, and at the optimum ``alpha`` is the
    tail's loss threshold while ``u_t`` carries the loss above it - so the
    first objective block is exactly the tail's expected loss.

    Parameters
    ----------
    scenarios : numpy.ndarray
        The validated ``(T, N)`` scenario matrix.
    tail_alpha : float
        The validated tail fraction.

    Returns
    -------
    tuple
        ``(weights, cvar, objective, converged, iterations)``: the optimal
        long-only weight vector, the tail's expected loss under it, the
        minimized program value (``cvar - w' mu``), and the solver's
        convergence flag and iteration count.
    """
    import scipy.optimize
    import scipy.sparse

    observation_count, option_count = scenarios.shape
    mean = scenarios.mean(axis=0)

    # Variables: the N weights, the tail threshold alpha, and one tail
    # surplus per scenario.
    objective = np.concatenate(
        [
            -mean,
            [1.0],
            np.full(observation_count, 1.0 / (tail_alpha * observation_count)),
        ]
    )
    # u_t >= -r_t . w - alpha, rewritten as -r_t . w - alpha - u_t <= 0.
    tail_block = scipy.sparse.hstack(
        [
            scipy.sparse.csr_matrix(-scenarios),
            scipy.sparse.csr_matrix(np.full((observation_count, 1), -1.0)),
            -scipy.sparse.eye(observation_count, format="csr"),
        ],
        format="csr",
    )
    budget_block = scipy.sparse.csr_matrix(
        np.concatenate([np.ones(option_count), np.zeros(observation_count + 1)])[
            None, :
        ]
    )
    constraints = scipy.sparse.vstack([tail_block, budget_block], format="csr")
    bounds = (
        [(0.0, 1.0)] * option_count + [(None, None)] + [(0.0, None)] * observation_count
    )
    result = scipy.optimize.linprog(
        objective,
        A_ub=constraints,
        b_ub=np.concatenate([np.zeros(observation_count), [1.0]]),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        # The program is always feasible (all-cash) and bounded below, so a
        # failure is the solver's, not the caller's - surface it rather than
        # quietly return a suboptimal allocation.
        raise RuntimeError(f"The mean-CVaR linear program failed: {result.message}")
    weights = np.clip(result.x[:option_count], 0.0, 1.0)
    total = weights.sum()
    if total > 1.0:
        # HiGHS satisfies the budget to its feasibility tolerance (about
        # 1e-7); the house weight contract is tighter (1e-12), so scale the
        # rounding noise back inside it.
        weights = weights / total
    cvar = float(result.fun + mean @ result.x[:option_count])
    return weights, cvar, float(result.fun), bool(result.success), int(result.nit)


class MeanCVaR(BaseAllocationStrategy, ModelInputMixin):
    """
    Mean-CVaR allocation over a scenario matrix.

    Sizes the portfolio against a sample of joint simple-return vectors -
    empirical rows, model draws, anything shape ``(T, N)`` - by maximizing
    the unit-risk-aversion tradeoff between expected return and tail risk:

        ``maximize  w' mu - CVaR_tail(w)``

    where ``mu`` is the scenarios' mean vector and ``CVaR_tail`` the
    expected portfolio loss in the worst ``tail_alpha`` fraction of
    scenarios, losses read as negative simple returns. The scenarios are
    the whole story: sample-CVaR is defined for any finite sample, so
    fat-tailed marginals, mixture jumps, and distributions without a finite
    variance all size here - the territory where the moment-based methods'
    second-moment assumption would already be meaningless. The price is the
    empirical estimate's sampling noise: the scenarios are the distribution
    as far as this allocator knows.

    Through the Rockafellar-Uryasev auxiliary-variable trick the tradeoff
    is exactly a linear program - one tail threshold ``alpha``, one
    nonnegative surplus ``u_t`` per scenario, and the house long-only
    budget - solved with ``scipy.optimize.linprog`` under the HiGHS
    backend, which is why the class is gated on the ``keeks[allocation]``
    optional extra. Unit risk aversion is the v1 posture: the objective
    trades expected return against CVaR one-for-one, and a risk-aversion
    dial is a documented follow-up, not a v1 parameter (spec decision D2).

    The derivation is public and inspectable: ``scenarios`` (the validated
    descriptor, also the key the simulator's descriptor-equality gate
    reads), ``tail_alpha``, ``weights``, ``cvar`` (the optimal tail
    expectation), ``objective`` (the minimized program value),
    ``converged``, and ``iterations`` are exposed attributes.

    Weights are long-only and sum to no more than one - the residual is
    cash at zero return - and are scale-free: all zeros for a nonpositive
    bankroll.

    Parameters
    ----------
    scenarios : array-like
        The ``(observations, options)`` matrix of joint simple returns.
        Validated like every scenario descriptor: non-empty,
        two-dimensional, finite.
    tail_alpha : float, default=0.05
        The tail fraction, strictly between 0 and 1: ``0.05`` averages the
        worst one in twenty scenarios.

    Raises
    ------
    ImportError
        When scipy is not installed - the linear program is the method, so
        there is no numpy-only fallback. Install the optional solver
        backend with ``pip install keeks[allocation]``; the first example
        below demonstrates the pointed error.

    Examples
    --------
    The scipy gate points at the optional extra, not a deep traceback:

    >>> import builtins
    >>> real_import = builtins.__import__
    >>> def without_scipy(name, *args, **kwargs):
    ...     if name.startswith("scipy"):
    ...         raise ImportError(f"No module named {name!r}")
    ...     return real_import(name, *args, **kwargs)
    >>> builtins.__import__ = without_scipy
    >>> try:
    ...     MeanCVaR([[0.01], [-0.01]])
    ... except ImportError as error:
    ...     print("points at the extra:", "keeks[allocation]" in str(error))
    ... finally:
    ...     builtins.__import__ = real_import
    points at the extra: True

    >>> import numpy as np
    >>> scenarios = np.array([
    ...     [0.03, 0.01],
    ...     [-0.01, 0.02],
    ...     [0.01, -0.01],
    ...     [-0.02, -0.02],
    ... ])
    >>> strategy = MeanCVaR(scenarios)
    >>> weights = strategy.evaluate(1000.0)
    >>> len(weights)
    2
    >>> all(0.0 <= weight <= 1.0 for weight in weights)
    True
    >>> sum(weights) <= 1 + 1e-12
    True
    >>> strategy.evaluate(0.0)
    (0.0, 0.0)

    ``optimize()`` carries the solver's diagnostics on the result object:

    >>> result = strategy.optimize()
    >>> result.converged
    True
    >>> result.iterations >= 0
    True

    Any joint-return model builds a scenario-bound allocator through
    ``from_model`` - keeks binary bets, fitted marginals, user samplers:

    >>> from keeks.allocation.models import binary_bets_model
    >>> model = binary_bets_model([(0.5, 2.0, 1.0), (0.25, 3.0, 1.0)])
    >>> strategy = MeanCVaR.from_model(model, n_samples=256, seed=7)
    >>> strategy.scenarios.shape
    (256, 2)
    """

    def __init__(
        self,
        scenarios: np.typing.ArrayLike,
        tail_alpha: float = 0.05,
    ) -> None:
        # The linear program is the method: gate before anything else, so a
        # caller without scipy gets the pointed install hint, not a deep
        # traceback from the solver import.
        _require_scipy("Mean-CVaR optimization")
        self.tail_alpha: float = _validate_tail_alpha(tail_alpha)
        self.scenarios: np.ndarray
        self.scenarios, _ = _validate_scenarios(scenarios)
        (
            self.weights,
            self.cvar,
            self.objective,
            self.converged,
            self.iterations,
        ) = _mean_cvar_lp(self.scenarios, self.tail_alpha)

    @classmethod
    def from_model(
        cls,
        model: JointReturnModel,
        n_samples: int = 10_000,
        seed: int | None = None,
        **kwargs,
    ) -> "MeanCVaR":
        """
        Build a scenario-bound allocator from a joint-return model's draws.

        The scenario-bound override of the moment-based mixin: draws, not
        moments, are this allocator's descriptor, so ``from_model`` samples
        the model and binds the draw matrix - the model's exact ``moments()``
        hook, when it has one, is never consulted. A
        :class:`keeks.allocation.models.ScenarioModel` carrying residual
        probability mass contributes all-cash zero rows proportionally, and
        the program reads them like any other scenario.

        Parameters
        ----------
        model : JointReturnModel
            The model to draw from. Any object with
            ``sample(n_samples, rng)`` works.
        n_samples : int, default=10000
            The number of scenarios to draw; the program has one surplus
            variable per scenario, so very large counts cost solve time.
        seed : int, optional
            Seed for the private sampling stream; omit it for fresh draws
            with no replay promised.
        ``**kwargs``
            Extra constructor keyword arguments for ``cls`` (a tail
            fraction, ...).

        Returns
        -------
        MeanCVaR
            The allocator bound to the model's draws.

        Examples
        --------
        >>> from keeks.allocation.models import scenario_model
        >>> strategy = MeanCVaR.from_model(
        ...     scenario_model([[0.02], [-0.01], [0.01]]), n_samples=3, seed=1
        ... )
        >>> strategy.scenarios.shape
        (3, 1)
        """
        seed = _validate_simulator_seed(seed)
        n_samples = _validate_n_samples(n_samples)
        rng = np.random.default_rng(seed)
        draws = _validate_draws(model.sample(n_samples, rng), n_samples)
        return cls(draws, **kwargs)

    def optimize(self) -> AllocationResult:
        """
        Return the allocation with its solver diagnostics.

        ``objective`` is the minimized program value (the tail expectation
        minus the scenarios' expected return under the optimal weights) and
        ``cvar`` the tail expectation itself; ``expected_growth`` estimates
        ``E[log(1 + w'R)]`` under the scenarios - ``None`` when the optimal
        portfolio wipes out in some scenario, where the log is undefined -
        and ``volatility`` reads the weights against the scenarios'
        empirical covariance from :func:`scenarios_to_moments`.

        When the program's optimum is full cash - every weight effectively
        zero, as on daily-frequency market data under the
        unit-risk-aversion objective - ``all_cash_reason`` carries that
        explanation on the result, in the result object rather than only in
        example output.

        Returns
        -------
        AllocationResult
            The weight vector with the program's diagnostics.

        Examples
        --------
        >>> import numpy as np
        >>> scenarios = np.array([
        ...     [0.03, 0.01],
        ...     [-0.01, 0.02],
        ...     [0.01, -0.01],
        ...     [-0.02, -0.02],
        ... ])
        >>> result = MeanCVaR(scenarios).optimize()
        >>> result.converged
        True
        >>> result.weights.shape
        (2,)

        Daily-frequency scenarios hold full cash, and the result says why:

        >>> daily = np.array([
        ...     [0.0002, 0.0001],
        ...     [-0.0002, -0.0001],
        ...     [0.0001, -0.0001],
        ...     [-0.0001, 0.0002],
        ... ])
        >>> MeanCVaR(daily).optimize().all_cash_reason is not None
        True
        """
        portfolio_returns = self.scenarios @ self.weights
        if np.all(1.0 + portfolio_returns > 0.0):
            expected_growth = float(np.mean(np.log1p(portfolio_returns)))
        else:
            expected_growth = None
        _, covariance = scenarios_to_moments(self.scenarios)
        # The empirical covariance is PSD by construction, but rank-
        # deficient books can leave the quadratic form at -1e-18 float
        # noise; a variance cannot be negative, so clamp before the root.
        variance = float(self.weights @ covariance @ self.weights)
        volatility = float(np.sqrt(max(variance, 0.0)))
        all_cash_reason = None
        if np.all(np.abs(self.weights) <= PROBABILITY_SUM_TOLERANCE):
            all_cash_reason = (
                "The optimal portfolio holds full cash: the "
                "unit-risk-aversion objective (maximize w'mu - CVaR) trades "
                "expected return against the tail's expected loss "
                "one-for-one and is scale-homogeneous, so on "
                "daily-frequency market data - expected return far below "
                "the tail's expected loss - the honest optimum is all cash. "
                "The risk-aversion dial is a documented follow-up (spec "
                "decision D2)."
            )
        return AllocationResult(
            weights=self.weights.copy(),
            objective=self.objective,
            converged=self.converged,
            iterations=self.iterations,
            expected_growth=expected_growth,
            volatility=volatility,
            all_cash_reason=all_cash_reason,
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

        Examples
        --------
        >>> scenarios = [[0.03, 0.01], [-0.01, 0.02], [0.01, -0.01]]
        >>> MeanCVaR(scenarios).evaluate(0.0)
        (0.0, 0.0)
        """
        option_count = self.scenarios.shape[1]
        if current_bankroll <= 0:
            return _validate_weights([0.0] * option_count, option_count=option_count)
        return _validate_weights(self.weights, option_count=option_count)


def scenarios_to_moments(
    scenarios: np.typing.ArrayLike,
    probabilities: np.typing.ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bridge a scenario matrix to the ``(mean, covariance)`` moment descriptor.

    The exact moments of the discrete distribution over the scenario rows -
    equally weighted, or weighted by ``probabilities`` - computed through
    :class:`keeks.allocation.ScenarioModel`, so the validation and the
    residual-probability semantics are the model's: probability mass below
    one is an all-cash period at zero return, never renormalized away. This
    is the door moment-based methods (``MeanVariance``, minimum variance,
    ...) walk through to consume scenario inputs.

    Parameters
    ----------
    scenarios : array-like
        The ``(observations, options)`` matrix of joint simple returns.
    probabilities : array-like, optional
        The probability of each scenario row, validated like every keeks
        probability vector. Omitted mass is cash at zero return.

    Returns
    -------
    tuple of numpy.ndarray
        The exact ``(mean, covariance)`` of the scenario distribution - a
        shape ``(N,)`` mean vector and an ``(N, N)`` covariance matrix.

    Raises
    ------
    ValueError
        If the scenarios or the probabilities are invalid.

    Examples
    --------
    >>> mean, covariance = scenarios_to_moments(
    ...     [[0.02, 0.01], [-0.01, 0.01]], probabilities=[0.25, 0.25]
    ... )
    >>> mean.tolist()  # the other half of the mass is an all-cash period
    [0.0025, 0.005]
    """
    return ScenarioModel(scenarios, probabilities).moments()
