"""
Joint-return input models: the allocation layer's universal input currency.

Every allocator consumes one of two descriptors - a ``(mean, covariance)``
pair or a scenario matrix - and this module supplies the single abstraction
both come from: a :class:`JointReturnModel`, a source of joint simple-return
draws across N options. The model object, not its hyperparameters, is the
currency (decision D8 of the allocation spec): keeks binary bets, empirical
scenario rows, per-option parametric marginals, and user samplers all enter
the allocators through one sampling contract, so "kelly style binary bets in
the allocation context" and "incredibly fat tailed distributions" are the
same kind of input.

The adapters:

- :func:`scenario_model` wraps empirical rows (an iid bootstrap when
  resampled with a seeded :class:`numpy.random.Generator`), so a historical
  returns series is a one-liner.
- :func:`binary_bets_model` turns ``(probability, payoff, loss)`` triples
  into per-period simple returns with the settlement arithmetic
  :class:`keeks.multi_outcome.PortfolioSimulator` uses, and keys each bet's
  stream by its exact values the same way the portfolio simulator does.
- :func:`marginals_model` builds a model from per-option marginals - normal,
  Student-t, Laplace, lognormal, binary - independent (numpy-only) or joined
  by a Gaussian copula (scipy-gated behind ``keeks[allocation]``).
- :func:`fit_marginals_model` fits those marginals to a historical returns
  series (moment-matched for the thin-tailed families, maximum likelihood
  via ``scipy.stats`` for the Student-t).
- Anything exposing ``sample(n_samples, rng)`` works as-is: the estimators
  below duck-type the sampling contract, which is the escape hatch for
  user-supplied samplers.

Models are deterministic given a :class:`numpy.random.Generator`; models
whose hyperparameters give closed-form moments expose them exactly through
``moments()``, and the rest return ``None`` and are estimated by sampling
(:func:`estimate_moments`). Moment-based methods assume finite second
moments and say so; fat-tailed books are the intended territory of the
scenario methods, where sample-CVaR stays well defined where variance does
not.
"""

import abc
import dataclasses
import hashlib
import math
import operator
import struct
from collections.abc import Callable

import numpy as np

from keeks.allocation.base import _validate_covariance, _validate_scenarios
from keeks.multi_outcome.simulators import _validate_bets
from keeks.utils import _require_finite, _validate_simulator_seed

__author__ = "willmcginnis"


class JointReturnModel(abc.ABC):
    """
    Source of joint simple-return draws across N options.

    The allocation layer's universal input: scenario-based methods consume
    draws directly, moment-based methods consume moments estimated from
    draws, and a model that knows its moments exactly exposes them so no
    estimation is needed. Implementations are deterministic given a
    :class:`numpy.random.Generator` - the same generator state produces the
    same draws.

    Row convention matches the scenario validators: one row per observation,
    one column per option, values are per-period simple returns.
    """

    @abc.abstractmethod
    def sample(self, n_samples, rng):
        """
        Return joint simple-return draws.

        Parameters
        ----------
        n_samples : int
            The number of joint draws to produce. Must be a positive integer.
        rng : numpy.random.Generator
            The generator driving the draws. Implementations consume it
            deterministically: the same generator state yields the same
            draws.

        Returns
        -------
        numpy.ndarray
            An ``(n_samples, N)`` matrix of finite simple returns, one row
            per draw and one column per option.
        """
        raise NotImplementedError

    def moments(self):
        """
        Return the model's exact ``(mean, covariance)`` when it knows them.

        A binary bet, for instance, has closed-form moments in its
        hyperparameters, so its adapter returns them exactly and allocators
        built through ``from_model`` skip estimation entirely. Models without
        closed-form moments - Student-t marginals with ``nu <= 2`` (infinite
        variance), Gaussian-copula dependence (no closed-form Pearson
        covariance for non-Gaussian marginals), user samplers - return
        ``None`` and are estimated from draws instead.

        Returns
        -------
        tuple of numpy.ndarray or None
            The exact ``(mean, covariance)`` - a shape ``(N,)`` mean vector
            and an ``(N, N)`` covariance matrix - or ``None`` when the model
            does not know its moments exactly.
        """
        return None


def _validate_n_samples(n_samples):
    """
    Validate a draw count, which must be a positive integer.

    Parameters
    ----------
    n_samples : int
        The requested number of joint draws.

    Returns
    -------
    int
        The validated count.

    Raises
    ------
    ValueError
        If the count is not an integer or is not positive.

    Examples
    --------
    >>> _validate_n_samples("12")
    Traceback (most recent call last):
        ...
    ValueError: Sample count must be a positive integer
    """
    try:
        n_samples = operator.index(n_samples)
    except TypeError as exc:
        raise ValueError("Sample count must be a positive integer") from exc
    if n_samples < 1:
        raise ValueError("Sample count must be a positive integer")
    return n_samples


def _validate_draws(draws, n_samples):
    """
    Validate a model's draws against the sampling contract.

    Parameters
    ----------
    draws : array-like
        The joint simple-return draws a model produced.
    n_samples : int
        The requested draw count; the draws must carry exactly this many
        rows, one column per option.

    Returns
    -------
    numpy.ndarray
        The validated draws as a two-dimensional float array.

    Raises
    ------
    ValueError
        If the draws are not a two-dimensional ``(n_samples, N)`` matrix
        with ``N >= 1`` and only finite values.
    """
    try:
        draws = np.asarray(draws, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Joint-return draws must be a finite sequence") from exc
    if draws.ndim != 2:
        raise ValueError("Joint-return draws must be two-dimensional")
    if draws.shape[0] != n_samples or draws.shape[1] == 0:
        raise ValueError(
            f"Joint-return draws must have shape ({n_samples}, N) with N >= 1, "
            f"got shape {draws.shape}"
        )
    if not np.all(np.isfinite(draws)):
        raise ValueError("Joint-return draws must contain only finite values")
    return draws


def _require_scipy(feature):
    """
    Raise a pointed ImportError when scipy is missing.

    The allocation layer gates its scipy-using features behind the
    ``keeks[allocation]`` optional extra (spec decision D3); every gate
    names the extra so the fix is one install away.

    Parameters
    ----------
    feature : str
        The feature requesting scipy, named in the error message.

    Raises
    ------
    ImportError
        When scipy is not installed.
    """
    try:
        import scipy.stats  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            f"{feature} requires scipy, which is not installed; install the "
            "optional solver backend with `pip install keeks[allocation]` "
            "(or add scipy to your environment directly)."
        ) from exc


def _validate_dependence(dependence):
    """
    Validate a Gaussian-copula correlation matrix and return it as an array.

    A correlation matrix is a covariance with a unit diagonal, so the
    square/symmetry/PSD gates are the covariance ones; the unit diagonal is
    checked separately.

    Parameters
    ----------
    dependence : array-like
        The (N, N) correlation matrix of the Gaussian copula.

    Returns
    -------
    numpy.ndarray
        The validated correlation matrix as a two-dimensional float array.

    Raises
    ------
    ValueError
        If the matrix is not a valid correlation matrix.
    """
    try:
        correlation = _validate_covariance(dependence)
    except ValueError as exc:
        raise ValueError(
            f"Dependence must be a valid correlation matrix: {exc}"
        ) from exc
    if not np.allclose(np.diag(correlation), 1.0):
        raise ValueError("Dependence must be a correlation matrix with a unit diagonal")
    return correlation


@dataclasses.dataclass(frozen=True)
class _Marginal:
    """One option's marginal: how to sample it, invert it, and moment it."""

    family: str
    sample: Callable  # (rng, n_samples) -> (n_samples,) column
    ppf: Callable  # (u) -> (n_samples,) column from copula uniforms
    moments: Callable  # () -> (mean, variance) | None


def _normal_marginal(mean, sd):
    """Build the ``("normal", mean, sd)`` marginal."""

    def sample(rng, n_samples):
        return rng.normal(mean, sd, n_samples)

    def ppf(u):
        import scipy.stats

        return scipy.stats.norm.ppf(u, loc=mean, scale=sd)

    def moments():
        return mean, sd * sd

    return _Marginal("normal", sample, ppf, moments)


def _student_t_marginal(nu, loc, scale):
    """Build the ``("student_t", nu[, loc, scale])`` marginal."""

    def sample(rng, n_samples):
        return loc + scale * rng.standard_t(nu, n_samples)

    def ppf(u):
        import scipy.stats

        return scipy.stats.t.ppf(u, nu, loc=loc, scale=scale)

    def moments():
        if nu <= 2:
            # Infinite variance: the model honestly cannot produce one.
            return None
        return loc, scale * scale * nu / (nu - 2)

    return _Marginal("student_t", sample, ppf, moments)


def _laplace_marginal(loc, scale):
    """Build the ``("laplace", loc, scale)`` marginal."""

    def sample(rng, n_samples):
        return rng.laplace(loc, scale, n_samples)

    def ppf(u):
        import scipy.stats

        return scipy.stats.laplace.ppf(u, loc=loc, scale=scale)

    def moments():
        return loc, 2.0 * scale * scale

    return _Marginal("laplace", sample, ppf, moments)


def _lognormal_marginal(mean, sd):
    """Build the ``("lognormal", mean, sd)`` marginal (the variable's own moments)."""
    sigma_squared = math.log1p((sd / mean) ** 2)
    mu = math.log(mean) - sigma_squared / 2.0

    def sample(rng, n_samples):
        return rng.lognormal(mu, math.sqrt(sigma_squared), n_samples)

    def ppf(u):
        import scipy.stats

        return scipy.stats.lognorm.ppf(
            u, s=math.sqrt(sigma_squared), scale=math.exp(mu)
        )

    def moments():
        return mean, sd * sd

    return _Marginal("lognormal", sample, ppf, moments)


def _binary_marginal(probability, payoff, loss):
    """Build the ``("binary", probability, payoff, loss)`` marginal."""

    def sample(rng, n_samples):
        return np.where(rng.random(n_samples) < probability, payoff, -loss)

    def ppf(u):
        return np.where(u < probability, payoff, -loss)

    def moments():
        mean = probability * payoff - (1.0 - probability) * loss
        second = probability * payoff**2 + (1.0 - probability) * loss**2
        return mean, second - mean * mean

    return _Marginal("binary", sample, ppf, moments)


def _parse_marginal(spec, index):
    """
    Parse one marginal specification into a :class:`_Marginal`.

    Parameters
    ----------
    spec : tuple
        A ``(family, *parameters)`` tuple naming one of the built-in
        families: ``("normal", mean, sd)``, ``("student_t", nu)`` or
        ``("student_t", nu, loc, scale)``, ``("laplace", loc, scale)``,
        ``("lognormal", mean, sd)`` (the lognormal variable's own mean and
        standard deviation), or ``("binary", probability, payoff, loss)``.
    index : int
        The option position, named in error messages.

    Returns
    -------
    _Marginal
        The parsed marginal.

    Raises
    ------
    ValueError
        If the spec is not a tuple naming a built-in family with valid
        parameters.
    """
    if not isinstance(spec, tuple) or not spec:
        raise ValueError(
            f"Marginal {index} must be a (family, *parameters) tuple naming one "
            f"of the built-in families: normal, student_t, laplace, lognormal, binary"
        )
    family = spec[0]
    if family == "normal":
        if len(spec) != 3:
            raise ValueError(f"Marginal {index} must be a ('normal', mean, sd) tuple")
        mean = _require_finite(spec[1], f"Marginal {index} mean")
        sd = _require_finite(spec[2], f"Marginal {index} sd")
        if sd < 0:
            raise ValueError(f"Marginal {index} sd must be non-negative")
        return _normal_marginal(mean, sd)
    if family == "student_t":
        if len(spec) not in (2, 4):
            raise ValueError(
                f"Marginal {index} must be a ('student_t', nu) or "
                "('student_t', nu, loc, scale) tuple"
            )
        nu = _require_finite(spec[1], f"Marginal {index} nu")
        if nu <= 0:
            raise ValueError(f"Marginal {index} nu must be positive")
        loc = 0.0
        scale = 1.0
        if len(spec) == 4:
            loc = _require_finite(spec[2], f"Marginal {index} loc")
            scale = _require_finite(spec[3], f"Marginal {index} scale")
            if scale < 0:
                raise ValueError(f"Marginal {index} scale must be non-negative")
        return _student_t_marginal(nu, loc, scale)
    if family == "laplace":
        if len(spec) != 3:
            raise ValueError(
                f"Marginal {index} must be a ('laplace', loc, scale) tuple"
            )
        loc = _require_finite(spec[1], f"Marginal {index} loc")
        scale = _require_finite(spec[2], f"Marginal {index} scale")
        if scale < 0:
            raise ValueError(f"Marginal {index} scale must be non-negative")
        return _laplace_marginal(loc, scale)
    if family == "lognormal":
        if len(spec) != 3:
            raise ValueError(
                f"Marginal {index} must be a ('lognormal', mean, sd) tuple"
            )
        mean = _require_finite(spec[1], f"Marginal {index} mean")
        sd = _require_finite(spec[2], f"Marginal {index} sd")
        if mean <= 0:
            raise ValueError(f"Marginal {index} mean must be positive")
        if sd < 0:
            raise ValueError(f"Marginal {index} sd must be non-negative")
        return _lognormal_marginal(mean, sd)
    if family == "binary":
        if len(spec) != 4:
            raise ValueError(
                f"Marginal {index} must be a ('binary', probability, payoff, loss) tuple"
            )
        probability = _require_finite(spec[1], f"Marginal {index} probability")
        if not 0 <= probability <= 1:
            raise ValueError(f"Marginal {index} probability must be between 0 and 1")
        payoff = _require_finite(spec[2], f"Marginal {index} payoff")
        if payoff <= 0:
            raise ValueError(f"Marginal {index} payoff must be greater than 0")
        loss = _require_finite(spec[3], f"Marginal {index} loss")
        if loss < 0:
            raise ValueError(f"Marginal {index} loss must be non-negative")
        return _binary_marginal(probability, payoff, loss)
    raise ValueError(
        f"Marginal {index} names an unknown family {family!r}; the built-in "
        "families are 'normal', 'student_t', 'laplace', 'lognormal', and 'binary'"
    )


class ScenarioModel(JointReturnModel):
    """
    Empirical scenario rows as a joint-return model.

    The rows of the scenario matrix are the model's outcomes: sampling picks
    rows. Equally-weighted rows resample through the generator as an iid
    bootstrap of the observations; weighted rows draw by probability, and
    any probability mass below one is an all-cash period at zero return (a
    row of zeros), never renormalized away.

    Parameters
    ----------
    scenarios : array-like
        The ``(observations, options)`` matrix of joint simple returns.
    probabilities : array-like, optional
        The probability of each scenario row, validated like every keeks
        probability vector. Omitted mass is cash at zero return.

    Examples
    --------
    >>> model = ScenarioModel([[0.01, 0.02], [-0.01, 0.0]])
    >>> model.sample(2, np.random.default_rng(5)).shape
    (2, 2)
    """

    def __init__(self, scenarios, probabilities=None):
        self.scenarios: np.ndarray
        self.probabilities: np.ndarray | None
        self.scenarios, self.probabilities = _validate_scenarios(
            scenarios, probabilities
        )

    def sample(self, n_samples, rng):
        """
        Resample scenario rows.

        Equally-weighted rows draw uniform row indices; weighted rows draw
        uniforms against the cumulative probabilities, mapping the residual
        mass to the all-cash zero row.

        Parameters
        ----------
        n_samples : int
            The number of joint draws to produce.
        rng : numpy.random.Generator
            The generator driving the draws.

        Returns
        -------
        numpy.ndarray
            An ``(n_samples, N)`` matrix of simple returns.
        """
        n_samples = _validate_n_samples(n_samples)
        if self.probabilities is None:
            rows = rng.integers(0, self.scenarios.shape[0], size=n_samples)
            draws = self.scenarios[rows]
        else:
            cumulative = np.cumsum(self.probabilities)
            extended = np.vstack(
                [self.scenarios, np.zeros((1, self.scenarios.shape[1]))]
            )
            rows = np.searchsorted(cumulative, rng.random(n_samples), side="right")
            draws = extended[rows]
        return _validate_draws(draws, n_samples)

    def moments(self):
        """
        Return the exact probability-weighted empirical moments.

        The model's distribution is the discrete one over its rows, so the
        moments are exact sums - and when probabilities leave residual mass,
        the all-cash zero row carries it.

        Returns
        -------
        tuple of numpy.ndarray
            The exact ``(mean, covariance)`` of the empirical distribution.
        """
        if self.probabilities is None:
            weights = np.full(self.scenarios.shape[0], 1.0 / self.scenarios.shape[0])
            rows = self.scenarios
        else:
            weights = np.append(self.probabilities, 1.0 - self.probabilities.sum())
            rows = np.vstack([self.scenarios, np.zeros((1, self.scenarios.shape[1]))])
        mean = weights @ rows
        centered = rows - mean
        covariance = (centered * weights[:, None]).T @ centered
        return mean, covariance


def scenario_model(scenarios, probabilities=None):
    """
    Build a joint-return model from empirical scenario rows.

    The thin factory over :class:`ScenarioModel`; see there for the
    semantics. A historical returns series passed as ``scenarios`` becomes a
    first-class input: resample it with a seeded generator and the model is
    an iid bootstrap of the observations.

    Parameters
    ----------
    scenarios : array-like
        The ``(observations, options)`` matrix of joint simple returns.
    probabilities : array-like, optional
        The probability of each scenario row. Omitted mass is cash at zero
        return.

    Returns
    -------
    ScenarioModel
        The empirical model.

    Examples
    --------
    >>> model = scenario_model([[0.02, 0.01], [-0.01, 0.03], [0.0, -0.02]])
    >>> model.sample(4, np.random.default_rng(9)).shape
    (4, 2)
    """
    return ScenarioModel(scenarios, probabilities)


def _bet_spawn_key(bet, occurrences):
    """
    Build a bet's BLAKE2b-keyed ``spawn_key`` and count its occurrence.

    Mirrors the portfolio simulator's stream family: the key digests the
    bet's exact three float values, and identical tuples disambiguate by
    their zero-based occurrence ordinal in bet order - so a heterogeneous
    bet keeps its stream when other bets are inserted, removed, or
    reordered.

    Parameters
    ----------
    bet : tuple of float
        The validated ``(probability, payoff, loss)`` triple.
    occurrences : dict
        Running occurrence counts keyed by the packed bet bytes.

    Returns
    -------
    tuple of int
        The ``SeedSequence`` spawn key for this bet.
    """
    packed_bet = struct.pack(">ddd", *bet)
    occurrence = occurrences.get(packed_bet, 0)
    occurrences[packed_bet] = occurrence + 1
    identity = hashlib.blake2b(packed_bet, digest_size=16).digest()
    return (*struct.unpack(">IIII", identity), occurrence)


class BinaryBetsModel(JointReturnModel):
    """
    Keeks-native binary bets as a joint-return model.

    Each ``(probability, payoff, loss)`` triple becomes one option whose
    per-period simple return is the bet's settlement per unit staked:
    ``(payoff - 1) - transaction_costs`` on a win, ``-(loss + transaction_costs)``
    on a loss. This is the arithmetic :class:`keeks.multi_outcome.PortfolioSimulator`
    settles with - a win adds ``(payoff - 1) * stake`` to the bankroll
    (``payoff`` is decimal-odds gross including the returned stake) minus the
    fee - so allocation weights over these options replay portfolio stakes.
    ``transaction_costs`` here is a per-unit-staked fee (the strategy-side
    fractional unit): it matches the portfolio simulator's flat absolute fee
    on a unit stake.

    Bets are independent: each settles on its own draw from its own stream,
    keyed by the bet's exact values (BLAKE2b digest plus the occurrence
    ordinal of identical duplicates) beneath one per-call child seed drawn
    from the caller's generator - so the same generator seed replays
    byte-identically, different seeds diverge, and adding, removing, or
    reordering bets never shifts a surviving bet's stream.

    Parameters
    ----------
    bets : sequence of (probability, payoff, loss) triples
        One triple per independent bet, validated like the portfolio
        simulator's bets.
    transaction_costs : float, default=0.0
        The per-unit-staked fee, subtracted from a win and added to a loss.

    Examples
    --------
    >>> model = BinaryBetsModel([(0.5, 2.0, 1.0), (0.25, 3.0, 1.0)])
    >>> mean, covariance = model.moments()
    >>> mean.tolist()
    [0.0, -0.25]
    >>> covariance.diagonal().tolist()
    [1.0, 1.6875]
    """

    def __init__(self, bets, transaction_costs=0.0):
        self.bets: tuple[tuple[float, float, float], ...] = _validate_bets(bets)
        transaction_costs = _require_finite(transaction_costs, "Transaction costs")
        if transaction_costs < 0:
            raise ValueError("Transaction costs must be non-negative")
        self.transaction_costs: float = transaction_costs

    def _bet_returns(self, uniforms, bet):
        """Map one bet's uniforms to its per-period simple returns."""
        _, payoff, loss = bet
        win_return = (payoff - 1.0) - self.transaction_costs
        loss_return = -(loss + self.transaction_costs)
        return np.where(uniforms < bet[0], win_return, loss_return)

    def sample(self, n_samples, rng):
        """
        Draw every bet's per-period returns.

        One uniform per bet per draw, from the bet's own value-keyed stream;
        the per-call child seed consumes the caller's generator, so repeated
        calls draw fresh independent streams while staying deterministic
        given the generator.

        Parameters
        ----------
        n_samples : int
            The number of joint draws to produce.
        rng : numpy.random.Generator
            The generator driving the draws.

        Returns
        -------
        numpy.ndarray
            An ``(n_samples, N)`` matrix of simple returns, one column per
            bet.
        """
        n_samples = _validate_n_samples(n_samples)
        child_entropy = int.from_bytes(rng.bytes(8), "big")
        occurrences: dict[bytes, int] = {}
        columns = []
        for bet in self.bets:
            spawn_key = _bet_spawn_key(bet, occurrences)
            stream = np.random.default_rng(
                np.random.SeedSequence(child_entropy, spawn_key=spawn_key)
            )
            columns.append(self._bet_returns(stream.random(n_samples), bet))
        return _validate_draws(np.stack(columns, axis=1), n_samples)

    def moments(self):
        """
        Return the exact closed-form moments of the independent bets.

        Per bet: ``mean = p * ((payoff - 1) - tc) - (1 - p) * (loss + tc)`` and
        variance from the second moment; independence across bets makes the
        covariance diagonal.

        Returns
        -------
        tuple of numpy.ndarray
            The exact ``(mean, covariance)``.
        """
        means = []
        variances = []
        for probability, payoff, loss in self.bets:
            win_return = (payoff - 1.0) - self.transaction_costs
            loss_return = loss + self.transaction_costs
            mean = probability * win_return - (1.0 - probability) * loss_return
            second = probability * win_return**2 + (1.0 - probability) * loss_return**2
            means.append(mean)
            variances.append(second - mean * mean)
        return np.array(means), np.diag(np.array(variances))


def binary_bets_model(bets, transaction_costs=0.0):
    """
    Build a joint-return model from keeks-native binary bets.

    The thin factory over :class:`BinaryBetsModel`; see there for the
    settlement arithmetic and the stream-keying contract.

    Parameters
    ----------
    bets : sequence of (probability, payoff, loss) triples
        One triple per independent bet.
    transaction_costs : float, default=0.0
        The per-unit-staked fee.

    Returns
    -------
    BinaryBetsModel
        The binary-bets model.

    Examples
    --------
    >>> model = binary_bets_model([(0.5, 2.0, 1.0), (0.25, 3.0, 1.0)])
    >>> model.sample(3, np.random.default_rng(11)).shape
    (3, 2)
    """
    return BinaryBetsModel(bets, transaction_costs)


class MarginalModel(JointReturnModel):
    """
    Per-option parametric marginals joined by a configurable dependence.

    Each option carries one built-in marginal family - ``normal``,
    ``student_t``, ``laplace``, ``lognormal``, or ``binary`` - and the
    dependence model joins them: ``None`` (the default) samples the
    marginals independently, which is numpy-only, and an ``(N, N)``
    correlation matrix joins them through a Gaussian copula, which needs
    scipy and is gated behind ``keeks[allocation]`` at construction (spec
    decision D3).

    Parameters
    ----------
    marginals : sequence of (family, ``*parameters``) tuples
        One spec per option: ``("normal", mean, sd)``,
        ``("student_t", nu)`` or ``("student_t", nu, loc, scale)``,
        ``("laplace", loc, scale)``, ``("lognormal", mean, sd)`` (the
        lognormal variable's own mean and standard deviation), or
        ``("binary", probability, payoff, loss)``.
    dependence : array-like, optional
        The Gaussian copula's correlation matrix. ``None`` means
        independent.

    Raises
    ------
    ImportError
        When ``dependence`` is given and scipy is not installed.

    Examples
    --------
    >>> model = MarginalModel([("normal", 0.01, 0.02), ("binary", 0.5, 2.0, 1.0)])
    >>> model.sample(3, np.random.default_rng(3)).shape
    (3, 2)
    """

    def __init__(self, marginals, dependence=None):
        try:
            specs = list(marginals)
        except TypeError as exc:
            raise ValueError(
                "Marginals must be a non-empty sequence of (family, *parameters) tuples"
            ) from exc
        if not specs:
            raise ValueError(
                "Marginals must be a non-empty sequence of (family, *parameters) tuples"
            )
        self.marginals: list[_Marginal] = [
            _parse_marginal(spec, index) for index, spec in enumerate(specs)
        ]
        if dependence is None:
            self.dependence: np.ndarray | None = None
        else:
            # Gate at construction, like every scipy-gated feature: a model
            # that cannot be sampled should never finish building.
            _require_scipy("Gaussian-copula dependence")
            self.dependence = _validate_dependence(dependence)

    def sample(self, n_samples, rng):
        """
        Draw the joint simple returns.

        Independent models draw each marginal's column from the shared
        generator in option order; the Gaussian copula draws correlated
        normals from the generator, maps them to uniforms through the normal
        CDF, and maps those through each marginal's inverse CDF.

        Parameters
        ----------
        n_samples : int
            The number of joint draws to produce.
        rng : numpy.random.Generator
            The generator driving the draws.

        Returns
        -------
        numpy.ndarray
            An ``(n_samples, N)`` matrix of simple returns.
        """
        n_samples = _validate_n_samples(n_samples)
        if self.dependence is None:
            columns = [marginal.sample(rng, n_samples) for marginal in self.marginals]
        else:
            import scipy.stats

            count = len(self.marginals)
            normals = rng.multivariate_normal(
                np.zeros(count), self.dependence, size=n_samples
            )
            uniforms = scipy.stats.norm.cdf(normals)
            columns = [
                marginal.ppf(uniforms[:, index])
                for index, marginal in enumerate(self.marginals)
            ]
        return _validate_draws(np.stack(columns, axis=1), n_samples)

    def moments(self):
        """
        Return the exact marginal moments when the model knows them.

        Independence makes the joint moments the per-option ones (a diagonal
        covariance); any option with infinite variance - a Student-t with
        ``nu <= 2`` - leaves the joint covariance undefined and returns
        ``None``. A Gaussian copula preserves every marginal's mean and
        variance but not the Pearson covariances, which have no closed form
        for non-Gaussian marginals, so copula models return ``None`` too.

        Returns
        -------
        tuple of numpy.ndarray or None
            The exact ``(mean, covariance)``, or ``None`` when the model
            does not know its joint moments exactly.
        """
        if self.dependence is not None:
            return None
        means = []
        variances = []
        for marginal in self.marginals:
            option = marginal.moments()
            if option is None:
                return None
            means.append(option[0])
            variances.append(option[1])
        return np.array(means), np.diag(np.array(variances))


def marginals_model(marginals, dependence=None):
    """
    Build a joint-return model from per-option parametric marginals.

    The thin factory over :class:`MarginalModel`; see there for the family
    specs and the dependence gating.

    Parameters
    ----------
    marginals : sequence of (family, ``*parameters``) tuples
        One spec per option.
    dependence : array-like, optional
        The Gaussian copula's correlation matrix. ``None`` means
        independent.

    Returns
    -------
    MarginalModel
        The marginals model.

    Examples
    --------
    >>> model = marginals_model([("normal", 0.01, 0.02), ("binary", 0.5, 2.0, 1.0)])
    >>> mean, covariance = model.moments()
    >>> mean.tolist()
    [0.01, 0.5]
    """
    return MarginalModel(marginals, dependence)


def fit_marginals_model(returns, family="student_t", dependence=None):
    """
    Fit per-option parametric marginals to a historical returns series.

    The "bring your own history" counterpart to :func:`marginals_model`:
    each column of the series is fitted to one built-in family and the
    fitted specs build a :class:`MarginalModel`. The thin-tailed families
    are moment-matched to the column's mean and (population) standard
    deviation; the Student-t is fitted by maximum likelihood through
    ``scipy.stats``, which gates on the ``keeks[allocation]`` extra.

    Parameters
    ----------
    returns : array-like
        The ``(observations, options)`` matrix of historical simple returns.
    family : str, default="student_t"
        One of ``"normal"``, ``"student_t"``, ``"laplace"``, or
        ``"lognormal"``.
    dependence : array-like, optional
        The Gaussian copula's correlation matrix for the fitted marginals.
        ``None`` means independent.

    Returns
    -------
    MarginalModel
        The fitted model.

    Raises
    ------
    ImportError
        When ``family`` is ``"student_t"`` and scipy is not installed.
    ValueError
        When the returns are invalid, the family is unknown, or a
        lognormal fit is asked for nonpositive observations.

    Examples
    --------
    >>> import numpy as np
    >>> returns = [[0.01, 0.03], [0.03, -0.01], [-0.01, 0.01]]
    >>> model = fit_marginals_model(returns, family="normal")
    >>> mean, _ = model.moments()
    >>> bool(np.allclose(mean, [0.01, 0.01]))
    True
    """
    scenarios, _ = _validate_scenarios(returns)
    if family == "normal":
        specs = [
            ("normal", float(column.mean()), float(column.std()))
            for column in scenarios.T
        ]
    elif family == "laplace":
        specs = [
            ("laplace", float(column.mean()), float(column.std() / math.sqrt(2.0)))
            for column in scenarios.T
        ]
    elif family == "lognormal":
        if not np.all(scenarios > 0):
            raise ValueError("Lognormal marginals require strictly positive returns")
        specs = [
            ("lognormal", float(column.mean()), float(column.std()))
            for column in scenarios.T
        ]
    elif family == "student_t":
        _require_scipy("Fitting Student-t marginals")
        import scipy.stats

        specs = []
        for column in scenarios.T:
            nu, loc, scale = scipy.stats.t.fit(column)
            specs.append(("student_t", float(nu), float(loc), float(scale)))
    else:
        raise ValueError(
            f"Unknown family {family!r}; fit_marginals_model fits 'normal', "
            "'student_t', 'laplace', or 'lognormal'"
        )
    return marginals_model(specs, dependence=dependence)


def estimate_moments(model, n_samples, seed=None):
    """
    Estimate a model's ``(mean, covariance)`` from its own draws.

    The fallback for models whose :meth:`JointReturnModel.moments` is
    ``None``: draw and take the sample mean and the unbiased sample
    covariance. Works on any object implementing the sampling contract -
    the escape hatch for user-supplied samplers.

    Parameters
    ----------
    model : JointReturnModel
        The model to draw from. Any object with ``sample(n_samples, rng)``
        works.
    n_samples : int
        The number of draws; at least two, since a covariance needs them.
    seed : int, optional
        Seed for the private estimation stream. When omitted, a fresh
        generator drives the draws and no replay is promised.

    Returns
    -------
    tuple of numpy.ndarray
        The estimated ``(mean, covariance)``.

    Raises
    ------
    ValueError
        If ``n_samples`` is not an integer of at least two, the seed is not
        a nonnegative integer or ``None``, or the model's draws violate the
        sampling contract.

    Examples
    --------
    >>> model = scenario_model([[0.01, -0.01], [0.03, 0.01]])
    >>> mean, covariance = estimate_moments(model, 512, seed=0)
    >>> mean.shape
    (2,)
    >>> mean2, _ = estimate_moments(model, 512, seed=0)
    >>> bool(np.array_equal(mean, mean2))
    True
    """
    seed = _validate_simulator_seed(seed)
    rng = np.random.default_rng(seed)
    n_samples = _validate_n_samples(n_samples)
    if n_samples < 2:
        raise ValueError("Estimating a covariance needs at least two samples")
    draws = _validate_draws(model.sample(n_samples, rng), n_samples)
    mean = draws.mean(axis=0)
    centered = draws - mean
    covariance = centered.T @ centered / (n_samples - 1)
    return mean, covariance


class ModelInputMixin:
    """
    Mixin giving moment-based allocators a ``from_model`` constructor.

    Allocators consume ``(mean, covariance)`` or scenario descriptors bound
    at construction; this mixin builds those descriptors from any
    :class:`JointReturnModel` - keeks binary bets, fitted marginals, user
    samplers - so every allocator accepts every input through one door. The
    classmethod prefers the model's exact ``moments()`` and falls back to
    :func:`estimate_moments` when the model cannot produce them.

    Scenario-bound allocators consume draws rather than moments and override
    ``from_model``; the constructor keyword arguments pass through to
    ``cls`` either way.

    Examples
    --------
    >>> class EqualWeights(ModelInputMixin):
    ...     def __init__(self, mean, covariance):
    ...         self.mean = mean
    ...         self.covariance = covariance
    >>> model = binary_bets_model([(0.5, 2.0, 1.0)])
    >>> EqualWeights.from_model(model).mean.tolist()
    [0.0]
    """

    @classmethod
    def from_model(cls, model, n_samples=10_000, seed=None, **kwargs):
        """
        Build ``cls`` from a joint-return model's inputs.

        Exact moments win when the model carries them, so a binary-bets or
        thin-tailed parametric model needs no Monte Carlo at all;
        ``n_samples`` and ``seed`` are only consumed on the estimation
        fallback.

        Parameters
        ----------
        model : JointReturnModel
            The model to build inputs from. Any object with
            ``sample(n_samples, rng)`` (and optionally ``moments()``) works.
        n_samples : int, default=10000
            The draw count for the estimation fallback.
        seed : int, optional
            Seed for the estimation fallback's private stream.
        **kwargs
            Extra constructor keyword arguments for ``cls`` (a risk
            aversion, risk budgets, ...).

        Returns
        -------
        BaseAllocationStrategy
            The allocator built from the model's inputs.
        """
        moments = getattr(model, "moments", None)
        exact = moments() if callable(moments) else None
        if exact is None:
            exact = estimate_moments(model, n_samples, seed=seed)
        mean, covariance = exact
        return cls(mean, covariance, **kwargs)
