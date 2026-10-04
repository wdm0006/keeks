"""
Covariance and mean estimators for the allocation layer.

Two numpy-only estimators prepare allocation inputs without sizing anything
themselves - neither is a :class:`BaseAllocationStrategy`; both compose with
any optimizer that consumes the moments they produce:

- :func:`shrink_covariance` shrinks a sample covariance toward a structured
  target (Ledoit & Wolf 2004): a convex combination that trades the sample
  matrix's noise for the target's stability, with the shrinkage intensity
  either fixed by the caller or estimated from the samples. It fixes Σ only,
  never μ - pair it with whatever mean estimate the caller trusts.
- :func:`black_litterman_mean` blends equilibrium market returns with
  discrete views into one posterior mean (Black & Litterman 1992). It fixes
  μ only, never Σ - the posterior mean feeds any optimizer that consumes a
  mean vector, with the covariance left untouched.

Both honor the subpackage's numeric discipline: matrices are validated with
the shared covariance gate, and nothing here imports scipy.
"""

import numpy as np

from keeks.allocation.base import _validate_covariance, _validate_weights
from keeks.allocation.online import _validate_positive_finite


def _validate_samples(samples):
    """
    Validate an estimation sample matrix and return it as a float array.

    One row per observation, one column per option, at least two rows - a
    covariance is not defined from fewer.

    Parameters
    ----------
    samples : array-like
        The (observations, options) matrix of joint simple returns.

    Returns
    -------
    numpy.ndarray
        The validated samples as a two-dimensional float array.

    Raises
    ------
    ValueError
        If the samples are not a two-dimensional sequence of at least two
        rows of finite numbers.

    Examples
    --------
    >>> _validate_samples([[0.01, -0.01], [0.03, 0.01]]).shape
    (2, 2)
    """
    try:
        samples = np.asarray(samples, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Estimation samples must be a finite sequence") from exc
    if samples.ndim != 2:
        raise ValueError("Estimation samples must be two-dimensional")
    if samples.size == 0:
        raise ValueError("Estimation samples must be non-empty")
    if not np.all(np.isfinite(samples)):
        raise ValueError("Estimation samples must contain only finite values")
    if samples.shape[0] < 2:
        raise ValueError("Estimation samples need at least two observations")
    return samples


def _validate_shrinkage_intensity(alpha):
    """
    Validate an explicit shrinkage intensity and return it as a ``float``.

    Parameters
    ----------
    alpha : float
        The shrinkage intensity, in ``[0, 1]``: 0 keeps the sample
        covariance, 1 replaces it with the target.

    Returns
    -------
    float
        The validated intensity.

    Raises
    ------
    ValueError
        If the intensity is not a finite number in ``[0, 1]``.

    Examples
    --------
    >>> _validate_shrinkage_intensity(0.25)
    0.25
    """
    try:
        alpha = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Shrinkage intensity must be a finite number between 0 and 1"
        ) from exc
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("Shrinkage intensity must be a finite number between 0 and 1")
    return alpha


def _ledoit_wolf_intensity(centered, sample_covariance, target):
    """
    Estimate the Ledoit-Wolf shrinkage intensity toward ``target``.

    The intensity is the ratio of the estimation noise to the distance the
    shrinkage would travel (Ledoit & Wolf 2004, theorem 2): squared Frobenius
    norms of the rank-one scatter matrices around the sample covariance,
    over the squared distance from the sample covariance to the target.
    Because the samples here are centered by their own mean and the sample
    covariance uses the unbiased ``(n - 1)`` denominator, the noise
    numerator is ``sum_i (z_i'z_i)^2 - (n - 2) * trace(S^2)`` - the identity
    ``sum_i ||z_i z_i' - S||^2 = sum_i (z_i'z_i)^2 - (n - 2) trace(S^2)``
    for centered ``z_i`` and the unbiased ``S``. The ratio is capped to
    ``[0, 1]``: never shrink past the target, never shrink away from it.

    Parameters
    ----------
    centered : numpy.ndarray
        The samples with their column means removed.
    sample_covariance : numpy.ndarray
        The unbiased sample covariance of ``centered``.
    target : numpy.ndarray
        The structured shrinkage target.

    Returns
    -------
    float
        The estimated intensity in ``[0, 1]``. Zero when the sample
        covariance already equals the target, where the ratio is undefined
        and there is nothing to shrink.
    """
    n_samples, n_options = centered.shape
    deviation = sample_covariance - target
    distance_squared = float(np.trace(deviation @ deviation)) / n_options
    if distance_squared <= 0.0:
        return 0.0

    squared_samples = centered**2
    squared_norm_total = float((squared_samples.T @ squared_samples).sum())
    trace_covariance_squared = float(np.trace(sample_covariance @ sample_covariance))
    estimation_noise = (
        squared_norm_total - (n_samples - 2) * trace_covariance_squared
    ) / (n_samples**2 * n_options)
    return max(0.0, min(1.0, estimation_noise / distance_squared))


def shrink_covariance(samples, alpha=None):
    """
    Ledoit-Wolf shrinkage estimate of a covariance matrix.

    The estimate is a convex combination of the unbiased sample covariance
    ``S`` and the structured target ``F = m * I``, where ``m`` is the
    average variance ``trace(S) / N`` (Ledoit & Wolf 2004)::

        shrunk = (1 - alpha) * S + alpha * F

    ``alpha = 0`` returns the sample covariance exactly and ``alpha = 1``
    the target exactly; in between, the blend trades the sample matrix's
    noise for the target's stability. When ``alpha`` is omitted, the
    intensity is estimated from the samples by the Ledoit-Wolf formula -
    the ratio of estimated estimation noise to the distance to the target,
    capped to ``[0, 1]`` - so the estimate adapts: trusted samples shrink
    barely, noisy ones shrink toward the average variance.

    The samples are centered by their own column means first, so the
    estimate describes dispersion around the observed means. Only the
    covariance is touched - there is no mean output, and no μ is consumed.

    Parameters
    ----------
    samples : array-like
        The (observations, options) matrix of joint simple returns. At
        least two rows - a covariance needs them.
    alpha : float, optional
        The shrinkage intensity in ``[0, 1]``. When omitted, the
        Ledoit-Wolf intensity is estimated from the samples.

    Returns
    -------
    tuple of numpy.ndarray and float
        The shrunk covariance as an ``(N, N)`` positive semidefinite matrix,
        and the intensity used (the estimated one when ``alpha`` is
        ``None``).

    Raises
    ------
    ValueError
        If the samples are not a two-dimensional sequence of at least two
        rows of finite numbers, or ``alpha`` is given and is not a finite
        number in ``[0, 1]``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> samples = rng.normal(scale=0.02, size=(128, 4))
    >>> shrunk, alpha = shrink_covariance(samples)
    >>> shrunk.shape
    (4, 4)
    >>> 0.0 <= alpha <= 1.0
    True
    >>> sample_only, _ = shrink_covariance(samples, alpha=0.0)
    >>> bool(np.allclose(sample_only, np.cov(samples, rowvar=False)))
    True
    >>> fully_shrunk, _ = shrink_covariance(samples, alpha=1.0)
    >>> target = np.eye(4) * np.trace(sample_only) / 4
    >>> bool(np.allclose(fully_shrunk, target))
    True
    """
    samples = _validate_samples(samples)
    centered = samples - samples.mean(axis=0)
    n_samples, n_options = centered.shape
    sample_covariance = centered.T @ centered / (n_samples - 1)
    mean_variance = float(np.trace(sample_covariance)) / n_options
    target = mean_variance * np.eye(n_options)

    if alpha is None:
        alpha = _ledoit_wolf_intensity(centered, sample_covariance, target)
    else:
        alpha = _validate_shrinkage_intensity(alpha)

    shrunk = (1.0 - alpha) * sample_covariance + alpha * target
    return shrunk, alpha


def _validate_views(views, option_count):
    """
    Validate the ``(P, q)`` view pair and return it as float arrays.

    ``P`` is the ``(K, N)`` matrix of view loadings - one row per view, one
    column per option, each row spelling out the portfolio the view is
    about - and ``q`` is the ``(K,)`` vector of simple returns each view
    expects for its portfolio.

    Parameters
    ----------
    views : tuple of array-like
        The ``(P, q)`` pair.
    option_count : int
        The number of options the view matrix must have one column per.

    Returns
    -------
    tuple of numpy.ndarray
        The validated ``(P, q)`` pair.

    Raises
    ------
    ValueError
        If ``views`` is not a two-element pair, ``P`` is not a
        two-dimensional finite matrix with one column per option, or ``q``
        is not a one-dimensional finite vector with one entry per view row.

    Examples
    --------
    >>> matrix, returns = _validate_views(
    ...     ([[1.0, -1.0]], [0.01]), 2
    ... )
    >>> matrix.shape
    (1, 2)
    >>> returns.shape
    (1,)
    """
    try:
        view_matrix, view_returns = views
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Views must be a (P, q) pair: the (K, N) view matrix and the "
            "(K,) view returns"
        ) from exc

    try:
        view_matrix = np.asarray(view_matrix, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("View matrix must be a finite sequence") from exc
    if view_matrix.ndim != 2:
        raise ValueError("View matrix must be two-dimensional")
    if not np.all(np.isfinite(view_matrix)):
        raise ValueError("View matrix must contain only finite values")
    if view_matrix.shape[1] != option_count:
        raise ValueError(
            f"View matrix must have exactly {option_count} columns, "
            f"got {view_matrix.shape[1]}"
        )

    try:
        view_returns = np.asarray(view_returns, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("View returns must be a finite sequence") from exc
    if view_returns.ndim != 1:
        raise ValueError("View returns must be one-dimensional")
    if not np.all(np.isfinite(view_returns)):
        raise ValueError("View returns must contain only finite values")
    if view_returns.shape[0] != view_matrix.shape[0]:
        raise ValueError("View returns must have one entry per view row")
    return view_matrix, view_returns


def black_litterman_mean(covariance, market_weights, risk_aversion, views, omega, tau):
    """
    Black-Litterman posterior mean: equilibrium returns blended with views.

    The equilibrium (prior) mean is the one implied by the market weights
    through reverse optimization, ``prior = risk_aversion * covariance @
    market_weights``. Each view - a row ``P_k`` of the view matrix whose
    portfolio the holder expects to return ``q_k`` - is a noisy observation
    of that portfolio's mean return, with uncertainty ``omega``. The
    posterior precision-weights the prior against the views (Black &
    Litterman 1992)::

        mu = [(tau * covariance)^-1 + P' omega^-1 P]^-1
             [(tau * covariance)^-1 prior + P' omega^-1 q]

    ``tau`` scales how much the prior is trusted relative to the views, and
    ``omega``'s diagonal how noisy each view is: a confident view (small
    uncertainty) pulls the posterior toward ``q``, an uninformative one
    leaves the prior standing. With no views the posterior is exactly the
    prior.

    The result is a mean preprocessor, not an allocator: it fixes μ only,
    never Σ - feed the posterior to any optimizer that consumes a mean
    vector and keep the covariance it was built from.

    Parameters
    ----------
    covariance : array-like
        The ``(N, N)`` covariance of the options' simple returns. Must be
        invertible - the posterior precision needs ``(tau * covariance)^-1``.
    market_weights : sequence of float
        One weight per option under the same long-only budget contract as
        strategy weights (each in ``[0, 1]``, summing to no more than one);
        the conventional fully-invested market portfolio passes with sum
        exactly one.
    risk_aversion : float
        The market risk aversion ``delta`` turning the market weights into
        equilibrium returns. Positive.
    views : tuple of array-like or None
        The ``(P, q)`` pair: the ``(K, N)`` view matrix and the ``(K,)``
        view returns. ``None``, or a pair whose matrix has zero rows
        (with ``omega`` an empty ``(0, 0)`` matrix), means no views.
    omega : array-like or None
        The ``(K, K)`` uncertainty covariance of the view returns.
        Provided together with ``views`` - both ``None`` or both set.
    tau : float
        The prior scaling. Positive.

    Returns
    -------
    numpy.ndarray
        The posterior mean, one entry per option.

    Raises
    ------
    ValueError
        If the covariance, view inputs, or omega fail the shared numeric
        gates; if exactly one of ``views`` and ``omega`` is ``None``; if
        either matrix is singular where an inverse is needed.

    Examples
    --------
    >>> import numpy as np
    >>> covariance = np.array([[0.02, 0.004], [0.004, 0.03]])
    >>> weights = [0.6, 0.4]
    >>> prior = black_litterman_mean(
    ...     covariance, weights, risk_aversion=2.5, views=None, omega=None,
    ...     tau=1.0,
    ... )
    >>> prior.shape
    (2,)
    >>> views = (np.array([[1.0, -1.0]]), np.array([0.01]))
    >>> posterior = black_litterman_mean(
    ...     covariance, weights, risk_aversion=2.5, views=views,
    ...     omega=np.array([[0.001]]), tau=1.0,
    ... )
    >>> posterior.shape
    (2,)
    """
    covariance = _validate_covariance(covariance)
    market_weights = np.asarray(
        _validate_weights(market_weights, covariance.shape[0]), dtype=float
    )
    risk_aversion = _validate_positive_finite(risk_aversion, "Risk aversion")
    tau = _validate_positive_finite(tau, "Tau")

    prior = risk_aversion * (covariance @ market_weights)

    if views is None and omega is None:
        return prior
    if views is None or omega is None:
        raise ValueError("Views and omega must be provided together")

    view_matrix, view_returns = _validate_views(views, covariance.shape[0])
    if view_matrix.shape[0] == 0:
        return prior
    omega = _validate_covariance(omega, view_matrix.shape[0])

    try:
        prior_precision = np.linalg.inv(tau * covariance)
        omega_inverse = np.linalg.inv(omega)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Black-Litterman needs an invertible covariance and view uncertainty matrix"
        ) from exc

    posterior_precision = prior_precision + view_matrix.T @ omega_inverse @ view_matrix
    posterior_rhs = (
        prior_precision @ prior + view_matrix.T @ omega_inverse @ view_returns
    )
    try:
        return np.linalg.solve(posterior_precision, posterior_rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Black-Litterman posterior precision is singular; check the "
            "covariance, omega, and tau"
        ) from exc
