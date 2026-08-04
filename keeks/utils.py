import math
import operator

import numpy as np

PROBABILITY_SUM_TOLERANCE = 1e-12

_UNSET = object()


class RuinError(Exception):
    """
    Exception raised when a bankroll experiences a drawdown exceeding the maximum allowed limit.

    This exception is typically raised by the BankRoll class when a withdrawal would cause
    the bankroll to drop below the configured maximum drawdown threshold.
    """

    pass


def crra_utility(wealth, risk_aversion=1.0):
    """
    Calculate CRRA (Constant Relative Risk Aversion) utility.

    For γ=1 (risk_aversion=1.0), uses log utility.
    For γ≠1, uses power utility: U(W) = W^(1-γ) / (1-γ)

    Parameters
    ----------
    wealth : float or array-like
        The wealth level(s) to evaluate
    risk_aversion : float, default=1.0
        Coefficient of relative risk aversion (γ)
        - γ=1.0: Log utility (Kelly Criterion)
        - γ=1.5-2.0: Moderate risk aversion
        - γ>2.0: High risk aversion

    Returns
    -------
    float or array-like
        The utility value(s)

    Notes
    -----
    CRRA utility exhibits constant relative risk aversion, meaning the
    fraction of wealth an agent is willing to risk remains constant
    as wealth changes.
    """
    if risk_aversion == 1.0:
        if np.any(wealth <= 0):
            return (
                -np.inf
                if np.isscalar(wealth)
                else np.where(wealth <= 0, -np.inf, np.log(wealth))
            )
        return np.log(wealth)
    else:
        if np.any(wealth <= 0):
            return (
                -np.inf
                if np.isscalar(wealth)
                else np.where(
                    wealth <= 0,
                    -np.inf,
                    (wealth ** (1 - risk_aversion)) / (1 - risk_aversion),
                )
            )
        return (wealth ** (1 - risk_aversion)) / (1 - risk_aversion)


def _normalize_gamble(outcomes, probabilities):
    """Validate a gamble and add its implicit zero-payout outcome."""
    try:
        outcomes = np.asarray(outcomes, dtype=float)
        probabilities = np.asarray(probabilities, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Outcomes and probabilities must be finite sequences") from exc

    if outcomes.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("Outcomes and probabilities must be one-dimensional")
    if outcomes.size == 0 or probabilities.size == 0:
        raise ValueError("Outcomes and probabilities must be non-empty")
    if outcomes.size != probabilities.size:
        raise ValueError("Outcomes and probabilities must have equal length")
    if not np.all(np.isfinite(outcomes)) or not np.all(np.isfinite(probabilities)):
        raise ValueError("Outcomes and probabilities must contain only finite values")
    if np.any(probabilities < 0):
        raise ValueError("Probabilities must be nonnegative")

    total_probability = probabilities.sum()
    if total_probability > 1 + PROBABILITY_SUM_TOLERANCE:
        raise ValueError("Probabilities must sum to no more than one")
    if total_probability > 1:
        probabilities = probabilities / total_probability
        total_probability = 1.0
    if total_probability < 1:
        outcomes = np.append(outcomes, 0.0)
        probabilities = np.append(probabilities, 1.0 - total_probability)

    return outcomes, probabilities


def _require_finite(value, name):
    """Coerce ``value`` to a finite float or raise ``ValueError``."""
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _validate_entry_price_scalars(
    current_wealth, tolerance, max_search_fraction, risk_aversion=_UNSET
):
    """
    Validate the scalar controls shared by every entry-price calculation.

    ``current_wealth`` and ``tolerance`` must be finite and positive,
    ``max_search_fraction`` must be finite and nonnegative (values above 1.0 are
    allowed), and ``risk_aversion`` — when the caller has one — must be finite
    and positive.

    Raises
    ------
    ValueError
        If any control is outside its accepted range.
    """
    if _require_finite(current_wealth, "Current wealth") <= 0:
        raise ValueError("Current wealth must be greater than 0")
    if _require_finite(tolerance, "Tolerance") <= 0:
        raise ValueError("Tolerance must be greater than 0")
    if _require_finite(max_search_fraction, "Maximum search fraction") < 0:
        raise ValueError("Maximum search fraction must be non-negative")
    if (
        risk_aversion is not _UNSET
        and _require_finite(risk_aversion, "Risk aversion") <= 0
    ):
        raise ValueError("Risk aversion must be greater than 0")


def _validate_simulator_controls(payoff, loss, transaction_costs, trials):
    """
    Validate the controls shared by every simulator constructor.

    ``payoff`` must be finite and positive, ``loss`` and the flat
    ``transaction_costs`` fee must be finite and nonnegative, and ``trials`` must
    be a nonnegative integer.

    Returns
    -------
    tuple
        The validated ``(payoff, loss, transaction_costs, trials)``, with the
        numeric controls coerced to ``float``.

    Raises
    ------
    ValueError
        If any control is outside its accepted range.
    """
    payoff = _require_finite(payoff, "Payoff")
    if payoff <= 0:
        raise ValueError("Payoff must be greater than 0")
    loss = _require_finite(loss, "Loss")
    if loss < 0:
        raise ValueError("Loss must be non-negative")
    transaction_costs = _require_finite(transaction_costs, "Transaction costs")
    if transaction_costs < 0:
        raise ValueError("Transaction costs must be non-negative")

    try:
        trials = operator.index(trials)
    except TypeError as exc:
        raise ValueError("Trials must be a nonnegative integer") from exc
    if trials < 0:
        raise ValueError("Trials must be a nonnegative integer")

    return payoff, loss, transaction_costs, trials


def _validate_simulator_probability(probability, name):
    """Validate a simulator's fixed probability, which must be finite in [0, 1]."""
    probability = _require_finite(probability, name)
    if not 0 <= probability <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return probability


def _validate_simulator_stdev(stdev, name):
    """Validate a simulator's standard deviation, which must be finite and >= 0."""
    stdev = _require_finite(stdev, name)
    if stdev < 0:
        raise ValueError(f"{name} must be non-negative")
    return stdev


def _expected_utility(
    outcomes, probabilities, current_wealth, entry_price, risk_aversion
):
    final_wealth = current_wealth - entry_price + outcomes
    utilities = crra_utility(final_wealth, risk_aversion)
    return np.sum(probabilities * utilities)


def expected_utility(
    outcomes, probabilities, current_wealth, entry_price, risk_aversion=1.0
):
    """
    Calculate expected utility of a gamble.

    Parameters
    ----------
    outcomes : array-like
        The possible payoffs from the gamble
    probabilities : array-like
        The probability of each outcome (must sum to no more than 1, within
        ``PROBABILITY_SUM_TOLERANCE``). Any omitted mass is treated as a
        zero-payout outcome.
    current_wealth : float
        Current wealth before the gamble
    entry_price : float
        Price to pay to participate in the gamble
    risk_aversion : float, default=1.0
        Coefficient of relative risk aversion (γ)

    Returns
    -------
    float
        The expected utility from participating in the gamble
    """
    outcomes, probabilities = _normalize_gamble(outcomes, probabilities)
    return _expected_utility(
        outcomes, probabilities, current_wealth, entry_price, risk_aversion
    )


def find_indifference_price(
    outcomes,
    probabilities,
    current_wealth,
    risk_aversion=1.0,
    tolerance=0.01,
    max_search_fraction=0.5,
):
    """
    Find maximum price willing to pay for a gamble using binary search.

    This function finds the entry price where the expected utility from
    participating equals the utility of not participating (indifference price).

    Parameters
    ----------
    outcomes : array-like
        The possible payoffs from the gamble
    probabilities : array-like
        The probability of each outcome (must sum to no more than 1, within
        ``PROBABILITY_SUM_TOLERANCE``). Any omitted mass is treated as a
        zero-payout outcome.
    current_wealth : float
        Current wealth before the gamble. Must be finite and greater than 0.
    risk_aversion : float, default=1.0
        Coefficient of relative risk aversion (γ). Must be finite and greater
        than 0.
    tolerance : float, default=0.01
        Convergence tolerance for binary search. Must be finite and greater
        than 0.
    max_search_fraction : float, default=0.5
        Maximum fraction of wealth to consider as upper bound. Must be finite
        and non-negative; values above 1.0 are allowed and search beyond
        current wealth.

    Returns
    -------
    float
        Maximum price willing to pay for the gamble

    Raises
    ------
    ValueError
        If the gamble arrays are malformed, or if any scalar control falls
        outside the ranges documented above.

    Examples
    --------
    >>> # Simple 50/50 bet: win $100 or lose $100
    >>> outcomes = [100, -100]
    >>> probabilities = [0.5, 0.5]
    >>> max_price = find_indifference_price(outcomes, probabilities,
    ...                                      current_wealth=1000, risk_aversion=2.0)
    >>> print(f"Willing to pay: ${max_price:.2f}")
    """
    outcomes, probabilities = _normalize_gamble(outcomes, probabilities)
    _validate_entry_price_scalars(
        current_wealth, tolerance, max_search_fraction, risk_aversion=risk_aversion
    )

    # Current utility without participating
    current_utility = crra_utility(current_wealth, risk_aversion)

    # Binary search bounds
    low = 0.0
    high = current_wealth * max_search_fraction

    while high - low > tolerance:
        mid = (low + high) / 2

        # Calculate expected utility at this price
        exp_util = _expected_utility(
            outcomes, probabilities, current_wealth, mid, risk_aversion
        )

        if exp_util > current_utility:
            # Willing to pay more
            low = mid
        else:
            # Paying too much
            high = mid

    return (low + high) / 2
