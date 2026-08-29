import math
import operator
import warnings

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
    if np.any(wealth <= 0):
        if np.isscalar(wealth):
            return -np.inf
        # Both arms of an ``np.where`` are evaluated, so masking afterwards would
        # still apply the log/power to the entries this branch exists to exclude
        # and warn once per one. Evaluate only where the utility is defined.
        wealth = np.asarray(wealth, dtype=float)
        defined = ~(wealth <= 0)
        utility = np.full(wealth.shape, -np.inf)
        if risk_aversion == 1.0:
            np.log(wealth, out=utility, where=defined)
        else:
            exponent = 1 - risk_aversion
            np.power(wealth, exponent, out=utility, where=defined)
            np.divide(utility, exponent, out=utility, where=defined)
        return utility

    if risk_aversion == 1.0:
        return np.log(wealth)
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
    current_wealth,
    tolerance=_UNSET,
    max_search_fraction=_UNSET,
    risk_aversion=_UNSET,
    entry_price=_UNSET,
):
    """
    Validate the scalar controls shared by every entry-price calculation.

    Every control except ``current_wealth`` is optional, so a caller supplies
    only the ones it has and each rule lives in one place. When supplied:
    ``current_wealth`` and ``tolerance`` must be finite and positive,
    ``max_search_fraction`` must be finite and nonnegative (values above 1.0 are
    allowed), ``risk_aversion`` must be finite and positive, and ``entry_price``
    must be finite — its sign is unconstrained, since a negative price models
    being paid to take the gamble.

    Raises
    ------
    ValueError
        If any supplied control is outside its accepted range.
    """
    if _require_finite(current_wealth, "Current wealth") <= 0:
        raise ValueError("Current wealth must be greater than 0")
    if tolerance is not _UNSET and _require_finite(tolerance, "Tolerance") <= 0:
        raise ValueError("Tolerance must be greater than 0")
    if (
        max_search_fraction is not _UNSET
        and _require_finite(max_search_fraction, "Maximum search fraction") < 0
    ):
        raise ValueError("Maximum search fraction must be non-negative")
    if (
        risk_aversion is not _UNSET
        and _require_finite(risk_aversion, "Risk aversion") <= 0
    ):
        raise ValueError("Risk aversion must be greater than 0")
    if entry_price is not _UNSET:
        _require_finite(entry_price, "Entry price")


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


def _validate_simulator_seed(seed):
    """Validate an optional simulator seed, which must be a nonnegative integer."""
    if seed is None:
        return None
    try:
        seed = operator.index(seed)
    except TypeError as exc:
        raise ValueError("Seed must be a nonnegative integer or None") from exc
    if seed < 0:
        raise ValueError("Seed must be a nonnegative integer or None")
    return seed


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


def _validate_strategy_odds(strategy, payoff, loss):
    """
    Reject a strategy whose odds contradict a simulator's settlement odds.

    A simulator sizes every bet through ``strategy.evaluate`` but settles it with
    its own ``payoff`` and ``loss``, so the two models have to agree for the run
    to describe anything. Only :class:`keeks.binary_strategies.base.BaseStrategy`
    instances are checked; a duck-typed strategy needs no ``payoff``/``loss`` at
    all and its compatibility stays the caller's responsibility.

    The strategies' fractional ``transaction_cost`` and the simulators' flat
    ``transaction_costs`` fee are deliberately different units and are never
    compared.

    Raises
    ------
    ValueError
        If the strategy's ``payoff`` or ``loss`` differs from the simulator's.
    """
    # Deferred like the strategy modules' own keeks.utils imports, since the
    # two packages reference each other.
    from keeks.binary_strategies.base import BaseStrategy

    if not isinstance(strategy, BaseStrategy):
        return

    for name, strategy_value, simulator_value in (
        ("payoff", strategy.payoff, payoff),
        ("loss", strategy.loss, loss),
    ):
        if strategy_value != simulator_value:
            raise ValueError(
                f"Strategy {name} ({strategy_value!r}) does not match simulator "
                f"{name} ({simulator_value!r}); the strategy sizes each bet with "
                "its own odds while the simulator settles with the simulator's, "
                "so the two must agree."
            )


def _validate_stake_fraction(value):
    """Coerce a strategy result to a finite float within ``[0, 1]``."""
    value = _require_finite(value, "Strategy stake fraction")
    if not 0 <= value <= 1:
        raise ValueError("Strategy stake fraction must be between 0 and 1")
    return value


def _update_strategy_bankroll(strategy, current_bankroll):
    """Update a strategy's bankroll state when it exposes a callable hook."""
    update_bankroll = getattr(strategy, "update_bankroll", None)
    if callable(update_bankroll):
        update_bankroll(current_bankroll)


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
        Current wealth before the gamble. Must be finite and greater than 0.
    entry_price : float
        Price to pay to participate in the gamble. Must be finite; its sign is
        unconstrained, because a negative price models being paid to take the
        gamble.
    risk_aversion : float, default=1.0
        Coefficient of relative risk aversion (γ). Must be finite and greater
        than 0.

    Returns
    -------
    float
        The expected utility from participating in the gamble

    Raises
    ------
    ValueError
        If the gamble arrays are malformed, or if any scalar argument falls
        outside the ranges documented above.
    """
    outcomes, probabilities = _normalize_gamble(outcomes, probabilities)
    _validate_entry_price_scalars(
        current_wealth, risk_aversion=risk_aversion, entry_price=entry_price
    )
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

    Warns
    -----
    RuntimeWarning
        If the gamble is still worth buying at the top of the search range,
        ``current_wealth * max_search_fraction``. The search cannot look past
        that bound, so the returned price is the bound itself rather than a
        solved indifference price, and the true price is at or above it. Raise
        ``max_search_fraction`` to search further.

    Examples
    --------
    >>> # A favourable gamble: 60% chance to win $200, 40% chance to lose $100
    >>> outcomes = [200, -100]
    >>> probabilities = [0.6, 0.4]
    >>> max_price = find_indifference_price(outcomes, probabilities,
    ...                                      current_wealth=1000, risk_aversion=2.0)
    >>> print(f"Willing to pay: ${max_price:.2f}")
    Willing to pay: $57.56
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

    # The search can only report a price inside [0, high]. If the gamble is
    # still worth buying at ``high``, every iteration pushes ``low`` up and the
    # returned price is the bound rather than a solution.
    if (
        _expected_utility(outcomes, probabilities, current_wealth, high, risk_aversion)
        > current_utility
    ):
        warnings.warn(
            f"find_indifference_price saturated at its search bound ({high} = "
            f"current_wealth * max_search_fraction={max_search_fraction}); the "
            "true indifference price is at or above this value. Raise "
            "max_search_fraction to search further.",
            RuntimeWarning,
            stacklevel=2,
        )

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
