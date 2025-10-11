import numpy as np


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
            return -np.inf if np.isscalar(wealth) else np.where(wealth <= 0, -np.inf, np.log(wealth))
        return np.log(wealth)
    else:
        if np.any(wealth <= 0):
            return -np.inf if np.isscalar(wealth) else np.where(wealth <= 0, -np.inf, (wealth ** (1 - risk_aversion)) / (1 - risk_aversion))
        return (wealth ** (1 - risk_aversion)) / (1 - risk_aversion)


def expected_utility(outcomes, probabilities, current_wealth, entry_price, risk_aversion=1.0):
    """
    Calculate expected utility of a gamble.

    Parameters
    ----------
    outcomes : array-like
        The possible payoffs from the gamble
    probabilities : array-like
        The probability of each outcome (must sum to ≤ 1)
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
    outcomes = np.array(outcomes)
    probabilities = np.array(probabilities)

    # Final wealth for each outcome = current wealth - entry price + payout
    final_wealth = current_wealth - entry_price + outcomes

    # Calculate utility for each outcome
    utilities = crra_utility(final_wealth, risk_aversion)

    # Return expected utility
    return np.sum(probabilities * utilities)


def find_indifference_price(outcomes, probabilities, current_wealth, risk_aversion=1.0,
                           tolerance=0.01, max_search_fraction=0.5):
    """
    Find maximum price willing to pay for a gamble using binary search.

    This function finds the entry price where the expected utility from
    participating equals the utility of not participating (indifference price).

    Parameters
    ----------
    outcomes : array-like
        The possible payoffs from the gamble
    probabilities : array-like
        The probability of each outcome (must sum to ≤ 1)
    current_wealth : float
        Current wealth before the gamble
    risk_aversion : float, default=1.0
        Coefficient of relative risk aversion (γ)
    tolerance : float, default=0.01
        Convergence tolerance for binary search
    max_search_fraction : float, default=0.5
        Maximum fraction of wealth to consider as upper bound

    Returns
    -------
    float
        Maximum price willing to pay for the gamble

    Examples
    --------
    >>> # Simple 50/50 bet: win $100 or lose $100
    >>> outcomes = [100, -100]
    >>> probabilities = [0.5, 0.5]
    >>> max_price = find_indifference_price(outcomes, probabilities,
    ...                                      current_wealth=1000, risk_aversion=2.0)
    >>> print(f"Willing to pay: ${max_price:.2f}")
    """
    # Current utility without participating
    current_utility = crra_utility(current_wealth, risk_aversion)

    # Binary search bounds
    low = 0.0
    high = current_wealth * max_search_fraction

    while high - low > tolerance:
        mid = (low + high) / 2

        # Calculate expected utility at this price
        exp_util = expected_utility(outcomes, probabilities, current_wealth,
                                   mid, risk_aversion)

        if exp_util > current_utility:
            # Willing to pay more
            low = mid
        else:
            # Paying too much
            high = mid

    return (low + high) / 2
