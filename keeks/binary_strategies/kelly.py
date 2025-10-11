from keeks.binary_strategies.base import BaseStrategy

__author__ = "willmcginnis"


class KellyCriterion(BaseStrategy):
    """
    Implementation of the Kelly Criterion for binary betting.

    The Kelly Criterion is a mathematical formula that determines the optimal
    size of a series of bets to maximize long-term growth rate.

    Parameters
    ----------
    payoff : float
        The amount won per unit bet on a successful outcome.
    loss : float
        The amount lost per unit bet on an unsuccessful outcome.
    transaction_cost : float
        The fixed cost per transaction, regardless of outcome.
    min_probability : float, default=0.5
        The minimum probability required to place a bet.
    """

    def __init__(self, payoff, loss, transaction_cost, min_probability=0.5):
        """
        Initialize the Kelly Criterion strategy.

        Parameters
        ----------
        payoff : float
            The amount won per unit bet on a successful outcome.
        loss : float
            The amount lost per unit bet on an unsuccessful outcome.
        transaction_cost : float
            The fixed cost per transaction, regardless of outcome.
        min_probability : float, default=0.5
            The minimum probability required to place a bet.
        """
        if not 0 <= min_probability <= 1:
            raise ValueError("Minimum probability must be between 0 and 1")

        super().__init__(payoff, loss, transaction_cost)
        self.min_probability = min_probability

    def evaluate(self, probability, current_bankroll):
        """
        Calculate the optimal Kelly bet size.

        The Kelly Criterion formula is:
        f* = (bp - q) / b

        where:
        f* = fraction of current bankroll to bet
        b = net odds received on the bet (payoff/loss)
        p = probability of winning
        q = probability of losing (1 - p)

        Parameters
        ----------
        probability : float
            The probability of a successful outcome, typically between 0 and 1.
        current_bankroll : float
            The current bankroll amount.

        Returns
        -------
        float
            The optimal proportion of the bankroll to bet.
        """
        if probability < self.min_probability:
            return 0.0

        # Calculate net odds
        self.payoff / self.loss

        # Calculate probability of losing
        q = 1 - probability

        # Calculate Kelly fraction with transaction costs incorporated
        # Adjust payoff and loss for transaction costs
        adjusted_payoff = self.payoff - self.transaction_cost
        adjusted_loss = self.loss + self.transaction_cost

        # Recalculate net odds with transaction costs
        if adjusted_payoff <= 0 or adjusted_loss <= 0:
            return 0.0  # If transaction costs make the bet unprofitable

        adjusted_b = adjusted_payoff / adjusted_loss

        # Calculate Kelly fraction with adjusted odds
        kelly_fraction = (adjusted_b * probability - q) / adjusted_b

        # Ensure we never bet more than would result in negative bankroll
        return min(max(0, kelly_fraction), self.get_max_safe_bet(current_bankroll))

    def calculate_max_entry_price(self, outcomes, probabilities, current_wealth,
                                  tolerance=0.01, max_search_fraction=0.5):
        """
        Calculate maximum price willing to pay for a one-time gamble.

        Kelly Criterion is derived from log utility maximization (CRRA with γ=1.0),
        so this uses log utility to find the indifference price.

        Parameters
        ----------
        outcomes : array-like
            The possible payoffs from the gamble
        probabilities : array-like
            The probability of each outcome (must sum to ≤ 1)
        current_wealth : float
            Current wealth before the gamble
        tolerance : float, default=0.01
            Convergence tolerance for binary search
        max_search_fraction : float, default=0.5
            Maximum fraction of wealth to consider as upper bound

        Returns
        -------
        float
            Maximum price willing to pay for the gamble

        Notes
        -----
        Kelly Criterion maximizes E[log(wealth)], which corresponds to
        CRRA utility with risk aversion γ=1.0 (log utility).
        """
        from keeks.utils import find_indifference_price

        return find_indifference_price(
            outcomes=outcomes,
            probabilities=probabilities,
            current_wealth=current_wealth,
            risk_aversion=1.0,  # Kelly uses log utility (γ=1)
            tolerance=tolerance,
            max_search_fraction=max_search_fraction
        )


class FractionalKellyCriterion(BaseStrategy):
    """
    Implementation of the Fractional Kelly Criterion for binary betting.

    The Fractional Kelly Criterion applies a fraction to the full Kelly bet size,
    which can reduce volatility at the expense of long-term growth rate.

    Parameters
    ----------
    payoff : float
        The amount won per unit bet on a successful outcome.
    loss : float
        The amount lost per unit bet on an unsuccessful outcome.
    transaction_cost : float
        The fixed cost per transaction, regardless of outcome.
    fraction : float
        The fraction of the full Kelly bet to use (typically between 0 and 1).
    """

    def __init__(self, payoff, loss, transaction_cost, fraction):
        if not 0 <= fraction <= 1:
            raise ValueError("Fraction must be between 0 and 1")

        super().__init__(payoff, loss, transaction_cost)
        self.fraction = fraction

    def evaluate(self, probability, current_bankroll):
        """
        Calculate the fractional Kelly bet size.

        Parameters
        ----------
        probability : float
            The probability of a successful outcome, typically between 0 and 1.
        current_bankroll : float
            The current bankroll amount.

        Returns
        -------
        float
            The optimal proportion of the bankroll to bet, multiplied by the fraction parameter.
        """
        kelly = KellyCriterion(self.payoff, self.loss, self.transaction_cost)
        return self.fraction * kelly.evaluate(probability, current_bankroll)

    def calculate_max_entry_price(self, outcomes, probabilities, current_wealth,
                                  tolerance=0.01, max_search_fraction=0.5):
        """
        Calculate maximum price willing to pay for a one-time gamble.

        Fractional Kelly is more conservative than full Kelly, so we scale down
        the entry price proportionally. This assumes the fraction reflects
        increased risk aversion beyond log utility.

        Parameters
        ----------
        outcomes : array-like
            The possible payoffs from the gamble
        probabilities : array-like
            The probability of each outcome (must sum to ≤ 1)
        current_wealth : float
            Current wealth before the gamble
        tolerance : float, default=0.01
            Convergence tolerance for binary search
        max_search_fraction : float, default=0.5
            Maximum fraction of wealth to consider as upper bound

        Returns
        -------
        float
            Maximum price willing to pay for the gamble

        Notes
        -----
        Fractional Kelly (e.g., Half Kelly) is more conservative than full Kelly.
        We scale the entry price by the fraction, which is a reasonable
        approximation though not derived from first principles.
        """
        # Get full Kelly price
        kelly = KellyCriterion(self.payoff, self.loss, self.transaction_cost)
        kelly_price = kelly.calculate_max_entry_price(
            outcomes, probabilities, current_wealth, tolerance, max_search_fraction
        )

        # Scale by the fraction (more conservative)
        return self.fraction * kelly_price


class DrawdownAdjustedKelly(BaseStrategy):
    """
    Implementation of the Drawdown-Adjusted Kelly Criterion for binary betting.

    This strategy adjusts the Kelly bet size based on a maximum acceptable drawdown.
    It provides a more conservative approach by reducing the bet size to minimize
    the risk of large drawdowns.

    Parameters
    ----------
    payoff : float
        The amount won per unit bet on a successful outcome.
    loss : float
        The amount lost per unit bet on an unsuccessful outcome.
    transaction_cost : float
        The fixed cost per transaction, regardless of outcome.
    max_acceptable_drawdown : float
        The maximum acceptable drawdown as a fraction of the bankroll (0 to 1).
    """

    def __init__(self, payoff, loss, transaction_cost, max_acceptable_drawdown=0.2):
        """
        Initialize the DrawdownAdjustedKelly strategy.

        Parameters
        ----------
        payoff : float
            The amount won per unit bet on a successful outcome.
        loss : float
            The amount lost per unit bet on an unsuccessful outcome.
        transaction_cost : float
            The fixed cost per transaction, regardless of outcome.
        max_acceptable_drawdown : float, default=0.2
            The maximum acceptable drawdown as a fraction of the bankroll.

        Raises
        ------
        ValueError
            If max_acceptable_drawdown is not between 0 and 1 (exclusive).
        """
        super().__init__(payoff, loss, transaction_cost)

        if not 0 < max_acceptable_drawdown < 1:
            raise ValueError(
                "Maximum acceptable drawdown must be between 0 and 1 (exclusive)"
            )

        self.max_acceptable_drawdown = max_acceptable_drawdown

    def evaluate(self, probability, current_bankroll):
        """
        Calculate the drawdown-adjusted Kelly bet size.

        The adjustment is an approximation based on the relationship between
        bet size and expected drawdown in repeated betting scenarios.

        Parameters
        ----------
        probability : float
            The probability of a successful outcome, typically between 0 and 1.
        current_bankroll : float
            The current bankroll amount.

        Returns
        -------
        float
            The drawdown-adjusted proportion of the bankroll to bet.
        """
        # Calculate the standard Kelly bet size
        kelly = KellyCriterion(self.payoff, self.loss, self.transaction_cost)
        full_kelly = kelly.evaluate(probability, current_bankroll)

        # Adjust the Kelly fraction based on maximum acceptable drawdown
        # This is a simplified approximation of the relationship
        # Various research suggests specific formulas, but a common
        # conservative approach is to scale Kelly by max drawdown / 0.5
        # (since full Kelly has an expected drawdown of around 50%)
        drawdown_factor = min(1.0, self.max_acceptable_drawdown / 0.5)

        # Apply the drawdown adjustment
        adjusted_kelly = drawdown_factor * full_kelly

        # Ensure we never bet more than would result in negative bankroll
        return min(adjusted_kelly, self.get_max_safe_bet(current_bankroll))

    def calculate_max_entry_price(self, outcomes, probabilities, current_wealth,
                                  tolerance=0.01, max_search_fraction=0.5):
        """
        Calculate maximum price willing to pay for a one-time gamble.

        Drawdown-Adjusted Kelly scales down Kelly based on drawdown tolerance.
        We apply the same scaling factor to the entry price.

        Parameters
        ----------
        outcomes : array-like
            The possible payoffs from the gamble
        probabilities : array-like
            The probability of each outcome (must sum to ≤ 1)
        current_wealth : float
            Current wealth before the gamble
        tolerance : float, default=0.01
            Convergence tolerance for binary search
        max_search_fraction : float, default=0.5
            Maximum fraction of wealth to consider as upper bound

        Returns
        -------
        float
            Maximum price willing to pay for the gamble

        Notes
        -----
        Uses the same drawdown adjustment factor as the betting strategy:
        drawdown_factor = min(1.0, max_acceptable_drawdown / 0.5)
        """
        # Get full Kelly price
        kelly = KellyCriterion(self.payoff, self.loss, self.transaction_cost)
        kelly_price = kelly.calculate_max_entry_price(
            outcomes, probabilities, current_wealth, tolerance, max_search_fraction
        )

        # Apply the same drawdown adjustment used in evaluate()
        drawdown_factor = min(1.0, self.max_acceptable_drawdown / 0.5)

        return drawdown_factor * kelly_price
