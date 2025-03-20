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
