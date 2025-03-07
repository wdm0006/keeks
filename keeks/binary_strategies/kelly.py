from keeks.binary_strategies.base import BaseStrategy

__author__ = "willmcginnis"


class KellyCriterion(BaseStrategy):
    """
    Implementation of the Kelly Criterion for binary betting.

    The Kelly Criterion is a formula for bet sizing that maximizes the expected
    logarithm of wealth. It calculates the optimal fraction of the bankroll to
    bet based on the probability of winning and the payoff odds.

    Parameters
    ----------
    payoff : float
        The amount won per unit bet on a successful outcome.
    loss : float
        The amount lost per unit bet on an unsuccessful outcome.
    transaction_cost : float
        The fixed cost per transaction, regardless of outcome.
    """

    def __init__(self, payoff, loss, transaction_cost):
        self.payoff = payoff
        self.loss = loss
        self.transaction_cost = transaction_cost

    def evaluate(self, probability):
        """
        Calculate the optimal bet size using the Kelly Criterion.

        The Kelly formula for binary outcomes:
        f* = (bp - q) / b
        where:
        - b = net odds received on the wager (payoff/loss)
        - p = probability of winning
        - q = probability of losing (1-p)

        The formula is adjusted to account for transaction costs.

        Parameters
        ----------
        probability : float
            The probability of a successful outcome, typically between 0 and 1.

        Returns
        -------
        float
            The optimal proportion of the bankroll to bet, adjusted for transaction costs.
            Returns a negative value if the expected value is negative.
        """

        b = self.payoff / self.loss
        p = probability
        q = 1 - p

        expected_kelly = (b * p - q) / b

        # Adjust for transaction costs
        if expected_kelly > 0:
            expected_kelly -= self.transaction_cost / self.payoff

        return expected_kelly


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
        self.payoff = payoff
        self.loss = loss
        self.transaction_cost = transaction_cost
        self.fraction = fraction

    def evaluate(self, probability):
        """
        Calculate the fractional Kelly bet size.

        Parameters
        ----------
        probability : float
            The probability of a successful outcome, typically between 0 and 1.

        Returns
        -------
        float
            The optimal proportion of the bankroll to bet, multiplied by the fraction parameter.
        """
        kelly = KellyCriterion(self.payoff, self.loss, self.transaction_cost)
        return self.fraction * kelly.evaluate(probability)
