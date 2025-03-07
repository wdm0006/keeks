from keeks.binary_strategies.base import BaseStrategy

__author__ = "willmcginnis"


class NaiveStrategy(BaseStrategy):
    """
    A simple binary betting strategy that bets the full amount when expected value is positive.

    This strategy uses a naive approach where it bets the maximum allowed amount
    if the expected value is positive, and nothing if the expected value is negative.
    It can also bet against an outcome if the inverse bet has positive expected value.

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
        """
        Initialize the NaiveStrategy with payoff, loss, and transaction cost parameters.

        Parameters
        ----------
        payoff : float
            The amount won per unit bet on a successful outcome.
        loss : float
            The amount lost per unit bet on an unsuccessful outcome.
        transaction_cost : float
            The fixed cost per transaction, regardless of outcome.
        """
        self.payoff = payoff
        self.loss = loss
        self.transaction_cost = transaction_cost

    def evaluate(self, probability):
        """
        Evaluate whether to bet based on expected value calculation.

        Calculates the expected value of the bet and returns:
        - 1.0 if betting on the outcome has positive expected value
        - -1.0 if betting against the outcome has positive expected value
        - 0.0 if neither bet has positive expected value

        Parameters
        ----------
        probability : float
            The probability of a successful outcome, typically between 0 and 1.

        Returns
        -------
        float
            Either 1.0 (bet full amount), -1.0 (bet against), or 0.0 (no bet).
        """
        expected_value = (
            (self.payoff * probability)
            - (self.loss * (1 - probability))
            - self.transaction_cost
        )
        expected_value_neg = (
            (self.payoff * (1 - probability))
            - (self.loss * probability)
            - self.transaction_cost
        )

        if expected_value > 0:
            return 1.0
        elif expected_value_neg > 0:
            return -1.0
        else:
            return 0.0
