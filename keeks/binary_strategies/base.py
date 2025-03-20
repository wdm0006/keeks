import abc

__author__ = "willmcginnis"


class BaseStrategy(abc.ABC):
    """
    Abstract base class for all binary betting strategies.

    This class defines the interface that all binary betting strategies must implement.
    Concrete strategy implementations should inherit from this class and implement
    the evaluate method.
    """

    def __init__(self, payoff: float, loss: float, transaction_cost: float = 0):
        """
        Initialize the strategy.

        Parameters
        ----------
        payoff : float
            The payoff multiplier for winning.
        loss : float
            The loss multiplier for losing.
        transaction_cost : float, optional
            The transaction cost as a fraction of the bet, by default 0.

        Raises
        ------
        ValueError
            If payoff is not positive, loss is negative, or loss + transaction_cost is not positive.
        """
        if payoff <= 0:
            raise ValueError("Payoff must be greater than 0")
        if loss < 0:
            raise ValueError("Loss must be non-negative")
        if transaction_cost < 0:
            raise ValueError("Transaction cost must be non-negative")
        if loss + transaction_cost <= 0:
            raise ValueError(
                "Total cost (loss + transaction_cost) must be greater than 0"
            )

        self.payoff = payoff
        self.loss = loss
        self.transaction_cost = transaction_cost

    def get_max_safe_bet(self, current_bankroll: float) -> float:
        """
        Calculate the maximum safe bet size based on current bankroll.

        Parameters
        ----------
        current_bankroll : float
            The current bankroll to use for calculations.

        Returns
        -------
        float
            The maximum safe bet size as a proportion of bankroll.
        """
        # Calculate maximum bet that won't result in negative bankroll
        # after accounting for loss multiplier and transaction costs
        max_bet = current_bankroll / (self.loss + self.transaction_cost)
        return min(1.0, max_bet / current_bankroll)

    @abc.abstractmethod
    def evaluate(self, probability: float, current_bankroll: float) -> float:
        """
        Evaluate the strategy for a given probability.

        Parameters
        ----------
        probability : float
            The probability of winning.
        current_bankroll : float
            The current bankroll to use for calculations.

        Returns
        -------
        float
            The proportion of the bankroll to bet.
        """
        pass
