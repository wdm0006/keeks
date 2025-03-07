import abc

__author__ = "willmcginnis"


class BaseStrategy(abc.ABC):
    """
    Abstract base class for all binary betting strategies.

    This class defines the interface that all binary betting strategies must implement.
    Concrete strategy implementations should inherit from this class and implement
    the evaluate method.
    """

    @abc.abstractmethod
    def evaluate(self, probability):
        """
        Evaluate the strategy for a given probability of success.

        Parameters
        ----------
        probability : float
            The probability of a successful outcome, typically between 0 and 1.

        Returns
        -------
        float
            The proportion of the bankroll to bet, where:
            - Positive values indicate a bet on success
            - Negative values indicate a bet against success (if supported)
            - Zero indicates no bet
        """
        pass
