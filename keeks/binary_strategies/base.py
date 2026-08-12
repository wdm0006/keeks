import abc

from keeks.utils import _require_finite

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
            The transaction cost as a fraction of each unit staked, by default 0.
            This is a per-unit *fractional* cost that enters the sizing formulas
            alongside ``payoff`` and ``loss`` (Kelly, for instance, computes
            ``payoff - transaction_cost``), so ``0.01`` means one percent of the
            stake and the fee it represents grows with the bet.

            The simulators in ``keeks.simulators`` take a near-identically named
            ``transaction_costs`` (plural) that is an *absolute* bankroll amount
            charged once per settled bet, independent of stake size. The two are
            different units: passing the same number to both models two very
            different costs.

        Raises
        ------
        ValueError
            If any of payoff, loss or transaction_cost is not a finite number, if
            payoff is not positive, loss is negative, or loss + transaction_cost
            is not positive.
        """
        payoff = _require_finite(payoff, "Payoff")
        if payoff <= 0:
            raise ValueError("Payoff must be greater than 0")
        loss = _require_finite(loss, "Loss")
        if loss < 0:
            raise ValueError("Loss must be non-negative")
        transaction_cost = _require_finite(transaction_cost, "Transaction cost")
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
            The maximum safe bet size as a proportion of bankroll. Zero when
            there is nothing left to stake (``current_bankroll <= 0``).

        Notes
        -----
        The maximum stake that cannot drive the bankroll negative is
        ``current_bankroll / (loss + transaction_cost)``; expressed as a
        proportion of the bankroll the bankroll term cancels, so for a positive
        bankroll this is exactly ``min(1.0, 1 / (loss + transaction_cost))``.
        A non-positive bankroll has no safe stake at all, so the answer there is
        ``0.0``.
        """
        if current_bankroll <= 0:
            return 0.0
        return min(1.0, 1.0 / (self.loss + self.transaction_cost))

    def calculate_max_entry_price(
        self,
        outcomes,
        probabilities,
        current_wealth,
        tolerance=0.01,
        max_search_fraction=0.5,
    ):
        """
        Calculate maximum price willing to pay for a one-time gamble.

        This method is only implemented for strategies derived from utility theory
        (e.g., KellyCriterion, MertonShare). For heuristic strategies without
        underlying utility functions, this method raises NotImplementedError.

        Parameters
        ----------
        outcomes : array-like
            The possible payoffs from the gamble
        probabilities : array-like
            The probability of each outcome (must sum to ≤ 1)
        current_wealth : float
            Current wealth before the gamble. Must be finite and greater than 0.
        tolerance : float, default=0.01
            Convergence tolerance for binary search. Must be finite and greater
            than 0.
        max_search_fraction : float, default=0.5
            Maximum fraction of wealth to consider as upper bound. Must be
            finite and non-negative; values above 1.0 are allowed.

        Returns
        -------
        float
            Maximum price willing to pay for the gamble

        Raises
        ------
        NotImplementedError
            If the strategy does not have a utility-theoretic basis for
            calculating indifference prices.
        ValueError
            Raised by every shipped implementation if the gamble arrays are
            malformed or if any scalar control falls outside the ranges
            documented above. Overrides should validate with
            ``keeks.utils._validate_entry_price_scalars`` before doing work or
            mutating internal state.

        Warns
        -----
        RuntimeWarning
            Emitted by the utility-based implementations, which delegate to
            ``keeks.utils.find_indifference_price``, when the gamble is still
            worth buying at ``current_wealth * max_search_fraction``. The
            returned price is then that bound rather than a solved indifference
            price. See ``find_indifference_price`` for details.

        Notes
        -----
        This method answers: "What's the maximum I'd pay to participate in this
        one-time gamble?" It's fundamentally different from evaluate(), which
        answers: "How much should I bet in a repeated betting scenario?"

        Only strategies with underlying utility functions can meaningfully answer
        the entry price question. Heuristic strategies (e.g., FixedFraction, CPPI)
        don't have utility functions to derive indifference prices from.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support one-time entry price "
            "calculation. This method is only available for utility-based strategies "
            "like KellyCriterion and MertonShare.\n\n"
            "Heuristic strategies (FixedFraction, CPPI, Dynamic, etc.) don't have "
            "underlying utility functions to derive indifference prices from."
        )

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
