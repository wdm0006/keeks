import warnings

import numpy as np

from keeks.binary_strategies.base import BaseStrategy
from keeks.utils import _require_finite, _validate_probability, find_indifference_price

__author__ = "willmcginnis"


class KellyCriterion(BaseStrategy):
    """
    Implementation of the Kelly Criterion for binary betting.

    The Kelly Criterion is a mathematical formula that determines the optimal
    size of a series of bets to maximize long-term growth rate.

    By default the gate is edge-aware: a bet the Kelly formula itself sizes
    positively is always sized, and a bet with no positive edge returns 0.0.
    An optional ``min_probability`` longshot gate can additionally refuse
    low-probability bets - loudly, when the refused bet is one the formula
    would have sized.

    Parameters
    ----------
    payoff : float
        The amount won per unit bet on a successful outcome: the *net* win,
        excluding the stake's return - decimal odds minus one, so even money
        is ``1.0``. Unlike here, ``BinaryBetsModel`` and the multi-outcome
        layer read ``payoff`` as the decimal odds themselves.
    loss : float
        The amount lost per unit bet on an unsuccessful outcome.
    transaction_cost_rate : float
        The transaction cost as a fraction of each unit staked (per-unit, not a fixed per-transaction amount).
    min_probability : float or None, optional
        An optional longshot gate: when set, bets with a win probability
        below it return 0.0 even when the Kelly fraction is positive, and a
        ``UserWarning`` names the suppressed fraction so the refusal is
        never silent. The default (``None``) sizes on edge alone: with
        ``payoff=10``, ``loss=1`` and probability 0.3, the true Kelly
        fraction of about 0.23 is placed instead of being zeroed.

    Examples
    --------
    >>> strategy = KellyCriterion(payoff=10, loss=1, transaction_cost_rate=0)
    >>> # Edge-aware by default: the formula sizes this longshot at 0.23.
    >>> round(strategy.evaluate(0.3, 1000.0), 4)
    0.23
    >>> # A bet with no positive edge refuses itself.
    >>> round(strategy.evaluate(0.05, 1000.0), 4)
    0
    """

    def __init__(
        self,
        payoff: float,
        loss: float,
        transaction_cost_rate: float,
        min_probability: float | None = None,
    ) -> None:
        """
        Initialize the Kelly Criterion strategy.

        Parameters
        ----------
        payoff : float
            The amount won per unit bet on a successful outcome: the *net*
            win, excluding the stake's return - decimal odds minus one, so
            even money is ``1.0``. Unlike here, ``BinaryBetsModel`` and the
            multi-outcome layer read ``payoff`` as the decimal odds itself.
        loss : float
            The amount lost per unit bet on an unsuccessful outcome.
        transaction_cost_rate : float
            The transaction cost as a fraction of each unit staked (per-unit, not a fixed per-transaction amount).
        min_probability : float or None, optional
            An optional longshot gate: when set, bets with a win probability
            below it return 0.0 even when the Kelly fraction is positive, and
            a ``UserWarning`` names the suppressed fraction. ``None`` (the
            default) sizes on edge alone.
        """
        if min_probability is not None and not 0 <= min_probability <= 1:
            raise ValueError("Minimum probability must be between 0 and 1")

        super().__init__(payoff, loss, transaction_cost_rate)
        self.min_probability = min_probability

    def evaluate(self, probability: float, current_bankroll: float) -> float:
        """
        Calculate the optimal Kelly bet size.

        The Kelly Criterion formula is:
        f* = p / l - q / a

        where:
        f* = fraction of current bankroll to bet
        a = payoff after transaction costs
        l = loss including transaction costs
        p = probability of winning
        q = probability of losing (1 - p)

        Unlike the classic formula, which assumes the entire stake is lost, this
        formula explicitly accounts for the loss multiplier used by this library.

        The gate is edge-aware: a bet the formula sizes positively is never
        silently zeroed. When ``min_probability`` is set and the probability
        falls below it, a formula-positive bet is refused with a warning
        naming the suppressed fraction; a formula-negative bet refuses
        silently, because the formula itself declines it.

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
        probability = _validate_probability(probability)
        current_bankroll = _require_finite(current_bankroll, "Current bankroll")

        # Calculate probability of losing
        q = 1 - probability

        # Calculate Kelly fraction with transaction costs incorporated
        # Adjust payoff and loss for transaction costs
        adjusted_payoff = self.payoff - self.transaction_cost_rate
        adjusted_loss = self.loss + self.transaction_cost_rate

        # Recalculate net odds with transaction costs
        if adjusted_payoff <= 0 or adjusted_loss <= 0:
            return 0.0  # If transaction costs make the bet unprofitable

        # Calculate Kelly fraction with adjusted payoff and loss
        kelly_fraction = probability / adjusted_loss - q / adjusted_payoff

        if self.min_probability is not None and probability < self.min_probability:
            if kelly_fraction > 0:
                warnings.warn(
                    f"min_probability gate: probability {probability} is below "
                    f"min_probability={self.min_probability}, but the Kelly "
                    f"formula sizes this bet positively at {kelly_fraction:.4f} "
                    "of bankroll; the bet is refused anyway. Pass "
                    "min_probability=None to size every positive-edge bet.",
                    stacklevel=2,
                )
            return 0.0

        # Ensure we never bet more than would result in negative bankroll
        return min(max(0, kelly_fraction), self.get_max_safe_bet(current_bankroll))

    def calculate_max_entry_price(
        self,
        outcomes: np.typing.ArrayLike,
        probabilities: np.typing.ArrayLike,
        current_wealth: float,
        tolerance: float = 0.01,
        max_search_fraction: float = 0.5,
    ) -> float:
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
        ValueError
            If the gamble arrays are malformed, or if any scalar control falls
            outside the ranges documented above.

        Notes
        -----
        Kelly Criterion maximizes E[log(wealth)], which corresponds to
        CRRA utility with risk aversion γ=1.0 (log utility).
        """
        return find_indifference_price(
            outcomes=outcomes,
            probabilities=probabilities,
            current_wealth=current_wealth,
            risk_aversion=1.0,  # Kelly uses log utility (γ=1)
            tolerance=tolerance,
            max_search_fraction=max_search_fraction,
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
    transaction_cost_rate : float
        The transaction cost as a fraction of each unit staked (per-unit, not a fixed per-transaction amount).
    fraction : float
        The fraction of the full Kelly bet to use (typically between 0 and 1).
    """

    def __init__(
        self, payoff: float, loss: float, transaction_cost_rate: float, fraction: float
    ) -> None:
        if not 0 <= fraction <= 1:
            raise ValueError("Fraction must be between 0 and 1")

        super().__init__(payoff, loss, transaction_cost_rate)
        self.fraction = fraction
        # Constructed once: the inner Kelly strategy depends only on
        # constructor arguments, and strategies are immutable after init.
        self._kelly = KellyCriterion(payoff, loss, transaction_cost_rate)

    def evaluate(self, probability: float, current_bankroll: float) -> float:
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
        # The inner Kelly evaluate validates both arguments, so the wrapper
        # only rescales its result.
        return self.fraction * self._kelly.evaluate(probability, current_bankroll)

    def calculate_max_entry_price(
        self,
        outcomes: np.typing.ArrayLike,
        probabilities: np.typing.ArrayLike,
        current_wealth: float,
        tolerance: float = 0.01,
        max_search_fraction: float = 0.5,
    ) -> float:
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
        ValueError
            If the gamble arrays are malformed, or if any scalar control falls
            outside the ranges documented above.

        Notes
        -----
        Fractional Kelly (e.g., Half Kelly) is more conservative than full Kelly.
        We scale the entry price by the fraction, which is a reasonable
        approximation though not derived from first principles.
        """
        # Get full Kelly price
        kelly_price = self._kelly.calculate_max_entry_price(
            outcomes, probabilities, current_wealth, tolerance, max_search_fraction
        )

        # Scale by the fraction (more conservative)
        return self.fraction * kelly_price


class DrawdownAdjustedKelly(BaseStrategy):
    """
    Implementation of the Drawdown-Adjusted Kelly Criterion for binary betting.

    This strategy adjusts the Kelly bet size based on the maximum per-bet
    transaction loss it tolerates.
    It provides a more conservative approach by reducing the bet size to minimize
    the risk of large drawdowns.

    Parameters
    ----------
    payoff : float
        The amount won per unit bet on a successful outcome.
    loss : float
        The amount lost per unit bet on an unsuccessful outcome.
    transaction_cost_rate : float
        The transaction cost as a fraction of each unit staked (per-unit, not a fixed per-transaction amount).
    max_transaction_loss : float
        The maximum transaction loss tolerated per bet, as a fraction of the
        bankroll (0 to 1, exclusive) - a per-removal cap the sizing scale
        keys off, not peak-to-trough drawdown monitoring.
    """

    def __init__(
        self,
        payoff: float,
        loss: float,
        transaction_cost_rate: float,
        max_transaction_loss: float = 0.2,
    ) -> None:
        """
        Initialize the DrawdownAdjustedKelly strategy.

        Parameters
        ----------
        payoff : float
            The amount won per unit bet on a successful outcome.
        loss : float
            The amount lost per unit bet on an unsuccessful outcome.
        transaction_cost_rate : float
            The transaction cost as a fraction of each unit staked (per-unit, not a fixed per-transaction amount).
        max_transaction_loss : float, default=0.2
            The maximum transaction loss tolerated per bet, as a fraction of
            the bankroll (0 to 1, exclusive) - a per-removal cap, not
            peak-to-trough drawdown monitoring.

        Raises
        ------
        ValueError
            If max_transaction_loss is not between 0 and 1 (exclusive).
        """
        super().__init__(payoff, loss, transaction_cost_rate)

        if not 0 < max_transaction_loss < 1:
            raise ValueError(
                "Maximum transaction loss must be between 0 and 1 (exclusive)"
            )

        self.max_transaction_loss = max_transaction_loss
        # Constructed once: the inner Kelly strategy depends only on
        # constructor arguments, and strategies are immutable after init.
        self._kelly = KellyCriterion(payoff, loss, transaction_cost_rate)
        self._drawdown_factor = min(1.0, max_transaction_loss / 0.5)

    def evaluate(self, probability: float, current_bankroll: float) -> float:
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
        # The inner Kelly evaluate validates both arguments and caps at the
        # max-safe fraction; the wrapper only rescales its result. Full Kelly
        # has an expected drawdown of around 50%, so scale by
        # max_transaction_loss / 0.5.
        full_kelly = self._kelly.evaluate(probability, current_bankroll)

        # Apply the drawdown adjustment
        adjusted_kelly = self._drawdown_factor * full_kelly

        # Ensure we never bet more than would result in negative bankroll
        return min(adjusted_kelly, self.get_max_safe_bet(current_bankroll))

    def calculate_max_entry_price(
        self,
        outcomes: np.typing.ArrayLike,
        probabilities: np.typing.ArrayLike,
        current_wealth: float,
        tolerance: float = 0.01,
        max_search_fraction: float = 0.5,
    ) -> float:
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
        ValueError
            If the gamble arrays are malformed, or if any scalar control falls
            outside the ranges documented above.

        Notes
        -----
        Uses the same drawdown adjustment factor as the betting strategy:
        drawdown_factor = min(1.0, max_transaction_loss / 0.5)
        """
        # Get full Kelly price
        kelly_price = self._kelly.calculate_max_entry_price(
            outcomes, probabilities, current_wealth, tolerance, max_search_fraction
        )

        # Apply the same drawdown adjustment used in evaluate()
        return self._drawdown_factor * kelly_price
