import abc
from collections.abc import Sequence

import numpy as np

from keeks.utils import PROBABILITY_SUM_TOLERANCE, _require_finite

__author__ = "willmcginnis"


def _validate_stake_fractions(stakes, leg_count=None):
    """
    Coerce a strategy's stake vector to a tuple of finite floats within ``[0, 1]``.

    The vector analogue of :func:`keeks.utils._validate_stake_fraction`: one
    stake fraction per mutually exclusive leg, every element finite in
    ``[0, 1]``, and the total at most one within ``PROBABILITY_SUM_TOLERANCE``
    — the legs together can never promise more of the bankroll than it holds.

    Parameters
    ----------
    stakes : sequence of float
        One stake fraction per leg. Any sequence is accepted; a bare scalar is
        not a one-dimensional sequence and is rejected.
    leg_count : int, optional
        When given, the vector must carry exactly that many fractions - one
        per leg of the market or portfolio it settles. A length mismatch would
        otherwise silently skip legs or settle stakes on legs that do not
        exist, so simulators pass their leg count and reject the vector.

    Returns
    -------
    tuple of float
        The validated stakes, one per leg, in leg order.

    Raises
    ------
    ValueError
        If the stakes are not a non-empty one-dimensional sequence of finite
        numbers, if any element falls outside ``[0, 1]``, if they sum to
        more than ``1 + PROBABILITY_SUM_TOLERANCE``, or if ``leg_count`` is
        given and the vector's length differs from it.

    Examples
    --------
    >>> _validate_stake_fractions([0.25, 0.25, 0.5])
    (0.25, 0.25, 0.5)
    """
    try:
        stakes = np.asarray(stakes, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Strategy stake fractions must be a finite sequence") from exc
    if stakes.ndim != 1:
        raise ValueError("Strategy stake fractions must be one-dimensional")
    if stakes.size == 0:
        raise ValueError("Strategy stake fractions must be non-empty")
    if leg_count is not None and stakes.size != leg_count:
        raise ValueError(
            f"Strategy must return exactly {leg_count} stake fractions, "
            f"got {stakes.size}"
        )
    if not np.all(np.isfinite(stakes)):
        raise ValueError("Strategy stake fractions must contain only finite values")
    if np.any((stakes < 0) | (stakes > 1)):
        raise ValueError("Strategy stake fractions must be between 0 and 1")
    if stakes.sum() > 1 + PROBABILITY_SUM_TOLERANCE:
        raise ValueError("Strategy stake fractions must sum to no more than one")
    return tuple(stakes.tolist())


class BaseMultiOutcomeStrategy(abc.ABC):
    """
    Abstract base class for all multi-outcome betting strategies.

    This class defines the interface that all multi-outcome betting strategies
    must implement. A multi-outcome strategy sizes stakes on the mutually
    exclusive legs of one market - a 1X2 football match, for instance - in a
    single decision: exactly one leg settles, every losing leg's stake is
    charged ``loss`` plus ``transaction_cost``, and the winning leg pays its
    payoff multiplier times its stake.

    Concrete strategy implementations should inherit from this class and
    implement the evaluate method.
    """

    def __init__(
        self, payoffs: Sequence[float], loss: float, transaction_cost: float = 0
    ):
        """
        Initialize the strategy.

        Parameters
        ----------
        payoffs : sequence of float
            The payoff multiplier for each mutually exclusive leg, in leg
            order. Every payoff must be finite and greater than 0. The legs
            are positional and payoffs are fixed at construction: leg ``i``
            of the probabilities passed to :meth:`evaluate` is settled with
            leg ``i`` of ``payoffs``, and a strategy reprices by fresh
            construction, not by mutating its odds.
        loss : float
            The loss multiplier applied to every losing leg's stake.
        transaction_cost : float, optional
            The transaction cost as a fraction of each unit staked, by default 0.
            This is a per-unit *fractional* cost that enters the sizing formulas
            alongside ``payoffs`` and ``loss``, so ``0.01`` means one percent of
            the stake and the fee it represents grows with the bet.

            The simulators in ``keeks.simulators`` take a near-identically named
            ``transaction_costs`` (plural) that is an *absolute* bankroll amount
            charged once per settled bet, independent of stake size. The two are
            different units: passing the same number to both models two very
            different costs.

        Raises
        ------
        ValueError
            If ``payoffs`` is not a non-empty one-dimensional sequence of
            finite numbers greater than 0, if any of ``loss`` or
            ``transaction_cost`` is not a finite number, if ``loss`` is
            negative, or if ``loss + transaction_cost`` is not positive.
        """
        try:
            payoff_array = np.asarray(payoffs, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Payoffs must be a finite sequence") from exc
        if payoff_array.ndim != 1:
            raise ValueError("Payoffs must be one-dimensional")
        if payoff_array.size == 0:
            raise ValueError("Payoffs must be non-empty")
        if not np.all(np.isfinite(payoff_array)):
            raise ValueError("Payoffs must contain only finite values")
        if np.any(payoff_array <= 0):
            raise ValueError("Payoffs must be greater than 0")

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

        self.payoffs: tuple[float, ...] = tuple(payoff_array.tolist())
        self.loss = loss
        self.transaction_cost = transaction_cost
        # Constructor-only constant: for a positive bankroll the bankroll term
        # in current_bankroll / (loss + transaction_cost) cancels, so the
        # aggregate cap is a fixed fraction (see get_max_safe_total_bet).
        self._max_safe_total_fraction = min(1.0, 1.0 / (loss + transaction_cost))

    def get_max_safe_total_bet(self, current_bankroll: float) -> float:
        """
        Calculate the maximum safe aggregate stake across all legs.

        Parameters
        ----------
        current_bankroll : float
            The current bankroll to use for calculations.

        Returns
        -------
        float
            The maximum safe total stake size as a proportion of bankroll.
            Zero when there is nothing left to stake
            (``current_bankroll <= 0``).

        Raises
        ------
        ValueError
            If ``current_bankroll`` is not finite.

        Notes
        -----
        Under net settlement exactly one leg of the market wins and every
        losing leg's stake is charged ``loss + transaction_cost``, so a total
        stake fraction ``F`` spread across the legs can lose at most
        ``F * (loss + transaction_cost)`` of the bankroll - the worst leg's
        charge being the binding one. Keeping that worst case within the
        bankroll requires ``F <= current_bankroll / (loss + transaction_cost)``;
        expressed as a proportion of the bankroll the bankroll term cancels,
        so for a positive bankroll this is exactly
        ``min(1.0, 1 / (loss + transaction_cost))``. Every leg shares one
        scalar ``loss`` and ``transaction_cost``, so the worst leg's per-leg
        bound is also the aggregate bound - the same value
        ``keeks.binary_strategies.base.BaseStrategy.get_max_safe_bet`` returns
        for a single binary bet with the same charges. A non-positive bankroll
        has no safe stake at all, so the answer there is ``0.0``.
        """
        current_bankroll = _require_finite(current_bankroll, "Current bankroll")
        if current_bankroll <= 0:
            return 0.0
        return self._max_safe_total_fraction

    @abc.abstractmethod
    def evaluate(
        self, probabilities: Sequence[float], current_bankroll: float
    ) -> tuple[float, ...]:
        """
        Evaluate the strategy for a given probability vector.

        Parameters
        ----------
        probabilities : sequence of float
            The probability of each mutually exclusive leg, in leg order. Must
            be a non-empty one-dimensional sequence of finite, nonnegative
            numbers whose sum is at most ``1 + PROBABILITY_SUM_TOLERANCE`` -
            the contract :func:`keeks.utils.normalize_probabilities` enforces.
            Probability mass below one models a void or push outcome on which
            no leg settles.
        current_bankroll : float
            The current bankroll to use for calculations.

        Returns
        -------
        tuple of float
            One stake fraction of the bankroll per leg: ``len(result) ==
            len(probabilities)``, every element finite and within ``[0, 1]``,
            and ``sum(result) <= 1 + PROBABILITY_SUM_TOLERANCE``.
            Implementations accept any sequence input and return a tuple.

        Raises
        ------
        ValueError
            If the probability vector is malformed (empty, non-finite,
            negative, or summing above ``1 + PROBABILITY_SUM_TOLERANCE``), if
            ``current_bankroll`` is not finite, or if the stake vector leaving
            this strategy violates the contract above.
        """
        pass
