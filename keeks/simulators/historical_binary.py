import warnings
from collections.abc import Sequence
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np

from keeks.binary_strategies.base import BaseStrategy
from keeks.utils import (
    RuinError,
    _validate_simulator_controls,
    _validate_simulator_probability,
    _validate_stake_fraction,
    _validate_strategy_odds,
)

if TYPE_CHECKING:
    from keeks.bankroll import BankRoll


def _validate_outcome(outcome: object, index: int) -> bool:
    if isinstance(outcome, bool | np.bool_):
        return bool(outcome)
    if isinstance(outcome, Integral) and outcome in (0, 1):
        return bool(outcome)
    raise ValueError(f"Outcome at index {index} must be a bool or 0/1, got {outcome!r}")


class HistoricalBinarySimulator:
    """
    Replay a recorded sequence of binary bets through a strategy.

    Each trial uses the recorded probability and settles on the recorded
    outcome, so the run answers "what would this strategy have done with my
    bets?". Settlement, validation, hooks and early-stop behaviour match
    :class:`~keeks.simulators.repeated_binary.RepeatedBinarySimulator`. No
    random numbers are drawn, so there is no ``seed``.

    Parameters
    ----------
    payoff : float
        The amount won per unit bet on a successful outcome.
    loss : float
        The amount lost per unit bet on an unsuccessful outcome.
    fee_per_bet : float
        The flat fee, in currency, charged once per settled bet.
    probabilities : sequence of float
        The probability estimate for each recorded bet, each in ``[0, 1]``.
    outcomes : sequence of bool or int
        Whether each recorded bet won: ``True``/``False`` or ``1``/``0``.
        Must be the same length as ``probabilities``.

    Attributes
    ----------
    trials : int
        The number of recorded bets, ``len(probabilities)``.

    Raises
    ------
    ValueError
        If the sequences differ in length, a probability is not finite within
        ``[0, 1]``, an outcome is not boolean-like, or the odds controls are
        invalid.
    """

    def __init__(
        self,
        payoff: float,
        loss: float,
        fee_per_bet: float,
        probabilities: Sequence[float],
        outcomes: Sequence[bool | int],
    ) -> None:
        """Validate the odds controls and the recorded bets."""
        for name, values in (("probabilities", probabilities), ("outcomes", outcomes)):
            if isinstance(values, str | bytes) or not hasattr(values, "__len__"):
                raise ValueError(f"{name.capitalize()} must be a sequence")
        if len(probabilities) != len(outcomes):
            raise ValueError(
                "Probabilities and outcomes must have the same length, got "
                f"{len(probabilities)} and {len(outcomes)}"
            )
        (
            self.payoff,
            self.loss,
            self.fee_per_bet,
            self.trials,
        ) = _validate_simulator_controls(payoff, loss, fee_per_bet, len(probabilities))
        self.probabilities = tuple(
            _validate_simulator_probability(p, f"Probability at index {i}")
            for i, p in enumerate(probabilities)
        )
        self.outcomes = tuple(_validate_outcome(o, i) for i, o in enumerate(outcomes))

    def evaluate_strategy(self, strategy: BaseStrategy, bankroll: "BankRoll") -> None:
        """
        Replay the recorded bets through ``strategy``, updating ``bankroll`` in place.

        The run stops early if the bankroll is depleted or a settlement raises
        :class:`~keeks.utils.RuinError` (reported as a warning).

        Parameters
        ----------
        strategy : BaseStrategy
            The betting strategy to evaluate.
        bankroll : BankRoll
            The bankroll to use for the simulation.

        Raises
        ------
        ValueError
            If the strategy's ``payoff`` or ``loss`` differs from this
            simulator's, or it returns an invalid stake fraction.
        """
        _validate_strategy_odds(strategy, self.payoff, self.loss)

        update_bankroll = getattr(strategy, "update_bankroll", None)
        if not callable(update_bankroll):
            update_bankroll = None
        record_settlement = getattr(strategy, "record_settlement", None)
        if not callable(record_settlement):
            record_settlement = None

        for probability, won in zip(self.probabilities, self.outcomes, strict=True):
            total_funds = bankroll.total_funds
            if total_funds <= 0:
                break

            if update_bankroll is not None:
                update_bankroll(total_funds)

            proportion = _validate_stake_fraction(
                strategy.evaluate(probability, total_funds)
            )

            if proportion > 0:
                bet_amount = bankroll.bettable_funds * proportion
                try:
                    if won:
                        amount = (self.payoff * bet_amount) - self.fee_per_bet
                        if amount >= 0:
                            bankroll.deposit(amount)
                        else:
                            bankroll.withdraw(abs(amount))
                        realized_return = amount / total_funds
                    else:
                        amount = (self.loss * bet_amount) + self.fee_per_bet
                        bankroll.withdraw(amount)
                        realized_return = -amount / total_funds
                except RuinError as exc:
                    warnings.warn(f"Simulation stopped early: {exc}", stacklevel=2)
                    break

                if record_settlement is not None:
                    record_settlement((won,), (realized_return,))
