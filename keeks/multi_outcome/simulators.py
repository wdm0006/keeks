"""
Repeated-play simulation over mutually exclusive market outcomes.

Generalizes the fixed-probability binary simulator to N-leg markets: one
categorical draw per trial realizes exactly one leg (the probability mass
below one is a void or push on which no leg settles), every leg settles
through the same net-settlement flow the binary simulators use, and a
``record_settlement`` hook reports the full N-ary result of each batch.
"""

import operator

import numpy as np

from keeks.multi_outcome.base import BaseMultiOutcomeStrategy, _validate_stake_fractions
from keeks.utils import (
    RuinError,
    _require_finite,
    _validate_simulator_seed,
    normalize_probabilities,
)

__author__ = "willmcginnis"


def _validate_payoffs(payoffs):
    """
    Validate a simulator's payoff vector and return it as a float tuple.

    Parameters
    ----------
    payoffs : sequence of float
        The payoff multiplier for each mutually exclusive leg, in leg order.

    Returns
    -------
    tuple of float
        The validated payoffs, one per leg, in leg order.

    Raises
    ------
    ValueError
        If ``payoffs`` is not a non-empty one-dimensional sequence of finite
        numbers greater than 0.
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
    return tuple(payoff_array.tolist())


def _validate_strategy_odds(strategy, payoffs, loss):
    """
    Reject a strategy whose odds contradict the simulator's settlement odds.

    The simulator sizes every stake through ``strategy.evaluate`` but settles
    it with its own ``payoffs`` and ``loss``, so the two models have to agree
    for the run to describe anything. Only
    :class:`keeks.multi_outcome.base.BaseMultiOutcomeStrategy` instances are
    checked; a duck-typed strategy needs no ``payoffs``/``loss`` at all and
    its compatibility stays the caller's responsibility.

    The strategy's fractional ``transaction_cost`` and the simulator's flat
    ``transaction_costs`` fee are deliberately different units and are never
    compared.

    Raises
    ------
    ValueError
        If the strategy's ``payoffs`` or ``loss`` differ from the simulator's.
    """
    if not isinstance(strategy, BaseMultiOutcomeStrategy):
        return

    if tuple(strategy.payoffs) != tuple(payoffs):
        raise ValueError(
            f"Strategy payoffs ({strategy.payoffs!r}) does not match simulator "
            f"payoffs ({tuple(payoffs)!r}); the strategy sizes each stake with "
            "its own odds while the simulator settles with the simulator's, "
            "so the two must agree."
        )
    if strategy.loss != loss:
        raise ValueError(
            f"Strategy loss ({strategy.loss!r}) does not match simulator loss "
            f"({loss!r}); the strategy sizes each stake with its own odds "
            "while the simulator settles with the simulator's, so the two "
            "must agree."
        )


class RepeatedMultiOutcomeSimulator:
    """
    Simulator for multi-outcome strategies on a fixed mutually exclusive market.

    Every trial bets on the legs of one market - a 1X2 football match, for
    instance - with the same win probabilities, payoff multipliers, and loss
    multiplier. Exactly one leg realizes per trial: the strategy returns one
    stake fraction per leg, one uniform draw picks the realized leg, the
    realized leg settles as a win, and every other leg the strategy staked
    settles as a loss.

    Parameters
    ----------
    payoffs : sequence of float
        The payoff multiplier for each mutually exclusive leg, in leg order.
        Every payoff must be finite and greater than 0. Leg ``i`` of
        ``probabilities`` is settled with leg ``i`` of ``payoffs``.
    loss : float
        The loss multiplier applied to every losing leg's stake.
    transaction_costs : float
        The flat fee charged once per settled leg, regardless of outcome. This
        is an absolute bankroll amount, not a fraction of the stake: it is
        subtracted from the winning leg's settlement and added to every
        losing leg's. Legs the strategy declines (zero stake) settle nothing
        and pay no fee. Note this differs in unit from the singular
        ``transaction_cost`` taken by strategies in ``keeks.multi_outcome``,
        which is a per-unit fraction of the stake used for sizing.
    probabilities : sequence of float
        The fixed probability of each leg for all trials. Must be a non-empty
        one-dimensional sequence of finite nonnegative numbers summing to at
        most ``1 + PROBABILITY_SUM_TOLERANCE``; probability mass below one is
        the chance of a void or push round on which no leg settles.
    trials : int, default=1000
        The number of betting trials to simulate.
    seed : int or None, default=None
        Seed for the private settlement stream. When omitted, numpy's global
        generator drives the draws and no replay is promised.

    Raises
    ------
    ValueError
        If ``payoffs`` is not a non-empty one-dimensional sequence of finite
        numbers greater than 0, if ``loss`` or ``transaction_costs`` is not
        finite and nonnegative, if ``probabilities`` is not a valid probability
        vector, or if ``trials`` is not a nonnegative integer, or if ``seed``
        is not a nonnegative integer or ``None``.

    Notes
    -----
    **Reproducibility contract (public behavior).** With a ``seed``, every
    draw comes from a private :class:`numpy.random.Generator` derived as the
    first child of ``numpy.random.SeedSequence(seed).spawn(1)``. Rerunning a
    seeded construction replays byte-identically: the same bankroll history
    and the same hook calls. Spawned children, not the raw seed, power the
    streams of this API family, so each stream owns an independent child seed
    and a stream added later cannot shift the settlement stream's draws.
    Without a seed the draws come from numpy's global generator.

    **Trial semantics.** Each trial runs the same flow as the binary
    simulators, generalized to N legs:

    1. Stop when the bankroll is depleted (``total_funds <= 0``).
    2. Fire the strategy's ``update_bankroll`` hook when it has one.
    3. Validate the stake vector returned by ``evaluate`` (one fraction per
       leg, each in ``[0, 1]``, sum at most ``1 + PROBABILITY_SUM_TOLERANCE``).
       A trial the strategy stakes on nothing (every fraction zero) is
       skipped entirely: no draw is consumed, no fee is charged.
    4. Draw one uniform and read it against the cumulative probabilities:
       the realized leg is the first whose cumulative band contains the draw,
       and a draw above the total probability mass is a void or push. A void
       or push round refunds every stake: no leg settles, no fee is charged,
       and the trial's draw is still consumed.
    5. Otherwise settle the batch in leg order. Stakes are computed once from
       the bankroll as it stood when the trial began, so settlements within
       the batch never resize later legs. Each staked leg settles through the
       same net-settlement flow the binary simulators use: the realized leg
       pays ``payoff * stake - transaction_costs`` (deposited, or withdrawn
       when the fee dominates) and every other staked leg is charged
       ``loss * stake + transaction_costs``.
    6. When a bankroll safeguard refuses a settlement (:class:`RuinError`),
       that leg's settlement leaves the bankroll unchanged and reports a 0.0
       return, the remaining legs of the batch still settle, and the
       simulation stops after the batch completes - never mid-batch.

    **The ``record_settlement`` hook.** Strategies expose an N-ary settlement
    hook with the signature ``record_settlement(won_leg, return_pcts)``:
    ``won_leg`` is the realized leg's index, or ``None`` for a void or push;
    ``return_pcts`` holds one signed net return per leg, as a fraction of the
    bankroll before the trial, with ``0.0`` for legs that settled nothing:
    declined legs, void rounds, and settlements a safeguard refused. The hook
    fires once per staked trial, including voids, after the batch settles.

    Examples
    --------
    >>> from keeks.multi_outcome.simulators import RepeatedMultiOutcomeSimulator
    >>> simulator = RepeatedMultiOutcomeSimulator(
    ...     payoffs=(3.2, 3.4, 2.4),
    ...     loss=1.0,
    ...     transaction_costs=0.0,
    ...     probabilities=(0.42, 0.27, 0.28),
    ...     trials=10_000,
    ...     seed=42,
    ... )
    >>> simulator.payoffs
    (3.2, 3.4, 2.4)
    >>> simulator.probabilities
    array([0.42, 0.27, 0.28])
    """

    def __init__(
        self,
        payoffs,
        loss,
        transaction_costs,
        probabilities,
        trials=1000,
        seed=None,
    ):
        self.payoffs: tuple[float, ...] = _validate_payoffs(payoffs)
        loss = _require_finite(loss, "Loss")
        if loss < 0:
            raise ValueError("Loss must be non-negative")
        transaction_costs = _require_finite(transaction_costs, "Transaction costs")
        if transaction_costs < 0:
            raise ValueError("Transaction costs must be non-negative")
        self.loss: float = loss
        self.transaction_costs: float = transaction_costs

        self.probabilities: np.ndarray = normalize_probabilities(probabilities)
        # Cumulative bands for the categorical read: leg j realizes when the
        # trial's uniform falls below cumulative[j] and above cumulative[j-1];
        # the mass above cumulative[-1] is the void region.
        self._cumulative = np.cumsum(self.probabilities)

        try:
            trials = operator.index(trials)
        except TypeError as exc:
            raise ValueError("Trials must be a nonnegative integer") from exc
        if trials < 0:
            raise ValueError("Trials must be a nonnegative integer")
        self.trials: int = trials

        self.seed: int | None = _validate_simulator_seed(seed)
        self._outcome_rng = (
            np.random.default_rng(np.random.SeedSequence(self.seed).spawn(1)[0])
            if self.seed is not None
            else None
        )

    def evaluate_strategy(self, strategy, bankroll) -> None:
        """
        Evaluate a multi-outcome strategy over multiple trials on a fixed market.

        For each trial, the strategy is evaluated with the fixed probability
        vector, one leg realizes, and the bankroll is updated through the
        settlement of the batch. The simulation stops early if the bankroll is
        depleted (bankruptcy) or a settlement batch trips a bankroll safeguard
        (after that batch completes - see the class notes).

        Parameters
        ----------
        strategy : BaseMultiOutcomeStrategy
            The betting strategy to evaluate. Its ``payoffs`` and ``loss``
            must match this simulator's.
        bankroll : BankRoll
            The bankroll to use for the simulation.

        Returns
        -------
        None
            The bankroll object is updated in-place with the results of the
            simulation.

        Raises
        ------
        ValueError
            If ``strategy`` is a ``BaseMultiOutcomeStrategy`` whose ``payoffs``
            or ``loss`` differ from this simulator's, since it would then size
            stakes against different odds than the ones the simulator settles
            at, or if the strategy returns an invalid stake vector.
        """
        _validate_strategy_odds(strategy, self.payoffs, self.loss)

        # Resolve state-dependent hooks once: neither the strategy's hook set
        # nor the bankroll value changes between the reads within one trial.
        update_bankroll = getattr(strategy, "update_bankroll", None)
        if not callable(update_bankroll):
            update_bankroll = None
        record_settlement = getattr(strategy, "record_settlement", None)
        if not callable(record_settlement):
            record_settlement = None

        for _ in range(self.trials):
            # Stop if bankrupt
            total_funds = bankroll.total_funds
            if total_funds <= 0:
                break

            if update_bankroll is not None:
                update_bankroll(total_funds)

            # Get the proportion to bet on each leg
            fractions = _validate_stake_fractions(
                strategy.evaluate(self.probabilities, total_funds)
            )

            # Only process the market if the strategy staked something (avoid
            # charging costs on no-bet)
            if not any(fractions):
                continue

            outcome = (
                self._outcome_rng.random()
                if self._outcome_rng is not None
                else np.random.random()
            )
            realized = int(np.searchsorted(self._cumulative, outcome, side="right"))
            won_leg = realized if realized < len(self.probabilities) else None

            returns = [0.0] * len(fractions)

            if won_leg is None:
                # Void or push: the market refunds every stake, so no leg
                # settles and no fee is charged.
                if record_settlement is not None:
                    record_settlement(None, tuple(returns))
                continue

            # Stakes come from the bankroll as it stood when the trial began;
            # settlements within the batch never resize later legs.
            bettable_funds = bankroll.bettable_funds
            bankroll_before = total_funds
            batch_ruined = False
            for leg, fraction in enumerate(fractions):
                if fraction <= 0:
                    continue
                stake = bettable_funds * fraction
                try:
                    if leg == won_leg:
                        amt = (self.payoffs[leg] * stake) - self.transaction_costs
                        if amt >= 0:
                            bankroll.deposit(amt)
                        else:
                            bankroll.withdraw(abs(amt))
                        returns[leg] = amt / bankroll_before
                    else:
                        amt = (self.loss * stake) + self.transaction_costs
                        bankroll.withdraw(amt)
                        returns[leg] = -amt / bankroll_before
                except RuinError:
                    # Settlement exceeded a bankroll safeguard: that leg's
                    # settlement leaves the bankroll unchanged. Finish the
                    # rest of the batch, then stop - never truncate mid-batch.
                    batch_ruined = True

            if record_settlement is not None:
                record_settlement(won_leg, tuple(returns))

            if batch_ruined:
                break
