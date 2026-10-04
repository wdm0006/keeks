import random
import warnings

from keeks.utils import (
    RuinError,
    _validate_simulator_controls,
    _validate_simulator_probability,
    _validate_simulator_seed,
    _validate_stake_fraction,
    _validate_strategy_odds,
)


class RepeatedBinarySimulator:
    """
    Simulator for binary betting strategies with a fixed probability.

    This simulator uses the same probability for each trial, simulating
    repeated bets on events with identical odds.

    Parameters
    ----------
    payoff : float
        The amount won per unit bet on a successful outcome.
    loss : float
        The amount lost per unit bet on an unsuccessful outcome.
    fee_per_bet : float
        The flat fee charged once per settled bet, regardless of outcome. This is
        an absolute bankroll amount, not a fraction of the stake, so it does not
        scale with bet size: it is subtracted from a winning settlement and added
        to a losing one. Note this differs in unit from the singular
        ``transaction_cost_rate`` taken by strategies in ``keeks.binary_strategies``,
        which is a per-unit fraction of the bet used for sizing.
    probability : float
        The fixed probability of a successful outcome for all trials.
    trials : int, default=1000
        The number of betting trials to simulate.
    seed : int or None, default=None
        Seed for a private outcome generator. When omitted, the process-global
        ``random`` generator is used for backward compatibility.

    Raises
    ------
    ValueError
        If ``payoff`` is not finite and positive, if ``loss`` or
        ``fee_per_bet`` is not finite and nonnegative, if ``probability``
        is not finite within ``[0, 1]``, or if ``trials`` is not a nonnegative
        integer, or if ``seed`` is not a nonnegative integer or ``None``.
    """

    def __init__(self, payoff, loss, fee_per_bet, probability, trials=1000, seed=None):
        (
            self.payoff,
            self.loss,
            self.fee_per_bet,
            self.trials,
        ) = _validate_simulator_controls(payoff, loss, fee_per_bet, trials)
        self.probability = _validate_simulator_probability(probability, "Probability")
        self.seed = _validate_simulator_seed(seed)
        self._outcome_rng = random.Random(self.seed) if self.seed is not None else None

    def evaluate_strategy(self, strategy, bankroll):
        """
        Evaluate a betting strategy over multiple trials with fixed probability.

        For each trial, the strategy is evaluated with the fixed probability,
        and the bankroll is updated based on the outcome. The simulation stops
        early if the bankroll is depleted (bankruptcy).

        Parameters
        ----------
        strategy : BaseStrategy
            The betting strategy to evaluate.
        bankroll : BankRoll
            The bankroll to use for the simulation.

        Returns
        -------
        None
            The bankroll object is updated in-place with the results of the simulation.

        Raises
        ------
        ValueError
            If ``strategy`` is a ``BaseStrategy`` whose ``payoff`` or ``loss``
            differs from this simulator's, since it would then size bets against
            different odds than the ones the simulator settles at, or if the
            strategy returns a non-finite or out-of-range stake fraction.
        """
        _validate_strategy_odds(strategy, self.payoff, self.loss)

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

            # Get the proportion to bet
            proportion = _validate_stake_fraction(
                strategy.evaluate(self.probability, total_funds)
            )

            # Only process the bet if proportion > 0 (avoid charging costs on no-bet)
            if proportion > 0:
                current_bankroll = total_funds
                bet_amount = bankroll.bettable_funds * proportion
                try:
                    outcome = (
                        random.random()
                        if self._outcome_rng is None
                        else self._outcome_rng.random()
                    )
                    won = outcome < self.probability
                    if won:
                        amount = (self.payoff * bet_amount) - self.fee_per_bet
                        if amount >= 0:
                            bankroll.deposit(amount)
                        else:
                            bankroll.withdraw(abs(amount))
                        realized_return = amount / current_bankroll
                    else:
                        amount = (self.loss * bet_amount) + self.fee_per_bet
                        bankroll.withdraw(amount)
                        realized_return = -amount / current_bankroll
                except RuinError as exc:
                    # Settlement exceeded a bankroll safeguard; stop the run
                    # loudly rather than silently: the warning carries the
                    # refused amount, the configured limit, and current funds.
                    warnings.warn(f"Simulation stopped early: {exc}", stacklevel=2)
                    break

                if record_settlement is not None:
                    record_settlement((won,), (realized_return,))
