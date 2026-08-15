import random

import numpy as np

from keeks.utils import (
    RuinError,
    _update_strategy_bankroll,
    _validate_simulator_controls,
    _validate_simulator_seed,
    _validate_simulator_stdev,
    _validate_strategy_odds,
)


class RandomUncertainBinarySimulator:
    """
    Simulator for binary betting strategies with random probabilities and uncertainty.

    This simulator generates random probabilities for each trial, centered around 0.5
    with a configurable standard deviation. It adds an additional uncertainty factor
    to the actual outcome probability, simulating imperfect information.

    Parameters
    ----------
    payoff : float
        The amount won per unit bet on a successful outcome.
    loss : float
        The amount lost per unit bet on an unsuccessful outcome.
    transaction_costs : float
        The flat fee charged once per settled bet, regardless of outcome. This is
        an absolute bankroll amount, not a fraction of the stake, so it does not
        scale with bet size: it is subtracted from a winning settlement and added
        to a losing one. Note this differs in unit from the singular
        ``transaction_cost`` taken by strategies in ``keeks.binary_strategies``,
        which is a per-unit fraction of the bet used for sizing.
    trials : int, default=1000
        The number of betting trials to simulate.
    stdev : float, default=0.1
        The standard deviation of the normal distribution used to generate probabilities.
        Samples are clamped to [0.0, 1.0].
    uncertainty_stdev : float, default=0.05
        The standard deviation of the normal distribution used to add uncertainty
        to the actual outcome probability. The resulting outcome probability is
        clamped to [0.0, 1.0].
    seed : int or None, default=None
        Seed for private outcome, probability, and uncertainty generators. When
        omitted, the process-global ``random`` and ``numpy.random`` generators
        are used for backward compatibility.

    Raises
    ------
    ValueError
        If ``payoff`` is not finite and positive, if ``loss``,
        ``transaction_costs``, ``stdev`` or ``uncertainty_stdev`` is not finite
        and nonnegative, if ``trials`` is not a nonnegative integer, or if
        ``seed`` is not a nonnegative integer or ``None``.
    """

    def __init__(
        self,
        payoff,
        loss,
        transaction_costs,
        trials=1000,
        stdev=0.1,
        uncertainty_stdev=0.05,
        seed=None,
    ):
        (
            self.payoff,
            self.loss,
            self.transaction_costs,
            self.trials,
        ) = _validate_simulator_controls(payoff, loss, transaction_costs, trials)
        self.stdev = _validate_simulator_stdev(stdev, "Standard deviation")
        self.uncertainty_stdev = _validate_simulator_stdev(
            uncertainty_stdev, "Uncertainty standard deviation"
        )
        self.seed = _validate_simulator_seed(seed)
        self._outcome_rng = random.Random(self.seed) if self.seed is not None else None
        self._probability_rng = (
            np.random.default_rng(self.seed) if self.seed is not None else None
        )

    def evaluate_strategy(self, strategy, bankroll):
        """
        Evaluate a betting strategy over multiple trials with uncertainty.

        For each trial, a random probability is generated, the strategy is evaluated
        with this probability, but the actual outcome is determined by the probability
        plus a random uncertainty factor. The simulation stops early if the bankroll
        is depleted (bankruptcy).

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
            different odds than the ones the simulator settles at.
        """
        _validate_strategy_odds(strategy, self.payoff, self.loss)

        for _ in range(self.trials):
            # Stop if bankrupt
            if bankroll.total_funds <= 0:
                break

            _update_strategy_bankroll(strategy, bankroll.total_funds)

            # Normal samples are unbounded; only [0, 1] values are probabilities.
            probability = min(
                1.0,
                max(
                    0.0,
                    np.random.normal(0.5, self.stdev, 1)[0]
                    if self._probability_rng is None
                    else self._probability_rng.normal(0.5, self.stdev),
                ),
            )
            proportion = strategy.evaluate(probability, bankroll.total_funds)

            # Only process the bet if proportion > 0 (avoid charging costs on no-bet)
            if proportion > 0:
                current_bankroll = bankroll.total_funds
                bet_amount = bankroll.bettable_funds * proportion
                outcome_probability = min(
                    1.0,
                    max(
                        0.0,
                        probability
                        + (
                            np.random.normal(0, self.uncertainty_stdev, 1)[0]
                            if self._probability_rng is None
                            else self._probability_rng.normal(0, self.uncertainty_stdev)
                        ),
                    ),
                )
                try:
                    outcome = (
                        random.random()
                        if self._outcome_rng is None
                        else self._outcome_rng.random()
                    )
                    won = outcome < outcome_probability
                    if won:
                        amt = (self.payoff * bet_amount) - self.transaction_costs
                        if amt >= 0:
                            bankroll.deposit(amt)
                        else:
                            bankroll.withdraw(abs(amt))
                        return_pct = amt / current_bankroll
                    else:
                        amt = (self.loss * bet_amount) + self.transaction_costs
                        bankroll.withdraw(amt)
                        return_pct = -amt / current_bankroll
                except RuinError:
                    # Settlement exceeded a bankroll safeguard; stop gracefully
                    break

                record_result = getattr(strategy, "record_result", None)
                if callable(record_result):
                    record_result(won, return_pct)
