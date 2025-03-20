import random

import numpy as np


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
        The fixed cost per transaction, regardless of outcome.
    trials : int, default=1000
        The number of betting trials to simulate.
    stdev : float, default=0.1
        The standard deviation of the normal distribution used to generate probabilities.
    uncertainty_stdev : float, default=0.05
        The standard deviation of the normal distribution used to add uncertainty
        to the actual outcome probability.
    """

    def __init__(
        self,
        payoff,
        loss,
        transaction_costs,
        trials=1000,
        stdev=0.1,
        uncertainty_stdev=0.05,
    ):
        self.payoff = payoff
        self.loss = loss
        self.transaction_costs = transaction_costs
        self.trials = trials
        self.stdev = stdev
        self.uncertainty_stdev = uncertainty_stdev

    def evaluate_strategy(self, strategy, bankroll):
        """
        Evaluate a betting strategy over multiple trials with uncertainty.

        For each trial, a random probability is generated, the strategy is evaluated
        with this probability, but the actual outcome is determined by the probability
        plus a random uncertainty factor.

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
        """
        for _ in range(self.trials):
            probability = np.random.normal(0.5, self.stdev, 1)[0]
            proportion = strategy.evaluate(probability, bankroll.total_funds)
            if (
                random.random()
                < probability + np.random.normal(0, self.uncertainty_stdev, 1)[0]
            ):
                amt = (
                    self.payoff * bankroll.bettable_funds * proportion
                ) - self.transaction_costs
                bankroll.deposit(amt)
            else:
                bankroll.withdraw(
                    (self.loss * bankroll.bettable_funds * proportion)
                    - self.transaction_costs
                )
