import random


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
    transaction_costs : float
        The fixed cost per transaction, regardless of outcome.
    probability : float
        The fixed probability of a successful outcome for all trials.
    trials : int, default=1000
        The number of betting trials to simulate.
    """

    def __init__(self, payoff, loss, transaction_costs, probability, trials=1000):
        self.payoff = payoff
        self.loss = loss
        self.transaction_costs = transaction_costs
        self.probability = probability
        self.trials = trials

    def evaluate_strategy(self, strategy, bankroll):
        """
        Evaluate a betting strategy over multiple trials with fixed probability.

        For each trial, the strategy is evaluated with the fixed probability,
        and the bankroll is updated based on the outcome.

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
            # Update the strategy's internal state with current bankroll if supported
            if hasattr(strategy, "update_bankroll"):
                strategy.update_bankroll(bankroll.total_funds)

            # Get the proportion to bet
            proportion = strategy.evaluate(self.probability, bankroll.total_funds)

            if random.random() < self.probability:
                amt = (
                    self.payoff * bankroll.bettable_funds * proportion
                ) - self.transaction_costs
                bankroll.deposit(amt)
            else:
                bankroll.withdraw(
                    (self.loss * bankroll.bettable_funds * proportion)
                    - self.transaction_costs
                )
