import numpy as np
import random


class RandomUncertainBinarySimulator:
    def __init__(self, payoff, loss, transaction_costs, trials=1000, stdev=0.1, uncertainty_stdev=0.05):
        self.payoff = payoff
        self.loss = loss
        self.transaction_costs = transaction_costs
        self.trials = trials
        self.stdev = stdev
        self.uncertainty_stdev = uncertainty_stdev

    def evaluate_strategy(self, strategy, bankroll):
        for _ in range(self.trials):
            probability = np.random.normal(0.5, self.stdev, 1)[0]
            proportion = strategy.evaluate(probability)
            if random.random() < probability + np.random.normal(0, self.uncertainty_stdev, 1)[0]:
                amt = (self.payoff * bankroll.bettable_funds * proportion) - self.transaction_costs
                bankroll.deposit(amt)
            else:
                bankroll.withdraw((self.loss * bankroll.bettable_funds * proportion) - self.transaction_costs)