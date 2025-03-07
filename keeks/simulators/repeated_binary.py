import random


class RepeatedBinarySimulator:
    def __init__(self, payoff, loss, transaction_costs, probability, trials=1000):
        self.payoff = payoff
        self.loss = loss
        self.transaction_costs = transaction_costs
        self.probability = probability
        self.trials = trials

    def evaluate_strategy(self, strategy, bankroll):
        for _ in range(self.trials):
            proportion = strategy.evaluate(self.probability)
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
