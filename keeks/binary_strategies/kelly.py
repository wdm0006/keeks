from keeks.binary_strategies.base import BaseStrategy

__author__ = 'willmcginnis'


class KellyCriterion(BaseStrategy):
    def __init__(self, payoff, loss, transaction_cost):
        """
        :param payoff:
        :param loss:
        :param transaction_cost:
        """
        self.payoff = payoff
        self.loss = loss
        self.transaction_cost = transaction_cost

    def evaluate(self, probability):
        expected_net_winnings = (self.payoff * probability) - (self.loss * (1 - probability)) - self.transaction_cost
        winnings_if_won = self.payoff - self.transaction_cost

        return expected_net_winnings / winnings_if_won


class FractionalKellyCriterion(BaseStrategy):
    def __init__(self, payoff, loss, transaction_cost, fraction):
        """
        :param payoff:
        :param loss:
        :param transaction_cost:
        """
        self.payoff = payoff
        self.loss = loss
        self.transaction_cost = transaction_cost
        self.fraction = fraction

    def evaluate(self, probability):
        expected_net_winnings = (self.payoff * probability) - (self.loss * (1 - probability)) - self.transaction_cost
        winnings_if_won = self.payoff - self.transaction_cost

        return self.fraction * (expected_net_winnings / winnings_if_won)