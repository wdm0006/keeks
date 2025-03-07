from keeks.binary_strategies.base import BaseStrategy

__author__ = "willmcginnis"


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
        """
        # The Kelly formula for binary outcomes:
        # f* = (bp - q) / b
        # where:
        # b = net odds received on the wager (payoff/loss)
        # p = probability of winning
        # q = probability of losing (1-p)
        """
        
        b = self.payoff / self.loss
        p = probability
        q = 1 - p
        
        expected_kelly = (b * p - q) / b
        
        # Adjust for transaction costs
        if expected_kelly > 0:
            expected_kelly -= self.transaction_cost / self.payoff
            
        return expected_kelly


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
        kelly = KellyCriterion(self.payoff, self.loss, self.transaction_cost)
        return self.fraction * kelly.evaluate(probability)
