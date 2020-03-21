import unittest
from keeks import KellyCriterion
import random

random.seed(42)

class TestKellyCriterion(unittest.TestCase):
    def test_EvenOdds(self):
        strategy = KellyCriterion(payoff=1, loss=1, transaction_cost=0)

        # low probability should result in low allocations and vice versa
        portion = strategy.evaluate(0.9)
        self.assertGreater(portion, 0.0)
        portion = strategy.evaluate(0.1)
        self.assertLess(portion, 0.0)
        portion = strategy.evaluate(1)
        self.assertEqual(portion, 1.0)
        portion = strategy.evaluate(0)
        self.assertEqual(portion, -1)

    def test_KnownCases(self):
        strategy = KellyCriterion(payoff=2, loss=1, transaction_cost=0)
        self.assertEqual(strategy.evaluate(0.5), 0.25)

        strategy = KellyCriterion(payoff=5, loss=1, transaction_cost=0)
        self.assertEqual(strategy.evaluate(0.5), 0.40)

    def test_Simulation(self):
        payoff = 1
        loss = 1
        transaction_cost = 0.01
        win_probability = 0.6
        wealth = [1_000_000]
        strategy = KellyCriterion(payoff=payoff, loss=loss, transaction_cost=transaction_cost)
        for trial in range(1000):
            proportion = strategy.evaluate(win_probability)
            if random.random() < win_probability:
                wealth.append(wealth[-1] - transaction_cost + (payoff * wealth[-1] * proportion))
            else:
                wealth.append(wealth[-1] - transaction_cost - (loss * wealth[-1] * proportion))

            if wealth[-1] < 0:
                raise Exception('Bankrupt')

            print('betting proportion: %s, wealth: %s' % (proportion, wealth[-1]))

