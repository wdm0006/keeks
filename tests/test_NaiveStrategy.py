import unittest
from keeks import NaiveStrategy
import random

random.seed(42)


class TestNaiveStrategy(unittest.TestCase):
    def test_EvenOdds(self):
        strategy = NaiveStrategy(payoff=1, loss=1, transaction_cost=0)

        # low probability should result in low allocations and vice versa
        portion = strategy.evaluate(0.9)
        self.assertGreater(portion, 0.0)
        portion = strategy.evaluate(0.1)
        self.assertLess(portion, 0.0)
        portion = strategy.evaluate(1)
        self.assertEqual(portion, 1.0)
        portion = strategy.evaluate(0)
        self.assertEqual(portion, -1)

    def test_Simulation(self):
        with self.assertRaises(ValueError):
            payoff = 1
            loss = 1
            transaction_cost = 0.01
            win_probability = 0.6
            wealth = [1_000_000]
            strategy = NaiveStrategy(payoff=payoff, loss=loss, transaction_cost=transaction_cost)
            for trial in range(1000):
                proportion = strategy.evaluate(win_probability)
                if random.random() < win_probability:
                    wealth.append(wealth[-1] - transaction_cost + (payoff * wealth[-1] * proportion))
                else:
                    wealth.append(wealth[-1] - transaction_cost - (loss * wealth[-1] * proportion))
                if wealth[-1] < 0:
                    raise ValueError('Bankrupt')
                print('betting proportion: %s, wealth: %s' % (proportion, wealth[-1]))