import random
import unittest

from keeks.bankroll import BankRoll
from keeks.binary_strategies import NaiveStrategy
from keeks.simulators.repeated_binary import RepeatedBinarySimulator
from keeks.utils import RuinError

random.seed(42)


class TestNaiveStrategy(unittest.TestCase):
    def test_even_odds(self):
        strategy = NaiveStrategy(payoff=1, loss=1, transaction_cost=0)

        # 60% chance of winning
        portion = strategy.evaluate(0.6)
        self.assertEqual(portion, 1)

        # 50% chance of winning
        portion = strategy.evaluate(0.5)
        self.assertEqual(portion, 0)

        # 40% chance of winning
        portion = strategy.evaluate(0.4)
        self.assertEqual(portion, -1)

    def test_simulation(self):
        with self.assertRaises(RuinError):
            payoff = 1
            loss = 1
            transaction_cost = 0.01
            probability = 0.45
            trials = 1_000_000

            bankroll = BankRoll(
                initial_funds=1000, percent_bettable=1, max_draw_down=0.3
            )
            strategy = NaiveStrategy(payoff, loss, transaction_cost)
            simulator = RepeatedBinarySimulator(
                payoff=payoff,
                loss=loss,
                transaction_costs=transaction_cost,
                probability=probability,
                trials=trials,
            )
            simulator.evaluate_strategy(strategy, bankroll)
