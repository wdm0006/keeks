import unittest
from keeks.binary_strategies.kelly import KellyCriterion
from keeks.simulators.repeated_binary import RepeatedBinarySimulator
from keeks.bankroll import BankRoll
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
        bankroll = BankRoll(initial_funds=1_000_000, max_draw_down=1, percent_bettable=1)
        simulator = RepeatedBinarySimulator(payoff, loss, transaction_cost, win_probability, trials=1000)
        strategy = KellyCriterion(payoff=payoff, loss=loss, transaction_cost=transaction_cost)
        simulator.evaluate_strategy(strategy, bankroll)
        self.assertGreater(bankroll.total_funds, 1_000_000)

