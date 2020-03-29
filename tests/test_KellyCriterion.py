import unittest
from keeks.binary_strategies.kelly import KellyCriterion, FractionalKellyCriterion
from keeks.binary_strategies.simple import NaiveStrategy
from keeks.simulators.repeated_binary import RepeatedBinarySimulator
from keeks.simulators.random_binary import RandomBinarySimulator
from keeks.simulators.random_uncertain_binary import RandomUncertainBinarySimulator
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

    def test_SimulationRepeated(self):
        payoff = 1
        loss = 1
        transaction_cost = 0.01
        win_probability = 0.6
        bankroll = BankRoll(initial_funds=1_000_000, max_draw_down=None, percent_bettable=1)
        simulator = RepeatedBinarySimulator(payoff, loss, transaction_cost, win_probability, trials=1000)
        strategy = KellyCriterion(payoff=payoff, loss=loss, transaction_cost=transaction_cost)
        simulator.evaluate_strategy(strategy, bankroll)
        self.assertGreater(bankroll.total_funds, 1_000_000)

    def test_SimulationRandom(self):
        payoff = 1
        loss = 1
        transaction_cost = 0.25
        simulator = RandomBinarySimulator(payoff, loss, transaction_cost, trials=1000, stdev=0.1)

        bankroll_kelly = BankRoll(initial_funds=1_000_000, max_draw_down=None, percent_bettable=1)
        bankroll_fractional_kelly = BankRoll(initial_funds=1_000_000, max_draw_down=None, percent_bettable=1)
        bankroll_naive = BankRoll(initial_funds=1_000_000, max_draw_down=None, percent_bettable=1)

        strategy_kelly = KellyCriterion(payoff=payoff, loss=loss, transaction_cost=transaction_cost)
        strategy_fractional_kelly = FractionalKellyCriterion(payoff=payoff, loss=loss,
                                                             transaction_cost=transaction_cost, fraction=0.1)
        strategy_naive = NaiveStrategy(payoff=payoff, loss=loss, transaction_cost=transaction_cost)

        simulator.evaluate_strategy(strategy_kelly, bankroll_kelly)
        simulator.evaluate_strategy(strategy_fractional_kelly, bankroll_fractional_kelly)
        simulator.evaluate_strategy(strategy_naive, bankroll_naive)

        self.assertGreater(bankroll_fractional_kelly.total_funds, 1_000_000)
        self.assertGreater(bankroll_fractional_kelly.total_funds, bankroll_kelly.total_funds)
        self.assertGreater(bankroll_fractional_kelly.total_funds, bankroll_naive.total_funds)

    def test_SimulationRandomUncertain(self):
        payoff = 0.15
        loss = 0.15
        transaction_cost = 0.25
        simulator = RandomUncertainBinarySimulator(payoff, loss, transaction_cost, trials=1000, stdev=0.1, uncertainty_stdev=0.05)

        bankroll_kelly = BankRoll(initial_funds=1_000_000, max_draw_down=None, percent_bettable=1)
        bankroll_fractional_kelly = BankRoll(initial_funds=1_000_000, max_draw_down=None, percent_bettable=1)
        bankroll_naive = BankRoll(initial_funds=1_000_000, max_draw_down=None, percent_bettable=1)

        strategy_kelly = KellyCriterion(payoff=payoff, loss=loss, transaction_cost=transaction_cost)
        strategy_fractional_kelly = FractionalKellyCriterion(payoff=payoff, loss=loss, transaction_cost=transaction_cost, fraction=0.1)
        strategy_naive = NaiveStrategy(payoff=payoff, loss=loss, transaction_cost=transaction_cost)

        simulator.evaluate_strategy(strategy_kelly, bankroll_kelly)
        simulator.evaluate_strategy(strategy_fractional_kelly, bankroll_fractional_kelly)
        simulator.evaluate_strategy(strategy_naive, bankroll_naive)

        bankroll_fractional_kelly.plot_history('fractional_kelly.png')
        # self.assertGreater(bankroll_fractional_kelly.total_funds, 1_000_000)
        # self.assertGreater(bankroll_fractional_kelly.total_funds, bankroll_kelly.total_funds)
        # self.assertGreater(bankroll_fractional_kelly.total_funds, bankroll_naive.total_funds)
