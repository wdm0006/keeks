import random
import unittest

from keeks.bankroll import BankRoll
from keeks.binary_strategies import KellyCriterion
from keeks.binary_strategies.kelly import FractionalKellyCriterion
from keeks.binary_strategies.simple import NaiveStrategy
from keeks.simulators.random_binary import RandomBinarySimulator
from keeks.simulators.random_uncertain_binary import RandomUncertainBinarySimulator
from keeks.simulators.repeated_binary import RepeatedBinarySimulator

random.seed(42)


class TestKellyCriterion(unittest.TestCase):
    def test_even_odds(self):
        strategy = KellyCriterion(payoff=1, loss=1, transaction_cost=0)

        # 60% chance of winning
        portion = strategy.evaluate(0.6)
        self.assertEqual(portion, 0.2)

        # 50% chance of winning
        portion = strategy.evaluate(0.5)
        self.assertEqual(portion, 0)

        # 40% chance of winning
        portion = strategy.evaluate(0.4)
        self.assertEqual(portion, -0.2)

        # 0% chance of winning
        portion = strategy.evaluate(0)
        self.assertEqual(portion, -1)

    def test_known_cases(self):
        strategy = KellyCriterion(payoff=2, loss=1, transaction_cost=0)
        self.assertEqual(strategy.evaluate(0.5), 0.25)

        strategy = KellyCriterion(payoff=10, loss=1, transaction_cost=0)
        self.assertEqual(strategy.evaluate(0.5), 0.45)

        strategy = KellyCriterion(payoff=1, loss=0.5, transaction_cost=0)
        self.assertEqual(strategy.evaluate(0.5), 0.40)

    def test_simulation_repeated(self):
        payoff = 1
        loss = 1
        transaction_cost = 0.01
        probability = 0.55
        trials = 1_000_000

        bankroll = BankRoll(initial_funds=1000, percent_bettable=1, max_draw_down=1)
        strategy = KellyCriterion(payoff, loss, transaction_cost)
        simulator = RepeatedBinarySimulator(
            payoff=payoff,
            loss=loss,
            transaction_costs=transaction_cost,
            probability=probability,
            trials=trials,
        )
        simulator.evaluate_strategy(strategy, bankroll)
        self.assertGreater(bankroll.total_funds, 1_000_000)

    def test_simulation_random(self):
        payoff = 1
        loss = 1
        transaction_cost = 0.01
        trials = 1_000_000
        stdev = 0.1

        bankroll_kelly = BankRoll(
            initial_funds=1000, percent_bettable=1, max_draw_down=1
        )
        bankroll_naive = BankRoll(
            initial_funds=1000, percent_bettable=1, max_draw_down=1
        )

        strategy_kelly = KellyCriterion(payoff, loss, transaction_cost)
        strategy_naive = NaiveStrategy(payoff, loss, transaction_cost)

        simulator_kelly = RandomBinarySimulator(
            payoff=payoff,
            loss=loss,
            transaction_costs=transaction_cost,
            trials=trials,
            stdev=stdev,
        )

        simulator_naive = RandomBinarySimulator(
            payoff=payoff,
            loss=loss,
            transaction_costs=transaction_cost,
            trials=trials,
            stdev=stdev,
        )

        simulator_kelly.evaluate_strategy(strategy_kelly, bankroll_kelly)
        simulator_naive.evaluate_strategy(strategy_naive, bankroll_naive)

        bankroll_kelly.plot_history("kelly.png")
        bankroll_naive.plot_history("naive.png")

    def test_simulation_random_uncertain(self):
        payoff = 0.15
        loss = 0.15
        transaction_cost = 0.01
        trials = 1_000_000
        stdev = 0.1
        uncertainty_stdev = 0.05

        bankroll_kelly = BankRoll(
            initial_funds=1000, percent_bettable=1, max_draw_down=1
        )
        bankroll_naive = BankRoll(
            initial_funds=1000, percent_bettable=1, max_draw_down=1
        )
        bankroll_fractional_kelly = BankRoll(
            initial_funds=1000, percent_bettable=1, max_draw_down=1
        )

        strategy_kelly = KellyCriterion(payoff, loss, transaction_cost)
        strategy_naive = NaiveStrategy(payoff, loss, transaction_cost)
        strategy_fractional_kelly = FractionalKellyCriterion(
            payoff, loss, transaction_cost, 0.5
        )

        simulator_kelly = RandomUncertainBinarySimulator(
            payoff=payoff,
            loss=loss,
            transaction_costs=transaction_cost,
            trials=trials,
            stdev=stdev,
            uncertainty_stdev=uncertainty_stdev,
        )

        simulator_naive = RandomUncertainBinarySimulator(
            payoff=payoff,
            loss=loss,
            transaction_costs=transaction_cost,
            trials=trials,
            stdev=stdev,
            uncertainty_stdev=uncertainty_stdev,
        )

        simulator_fractional_kelly = RandomUncertainBinarySimulator(
            payoff=payoff,
            loss=loss,
            transaction_costs=transaction_cost,
            trials=trials,
            stdev=stdev,
            uncertainty_stdev=uncertainty_stdev,
        )

        simulator_kelly.evaluate_strategy(strategy_kelly, bankroll_kelly)
        simulator_naive.evaluate_strategy(strategy_naive, bankroll_naive)
        simulator_fractional_kelly.evaluate_strategy(
            strategy_fractional_kelly, bankroll_fractional_kelly
        )

        bankroll_kelly.plot_history("kelly.png")
        bankroll_naive.plot_history("naive.png")
        bankroll_fractional_kelly.plot_history("fractional_kelly.png")
        # self.assertGreater(bankroll_fractional_kelly.total_funds, 1_000_000)
        # self.assertGreater(
        #     bankroll_fractional_kelly.total_funds, bankroll_kelly.total_funds
        # )
        # self.assertGreater(
        #     bankroll_fractional_kelly.total_funds, bankroll_naive.total_funds
        # )
