import unittest

from keeks.bankroll import BankRoll
from keeks.utils import RuinError


class TestBankroll(unittest.TestCase):
    def test_transactions(self):
        br = BankRoll(initial_funds=1000, percent_bettable=1, max_draw_down=1)
        self.assertEqual(br.bettable_funds, 1000)
        self.assertEqual(br.total_funds, 1000)

        br.deposit(500)
        self.assertEqual(br.bettable_funds, 1500)
        self.assertEqual(br.total_funds, 1500)

        br.withdraw(500)
        self.assertEqual(br.bettable_funds, 1000)
        self.assertEqual(br.total_funds, 1000)

        br.deposit(500)
        self.assertEqual(br.bettable_funds, 1500)
        self.assertEqual(br.total_funds, 1500)

    def test_percent_bettable(self):
        br = BankRoll(initial_funds=1000, percent_bettable=0.5, max_draw_down=1)
        self.assertEqual(br.bettable_funds, 500)
        self.assertEqual(br.total_funds, 1000)

        br.deposit(500)
        self.assertEqual(br.bettable_funds, 750)
        self.assertEqual(br.total_funds, 1500)

        br.withdraw(500)
        self.assertEqual(br.bettable_funds, 500)
        self.assertEqual(br.total_funds, 1000)

    def test_drawdown_limit(self):
        br = BankRoll(initial_funds=1000, percent_bettable=0.5, max_draw_down=0.3)
        with self.assertRaises(RuinError):
            br.withdraw(400)
