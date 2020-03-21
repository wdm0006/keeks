import unittest
from keeks import BankRoll


class TestBankroll(unittest.TestCase):
    def test_Transactions(self):
        br = BankRoll(initial_funds=1000, percent_bettable=1, max_draw_down=1)
        self.assertEqual(br.bettable_funds, 1000)
        self.assertEqual(br.total_funds, 1000)

        br.deposit(1000)
        self.assertEqual(br.bettable_funds, 2000)
        self.assertEqual(br.total_funds, 2000)

        br.withdraw(500)
        self.assertEqual(br.bettable_funds, 1500)
        self.assertEqual(br.total_funds, 1500)

    def test_PercentBettable(self):
        br = BankRoll(initial_funds=1000, percent_bettable=0.5, max_draw_down=1)
        self.assertEqual(br.bettable_funds, 500)
        self.assertEqual(br.total_funds, 1000)

        br.deposit(1000)
        self.assertEqual(br.bettable_funds, 1000)
        self.assertEqual(br.total_funds, 2000)

        br.withdraw(500)
        self.assertEqual(br.bettable_funds, 750)
        self.assertEqual(br.total_funds, 1500)

    def test_Drawdown_Limit(self):
        br = BankRoll(initial_funds=1000, percent_bettable=0.5, max_draw_down=0.3)
        with self.assertRaises(ValueError):
            br.withdraw(500)