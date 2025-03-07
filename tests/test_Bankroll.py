import pytest

from keeks.bankroll import BankRoll
from keeks.utils import RuinError


def test_transactions():
    br = BankRoll(initial_funds=1000, percent_bettable=1, max_draw_down=1)
    assert br.bettable_funds == 1000
    assert br.total_funds == 1000

    br.deposit(500)
    assert br.bettable_funds == 1500
    assert br.total_funds == 1500

    br.withdraw(500)
    assert br.bettable_funds == 1000
    assert br.total_funds == 1000

    br.deposit(500)
    assert br.bettable_funds == 1500
    assert br.total_funds == 1500


def test_percent_bettable():
    br = BankRoll(initial_funds=1000, percent_bettable=0.5, max_draw_down=1)
    assert br.bettable_funds == 500
    assert br.total_funds == 1000

    br.deposit(500)
    assert br.bettable_funds == 750
    assert br.total_funds == 1500

    br.withdraw(500)
    assert br.bettable_funds == 500
    assert br.total_funds == 1000


def test_drawdown_limit():
    br = BankRoll(initial_funds=1000, percent_bettable=0.5, max_draw_down=0.3)
    with pytest.raises(RuinError):
        br.withdraw(400)
