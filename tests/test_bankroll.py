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


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("initial_funds", -1),
        ("initial_funds", float("nan")),
        ("initial_funds", float("inf")),
        ("initial_funds", float("-inf")),
        ("percent_bettable", -0.1),
        ("percent_bettable", 1.1),
        ("percent_bettable", float("nan")),
        ("max_draw_down", -0.1),
        ("max_draw_down", 1.1),
        ("max_draw_down", float("inf")),
    ],
)
def test_invalid_configuration_raises_value_error(argument, value):
    with pytest.raises(ValueError):
        BankRoll(**{argument: value})


def test_none_disables_drawdown_limit():
    br = BankRoll(initial_funds=100, max_draw_down=None)

    br.withdraw(100)

    assert br.total_funds == 0


@pytest.mark.parametrize(
    "method_name",
    ["deposit", "withdraw", "bet", "add_funds", "remove_funds"],
)
@pytest.mark.parametrize("amount", [-1, float("nan"), float("inf"), float("-inf")])
def test_invalid_transaction_does_not_mutate_bankroll(method_name, amount):
    br = BankRoll(initial_funds=100, max_draw_down=1)
    original_history = br.history.copy()

    with pytest.raises(ValueError):
        getattr(br, method_name)(amount)

    assert br.total_funds == 100
    assert br.history == original_history


@pytest.mark.parametrize("method_name", ["withdraw", "remove_funds", "bet"])
def test_zero_drawdown_rejects_positive_removal_without_mutation(method_name):
    br = BankRoll(initial_funds=100, max_draw_down=0)
    original_history = br.history.copy()

    with pytest.raises(RuinError):
        getattr(br, method_name)(1)

    assert br.total_funds == 100
    assert br.history == original_history


@pytest.mark.parametrize("method_name", ["withdraw", "remove_funds", "bet"])
def test_zero_amount_is_allowed_with_zero_drawdown(method_name):
    br = BankRoll(initial_funds=100, max_draw_down=0)

    getattr(br, method_name)(0)

    assert br.total_funds == 100
    assert br.history == [100, 100]


def test_bet_above_drawdown_limit_does_not_mutate_bankroll():
    br = BankRoll(initial_funds=100, max_draw_down=0.5)
    original_history = br.history.copy()

    with pytest.raises(RuinError):
        br.bet(50.01)

    assert br.total_funds == 100
    assert br.history == original_history


def test_bet_at_drawdown_limit_succeeds():
    br = BankRoll(initial_funds=100, max_draw_down=0.5)

    br.bet(50)

    assert br.total_funds == 50
    assert br.history == [100, 50]


def test_bet_above_bettable_funds_raises_before_drawdown_check():
    br = BankRoll(initial_funds=100, percent_bettable=0.1, max_draw_down=None)
    original_history = br.history.copy()

    with pytest.raises(ValueError):
        br.bet(20)

    assert br.total_funds == 100
    assert br.history == original_history


def test_none_disables_drawdown_limit_for_bet():
    br = BankRoll(initial_funds=100, max_draw_down=None)

    br.bet(100)

    assert br.total_funds == 0
    assert br.history == [100, 0]
