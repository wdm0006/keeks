import random

import pytest

from keeks.bankroll import BankRoll
from keeks.binary_strategies import DrawdownAdjustedKelly, KellyCriterion
from keeks.simulators.repeated_binary import RepeatedBinarySimulator

random.seed(42)


def test_basic_functionality():
    """Test that DrawdownAdjustedKelly properly scales the Kelly criterion."""
    payoff = 2
    loss = 1
    transaction_cost = 0
    current_bankroll = 1000

    # Standard Kelly for probability 0.6 with 2:1 payoff
    regular_kelly = KellyCriterion(payoff, loss, transaction_cost)
    regular_kelly_bet = regular_kelly.evaluate(0.6, current_bankroll)

    # Drawdown-adjusted Kelly with 20% max drawdown (less aggressive)
    adjusted_kelly = DrawdownAdjustedKelly(
        payoff, loss, transaction_cost, max_acceptable_drawdown=0.2
    )
    adjusted_kelly_bet = adjusted_kelly.evaluate(0.6, current_bankroll)

    # Adjusted Kelly should be smaller than regular Kelly
    assert adjusted_kelly_bet < regular_kelly_bet

    # The adjustment should scale by approximately max_drawdown / 0.5
    # (since full Kelly has expected drawdown around 50%)
    expected_scale = 0.2 / 0.5  # 0.4
    assert adjusted_kelly_bet == pytest.approx(regular_kelly_bet * expected_scale)


def test_different_drawdown_levels():
    """Test behavior with different drawdown tolerance levels."""
    payoff = 2
    loss = 1
    transaction_cost = 0
    probability = 0.6
    current_bankroll = 1000

    # Regular Kelly
    regular_kelly = KellyCriterion(payoff, loss, transaction_cost)
    regular_kelly_bet = regular_kelly.evaluate(probability, current_bankroll)

    # Test various drawdown levels
    drawdown_levels = [0.1, 0.2, 0.3, 0.4, 0.5]

    for drawdown in drawdown_levels:
        adjusted_kelly = DrawdownAdjustedKelly(
            payoff, loss, transaction_cost, max_acceptable_drawdown=drawdown
        )
        adjusted_kelly_bet = adjusted_kelly.evaluate(probability, current_bankroll)

        # Check scaling is approximately proportional to drawdown level
        expected_scale = min(1.0, drawdown / 0.5)
        assert adjusted_kelly_bet == pytest.approx(regular_kelly_bet * expected_scale)

    # A drawdown of 0.5 or higher should give approximately full Kelly
    high_drawdown_kelly = DrawdownAdjustedKelly(
        payoff, loss, transaction_cost, max_acceptable_drawdown=0.6
    )
    assert high_drawdown_kelly.evaluate(probability, current_bankroll) == pytest.approx(
        regular_kelly_bet
    )


def test_negative_ev_cases():
    """Test behavior when expected value is negative."""
    payoff = 1
    loss = 1
    transaction_cost = 0
    probability = 0.4  # Negative EV
    current_bankroll = 1000

    # Regular Kelly would return 0 for negative EV
    regular_kelly = KellyCriterion(payoff, loss, transaction_cost)
    regular_kelly_bet = regular_kelly.evaluate(probability, current_bankroll)
    assert regular_kelly_bet == 0

    # Drawdown-adjusted Kelly should also return 0
    drawdown_kelly = DrawdownAdjustedKelly(
        payoff, loss, transaction_cost, max_acceptable_drawdown=0.2
    )
    drawdown_kelly_bet = drawdown_kelly.evaluate(probability, current_bankroll)
    assert drawdown_kelly_bet == 0


def test_invalid_parameters():
    """Test that invalid parameters raise appropriate exceptions."""
    payoff = 2
    loss = 1
    transaction_cost = 0

    # Max acceptable drawdown must be between 0 and 1 (exclusive)
    with pytest.raises(
        ValueError, match="Maximum acceptable drawdown must be between 0 and 1 .*"
    ):
        DrawdownAdjustedKelly(payoff, loss, transaction_cost, max_acceptable_drawdown=0)

    with pytest.raises(
        ValueError, match="Maximum acceptable drawdown must be between 0 and 1 .*"
    ):
        DrawdownAdjustedKelly(payoff, loss, transaction_cost, max_acceptable_drawdown=1)


def test_simulation_comparison():
    """Compare performance of regular Kelly vs drawdown-adjusted Kelly."""
    payoff = 1
    loss = 1
    transaction_cost = 0.01
    probability = 0.55  # Slight edge
    trials = 1000

    # Regular Kelly
    bankroll_regular = BankRoll(initial_funds=1000, percent_bettable=1, max_draw_down=1)
    strategy_regular = KellyCriterion(payoff, loss, transaction_cost)

    # Drawdown-adjusted Kelly (more conservative)
    bankroll_adjusted = BankRoll(
        initial_funds=1000, percent_bettable=1, max_draw_down=1
    )
    strategy_adjusted = DrawdownAdjustedKelly(
        payoff, loss, transaction_cost, max_acceptable_drawdown=0.2
    )

    # Use the same random seed for fair comparison
    simulator_regular = RepeatedBinarySimulator(
        payoff=payoff,
        loss=loss,
        transaction_costs=transaction_cost,
        probability=probability,
        trials=trials,
    )

    simulator_adjusted = RepeatedBinarySimulator(
        payoff=payoff,
        loss=loss,
        transaction_costs=transaction_cost,
        probability=probability,
        trials=trials,
    )

    # Run simulations with the same seed
    random.seed(42)
    simulator_regular.evaluate_strategy(strategy_regular, bankroll_regular)

    random.seed(42)
    simulator_adjusted.evaluate_strategy(strategy_adjusted, bankroll_adjusted)

    # Both should grow with a positive edge
    assert bankroll_regular.total_funds > 1000
    assert bankroll_adjusted.total_funds > 1000

    # Check that history exists for comparison
    assert len(bankroll_regular.history) > 1
    assert len(bankroll_adjusted.history) > 1
