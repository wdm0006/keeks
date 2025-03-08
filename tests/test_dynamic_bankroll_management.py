import random

import pytest
import numpy as np

from keeks.bankroll import BankRoll
from keeks.binary_strategies import DynamicBankrollManagement
from keeks.simulators.repeated_binary import RepeatedBinarySimulator

random.seed(42)


def test_basic_functionality():
    """Test that DynamicBankrollManagement strategy returns the correct base fraction initially."""
    payoff = 1
    loss = 1
    base_fraction = 0.1
    
    # Initialize with a 0.1 (10%) base fraction
    strategy = DynamicBankrollManagement(
        base_fraction=base_fraction,
        payoff=payoff,
        loss=loss
    )
    
    # Initially should return the base fraction when probability >= min_probability
    assert strategy.evaluate(0.6) == pytest.approx(base_fraction)
    
    # Should return 0 when probability < min_probability
    assert strategy.evaluate(0.4) == pytest.approx(0)


def test_streak_factor():
    """Test that streak factor increases with winning streak and decreases with losing streak."""
    strategy = DynamicBankrollManagement(
        base_fraction=0.1,
        payoff=1,
        loss=1
    )
    
    # Initially streak factor should be 1.0 (no streak)
    assert strategy.get_streak_factor() == 1.0
    
    # Record a win
    strategy.record_result(won=True, return_pct=0.1, current_bankroll=1.1)
    assert strategy.get_streak_factor() > 1.0
    
    # Record multiple wins to build a streak
    for _ in range(4):
        strategy.record_result(won=True, return_pct=0.1, current_bankroll=1.2)
    
    # Should have a significant increase due to 5-win streak
    assert strategy.get_streak_factor() > 1.3
    
    # Record a loss to break the streak
    strategy.record_result(won=False, return_pct=-0.05, current_bankroll=1.15)
    
    # Factor should now be less than 1.0 due to the loss
    assert strategy.get_streak_factor() < 1.0
    
    # Record multiple losses to build a losing streak
    for _ in range(4):
        strategy.record_result(won=False, return_pct=-0.05, current_bankroll=1.0)
    
    # Should be significantly reduced due to 5-loss streak
    assert strategy.get_streak_factor() < 0.7


def test_volatility_factor():
    """Test that volatility factor decreases with increasing volatility."""
    strategy = DynamicBankrollManagement(
        base_fraction=0.1,
        payoff=1,
        loss=1
    )
    
    # Initially volatility factor should be 1.0 (not enough data)
    assert strategy.get_volatility_factor() == 1.0
    
    # Record stable returns
    for _ in range(5):
        strategy.record_result(won=True, return_pct=0.05, current_bankroll=1.05)
    
    # Low volatility should give a higher factor
    low_volatility_factor = strategy.get_volatility_factor()
    assert low_volatility_factor > 0.9
    
    # Reset strategy
    strategy = DynamicBankrollManagement(
        base_fraction=0.1,
        payoff=1,
        loss=1
    )
    
    # Record highly volatile returns
    strategy.record_result(won=True, return_pct=0.2, current_bankroll=1.2)
    strategy.record_result(won=False, return_pct=-0.15, current_bankroll=1.05)
    strategy.record_result(won=True, return_pct=0.25, current_bankroll=1.3)
    strategy.record_result(won=False, return_pct=-0.2, current_bankroll=1.1)
    strategy.record_result(won=True, return_pct=0.15, current_bankroll=1.25)
    
    # High volatility should give a lower factor
    high_volatility_factor = strategy.get_volatility_factor()
    assert high_volatility_factor < low_volatility_factor


def test_drawdown_factor():
    """Test that drawdown factor decreases as drawdown increases."""
    strategy = DynamicBankrollManagement(
        base_fraction=0.1,
        payoff=1,
        loss=1
    )
    
    # Initially at peak, so drawdown factor should be 1.0
    assert strategy.get_drawdown_factor() == 1.0
    
    # At peak
    strategy.current_bankroll = 1.0
    strategy.peak_bankroll = 1.0
    assert strategy.get_drawdown_factor() == 1.0
    
    # Small drawdown (10%)
    strategy.current_bankroll = 0.9
    strategy.peak_bankroll = 1.0
    small_drawdown_factor = strategy.get_drawdown_factor()
    assert small_drawdown_factor < 1.0
    assert small_drawdown_factor > 0.7
    
    # Large drawdown (30%)
    strategy.current_bankroll = 0.7
    strategy.peak_bankroll = 1.0
    large_drawdown_factor = strategy.get_drawdown_factor()
    assert large_drawdown_factor < small_drawdown_factor
    
    # Above peak
    strategy.current_bankroll = 1.1
    strategy.peak_bankroll = 1.0
    # Should update peak and return 1.0
    assert strategy.get_drawdown_factor() == 1.0
    assert strategy.peak_bankroll == 1.1


def test_combined_adjustments():
    """Test how multiple factors combine to adjust the base fraction."""
    strategy = DynamicBankrollManagement(
        base_fraction=0.1,
        payoff=1,
        loss=1,
        max_fraction=0.3,  # Cap at 30%
        min_fraction=0.01  # Floor at 1%
    )
    
    # Simulate a favorable scenario (winning streak, low volatility, no drawdown)
    for _ in range(5):
        strategy.record_result(won=True, return_pct=0.05, current_bankroll=1.0 + (_ + 1) * 0.05)
    
    favorable_bet = strategy.evaluate(0.6)
    # Should be above base fraction due to positive adjustments
    assert favorable_bet > 0.1
    
    # Simulate an unfavorable scenario (losing streak, high volatility, significant drawdown)
    strategy = DynamicBankrollManagement(
        base_fraction=0.1,
        payoff=1,
        loss=1,
        max_fraction=0.3,
        min_fraction=0.01
    )
    
    # Set peak higher than current to simulate drawdown
    strategy.peak_bankroll = 1.5
    strategy.current_bankroll = 1.0
    
    # Record volatile losing streak
    for i in range(5):
        strategy.record_result(
            won=False,
            return_pct=-0.1 - (i * 0.02),  # Increasing losses
            current_bankroll=1.0 - (i + 1) * 0.1
        )
    
    unfavorable_bet = strategy.evaluate(0.6)
    # Should be below base fraction due to negative adjustments
    assert unfavorable_bet < 0.1
    
    # But should not go below min_fraction
    assert unfavorable_bet >= 0.01


def test_min_max_bounds():
    """Test that bet size is properly bounded by min and max fractions."""
    strategy = DynamicBankrollManagement(
        base_fraction=0.1,
        payoff=1,
        loss=1,
        max_fraction=0.15,
        min_fraction=0.05
    )
    
    # Force a very positive scenario
    strategy.get_streak_factor = lambda: 2.0  # Mock a strong winning streak
    strategy.get_volatility_factor = lambda: 1.5  # Mock low volatility
    strategy.get_drawdown_factor = lambda: 1.5  # Mock no drawdown
    
    # Combined adjustment would be 0.1 * 2.0 * 1.5 * 1.5 = 0.45, but should be capped at 0.15
    assert strategy.evaluate(0.6) == pytest.approx(0.15)
    
    # Force a very negative scenario
    strategy.get_streak_factor = lambda: 0.5  # Mock a losing streak
    strategy.get_volatility_factor = lambda: 0.5  # Mock high volatility
    strategy.get_drawdown_factor = lambda: 0.5  # Mock significant drawdown
    
    # Combined adjustment would be 0.1 * 0.5 * 0.5 * 0.5 = 0.0125, but should be floored at 0.05
    assert strategy.evaluate(0.6) == pytest.approx(0.05)


def test_invalid_parameters():
    """Test that invalid parameters raise appropriate exceptions."""
    payoff = 1
    loss = 1
    
    # Base fraction must be between 0 and 1
    with pytest.raises(ValueError, match="Base fraction must be between 0 and 1"):
        DynamicBankrollManagement(base_fraction=0, payoff=payoff, loss=loss)
    
    with pytest.raises(ValueError, match="Base fraction must be between 0 and 1"):
        DynamicBankrollManagement(base_fraction=1.1, payoff=payoff, loss=loss)
    
    # Min fraction must be <= max fraction, both between 0 and 1
    with pytest.raises(ValueError, match="Min fraction must be <= max fraction"):
        DynamicBankrollManagement(
            base_fraction=0.1,
            payoff=payoff,
            loss=loss,
            min_fraction=0.2,
            max_fraction=0.1
        )
    
    # Window size must be positive
    with pytest.raises(ValueError, match="Window size must be positive"):
        DynamicBankrollManagement(
            base_fraction=0.1,
            payoff=payoff,
            loss=loss,
            window_size=0
        )


def test_window_size_limiting():
    """Test that history is limited to the window size."""
    window_size = 3
    strategy = DynamicBankrollManagement(
        base_fraction=0.1,
        payoff=1,
        loss=1,
        window_size=window_size
    )
    
    # Add more results than the window size
    for i in range(5):
        strategy.record_result(won=True, return_pct=0.1, current_bankroll=1.0 + (i + 1) * 0.1)
    
    # History should be limited to window size
    assert len(strategy.results_history) == window_size
    assert len(strategy.returns_history) == window_size
    
    # Should only have the most recent entries
    assert strategy.returns_history == [0.1, 0.1, 0.1] 