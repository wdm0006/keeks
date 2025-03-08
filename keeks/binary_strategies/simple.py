from keeks.binary_strategies.base import BaseStrategy
import numpy as np

__author__ = "willmcginnis"


class NaiveStrategy(BaseStrategy):
    """
    A simple binary betting strategy that bets the full amount when expected value is positive.

    This strategy uses a naive approach where it bets the maximum allowed amount
    if the expected value is positive, and nothing if the expected value is negative.
    It can also bet against an outcome if the inverse bet has positive expected value.

    Parameters
    ----------
    payoff : float
        The amount won per unit bet on a successful outcome.
    loss : float
        The amount lost per unit bet on an unsuccessful outcome.
    transaction_cost : float
        The fixed cost per transaction, regardless of outcome.
    """

    def __init__(self, payoff, loss, transaction_cost):
        """
        Initialize the NaiveStrategy with payoff, loss, and transaction cost parameters.

        Parameters
        ----------
        payoff : float
            The amount won per unit bet on a successful outcome.
        loss : float
            The amount lost per unit bet on an unsuccessful outcome.
        transaction_cost : float
            The fixed cost per transaction, regardless of outcome.
        """
        self.payoff = payoff
        self.loss = loss
        self.transaction_cost = transaction_cost

    def evaluate(self, probability):
        """
        Evaluate whether to bet based on expected value calculation.

        Calculates the expected value of the bet and returns:
        - 1.0 if betting on the outcome has positive expected value
        - -1.0 if betting against the outcome has positive expected value
        - 0.0 if neither bet has positive expected value

        Parameters
        ----------
        probability : float
            The probability of a successful outcome, typically between 0 and 1.

        Returns
        -------
        float
            Either 1.0 (bet full amount), -1.0 (bet against), or 0.0 (no bet).
        """
        expected_value = (
            (self.payoff * probability)
            - (self.loss * (1 - probability))
            - self.transaction_cost
        )
        expected_value_neg = (
            (self.payoff * (1 - probability))
            - (self.loss * probability)
            - self.transaction_cost
        )

        if expected_value > 0:
            return 1.0
        elif expected_value_neg > 0:
            return -1.0
        else:
            return 0.0


class FixedFractionStrategy(BaseStrategy):
    """
    A simple strategy that bets a fixed percentage of the bankroll.
    
    This strategy ignores probabilities and odds completely, and simply
    wagers a fixed fraction of the bankroll. It can be used as a baseline
    for comparison or as a simple approach for risk management.
    
    Parameters
    ----------
    fraction : float
        The fixed fraction of the bankroll to bet (between 0 and 1).
    min_probability : float, default=0.5
        The minimum probability required to place a bet.
    """
    
    def __init__(self, fraction, min_probability=0.5):
        """
        Initialize the FixedFractionStrategy.
        
        Parameters
        ----------
        fraction : float
            The fixed fraction of the bankroll to bet (between 0 and 1).
        min_probability : float, default=0.5
            The minimum probability required to place a bet.
        """
        if not 0 <= fraction <= 1:
            raise ValueError("Fraction must be between 0 and 1")
        if not 0 <= min_probability <= 1:
            raise ValueError("Minimum probability must be between 0 and 1")
        
        self.fraction = fraction
        self.min_probability = min_probability
    
    def evaluate(self, probability):
        """
        Return the fixed fraction if the probability meets the minimum threshold.
        
        Parameters
        ----------
        probability : float
            The probability of a successful outcome, typically between 0 and 1.
            
        Returns
        -------
        float
            The fixed fraction if probability >= min_probability, otherwise 0.
        """
        if probability >= self.min_probability:
            return self.fraction
        else:
            return 0.0


class CPPIStrategy(BaseStrategy):
    """
    Constant Proportion Portfolio Insurance (CPPI) strategy for bankroll allocation.
    
    CPPI is a dynamic strategy that allocates funds based on a cushion value
    above a fixed floor. The strategy aims to maintain a minimum amount of capital
    (the floor) while allowing for upside participation through risky bets.
    
    The amount to bet is calculated as: multiplier * (current_bankroll - floor)
    
    Parameters
    ----------
    floor_fraction : float
        The fraction of the initial bankroll to protect as the floor (between 0 and 1).
    multiplier : float
        The leverage multiplier applied to the cushion (typically between 1 and 5).
    initial_bankroll : float
        The starting bankroll amount to calculate the initial floor.
    min_probability : float, default=0.5
        The minimum probability required to place a bet.
    """
    
    def __init__(self, floor_fraction, multiplier, initial_bankroll, min_probability=0.5):
        """
        Initialize the CPPI strategy.
        
        Parameters
        ----------
        floor_fraction : float
            The fraction of the initial bankroll to protect as the floor (between 0 and 1).
        multiplier : float
            The leverage multiplier applied to the cushion (typically between 1 and 5).
        initial_bankroll : float
            The starting bankroll amount to calculate the initial floor.
        min_probability : float, default=0.5
            The minimum probability required to place a bet.
        """
        if not 0 <= floor_fraction < 1:
            raise ValueError("Floor fraction must be between 0 and 1")
        if multiplier <= 0:
            raise ValueError("Multiplier must be greater than 0")
        if initial_bankroll <= 0:
            raise ValueError("Initial bankroll must be greater than 0")
        
        self.floor_fraction = floor_fraction
        self.multiplier = multiplier
        self.floor = floor_fraction * initial_bankroll
        self.min_probability = min_probability
        self.current_bankroll = initial_bankroll
    
    def update_bankroll(self, new_bankroll):
        """
        Update the current bankroll value to calculate future allocations.
        
        Parameters
        ----------
        new_bankroll : float
            The current value of the bankroll.
        """
        self.current_bankroll = new_bankroll
    
    def evaluate(self, probability):
        """
        Calculate the CPPI bet size based on the cushion above the floor.
        
        Parameters
        ----------
        probability : float
            The probability of a successful outcome, typically between 0 and 1.
            
        Returns
        -------
        float
            The proportion of the current bankroll to bet, or 0 if below
            the minimum probability threshold.
        """
        if probability < self.min_probability:
            return 0.0
        
        # Calculate the cushion (amount above the floor)
        cushion = max(0, self.current_bankroll - self.floor)
        
        # Calculate the exposure (amount to bet)
        exposure = self.multiplier * cushion
        
        # Convert to a proportion of the current bankroll
        if self.current_bankroll <= 0:
            return 0.0
        
        proportion = min(1.0, exposure / self.current_bankroll)
        return proportion


class DynamicBankrollManagement(BaseStrategy):
    """
    Dynamic Bankroll Management strategy that adapts bet sizing based on recent performance.
    
    This strategy adjusts the bet size based on:
    1. Recent win/loss streak
    2. Volatility of recent returns
    3. Current drawdown from peak
    
    The strategy becomes more conservative during losing streaks, high volatility,
    or significant drawdowns, and more aggressive during winning streaks, low
    volatility, and when near equity peaks.
    
    Parameters
    ----------
    base_fraction : float
        The baseline fraction of bankroll to bet (between 0 and 1).
    payoff : float
        The amount won per unit bet on a successful outcome.
    loss : float
        The amount lost per unit bet on an unsuccessful outcome.
    window_size : int, default=10
        The number of recent bets to consider for performance evaluation.
    max_fraction : float, default=0.25
        The maximum fraction of bankroll that can be bet.
    min_fraction : float, default=0.01
        The minimum fraction of bankroll that will be bet if a bet is placed.
    min_probability : float, default=0.5
        The minimum probability required to place a bet.
    """
    
    def __init__(
        self, 
        base_fraction, 
        payoff, 
        loss, 
        window_size=10,
        max_fraction=0.25, 
        min_fraction=0.01,
        min_probability=0.5
    ):
        """
        Initialize the DynamicBankrollManagement strategy.
        
        Parameters
        ----------
        base_fraction : float
            The baseline fraction of bankroll to bet (between 0 and 1).
        payoff : float
            The amount won per unit bet on a successful outcome.
        loss : float
            The amount lost per unit bet on an unsuccessful outcome.
        window_size : int, default=10
            The number of recent bets to consider for performance evaluation.
        max_fraction : float, default=0.25
            The maximum fraction of bankroll that can be bet.
        min_fraction : float, default=0.01
            The minimum fraction of bankroll that will be bet if a bet is placed.
        min_probability : float, default=0.5
            The minimum probability required to place a bet.
        """
        if not 0 < base_fraction <= 1:
            raise ValueError("Base fraction must be between 0 and 1")
        if not 0 < min_fraction <= max_fraction <= 1:
            raise ValueError("Min fraction must be <= max fraction, both between 0 and 1")
        if window_size < 1:
            raise ValueError("Window size must be positive")
            
        self.base_fraction = base_fraction
        self.payoff = payoff
        self.loss = loss
        self.window_size = window_size
        self.max_fraction = max_fraction
        self.min_fraction = min_fraction
        self.min_probability = min_probability
        
        # Performance tracking
        self.results_history = []  # 1 for win, -1 for loss
        self.returns_history = []  # Percentage returns
        self.peak_bankroll = 1.0   # Start at 1.0 (will be updated)
        self.current_bankroll = 1.0
        
    def record_result(self, won, return_pct, current_bankroll):
        """
        Record the result of a bet to update the strategy's state.
        
        Parameters
        ----------
        won : bool
            Whether the bet was won (True) or lost (False).
        return_pct : float
            The return as a percentage of the bankroll.
        current_bankroll : float
            The current bankroll value after the bet.
        """
        self.results_history.append(1 if won else -1)
        self.returns_history.append(return_pct)
        
        # Trim history to window size
        if len(self.results_history) > self.window_size:
            self.results_history.pop(0)
            self.returns_history.pop(0)
            
        # Update bankroll tracking
        self.current_bankroll = current_bankroll
        self.peak_bankroll = max(self.peak_bankroll, current_bankroll)
    
    def get_streak_factor(self):
        """
        Calculate a factor based on recent win/loss streak.
        
        Returns
        -------
        float
            A multiplier that increases with winning streaks and
            decreases with losing streaks.
        """
        if not self.results_history:
            return 1.0
            
        # Count consecutive results of the same sign
        streak = 0
        current_sign = self.results_history[-1]
        
        for i in range(len(self.results_history) - 1, -1, -1):
            if self.results_history[i] == current_sign:
                streak += 1
            else:
                break
                
        # Convert streak to a factor
        if current_sign > 0:  # Winning streak
            return min(1.5, 1.0 + (streak * 0.1))
        else:  # Losing streak
            return max(0.5, 1.0 - (streak * 0.1))
    
    def get_volatility_factor(self):
        """
        Calculate a factor based on recent return volatility.
        
        Returns
        -------
        float
            A multiplier that decreases as volatility increases.
        """
        if len(self.returns_history) < 2:
            return 1.0
            
        # Calculate standard deviation of returns
        std_dev = np.std(self.returns_history) if self.returns_history else 0
        
        # Convert to a factor (higher volatility = lower factor)
        return max(0.5, min(1.5, 1.0 - (std_dev * 2)))
    
    def get_drawdown_factor(self):
        """
        Calculate a factor based on current drawdown from peak.
        
        Returns
        -------
        float
            A multiplier that decreases as drawdown increases.
        """
        if self.current_bankroll >= self.peak_bankroll:
            # Update peak when current bankroll is higher
            self.peak_bankroll = self.current_bankroll
            return 1.0
            
        # Calculate current drawdown percentage
        drawdown_pct = 1.0 - (self.current_bankroll / self.peak_bankroll)
        
        # Convert to a factor (higher drawdown = lower factor)
        return max(0.5, 1.0 - (drawdown_pct * 2))
    
    def evaluate(self, probability):
        """
        Calculate the dynamic bet size based on recent performance metrics.
        
        Parameters
        ----------
        probability : float
            The probability of a successful outcome, typically between 0 and 1.
            
        Returns
        -------
        float
            The dynamically adjusted proportion of the bankroll to bet.
        """
        if probability < self.min_probability:
            return 0.0
            
        # Calculate adjustment factors
        streak_factor = self.get_streak_factor()
        volatility_factor = self.get_volatility_factor()
        drawdown_factor = self.get_drawdown_factor()
        
        # Apply all factors to the base fraction
        adjusted_fraction = self.base_fraction * streak_factor * volatility_factor * drawdown_factor
        
        # Ensure result is within bounds
        return max(self.min_fraction, min(self.max_fraction, adjusted_fraction))
