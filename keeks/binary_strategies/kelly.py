from keeks.binary_strategies.base import BaseStrategy

__author__ = "willmcginnis"


class KellyCriterion(BaseStrategy):
    """
    Implementation of the Kelly Criterion for binary betting.

    The Kelly Criterion is a formula for bet sizing that maximizes the expected
    logarithm of wealth. It calculates the optimal fraction of the bankroll to
    bet based on the probability of winning and the payoff odds.

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
        self.payoff = payoff
        self.loss = loss
        self.transaction_cost = transaction_cost

    def evaluate(self, probability):
        """
        Calculate the optimal bet size using the Kelly Criterion.

        The Kelly formula for binary outcomes:
        f* = (bp - q) / b
        where:
        - b = net odds received on the wager (payoff/loss)
        - p = probability of winning
        - q = probability of losing (1-p)

        The formula is adjusted to account for transaction costs.

        Parameters
        ----------
        probability : float
            The probability of a successful outcome, typically between 0 and 1.

        Returns
        -------
        float
            The optimal proportion of the bankroll to bet, adjusted for transaction costs.
            Returns a negative value if the expected value is negative.
        """

        b = self.payoff / self.loss
        p = probability
        q = 1 - p

        expected_kelly = (b * p - q) / b

        # Adjust for transaction costs
        if expected_kelly > 0:
            expected_kelly -= self.transaction_cost / self.payoff

        return expected_kelly


class FractionalKellyCriterion(BaseStrategy):
    """
    Implementation of the Fractional Kelly Criterion for binary betting.

    The Fractional Kelly Criterion applies a fraction to the full Kelly bet size,
    which can reduce volatility at the expense of long-term growth rate.

    Parameters
    ----------
    payoff : float
        The amount won per unit bet on a successful outcome.
    loss : float
        The amount lost per unit bet on an unsuccessful outcome.
    transaction_cost : float
        The fixed cost per transaction, regardless of outcome.
    fraction : float
        The fraction of the full Kelly bet to use (typically between 0 and 1).
    """

    def __init__(self, payoff, loss, transaction_cost, fraction):
        self.payoff = payoff
        self.loss = loss
        self.transaction_cost = transaction_cost
        self.fraction = fraction

    def evaluate(self, probability):
        """
        Calculate the fractional Kelly bet size.

        Parameters
        ----------
        probability : float
            The probability of a successful outcome, typically between 0 and 1.

        Returns
        -------
        float
            The optimal proportion of the bankroll to bet, multiplied by the fraction parameter.
        """
        kelly = KellyCriterion(self.payoff, self.loss, self.transaction_cost)
        return self.fraction * kelly.evaluate(probability)


class DrawdownAdjustedKelly(BaseStrategy):
    """
    Drawdown-Adjusted Kelly Criterion for binary betting.
    
    This variant of the Kelly Criterion adjusts the bet size based on the user's
    maximum acceptable drawdown, effectively managing risk more conservatively
    than the standard Kelly approach.
    
    The adjustment is based on the relationship between drawdown and bet size
    as described in risk management literature. The formula approximates the
    relationship between maximum drawdown and the fraction of Kelly to use.
    
    Parameters
    ----------
    payoff : float
        The amount won per unit bet on a successful outcome.
    loss : float
        The amount lost per unit bet on an unsuccessful outcome.
    transaction_cost : float
        The fixed cost per transaction, regardless of outcome.
    max_acceptable_drawdown : float
        The maximum drawdown the investor is willing to tolerate,
        expressed as a fraction between 0 and 1 (e.g., 0.2 for 20%).
    """
    
    def __init__(self, payoff, loss, transaction_cost, max_acceptable_drawdown):
        """
        Initialize the DrawdownAdjustedKelly strategy.
        
        Parameters
        ----------
        payoff : float
            The amount won per unit bet on a successful outcome.
        loss : float
            The amount lost per unit bet on an unsuccessful outcome.
        transaction_cost : float
            The fixed cost per transaction, regardless of outcome.
        max_acceptable_drawdown : float
            The maximum drawdown the investor is willing to tolerate,
            expressed as a fraction between 0 and 1 (e.g., 0.2 for 20%).
        """
        if not 0 < max_acceptable_drawdown < 1:
            raise ValueError("Maximum acceptable drawdown must be between 0 and 1")
            
        self.payoff = payoff
        self.loss = loss
        self.transaction_cost = transaction_cost
        self.max_acceptable_drawdown = max_acceptable_drawdown
        
    def evaluate(self, probability):
        """
        Calculate the drawdown-adjusted Kelly bet size.
        
        The adjustment is an approximation based on the relationship between
        bet size and expected drawdown in repeated betting scenarios.
        
        Parameters
        ----------
        probability : float
            The probability of a successful outcome, typically between 0 and 1.
            
        Returns
        -------
        float
            The drawdown-adjusted proportion of the bankroll to bet.
        """
        # Calculate the standard Kelly bet size
        kelly = KellyCriterion(self.payoff, self.loss, self.transaction_cost)
        full_kelly = kelly.evaluate(probability)
        
        # Adjust the Kelly fraction based on maximum acceptable drawdown
        # This is a simplified approximation of the relationship
        # Various research suggests specific formulas, but a common
        # conservative approach is to scale Kelly by max drawdown / 0.5
        # (since full Kelly has an expected drawdown of around 50%)
        drawdown_factor = min(1.0, self.max_acceptable_drawdown / 0.5)
        
        # Apply the drawdown adjustment
        return drawdown_factor * full_kelly


class OptimalF(BaseStrategy):
    """
    Implementation of Ralph Vince's Optimal f for binary betting.
    
    Optimal f is a strategy that aims to maximize the geometric growth rate
    of a bankroll over time. It's similar to the Kelly Criterion but can be
    more adaptable to certain scenarios and often less aggressive in its allocations.
    
    The strategy uses past or simulated performance data to find the fraction
    that would have maximized growth, then applies that fraction to future bets.
    
    Parameters
    ----------
    payoff : float
        The amount won per unit bet on a successful outcome.
    loss : float
        The amount lost per unit bet on an unsuccessful outcome.
    transaction_cost : float
        The fixed cost per transaction, regardless of outcome.
    win_rate : float
        The historical or expected win rate (between 0 and 1).
    max_risk_fraction : float, default=0.2
        The maximum fraction of bankroll that can be risked on a single bet.
    """
    
    def __init__(self, payoff, loss, transaction_cost, win_rate, max_risk_fraction=0.2):
        """
        Initialize the OptimalF strategy.
        
        Parameters
        ----------
        payoff : float
            The amount won per unit bet on a successful outcome.
        loss : float
            The amount lost per unit bet on an unsuccessful outcome.
        transaction_cost : float
            The fixed cost per transaction, regardless of outcome.
        win_rate : float
            The historical or expected win rate (between 0 and 1).
        max_risk_fraction : float, default=0.2
            The maximum fraction of bankroll that can be risked on a single bet.
        """
        if not 0 <= win_rate <= 1:
            raise ValueError("Win rate must be between 0 and 1")
        if not 0 < max_risk_fraction <= 1:
            raise ValueError("Maximum risk fraction must be between 0 and 1")
        
        self.payoff = payoff
        self.loss = loss
        self.transaction_cost = transaction_cost
        self.win_rate = win_rate
        self.max_risk_fraction = max_risk_fraction
        
    def evaluate(self, probability):
        """
        Calculate the optimal f bet size based on win rate and payoff ratio.
        
        This implementation uses a simplified approach to Optimal f, where
        we calculate it as (expected_win_rate * payoff - expected_loss_rate) / payoff.
        
        Parameters
        ----------
        probability : float
            The probability of a successful outcome, typically between 0 and 1.
            
        Returns
        -------
        float
            The optimal proportion of the bankroll to bet.
        """
        # Use the provided probability to adjust our expectations
        # If no specific edge is known, the win_rate will be used directly
        adjusted_win_rate = probability if probability > 0 else self.win_rate
        
        # Calculate expected loss rate
        expected_loss_rate = 1 - adjusted_win_rate
        
        # Calculate the risk-to-reward ratio
        reward_to_risk = self.payoff / self.loss
        
        # Calculate optimal f - basic form: (W * R - L) / R
        # where W = win rate, L = loss rate, R = reward-to-risk ratio
        optimal_f = (adjusted_win_rate * reward_to_risk - expected_loss_rate) / reward_to_risk
        
        # Adjust for transaction costs
        if optimal_f > 0:
            optimal_f -= self.transaction_cost / self.payoff
        
        # Cap at our maximum risk fraction
        optimal_f = min(max(0, optimal_f), self.max_risk_fraction)
        
        return optimal_f
