import numpy as np

from keeks.binary_strategies.base import BaseStrategy

__author__ = "willmcginnis"


class NaiveStrategy(BaseStrategy):
    """
    A simple betting strategy that bets based on expected value.

    This strategy calculates the expected value of a bet and bets a fixed
    fraction of the bankroll if the expected value is positive.

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
        Initialize the NaiveStrategy.

        Parameters
        ----------
        payoff : float
            The amount won per unit bet on a successful outcome.
        loss : float
            The amount lost per unit bet on an unsuccessful outcome.
        transaction_cost : float
            The fixed cost per transaction, regardless of outcome.
        """
        super().__init__(payoff, loss, transaction_cost)

    def evaluate(self, probability, current_bankroll):
        """
        Calculate the bet size based on expected value.

        The expected value is calculated as:
        EV = (probability * payoff) - ((1 - probability) * loss) - transaction_cost

        If EV is positive, bet a fixed fraction of the bankroll.
        If EV is negative, do not bet.

        Parameters
        ----------
        probability : float
            The probability of a successful outcome, typically between 0 and 1.
        current_bankroll : float
            The current bankroll amount.

        Returns
        -------
        float
            The proportion of the bankroll to bet.
        """
        # Calculate expected value
        expected_value = (
            (probability * self.payoff)
            - ((1 - probability) * self.loss)
            - self.transaction_cost
        )

        # If expected value is negative, do not bet
        if expected_value <= 0:
            return 0.0

        # Calculate bet size based on expected value
        bet_size = expected_value / self.payoff

        # Ensure we never bet more than would result in negative bankroll
        return min(max(0, bet_size), self.get_max_safe_bet(current_bankroll))

    def calculate_max_entry_price(self, outcomes, probabilities, current_wealth,
                                  tolerance=0.01, max_search_fraction=0.5):
        """
        Calculate maximum price willing to pay for a one-time gamble.

        Naive strategy is risk-neutral and simply pays the expected value.

        Parameters
        ----------
        outcomes : array-like
            The possible payoffs from the gamble
        probabilities : array-like
            The probability of each outcome (must sum to ≤ 1)
        current_wealth : float
            Current wealth before the gamble
        tolerance : float, default=0.01
            Convergence tolerance (unused, kept for API consistency)
        max_search_fraction : float, default=0.5
            Maximum fraction of wealth to consider as upper bound (ignored in this method)

        Returns
        -------
        float
            Maximum price willing to pay (the expected value)

        Notes
        -----
        Naive strategy is risk-neutral: willing to pay exactly the expected value.
        This is the classic "expected value maximizer" that doesn't account for risk.
        """
        import numpy as np

        # Suppress unused parameter warning - kept for API consistency
        _ = tolerance

        outcomes = np.array(outcomes)
        probabilities = np.array(probabilities)

        # Calculate expected value
        expected_value = np.sum(probabilities * outcomes)

        # For naive strategy, willing to pay expected value
        # This can be infinite (e.g., St. Petersburg paradox)
        # We cap at current_wealth for practical reasons
        return min(current_wealth, max(0.0, expected_value))


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
    payoff : float
        The amount won per unit bet on a successful outcome.
    loss : float
        The amount lost per unit bet on an unsuccessful outcome.
    transaction_cost : float, default=0
        The fixed cost per transaction, regardless of outcome.
    min_probability : float, default=0.5
        The minimum probability required to place a bet.
    """

    def __init__(self, fraction, payoff, loss, transaction_cost=0, min_probability=0.5):
        """
        Initialize the FixedFractionStrategy.

        Parameters
        ----------
        fraction : float
            The fixed fraction of the bankroll to bet (between 0 and 1).
        payoff : float
            The amount won per unit bet on a successful outcome.
        loss : float
            The amount lost per unit bet on an unsuccessful outcome.
        transaction_cost : float, default=0
            The fixed cost per transaction, regardless of outcome.
        min_probability : float, default=0.5
            The minimum probability required to place a bet.
        """
        if not 0 <= fraction <= 1:
            raise ValueError("Fraction must be between 0 and 1")
        if not 0 <= min_probability <= 1:
            raise ValueError("Minimum probability must be between 0 and 1")

        super().__init__(payoff, loss, transaction_cost)
        self.fraction = fraction
        self.min_probability = min_probability

    def evaluate(self, probability, current_bankroll):
        """
        Return the fixed fraction if the probability meets the minimum threshold.

        Parameters
        ----------
        probability : float
            The probability of a successful outcome, typically between 0 and 1.
        current_bankroll : float
            The current bankroll amount.

        Returns
        -------
        float
            The fixed fraction if probability >= min_probability, otherwise 0.
        """
        if probability >= self.min_probability:
            # Ensure we never bet more than would result in negative bankroll
            return min(self.fraction, self.get_max_safe_bet(current_bankroll))
        else:
            return 0.0

    def calculate_max_entry_price(self, outcomes, probabilities, current_wealth,
                                  tolerance=0.01, max_search_fraction=0.5):
        """
        Calculate maximum price willing to pay for a one-time gamble.

        Fixed fraction strategy simply pays a fixed fraction of current wealth.

        Parameters
        ----------
        outcomes : array-like
            The possible payoffs from the gamble (unused)
        probabilities : array-like
            The probability of each outcome (unused)
        current_wealth : float
            Current wealth before the gamble
        tolerance : float, default=0.01
            Convergence tolerance (unused, kept for API consistency)
        max_search_fraction : float, default=0.5
            Maximum fraction of wealth to consider as upper bound

        Returns
        -------
        float
            Maximum price willing to pay (fixed fraction of wealth)

        Notes
        -----
        Fixed fraction strategy doesn't analyze the gamble - it simply
        commits a fixed fraction of wealth regardless of the opportunity.
        This is a mechanical rule-based approach, not optimization-based.
        """
        # Suppress unused parameter warnings - kept for API consistency
        _ = outcomes, probabilities, tolerance

        # Pay the fixed fraction of current wealth
        return min(self.fraction * current_wealth, max_search_fraction * current_wealth)


class CPPIStrategy(BaseStrategy):
    """
    Implementation of Constant Proportion Portfolio Insurance (CPPI) for binary betting.

    This strategy maintains a dynamic floor below which the bankroll should not fall,
    and invests a multiple of the cushion (amount above floor) in each bet.

    Parameters
    ----------
    floor_fraction : float
        The fraction of initial bankroll to maintain as a floor.
    multiplier : float
        The multiplier to apply to the cushion for determining exposure.
    initial_bankroll : float
        The initial bankroll amount.
    payoff : float
        The amount won per unit bet on a successful outcome.
    loss : float
        The amount lost per unit bet on an unsuccessful outcome.
    transaction_cost : float
        The fixed cost per transaction, regardless of outcome.
    min_probability : float, default=0.5
        The minimum probability required to place a bet.
    """

    def __init__(
        self,
        floor_fraction,
        multiplier,
        initial_bankroll,
        payoff,
        loss,
        transaction_cost=0,
        min_probability=0.5,
    ):
        """Initialize the CPPI strategy."""
        if not 0 < floor_fraction < 1:
            raise ValueError("Floor fraction must be between 0 and 1")
        if multiplier <= 0:
            raise ValueError("Multiplier must be greater than 0")
        if initial_bankroll <= 0:
            raise ValueError("Initial bankroll must be greater than 0")
        if not 0 <= min_probability <= 1:
            raise ValueError("Minimum probability must be between 0 and 1")

        super().__init__(payoff, loss, transaction_cost)
        self.floor_fraction = floor_fraction
        self.multiplier = multiplier
        self.floor = floor_fraction * initial_bankroll
        self.min_probability = min_probability
        self.current_bankroll = initial_bankroll
        self.peak_bankroll = initial_bankroll

    def update_bankroll(self, new_bankroll):
        """
        Update the current bankroll value and adjust floor based on peak value.

        Parameters
        ----------
        new_bankroll : float
            The current value of the bankroll.
        """
        self.current_bankroll = new_bankroll
        if new_bankroll > self.peak_bankroll:
            self.peak_bankroll = new_bankroll
            # Adjust floor to maintain the same fraction of peak value
            self.floor = self.floor_fraction * self.peak_bankroll

    def evaluate(self, probability, current_bankroll):
        """
        Calculate the CPPI bet size based on the cushion above the floor.

        Parameters
        ----------
        probability : float
            The probability of a successful outcome, typically between 0 and 1.
        current_bankroll : float
            The current bankroll amount.

        Returns
        -------
        float
            The proportion of the current bankroll to bet, or 0 if below
            the minimum probability threshold.
        """
        # Update current bankroll
        self.update_bankroll(current_bankroll)

        # Check probability threshold
        if probability < self.min_probability:
            return 0.0

        # Calculate expected value per unit bet
        expected_value = probability * (self.payoff - self.transaction_cost) - (
            1 - probability
        ) * (self.loss + self.transaction_cost)

        # If expected value is negative, don't bet
        # Note: We allow small positive edges (even < 1%) for realistic scenarios
        # Professional bettors often operate with 0.5-2% edges
        if expected_value <= 0:
            return 0.0

        # Calculate the cushion (amount above the floor)
        cushion = max(0, current_bankroll - self.floor)

        # Calculate the exposure (amount to bet)
        # Scale the multiplier by the expected value to be more conservative
        # when the edge is smaller
        adjusted_multiplier = self.multiplier * min(1.0, expected_value)
        exposure = adjusted_multiplier * cushion

        # Convert to a proportion of the current bankroll
        if current_bankroll <= 0:
            return 0.0

        # Calculate proportion
        proportion = min(1.0, exposure / current_bankroll)

        # Adjust for transaction costs and ensure we don't risk going below floor
        if proportion > 0:
            # Calculate the maximum bet that ensures we stay above the floor
            # even in the worst case (loss + transaction cost)
            max_floor_bet = (current_bankroll - self.floor) / (
                current_bankroll * (self.loss + self.transaction_cost)
            )

            # Get the maximum safe bet considering ruin
            max_safe_bet = self.get_max_safe_bet(current_bankroll)

            # Take the minimum of all constraints
            proportion = min(proportion, max_floor_bet, max_safe_bet)

        return max(0, proportion)

    def calculate_max_entry_price(self, outcomes, probabilities, current_wealth,
                                  tolerance=0.01, max_search_fraction=0.5):
        """
        Calculate maximum price willing to pay for a one-time gamble.

        CPPI protects a floor value, so entry price is based on the cushion
        above the floor, scaled by the multiplier.

        Parameters
        ----------
        outcomes : array-like
            The possible payoffs from the gamble (unused in basic calculation)
        probabilities : array-like
            The probability of each outcome (unused in basic calculation)
        current_wealth : float
            Current wealth before the gamble
        tolerance : float, default=0.01
            Convergence tolerance (unused, kept for API consistency)
        max_search_fraction : float, default=0.5
            Maximum fraction of wealth to consider as upper bound

        Returns
        -------
        float
            Maximum price willing to pay (based on cushion above floor)

        Notes
        -----
        CPPI maintains a floor value and bets a multiple of the cushion.
        For entry price, we apply the same logic: pay multiplier × cushion,
        but never more than the cushion itself (to maintain floor).
        """
        # Suppress unused parameter warnings
        _ = outcomes, probabilities, tolerance

        # Update internal state with current wealth
        self.update_bankroll(current_wealth)

        # Calculate cushion above floor
        cushion = max(0, current_wealth - self.floor)

        # Pay up to multiplier × cushion, but capped at the cushion itself
        # (can't pay more than cushion without violating floor)
        max_price = min(self.multiplier * cushion, cushion)

        # Also respect max_search_fraction
        return min(max_price, max_search_fraction * current_wealth)


class DynamicBankrollManagement(BaseStrategy):
    """
    A dynamic bankroll management strategy that adjusts bet sizes based on performance.

    This strategy adjusts the base bet fraction based on:
    1. Recent performance (win/loss streak)
    2. Volatility of returns
    3. Current drawdown level
    4. Probability of success

    Parameters
    ----------
    base_fraction : float
        The base fraction of bankroll to bet.
    payoff : float
        The amount won per unit bet on a successful outcome.
    loss : float
        The amount lost per unit bet on an unsuccessful outcome.
    transaction_cost : float
        The fixed cost per transaction, regardless of outcome.
    window_size : int, default=10
        The number of recent results to consider for adjustments.
    max_fraction : float, default=0.2
        The maximum fraction of bankroll that can be bet.
    min_fraction : float, default=0.05
        The minimum fraction of bankroll to bet.
    """

    def __init__(
        self,
        base_fraction,
        payoff,
        loss,
        transaction_cost,
        window_size=10,
        max_fraction=0.2,
        min_fraction=0.05,
    ):
        """
        Initialize the DynamicBankrollManagement strategy.
        """
        if not 0 <= base_fraction <= 1:
            raise ValueError("Base fraction must be between 0 and 1")
        if not window_size > 0:
            raise ValueError("Window size must be positive")
        if not 0 <= min_fraction <= max_fraction <= 1:
            raise ValueError(
                "Min fraction must be between 0 and max fraction, and max fraction must be between 0 and 1"
            )

        super().__init__(payoff, loss, transaction_cost)
        self.base_fraction = base_fraction
        self.window_size = window_size
        self.max_fraction = max_fraction
        self.min_fraction = min_fraction
        self.results = []
        self.initial_bankroll = None
        self.current_bankroll = None
        self.peak_bankroll = None

    def record_result(self, won, return_pct=None):
        """
        Record the result of a bet.

        Parameters
        ----------
        won : bool
            Whether the bet was won.
        return_pct : float, optional
            The return percentage of the bet. If not provided, calculated from won/loss.
        """
        if return_pct is None:
            return_pct = self.payoff if won else -self.loss

        self.results.append(return_pct)
        if len(self.results) > self.window_size:
            self.results.pop(0)

    def get_streak_factor(self):
        """Calculate the adjustment factor based on recent performance."""
        if not self.results:
            return 1.0

        # Scale factor by window size to make smaller windows more responsive
        scale = min(len(self.results), self.window_size) / self.window_size

        wins = sum(1 for r in self.results if r > 0)
        losses = sum(1 for r in self.results if r < 0)

        if losses == 0:
            return 1.0 + (0.5 * scale)  # Maximum boost for all wins
        if wins == 0:
            return 1.0 - (0.5 * scale)  # Maximum reduction for all losses

        win_ratio = wins / (wins + losses)
        return 1.0 + ((win_ratio - 0.5) * scale)

    def get_volatility_factor(self):
        """Calculate the adjustment factor based on return volatility."""
        if not self.results:
            return 1.0

        returns = np.array(self.results)
        volatility = np.std(returns)

        if volatility == 0:
            return 1.0

        # Scale factor by window size to make smaller windows more responsive
        scale = min(len(self.results), self.window_size) / self.window_size
        return max(0.5, 1.0 - (volatility * scale))

    def get_drawdown_factor(self):
        """Calculate the adjustment factor based on current drawdown."""
        if self.current_bankroll is None or self.peak_bankroll is None:
            return 1.0

        drawdown = 1.0 - (self.current_bankroll / self.peak_bankroll)
        return max(0.5, 1.0 - drawdown)

    def get_probability_factor(self, probability):
        """Calculate the adjustment factor based on probability."""
        # Only apply probability factor if we have some results
        if not self.results:
            return 1.0

        # Scale linearly from 0.5 at 50% probability to 1.5 at 100% probability
        return max(0.5, min(1.5, 1.0 + (probability - 0.5)))

    def evaluate(self, probability, current_bankroll):
        """
        Calculate the bet size based on all adjustment factors.

        Parameters
        ----------
        probability : float
            The probability of a successful outcome.
        current_bankroll : float
            The current bankroll amount.

        Returns
        -------
        float
            The proportion of the current bankroll to bet.
        """
        # Initialize or update bankroll tracking
        if self.initial_bankroll is None:
            self.initial_bankroll = current_bankroll
            self.peak_bankroll = current_bankroll
        self.current_bankroll = current_bankroll
        self.peak_bankroll = max(self.peak_bankroll, current_bankroll)

        # Get all adjustment factors
        streak_factor = self.get_streak_factor()
        volatility_factor = self.get_volatility_factor()
        drawdown_factor = self.get_drawdown_factor()
        probability_factor = self.get_probability_factor(probability)

        # Combine all factors
        combined_factor = (
            streak_factor * volatility_factor * drawdown_factor * probability_factor
        )

        # Calculate adjusted bet size
        bet_size = self.base_fraction * combined_factor

        # Ensure bet size is within bounds
        return max(self.min_fraction, min(self.max_fraction, bet_size))

    def calculate_max_entry_price(self, outcomes, probabilities, current_wealth,
                                  tolerance=0.01, max_search_fraction=0.5):
        """
        Calculate maximum price willing to pay for a one-time gamble.

        Dynamic strategy adjusts based on history, but for a one-time decision
        with no history, we use the base fraction.

        Parameters
        ----------
        outcomes : array-like
            The possible payoffs from the gamble (unused)
        probabilities : array-like
            The probability of each outcome (unused)
        current_wealth : float
            Current wealth before the gamble
        tolerance : float, default=0.01
            Convergence tolerance (unused, kept for API consistency)
        max_search_fraction : float, default=0.5
            Maximum fraction of wealth to consider as upper bound

        Returns
        -------
        float
            Maximum price willing to pay (base fraction of wealth)

        Notes
        -----
        Dynamic strategy normally adjusts based on recent performance history.
        For a one-time decision with no history, we fall back to the base_fraction.
        This represents a neutral starting point before dynamic adjustments.
        """
        # Suppress unused parameter warnings
        _ = outcomes, probabilities, tolerance

        # Use base fraction since we have no history for a one-time decision
        return min(self.base_fraction * current_wealth, max_search_fraction * current_wealth)


class OptimalF(BaseStrategy):
    """
    Implementation of the Optimal f strategy for binary betting.

    This strategy calculates the optimal fraction of bankroll to bet based on
    the win rate and payoff ratio. It's similar to Kelly but uses a different
    approach to risk management.

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

        super().__init__(payoff, loss, transaction_cost)
        self.win_rate = win_rate
        self.max_risk_fraction = max_risk_fraction

    def evaluate(self, probability, current_bankroll):
        """
        Calculate the optimal f bet size based on win rate and payoff ratio.

        This implementation follows Ralph Vince's approach to Optimal f, which
        is based on maximizing the terminal wealth relative (TWR) for a given
        risk-to-reward profile.

        Parameters
        ----------
        probability : float
            The probability of a successful outcome, typically between 0 and 1.
        current_bankroll : float
            The current bankroll amount.

        Returns
        -------
        float
            The optimal proportion of the bankroll to bet.
        """
        if probability < 0.5:  # Use 0.5 as default minimum probability
            return 0.0

        # Use the provided probability to adjust our expectations
        # If no specific edge is known, the win_rate will be used directly
        adjusted_win_rate = probability if probability > 0 else self.win_rate
        adjusted_loss_rate = 1 - adjusted_win_rate

        # Calculate the risk-to-reward ratio (R-multiple)
        reward = self.payoff - self.transaction_cost
        risk = self.loss + self.transaction_cost

        # If transaction costs make the bet unprofitable, don't bet
        if reward <= 0:
            return 0.0

        # Calculate optimal f using the formula:
        # f* = W - (1-W)/(R/L) where W is win rate, R is reward, L is risk
        # This is more aligned with Ralph Vince's approach
        optimal_f = adjusted_win_rate - (adjusted_loss_rate / (reward / risk))

        # Cap at our maximum risk fraction
        optimal_f = min(max(0, optimal_f), self.max_risk_fraction)

        # Ensure we never bet more than would result in negative bankroll
        return min(optimal_f, self.get_max_safe_bet(current_bankroll))

    def calculate_max_entry_price(self, outcomes, probabilities, current_wealth,
                                  tolerance=0.01, max_search_fraction=0.5):
        """
        Calculate maximum price willing to pay for a one-time gamble.

        Optimal F maximizes geometric growth (like Kelly), so we use log utility.

        Parameters
        ----------
        outcomes : array-like
            The possible payoffs from the gamble
        probabilities : array-like
            The probability of each outcome (must sum to ≤ 1)
        current_wealth : float
            Current wealth before the gamble
        tolerance : float, default=0.01
            Convergence tolerance for binary search
        max_search_fraction : float, default=0.5
            Maximum fraction of wealth to consider as upper bound

        Returns
        -------
        float
            Maximum price willing to pay for the gamble

        Notes
        -----
        Optimal F, like Kelly Criterion, aims to maximize geometric growth,
        which corresponds to log utility (γ=1.0).
        """
        from keeks.utils import find_indifference_price

        return find_indifference_price(
            outcomes=outcomes,
            probabilities=probabilities,
            current_wealth=current_wealth,
            risk_aversion=1.0,  # Optimal F uses log utility like Kelly
            tolerance=tolerance,
            max_search_fraction=max_search_fraction
        )


class MertonShare(BaseStrategy):
    """
    Implementation of the Merton Share strategy using CRRA utility.

    This strategy is based on Robert Merton's portfolio optimization problem
    with Constant Relative Risk Aversion (CRRA) utility. The optimal fraction
    to invest is proportional to the expected excess return divided by the
    product of risk aversion and variance.

    The formula is: f* = μ / (γ × σ²)

    Where:
    - f* is the optimal fraction of wealth to invest
    - μ is the expected excess return (return above risk-free rate)
    - γ (gamma) is the coefficient of relative risk aversion
    - σ² (sigma squared) is the variance of returns

    For binary betting, we adapt this by:
    - Expected return is calculated from probability and payoff/loss ratios
    - Variance is estimated from the binary outcome distribution
    - Transaction costs are incorporated into the expected return

    Parameters
    ----------
    payoff : float
        The amount won per unit bet on a successful outcome.
    loss : float
        The amount lost per unit bet on an unsuccessful outcome.
    transaction_cost : float
        The fixed cost per transaction, regardless of outcome.
    risk_aversion : float, default=2.0
        The coefficient of relative risk aversion (γ). Common values:
        - 1.0: Low risk aversion
        - 2.0: Moderate risk aversion (most common empirical estimate)
        - 3.0-5.0: High risk aversion
    min_probability : float, default=0.5
        The minimum probability required to place a bet.
    max_fraction : float, default=1.0
        The maximum fraction of bankroll to bet (safety cap).

    References
    ----------
    .. [1] Merton, R. C. (1969). "Lifetime Portfolio Selection under Uncertainty:
           The Continuous-Time Case". The Review of Economics and Statistics.
    .. [2] https://elmwealth.com/merton-share-derivations/
    """

    def __init__(
        self,
        payoff,
        loss,
        transaction_cost,
        risk_aversion=2.0,
        min_probability=0.5,
        max_fraction=1.0,
    ):
        """
        Initialize the MertonShare strategy.

        Parameters
        ----------
        payoff : float
            The amount won per unit bet on a successful outcome.
        loss : float
            The amount lost per unit bet on an unsuccessful outcome.
        transaction_cost : float
            The fixed cost per transaction, regardless of outcome.
        risk_aversion : float, default=2.0
            The coefficient of relative risk aversion.
        min_probability : float, default=0.5
            The minimum probability required to place a bet.
        max_fraction : float, default=1.0
            The maximum fraction of bankroll to bet.
        """
        if risk_aversion <= 0:
            raise ValueError("Risk aversion must be greater than 0")
        if not 0 <= min_probability <= 1:
            raise ValueError("Minimum probability must be between 0 and 1")
        if not 0 < max_fraction <= 1:
            raise ValueError("Maximum fraction must be between 0 and 1")

        super().__init__(payoff, loss, transaction_cost)
        self.risk_aversion = risk_aversion
        self.min_probability = min_probability
        self.max_fraction = max_fraction

    def evaluate(self, probability, current_bankroll):
        """
        Calculate the Merton Share bet size using CRRA utility.

        Parameters
        ----------
        probability : float
            The probability of a successful outcome, typically between 0 and 1.
        current_bankroll : float
            The current bankroll amount.

        Returns
        -------
        float
            The optimal proportion of the bankroll to bet based on Merton's formula.
        """
        if probability < self.min_probability:
            return 0.0

        # Calculate expected return accounting for transaction costs
        # Expected return = p * (payoff - tc) - (1-p) * (loss + tc)
        expected_return = probability * (
            self.payoff - self.transaction_cost
        ) - (1 - probability) * (self.loss + self.transaction_cost)

        # Don't bet if expected return is non-positive
        if expected_return <= 0:
            return 0.0

        # Calculate variance of returns for a binary outcome
        # For a binary bet: Var(R) = p * (payoff)^2 + (1-p) * (-loss)^2 - E[R]^2
        # We use gross returns (before transaction costs) for variance calculation
        mean_squared_return = (
            probability * (self.payoff**2) + (1 - probability) * (self.loss**2)
        )
        variance = mean_squared_return - (
            probability * self.payoff - (1 - probability) * self.loss
        ) ** 2

        # Avoid division by zero
        if variance <= 0:
            return 0.0

        # Apply Merton's formula: f* = μ / (γ × σ²)
        merton_fraction = expected_return / (self.risk_aversion * variance)

        # Apply safety constraints
        merton_fraction = max(0.0, min(merton_fraction, self.max_fraction))

        # Ensure we never bet more than would result in negative bankroll
        return min(merton_fraction, self.get_max_safe_bet(current_bankroll))

    def calculate_max_entry_price(self, outcomes, probabilities, current_wealth,
                                  tolerance=0.01, max_search_fraction=0.5):
        """
        Calculate maximum price willing to pay for a one-time gamble.

        MertonShare is derived from CRRA utility maximization with the
        risk_aversion parameter (γ) that was set during initialization.

        Parameters
        ----------
        outcomes : array-like
            The possible payoffs from the gamble
        probabilities : array-like
            The probability of each outcome (must sum to ≤ 1)
        current_wealth : float
            Current wealth before the gamble
        tolerance : float, default=0.01
            Convergence tolerance for binary search
        max_search_fraction : float, default=0.5
            Maximum fraction of wealth to consider as upper bound

        Returns
        -------
        float
            Maximum price willing to pay for the gamble

        Notes
        -----
        Uses the risk_aversion (γ) parameter set during initialization.
        For γ=1.0, this is equivalent to Kelly Criterion (log utility).
        Higher γ values indicate more risk aversion and lower willing payments.
        """
        from keeks.utils import find_indifference_price

        return find_indifference_price(
            outcomes=outcomes,
            probabilities=probabilities,
            current_wealth=current_wealth,
            risk_aversion=self.risk_aversion,  # Use the strategy's γ parameter
            tolerance=tolerance,
            max_search_fraction=max_search_fraction
        )
