class RuinError(Exception):
    """
    Exception raised when a bankroll experiences a drawdown exceeding the maximum allowed limit.

    This exception is typically raised by the BankRoll class when a withdrawal would cause
    the bankroll to drop below the configured maximum drawdown threshold.
    """

    pass
