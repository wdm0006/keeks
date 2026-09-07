"""Contract tests for BaseStrategy's reserved entry-price hook.

All nine shipped strategies override ``calculate_max_entry_price``; the base
implementation is a deliberate contract for subclasses without utility
functions, so it is tested with a minimal subclass rather than removed.
"""

import pytest

from keeks.binary_strategies.base import BaseStrategy


class MinimalStrategy(BaseStrategy):
    """Implements only the abstract evaluate method."""

    def evaluate(self, _probability: float, _current_bankroll: float) -> float:
        return 0.0


def test_base_entry_price_contract_raises_not_implemented():
    """The base hook exists but refuses to price one-time gambles."""
    strategy = MinimalStrategy(payoff=2.0, loss=1.0)

    with pytest.raises(
        NotImplementedError,
        match=r"^MinimalStrategy does not support one-time entry price",
    ):
        strategy.calculate_max_entry_price([100], [0.5], 1000)
