"""
Multi-outcome betting strategies and simulators.

Generalizes the binary strategy contract to mutually exclusive markets with
N legs and to portfolios of simultaneous independent bets: one abstract base
class (:class:`BaseMultiOutcomeStrategy`), its vector stake validation, the
aggregate safe-stake cap, the log-growth optimal allocation
(:class:`MultiOutcomeKellyCriterion`), the repeated-play market simulator
(:class:`RepeatedMultiOutcomeSimulator`) that settles categorical outcomes
through the bankroll, and the portfolio simulator
(:class:`PortfolioSimulator`) that settles M independent binary bets per
trial in one net batch.
"""

from keeks.multi_outcome.base import BaseMultiOutcomeStrategy
from keeks.multi_outcome.kelly import MultiOutcomeKellyCriterion
from keeks.multi_outcome.simulators import (
    PortfolioSimulator,
    RepeatedMultiOutcomeSimulator,
)

__all__ = [
    "BaseMultiOutcomeStrategy",
    "MultiOutcomeKellyCriterion",
    "PortfolioSimulator",
    "RepeatedMultiOutcomeSimulator",
]
