"""
Multi-outcome betting strategies and simulators.

Generalizes the binary strategy contract to mutually exclusive markets with
N legs: one abstract base class
(:class:`BaseMultiOutcomeStrategy`), its vector stake validation, the
aggregate safe-stake cap, and the repeated-play simulator
(:class:`RepeatedMultiOutcomeSimulator`) that settles categorical outcomes
through the bankroll.
"""

from keeks.multi_outcome.base import BaseMultiOutcomeStrategy
from keeks.multi_outcome.simulators import RepeatedMultiOutcomeSimulator

__all__ = ["BaseMultiOutcomeStrategy", "RepeatedMultiOutcomeSimulator"]
