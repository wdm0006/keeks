"""
Multi-outcome betting strategies.

Generalizes the binary strategy contract to mutually exclusive markets with
N legs: one abstract base class
(:class:`BaseMultiOutcomeStrategy`), its vector stake validation, and the
aggregate safe-stake cap.
"""

from keeks.multi_outcome.base import BaseMultiOutcomeStrategy

__all__ = ["BaseMultiOutcomeStrategy"]
