"""
Binary betting strategies for the keeks package.

This module provides various strategies for binary betting scenarios, including:
- Kelly Criterion: Optimal bet sizing based on probability and odds
- Naive Strategy: Simple strategy that bets full amount when expected value is positive

All strategies implement the BaseStrategy interface.
"""

from keeks.binary_strategies.kelly import KellyCriterion
from keeks.binary_strategies.simple import NaiveStrategy

__author__ = "willmcginnis"

__all__ = ["KellyCriterion", "NaiveStrategy"]
