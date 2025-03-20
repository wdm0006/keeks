"""
Binary betting strategies for the keeks package.

This module provides various strategies for binary betting scenarios, including:
- Kelly Criterion: Optimal bet sizing based on probability and odds
- Fractional Kelly: A more conservative version of Kelly using a fraction
- Drawdown-Adjusted Kelly: Kelly variant that accounts for drawdown tolerance
- Optimal f: Ralph Vince's method for maximizing geometric growth rate
- Fixed Fraction: Simple strategy that bets a constant percentage
- CPPI: Constant Proportion Portfolio Insurance for preserving capital
- Dynamic Bankroll Management: Adaptive allocation based on recent performance
- Naive Strategy: Simple strategy that bets full amount when expected value is positive

All strategies implement the BaseStrategy interface.
"""

from keeks.binary_strategies.kelly import (
    DrawdownAdjustedKelly,
    FractionalKellyCriterion,
    KellyCriterion,
)
from keeks.binary_strategies.simple import (
    CPPIStrategy,
    DynamicBankrollManagement,
    FixedFractionStrategy,
    NaiveStrategy,
    OptimalF,
)

__author__ = "willmcginnis"

__all__ = [
    "KellyCriterion",
    "FractionalKellyCriterion",
    "DrawdownAdjustedKelly",
    "OptimalF",
    "NaiveStrategy",
    "FixedFractionStrategy",
    "CPPIStrategy",
    "DynamicBankrollManagement",
]
