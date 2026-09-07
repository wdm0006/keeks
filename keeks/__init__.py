"""
Keeks: A Python package for simulating and evaluating betting strategies.

Keeks provides tools for implementing and testing various betting strategies,
particularly focusing on the Kelly Criterion and its variants. The package
includes:

- Bankroll management: Track and manage funds for betting
- Binary betting strategies: Implementations of Kelly Criterion and other strategies
- Simulators: Tools to evaluate strategies under different conditions
- Decision-theory utilities: CRRA utility and one-time gamble pricing

The documented public API is re-exported here, so ``from keeks import ...``
works for bankroll, strategy, simulator, and utility names alike.

The package is designed for educational purposes and to help understand
optimal betting strategies in various scenarios.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version

from keeks.bankroll import BankRoll
from keeks.binary_strategies import (
    CPPIStrategy,
    DrawdownAdjustedKelly,
    DynamicBankrollManagement,
    FixedFractionStrategy,
    FractionalKellyCriterion,
    KellyCriterion,
    MertonShare,
    NaiveStrategy,
    OptimalF,
)
from keeks.binary_strategies.base import BaseStrategy
from keeks.simulators import (
    RandomBinarySimulator,
    RandomUncertainBinarySimulator,
    RepeatedBinarySimulator,
)
from keeks.utils import (
    RuinError,
    crra_utility,
    expected_utility,
    find_indifference_price,
    normalize_probabilities,
)

try:
    __version__ = _package_version("keeks")
except PackageNotFoundError:  # pragma: no cover - uninstalled source checkout
    __version__ = "unknown"

__all__ = [
    "__version__",
    "BankRoll",
    "BaseStrategy",
    "CPPIStrategy",
    "DrawdownAdjustedKelly",
    "DynamicBankrollManagement",
    "FixedFractionStrategy",
    "FractionalKellyCriterion",
    "KellyCriterion",
    "MertonShare",
    "NaiveStrategy",
    "OptimalF",
    "RandomBinarySimulator",
    "RandomUncertainBinarySimulator",
    "RepeatedBinarySimulator",
    "RuinError",
    "crra_utility",
    "expected_utility",
    "find_indifference_price",
    "normalize_probabilities",
]
