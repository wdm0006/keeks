"""
Keeks: A Python package for simulating and evaluating betting strategies.

Keeks provides tools for implementing and testing various betting strategies,
particularly focusing on the Kelly Criterion and its variants. The package
includes:

- Bankroll management: Track and manage funds for betting
- Binary betting strategies: Implementations of Kelly Criterion and other strategies
- Multi-outcome betting: Strategies and simulators for mutually exclusive markets
  and portfolios of independent bets
- Portfolio allocation: Size a portfolio across distribution-valued options —
  mean-variance, minimum-variance, maximum-Sharpe, risk-budgeting, hierarchical,
  mean-CVaR, and online methods over configurable joint-return models
- Simulators: Tools to evaluate strategies under different conditions
- Contract checks: ``check_strategy``, ``check_allocation_strategy``, and
  ``check_model`` verify a contributor's implementation against the
  documented contracts, on the ``check_estimator`` pattern
- Decision-theory utilities: CRRA utility and one-time gamble pricing

The documented public API is re-exported here, so ``from keeks import ...``
works for bankroll, strategy, simulator, allocation, and utility names alike.

The package is designed for educational purposes and to help understand
optimal betting strategies in various scenarios.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version

from keeks.allocation import (
    AllocationResult,
    AllocationSimulator,
    BaseAllocationStrategy,
    BinaryBetsModel,
    ExponentialGradient,
    FixedWeights,
    GlobalMinimumVariance,
    HierarchicalRiskParity,
    JointReturnModel,
    MarginalModel,
    MaximumDiversification,
    MaximumSharpe,
    MeanCVaR,
    MeanVariance,
    ModelInputMixin,
    OnlineNewtonStep,
    RiskAversionScaling,
    RiskBudgeting,
    ScenarioModel,
    bankroll_paths,
    binary_bets_model,
    black_litterman_mean,
    correlation_heatmap,
    dendrogram,
    drawdown_history,
    efficient_frontier,
    estimate_moments,
    fit_marginals_model,
    marginals_model,
    risk_contributions,
    scenario_losses,
    scenario_model,
    scenarios_to_moments,
    shrink_covariance,
    weight_evolution,
)
from keeks.allocation.base import (
    COVARIANCE_SYMMETRY_TOLERANCE,
    EIGENVALUE_FLOOR,
)
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
from keeks.calibration import (
    CalibrationBin,
    CalibrationReport,
    calibration_report,
)
from keeks.checks import check_allocation_strategy, check_model, check_strategy
from keeks.metrics import HistorySummary, summarize_history
from keeks.multi_outcome import (
    BaseMultiOutcomeStrategy,
    HistoricalMultiOutcomeSimulator,
    MultiOutcomeKellyCriterion,
    PortfolioSimulator,
    RepeatedMultiOutcomeSimulator,
)
from keeks.simulators import (
    HistoricalBinarySimulator,
    RandomBinarySimulator,
    RandomUncertainBinarySimulator,
    RepeatedBinarySimulator,
)
from keeks.utils import (
    PROBABILITY_SUM_TOLERANCE,
    RuinError,
    crra_utility,
    expected_utility,
    find_indifference_price,
    validate_probabilities,
)

try:
    __version__ = _package_version("keeks")
except PackageNotFoundError:  # pragma: no cover - uninstalled source checkout
    __version__ = "unknown"

__all__ = [
    "__version__",
    "AllocationResult",
    "AllocationSimulator",
    "BankRoll",
    "BaseAllocationStrategy",
    "BaseMultiOutcomeStrategy",
    "BaseStrategy",
    "BinaryBetsModel",
    "CalibrationBin",
    "CalibrationReport",
    "check_allocation_strategy",
    "check_model",
    "check_strategy",
    "COVARIANCE_SYMMETRY_TOLERANCE",
    "CPPIStrategy",
    "DrawdownAdjustedKelly",
    "DynamicBankrollManagement",
    "EIGENVALUE_FLOOR",
    "ExponentialGradient",
    "FixedFractionStrategy",
    "FixedWeights",
    "FractionalKellyCriterion",
    "GlobalMinimumVariance",
    "HierarchicalRiskParity",
    "HistoricalBinarySimulator",
    "HistorySummary",
    "JointReturnModel",
    "KellyCriterion",
    "MarginalModel",
    "MaximumDiversification",
    "MaximumSharpe",
    "MeanCVaR",
    "MeanVariance",
    "MertonShare",
    "ModelInputMixin",
    "HistoricalMultiOutcomeSimulator",
    "MultiOutcomeKellyCriterion",
    "NaiveStrategy",
    "OnlineNewtonStep",
    "OptimalF",
    "PortfolioSimulator",
    "PROBABILITY_SUM_TOLERANCE",
    "RandomBinarySimulator",
    "RandomUncertainBinarySimulator",
    "RepeatedBinarySimulator",
    "RepeatedMultiOutcomeSimulator",
    "RiskAversionScaling",
    "RiskBudgeting",
    "RuinError",
    "ScenarioModel",
    "bankroll_paths",
    "binary_bets_model",
    "black_litterman_mean",
    "calibration_report",
    "correlation_heatmap",
    "crra_utility",
    "dendrogram",
    "drawdown_history",
    "efficient_frontier",
    "estimate_moments",
    "expected_utility",
    "find_indifference_price",
    "fit_marginals_model",
    "marginals_model",
    "validate_probabilities",
    "risk_contributions",
    "scenario_losses",
    "scenario_model",
    "scenarios_to_moments",
    "shrink_covariance",
    "summarize_history",
    "weight_evolution",
]
