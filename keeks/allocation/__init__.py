"""
Portfolio-style allocation strategies for distribution-valued options.

The allocation layer sizes portfolios across N distribution-valued options
rather than fixed payoffs: one abstract base class
(:class:`BaseAllocationStrategy`) whose ``evaluate(current_bankroll)`` returns
one long-only weight per option - each in ``[0, 1]``, summing to no more than
one, with the residual held as cash at zero return - the shared weight, mean,
covariance, and scenario validators, and the :class:`AllocationResult`
diagnostics object.

Descriptive inputs bind at construction, and an allocator reprices by fresh
construction, matching the multi-outcome convention. Weights are scale-free:
they ignore the bankroll level, and every allocator returns all-zero weights
when the bankroll is nonpositive. Online allocators are the same ABC plus one
stateful hook - ``record_settlement(won, realized_returns)``, called once per
staked period with the per-option outcome flags and the realized joint
simple-return vector.

The joint-return input models (:mod:`keeks.allocation.models`) are the
allocation layer's universal input currency: one sampling contract -
``JointReturnModel.sample(n, rng)`` returning an ``(n, N)`` matrix of joint
simple returns - with adapters for empirical scenarios, keeks-native binary
bets, parametric marginals, and user callables. Moment-based allocators
consume exact model moments when available and Monte Carlo estimates
otherwise.

The estimators also live here (:mod:`keeks.allocation.estimators`):
:func:`shrink_covariance` and :func:`black_litterman_mean` preprocess Σ
and μ for any optimizer, numpy-only.

The allocation simulator (:mod:`keeks.allocation.simulators`) replays any
allocator over realizations from any joint-return model through the bankroll
machinery: :class:`AllocationSimulator` settles batch-net per period with
keeks' seeding and refuse-then-stop conventions, fires the
``record_settlement(won, realized_returns)`` hook with the per-option
outcome flags and the realized joint simple-return vector for
the online family, and treats residual probability mass as an all-cash
period.

The online family lives in :mod:`keeks.allocation.online` -
:class:`FixedWeights`, :class:`ExponentialGradient`, and
:class:`OnlineNewtonStep` adapt their weights through the
``record_settlement`` hook alone, numpy-only - the static moment-based
family in :mod:`keeks.allocation.moments`: :class:`MeanVariance`,
:class:`GlobalMinimumVariance`, :class:`MaximumSharpe`,
:class:`MaximumDiversification`, :class:`RiskBudgeting`, and the
:class:`RiskAversionScaling` wrapper - and the scenario family in
:mod:`keeks.allocation.scenarios`: :class:`MeanCVaR` sizes against a
scenario matrix through the Rockafellar-Uryasev linear program
(scipy-gated), and :func:`scenarios_to_moments` bridges scenarios to the
moment-based methods' ``(mean, covariance)`` descriptor.

The visualization helpers live in :mod:`keeks.allocation.plots`:
deterministic, matplotlib-only figures for the layer's outputs - bankroll
growth paths, drawdown curves, weight evolution, risk contributions, the
efficient frontier, correlation heatmaps, the HRP dendrogram, and
scenario-loss histograms with VaR/CVaR markers. Each helper takes objects
this package already returns and returns the Axes it drew on: series and
legends sort by name, colors come from one small fixed palette, and no
helper touches matplotlib's global state.
"""

from keeks.allocation.base import AllocationResult, BaseAllocationStrategy
from keeks.allocation.estimators import black_litterman_mean, shrink_covariance
from keeks.allocation.hierarchical import HierarchicalRiskParity
from keeks.allocation.models import (
    BinaryBetsModel,
    JointReturnModel,
    MarginalModel,
    ModelInputMixin,
    ScenarioModel,
    binary_bets_model,
    estimate_moments,
    fit_marginals_model,
    marginals_model,
    scenario_model,
)
from keeks.allocation.moments import (
    GlobalMinimumVariance,
    MaximumDiversification,
    MaximumSharpe,
    MeanVariance,
    RiskAversionScaling,
    RiskBudgeting,
)
from keeks.allocation.online import (
    ExponentialGradient,
    FixedWeights,
    OnlineNewtonStep,
)
from keeks.allocation.plots import (
    bankroll_paths,
    correlation_heatmap,
    dendrogram,
    drawdown_history,
    efficient_frontier,
    risk_contributions,
    scenario_losses,
    weight_evolution,
)
from keeks.allocation.scenarios import MeanCVaR, scenarios_to_moments
from keeks.allocation.simulators import AllocationSimulator

__all__ = [
    "AllocationResult",
    "AllocationSimulator",
    "bankroll_paths",
    "BaseAllocationStrategy",
    "BinaryBetsModel",
    "black_litterman_mean",
    "correlation_heatmap",
    "dendrogram",
    "drawdown_history",
    "efficient_frontier",
    "ExponentialGradient",
    "FixedWeights",
    "GlobalMinimumVariance",
    "HierarchicalRiskParity",
    "JointReturnModel",
    "MarginalModel",
    "MaximumDiversification",
    "MaximumSharpe",
    "MeanCVaR",
    "MeanVariance",
    "ModelInputMixin",
    "OnlineNewtonStep",
    "RiskAversionScaling",
    "RiskBudgeting",
    "risk_contributions",
    "scenario_losses",
    "ScenarioModel",
    "binary_bets_model",
    "estimate_moments",
    "fit_marginals_model",
    "marginals_model",
    "scenario_model",
    "scenarios_to_moments",
    "shrink_covariance",
    "weight_evolution",
]
