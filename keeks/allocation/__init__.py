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
stateful hook - ``record_settlement(realized_returns)``, called once per
staked period with the realized joint simple-return vector.

The joint-return input models (:mod:`keeks.allocation.models`) are the
allocation layer's universal input currency: one sampling contract -
``JointReturnModel.sample(n, rng)`` returning an ``(n, N)`` matrix of joint
simple returns - with adapters for empirical scenarios, keeks-native binary
bets, parametric marginals, and user callables. Moment-based allocators
consume exact model moments when available and Monte Carlo estimates
otherwise.

Subsequent modules in this subpackage add the remaining method families
(mean-variance, minimum variance, maximum Sharpe, risk budgeting,
mean-CVaR), and the allocation simulator; solvers that need scipy arrive
behind the ``keeks[allocation]`` optional extra. The online family already
lives here (:mod:`keeks.allocation.online`): :class:`FixedWeights`,
:class:`ExponentialGradient`, and :class:`OnlineNewtonStep` adapt their
weights through the ``record_settlement`` hook alone, numpy-only.
"""

from keeks.allocation.base import AllocationResult, BaseAllocationStrategy
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
from keeks.allocation.online import (
    ExponentialGradient,
    FixedWeights,
    OnlineNewtonStep,
)

__all__ = [
    "AllocationResult",
    "BaseAllocationStrategy",
    "BinaryBetsModel",
    "ExponentialGradient",
    "FixedWeights",
    "HierarchicalRiskParity",
    "JointReturnModel",
    "MarginalModel",
    "ModelInputMixin",
    "OnlineNewtonStep",
    "ScenarioModel",
    "binary_bets_model",
    "estimate_moments",
    "fit_marginals_model",
    "marginals_model",
    "scenario_model",
]
