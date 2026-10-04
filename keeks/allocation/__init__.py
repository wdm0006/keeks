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

Subsequent modules in this subpackage add the joint-return input models, the
method families (mean-variance, minimum variance, maximum Sharpe, risk
budgeting, hierarchical risk parity, mean-CVaR, online methods), and the
allocation simulator; solvers that need scipy arrive behind the
``keeks[allocation]`` optional extra.
"""

from keeks.allocation.base import AllocationResult, BaseAllocationStrategy

__all__ = [
    "AllocationResult",
    "BaseAllocationStrategy",
]
