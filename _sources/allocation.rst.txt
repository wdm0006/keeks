Portfolio Allocation
====================

The allocation layer — :mod:`keeks.allocation` — sizes portfolios across N
**distribution-valued options**: options described by a mean vector and
covariance, a scenario matrix of joint simple returns, keeks-native binary
bets, fitted parametric marginals, or any user-supplied sampler. Where the
binary strategies return one stake fraction and the multi-outcome strategies
return stake fractions over mutually exclusive legs, the allocation
strategies return **weights**: one long-only weight per option, each in
``[0, 1]``, summing to at most one. The residual is cash held at zero
return, so a :class:`keeks.bankroll.BankRoll` is reused untouched and the
aggregate-exposure guarantee transfers.

The input model is deliberately diverse and configurable. Allocators accept
anything that can produce joint simple-return draws through one sampling
contract — :class:`keeks.allocation.models.JointReturnModel` — so binary
bets, empirical histories, thin and incredibly fat-tailed marginals, and
custom samplers all feed the same allocators. Models that know their
closed-form moments expose them exactly via ``moments()``; the rest are
estimated from draws. The Kelly identity is the layer's accent:
mean-variance at risk aversion λ = 1 is the second-order Kelly allocation,
and :class:`keeks.allocation.moments.RiskAversionScaling` is the
fractional-Kelly story — shrinkage toward cash.

Descriptive inputs bind at construction and are immutable thereafter — an
allocator reprices by fresh construction, matching the multi-outcome
doctrine. scipy is an optional extra, ``keeks[allocation]``: the QP/LP
solvers (``MeanVariance``, ``GlobalMinimumVariance``, ``MaximumSharpe``,
``MaximumDiversification``) and ``MeanCVaR`` point at the extra when it is
absent; everything else is numpy-only. For visualization see
:doc:`allocation_plots` — matplotlib helpers for every object this API
returns — and ``examples/allocation_etfs.py`` for the worked real-data
example, which runs a six-ETF book through both input-model helpers and the
simulator offline from a committed fixture.

The Strategy Contract
---------------------

.. autoclass:: keeks.allocation.base.BaseAllocationStrategy
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: keeks.allocation.base.AllocationResult
    :members:
    :exclude-members: weights, objective, converged, iterations, expected_growth, volatility, all_cash_reason

.. autoclass:: keeks.allocation.models.ModelInputMixin
    :members:
    :undoc-members:

Joint-Return Input Models
-------------------------

The universal input currency is the model object: hyperparameters live in
adapter constructors, and consumers see one sampling contract.

.. autoclass:: keeks.allocation.models.JointReturnModel
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: keeks.allocation.models.ScenarioModel
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: keeks.allocation.models.scenario_model

.. autoclass:: keeks.allocation.models.BinaryBetsModel
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: keeks.allocation.models.binary_bets_model

.. autoclass:: keeks.allocation.models.MarginalModel
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: keeks.allocation.models.marginals_model

.. autofunction:: keeks.allocation.models.fit_marginals_model

.. autofunction:: keeks.allocation.models.estimate_moments

.. autofunction:: keeks.allocation.scenarios.scenarios_to_moments

Moment-Based Methods
--------------------

Static allocators over a mean vector and covariance matrix. Mean-variance
at λ = 1 is the second-order Kelly allocation; the covariance-only methods
(GMV, risk budgeting, maximum diversification) need no expected returns at
all.

.. autoclass:: keeks.allocation.moments.MeanVariance
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: keeks.allocation.moments.GlobalMinimumVariance
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: keeks.allocation.moments.MaximumSharpe
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: keeks.allocation.moments.MaximumDiversification
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: keeks.allocation.moments.RiskBudgeting
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: keeks.allocation.moments.RiskAversionScaling
    :members:
    :undoc-members:
    :show-inheritance:

Scenario-Based Methods
----------------------

Scenario methods consume joint-return draws directly — sample-CVaR is
defined for any finite sample, so fat-tailed marginals and distributions
without a finite variance size here, where the moment-based methods'
second-moment assumption would already be meaningless. Unit risk aversion
is the v1 posture: the objective trades expected return against the tail's
expected loss one-for-one, so on daily-frequency market data — expected
return far below the tail expectation — the optimum is typically full cash.

.. autoclass:: keeks.allocation.scenarios.MeanCVaR
    :members:
    :undoc-members:
    :show-inheritance:

Hierarchical Risk Parity
------------------------

HRP builds the cluster tree from correlation distance, quasi-diagonalizes,
and bisects recursively with inverse-variance splits — it never inverts the
covariance matrix, so it sizes ill-conditioned books.

.. autoclass:: keeks.allocation.hierarchical.HierarchicalRiskParity
    :members:
    :undoc-members:
    :show-inheritance:

Estimators
----------

Preprocessors that compose with any optimizer above: shrinkage fixes Σ,
Black-Litterman fixes μ.

.. autofunction:: keeks.allocation.estimators.shrink_covariance

.. autofunction:: keeks.allocation.estimators.black_litterman_mean

Online Allocation
-----------------

Online allocators add one stateful hook —
``record_settlement(realized_returns)``, called once per staked period with
the realized joint simple-return vector — and adapt their weights from
realized returns. ``FixedWeights`` is the constant rebalanced portfolio,
the regret benchmark the adaptive methods compete against.

.. autoclass:: keeks.allocation.online.FixedWeights
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: keeks.allocation.online.ExponentialGradient
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: keeks.allocation.online.OnlineNewtonStep
    :members:
    :undoc-members:
    :show-inheritance:

Simulator
---------

The simulator replays an allocator over scenario realizations — batch-net
settlement through the bankroll machinery, refuse-then-stop ruin policy,
flat per-period costs, and the house single-seed-stream reproducibility
contract. It is what makes the static/online distinction observable: the
settlement hook feeds online allocators' ``record_settlement``.

.. autoclass:: keeks.allocation.simulators.AllocationSimulator
    :members:
    :undoc-members:
    :show-inheritance:
