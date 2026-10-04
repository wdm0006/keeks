Allocation Visualization Helpers
================================

The allocation layer's visualization helpers live in
:mod:`keeks.allocation.plots`: matplotlib-only figures for the objects the
API already returns — bankroll histories, allocation results, weight
matrices, covariance descriptors, the hand-rolled HRP linkage, and scenario
matrices — so examples and benchmarks can render the layer without gluing
matplotlib calls by hand.

Every helper draws one figure and returns the
:class:`matplotlib.axes.Axes` it drew on, so callers can style further,
save, or embed the result. The helpers are deterministic by construction:
series and legends sort by name, colors come from one small fixed palette
(:data:`keeks.allocation.plots.PALETTE`), and nothing reads or writes
matplotlib's global state — figures are bare
:class:`matplotlib.figure.Figure` objects, never pyplot-managed, which
makes the helpers safe to call inside simulation loops and headless (Agg)
runs.

Bankroll growth and drawdown
----------------------------

.. autofunction:: keeks.allocation.plots.bankroll_paths

.. autofunction:: keeks.allocation.plots.drawdown_history

Weights and risk
----------------

.. autofunction:: keeks.allocation.plots.weight_evolution

.. autofunction:: keeks.allocation.plots.risk_contributions

.. autofunction:: keeks.allocation.plots.efficient_frontier

Structure and tails
-------------------

.. autofunction:: keeks.allocation.plots.correlation_heatmap

.. autofunction:: keeks.allocation.plots.dendrogram

.. autofunction:: keeks.allocation.plots.scenario_losses
