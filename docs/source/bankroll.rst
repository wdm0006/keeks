Bankroll
========

The bankroll object simulates the financial side of simulations. It takes into account things like max drawdown limits
and in the future may include risk free rate investments, interest and concepts like that.

Exceptions
----------

.. autoexception:: keeks.utils.RuinError
    :members:
    :undoc-members:
    :show-inheritance:
    :noindex:

Summarizing a run
-----------------

``evaluate_strategy`` mutates the bankroll and returns nothing, so
:func:`keeks.metrics.summarize_history` reduces ``bankroll.history`` to the
numbers that distinguish strategies: bets recorded, total return, geometric
growth per bet, maximum drawdown against the running peak, and whether the run
ended ruined. Values that cannot be defined (growth with no recorded bets or
nonpositive end funds, return from a zero start) are ``None`` rather than
``0.0``; an empty, non-finite or negative history raises ``ValueError``.

.. code-block:: python

    from keeks import summarize_history

    summary = summarize_history(bankroll.history)
    print(summary.max_drawdown, summary.geometric_growth_per_period)

.. autofunction:: keeks.metrics.summarize_history

.. autoclass:: keeks.metrics.HistorySummary
    :no-members:

Comparing strategies on a recorded log
--------------------------------------

:func:`keeks.comparison.compare_strategies` replays one recorded log through
several strategies, each from a fresh strategy and a fresh bankroll, and
returns one :class:`~keeks.comparison.StrategyComparison` per strategy (name
plus its ``HistorySummary``), ranked by ``sort_by`` with undefined (``None``)
values last.

.. code-block:: python

    from keeks import FixedFractionStrategy, KellyCriterion, compare_strategies

    odds = {"payoff": 1.0, "loss": 1.0, "transaction_cost_rate": 0.0}
    rows = compare_strategies(
        {
            "Kelly": lambda: KellyCriterion(**odds),
            "Fixed 5%": lambda: FixedFractionStrategy(**odds, fraction=0.05),
        },
        probabilities,
        outcomes,
        payoff=1.0,
        loss=1.0,
        fee_per_bet=0.0,
        initial_funds=1000.0,
    )
    for row in rows:
        print(row.name, row.summary.max_drawdown)

.. autofunction:: keeks.comparison.compare_strategies

For mutually exclusive markets (such as 1X2), see
:func:`keeks.comparison.compare_multi_outcome_strategies` in the
:doc:`multi-outcome guide <multi_outcome>`.

.. autoclass:: keeks.comparison.StrategyComparison
    :no-members:

How sure is the ranking?
~~~~~~~~~~~~~~~~~~~~~~~~

A single replay is one draw. :func:`keeks.comparison.bootstrap_comparison`
resamples the log's bets (``block_size`` > 1 resamples moving blocks), replays
every strategy on the same resample so differences are paired, and returns a
:class:`~keeks.comparison.BootstrapRow` per strategy: the full-log ``point``
value, percentile ``lower``/``upper`` bounds, and ``rank_counts`` (how often it
ranked first, second, ...). ``seed`` is required and uses a private generator.
Resamples where the metric is undefined are excluded from the bounds and
counted in ``undefined``.

.. code-block:: pycon

    >>> from keeks import FixedFractionStrategy, bootstrap_comparison
    >>> odds = {"payoff": 1.0, "loss": 1.0, "transaction_cost_rate": 0.0}
    >>> rows = bootstrap_comparison(
    ...     {
    ...         "Fixed 1%": lambda: FixedFractionStrategy(**odds, fraction=0.01),
    ...         "Fixed 10%": lambda: FixedFractionStrategy(**odds, fraction=0.1),
    ...     },
    ...     [0.7] * 8,
    ...     [1] * 8,
    ...     payoff=1.0, loss=1.0, fee_per_bet=0.0, initial_funds=100.0,
    ...     n_resamples=20, seed=1,
    ... )
    >>> [(row.name, round(row.point, 3), row.rank_counts) for row in rows]
    [('Fixed 1%', 0.01, (0, 20)), ('Fixed 10%', 0.1, (20, 0))]

.. note::

   The intervals reflect sampling of the recorded log only. They say nothing
   about model error, a changed edge, or a different future. Runtime is
   ``n_resamples`` times one :func:`~keeks.comparison.compare_strategies` call.

.. autofunction:: keeks.comparison.bootstrap_comparison

.. autoclass:: keeks.comparison.BootstrapRow
    :no-members:
