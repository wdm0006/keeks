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

.. autoclass:: keeks.comparison.StrategyComparison
    :no-members:
