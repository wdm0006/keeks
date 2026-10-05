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
