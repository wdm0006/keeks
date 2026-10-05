Strategy lab notebook
=====================

``examples/keeks_strategy_lab.ipynb`` is the interactive counterpart to
the benchmark pages: a runnable notebook that swaps four bet-sizing rules —
full Kelly, half Kelly, a flat 5% fixed fraction, and dynamic bankroll
management — under identical conditions, on 300 paired paths of 200 bets
at a 55% win probability, and charts the *distribution* of terminal
balance and of maximum drawdown side by side.

It is also a guided tour of the two mistakes that most quietly corrupt
bankroll simulations, each with a section that demonstrates the damage:

- **Two costs that look alike.** The strategy's ``transaction_cost_rate``
  (singular, a fraction of each unit staked, used only for sizing) and the
  simulator's ``fee_per_bet`` (plural, an absolute amount charged once per
  settled bet) are different units. Passing the same number to both does
  not mean what you expect, and the notebook's parameter cell separates
  them by name.
- **State that leaks across paths.** ``BankRoll`` accumulates its history
  and enforces its drawdown limit as it goes, and
  ``DynamicBankrollManagement`` keeps a window of recent results — so
  reusing either object across paths silently lets one path contaminate
  the next. Every path in the lab gets a fresh strategy and a fresh
  bankroll, and a later section shows what happens when you do not.

Everything you might want to change sits in one parameter cell, the run is
seeded, and the first cell installs a pinned keeks version into the
kernel's environment — no local setup beyond Jupyter itself.

.. warning::

   The notebook simulates a made-up repeated bet with a probability you
   set. It is not betting, investment, legal or tax advice, and its output
   is not a forecast of any real result.

Read or run it
--------------

The notebook is linked here rather than rendered in these docs, so the
docs build stays free of notebook-rendering dependencies. Read it as
rendered cells on GitHub, or run it from a checkout in Jupyter:

.. code-block:: bash

   jupyter lab examples/keeks_strategy_lab.ipynb

`examples/keeks_strategy_lab.ipynb on GitHub
<https://github.com/wdm0006/keeks/blob/master/examples/keeks_strategy_lab.ipynb>`_
