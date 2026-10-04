Keeks vs betting-math-kit: Which Python Betting Library Fits Your Job?
=======================================================================

.. meta::
   :description: Compare Keeks and betting-math-kit on intended job, outcome scope, sizing strategies, de-vigging, calibration, simulation, safeguards, dependencies, and maturity, with guidance on when to choose each or use both.

Keeks and betting-math-kit both size bets with the Kelly criterion, and they
are built for different jobs. This page states the boundary so you can pick
the right one, or use both.

.. note::

   Statements about betting-math-kit were read from its public README,
   ``pyproject.toml`` and ``simulation.py`` on **2026-10-04** at commit
   ``ebb3d00`` (version 0.3.0); see :ref:`bmk-sources`. That project can change
   after this date, so check its repository before relying on a row. Where a
   feature is not described on the pages read, this page says "not described on
   the pages read" rather than saying it is missing. Statements about Keeks
   describe version 0.8.0.

The short version
-----------------

* Choose **betting-math-kit** when you start from sportsbook odds and want one
  pipeline from odds conversion and de-vigging through edge, stake size,
  calibration metrics, and a bankroll simulation.
* Choose **Keeks** when you already have a probability, a payoff, and a loss,
  and the question is how different sizing policies behave under the same
  controlled, repeatable bankroll simulation.
* Use **both** when you want fair probabilities from one and a strategy
  comparison from the other (see :ref:`bmk-pipeline`).

Neither library is a prediction model, and neither is investment or betting
advice. Keeks is educational (see the disclaimer on the :doc:`index`).

Side-by-side
------------

.. list-table::
   :header-rows: 1
   :widths: 16 42 42

   * - Dimension
     - Keeks
     - betting-math-kit
   * - Intended job
     - Compare repeated-bet sizing strategies and simulate how a bankroll
       behaves under each. Starts from a probability, a payoff, a loss, and a
       cost input (:doc:`getting_started`).
     - Sports-betting math as one pipeline: "odds conversion, de-vigging, Kelly
       criterion, calibration metrics, Monte Carlo simulation" (``pyproject.toml``
       description).
   * - Binary vs multi-outcome
     - Binary repeated bets, plus ``MultiOutcomeKellyCriterion`` for one
       mutually exclusive market and ``PortfolioSimulator`` for independent
       bets (:doc:`multi_outcome`).
     - De-vigging supports two-outcome and n-outcome markets (multiplicative,
       power, and Shin for n outcomes). Pari-mutuel Kelly sizes runners in a
       race with an exposure cap (README, "Modules" and "Technical notes").
   * - Sizing-strategy breadth
     - Nine binary strategies behind one ``evaluate(probability,
       current_bankroll)`` call: Kelly, fractional Kelly, drawdown-adjusted
       Kelly, OptimalF, fixed fraction, CPPI, dynamic bankroll management,
       Merton share, and a naive strategy (:doc:`binary_strategies`).
     - Fixed-odds Kelly with a fractional multiplier and a minimum-edge gate,
       and pari-mutuel Kelly with takeout and pool-size limits. Its simulation
       module simulates "fixed fractional Kelly betting"
       (``simulation.py`` docstring).
   * - Odds, de-vig, calibration
     - Not provided. Inputs are a probability, a payoff multiplier, and a
       loss multiplier; there is no odds-conversion, de-vig, or calibration
       API in the package or its docs as of 0.8.0.
     - Odds conversion (American, decimal, implied probability), four de-vig
       methods, edge against the fair line, and Brier score, log loss,
       expected calibration error, calibration buckets, and closing line value
       (README, "Modules").
   * - Simulation model
     - Strategy-agnostic. A simulator takes fixed payoff, loss, cost, and
       probability inputs, applies any strategy's fraction to a fresh
       ``BankRoll`` once per trial, and accepts an optional ``seed``
       (:doc:`simulators`). Multi-outcome and portfolio simulators are
       seeded as well (:doc:`multi_outcome`).
     - Monte Carlo over many trials of a single fixed-odds, fixed-edge bet
       sequence at one Kelly fraction, with Python's ``random.Random`` and an
       optional seed. Returns ruin rate, median, mean, 5th and 95th percentile
       final bankroll, median maximum drawdown, and growth rate
       (``simulation.py``).
   * - Bankroll and drawdown safeguards
     - ``BankRoll`` can hold back part of the funds and refuses any single
       withdrawal larger than ``max_draw_down`` times current funds, raising
       ``RuinError`` that the simulators catch. This is a per-settlement cap,
       not a cumulative drawdown budget, and it does not protect against loss
       (:doc:`bankroll`).
     - A configurable ``ruin_threshold`` and a median maximum drawdown statistic
       in the simulation result; a minimum-edge gate, and pool and race-exposure
       caps for pari-mutuel bets (README, ``simulation.py``). A per-settlement
       cap like Keeks's is not described on the pages read.
   * - Dependencies
     - NumPy and Matplotlib at runtime; Python 3.10 to 3.14 (``pyproject.toml``
       and the package classifiers).
     - "Zero dependencies" (README); Python 3.10 or newer (``pyproject.toml``).
   * - Entry-price utility
     - Separate one-time-gamble tools: CRRA utility, ``expected_utility``,
       ``find_indifference_price``, and ``calculate_max_entry_price`` on the
       strategies. These answer a different question from repeated-bet sizing
       (:doc:`utils`).
     - Not described on the pages read.
   * - Maturity
     - Version 0.8.0, MIT license, development status Alpha.
     - Version 0.3.0, MIT license, "Development Status :: 4 - Beta"
       (``pyproject.toml``). The README describes it as a "snapshot" pulled
       from a larger private system, reports 195 tests, and runs CI on Python
       3.10 to 3.13.

Choose betting-math-kit when
----------------------------

* Your inputs are sportsbook prices. It converts American and decimal odds,
  removes the margin, and measures edge against the fair line, which Keeks does
  not do.
* You want to evaluate whether your probabilities are any good over time:
  calibration and closing-line-value metrics are part of its pipeline.
* You bet pari-mutuel pools and need takeout and pool-size limits in the stake.
* You want no runtime dependencies at all.

Choose Keeks when
-----------------

* The decision is between sizing policies. Nine strategies share one calling
  convention, so a simulator can run each on the same inputs and bankroll
  rules without rewriting anything.
* You want a seeded, repeatable run you can inspect, with a bankroll object
  that records its history and stops a run when a settlement breaches its
  cap.
* You need a market with several mutually exclusive legs, or several
  independent bets netted into one bankroll.
* You want to price a one-time entry with CRRA utility separately from
  repeated-bet sizing.

What neither one does
---------------------

Neither library predicts outcomes. Keeks models transaction cost as a single
normalized per-bet input and does not model spreads, slippage, market impact,
venue commissions, or correlated positions. Simulated results show how a model
behaves under assumptions you supply; they are not evidence of how a strategy
will perform on real bets.

.. _bmk-pipeline:

Use both (conceptual)
---------------------

.. warning::

   This pipeline is **conceptual**. It is not part of either project's test
   suite, and neither project documents or supports the combination. The
   snippet was run once without error on 2026-10-04 against Keeks 0.8.0 and
   betting-math-kit commit ``ebb3d00``, but that is not a tested integration.
   It publishes no result, and you are responsible for checking every
   conversion.

The idea: use betting-math-kit to turn a quoted line and your model's
probability into an edge measured against the fair price, then give the
resulting probability and payoff to Keeks to compare sizing policies.

.. code-block:: python

   from betting_math_kit import calculate_edge_calibrated
   from keeks.bankroll import BankRoll
   from keeks.binary_strategies import FractionalKellyCriterion, KellyCriterion
   from keeks.simulators.repeated_binary import RepeatedBinarySimulator

   model_prob = 0.55
   edge = calculate_edge_calibrated(
       model_prob=model_prob, home_odds=-110, away_odds=-110
   )
   print(edge.raw_edge, edge.true_edge)  # vigged vs fair-line edge

   # -110 pays 100/110 per unit staked; Keeks takes the payoff as a multiplier.
   payoff = 100 / 110

   strategies = {
       "kelly": KellyCriterion(payoff=payoff, loss=1.0, transaction_cost=0.0),
       "half kelly": FractionalKellyCriterion(
           payoff=payoff, loss=1.0, fraction=0.5, transaction_cost=0.0
       ),
   }
   for name, strategy in strategies.items():
       bankroll = BankRoll(initial_funds=1000.0, max_draw_down=0.3)
       simulator = RepeatedBinarySimulator(
           payoff=payoff, loss=1.0, transaction_costs=0.0,
           probability=model_prob, trials=200, seed=7,
       )
       simulator.evaluate_strategy(strategy, bankroll)
       print(name, strategy.evaluate(model_prob, 1000.0), bankroll.total_funds)

Two cautions that apply to any such pairing. First, Keeks does not check that a
strategy's payoff and loss agree with the simulator's, so pass the same values
to both. Second, Kelly-family strategies in Keeks return ``0.0`` below a
``min_probability`` of 0.5 by default, regardless of payoff.

.. _bmk-sources:

Sources for betting-math-kit statements
---------------------------------------

All read on 2026-10-04 at commit ``ebb3d00d2f8d0e448e5be821e581938579528369``
(committed 2026-07-13):

* README: https://github.com/bene-art/betting-math-kit/blob/ebb3d00d2f8d0e448e5be821e581938579528369/README.md
* ``pyproject.toml``: https://github.com/bene-art/betting-math-kit/blob/ebb3d00d2f8d0e448e5be821e581938579528369/pyproject.toml
* ``simulation.py``: https://github.com/bene-art/betting-math-kit/blob/ebb3d00d2f8d0e448e5be821e581938579528369/src/betting_math_kit/simulation.py

Keeks is a separate project with no affiliation to betting-math-kit or its
author.
