Kelly Criterion in Python: Calculate a Bankroll Fraction with Keeks
=====================================================================

.. meta::
   :description: Calculate a Kelly bankroll fraction in Python from win probability, payoff, loss, and a normalized transaction cost using a runnable Keeks example.

.. raw:: html

   <script type="application/ld+json">
   {
     "@context": "https://schema.org",
     "@type": "TechArticle",
     "headline": "Kelly Criterion in Python: Calculate a Bankroll Fraction with Keeks",
     "description": "Calculate a Kelly bankroll fraction in Python from win probability, payoff, loss, and a normalized transaction cost using a runnable Keeks example.",
     "dateModified": "2026-08-03",
     "author": {
       "@type": "Organization",
       "name": "Keeks Contributors"
     },
     "hasPart": {
       "@type": "SoftwareSourceCode",
       "programmingLanguage": "Python",
       "runtimePlatform": "Python 3",
       "codeRepository": "https://github.com/wdm0006/keeks"
     }
   }
   </script>

Quick answer
------------

Given a win probability, a payoff multiplier, a loss multiplier, and a
normalized transaction cost, Keeks' ``KellyCriterion.evaluate()`` returns the
fraction of the current bankroll to stake on a repeated binary bet:

.. code-block:: python

   from keeks.binary_strategies import KellyCriterion

   bankroll = 1_000.0
   strategy = KellyCriterion(
       payoff=1.0,
       loss=1.0,
       transaction_cost=0.01,
   )

   fraction = strategy.evaluate(probability=0.55, current_bankroll=bankroll)
   print(f"Bankroll fraction: {fraction:.4%}")

.. code-block:: text

   Bankroll fraction: 9.0009%

This is a repeated-bet answer, not a one-time price. See
`Different problem: price a one-time gamble`_ below if that is what you need.

Install Keeks
-------------

.. code-block:: bash

   pip install keeks

Keeks supports Python 3.10 through 3.14.

Run the smallest complete example
----------------------------------

.. code-block:: python

   from keeks.binary_strategies import KellyCriterion

   bankroll = 1_000.0
   strategy = KellyCriterion(
       payoff=1.0,
       loss=1.0,
       transaction_cost=0.01,
   )

   fraction = strategy.evaluate(probability=0.55, current_bankroll=bankroll)
   amount = bankroll * fraction

   print(f"Bankroll fraction: {fraction:.4%}")
   print(f"Amount from a $1,000 bankroll: ${amount:.2f}")

.. code-block:: text

   Bankroll fraction: 9.0009%
   Amount from a $1,000 bankroll: $90.01

Read the result
----------------

``evaluate()`` returns a fraction of the bankroll, not a currency amount. The
example above multiplies that fraction by the current bankroll only to make
the result concrete — the fraction itself is what you should carry into a
simulation or your own accounting.

How Keeks handles payoff, loss, and cost
------------------------------------------

Keeks calculates:

.. code-block:: text

   p / (loss + cost) - (1 - p) / (payoff - cost)

where ``p`` is the win probability, ``payoff`` and ``loss`` are the
per-unit multipliers for a win and a loss, and ``cost`` is
``transaction_cost``. The cost is added to the loss side and subtracted from
the payoff side before the ratio is taken, so it makes both a win pay a
little less and a loss cost a little more.

``transaction_cost`` is Keeks' normalized, per-unit fractional model input —
not a synonym for a broker commission, a bid-ask spread, or slippage. It does
not model market impact, venue-specific fees, correlated positions, or
portfolio rebalancing. It is also not the same quantity as the ``keeks.simulators``
classes' ``transaction_costs`` (plural): that one is a flat, absolute
bankroll amount charged once per settled bet, independent of stake size.
Passing the same number to both does not mean the same real-world cost, and
Keeks does not convert between them.

Why the result can be zero or capped
---------------------------------------

The raw formula above can be negative, and a raw fraction can also exceed
what the bankroll can safely support. ``KellyCriterion.evaluate()`` applies
two adjustments after the formula:

- **Zero floor.** If the win probability is below the strategy's
  ``min_probability`` (0.5 by default), or if the cost-adjusted payoff or
  loss is not positive, the strategy returns ``0.0`` rather than a negative
  or undefined fraction.
- **Maximum-safe-bet clamp.** The result is capped at
  ``get_max_safe_bet(current_bankroll)``, the largest stake that cannot drive
  the bankroll negative given ``loss`` and ``transaction_cost``. A
  non-positive bankroll has no safe stake at all, so the clamp returns
  ``0.0`` in that case too.

Neither adjustment prevents a real loss on a given bet — they only keep the
*sizing* from allocating more than the model can support.

Try fractional Kelly
---------------------

Full Kelly is one specific scaling choice. If you want to bet a fraction of
what full Kelly would allocate, see
:doc:`fractional-kelly-vs-kelly` for a side-by-side comparison with runnable
code.

Different problem: price a one-time gamble
---------------------------------------------

Everything above answers a repeated-bet question: given this binary model,
what fraction of the current bankroll should this rule allocate now? A
different question — what is the maximum price you would pay to enter a
single, one-time gamble with known outcomes and probabilities — is answered
by :doc:`utils`'s ``find_indifference_price()`` and the
``calculate_max_entry_price()`` method that utility-based strategies
support. Do not blend the two: a bet-sizing fraction is not an entry price.

Limits and disclaimer
------------------------

This page models a known win probability and a single binary outcome per
bet. It does not model execution venues, order books, correlated positions,
or promise investment performance. Keeks is an educational library; treat
every number above as a property of the model, not a forecast.

Next steps
----------

- :doc:`getting_started` — installation and the full bankroll/simulator flow.
- :doc:`binary_strategies` — constructor and method details for every strategy.
- :doc:`bankroll` — ``bettable_funds``, history, and drawdown protection.
- :doc:`utils` — one-time CRRA indifference pricing.
- :doc:`fractional-kelly-vs-kelly` — compare full Kelly against half and
  quarter Kelly with identical inputs.
