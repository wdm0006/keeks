Full Kelly vs Fractional Kelly in Python
===========================================

.. meta::
   :description: Compare full, half, and quarter Kelly bet fractions with identical inputs and runnable Python code using Keeks.

.. raw:: html

   <script type="application/ld+json">
   {
     "@context": "https://schema.org",
     "@type": "TechArticle",
     "headline": "Full Kelly vs Fractional Kelly in Python",
     "description": "Compare full, half, and quarter Kelly bet fractions with identical inputs and runnable Python code using Keeks.",
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

This page builds on :doc:`kelly-criterion-python`, which covers the full
Kelly formula, its inputs, and its clamps in detail. Read that page first if
you have not already.

Quick answer
------------

Fractional Kelly multiplies the full-Kelly fraction by a selected value
between zero and one. Keeks does the scaling; you supply the fraction and
the probability estimate.

Full Kelly versus fractional Kelly
-------------------------------------

.. list-table::
   :header-rows: 1

   * - Question
     - Full Kelly
     - Fractional Kelly
   * - What Keeks calculates
     - The model's full repeated-bet fraction
     - ``fraction ×`` the full-Kelly result
   * - Extra user input
     - None
     - A number from 0 through 1
   * - Same payoff/loss/cost inputs?
     - Yes
     - Yes
   * - Does the result predict profit?
     - No
     - No
   * - Can it price a one-time gamble?
     - Separate method
     - Separate method; do not mix it into this comparison

Smaller stake sizes change the strategy's exposure to variance. They do not
change, and cannot express, anything about whether the underlying
probability estimate is correct.

Python comparison with identical inputs
------------------------------------------

Both classes take the same ``payoff``, ``loss``, and ``transaction_cost``
inputs and share the same ``evaluate()`` method:

.. code-block:: python

   from keeks.binary_strategies import (
       FractionalKellyCriterion,
       KellyCriterion,
   )

   inputs = {
       "payoff": 1.0,
       "loss": 1.0,
       "transaction_cost": 0.01,
   }
   bankroll = 1_000.0
   probability = 0.55

   full = KellyCriterion(**inputs)
   half = FractionalKellyCriterion(**inputs, fraction=0.5)
   quarter = FractionalKellyCriterion(**inputs, fraction=0.25)

   for label, strategy in [
       ("Full Kelly", full),
       ("Half Kelly", half),
       ("Quarter Kelly", quarter),
   ]:
       fraction = strategy.evaluate(probability, bankroll)
       print(f"{label}: {fraction:.4%} (${bankroll * fraction:.2f})")

.. code-block:: text

   Full Kelly: 9.0009% ($90.01)
   Half Kelly: 4.5005% ($45.00)
   Quarter Kelly: 2.2502% ($22.50)

The values demonstrate scaling, not realized returns. Fractional Kelly
inherits the full strategy's zero floor and maximum-safe-bet clamp before
applying the selected fraction — see
:doc:`kelly-criterion-python`'s `Why the result can be zero or capped`
section for what those clamps do.

Half Kelly and quarter Kelly
-------------------------------

``fraction=0.5`` ("half Kelly") and ``fraction=0.25`` ("quarter Kelly") are
just ``FractionalKellyCriterion`` constructor arguments — there is no
separate class for them. Lower fractions reduce both the size of individual
stakes and the variance of the resulting bankroll path; they do not change
whether the strategy is right about the win probability.

What Keeks does and does not decide
---------------------------------------

``FractionalKellyCriterion`` performs the scaling arithmetic and applies the
same safety floor and clamp as full Kelly. It does not decide which fraction
to use, and it does not validate your probability estimate. Those are
choices you make going in.

How to compare strategies in a simulation
---------------------------------------------

A fair comparison between strategies needs:

- a **fresh bankroll and a fresh strategy object** per run — simulators
  mutate the bankroll in place, and some strategies carry state across
  ``evaluate()`` calls;
- **identical simulator inputs** (``payoff``, ``loss``,
  ``transaction_costs``, ``probability``) across every strategy being
  compared;
- **multiple seeded runs**, because a single stochastic path does not
  support a general claim about which strategy is "better"; and
- **a distribution of terminal outcomes**, not one number, since the mean of
  a Kelly-like strategy's terminal bankroll is often dominated by a handful
  of extreme paths.

Keeks ships a seeded, reproducible benchmark that runs all nine strategies
under identical assumptions and reports exactly this kind of distribution —
median and percentile terminal bankroll, drawdown, growth rate, and
early-stop rates. See :doc:`strategy_benchmark` for the current figures;
this page does not restate them, so there is only one place they can go
stale.

Repeated sizing is not one-time pricing
-------------------------------------------

Both classes on this page answer a repeated-bet question. A different
question — the maximum price to pay for a single, one-time gamble with known
outcomes and probabilities — is answered by :doc:`utils`'s
``find_indifference_price()`` instead. Keep the two separate.

Next step
---------

Run the comparison code above with your own probability and fraction, then
see :doc:`kelly-criterion-python` for the full-Kelly formula and clamp
details, or :doc:`strategy_benchmark` for how all nine strategies behave
under identical, seeded assumptions.
