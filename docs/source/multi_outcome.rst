Multi-Outcome Strategies and Simulators
=======================================

Multi-outcome tools generalize the binary strategy surface to **mutually
exclusive markets** — a 1X2 football match (home win, draw, away win), for
instance — where exactly one leg settles per round, and to **portfolios of
independent binary bets** — M bookmaker markets all settling at once.

The strategy contract is the binary contract made vector-valued: a strategy
receives one probability per leg and returns one stake fraction of the
bankroll per leg. Probabilities are validated by
:func:`keeks.utils.normalize_probabilities` — finite, nonnegative, summing to
no more than one within tolerance — and probability mass below one models a
void or push round on which no leg settles. The returned fractions are each
in ``[0, 1]`` and sum to at most one: the legs together can never promise
more of the bankroll than it holds. Payoffs are decimal odds (a winning leg
pays its payoff multiplier times its stake, stake included), and strategies
price with a per-unit fractional ``transaction_cost`` while the simulators
charge a flat per-settlement ``transaction_costs`` fee — the two units are
documented at every entry point and are never interchangeable.

The Strategy Contract
---------------------

.. autoclass:: keeks.multi_outcome.base.BaseMultiOutcomeStrategy
    :members:
    :undoc-members:
    :show-inheritance:

Multi-Outcome Kelly Criterion
-----------------------------

.. autoclass:: keeks.multi_outcome.kelly.MultiOutcomeKellyCriterion
    :members:
    :undoc-members:
    :show-inheritance:

Repeated Multi-Outcome Simulator
--------------------------------

.. autoclass:: keeks.multi_outcome.simulators.RepeatedMultiOutcomeSimulator
    :members:
    :undoc-members:
    :show-inheritance:

Portfolio Simulator
-------------------

.. autoclass:: keeks.multi_outcome.simulators.PortfolioSimulator
    :members:
    :undoc-members:
    :show-inheritance:

Seeding and Reproducibility
---------------------------

Both simulators treat seeding as public, testable behavior — the same
contract the binary simulators introduced in v0.6.0, generalized to
per-stream spawned children:

- **With a ``seed``, a run replays byte-identically.** Every draw comes
  from a private :class:`numpy.random.Generator`, never from numpy's global
  state, so rerunning a seeded construction reproduces the same bankroll
  history and the same ``record_settlement`` hook calls.
- **Streams are spawned children of the seed, not the seed itself.**
  :class:`RepeatedMultiOutcomeSimulator` derives its single settlement
  stream from the first child of ``numpy.random.SeedSequence(seed).spawn(1)``;
  :class:`PortfolioSimulator` derives one stream per bet from
  ``numpy.random.SeedSequence(seed).spawn(len(bets))``, with bet ``m``
  owning child ``m``. Each stream owns an independent child seed, so a
  stream added later cannot shift the settlement stream's draws — adding,
  removing, or reordering portfolio bets never shifts a surviving bet's
  stream.
- **Without a seed, no replay is promised.** The draws then come from
  numpy's global generator, matching the unseeded binary simulators.

The test suite pins this contract with bit-exactness golden files for both
simulators, so a change that alters a seeded stream fails CI before it
reaches a user.

A runnable 1X2 example — the multi-outcome Kelly criterion against flat and
favorite-only staking on a home-draw-away market — ships as
``examples/multi_outcome_1x2.py`` in the repository.
