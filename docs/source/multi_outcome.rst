Multi-Outcome Strategies and Simulators
=======================================

Multi-outcome tools generalize the binary strategy surface to **mutually
exclusive markets** — a 1X2 football match (home win, draw, away win), for
instance — where exactly one leg settles per round, and to **portfolios of
independent binary bets** — M bookmaker markets all settling at once.

The strategy contract is the binary contract made vector-valued: a strategy
receives one probability per leg and returns one stake fraction of the
bankroll per leg. Probabilities are validated by
:func:`keeks.utils.validate_probabilities` — finite, nonnegative, summing to
no more than one within tolerance — and probability mass below one models a
void or push round on which no leg settles. The returned fractions are each
in ``[0, 1]`` and sum to at most one: the legs together can never promise
more of the bankroll than it holds. Payoffs are decimal odds (a winning leg
pays its payoff multiplier times its stake, stake included), and strategies
price with a per-unit fractional ``transaction_cost_rate`` while the simulators
charge a flat per-settlement ``fee_per_bet`` fee — the two units are
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

Historical Multi-Outcome Simulator
----------------------------------

Replay chronological estimates and recorded winning leg indices with
``HistoricalMultiOutcomeSimulator``. Use ``None`` for a void or push round;
no random draws or seed are involved. The fixed decimal odds and settlement
rules match the repeated simulator, while estimates can change each trial.

.. autoclass:: keeks.multi_outcome.simulators.HistoricalMultiOutcomeSimulator
    :members:

Comparing Sizing Policies on a Recorded Log
-------------------------------------------

:func:`keeks.comparison.compare_multi_outcome_strategies` replays one recorded
market log through several policies, each on a fresh bankroll, and ranks them
like :func:`~keeks.comparison.compare_strategies`. Pass name-to-factory
mappings (each factory must return a fresh vector strategy), the probability
rows, and the winning leg per market (``None`` for a void). The decimal-odds
``payoffs``, ``loss`` and flat ``fee_per_bet`` are fixed for the whole log and
every strategy must be built with the same ``payoffs`` and ``loss``.

Rows of the log are markets, but ``summary.bets`` counts settled transactions:
a void or a declined market adds none, and a market staking two legs adds two.
A void charges no fee.

.. code-block:: pycon

    >>> from keeks import MultiOutcomeKellyCriterion, compare_multi_outcome_strategies
    >>> class CappedKelly(MultiOutcomeKellyCriterion):
    ...     """Kelly stakes scaled down to a fixed share."""
    ...     def __init__(self, payoffs, loss, scale):
    ...         super().__init__(payoffs, loss)
    ...         self.scale = scale
    ...     def evaluate(self, probabilities, current_bankroll):
    ...         stakes = super().evaluate(probabilities, current_bankroll)
    ...         return tuple(self.scale * stake for stake in stakes)
    >>> odds = (3.2, 3.4, 2.4)
    >>> rows = compare_multi_outcome_strategies(
    ...     {
    ...         "Full Kelly": lambda: MultiOutcomeKellyCriterion(odds, 1.0),
    ...         "Quarter Kelly": lambda: CappedKelly(odds, 1.0, 0.25),
    ...     },
    ...     [[0.42, 0.27, 0.28]] * 4,
    ...     [0, 2, None, 1],
    ...     payoffs=odds, loss=1.0, fee_per_bet=0.0, initial_funds=1000.0,
    ... )
    >>> [(row.name, round(row.summary.end, 2), row.summary.bets) for row in rows]
    [('Quarter Kelly', 1009.71, 6), ('Full Kelly', 962.21, 6)]

.. autofunction:: keeks.comparison.compare_multi_outcome_strategies

Portfolio Simulator
-------------------

.. autoclass:: keeks.multi_outcome.simulators.PortfolioSimulator
    :members:
    :undoc-members:
    :show-inheritance:

Seeding and Reproducibility
---------------------------

The two random simulators treat seeding as public, testable behavior — the same
contract the binary simulators introduced in v0.6.0, generalized to
per-stream spawned children:

- **With a ``seed``, a run replays byte-identically.** Every draw comes
  from a private :class:`numpy.random.Generator`, never from numpy's global
  state, so rerunning a seeded construction reproduces the same bankroll
  history and the same ``record_settlement`` hook calls.
- **Streams are deterministic derivatives of the seed.**
  :class:`RepeatedMultiOutcomeSimulator` derives its single settlement
  stream from the first child of ``numpy.random.SeedSequence(seed).spawn(1)``;
  :class:`PortfolioSimulator` derives each stream from the seed and a stable
  BLAKE2 digest of the validated bet's exact three IEEE-754 float values.
  Adding, removing, or reordering heterogeneous bets therefore never shifts
  a surviving bet's stream. Duplicate identical tuples use zero-based
  occurrence ordinals in portfolio order to receive independent deterministic
  streams. Since identical bets are indistinguishable, reordering duplicates
  need not preserve a distinguishable stream for a particular duplicate.
- **Without a seed, no replay is promised.** The draws then come from
  numpy's global generator, matching the unseeded binary simulators.

The test suite pins this contract with bit-exactness golden files for both
simulators, so a change that alters a seeded stream fails CI before it
reaches a user.

A runnable 1X2 example — the multi-outcome Kelly criterion against flat and
favorite-only staking on a home-draw-away market — ships as
``examples/multi_outcome_1x2.py`` in the repository.
