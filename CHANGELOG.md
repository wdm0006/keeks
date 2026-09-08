v0.8.0 (2026-09-08)
===================

 * pandas is now a development-only dependency rather than part of the core install
 * `OptimalF.evaluate` now returns Ralph Vince's optimal f converted from a risk fraction to a stake fraction (`f* / (loss + transaction_cost)`), so it is the TWR-optimal stake; sizes are unchanged when `loss + transaction_cost = 1` and otherwise grow for losses under 1 and shrink for losses over 1
 * `find_indifference_price` and `expected_utility` no longer let explicit zero-probability outcomes poison the expected utility with NaN when their final wealth is nonpositive; affected gambles now price correctly and the saturation warning fires instead of being silently suppressed
 * `MertonShare` no longer raises `OverflowError` for extreme payoffs (about 1.34e154 and above); the variance saturates and the strategy returns 0.0
 * Strategy docstrings now describe `transaction_cost` as the per-unit fractional cost it is, not a fixed per-transaction amount
 * `keeks.__version__` now reports the installed version, with `version.py` as the single source: the Sphinx build imports it and hatchling's dynamic versioning consumes it, so `pyproject.toml` no longer duplicates the number
 * The package root re-exports the documented public API — `BankRoll`, `RuinError`, `BaseStrategy`, all nine binary strategies, the three binary simulators, the multi-outcome surface (`BaseMultiOutcomeStrategy`, `MultiOutcomeKellyCriterion`, `RepeatedMultiOutcomeSimulator`, `PortfolioSimulator`), and the CRRA utilities (`crra_utility`, `expected_utility`, `find_indifference_price`) — so `from keeks import BankRoll` works alongside the module paths
 * New public `keeks.utils.normalize_probabilities` (also re-exported from the package root) validates a probability vector — finite, nonnegative, summing to no more than one within `PROBABILITY_SUM_TOLERANCE` — and returns it as a float array; `_normalize_gamble` now delegates its validation there, leaving gamble behavior unchanged
 * New `keeks.multi_outcome` package: `BaseMultiOutcomeStrategy` generalizes the strategy contract to mutually exclusive markets — `evaluate(probabilities, current_bankroll)` returns one stake fraction per leg as a tuple (`len == len(probabilities)`, each in `[0, 1]`, the sum at most `1 + PROBABILITY_SUM_TOLERANCE`), a vector stake validator mirrors the scalar stake gate's error discipline, and `get_max_safe_total_bet` caps the aggregate stake at the worst leg's `min(1, 1 / (loss + transaction_cost))` bound; the binary surface is untouched
 * New `MultiOutcomeKellyCriterion` in `keeks.multi_outcome` sizes one stake fraction per leg of a mutually exclusive market to maximize expected log growth `sum_i p_i * log(1 + a_i * f_i - l * sum_{k != i} f_k)` with `a_i = payoff_i - 1 - transaction_cost` and `l = loss + transaction_cost`: at two legs on a fully priced book whose delegated binary point passes the joint KKT check it returns exactly what `KellyCriterion` returns per leg, and otherwise a deterministic coordinate-ascent solver with floor-capped pairwise transfers finds the joint optimum under the aggregate safe-stake cap while keeping every positive-probability leg's realized multiplier above a ruin floor; zero-edge and zero-probability legs are never staked
 * New `RepeatedMultiOutcomeSimulator` in `keeks.multi_outcome` evaluates multi-outcome strategies on one fixed mutually exclusive market: one categorical draw per trial realizes exactly one leg (probability mass below one is a void or push round that refunds every stake), the batch settles leg by leg through the same net-settlement flow the binary simulators use with stakes fixed from the bankroll as it stood when the trial began, a `record_settlement` hook reports every settled batch, and a settlement that trips a bankroll safeguard leaves that settlement unchanged and stops the simulation after the batch completes — never mid-batch
 * New `PortfolioSimulator` in `keeks.multi_outcome` evaluates multi-outcome strategies over a portfolio of M independent binary bets: every staked bet settles win or lose on its own draw, the batch's settlements net into exactly one bankroll transaction so drawdown safeguards evaluate the batch total once, an aggregate-exposure check rejects stake vectors worth more than the bettable funds rather than silently reducing them, and a refused batch leaves the bankroll unchanged and stops the simulation after the batch completes — never mid-portfolio
 * Both multi-outcome simulators pin a reproducibility contract as public behavior: with a `seed`, every draw comes from a private `numpy.random.Generator` on a spawned `SeedSequence` child (one for the repeated simulator, one per bet for the portfolio), so a seeded run replays byte-identically — same bankroll history, same hook calls — and adding, removing, or reordering portfolio bets never shifts a surviving bet's stream; without a seed, numpy's global generator drives the draws
 * A Sphinx documentation page covers the multi-outcome surface — the strategy contract, the multi-outcome Kelly, both simulators, and the seeding and reproducibility contract — and a runnable 1X2 home-draw-away example (`examples/multi_outcome_1x2.py`, with committed CSV and chart outputs) compares the multi-outcome Kelly against flat and favorite-only staking on a seeded market with void rounds

v0.6.0 (2026-08-22)
===================

**Added:**
 * Deterministic nine-strategy risk benchmark (`benchmarks/strategy_benchmark.py`, `make benchmark`) with committed CSV and chart artifacts and a documentation page covering growth, drawdown, percentile bands and early stops across edge, cost, estimate-error and drawdown-cap scenarios
 * Explicit simulator seeds, so a strategy comparison reproduces run to run
 * A bankroll update hook that fires in every simulator, which is the feedback path an adaptive strategy needs to size from settled outcomes
 * Simulator classes are re-exported from the package root

**Changed:**
 * `find_indifference_price` now signals when it saturates at its search bound instead of returning the bound silently
 * The transaction-cost unit difference across the API boundary is documented at every entry point
 * Docstring examples are gated in CI, so the getting-started snippets stay runnable

**Validation:**
 * Strategy economics must be finite at construction, and stake fractions must be valid
 * Simulator configuration is validated at construction, and a simulation is refused when its strategy odds contradict its settlement odds
 * `crra_utility` no longer evaluates a log or power over nonpositive wealth
 * `get_max_safe_bet` is guarded against zero and negative bankrolls, and drawdown safeguards are enforced when a bet is placed
 * Scalar inputs and strategy-specific numeric controls are validated for entry-price calculations

**Note:** v0.5.0 was tagged in the changelog but never published to PyPI, so its OptimalF sizing change ships here.

v0.5.0 (never published)
=========================

**Changed:**
 * OptimalF now uses its historical or expected win rate for bet sizing while retaining the per-trial probability gate

v0.4.0 (2026-08-01)
====================

**Changed:**
 * Dynamic bankroll management now skips bets below its configurable minimum probability, respects the maximum safe bet, and adapts from settled simulator outcomes
 * Python 3.10 is now the minimum supported version, with support extended through Python 3.13

**Critical Bug Fixes:**
 * Corrected Kelly-family bet sizing when using an explicit loss multiplier
 * Added finite, nonnegative validation for bankroll configuration and transactions, and made zero drawdown enforce a true zero-loss limit
 * Accounted for omitted probability mass as a zero-payout outcome in expected utility and indifference pricing, while rejecting malformed probability distributions
 * Clamped random simulator probabilities to valid bounds and stopped simulations cleanly when bankroll safeguards reject a settlement
 * Routed fee-dominated wins through bankroll withdrawal safeguards instead of depositing a negative amount
 * Corrected bankroll history plotting for current Matplotlib versions

v0.3.0 (2025-10-11)
====================

**New Features:**
 * Added MertonShare strategy based on Merton's portfolio problem with CRRA utility
 * MertonShare supports configurable risk aversion parameter (gamma)
 * Added CRRA utility functions to keeks.utils: `crra_utility()`, `expected_utility()`, `find_indifference_price()`
 * Added `calculate_max_entry_price()` method to ALL 9 strategies:
   - **Utility-based**: KellyCriterion, MertonShare, OptimalF use CRRA utility
   - **Kelly variants**: FractionalKelly, DrawdownAdjustedKelly scale Kelly's price
   - **Rule-based**: FixedFraction, CPPI, Dynamic apply mechanical rules
   - **Risk-neutral**: NaiveStrategy pays expected value
 * Added St. Petersburg paradox example demonstrating all 9 strategies
 * Updated strategy comparison example to include MertonShare with different risk aversion levels
 * Added comprehensive tests for MertonShare strategy (16 test cases)
 * Added comprehensive tests for utility functions (18 test cases)
 * Added comprehensive tests for strategy entry price methods (17 test cases)
 * Updated documentation to include MertonShare, utility functions, and entry price methods
 * Example outputs now include both CSV and PNG visualizations

**Critical Bug Fixes:**
 * Fixed bankruptcy protection - bankrolls can no longer go negative
 * Fixed transaction cost bug in simulators - costs now correctly increase losses instead of decreasing them
 * Fixed transaction costs being charged on zero bets - costs only apply when actually betting
 * Fixed CPPI hardcoded 1% edge threshold - now works with realistic 0.5-2% edges
 * Added bankruptcy checks to all simulators - simulations now stop when bankroll is depleted
 * Fixed drawdown check ordering - bankruptcy is now checked before drawdown limits
 * Fixed ruin rate detection in examples - now properly detects early termination
 * Added 9 comprehensive bankruptcy protection tests
 * Updated example parameters to realistic professional bettor scenario (52% win, 0.95x payoff, 0.4% edge)

v0.2.0 (2025-03-20)
====================

 * no longer allowing negative bets

v0.1.0 (2025-03-09)
====================

 * modernizing, added new methods, examples fixes

v0.0.2 (never published)
=========================

 * adding CI, docs, and testing

v0.0.1 (never published)
=========================

 * first release
