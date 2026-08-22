v0.6.0
======

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

v0.5.0
======

**Changed:**
 * OptimalF now uses its historical or expected win rate for bet sizing while retaining the per-trial probability gate

v0.4.0
======

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

v0.3.0
======

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

v0.2.0
======

 * no longer allowing negative bets

v0.1.0
======

 * modernizing, added new methods, examples fixes

v0.0.2
======

 * adding CI, docs, and testing

v0.0.1
======

 * first release
