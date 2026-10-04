Unreleased
==========

 * Strategy contracts are enforced at the base classes: `BaseAllocationStrategy.evaluate` and `BaseMultiOutcomeStrategy.evaluate` now run the returned weight/stake vector through the same validators the simulator boundary applies, so a subclass handing back an over-budget vector — e.g. `(0.6, 0.6)` — fails at the call site with a `ValueError` naming the expectation and the received values, instead of passing silently until settlement
 * New `keeks.checks` module (re-exported at the package root): `check_strategy`, `check_allocation_strategy`, and `check_model` are runnable contract harnesses on the check_estimator pattern — probe calls that pass silently for a contract-honest implementation and raise a `ValueError` naming the violated expectation and the received value otherwise, covering fraction/weight validity, nonpositive-bankroll behavior, hook callability, draw-shape and option-count stability, seeded determinism, and `moments()` self-consistency
 * `BaseStrategy` now documents the simulator-facing hooks — `update_bankroll(new_bankroll)` and `record_result(won, return_pct=None)` — with exact signatures, when each simulator fires them, the `RuinError` refuse-and-stop settlement policy around them, and the fact that omitting them breaks nothing (they are optional; an absent hook is skipped)
 * `AllocationResult` gains an `all_cash_reason` field, populated by `MeanCVaR.optimize()` when the unit-risk-aversion objective's one-for-one trade-off holds the full-cash corner (the all-cash outcome on daily-frequency market data the ETF example footnotes), so the result object itself carries the explanation instead of returning a bare `(−0.0, −0.0)`
 * `KellyCriterion` sizing is edge-aware by default: `min_probability` now defaults to `None`, so a bet the Kelly formula itself sizes positively is placed instead of being silently zeroed by the old `0.5` probability gate (a 2:1 payoff at 40% win probability now sizes its ~9.5% Kelly fraction); an explicitly configured `min_probability` still refuses below-gate bets, and emits a `UserWarning` naming the suppressed fraction whenever it zeroes a formula-positive bet, while formula-negative bets keep refusing silently
 * `BankRoll` no longer caps withdrawals by default: `max_draw_down` defaults to `None` (no per-removal cap) because the old `0.3` default vetoed full-Kelly settlements and turned simulations into silent no-ops; passing a fraction keeps the cap
 * `RuinError` messages now carry the numbers a refusal needs — the attempted amount, the configured limit, and the current funds — for both the drawdown and the bankruptcy causes
 * Every simulator re-reports a refused settlement as a `UserWarning` carrying the attempted amount, the configured limit, and the current funds before stopping, so an early-stopped run is never silent — the three binary simulators, both multi-outcome simulators, and the allocation simulator
 * New `keeks.allocation` package sizes portfolios across N distribution-valued options — a mean vector and covariance, a scenario matrix, keeks binary bets, fitted parametric marginals, or a user sampler — under one long-only weight contract: `evaluate(current_bankroll)` returns one weight per option, each in `[0, 1]` and summing to at most `1 + PROBABILITY_SUM_TOLERANCE`, the residual being cash held at zero return; descriptors bind at construction and `optimize()` returns an `AllocationResult` (weights plus solver diagnostics: objective, convergence, iterations, estimated expected log growth, volatility) alongside the simulator-facing tuple
 * Moment-based allocators: `MeanVariance` (risk aversion λ = 1 is the second-order Kelly allocation), `GlobalMinimumVariance`, `MaximumSharpe` (Cornuéjols–Tütüncü convex reformulation of the tangency portfolio), `MaximumDiversification`, and `RiskBudgeting` (Spinu's convex objective with cyclical coordinate descent; equal budgets give equal risk contribution), plus `RiskAversionScaling`, the wrapper that shrinks any allocator toward cash — the fractional-Kelly dial (κ ↦ CRRA γ = 1/κ)
 * Scenario- and structure-based allocators: `MeanCVaR` solves the Rockafellar–Uryasev linear program over scenario rows (sample-CVaR is defined without finite variance; the shipped unit-risk-aversion objective trades expected return against the tail's expected loss one-for-one, which holds full cash on daily-frequency market data), and `HierarchicalRiskParity` clusters correlation distance, quasi-diagonalizes, and bisects recursively with inverse-variance splits — it never inverts the covariance matrix
 * Joint-return input models are the layer's universal input currency: any object exposing `sample(n_samples, rng) -> (n, N)` simple-return draws feeds every allocator through `from_model`, with adapters `binary_bets_model` (keeks-native binary bets, per-bet BLAKE2b-keyed streams, net-win settlement matching the portfolio simulator, and exact closed-form moments), `scenario_model` (empirical rows — resample with a Generator for an iid bootstrap), `marginals_model` (normal, Student-t, Laplace, lognormal, and binary marginals; independence is numpy-only, Gaussian-copula dependence is scipy-gated), `fit_marginals_model` (parametric marginals fitted to a returns series), and `estimate_moments` (Monte Carlo mean and covariance for models without closed form)
 * Estimators compose with any optimizer: `shrink_covariance` (Ledoit–Wolf shrinkage toward the structured target, with the `alpha` endpoints documented) and `black_litterman_mean` (posterior mean from market weights and views; equals the prior with no views)
 * Online allocators `FixedWeights`, `ExponentialGradient`, and `OnlineNewtonStep` adapt weights from realized joint returns through one sanctioned hook — `record_settlement(realized_returns)`, fired once per staked period — with `FixedWeights` the constant-rebalanced regret benchmark
 * `AllocationSimulator` replays an allocator over joint-return realizations: it accepts a scenario matrix or any joint-return model, weights each period's settlement as one net bankroll transaction (so drawdown evaluates the period once), charges a flat per-period cost, treats residual probability mass as an all-cash period, refuses-then-stops on a bankroll safeguard, rejects an invalid weight vector before consuming any draw, enforces a descriptor-equality gate (a scenario-bound allocator is sized with the scenarios it settles with), and pins the house reproducibility contract — a seeded run replays byte-identically from one private spawned stream
 * scipy arrives as the optional `keeks[allocation]` extra: `MeanVariance`, `GlobalMinimumVariance`, `MaximumSharpe`, `MaximumDiversification`, and `MeanCVaR` raise a pointed `ImportError` at construction without it; every other allocation method is numpy-only, and nothing else in keeks imports scipy
 * The package root re-exports the allocation surface — `BaseAllocationStrategy`, `AllocationResult`, `AllocationSimulator`, the moment/scenario/hierarchy/online allocators, the model adapters, the estimators, and the visualization helpers — so `from keeks import MeanVariance` works alongside the module paths
 * New deterministic matplotlib visualization helpers in `keeks.allocation.plots` (re-exported at the root): `bankroll_paths`, `drawdown`, `weight_evolution`, `risk_contributions`, `efficient_frontier`, `correlation_heatmap`, `dendrogram`, and `scenario_losses` — each takes the objects the API already returns, draws one bare figure, and returns the axes for further styling
 * New worked example `examples/allocation_etfs.py`: a six-ETF book (SPY, QQQ, IWM, TLT, GLD, VNQ) flows through both input-model helpers (empirical scenarios and fitted Student-t marginals) and races MeanVariance, RiskBudgeting, HRP, MeanCVaR, and ExponentialGradient against equal-weight and GMV benchmarks through the AllocationSimulator under common random numbers, printing growth/volatility/drawdown tables and rendering four charts into `examples/output/`; the default run is offline from the committed fixture `examples/data/etf_returns.csv` (provenance header with tickers, window, and refresh command) and `--refresh` re-downloads via yfinance, a development-only dependency that keeks never imports at runtime
 * The strategy benchmark (`make benchmark`) adds an allocation comparison: the allocation families run through the AllocationSimulator on a synthetic six-asset factor market under common random numbers, with the growth comparison chart drawn by the plots helpers
 * Sphinx documentation gains an allocation API page (strategy contract, input models, every allocator family, estimators, the simulator) wired into the toctree alongside the visualization-helpers page, and the README documents the extra, the input models, the method families, the simulator, and the example

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
