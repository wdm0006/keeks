# Keeks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)

**Python bet sizing and bankroll simulation for developers modeling repeated
binary outcomes.**

Given your estimated win probability, payoff, loss, and transaction cost, Keeks
returns the fraction of your bankroll to stake — and can simulate that same rule
over repeated trials to show the bankroll path it produces. For portfolios of
options described by return distributions rather than single bets, the
[`keeks.allocation`](https://keeks.mcginniscommawill.com/allocation.html) layer
sizes them together and returns one long-only weight per option.

## Install

```bash
pip install keeks
```

Keeks supports Python 3.10 through 3.14. The portfolio-allocation layer's
QP/LP solvers need one optional extra (everything else is numpy-only):
`pip install "keeks[allocation]"`.

## Quick start: size a bet

```python
from keeks.binary_strategies import KellyCriterion

bankroll = 1_000.0
strategy = KellyCriterion(
    payoff=1.0,
    loss=1.0,
    transaction_cost_rate=0.01,
)

fraction = strategy.evaluate(probability=0.55, current_bankroll=bankroll)
amount = bankroll * fraction

print(f"Bankroll fraction: {fraction:.4%}")
print(f"Amount from a $1,000 bankroll: ${amount:.2f}")
```

```text
Bankroll fraction: 9.0009%
Amount from a $1,000 bankroll: $90.01
```

`evaluate()` returns a fraction, not a currency amount; the example multiplies it by the bankroll only to make the result concrete.

## Quick start: allocate a portfolio

Binary strategies size one bet at a time; the allocation layer sizes a whole
portfolio at once, returning one long-only weight per option (each in `[0, 1]`,
summing to at most one — the residual is cash held at zero return). Adapters
build a joint-return model from what you have — binary bets, a scenario matrix,
fitted marginals, or any callable sampler — and allocators consume it through
one sampling contract. `MeanVariance` at risk aversion λ = 1 is the
second-order Kelly allocation; the
[allocation documentation](https://keeks.mcginniscommawill.com/allocation.html)
covers the other method families (scenario-based, hierarchical, online, and the
covariance estimators).

```python
from keeks import MeanVariance, binary_bets_model

# Three simultaneous binary bets: (win probability, decimal odds, loss)
model = binary_bets_model([(0.55, 2.0, 1.0), (0.42, 2.5, 1.0), (0.70, 1.5, 1.0)])

# λ = 1 mean-variance is the second-order Kelly allocation
strategy = MeanVariance.from_model(model, n_samples=4096, seed=42)
print(strategy.evaluate(current_bankroll=1_000.0))  # weights, not amounts
```

## Choose a strategy

All nine strategies expose `evaluate(probability, current_bankroll)`, but their
constructors and sizing rules differ.

| Strategy | Choose it when you want to model |
|---|---|
| `KellyCriterion` | Full Kelly sizing from a binary win probability, payoff, loss, and cost. |
| `FractionalKellyCriterion` | A fixed fraction of the full-Kelly result. |
| `DrawdownAdjustedKelly` | Kelly sizing scaled by an acceptable-drawdown input. |
| `OptimalF` | A geometric-growth rule based on a supplied win rate, with a risk-fraction cap. |
| `FixedFractionStrategy` | A constant fraction above a minimum probability; useful as a baseline. |
| `CPPIStrategy` | A cushion-based rule relative to a bankroll floor. |
| `DynamicBankrollManagement` | A fraction adjusted from recent recorded outcomes. |
| `MertonShare` | A CRRA risk-aversion rule adapted to binary outcomes. |
| `NaiveStrategy` | A positive-expected-value rule without utility-based sizing. |

See the [strategy API](https://keeks.mcginniscommawill.com/binary_strategies.html)
for constructor parameters and formulas.

## Simulate a bankroll

Once you have chosen a strategy, pass it and a fresh `BankRoll` to a simulator:

```python
from keeks.bankroll import BankRoll
from keeks.binary_strategies import FractionalKellyCriterion
from keeks.simulators.repeated_binary import RepeatedBinarySimulator

bankroll = BankRoll(
    initial_funds=1_000.0,
    percent_bettable=0.8,
    max_transaction_loss=0.3,
)
strategy = FractionalKellyCriterion(
    payoff=1.0,
    loss=1.0,
    transaction_cost_rate=0.01,
    fraction=0.5,
)
simulator = RepeatedBinarySimulator(
    payoff=1.0,
    loss=1.0,
    fee_per_bet=0.01,
    probability=0.55,
    trials=1_000,
)

simulator.evaluate_strategy(strategy, bankroll)

print(f"Final bankroll: ${bankroll.total_funds:.2f}")
bankroll.plot_history(fname="bankroll-history.png")
```

The test suite executes this exact block as a seeded, down-scaled demo — the
simulator clamped to 50 trials with a fixed seed and the plot rendered
headlessly — so the example stays runnable; the narrative above describes the
full 1,000-trial run.

Simulation mutates the bankroll and records its history. Use matching payoff and
loss assumptions in the strategy and simulator: Keeks enforces the match — every
simulator's `evaluate_strategy` checks that a `BaseStrategy` instance's `payoff`
and `loss` agree with the odds it settles at, and raises `ValueError` on a
mismatch. Duck-typed strategies that carry no odds of their own are not checked —
their compatibility stays your responsibility.

Portfolios of distribution-valued options simulate through the same bankroll machinery — see the [allocation documentation](https://keeks.mcginniscommawill.com/allocation.html) and the [six-ETF worked example](examples/allocation_etfs.py).

## Where to go next

- [Getting started](https://keeks.mcginniscommawill.com/getting_started.html) — a fuller walkthrough of both layers
- [Portfolio allocation](https://keeks.mcginniscommawill.com/allocation.html) — joint-return models, allocator families, and the simulator; plus the [visualization helpers](https://keeks.mcginniscommawill.com/allocation_plots.html)
- API references: [strategies](https://keeks.mcginniscommawill.com/binary_strategies.html), [bankroll](https://keeks.mcginniscommawill.com/bankroll.html), [simulators](https://keeks.mcginniscommawill.com/simulators.html)
- [Contract checks](https://keeks.mcginniscommawill.com/checks.html) — runnable harnesses for extending the library
- [Nine-strategy risk benchmark](https://keeks.mcginniscommawill.com/strategy_benchmark.html) — what each shipped strategy actually produces under identical, seeded assumptions; regenerate every number with `uv run python benchmarks/strategy_benchmark.py`
- Examples: [`examples/strategy_comparison.py`](examples/strategy_comparison.py) (all nine strategies, headless), [`examples/st_petersburg_paradox.py`](examples/st_petersburg_paradox.py) (repeated sizing is not one-time pricing), and [`examples/allocation_etfs.py`](examples/allocation_etfs.py) (a six-ETF portfolio)

## Good to know

- **Cost units differ.** A strategy's `transaction_cost_rate` is a per-unit
  *fractional* cost that scales with stake size; a simulator's `fee_per_bet` is
  a flat, *absolute* bankroll amount charged once per settled bet. The two are
  different units — passing the same number to both models as two different
  real-world costs, and Keeks does not convert between them.
- **Edge-aware gating.** `KellyCriterion` sizes on edge by default
  (`min_probability=None`): any bet its formula prices positively is placed, and
  a bet with no positive edge returns 0.0. Set `min_probability` to add a
  longshot gate — a trial probability below it returns 0.0 even when the Kelly
  fraction is positive, with a `UserWarning` naming the suppressed fraction.
- Fractions are floored at zero and capped so the modeled loss plus transaction
  cost cannot exceed the current bankroll — but constraints do not prevent
  losses or verify your probability estimate. The models are not advice.

## References

- [1] [A New Interpretation of Information Rate](http://www.herrold.com/brokerage/kelly.pdf) - The original Kelly Criterion paper
- [2] [The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market](https://www.amazon.com/Kelly-Criterion-Blackjack-Sports-Betting/dp/1096432366) - A practical guide to applying the Kelly Criterion
- [3] [Fortune's Formula](https://www.amazon.com/Fortunes-Formula-Scientific-Betting-Casinos/dp/0809045990) - The untold story of the scientific betting system that beat the casinos and Wall Street

## Disclaimer

Keeks is for educational purposes. It does not provide investment, legal, or tax
advice. Models and simulations can be wrong, and financial loss is possible. You
are responsible for validating your inputs and deciding whether any real-world use
is appropriate.

## Contributing

Contributions are welcome. To set up the project and run its checks:

```bash
git clone https://github.com/wdm0006/keeks.git
cd keeks
make setup
make install-dev
make test
make lint
```

Extending a strategy, allocator, or model? Verify it against the library's contracts before opening a PR — see [Contract checks](https://keeks.mcginniscommawill.com/checks.html).

Keeks is available under the [MIT License](LICENSE.md).
