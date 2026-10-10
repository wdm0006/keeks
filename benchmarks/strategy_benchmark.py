"""Deterministic nine-strategy risk benchmark.

Runs every strategy exported from ``keeks.binary_strategies`` through
``RepeatedBinarySimulator`` over a fixed scenario matrix and writes the growth,
drawdown and early-stop metrics to ``benchmarks/output/``. It then races the
portfolio-allocation families through ``AllocationSimulator`` on a synthetic
six-asset factor market and writes their growth comparison with the
``keeks.allocation`` plots helpers.

Reproduce with::

    uv run python benchmarks/strategy_benchmark.py

Design notes that the numbers depend on:

* **Fresh state per run.** Every (scenario, strategy, path) triple builds a new
  ``BankRoll`` and a new strategy instance, because simulators mutate the bankroll
  in place and ``CPPIStrategy`` / ``DynamicBankrollManagement`` carry state between
  ``evaluate`` calls.
* **Common random numbers.** Outcomes are drawn once per path from a seeded
  ``random.Random`` and replayed by trial index, so trial *t* of a given path
  resolves identically for all nine strategies even when some of them decline to
  bet. Seeding the global RNG alone would not achieve this: the simulator only
  draws when a bet is placed, so a strategy that skips a trial would otherwise
  shift every later outcome.
* **Estimate error is applied at the strategy boundary.** The shipped uncertain
  simulator centres its probability draws on 0.5, which cannot express an edge, so
  the estimate-noise and estimate-bias axes instead perturb the probability handed to
  ``strategy.evaluate`` while the simulator settles against the true probability.
* **Cost units differ by design of the library.** Strategies treat
  ``transaction_cost_rate`` as a per-unit fractional cost; simulators subtract
  ``fee_per_bet`` as a flat fee per settled bet. The same scalar is passed to
  both, and the realised fee is measured and reported so the asymmetry is visible
  rather than assumed away.
"""

import math
import random
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from keeks import (  # noqa: E402
    AllocationSimulator,
    ExponentialGradient,
    FixedWeights,
    GlobalMinimumVariance,
    HierarchicalRiskParity,
    MeanVariance,
    RiskBudgeting,
    bankroll_paths,
    scenario_model,
    summarize_history,
)
from keeks.bankroll import BankRoll  # noqa: E402
from keeks.binary_strategies import (  # noqa: E402
    CPPIStrategy,
    DrawdownAdjustedKelly,
    DynamicBankrollManagement,
    FixedFractionStrategy,
    FractionalKellyCriterion,
    KellyCriterion,
    MertonShare,
    NaiveStrategy,
    OptimalF,
)
from keeks.simulators import repeated_binary  # noqa: E402
from keeks.simulators.repeated_binary import RepeatedBinarySimulator  # noqa: E402
from keeks.utils import RuinError  # noqa: E402

SEED = 20260803
INITIAL_FUNDS = 1000.0
TRIALS = 500
PATHS = 200
# Terminal wealth floor used only so a fully depleted path has a finite log growth
# rate; a depleted path is reported as an early stop in its own right.
WEALTH_FLOOR = 0.01

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


@dataclass(frozen=True)
class Scenario:
    """One point of the benchmark matrix."""

    key: str
    axis: str
    label: str
    probability: float
    cost: float
    estimate_stdev: float
    max_transaction_loss: float | None
    payoff: float = 1.0
    loss: float = 1.0
    estimate_bias: float = 0.0


BASE = Scenario(
    key="base",
    axis="base",
    label="Base: 55% edge, even money, no cost, no estimate error, max_transaction_loss=0.3",
    probability=0.55,
    cost=0.0,
    estimate_stdev=0.0,
    max_transaction_loss=0.3,
)


def _variant(key, axis, label, **overrides):
    return replace(BASE, key=key, axis=axis, label=label, **overrides)


SCENARIOS = [
    BASE,
    _variant("edge-51", "edge", "Edge: 51% win probability", probability=0.51),
    _variant("edge-60", "edge", "Edge: 60% win probability", probability=0.60),
    _variant("cost-01", "cost", "Cost input: 0.01", cost=0.01),
    _variant("cost-05", "cost", "Cost input: 0.05", cost=0.05),
    _variant(
        "noise-03",
        "estimate error",
        "Estimate error: probability known to +/- 0.03 (1 sd)",
        estimate_stdev=0.03,
    ),
    _variant(
        "noise-06",
        "estimate error",
        "Estimate error: probability known to +/- 0.06 (1 sd)",
        estimate_stdev=0.06,
    ),
    _variant(
        "bias+03",
        "estimate bias",
        "Estimate bias: +3 percentage points",
        estimate_bias=0.03,
    ),
    _variant(
        "bias+05",
        "estimate bias",
        "Estimate bias: +5 percentage points",
        estimate_bias=0.05,
    ),
    _variant(
        "bias-03",
        "estimate bias",
        "Estimate bias: -3 percentage points",
        estimate_bias=-0.03,
    ),
    _variant(
        "drawdown-08",
        "drawdown limit",
        "Drawdown limit: 8% of funds per settlement",
        max_transaction_loss=0.08,
    ),
    _variant(
        "drawdown-03",
        "drawdown limit",
        "Drawdown limit: 3% of funds per settlement",
        max_transaction_loss=0.03,
    ),
    _variant(
        "drawdown-off",
        "drawdown limit",
        "Drawdown limit: disabled (max_transaction_loss=None)",
        max_transaction_loss=None,
    ),
]

# One factory per name in keeks.binary_strategies.__all__. Factories rather than
# instances because CPPIStrategy and DynamicBankrollManagement carry state across
# evaluate() calls, so each path needs its own object.
STRATEGY_FACTORIES = {
    "Kelly": lambda s: KellyCriterion(
        payoff=s.payoff, loss=s.loss, transaction_cost_rate=s.cost
    ),
    "Half Kelly": lambda s: FractionalKellyCriterion(
        payoff=s.payoff, loss=s.loss, transaction_cost_rate=s.cost, fraction=0.5
    ),
    "Drawdown-adjusted Kelly": lambda s: DrawdownAdjustedKelly(
        payoff=s.payoff,
        loss=s.loss,
        transaction_cost_rate=s.cost,
        max_transaction_loss=0.2,
    ),
    "Optimal f": lambda s: OptimalF(
        payoff=s.payoff,
        loss=s.loss,
        transaction_cost_rate=s.cost,
        win_rate=s.probability,
        max_risk_fraction=0.2,
    ),
    "Naive": lambda s: NaiveStrategy(
        payoff=s.payoff, loss=s.loss, transaction_cost_rate=s.cost
    ),
    "Fixed fraction 2%": lambda s: FixedFractionStrategy(
        fraction=0.02, payoff=s.payoff, loss=s.loss, transaction_cost_rate=s.cost
    ),
    "CPPI": lambda s: CPPIStrategy(
        floor_fraction=0.8,
        multiplier=2.0,
        initial_bankroll=INITIAL_FUNDS,
        payoff=s.payoff,
        loss=s.loss,
        transaction_cost_rate=s.cost,
    ),
    "Dynamic": lambda s: DynamicBankrollManagement(
        base_fraction=0.05, payoff=s.payoff, loss=s.loss, transaction_cost_rate=s.cost
    ),
    "Merton share": lambda s: MertonShare(
        payoff=s.payoff, loss=s.loss, transaction_cost_rate=s.cost, risk_aversion=2.0
    ),
}


class _StoppedBankRoll(BankRoll):
    """A BankRoll that records why a settlement was refused.

    The simulator catches ``RuinError`` and breaks, which leaves no signal of the
    reason. Recording it here is what lets the benchmark separate a drawdown-limit
    stop from a bankruptcy stop instead of inferring both from a short history.
    """

    def __init__(self, initial_funds, max_transaction_loss):
        super().__init__(
            initial_funds=initial_funds, max_transaction_loss=max_transaction_loss
        )
        self.stop_reason = ""

    def withdraw(self, amount):
        try:
            super().withdraw(amount)
        except RuinError:
            self.stop_reason = (
                "bankruptcy" if self.total_funds - amount < 0 else "drawdown-limit"
            )
            raise


class _Clock:
    """Shared trial index between the strategy wrapper and the outcome source."""

    def __init__(self):
        self.trial = -1


class _ReplayedOutcomes:
    """Stand-in for the ``random`` module inside the simulator module.

    Exposes only ``random()``, returning the pre-drawn uniform for the current
    trial so that every strategy meets the same outcome at the same trial index.
    """

    def __init__(self, draws, clock):
        self._draws = draws
        self._clock = clock

    def random(self):
        return self._draws[self._clock.trial]


@dataclass
class PathResult:
    terminal: float
    trials_started: int
    bets_placed: int
    max_drawdown: float
    growth_rate: float
    stop_reason: str
    fees_paid: float
    staked: float
    first_bet_fraction: float


def run_path(scenario, strategy_name, path_index):
    """Run one strategy over one seeded path and return its metrics."""
    # random.Random seeds a string deterministically (SHA-512 of the bytes), unlike
    # hash(), which is salted per process. Both streams are keyed on the path index
    # alone, so the same 200 outcome sequences and the same estimate-error shocks
    # are reused by every strategy *and* every scenario: a cell that a scenario axis
    # does not touch reproduces its base-scenario number exactly rather than
    # re-rolling it.
    outcome_rng = random.Random(f"outcomes|{SEED}|{path_index}")
    draws = [outcome_rng.random() for _ in range(TRIALS)]
    if scenario.estimate_stdev:
        shock_rng = random.Random(f"shocks|{SEED}|{path_index}")
        beliefs = [
            min(
                1.0,
                max(
                    0.0,
                    scenario.probability
                    + scenario.estimate_bias
                    + scenario.estimate_stdev * shock_rng.gauss(0, 1),
                ),
            )
            for _ in range(TRIALS)
        ]
    else:
        beliefs = [
            min(1.0, max(0.0, scenario.probability + scenario.estimate_bias))
        ] * TRIALS

    bankroll = _StoppedBankRoll(INITIAL_FUNDS, scenario.max_transaction_loss)
    strategy = STRATEGY_FACTORIES[strategy_name](scenario)
    clock = _Clock()
    counters = {"bets": 0, "staked": 0.0, "first": 0.0}

    original_evaluate = strategy.evaluate

    def evaluate(_probability, current_bankroll):
        clock.trial += 1
        proportion = original_evaluate(beliefs[clock.trial], current_bankroll)
        if clock.trial == 0:
            counters["first"] = proportion
        if proportion > 0:
            counters["bets"] += 1
            counters["staked"] += (
                round(current_bankroll * bankroll.percent_bettable, 2) * proportion
            )
        return proportion

    strategy.evaluate = evaluate

    simulator = RepeatedBinarySimulator(
        payoff=scenario.payoff,
        loss=scenario.loss,
        fee_per_bet=scenario.cost,
        probability=scenario.probability,
        trials=TRIALS,
    )

    saved_random = repeated_binary.random
    repeated_binary.random = _ReplayedOutcomes(draws, clock)
    try:
        simulator.evaluate_strategy(strategy, bankroll)
    finally:
        repeated_binary.random = saved_random

    terminal = bankroll.total_funds
    trials_started = clock.trial + 1
    stop_reason = bankroll.stop_reason
    if not stop_reason and trials_started < TRIALS:
        stop_reason = "bankruptcy"
    max_drawdown = summarize_history(bankroll.history).max_drawdown
    return PathResult(
        terminal=terminal,
        trials_started=trials_started,
        bets_placed=counters["bets"],
        max_drawdown=0.0 if max_drawdown is None else max_drawdown,
        growth_rate=math.log(max(terminal, WEALTH_FLOOR) / INITIAL_FUNDS) / TRIALS,
        stop_reason=stop_reason,
        fees_paid=counters["bets"] * scenario.cost,
        staked=counters["staked"],
        first_bet_fraction=counters["first"],
    )


def summarise(scenario, strategy_name, results):
    """Reduce the per-path results for one cell of the matrix to one row."""
    terminals = pd.Series([r.terminal for r in results])
    drawdowns = pd.Series([r.max_drawdown for r in results])
    growth = pd.Series([r.growth_rate for r in results])
    early = [r for r in results if r.trials_started < TRIALS]
    staked = sum(r.staked for r in results)
    fees = sum(r.fees_paid for r in results)
    return {
        "scenario": scenario.key,
        "axis": scenario.axis,
        "scenario_label": scenario.label,
        "probability": scenario.probability,
        "cost_input": scenario.cost,
        "estimate_stdev": scenario.estimate_stdev,
        "estimate_bias": scenario.estimate_bias,
        "max_transaction_loss": "none"
        if scenario.max_transaction_loss is None
        else scenario.max_transaction_loss,
        "strategy": strategy_name,
        "paths": len(results),
        "median_first_bet_fraction": round(
            pd.Series([r.first_bet_fraction for r in results]).median(), 6
        ),
        "median_terminal": round(terminals.median(), 2),
        "mean_terminal": round(terminals.mean(), 2),
        "p5_terminal": round(terminals.quantile(0.05), 2),
        "p25_terminal": round(terminals.quantile(0.25), 2),
        "p75_terminal": round(terminals.quantile(0.75), 2),
        "p95_terminal": round(terminals.quantile(0.95), 2),
        "median_max_drawdown": round(drawdowns.median(), 4),
        "p95_max_drawdown": round(drawdowns.quantile(0.95), 4),
        "median_growth_rate_per_trial": round(growth.median(), 6),
        "early_stop_rate": round(len(early) / len(results), 4),
        "early_stop_drawdown_rate": round(
            sum(r.stop_reason == "drawdown-limit" for r in results) / len(results), 4
        ),
        "early_stop_bankruptcy_rate": round(
            sum(r.stop_reason == "bankruptcy" for r in results) / len(results), 4
        ),
        "median_trials_started": int(
            pd.Series([r.trials_started for r in results]).median()
        ),
        "median_bets_placed": int(pd.Series([r.bets_placed for r in results]).median()),
        "realised_fee_pct_of_stake": round(100 * fees / staked, 6) if staked else 0.0,
    }


def run_matrix():
    """Run the whole matrix and return one row per (scenario, strategy)."""
    rows = []
    for scenario in SCENARIOS:
        for strategy_name in STRATEGY_FACTORIES:
            results = [run_path(scenario, strategy_name, i) for i in range(PATHS)]
            rows.append(summarise(scenario, strategy_name, results))
        print(f"  {scenario.key}: {len(STRATEGY_FACTORIES)} strategies")
    return pd.DataFrame(rows)


# --- charts -----------------------------------------------------------------
# Every chart is drawn in greyscale with distinct markers, hatches or line styles
# so it stays readable when printed without colour.

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h"]
HATCHES = ["", "///", "...", "xxx", "\\\\\\", "+++", "ooo", "**", "||"]
GREYS = [str(round(0.15 + 0.08 * i, 2)) for i in range(9)]


def _base_frame(frame):
    return frame[frame["scenario"] == "base"].sort_values("median_terminal")


def chart_terminal_bands(frame, path):
    data = _base_frame(frame)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = range(len(data))
    for i, (_, row) in enumerate(data.iterrows()):
        ax.plot(
            [row["p5_terminal"], row["p95_terminal"]],
            [i, i],
            color="0.55",
            linewidth=1.2,
            zorder=1,
        )
        ax.plot(
            [row["p25_terminal"], row["p75_terminal"]],
            [i, i],
            color="0.15",
            linewidth=4.5,
            solid_capstyle="butt",
            zorder=2,
        )
        ax.plot(
            row["median_terminal"],
            i,
            marker=MARKERS[i % len(MARKERS)],
            color="white",
            markeredgecolor="black",
            markersize=9,
            zorder=3,
        )
    ax.axvline(INITIAL_FUNDS, color="black", linestyle=":", linewidth=1)
    ax.set_yticks(list(y))
    ax.set_yticklabels(data["strategy"])
    ax.set_xscale("log")
    ax.set_xlabel(f"Terminal bankroll after {TRIALS} bets (log scale, $)")
    ax.set_title(
        "Terminal bankroll: median, quartiles and 5th-95th percentile band\n"
        f"{BASE.label}\n{PATHS} paths, {TRIALS} bets, seed {SEED}",
        fontsize=11,
    )
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def chart_growth_vs_drawdown(frame, path):
    data = _base_frame(frame)
    # Strategies that coincide exactly share one point; joining their names is the
    # only honest way to label it, and it is itself a result worth showing.
    points = {}
    for _, row in data.iterrows():
        key = (
            round(row["median_max_drawdown"] * 100, 4),
            round(row["median_growth_rate_per_trial"] * 100, 4),
        )
        points.setdefault(key, []).append(row["strategy"])
    fig, ax = plt.subplots(figsize=(10, 6))
    ordered = sorted(points.items())
    span = max(y for _, y in points) - min(y for _, y in points)
    gap = span * 0.07
    label_y = None
    for i, ((x, y), names) in enumerate(ordered):
        ax.scatter(
            x,
            y,
            marker=MARKERS[i % len(MARKERS)],
            s=120,
            facecolor="white",
            edgecolor="black",
            zorder=3,
        )
        # Nudge labels apart when points nearly coincide, with a leader line back
        # to the marker so the pairing stays unambiguous in print.
        label_y = y if label_y is None else max(y, label_y + gap)
        ax.annotate(
            " = ".join(names),
            xy=(x, y),
            xytext=(x + 2.5, label_y),
            fontsize=9,
            va="center",
            arrowprops={"arrowstyle": "-", "linewidth": 0.6, "color": "0.4"},
        )
    ax.axhline(0, color="black", linewidth=1, linestyle=":")
    ax.set_xlabel("Median maximum drawdown (%)")
    ax.set_ylabel("Median growth rate per bet (%)")
    ax.set_xlim(0, 105)
    ax.set_title(
        "Growth against drawdown; joined names are strategies that returned "
        "identical results\n"
        f"{BASE.label}\n{PATHS} paths, {TRIALS} bets, seed {SEED}",
        fontsize=11,
    )
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def chart_early_stops(frame, path):
    order = ["drawdown-03", "drawdown-08", "base", "drawdown-off"]
    names = {
        "drawdown-03": "0.03",
        "drawdown-08": "0.08",
        "base": "0.30 (default)",
        "drawdown-off": "None",
    }
    axis = frame[frame["scenario"].isin(order)]
    strategies = list(STRATEGY_FACTORIES)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    width = 0.2
    for j, scenario_key in enumerate(order):
        subset = axis[axis["scenario"] == scenario_key].set_index("strategy")
        values = [subset.loc[name, "early_stop_rate"] * 100 for name in strategies]
        ax.bar(
            [i + (j - 1.5) * width for i in range(len(strategies))],
            values,
            width=width,
            facecolor=GREYS[j * 2],
            edgecolor="black",
            hatch=HATCHES[j + 1],
            label=names[scenario_key],
        )
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=30, ha="right")
    ax.set_ylabel(f"Runs stopped before {TRIALS} bets (%)")
    ax.set_ylim(0, 118)
    ax.set_title(
        "Early stops by per-settlement loss cap\n"
        f"55% edge, even money, no cost, no estimate error, "
        f"{PATHS} paths, {TRIALS} bets, seed {SEED}",
        fontsize=11,
    )
    ax.legend(title="max_transaction_loss", ncol=4, loc="upper center", framealpha=1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def build_allocation_market(observations=500, assets=6):
    """A deterministic six-asset market with one common risk factor.

    Returns share a factor with asset-specific loadings plus idiosyncratic
    noise, so the covariance the risk-based allocators read is structured
    rather than diagonal, and every draw comes from one seeded generator.
    """
    rng = np.random.default_rng(SEED)
    loadings = np.linspace(0.6, 1.4, assets)
    means = np.linspace(0.0002, 0.0006, assets)
    factor = rng.normal(0.0004, 0.008, size=observations)
    idiosyncratic = rng.normal(0.0, 0.01, size=(observations, assets))
    # (T, 1) * (1, N): each asset loads the common factor by its own beta.
    returns = means[None, :] + loadings[None, :] * factor[:, None] + idiosyncratic
    return scenario_model(returns)


def run_allocation_comparison(model, trials=1_000):
    """Race the allocation families through ``AllocationSimulator``.

    Common random numbers: every allocator sees its own simulator seeded
    with the same seed, so trial *t* draws the same scenario row for all
    of them and the growth comparison is matched. scipy-gated allocators
    are skipped with a note when the optional extra is absent.
    """
    mean, covariance = model.moments()
    option_count = covariance.shape[0]
    allocators = {
        "Equal weight": FixedWeights([1.0 / option_count] * option_count),
        "RiskBudgeting (ERC)": RiskBudgeting(covariance),
        "HierarchicalRiskParity": HierarchicalRiskParity(covariance),
        "ExponentialGradient": ExponentialGradient(option_count, learning_rate=0.05),
    }
    try:
        allocators["Global min variance"] = GlobalMinimumVariance(covariance)
        allocators["MeanVariance (lambda=1)"] = MeanVariance(
            mean, covariance, risk_aversion=1.0
        )
    except ImportError as error:
        print(f"skipping the scipy-gated allocators: {error}")
    histories = {}
    for name, allocation in allocators.items():
        bankroll = BankRoll(initial_funds=1_000.0, max_transaction_loss=None)
        simulator = AllocationSimulator(model, trials=trials, seed=SEED)
        simulator.evaluate_strategy(allocation, bankroll)
        histories[name] = [float(value) for value in bankroll.history]
    return histories


def allocation_metrics(histories):
    """Growth, volatility, and drawdown per allocator, as a frame."""
    records = []
    for name, history in histories.items():
        values = np.asarray(history, dtype=float)
        previous = values[:-1]
        returns = np.divide(
            values[1:] - previous,
            previous,
            out=np.zeros_like(values[1:]),
            where=previous > 0,
        )
        records.append(
            {
                "strategy": name,
                "final": values[-1],
                "growth": values[-1] / values[0],
                "vol/period": (float(returns.std(ddof=1)) if returns.size > 1 else 0.0),
                "max dd": float(np.max(1.0 - values / np.maximum.accumulate(values))),
            }
        )
    return pd.DataFrame(records).set_index("strategy")


def chart_allocation_growth(histories, path, trials):
    """Growth-path comparison, via the allocation plots helper."""
    axes = bankroll_paths(histories)
    axes.set_title(
        "Allocation strategies through AllocationSimulator\n"
        f"six-asset factor market, {trials} periods, seed {SEED}",
        fontsize=11,
    )
    axes.figure.savefig(path, dpi=200)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"keeks strategy benchmark: {len(SCENARIOS)} scenarios x "
        f"{len(STRATEGY_FACTORIES)} strategies x {PATHS} paths x {TRIALS} bets"
    )
    frame = run_matrix()
    csv_path = OUTPUT_DIR / "strategy_benchmark.csv"
    frame.to_csv(csv_path, index=False)
    chart_terminal_bands(frame, OUTPUT_DIR / "terminal_bankroll_bands.png")
    chart_growth_vs_drawdown(frame, OUTPUT_DIR / "growth_vs_drawdown.png")
    chart_early_stops(frame, OUTPUT_DIR / "early_stops_by_drawdown_limit.png")
    print(f"wrote {csv_path} and 3 charts to {OUTPUT_DIR}")

    # Allocation-layer comparison: same benchmark discipline (fresh state,
    # common random numbers, one seeded generator) applied to the
    # portfolio allocators, with the growth chart drawn by the plots
    # helpers the layer ships.
    allocation_model = build_allocation_market()
    allocation_histories = run_allocation_comparison(allocation_model)
    allocation_frame = allocation_metrics(allocation_histories)
    print(
        "\n=== Allocation comparison (AllocationSimulator, common random numbers) ==="
    )
    print(
        allocation_frame.to_string(
            float_format=lambda value: f"{value:,.4f}",
        )
    )
    chart_allocation_growth(
        allocation_histories,
        OUTPUT_DIR / "allocation_growth_comparison.png",
        trials=1_000,
    )
    print(
        f"wrote {OUTPUT_DIR / 'allocation_growth_comparison.png'} "
        "(allocation comparison chart)"
    )


if __name__ == "__main__":
    main()
