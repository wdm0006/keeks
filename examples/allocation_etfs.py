"""
Portfolio allocation over a real ETF book, offline by default.

Runs keeks' allocation layer over six exchange-traded funds spanning US large
and small equities, growth equities, long treasuries, gold, and real estate
(SPY, QQQ, IWM, TLT, GLD, VNQ). Two joint-return models are built from the
same historical daily simple returns — the empirical bootstrap
(``scenario_model``) and per-option Student-t marginals fitted by maximum
likelihood (``fit_marginals_model``, scipy-gated) — and five allocators
(MeanVariance, RiskBudgeting, HierarchicalRiskParity, MeanCVaR,
ExponentialGradient) are replayed against two benchmarks (equal weight and
global minimum variance) through the :class:`AllocationSimulator`. The
growth / volatility / drawdown comparison is printed per pass, and the
growth-path, drawdown, and weight-evolution charts are written into
``examples/output/`` with the visualization helpers.

The default run is fully offline: it loads the committed fixture
``examples/data/etf_returns.csv``, whose provenance header names the tickers,
the date window, and the refresh command. Pass ``--refresh`` to re-download
the window from Yahoo Finance via yfinance and rewrite the fixture. yfinance
is a development-only dependency — keeks never imports it at runtime; the
refresh path imports it lazily with a pointed error when it is missing.

Usage::

    uv run python examples/allocation_etfs.py            # offline, from the fixture
    uv run python examples/allocation_etfs.py --refresh  # re-download, then run

Outputs (committed; regenerate via ``make examples``):

    examples/output/allocation_etf_growth_paths.png
    examples/output/allocation_etf_growth_paths_student_t.png
    examples/output/allocation_etf_drawdown.png
    examples/output/allocation_etf_weight_evolution.png

Notes on honesty of the numbers:

* The two passes share one seed, and each simulator draws its realizations in
  one call from its model, so every allocator inside a pass meets the same
  realizations (common random numbers) — the comparison is the allocators',
  not the draw luck's.
* ``transaction_costs`` stays at zero: the simulator's fee is a flat absolute
  amount per staked period, and a nonzero flat fee on a 1000-period daily
  replay would measure the fee, not the allocators. The cost-unit split is
  documented in the README.
* The MeanCVaR allocator is bound to the empirical scenario rows in both
  passes — the tail information the book actually observed. Under the
  Student-t pass the simulator draws from the fitted marginals, which carry
  no scenario rows, so the descriptor-equality gate passes untouched.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from keeks import (  # noqa: E402
    AllocationSimulator,
    BankRoll,
    BaseAllocationStrategy,
    ExponentialGradient,
    FixedWeights,
    GlobalMinimumVariance,
    HierarchicalRiskParity,
    MeanCVaR,
    MeanVariance,
    RiskBudgeting,
    bankroll_paths,
    drawdown,
    fit_marginals_model,
    scenario_model,
    weight_evolution,
)

TICKERS = ("SPY", "QQQ", "IWM", "TLT", "GLD", "VNQ")
START = "2022-10-03"
END = "2025-10-01"
FIXTURE_PATH = Path(__file__).resolve().parent / "data" / "etf_returns.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

INITIAL_FUNDS = 1000.0
TRIALS = 1000
SEED = 20260803
TRANSACTION_COSTS = 0.0

FIXTURE_HEADER = """\
# ETF daily simple returns for examples/allocation_etfs.py
#   tickers: SPY, QQQ, IWM, TLT, GLD, VNQ (US large-cap, growth, small-cap,
#            long treasuries, gold, real estate)
#   prices:  Yahoo Finance adjusted close, converted to daily simple returns
#   window:  2022-10-03 .. 2025-10-01 (yfinance start/end)
#   refresh: uv run python examples/allocation_etfs.py --refresh
#   format:  one header row, then one Date column and one simple-return
#            column per ticker
"""


def load_returns(fixture_path=FIXTURE_PATH):
    """
    Load the committed returns fixture.

    Returns
    -------
    tuple
        ``(dates, returns)`` — the observation date strings and the
        ``(observations, options)`` matrix of daily simple returns.
    """
    lines = fixture_path.read_text(encoding="utf-8").splitlines()
    rows = [line for line in lines if not line.startswith("#")]
    reader = csv.reader(rows)
    header = next(reader)
    if header != ["Date", *TICKERS]:
        raise ValueError(
            f"{fixture_path} header {header!r} does not match "
            f"{['Date', *TICKERS]!r}; refresh the fixture with --refresh"
        )
    dates = []
    returns = []
    for row in reader:
        if not row:
            continue
        dates.append(row[0])
        returns.append([float(value) for value in row[1:]])
    return dates, np.asarray(returns, dtype=float)


def download_returns(start=START, end=END):
    """
    Download the ETF window from Yahoo Finance.

    Returns
    -------
    tuple
        ``(dates, returns)`` in the same shape :func:`load_returns` gives.

    Raises
    ------
    ImportError
        When yfinance is not installed — it is a development-only
        dependency, never imported at keeks runtime.
    """
    try:
        import yfinance
    except ImportError as exc:
        raise ImportError(
            "The --refresh path needs yfinance, a development-only dependency: "
            'install it with `uv pip install -e ".[dev]"` (or pip install '
            "yfinance). keeks itself never imports yfinance."
        ) from exc
    prices = yfinance.download(
        list(TICKERS), start=start, end=end, progress=False, auto_adjust=True
    )["Close"][list(TICKERS)].dropna()
    returns_frame = prices.pct_change().dropna()
    dates = [stamp.date().isoformat() for stamp in returns_frame.index]
    return dates, returns_frame.to_numpy(dtype=float)


def write_fixture(dates, returns, fixture_path=FIXTURE_PATH):
    """Write the fixture with its provenance header."""
    lines = FIXTURE_HEADER.splitlines()
    lines.append("Date," + ",".join(TICKERS))
    for date, row in zip(dates, returns, strict=True):
        lines.append(date + "," + ",".join(repr(float(value)) for value in row))
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    fixture_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class WeightRecording(BaseAllocationStrategy):
    """
    Wrap an allocator, recording the weight tuple it returns each trial.

    Pure observer: ``evaluate`` delegates and records, and the settlement
    hook delegates so the wrapped online allocator keeps adapting.
    """

    def __init__(self, inner):
        self.inner = inner
        self.recorded = []

    def evaluate(self, current_bankroll):
        weights = self.inner.evaluate(current_bankroll)
        self.recorded.append([float(weight) for weight in weights])
        return weights

    def record_settlement(self, realized_returns):
        self.inner.record_settlement(realized_returns)


def build_allocators(scenarios, mean, covariance):
    """
    Construct the seven-book comparison: five allocators, two benchmarks.

    Fresh construction per simulator pass, matching the layer's repricing
    doctrine. ``scenarios`` is the empirical rows matrix (MeanCVaR's
    descriptor, and the source of the option count); ``mean`` and
    ``covariance`` are the moment descriptors (MeanVariance, GMV,
    RiskBudgeting, HRP).
    """
    option_count = scenarios.shape[1]
    equal_weight = 1.0 / option_count
    return {
        "Equal weight": FixedWeights([equal_weight] * option_count),
        "Global min variance": GlobalMinimumVariance(covariance),
        "MeanVariance (lambda=1)": MeanVariance(mean, covariance, risk_aversion=1.0),
        "RiskBudgeting (ERC)": RiskBudgeting(covariance),
        "HierarchicalRiskParity": HierarchicalRiskParity(covariance),
        # Unit risk aversion is MeanCVaR's shipped v1 posture: the objective
        # trades expected return against the tail's expected loss
        # one-for-one, and the program is positively homogeneous in the
        # return scale, so on daily-frequency market data - expected return
        # far below the tail expectation - the honest optimum is full cash.
        # The row below demonstrates exactly that, not a failure to solve.
        "MeanCVaR (alpha=5%)": MeanCVaR(scenarios, tail_alpha=0.05),
        "ExponentialGradient": ExponentialGradient(option_count, learning_rate=0.05),
    }


def run_pass(model, allocators, trials=TRIALS, seed=SEED):
    """
    Replay every allocator through its own seeded simulator on one model.

    Each simulator reseeds the same private stream, and the model draws its
    realizations in one call, so every allocator meets the same realization
    sequence. Returns ``name -> bankroll history``.
    """
    histories = {}
    for name, allocation in allocators.items():
        bankroll = BankRoll(initial_funds=INITIAL_FUNDS, max_draw_down=None)
        simulator = AllocationSimulator(
            model, transaction_costs=TRANSACTION_COSTS, trials=trials, seed=seed
        )
        simulator.evaluate_strategy(allocation, bankroll)
        histories[name] = [float(value) for value in bankroll.history]
    return histories


def period_returns(history):
    """Per-period simple returns of a bankroll history, zero where stalled."""
    values = np.asarray(history, dtype=float)
    previous = values[:-1]
    current = values[1:]
    return np.divide(
        current - previous,
        previous,
        out=np.zeros_like(current),
        where=previous > 0,
    )


def growth(history):
    """Total growth multiple of a bankroll history."""
    values = np.asarray(history, dtype=float)
    return float(values[-1] / values[0])


def volatility(history):
    """Per-period standard deviation of a bankroll history's returns."""
    returns = period_returns(history)
    return float(returns.std(ddof=1)) if returns.size > 1 else 0.0


def max_drawdown(history):
    """Peak-to-trough fractional loss of a bankroll history."""
    values = np.asarray(history, dtype=float)
    peaks = np.maximum.accumulate(values)
    positive = peaks > 0
    if not positive.any():
        return 0.0
    drawdowns = np.zeros_like(values)
    drawdowns[positive] = (peaks[positive] - values[positive]) / peaks[positive]
    return float(drawdowns.max())


def comparison_table(histories):
    """One row per allocator: final funds, growth, per-period volatility, max drawdown."""
    rows = {
        name: {
            "final": f"${history[-1]:,.2f}",
            "growth": f"{growth(history):.3f}x",
            "vol/period": f"{volatility(history):.4f}",
            "max dd": f"{max_drawdown(history):.2%}",
        }
        for name, history in histories.items()
    }
    frame = pd.DataFrame.from_dict(rows, orient="index")
    return frame[["final", "growth", "vol/period", "max dd"]]


def save_growth_chart(histories, path, title):
    """Growth paths of every allocator in a pass, via ``bankroll_paths``."""
    axes = bankroll_paths(histories)
    axes.set_title(title, fontsize=11)
    axes.figure.savefig(path, dpi=200)


def save_drawdown_chart(history, path, title):
    """Drawdown curve of one allocator's replay, via ``drawdown``."""
    axes = drawdown(history)
    axes.set_title(title, fontsize=11)
    axes.figure.savefig(path, dpi=200)


def save_weight_chart(weight_matrix, path, title, labels=None):
    """Weight evolution, via ``weight_evolution``.

    ``labels`` names the bands on the legend; the default keeps the
    helper's generic "Option i" labels, and the ETF run passes its tickers.
    """
    axes = weight_evolution(weight_matrix)
    if labels is not None:
        for text, label in zip(axes.get_legend().get_texts(), labels, strict=True):
            text.set_text(label)
    axes.set_title(title, fontsize=11)
    axes.figure.savefig(path, dpi=200)


def main(argv=None):
    """Run both simulator passes, print the comparison, write the charts."""
    parser = argparse.ArgumentParser(
        description="Portfolio allocation over an ETF book, offline by default"
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "re-download the ETF window from Yahoo Finance via yfinance "
            "(a development-only dependency) and rewrite the fixture"
        ),
    )
    args = parser.parse_args(argv)

    if args.refresh:
        dates, returns = download_returns()
        write_fixture(dates, returns)
        print(f"refreshed fixture: {FIXTURE_PATH}")
    else:
        dates, returns = load_returns()
        print(
            f"loaded fixture: {FIXTURE_PATH} "
            f"({len(dates)} observations, {returns.shape[1]} tickers, offline)"
        )

    # One empirical joint-return model, two doors into it: the exact
    # probability-weighted moments for the moment-based methods, and the
    # validated rows for the scenario method and the replay itself.
    empirical_model = scenario_model(returns)
    mean, covariance = empirical_model.moments()
    scenarios = empirical_model.scenarios
    student_t_model = fit_marginals_model(returns, family="student_t")

    # Pass 1 — the empirical bootstrap, with the online allocator observed.
    allocators = build_allocators(scenarios, mean, covariance)
    recorder = WeightRecording(allocators["ExponentialGradient"])
    allocators["ExponentialGradient"] = recorder
    empirical_histories = run_pass(empirical_model, allocators)

    # Pass 2 — the same book under fitted fat-tailed marginals.
    fat_histories = run_pass(
        student_t_model, build_allocators(scenarios, mean, covariance)
    )

    print(
        f"\n=== Replay: empirical bootstrap (scenario_model), "
        f"{TRIALS} periods, seed {SEED} ==="
    )
    print(comparison_table(empirical_histories).to_string())
    print(
        f"\n=== Replay: Student-t marginals (fit_marginals_model), "
        f"{TRIALS} periods, seed {SEED} ==="
    )
    print(comparison_table(fat_histories).to_string())
    print(
        "\nNote: MeanCVaR's unit-risk-aversion objective (maximize w'mu - "
        "CVaR, one-for-one) holds full cash on daily-frequency market data:\n"
        "expected daily return sits far below the tail's expected loss, and "
        "the program is scale-homogeneous, so no unit choice changes it.\n"
        "The risk-aversion dial is a documented follow-up (spec decision D2)."
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_growth_chart(
        empirical_histories,
        OUTPUT_DIR / "allocation_etf_growth_paths.png",
        f"ETF allocation, empirical bootstrap ({TRIALS} periods, seed {SEED})",
    )
    save_growth_chart(
        fat_histories,
        OUTPUT_DIR / "allocation_etf_growth_paths_student_t.png",
        f"ETF allocation, Student-t marginals ({TRIALS} periods, seed {SEED})",
    )
    save_drawdown_chart(
        empirical_histories["MeanVariance (lambda=1)"],
        OUTPUT_DIR / "allocation_etf_drawdown.png",
        f"Drawdown, MeanVariance at lambda=1 ({TRIALS} periods, seed {SEED})",
    )
    save_weight_chart(
        recorder.recorded,
        OUTPUT_DIR / "allocation_etf_weight_evolution.png",
        "Weight evolution, ExponentialGradient (empirical bootstrap)",
        labels=TICKERS,
    )
    print(f"\nwrote 4 charts to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
