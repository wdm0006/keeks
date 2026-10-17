"""Replay a synthetic, recorded binary bet log; educational only, not advice."""

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from keeks import (
    BankRoll,
    FixedFractionStrategy,
    FractionalKellyCriterion,
    HistoricalBinarySimulator,
    KellyCriterion,
    bootstrap_comparison,
    calibration_report,
    compare_strategies,
    load_bet_log,
    summarize_history,
)

DATA_PATH = Path(__file__).resolve().parent / "data" / "synthetic_bet_log.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
INITIAL_BANKROLL = 1000.0
ECONOMICS = {
    "payoff": 1.0,
    "loss": 1.0,
    "fee_per_bet": 0.0,
    "initial_funds": INITIAL_BANKROLL,
    "percent_bettable": 1.0,
    "max_transaction_loss": None,
}
TRANSACTION_COST_RATE = 0.0
SORT_BY = "geometric_growth_per_period"
BOOTSTRAP = {"seed": 153, "n_resamples": 200, "block_size": 1, "confidence": 0.9}
N_BINS = 10


def strategy_factories():
    """Build fresh policies with the same odds and per-unit costs for every replay."""
    odds = {
        "payoff": ECONOMICS["payoff"],
        "loss": ECONOMICS["loss"],
        "transaction_cost_rate": TRANSACTION_COST_RATE,
    }
    return {
        "Kelly": lambda: KellyCriterion(**odds),
        "Half Kelly": lambda: FractionalKellyCriterion(**odds, fraction=0.5),
        "Fixed 5%": lambda: FixedFractionStrategy(**odds, fraction=0.05),
    }


def replay_bet_log(csv_path=DATA_PATH):
    """Compare fresh strategies on the same recorded, even-money bets."""
    probabilities, outcomes = load_bet_log(csv_path)
    return replay_rows(probabilities, outcomes, strategy_factories())


def replay_rows(probabilities, outcomes, factories):
    """Replay already loaded rows for the legacy CSV and settled-bet paths."""
    simulator = HistoricalBinarySimulator(
        payoff=ECONOMICS["payoff"],
        loss=ECONOMICS["loss"],
        fee_per_bet=ECONOMICS["fee_per_bet"],
        probabilities=probabilities,
        outcomes=outcomes,
    )
    results = []
    histories = {}
    for name, factory in factories.items():
        bankroll = BankRoll(
            ECONOMICS["initial_funds"],
            ECONOMICS["percent_bettable"],
            ECONOMICS["max_transaction_loss"],
        )
        simulator.evaluate_strategy(factory(), bankroll)
        results.append(
            {
                "strategy": name,
                "input_rows": len(probabilities),
                **asdict(summarize_history(bankroll.history)),
            }
        )
        histories[name] = bankroll.history
    return results, histories


def main(csv_path=DATA_PATH, output_dir=OUTPUT_DIR):
    """Write CSV, JSON diagnostics and one headless bankroll-path chart."""
    probabilities, outcomes = load_bet_log(csv_path)
    factories = strategy_factories()
    results, histories = replay_rows(probabilities, outcomes, factories)
    report = {
        "settings": {
            "input_rows": len(probabilities),
            **ECONOMICS,
            "transaction_cost_rate": TRANSACTION_COST_RATE,
            "sort_by": SORT_BY,
            "n_bins": N_BINS,
            **BOOTSTRAP,
        },
        "calibration": asdict(
            calibration_report(probabilities, outcomes, n_bins=N_BINS)
        ),
        "rankings": [
            asdict(row)
            for row in compare_strategies(
                factories, probabilities, outcomes, **ECONOMICS, sort_by=SORT_BY
            )
        ],
        "uncertainty": [
            asdict(row)
            for row in bootstrap_comparison(
                factories,
                probabilities,
                outcomes,
                **ECONOMICS,
                sort_by=SORT_BY,
                **BOOTSTRAP,
            )
        ],
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "replay_bet_log.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (output_dir / "replay_bet_log.csv").open(
        "w", newline="", encoding="utf-8"
    ) as destination:
        writer = csv.DictWriter(
            destination, fieldnames=list(results[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(results)

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, history in histories.items():
        ax.plot(history, label=name)
    ax.set(
        xlabel="Settled bets",
        ylabel="Bankroll ($)",
        title=f"Bet log replay: {Path(csv_path).stem} (initial bankroll $1,000)",
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "replay_bet_log.png", dpi=150)
    plt.close(fig)
    for row in results:
        print(
            f"{row['strategy']}: end ${row['end']:.2f}, "
            f"max drawdown {row['max_drawdown']:.2%}"
        )
    print(f"Educational only; not financial advice. Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", nargs="?", type=Path, default=DATA_PATH)
    main(csv_path=parser.parse_args().csv_path)
