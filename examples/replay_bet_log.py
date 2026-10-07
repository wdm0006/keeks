"""Replay a synthetic, recorded binary bet log; educational only, not advice."""

import argparse
import csv
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
    summarize_history,
)

DATA_PATH = Path(__file__).resolve().parent / "data" / "synthetic_bet_log.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
INITIAL_BANKROLL = 1000.0


def load_bet_log(csv_path=DATA_PATH):
    """Load chronological probability estimates and integer 0/1 outcomes."""
    with Path(csv_path).open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    return (
        [float(row["probability"]) for row in rows],
        [int(row["outcome"]) for row in rows],
    )


def replay_bet_log(csv_path=DATA_PATH):
    """Compare fresh strategies on the same recorded, even-money bets."""
    probabilities, outcomes = load_bet_log(csv_path)
    simulator = HistoricalBinarySimulator(
        payoff=1.0,
        loss=1.0,
        fee_per_bet=0.0,
        probabilities=probabilities,
        outcomes=outcomes,
    )
    odds = {"payoff": 1.0, "loss": 1.0, "transaction_cost_rate": 0.0}
    strategies = {
        "Kelly": KellyCriterion(**odds),
        "Half Kelly": FractionalKellyCriterion(**odds, fraction=0.5),
        "Fixed 5%": FixedFractionStrategy(**odds, fraction=0.05),
    }
    results = []
    histories = {}
    for name, strategy in strategies.items():
        bankroll = BankRoll(INITIAL_BANKROLL)
        simulator.evaluate_strategy(strategy, bankroll)
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
    """Write the summary CSV and one headless bankroll-path chart."""
    results, histories = replay_bet_log(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
