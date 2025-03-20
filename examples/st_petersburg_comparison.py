"""
St. Petersburg Paradox Simulation using Binary Strategies

This example demonstrates a simplified version of the St. Petersburg paradox
using the keeks library. The St. Petersburg paradox involves a game with potentially
infinite expected value, but finite practical outcomes.

Instead of implementing the classical version (which would require a custom simulator),
we use a simplified binary model with favorable odds to compare different bankroll
management strategies.
"""

import matplotlib

# Use non-interactive backend to avoid tkinter issues
matplotlib.use("Agg")
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from keeks.bankroll import BankRoll
from keeks.binary_strategies.kelly import (
    DrawdownAdjustedKelly,
    FractionalKellyCriterion,
    KellyCriterion,
)
from keeks.binary_strategies.simple import (
    CPPIStrategy,
    DynamicBankrollManagement,
    FixedFractionStrategy,
    NaiveStrategy,
    OptimalF,
)
from keeks.simulators.repeated_binary import RepeatedBinarySimulator
from keeks.utils import RuinError

# Define St. Petersburg-like parameters
# Using favorable odds to simulate the high expected value of St. Petersburg
INITIAL_BANKROLL = 1000.0
PAYOFF = 1.1  # Win 1.1x your bet (reduced from 2.0 for more realism)
LOSS = 1.0  # Lose your entire bet
TRANS_COST = 0.01  # Small transaction cost
PROBABILITY = 0.55  # 55% chance of winning (slightly favorable)
NUM_TRIALS = 100  # Reduced number of betting rounds
NUM_SIMULATIONS = 30  # Run multiple simulations to see distribution


def run_strategy_simulation(strategy_class, strategy_name, strategy_params=None):
    """Run multiple simulations for a given strategy and return results."""
    results = []
    ruin_count = 0
    max_bankroll = 0

    for _i in range(NUM_SIMULATIONS):
        try:
            # Initialize the bankroll and strategy
            bankroll = BankRoll(INITIAL_BANKROLL)

            # Create and initialize the strategy with required parameters
            if strategy_params:
                strategy = strategy_class(**strategy_params)
            else:
                strategy = strategy_class()

            # Use standard simulator for all strategies
            simulator = RepeatedBinarySimulator(
                payoff=PAYOFF,
                loss=LOSS,
                transaction_costs=TRANS_COST,
                probability=PROBABILITY,
                trials=NUM_TRIALS,
            )
            simulator.evaluate_strategy(strategy, bankroll)

            # Record the final bankroll
            final_bankroll = bankroll.total_funds

            # Cap the final bankroll at a reasonable maximum to avoid overflow
            final_bankroll = min(final_bankroll, 1e9)
            max_bankroll = max(max_bankroll, final_bankroll)

            results.append(final_bankroll)
        except RuinError:
            # If ruin occurs, record a final bankroll of 0
            results.append(0.0)
            ruin_count += 1
        except Exception as e:
            # Handle other exceptions
            print(f"  Error with {strategy_name}: {str(e)}")
            results.append(0.0)
            ruin_count += 1

    # If we had ruin cases, report them
    if ruin_count > 0:
        print(
            f"  Warning: {ruin_count} out of {NUM_SIMULATIONS} simulations ended in ruin"
        )

    return {
        "name": strategy_name,
        "mean": np.mean(results),
        "median": np.median(results),
        "min": np.min(results),
        "max": np.max(results),
        "std": np.std(results),
        "results": results,
        "ruin_rate": ruin_count / NUM_SIMULATIONS,
    }


def main():
    """Run the simulation for all strategies and plot results."""
    # Define all strategies with parameters - using more conservative parameters
    strategies = [
        {
            "class": KellyCriterion,
            "name": "Kelly Criterion",
            "params": {"payoff": PAYOFF, "loss": LOSS, "transaction_cost": TRANS_COST},
        },
        {
            "class": FractionalKellyCriterion,
            "name": "Half Kelly",
            "params": {
                "payoff": PAYOFF,
                "loss": LOSS,
                "transaction_cost": TRANS_COST,
                "fraction": 0.5,
            },
        },
        {
            "class": FractionalKellyCriterion,
            "name": "Quarter Kelly",
            "params": {
                "payoff": PAYOFF,
                "loss": LOSS,
                "transaction_cost": TRANS_COST,
                "fraction": 0.25,
            },
        },
        {
            "class": DrawdownAdjustedKelly,
            "name": "Drawdown Kelly",
            "params": {
                "payoff": PAYOFF,
                "loss": LOSS,
                "transaction_cost": TRANS_COST,
                "max_acceptable_drawdown": 0.2,
            },
        },
        {
            "class": OptimalF,
            "name": "Optimal F",
            "params": {
                "payoff": PAYOFF,
                "loss": LOSS,
                "transaction_cost": TRANS_COST,
                "win_rate": PROBABILITY,
                "max_risk_fraction": 0.15,
            },
        },
        {
            "class": FixedFractionStrategy,
            "name": "Fixed 5%",
            "params": {
                "fraction": 0.05,
                "payoff": PAYOFF,
                "loss": LOSS,
                "transaction_cost": TRANS_COST,
                "min_probability": 0.0,  # Always bet regardless of probability
            },
        },
        {
            "class": FixedFractionStrategy,
            "name": "Fixed 10%",
            "params": {
                "fraction": 0.1,
                "payoff": PAYOFF,
                "loss": LOSS,
                "transaction_cost": TRANS_COST,
                "min_probability": 0.0,  # Always bet regardless of probability
            },
        },
        {
            "class": NaiveStrategy,
            "name": "Naive Strategy",
            "params": {"payoff": PAYOFF, "loss": LOSS, "transaction_cost": TRANS_COST},
        },
        {
            "class": CPPIStrategy,
            "name": "CPPI",
            "params": {
                "floor_fraction": 0.9,  # Changed from 0.8 to 0.9 based on testing
                "multiplier": 0.5,  # Changed from 2 to 0.5 based on testing
                "initial_bankroll": INITIAL_BANKROLL,
                "payoff": PAYOFF,
                "loss": LOSS,
                "transaction_cost": TRANS_COST,
                "min_probability": 0.0,  # Always bet regardless of probability
            },
        },
        {
            "class": DynamicBankrollManagement,
            "name": "Dynamic",
            "params": {
                "base_fraction": 0.1,
                "payoff": PAYOFF,
                "loss": LOSS,
                "transaction_cost": TRANS_COST,
                "window_size": 10,
                "max_fraction": 0.15,
                "min_fraction": 0.01,
            },
        },
    ]

    # Run simulations for each strategy
    print(f"Running {NUM_SIMULATIONS} simulations for each strategy...")
    results = []

    for strategy in strategies:
        print(f"Simulating {strategy['name']}...")
        params = strategy.get("params", None)
        result = run_strategy_simulation(strategy["class"], strategy["name"], params)
        results.append(result)
        print(f"  Mean final bankroll: ${result['mean']:.2f}")

    # Sort results by mean performance
    results.sort(key=lambda x: x["mean"], reverse=True)

    # Create results table
    results_table = pd.DataFrame(
        [
            {
                "Strategy": r["name"],
                "Mean": f"${r['mean']:.2f}",
                "Median": f"${r['median']:.2f}",
                "Min": f"${r['min']:.2f}",
                "Max": f"${r['max']:.2f}",
                "Std Dev": f"${r['std']:.2f}",
                "Ruin Rate": f"{r['ruin_rate'] * 100:.1f}%",
            }
            for r in results
        ]
    )

    print("\nStrategy Comparison Results:")
    print(results_table.to_string(index=False))

    # Plot the distribution of final bankrolls for each strategy
    plt.figure(figsize=(12, 8))

    # Box plot of final bankrolls
    plt.subplot(2, 1, 1)
    boxplot_data = [r["results"] for r in results]
    boxplot_labels = [r["name"] for r in results]
    plt.boxplot(boxplot_data, vert=True, labels=boxplot_labels)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Final Bankroll ($)")
    plt.title("Distribution of Final Bankrolls Across Strategies")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Bar chart of mean bankrolls with standard deviation
    plt.subplot(2, 1, 2)
    means = [r["mean"] for r in results]
    stds = [r["std"] for r in results]
    names = [r["name"] for r in results]

    plt.bar(names, means, yerr=stds, capsize=5, alpha=0.7)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean Final Bankroll ($)")
    plt.title("Mean Final Bankroll with Standard Deviation")
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()

    # Create the output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    # Save the figure and table
    plt.savefig(
        os.path.join(output_dir, "st_petersburg_strategies_comparison.png"), dpi=300
    )
    results_table.to_csv(
        os.path.join(output_dir, "st_petersburg_strategies_comparison.csv"), index=False
    )

    print(f"\nResults saved to {output_dir}")
    print("Plot saved to file (using non-interactive backend)")


if __name__ == "__main__":
    main()
