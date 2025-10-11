"""
True St. Petersburg Paradox: What Would You Pay to Play?

The St. Petersburg paradox is a probability theory paradox:
- Flip a fair coin repeatedly until it comes up tails
- Payout is $2^n where n is the number of flips
- Expected value is infinite: E[X] = Σ(1/2^n × 2^n) = Σ(1) = ∞

The paradox: Despite infinite expected value, no rational person would pay much to play.

This example calculates the maximum entry price different betting strategies would pay
to play one round of the St. Petersburg game at various wealth levels.
"""

import os

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from keeks.binary_strategies import KellyCriterion
from keeks.binary_strategies.kelly import DrawdownAdjustedKelly, FractionalKellyCriterion
from keeks.binary_strategies.simple import (
    CPPIStrategy,
    DynamicBankrollManagement,
    FixedFractionStrategy,
    MertonShare,
    NaiveStrategy,
    OptimalF,
)


def get_st_petersburg_outcomes(max_flips=30):
    """
    Generate outcomes and probabilities for St. Petersburg game.

    Parameters
    ----------
    max_flips : int
        Maximum coin flips to consider (practical cap for infinite series)
        Note: Limited to ~1000 due to float64 overflow at 2^1024

    Returns
    -------
    tuple
        (outcomes, probabilities) arrays
    """
    import numpy as np

    outcomes = []
    probabilities = []

    for n in range(1, max_flips + 1):
        prob = (0.5) ** n  # Probability of exactly n flips
        payout = 2.0 ** n    # Payout is 2^n (use 2.0 to keep as float)
        outcomes.append(payout)
        probabilities.append(prob)

    return np.array(outcomes, dtype=float), np.array(probabilities, dtype=float)


def main():
    """Calculate maximum payment for St. Petersburg game across different scenarios."""

    print("=" * 80)
    print("ST. PETERSBURG PARADOX: WHAT WOULD YOU PAY TO PLAY?")
    print("=" * 80)
    print()
    print("The Game:")
    print("  - Flip a fair coin repeatedly until tails")
    print("  - Payout is $2^n where n is the number of flips")
    print("  - Expected value is INFINITE")
    print()
    print("The Question:")
    print("  Given your wealth, what's the maximum you'd rationally pay to play once?")
    print()
    print("=" * 80)
    print()

    # Different bankroll levels to test
    bankrolls = [1, 10, 100, 1_000]

    # Create strategy instances - all 9 strategies!
    strategies = {
        # Utility-based strategies
        "Kelly": KellyCriterion(
            payoff=1.0, loss=1.0, transaction_cost=0.0
        ),
        "Half Kelly": FractionalKellyCriterion(
            payoff=1.0, loss=1.0, transaction_cost=0.0, fraction=0.5
        ),
        "Drawdown Kelly": DrawdownAdjustedKelly(
            payoff=1.0, loss=1.0, transaction_cost=0.0, max_acceptable_drawdown=0.2
        ),
        "Optimal F": OptimalF(
            payoff=1.0, loss=1.0, transaction_cost=0.0, win_rate=0.55
        ),
        "Merton (γ=2.0)": MertonShare(
            payoff=1.0, loss=1.0, transaction_cost=0.0, risk_aversion=2.0
        ),
        "Merton (γ=5.0)": MertonShare(
            payoff=1.0, loss=1.0, transaction_cost=0.0, risk_aversion=5.0
        ),
        # Rule-based strategies
        "Naive": NaiveStrategy(
            payoff=1.0, loss=1.0, transaction_cost=0.0
        ),
        "Fixed 5%": FixedFractionStrategy(
            payoff=1.0, loss=1.0, transaction_cost=0.0, fraction=0.05
        ),
        "Dynamic 10%": DynamicBankrollManagement(
            payoff=1.0, loss=1.0, transaction_cost=0.0, base_fraction=0.1, window_size=10
        ),
    }

    # Get St. Petersburg outcomes
    # Using 1,000 flips gives EV = $1,000, approaching the infinite EV of the true paradox
    # Each flip contributes exactly $1 to expected value: (1/2^n) × 2^n = 1
    # True paradox has infinite flips → infinite EV, but we cap at 1000 due to float64 limits
    outcomes, probabilities = get_st_petersburg_outcomes(max_flips=1_000)

    print("Calculating optimal entry prices for each strategy...")
    print()

    # Calculate results
    all_results = []

    for bankroll in bankrolls:
        row = {"Bankroll": f"${bankroll:,}"}

        for strategy_name, strategy in strategies.items():
            # Use the strategy's calculate_max_entry_price method
            max_price = strategy.calculate_max_entry_price(
                outcomes=outcomes,
                probabilities=probabilities,
                current_wealth=bankroll
            )
            row[f"{strategy_name}"] = max_price
            row[f"{strategy_name} (%BR)"] = (max_price / bankroll) * 100

        all_results.append(row)

    # Create DataFrame
    df_results = pd.DataFrame(all_results)

    # Display price table
    print("\nMAXIMUM ENTRY PRICE BY STRATEGY AND WEALTH LEVEL:")
    print("=" * 80)
    print()

    # Create price-only table for display
    price_cols = ["Bankroll"] + list(strategies.keys())
    df_prices = df_results[price_cols].copy()

    # Format prices as currency
    for col in strategies.keys():
        df_prices[col] = df_results[col].apply(lambda x: f"${x:.2f}")

    print(df_prices.to_string(index=False))

    # Create visualization
    print("Creating visualization...")

    # Single chart: Entry price by strategy by bankroll
    fig, ax = plt.subplots(figsize=(16, 10))

    # Prepare data for plotting
    bankroll_labels = [f"${b:,}" for b in bankrolls]
    x = np.arange(len(bankroll_labels))
    n_strategies = len(strategies)
    width = 0.08  # Width of bars

    # Define colors for different strategy types
    colors = {
        'Kelly': '#1f77b4',
        'Half Kelly': '#aec7e8',
        'Drawdown Kelly': '#7fbfe8',
        'Optimal F': '#4a90c4',
        'Merton (γ=2.0)': '#ff7f0e',
        'Merton (γ=5.0)': '#ffbb78',
        'Naive': '#2ca02c',
        'Fixed 5%': '#d62728',
        'Dynamic 10%': '#9467bd',
    }

    # Plot bars for each strategy
    for i, strategy_name in enumerate(strategies.keys()):
        prices = [all_results[j][strategy_name] for j in range(len(bankrolls))]
        offset = (i - n_strategies / 2 + 0.5) * width
        ax.bar(x + offset, prices, width, label=strategy_name,
               color=colors.get(strategy_name, f'C{i}'))

    ax.set_xlabel('Bankroll Level', fontsize=12, fontweight='bold')
    ax.set_ylabel('Maximum Entry Price ($, log scale)', fontsize=12, fontweight='bold')
    ax.set_title('St. Petersburg Paradox: Maximum Entry Price by Strategy\n' +
                 '(Despite Infinite Expected Value, Agents Pay Finite Amounts)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(bankroll_labels, rotation=0)
    ax.set_yscale('log')
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    ax.grid(axis='y', alpha=0.3, linestyle='--', which='both')

    # Add annotation
    ax.text(0.98, 0.02,
            'Utility-based strategies (Kelly, Merton, Optimal F) analyze risk\n' +
            'Rule-based strategies (Fixed, Dynamic) apply mechanical rules\n' +
            'Naive pays expected value ($1,000 for 1k flips), capped at wealth',
            transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    # Save figure
    fig_path = os.path.join(output_dir, "st_petersburg_paradox.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Chart saved to: {fig_path}")

    # Save CSV with better formatting
    csv_data = []
    for result in all_results:
        row = {"Bankroll": result["Bankroll"]}
        for strategy_name in strategies.keys():
            row[f"{strategy_name} ($)"] = f"${result[strategy_name]:.2f}"
            row[f"{strategy_name} (%)"] = f"{result[f'{strategy_name} (%BR)']:.3f}%"
        csv_data.append(row)

    df_csv = pd.DataFrame(csv_data)
    csv_path = os.path.join(output_dir, "st_petersburg_paradox.csv")
    df_csv.to_csv(csv_path, index=False)
    print(f"✓ Data saved to: {csv_path}")



if __name__ == "__main__":
    main()
