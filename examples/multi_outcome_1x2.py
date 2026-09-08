"""
1X2 (Home-Draw-Away) Betting with the Multi-Outcome Kelly Criterion

This example bets the legs of one mutually exclusive football market - a 1X2
book - with the same fixed probabilities every match:

    home win:  probability 0.42 at decimal odds 3.2  (edge: 0.42 * 3.2 - 1 = +0.344)
    draw:      probability 0.27 at decimal odds 3.4  (edge: 0.27 * 3.4 - 1 = -0.082)
    away win:  probability 0.28 at decimal odds 2.4  (edge: 0.28 * 2.4 - 1 = -0.328)

The probabilities sum to 0.97, so 3% of the mass is a void or push round on
which no leg settles and every stake is refunded.

Three strategies are compared over seeded simulations:

    Multi-outcome Kelly - the log-growth optimal stake per leg. It backs the
    home leg most, hedges a little on the draw's rich odds, and declines the
    negative-edge away leg entirely.
    Flat 5% - stakes 5% of the bankroll on every leg. Two of the three legs
    are negative-edge, so most of the stake works against the bettor; the
    home edge still carries the portfolio, but the growth stays orders of
    magnitude behind Kelly's log-optimal split.
    Favorite-only 10% - stakes 10% on the most probable leg only. Positive
    expectation, but a cruder allocation than Kelly's.

The example also demonstrates the seeding and reproducibility contract: a
seeded simulator rerun replays the identical bankroll history.

Note: settlement and trial hooks (record_settlement / update_bankroll) are
attached to the example strategies to count void rounds and early stops.
"""

import matplotlib

# Use non-interactive backend to avoid tkinter issues
matplotlib.use("Agg")
import os

import matplotlib.pyplot as plt
import pandas as pd

from keeks.bankroll import BankRoll
from keeks.multi_outcome import (
    BaseMultiOutcomeStrategy,
    MultiOutcomeKellyCriterion,
    RepeatedMultiOutcomeSimulator,
)

# Market definition: one 1X2 book, fixed prices and probabilities
PAYOFFS = (3.2, 3.4, 2.4)  # decimal odds: home, draw, away
PROBABILITIES = (0.42, 0.27, 0.28)  # 3% residual mass = void/push
LOSS = 1.0  # losing legs forfeit the full stake
TRANSACTION_COSTS = 0.0  # simulator's flat per-settlement fee

INITIAL_BANKROLL = 1000.0
NUM_TRIALS = 250  # matches per simulation
NUM_SIMULATIONS = 30  # seeded runs per strategy
REPLAY_TRIALS = 300  # trials for the seeding-replay demonstration
BASE_SEED = 42  # every simulation gets seed BASE_SEED + simulation index


class SettlementCounter:
    """Mixin that counts settled batches, void rounds, and trials played.

    The simulator looks the hooks up on the strategy: ``update_bankroll``
    fires once per trial before staking, and the N-ary
    ``record_settlement(won_leg, return_pcts)`` fires after a staked batch
    settles - ``won_leg`` is ``None`` exactly for a void or push round.
    """

    def __init__(self, *args, **kwargs):
        self.trials_played = 0
        self.settlements = 0
        self.void_rounds = 0
        super().__init__(*args, **kwargs)

    def update_bankroll(self, _total_funds):
        self.trials_played += 1

    def record_settlement(self, won_leg, _return_pcts):
        self.settlements += 1
        if won_leg is None:
            self.void_rounds += 1


class CountingKelly(SettlementCounter, MultiOutcomeKellyCriterion):
    """Multi-outcome Kelly with the example's settlement counters attached."""

    def __init__(self):
        super().__init__(payoffs=PAYOFFS, loss=LOSS, transaction_cost=0)


class FlatStakesStrategy(SettlementCounter, BaseMultiOutcomeStrategy):
    """Stake a fixed fraction of the bankroll on every leg, win or lose."""

    def __init__(self, fraction):
        super().__init__(payoffs=PAYOFFS, loss=LOSS, transaction_cost=0)
        self.fraction = fraction

    def evaluate(self, probabilities, _current_bankroll):
        return tuple(self.fraction for _ in probabilities)


class FavoriteOnlyStrategy(SettlementCounter, BaseMultiOutcomeStrategy):
    """Stake a fixed fraction on the single most probable leg only."""

    def __init__(self, fraction):
        super().__init__(payoffs=PAYOFFS, loss=LOSS, transaction_cost=0)
        self.fraction = fraction

    def evaluate(self, probabilities, _current_bankroll):
        favorite = max(range(len(probabilities)), key=lambda leg: probabilities[leg])
        stakes = [0.0] * len(probabilities)
        stakes[favorite] = self.fraction
        return tuple(stakes)


def run_simulation(strategy, simulation_index):
    """Run one seeded simulation of a fresh strategy and report its outcome."""
    bankroll = BankRoll(INITIAL_BANKROLL)
    simulator = RepeatedMultiOutcomeSimulator(
        payoffs=PAYOFFS,
        loss=LOSS,
        transaction_costs=TRANSACTION_COSTS,
        probabilities=PROBABILITIES,
        trials=NUM_TRIALS,
        seed=BASE_SEED + simulation_index,
    )
    simulator.evaluate_strategy(strategy, bankroll)
    stopped_early = strategy.trials_played < NUM_TRIALS
    return bankroll.total_funds, stopped_early, strategy.void_rounds


def demonstrate_seeded_replay():
    """Show the reproducibility contract: a seeded run replays identically."""

    def seeded_run():
        bankroll = BankRoll(INITIAL_BANKROLL)
        strategy = CountingKelly()
        simulator = RepeatedMultiOutcomeSimulator(
            payoffs=PAYOFFS,
            loss=LOSS,
            transaction_costs=TRANSACTION_COSTS,
            probabilities=PROBABILITIES,
            trials=REPLAY_TRIALS,
            seed=BASE_SEED,
        )
        simulator.evaluate_strategy(strategy, bankroll)
        return bankroll.history

    history = seeded_run()
    replay = seeded_run()
    assert history == replay, "seeded run did not replay identically"
    print(
        f"Seeded replay: {REPLAY_TRIALS} trials ran twice and produced the "
        f"identical {len(history)}-entry bankroll history."
    )


def main():
    """Run the 1X2 demonstration and write the comparison chart and table."""
    print("1X2 market: home 3.2 (42%), draw 3.4 (27%), away 2.4 (28%), 3% void\n")

    # Opening stake vector: the sizes Kelly actually recommends.
    kelly = MultiOutcomeKellyCriterion(payoffs=PAYOFFS, loss=LOSS, transaction_cost=0)
    stakes = kelly.evaluate(PROBABILITIES, INITIAL_BANKROLL)
    print("Multi-outcome Kelly opening stakes (fraction of bankroll per leg):")
    print(f"  home {stakes[0]:.4f}, draw {stakes[1]:.4f}, away {stakes[2]:.4f}")
    print("  (the negative-edge away leg is declined entirely)\n")

    # Part 1: the seeding and reproducibility contract as public behavior.
    demonstrate_seeded_replay()

    # Part 2: strategy comparison over seeded simulations.
    strategies = {
        "Multi-outcome Kelly": CountingKelly,
        "Flat 5% (all legs)": lambda: FlatStakesStrategy(0.05),
        "Favorite-only 10%": lambda: FavoriteOnlyStrategy(0.10),
    }

    print(
        f"\nRunning {NUM_SIMULATIONS} seeded simulations of {NUM_TRIALS} "
        "matches per strategy..."
    )
    results = {}
    for name, factory in strategies.items():
        finals = []
        stops = 0
        voids = 0
        for simulation_index in range(NUM_SIMULATIONS):
            final, stopped_early, void_rounds = run_simulation(
                factory(), simulation_index
            )
            finals.append(final)
            stops += stopped_early
            voids += void_rounds
        results[name] = finals
        print(
            f"  {name}: mean ${sum(finals) / len(finals):,.0f}, "
            f"void rounds {voids}, early stops {stops}/{NUM_SIMULATIONS}"
        )

    results_table = pd.DataFrame(
        [
            {
                "Strategy": name,
                "Mean": f"${sum(finals) / len(finals):,.0f}",
                "Median": f"${sorted(finals)[len(finals) // 2]:,.0f}",
                "Min": f"${min(finals):,.0f}",
                "Max": f"${max(finals):,.0f}",
            }
            for name, finals in results.items()
        ]
    )
    print(f"\nFinal bankroll distribution (initial bankroll ${INITIAL_BANKROLL:,.0f}):")
    print(results_table.to_string(index=False))

    plt.figure(figsize=(10, 6))
    plt.boxplot(results.values())
    plt.xticks(range(1, len(results) + 1), results.keys())
    plt.yscale("symlog", linthresh=1.0)
    plt.ylabel("Final bankroll ($)")
    plt.title(
        "1X2 final bankrolls over "
        f"{NUM_SIMULATIONS} seeded simulations of {NUM_TRIALS} matches"
    )
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Create the output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(os.path.join(output_dir, "multi_outcome_1x2.png"), dpi=300)
    results_table.to_csv(os.path.join(output_dir, "multi_outcome_1x2.csv"), index=False)

    print(f"\nResults saved to {output_dir}")
    print("Plot saved to file (using non-interactive backend)")


if __name__ == "__main__":
    main()
