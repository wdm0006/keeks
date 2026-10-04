"""
Seeding, replay, and stream-purity tests for the allocation simulator.

The simulator's reproducibility contract is the documented public behavior
of the multi-outcome family: a seeded run consumes exactly one private
stream - ``np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])``
- so same-seed reruns replay byte-identically, different seeds diverge, and
the seeded path draws nothing from the process-global generators. A
validation failure consumes no draws, so the simulator that rejected a run
still replays a fresh simulation exactly.
"""

import numpy as np
import pytest

from keeks import BankRoll
from keeks.allocation import AllocationSimulator, FixedWeights, scenario_model

# Six joint realizations over three options; the run helper simulates over
# these with fixed weights so every trial stakes and settles.
MATRIX = [
    [0.02, -0.01, 0.015],
    [-0.005, 0.012, -0.008],
    [0.018, 0.004, 0.011],
    [-0.012, -0.006, 0.009],
    [0.007, 0.015, -0.004],
    [0.001, -0.002, 0.003],
]

WEIGHTS = (0.4, 0.35, 0.25)


def run_simulation(seed, trials=50, probabilities=None, fee_per_bet=0.01):
    simulator = AllocationSimulator(
        scenario_model(MATRIX),
        probabilities=probabilities,
        fee_per_bet=fee_per_bet,
        trials=trials,
        seed=seed,
    )
    bankroll = BankRoll(initial_funds=1000.0, max_transaction_loss=None)
    simulator.evaluate_strategy(FixedWeights(WEIGHTS), bankroll)
    return tuple(bankroll.history)


def test_same_seed_replays_identical_history():
    assert run_simulation(seed=42) == run_simulation(seed=42)


def test_different_seed_produces_different_history():
    assert run_simulation(seed=42) != run_simulation(seed=43)


def test_seeded_simulation_does_not_consume_global_generators():
    before = np.random.get_state()
    run_simulation(seed=20260803)
    run_simulation(seed=7)
    after = np.random.get_state()
    assert np.array_equal(before[1], after[1])
    assert before[2] == after[2]


def test_unweighted_runs_replay_regardless_of_global_seed_state():
    # Changing the legacy global seed must not perturb a seeded run.
    np.random.seed(1234)
    first = run_simulation(seed=11)
    np.random.seed(5678)
    second = run_simulation(seed=11)
    assert first == second


def test_reweighting_the_same_universe_changes_the_realized_sequence():
    # Same seed, same draws of uniform mass - different probability weights
    # remap the band lookup, so the realized trial sequence differs.
    even = run_simulation(seed=5, trials=40, probabilities=[1.0 / 6.0] * 6)
    skewed = run_simulation(
        seed=5, trials=40, probabilities=[0.8, 0.02, 0.02, 0.02, 0.02, 0.02]
    )
    assert even != skewed
    # Both replays are stable under their own weighting.
    assert skewed == run_simulation(
        seed=5, trials=40, probabilities=[0.8, 0.02, 0.02, 0.02, 0.02, 0.02]
    )


class _RejectingAllocator:
    """Returns weights that break the long-only budget."""

    def evaluate(self, _current_bankroll):
        return (0.6, 0.6, 0.0)


class _FixedFraction:
    """Duck-typed static allocator staking fixed weights."""

    def evaluate(self, _current_bankroll):
        return WEIGHTS


def test_validation_failure_consumes_no_draws():
    simulator = AllocationSimulator(
        scenario_model(MATRIX), fee_per_bet=0.0, trials=25, seed=99
    )
    bankroll = BankRoll(initial_funds=1000.0, max_transaction_loss=None)
    with pytest.raises(ValueError, match="sum to no more than one"):
        simulator.evaluate_strategy(_RejectingAllocator(), bankroll)
    # The refused run left nothing behind: the same simulator still replays
    # a fresh same-seed run exactly.
    pristine = AllocationSimulator(
        scenario_model(MATRIX), fee_per_bet=0.0, trials=25, seed=99
    )
    other = BankRoll(initial_funds=1000.0, max_transaction_loss=None)
    pristine.evaluate_strategy(_FixedFraction(), other)
    simulator.evaluate_strategy(_FixedFraction(), bankroll)
    assert bankroll.history == other.history


class _CountingModel:
    """Opaque model that counts sampling calls."""

    def __init__(self):
        self.calls = []

    def sample(self, n_samples, _rng):
        self.calls.append(n_samples)
        return np.zeros((n_samples, 3))


def test_seeded_run_draws_one_batch_per_evaluation():
    # The simulator spends exactly one sampling call per run, sized to the
    # trial count - not one call per trial.
    model = _CountingModel()
    simulator = AllocationSimulator(model, fee_per_bet=0.0, trials=13, seed=3)
    bankroll = BankRoll(initial_funds=1000.0, max_transaction_loss=None)
    simulator.evaluate_strategy(_FixedFraction(), bankroll)
    assert model.calls == [13]
