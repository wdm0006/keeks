"""Guards for the published strategy benchmark.

The benchmark's value is that its numbers can be regenerated, so the properties
worth testing are the ones a reader otherwise has to take on trust: the run is
reproducible, every strategy meets the same outcomes from the same starting state,
and an early stop is reported rather than inferred from a short history.
"""

import importlib.util
import random
from pathlib import Path

import pytest

import keeks.binary_strategies as binary_strategies
from keeks.simulators import repeated_binary

BENCHMARK_PATH = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "strategy_benchmark.py"
)


def _load_benchmark():
    spec = importlib.util.spec_from_file_location("strategy_benchmark", BENCHMARK_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BENCHMARK = _load_benchmark()


@pytest.fixture
def short_run(monkeypatch):
    """The published code path, over few enough bets to stay quick."""
    monkeypatch.setattr(BENCHMARK, "TRIALS", 40)
    return BENCHMARK


def test_every_exported_strategy_is_benchmarked():
    built = {
        type(factory(BENCHMARK.BASE)).__name__
        for factory in BENCHMARK.STRATEGY_FACTORIES.values()
    }
    assert built == set(binary_strategies.__all__)


@pytest.mark.parametrize("strategy_name", list(BENCHMARK.STRATEGY_FACTORIES))
def test_path_is_reproducible(short_run, strategy_name):
    first = short_run.run_path(short_run.BASE, strategy_name, 3)
    second = short_run.run_path(short_run.BASE, strategy_name, 3)
    assert first == second


def test_result_does_not_depend_on_the_global_random_state(short_run):
    random.seed(1)
    first = short_run.run_path(short_run.BASE, "Kelly", 7)
    random.seed(999_999)
    second = short_run.run_path(short_run.BASE, "Kelly", 7)
    assert first == second


def test_simulator_random_source_is_restored(short_run):
    before = repeated_binary.random
    short_run.run_path(short_run.BASE, "Kelly", 0)
    assert repeated_binary.random is before


def test_outcomes_are_indexed_by_trial_not_by_call():
    """The property that makes the comparison paired.

    Seeding the global RNG would not give this: the simulator draws only when a bet
    is placed, so one declined trial would shift every later outcome for that
    strategy alone.
    """
    clock = BENCHMARK._Clock()
    source = BENCHMARK._ReplayedOutcomes([0.1, 0.2, 0.3], clock)
    clock.trial = 2
    assert source.random() == 0.3
    clock.trial = 0
    assert source.random() == 0.1
    assert source.random() == 0.1


def test_strategies_meet_the_same_outcome_at_the_same_trial(short_run):
    """End-to-end form of the above, in a scenario where strategies do skip bets."""
    scenario = short_run._variant(
        "test-noise", "test", "test", estimate_stdev=0.2, probability=0.52
    )
    consumed = {}
    original = short_run._ReplayedOutcomes.random
    for name in ("Kelly", "Fixed fraction 2%", "CPPI"):
        seen = {}

        def record(self, _seen=seen, _original=original):
            value = _original(self)
            _seen[self._clock.trial] = value
            return value

        short_run._ReplayedOutcomes.random = record
        try:
            short_run.run_path(scenario, name, 11)
        finally:
            short_run._ReplayedOutcomes.random = original
        consumed[name] = seen

    trials = [set(seen) for seen in consumed.values()]
    assert min(len(t) for t in trials) < short_run.TRIALS, (
        "scenario no longer exercises skipped bets, so the test proves nothing"
    )
    shared = set.intersection(*trials)
    assert shared
    for trial in shared:
        values = {consumed[name][trial] for name in consumed}
        assert len(values) == 1


def test_stateful_strategies_start_each_path_from_scratch(short_run):
    """CPPI ratchets its floor, so a reused instance would drift between paths."""
    first = short_run.run_path(short_run.BASE, "CPPI", 0)
    short_run.run_path(short_run.BASE, "CPPI", 1)
    assert short_run.run_path(short_run.BASE, "CPPI", 0) == first


def test_early_stops_are_reported_with_a_reason(short_run):
    """A cap below the stake ends every run, and the cause is recorded, not guessed."""
    scenario = short_run._variant("test-tight", "test", "test", max_draw_down=0.01)
    results = [short_run.run_path(scenario, "Kelly", i) for i in range(10)]
    summary = short_run.summarise(scenario, "Kelly", results)
    assert summary["early_stop_rate"] == 1.0
    assert summary["early_stop_drawdown_rate"] == 1.0
    assert summary["early_stop_bankruptcy_rate"] == 0.0
    assert summary["median_trials_started"] < short_run.TRIALS


def test_the_default_cap_does_not_stop_the_base_scenario(short_run):
    results = [short_run.run_path(short_run.BASE, "Kelly", i) for i in range(10)]
    summary = short_run.summarise(short_run.BASE, "Kelly", results)
    assert summary["early_stop_rate"] == 0.0
    assert summary["median_trials_started"] == short_run.TRIALS
