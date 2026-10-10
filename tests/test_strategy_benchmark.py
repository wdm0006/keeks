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
    # BaseStrategy is exported (like every generation's ABC) but is abstract,
    # so it builds nothing and benchmarks nothing.
    assert built == set(binary_strategies.__all__) - {"BaseStrategy"}


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
    scenario = short_run._variant(
        "test-tight", "test", "test", max_transaction_loss=0.01
    )
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


@pytest.mark.parametrize(
    "probability,bias,stdev",
    [
        (0.55, 0.03, 0.0),
        (0.55, 0.05, 0.0),
        (0.55, -0.03, 0.0),
        (0.99, 0.05, 0.0),
        (0.01, -0.03, 0.0),
        (0.55, 0.03, 0.06),
        (0.55, 2.0, 0.06),
        (0.55, -2.0, 0.06),
    ],
)
def test_bias_and_noise_reach_strategy_with_clamped_probability(
    short_run, monkeypatch, probability, bias, stdev
):
    scenario = short_run._variant(
        "test-bias",
        "test",
        "test",
        probability=probability,
        estimate_bias=bias,
        estimate_stdev=stdev,
    )
    seen = []

    def record(_self, probability, _current_bankroll):
        seen.append(probability)
        return 0.0

    monkeypatch.setattr(BENCHMARK.KellyCriterion, "evaluate", record)
    short_run.run_path(scenario, "Kelly", 3)
    rng = random.Random(f"shocks|{short_run.SEED}|3")
    expected = [
        min(
            1.0,
            max(0.0, probability + bias + (stdev * rng.gauss(0, 1) if stdev else 0.0)),
        )
        for _ in range(short_run.TRIALS)
    ]
    assert seen == expected


def test_bias_does_not_change_settlement_probability(short_run, monkeypatch):
    def fixed_stake(_self, _probability, _current_bankroll):
        return 0.02

    monkeypatch.setattr(BENCHMARK.KellyCriterion, "evaluate", fixed_stake)
    biased = short_run._variant("test-bias", "test", "test", estimate_bias=0.4)
    assert short_run.run_path(biased, "Kelly", 3) == short_run.run_path(
        short_run.BASE, "Kelly", 3
    )


def test_bias_scenarios_and_csv_column(short_run):
    scenarios = [s for s in short_run.SCENARIOS if s.axis == "estimate bias"]
    assert {s.key: s.estimate_bias for s in scenarios} == {
        "bias+03": 0.03,
        "bias+05": 0.05,
        "bias-03": -0.03,
    }
    for scenario in scenarios:
        result = short_run.run_path(scenario, "Kelly", 0)
        assert (
            short_run.summarise(scenario, "Kelly", [result])["estimate_bias"]
            == scenario.estimate_bias
        )


def test_documented_bias_figures_match_committed_csv():
    import csv

    root = BENCHMARK_PATH.parents[1]
    with (root / "benchmarks/output/strategy_benchmark.csv").open() as source:
        rows = {(r["strategy"], r["scenario"]): r for r in csv.DictReader(source)}
    docs = (root / "docs/source/strategy_benchmark.rst").read_text()
    section = docs.split(
        ".. csv-table:: Terminal bankroll under constant estimate bias"
    )[1]
    table = [line.strip() for line in section.splitlines() if line.startswith('   "')]
    assert len(table) == 8
    for name, scenario, median, p5, p95, early in csv.reader(
        table, skipinitialspace=True
    ):
        row = rows[name, scenario]
        assert [float(median), float(p5), float(p95)] == [
            float(row[key])
            for key in ("median_terminal", "p5_terminal", "p95_terminal")
        ]
        assert float(early) == 100 * float(row["early_stop_rate"])


def test_undefined_drawdown_keeps_zero_for_bankrupt_path(short_run, monkeypatch):
    original_summary = short_run.summarize_history
    seen = []

    def summarize(history):
        summary = original_summary(history)
        seen.append(summary)
        return summary

    def bankrupt(_self, _strategy, bankroll):
        bankroll.history[:] = [0.0, 0.0]
        bankroll._bank = 0.0

    monkeypatch.setattr(short_run, "summarize_history", summarize)
    monkeypatch.setattr(
        short_run.RepeatedBinarySimulator, "evaluate_strategy", bankrupt
    )
    result = short_run.run_path(short_run.BASE, "Kelly", 0)
    assert len(seen) == 1
    assert seen[0].max_drawdown is None
    assert result.stop_reason == "bankruptcy"
    assert result.max_drawdown == 0.0
