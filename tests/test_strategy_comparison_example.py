"""Guards for the strategy-comparison example's plotting call.

The example's simulation takes minutes, so the figure-building step is the part
worth pinning: it is the one that broke silently when matplotlib renamed the
``boxplot`` keywords, and it is reachable without running any simulation.
"""

import importlib.util
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from keeks.binary_strategies.kelly import FractionalKellyCriterion, KellyCriterion

EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "strategy_comparison.py"
)


def _load_example():
    spec = importlib.util.spec_from_file_location("strategy_comparison", EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXAMPLE = _load_example()


@pytest.fixture
def results():
    """A synthetic stand-in for ``run_strategy_simulation``'s output."""
    return [
        {
            "name": "Strategy A",
            "results": [900.0, 1000.0, 1100.0, 1250.0],
            "mean": 1062.5,
            "std": 130.0,
        },
        {
            "name": "Strategy B",
            "results": [0.0, 800.0, 1000.0, 3000.0],
            "mean": 1200.0,
            "std": 1100.0,
        },
    ]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_builds_and_saves_without_warnings(results, tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        EXAMPLE.build_comparison_figure(results)
        destination = tmp_path / "strategy_comparison.png"
        plt.savefig(destination, dpi=50)

    assert destination.exists()
    assert destination.stat().st_size > 0


def test_box_plot_is_labelled_with_the_strategy_names(results):
    EXAMPLE.build_comparison_figure(results)

    box_axes = plt.gcf().axes[0]
    labels = [label.get_text() for label in box_axes.get_xticklabels()]
    assert labels == [r["name"] for r in results]


def test_box_plot_is_vertical(results):
    """Dropping ``vert=True`` must not flip the orientation."""
    EXAMPLE.build_comparison_figure(results)

    box_axes = plt.gcf().axes[0]
    # A vertical box plot spans the y axis over the bankroll values, and the
    # x axis over one position per strategy.
    assert box_axes.get_xlim()[1] <= len(results) + 1
    assert box_axes.get_ylim()[1] >= max(max(r["results"]) for r in results)


def test_strategy_simulation_is_reproducible_with_paired_indexed_seeds(monkeypatch):
    monkeypatch.setattr(EXAMPLE, "NUM_TRIALS", 50)
    monkeypatch.setattr(EXAMPLE, "NUM_SIMULATIONS", 8)
    simulator_class = EXAMPLE.RepeatedBinarySimulator
    seeds = []

    def build_simulator(**kwargs):
        seeds.append(kwargs["seed"])
        return simulator_class(**kwargs)

    monkeypatch.setattr(EXAMPLE, "RepeatedBinarySimulator", build_simulator)
    params = {
        "payoff": EXAMPLE.PAYOFF,
        "loss": EXAMPLE.LOSS,
        "transaction_cost": EXAMPLE.TRANS_COST,
    }

    first = EXAMPLE.run_strategy_simulation(KellyCriterion, "Kelly", params)
    second = EXAMPLE.run_strategy_simulation(KellyCriterion, "Kelly", params)
    fractional_params = {**params, "fraction": 0.5}
    EXAMPLE.run_strategy_simulation(
        FractionalKellyCriterion, "Half Kelly", fractional_params
    )

    expected = [
        1002.03,
        985.7,
        1018.64,
        1002.04,
        1027.04,
        1002.04,
        1061.36,
        1018.64,
    ]
    assert first["results"] == expected
    assert second["results"] == expected
    indexed_seeds = [EXAMPLE.BASE_SEED + i for i in range(EXAMPLE.NUM_SIMULATIONS)]
    assert seeds == indexed_seeds * 3
