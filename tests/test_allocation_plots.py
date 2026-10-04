"""Tests for the allocation layer's visualization helpers.

Every helper is a pure function of its inputs: figures are bare
``matplotlib.figure.Figure`` objects (never pyplot-managed), series and
legends sort by name, and colors come from the fixed palette. Tests run
headless under Agg and assert on the returned artists - line counts, label
sets, data limits, bar heights - never on pixels.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from keeks import BankRoll
from keeks.allocation import HierarchicalRiskParity
from keeks.allocation.plots import (
    PALETTE,
    bankroll_paths,
    correlation_heatmap,
    dendrogram,
    drawdown_history,
    efficient_frontier,
    risk_contributions,
    scenario_losses,
    weight_evolution,
)

matplotlib.use("Agg", force=True)

COVARIANCE = [[0.04, 0.004], [0.004, 0.01]]


# ---
# bankroll_paths: one line per named history, sorted by name, log scale
# by default, palette cycling, histories or owning bankrolls accepted.
# ---


def test_bankroll_paths_sorts_series_and_legend():
    """Series draw and legend in sorted-name order regardless of dict order."""
    axes = bankroll_paths({"zeta": [100.0, 110.0], "alpha": [100.0, 120.0]})
    labels = [line.get_label() for line in axes.get_lines()]
    assert labels == ["alpha", "zeta"]
    legend_labels = [text.get_text() for text in axes.get_legend().get_texts()]
    assert legend_labels == ["alpha", "zeta"]


def test_bankroll_paths_lines_carry_the_histories():
    """Each line plots its history against its period index."""
    axes = bankroll_paths({"a": [1000.0, 1100.0], "b": [500.0, 400.0, 450.0]})
    first, second = axes.get_lines()
    assert first.get_ydata().tolist() == [1000.0, 1100.0]
    assert first.get_xdata().tolist() == [0.0, 1.0]
    assert second.get_ydata().tolist() == [500.0, 400.0, 450.0]


def test_bankroll_paths_accepts_bankroll_objects():
    """A BankRoll contributes its history attribute."""
    bankroll = BankRoll(initial_funds=1000.0)
    bankroll.deposit(250.0)
    axes = bankroll_paths({"run": bankroll})
    assert axes.get_lines()[0].get_ydata().tolist() == [1000.0, 1250.0]


def test_bankroll_paths_accepts_a_bare_history():
    """A single bare sequence plots under the name 'bankroll'."""
    axes = bankroll_paths([100.0, 125.0])
    assert [line.get_label() for line in axes.get_lines()] == ["bankroll"]


def test_bankroll_paths_log_scale_is_the_default_and_flippable():
    axes = bankroll_paths({"a": [1.0, 2.0]})
    assert axes.get_yscale() == "log"
    linear = bankroll_paths({"a": [1.0, 2.0]}, log_scale=False)
    assert linear.get_yscale() == "linear"


def test_bankroll_paths_cycles_the_palette():
    """More series than palette entries cycle the colors from the start."""
    series = {f"s{index}": [1.0, 2.0] for index in range(len(PALETTE) + 1)}
    axes = bankroll_paths(series)
    colors = [line.get_color() for line in axes.get_lines()]
    assert colors[0] == colors[-1] == PALETTE[0]
    assert len(set(colors)) == len(PALETTE)


def test_bankroll_paths_rejects_an_empty_mapping():
    with pytest.raises(ValueError, match="At least one bankroll history"):
        bankroll_paths({})


def test_bankroll_paths_validates_histories():
    with pytest.raises(ValueError, match="must be a finite sequence"):
        bankroll_paths({"a": "junk"})
    with pytest.raises(ValueError, match="one-dimensional"):
        bankroll_paths({"a": [[1.0, 2.0], [3.0, 4.0]]})
    with pytest.raises(ValueError, match="non-empty"):
        bankroll_paths({"a": []})
    with pytest.raises(ValueError, match="only finite values"):
        bankroll_paths({"a": [1.0, float("nan")]})
    with pytest.raises(ValueError, match="nonnegative"):
        bankroll_paths({"a": [1.0, -0.5]})
    with pytest.raises(ValueError, match="History for 'a'"):
        bankroll_paths({"a": [float("nan")]})


# ---
# drawdown_history: the peak-to-trough fraction, zero before the first positive
# bankroll, accepted as a sequence or an owning BankRoll.
# ---


def test_drawdown_history_plots_the_peak_to_trough_fraction():
    axes = drawdown_history([1000.0, 1250.0, 1000.0, 1500.0])
    (line,) = axes.get_lines()
    assert line.get_ydata().tolist() == [0.0, 0.0, 0.2, 0.0]
    assert len(axes.collections) == 1  # the light fill beneath the curve


def test_drawdown_history_accepts_a_bankroll():
    bankroll = BankRoll(initial_funds=1000.0)
    bankroll.withdraw(250.0)
    (line,) = drawdown_history(bankroll).get_lines()
    assert line.get_ydata().tolist() == [0.0, 0.25]


def test_drawdown_history_is_zero_before_the_first_positive_value():
    """No banked funds means nothing to lose - zero drawdown, no crash."""
    (line,) = drawdown_history([0.0, 0.0, 100.0, 50.0]).get_lines()
    assert line.get_ydata().tolist() == [0.0, 0.0, 0.0, 0.5]


def test_drawdown_history_validates_the_history():
    with pytest.raises(ValueError, match="one-dimensional"):
        drawdown_history([[1.0, 2.0]])
    with pytest.raises(ValueError, match="non-empty"):
        drawdown_history([])
    with pytest.raises(ValueError, match="only finite values"):
        drawdown_history([1.0, float("inf")])
    with pytest.raises(ValueError, match="nonnegative"):
        drawdown_history([-1.0])


# ---
# weight_evolution: stacked areas bottom-up in option order, y pinned to
# the [0, 1] budget, labels in option order.
# ---


def test_weight_evolution_labels_and_limits():
    axes = weight_evolution([[0.5, 0.25], [0.25, 0.5]])
    assert [text.get_text() for text in axes.get_legend().get_texts()] == [
        "Option 0",
        "Option 1",
    ]
    assert axes.get_ylim() == (0.0, 1.0)
    assert len(axes.collections) == 2  # one area band per option


def test_weight_evolution_cycles_the_palette():
    matrix = np.tile(np.eye(len(PALETTE) + 1)[:1] * 0.5, (2, 1))
    axes = weight_evolution(matrix)
    assert len(axes.collections) == len(PALETTE) + 1


def test_weight_evolution_validates_the_matrix():
    with pytest.raises(ValueError, match="must be a finite sequence"):
        weight_evolution("junk")
    with pytest.raises(ValueError, match="two-dimensional"):
        weight_evolution([0.5, 0.5])
    with pytest.raises(ValueError, match="non-empty"):
        weight_evolution([[]])
    with pytest.raises(ValueError, match="only finite values"):
        weight_evolution([[float("nan"), 0.5]])
    with pytest.raises(ValueError, match="between 0 and 1"):
        weight_evolution([[1.5, 0.5]])
    with pytest.raises(ValueError, match="sum to no more than one"):
        weight_evolution([[0.8, 0.8]])


# ---
# risk_contributions: variance shares as bars, one per option in order.
# ---


def test_risk_contributions_shares_sum_to_one():
    axes = risk_contributions([0.5, 0.5], COVARIANCE)
    heights = [patch.get_height() for patch in axes.patches]
    assert len(heights) == 2
    assert sum(heights) == pytest.approx(1.0)
    # w_i (Sigma w)_i / (w' Sigma w): the cov term splits the remainder
    # off the naive 80/20 variance share.
    covariance = np.asarray(COVARIANCE)
    weights = np.array([0.5, 0.5])
    expected = weights * (covariance @ weights) / (weights @ covariance @ weights)
    assert heights[0] == pytest.approx(float(expected[0]))
    assert heights[1] == pytest.approx(float(expected[1]))
    assert heights[0] > heights[1]  # the riskier option still drives risk


def test_risk_contributions_labels_options_in_order():
    axes = risk_contributions([0.5, 0.5], COVARIANCE)
    assert [tick.get_text() for tick in axes.get_xticklabels()] == [
        "Option 0",
        "Option 1",
    ]


def test_risk_contributions_rejects_a_riskless_portfolio():
    with pytest.raises(ValueError, match="variance must be positive"):
        risk_contributions([0.0, 0.0], COVARIANCE)


def test_risk_contributions_propagates_descriptor_validation():
    with pytest.raises(ValueError, match="between 0 and 1"):
        risk_contributions([1.5, 0.5], COVARIANCE)
    with pytest.raises(ValueError, match="exactly 2x2"):
        risk_contributions(
            [0.5, 0.5], [[0.04, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.02]]
        )


# ---
# efficient_frontier: the swept mean-variance optima as a line, the
# options as a labeled scatter, points sorted by volatility.
# ---


def test_efficient_frontier_line_and_scatter():
    mean = [0.02, 0.01]
    grid = [1.0, 0.5, 2.0]
    axes = efficient_frontier(mean, COVARIANCE, grid)
    assert [text.get_text() for text in axes.get_legend().get_texts()] == [
        "Frontier",
        "Option 0",
        "Option 1",
    ]
    (line,) = axes.get_lines()
    # The grid arrives unsorted; the frontier draws low-volatility first.
    assert line.get_xdata().tolist() == sorted(line.get_xdata().tolist())
    assert len(line.get_xdata()) == len(grid)
    # The option scatter sits at each option's own (volatility, mean);
    # each single-point scatter call is its own collection.
    points = np.vstack([collection.get_offsets() for collection in axes.collections])
    assert points[0].tolist() == pytest.approx([0.2, 0.02])
    assert points[1].tolist() == pytest.approx([0.1, 0.01])


def test_efficient_frontier_validates_the_grid():
    with pytest.raises(ValueError, match="non-empty one-dimensional"):
        efficient_frontier([0.02, 0.01], COVARIANCE, [])
    with pytest.raises(ValueError, match="non-empty one-dimensional"):
        efficient_frontier([0.02, 0.01], COVARIANCE, [[1.0]])
    with pytest.raises(ValueError, match="positive finite risk aversions"):
        efficient_frontier([0.02, 0.01], COVARIANCE, [0.0, -1.0])
    with pytest.raises(ValueError, match="positive finite risk aversions"):
        efficient_frontier([0.02, 0.01], COVARIANCE, [float("nan")])
    with pytest.raises(ValueError, match="must be a finite sequence"):
        efficient_frontier([0.02, 0.01], COVARIANCE, "junk")


# ---
# correlation_heatmap: the clipped correlation matrix as an image on the
# fixed diverging colormap, colorbar attached.
# ---


def test_correlation_heatmap_image_and_clim():
    axes = correlation_heatmap(COVARIANCE)
    assert len(axes.images) == 1
    assert axes.images[0].get_clim() == (-1.0, 1.0)
    np.testing.assert_allclose(
        axes.images[0].get_array(), [[1.0, 0.2], [0.2, 1.0]], atol=1e-12
    )
    assert len(axes.figure.axes) == 2  # the heatmap plus its colorbar


def test_correlation_heatmap_validates_the_covariance():
    with pytest.raises(ValueError, match="strictly positive variances"):
        correlation_heatmap([[0.04, 0.0], [0.0, 0.0]])
    with pytest.raises(ValueError, match="symmetric"):
        correlation_heatmap([[0.04, 0.1], [0.0, 0.01]])


# ---
# dendrogram: u-shaped merge connectors over the quasi-diagonal leaf
# order, leaf ticks labeled by option index, the empty linkage for a
# single option.
# ---


def test_dendrogram_single_merge():
    strategy = HierarchicalRiskParity(COVARIANCE)
    axes = dendrogram(strategy.linkage)
    (line,) = axes.get_lines()
    assert line.get_xdata().tolist() == [0.0, 0.0, 1.0, 1.0]
    assert line.get_ydata().tolist() == [
        0.0,
        pytest.approx(float(strategy.linkage[0, 2])),
        pytest.approx(float(strategy.linkage[0, 2])),
        0.0,
    ]
    assert [tick.get_text() for tick in axes.get_xticklabels()] == ["0", "1"]
    assert axes.get_xlim() == (-0.5, 1.5)


def test_dendrogram_nested_clusters():
    """Children that are themselves clusters hang from their merge heights."""
    # Two correlated pairs: options {0, 1} and {2, 3} merge first, then the
    # root merges the two clusters - so the root's children are clusters.
    covariance = 0.01 * np.array(
        [
            [1.0, 0.9, 0.1, 0.1],
            [0.9, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.9],
            [0.1, 0.1, 0.9, 1.0],
        ]
    )
    linkage = HierarchicalRiskParity(covariance).linkage
    axes = dendrogram(linkage)
    assert len(axes.get_lines()) == 3
    # Leaves sit at consecutive positions in the tree's left-to-right order.
    assert [tick.get_text() for tick in axes.get_xticklabels()] == ["0", "1", "2", "3"]
    # The root connector spans the two first-merge heights at its legs and
    # the root height at its horizontal bar.
    root = axes.get_lines()[-1]
    assert max(root.get_ydata()) == pytest.approx(float(linkage[2, 2]))
    assert min(root.get_ydata()) == pytest.approx(float(linkage[0, 2]))


def test_dendrogram_single_option_has_no_merges():
    strategy = HierarchicalRiskParity([[0.04]])
    assert strategy.linkage.shape == (0, 4)
    axes = dendrogram(strategy.linkage)
    assert axes.get_lines() == []
    assert [tick.get_text() for tick in axes.get_xticklabels()] == ["0"]
    assert axes.get_ylim() == (0.0, 1.0)


def test_dendrogram_validates_the_linkage():
    with pytest.raises(ValueError, match="must be a finite sequence"):
        dendrogram("junk")
    with pytest.raises(ValueError, match="four columns"):
        dendrogram([[0.0, 1.0, 0.5]])
    with pytest.raises(ValueError, match="four columns"):
        dendrogram([0.0, 1.0, 0.5, 2.0])
    with pytest.raises(ValueError, match="only finite values"):
        dendrogram([[0.0, 1.0, float("nan"), 2.0]])
    with pytest.raises(ValueError, match="nonnegative"):
        dendrogram([[0.0, 1.0, -0.5, 2.0]])


# ---
# scenario_losses: the loss histogram with optional VaR/CVaR markers, a
# legend only when a marker carries one.
# ---


SCENARIOS = [[0.03, 0.01], [-0.01, 0.02], [0.01, -0.01], [-0.02, -0.02]]


def test_scenario_losses_markers_and_legend():
    axes = scenario_losses(SCENARIOS, [0.5, 0.5], var=0.005, cvar=0.015)
    assert len(axes.patches) > 0  # the histogram bars
    assert len(axes.lines) == 2  # one axvline marker each
    assert [text.get_text() for text in axes.get_legend().get_texts()] == [
        "VaR",
        "CVaR",
    ]
    var_line, cvar_line = axes.get_lines()
    assert var_line.get_xdata()[0] == 0.005
    assert cvar_line.get_xdata()[0] == 0.015


def test_scenario_losses_without_markers_has_no_legend():
    axes = scenario_losses(SCENARIOS, [0.5, 0.5])
    assert axes.get_legend() is None
    assert axes.get_lines() == []


def test_scenario_losses_validates_the_inputs():
    with pytest.raises(ValueError, match="two-dimensional"):
        scenario_losses([0.01, 0.02], [0.5, 0.5])
    with pytest.raises(ValueError, match="between 0 and 1"):
        scenario_losses(SCENARIOS, [0.5, -0.5])
    with pytest.raises(ValueError, match="exactly 2 entries"):
        scenario_losses(SCENARIOS, [0.5])
    with pytest.raises(ValueError, match="Bins must be a positive integer"):
        scenario_losses(SCENARIOS, [0.5, 0.5], bins="30")
    with pytest.raises(ValueError, match="Bins must be a positive integer"):
        scenario_losses(SCENARIOS, [0.5, 0.5], bins=0)
    with pytest.raises(ValueError, match="VaR must be a finite number"):
        scenario_losses(SCENARIOS, [0.5, 0.5], var=float("nan"))
    with pytest.raises(ValueError, match="CVaR must be a finite number"):
        scenario_losses(SCENARIOS, [0.5, 0.5], cvar=float("inf"))


# ---
# Cross-cutting determinism and global-state rules.
# ---


def test_helpers_are_deterministic():
    """Identical inputs draw identical artists, twice in a row."""
    first = drawdown_history([1000.0, 1250.0, 1000.0])
    second = drawdown_history([1000.0, 1250.0, 1000.0])
    assert first.get_lines()[0].get_ydata().tolist() == (
        second.get_lines()[0].get_ydata().tolist()
    )
    assert first.get_lines()[0].get_color() == second.get_lines()[0].get_color()


def test_helpers_leave_pyplot_empty():
    """No helper registers a figure in pyplot's global manager."""
    bankroll_paths({"a": [1.0, 2.0]})
    drawdown_history([1.0, 2.0])
    weight_evolution([[0.5, 0.5]])
    risk_contributions([0.5, 0.5], COVARIANCE)
    efficient_frontier([0.02, 0.01], COVARIANCE, [1.0])
    correlation_heatmap(COVARIANCE)
    dendrogram(HierarchicalRiskParity(COVARIANCE).linkage)
    scenario_losses(SCENARIOS, [0.5, 0.5])
    assert plt.get_fignums() == []


def test_helpers_return_the_axes_they_drew_on():
    for axes in (
        bankroll_paths({"a": [1.0, 2.0]}),
        drawdown_history([1.0, 2.0]),
        weight_evolution([[0.5, 0.5]]),
        risk_contributions([0.5, 0.5], COVARIANCE),
        efficient_frontier([0.02, 0.01], COVARIANCE, [1.0]),
        correlation_heatmap(COVARIANCE),
        dendrogram(HierarchicalRiskParity(COVARIANCE).linkage),
        scenario_losses(SCENARIOS, [0.5, 0.5]),
    ):
        assert isinstance(axes, matplotlib.axes.Axes)
        assert axes.get_xlabel()
        assert axes.get_ylabel()
