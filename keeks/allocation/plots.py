"""
Deterministic visualization helpers for the allocation layer.

Every helper here takes objects the allocation API already returns - a
bankroll history (or the :class:`~keeks.bankroll.BankRoll` that owns one),
an :class:`~keeks.allocation.base.AllocationResult`, a weight matrix, a
covariance matrix, the hand-rolled HRP linkage, a scenario matrix - and
draws one figure, returning the :class:`matplotlib.axes.Axes` it drew on so
callers can style further, save, or embed.

The helpers are deterministic by construction: series and legends sort by
name, colors come from one small fixed :data:`PALETTE`, and nothing reads or
writes matplotlib's global state. Figures are plain
:class:`matplotlib.figure.Figure` objects - never pyplot-managed - so
repeated calls never leak figures into pyplot's registry, and the helpers
are safe to call inside simulation loops and headless (Agg) test runs.
"""

import operator
from collections.abc import Mapping
from typing import TYPE_CHECKING

import matplotlib.axes
import matplotlib.figure
import numpy as np

from keeks.allocation.base import (
    _validate_covariance,
    _validate_mean,
    _validate_scenarios,
    _validate_weights,
)
from keeks.allocation.moments import MeanVariance
from keeks.utils import PROBABILITY_SUM_TOLERANCE, _require_finite

if TYPE_CHECKING:
    from keeks.bankroll import BankRoll

__author__ = "willmcginnis"

# One small fixed palette: helpers draw from these in order, so a figure is
# a pure function of its inputs and repeated runs replay identically.
PALETTE = (
    "#1f77b4",  # blue
    "#d62728",  # red
    "#2ca02c",  # green
    "#9467bd",  # purple
    "#ff7f0e",  # orange
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
)

# The fixed diverging colormap for correlation heatmaps: symmetric around
# zero, matching the correlation scale the color reads.
COLORMAP = "RdBu_r"

__all__ = [
    "COLORMAP",
    "PALETTE",
    "bankroll_paths",
    "correlation_heatmap",
    "dendrogram",
    "drawdown_history",
    "efficient_frontier",
    "risk_contributions",
    "scenario_losses",
    "weight_evolution",
]


def _new_axes(xlabel: str, ylabel: str) -> matplotlib.axes.Axes:
    """
    Open a fresh figure and label its axes, touching no global state.

    The figure is a bare :class:`matplotlib.figure.Figure`, not a pyplot
    figure: nothing registers it in pyplot's global manager, so a helper
    called inside a simulation loop cannot leak figures and a headless run
    needs no close bookkeeping.
    """
    figure = matplotlib.figure.Figure()
    axes = figure.add_subplot()
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    return axes


def _history_array(
    history: "np.typing.ArrayLike | BankRoll", label: str = "History"
) -> np.ndarray:
    """
    Coerce a bankroll history to a validated one-dimensional float array.

    A :class:`~keeks.bankroll.BankRoll` (or any object exposing a
    ``history`` attribute) is unwrapped first, so callers can pass either
    the recorded list or the bankroll that owns it. Histories are
    nonnegative by construction - a bankroll never settles below zero -
    and anything else is rejected loudly.

    Parameters
    ----------
    history : sequence of float or object
        The bankroll history, or the object whose ``history`` attribute
        holds one.
    label : str
        The name the validation errors carry.

    Returns
    -------
    numpy.ndarray
        The validated history as a one-dimensional nonnegative float array.

    Raises
    ------
    ValueError
        If the history is not a non-empty one-dimensional sequence of
        finite nonnegative numbers.
    """
    attribute = getattr(history, "history", None)
    if attribute is not None:
        history = attribute
    try:
        values = np.asarray(history, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite sequence") from exc
    if values.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional")
    if values.size == 0:
        raise ValueError(f"{label} must be non-empty")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} must contain only finite values")
    if np.any(values < 0):
        raise ValueError(
            f"{label} must be nonnegative: a bankroll never goes below zero"
        )
    return values


def _validate_weight_history(weights: np.typing.ArrayLike) -> np.ndarray:
    """
    Validate a per-period weight matrix and return it as a float array.

    One row per period, one column per option, every entry in ``[0, 1]``
    and every row summing to no more than one within
    ``PROBABILITY_SUM_TOLERANCE`` - the same long-only contract
    :func:`keeks.allocation.base._validate_weights` enforces on a single
    vector, read across time.

    Parameters
    ----------
    weights : array-like
        The ``(periods, options)`` weight matrix.

    Returns
    -------
    numpy.ndarray
        The validated matrix as a two-dimensional float array.

    Raises
    ------
    ValueError
        If the matrix is not a non-empty two-dimensional sequence of finite
        numbers, any entry falls outside ``[0, 1]``, or any row sums above
        ``1 + PROBABILITY_SUM_TOLERANCE``.
    """
    try:
        weights = np.asarray(weights, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Weights must be a finite sequence") from exc
    if weights.ndim != 2:
        raise ValueError(
            "Weights must be two-dimensional: one row per period, one column per option"
        )
    if weights.size == 0:
        raise ValueError("Weights must be non-empty")
    if not np.all(np.isfinite(weights)):
        raise ValueError("Weights must contain only finite values")
    if np.any((weights < 0) | (weights > 1)):
        raise ValueError("Weights must be between 0 and 1")
    if np.any(weights.sum(axis=1) > 1 + PROBABILITY_SUM_TOLERANCE):
        raise ValueError("Weights must sum to no more than one in every period")
    return weights


def _validate_grid(grid: np.typing.ArrayLike) -> np.ndarray:
    """
    Validate a grid of risk-aversion values and return it as a float array.

    Parameters
    ----------
    grid : sequence of float
        The risk aversions the efficient frontier sweeps. Must be a
        non-empty one-dimensional sequence of finite positive numbers.

    Returns
    -------
    numpy.ndarray
        The validated grid as a one-dimensional float array.

    Raises
    ------
    ValueError
        If the grid is not a non-empty one-dimensional sequence of finite
        positive numbers.
    """
    try:
        grid = np.asarray(grid, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Grid must be a finite sequence") from exc
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError("Grid must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(grid)) or np.any(grid <= 0):
        raise ValueError("Grid must contain only positive finite risk aversions")
    return grid


def _validate_linkage(linkage: np.typing.ArrayLike) -> np.ndarray:
    """
    Validate an agglomerative linkage matrix and return it as a float array.

    The ``(n - 1, 4)`` matrix
    :func:`keeks.allocation.hierarchical._agglomerative_linkage` produces:
    one row per merge, ``[first_id, second_id, distance, size]``, where
    leaves carry their option index and the merge on row ``k`` forms
    cluster ``n + k``. A single option has nothing to merge - the empty
    ``(0, 4)`` matrix - and is the one empty shape accepted.

    Parameters
    ----------
    linkage : array-like
        The linkage matrix.

    Returns
    -------
    numpy.ndarray
        The validated linkage as a two-dimensional float array.

    Raises
    ------
    ValueError
        If the matrix is not two-dimensional with four columns, is not
        finite, or carries a negative merge distance.
    """
    try:
        linkage = np.asarray(linkage, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Linkage must be a finite sequence") from exc
    if linkage.ndim != 2 or linkage.shape[1] != 4:
        raise ValueError(
            "Linkage must have four columns: [first_id, second_id, distance, size]"
        )
    if linkage.size == 0:
        return linkage
    if not np.all(np.isfinite(linkage)):
        raise ValueError("Linkage must contain only finite values")
    if np.any(linkage[:, 2] < 0):
        raise ValueError("Linkage distances must be nonnegative")
    return linkage


def _leaf_positions(linkage: np.ndarray, option_count: int) -> dict[int, float]:
    """
    Lay out every node of the linkage tree on the x-axis, iteratively.

    Leaves sit at consecutive integer positions in the tree's left-to-right
    order - the quasi-diagonal order the linkage encodes - and each cluster
    sits at the midpoint of its two children. The walk is an explicit-stack
    post-order, so deep trees cannot overflow the recursion limit.

    Returns a dict mapping every node id (leaves ``0..n-1``, clusters
    ``n..2n-2``) to its x position; every node is reachable from the root,
    which is why no unknown id can survive.
    """
    positions: dict[int, float] = {}
    root = 2 * option_count - 2
    stack = [(root, False)]
    while stack:
        node, expanded = stack.pop()
        if node < option_count:
            positions[node] = float(len(positions))
        elif expanded:
            row = linkage[node - option_count]
            positions[node] = (positions[int(row[0])] + positions[int(row[1])]) / 2.0
        else:
            stack.append((node, True))
            stack.append((int(linkage[node - option_count, 1]), False))
            stack.append((int(linkage[node - option_count, 0]), False))
    return positions


def bankroll_paths(
    histories: "Mapping[str, np.typing.ArrayLike | BankRoll]"
    " | np.typing.ArrayLike | BankRoll",
    log_scale: bool = True,
) -> matplotlib.axes.Axes:
    """
    Plot the growth curves of one or more bankroll histories.

    Each history becomes one line against its period index. Series sort by
    name - draw order and legend alike - so a figure is a pure function of
    its inputs, and colors cycle through :data:`PALETTE`. The y-axis is
    logarithmic by default: compounding bankrolls span orders of magnitude,
    and zero entries (a bankrupt period) are simply clipped off a log axis.
    Pass ``log_scale=False`` for a linear axis.

    Parameters
    ----------
    histories : mapping or sequence
        Either a mapping of series name to history - each history a
        sequence of nonnegative bankroll values, or the
        :class:`~keeks.bankroll.BankRoll` whose ``history`` attribute holds
        one - or a single bare history, which plots under the name
        ``"bankroll"``.
    log_scale : bool, default=True
        Whether the y-axis is logarithmic.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the paths were drawn on, one line per series, legend
        sorted by name.

    Raises
    ------
    ValueError
        If a history is not a non-empty one-dimensional sequence of finite
        nonnegative numbers, or the mapping is empty.

    Examples
    --------
    >>> from keeks import BankRoll
    >>> from keeks.allocation.plots import bankroll_paths
    >>> bankroll = BankRoll(initial_funds=1000.0)
    >>> bankroll.deposit(250.0)
    >>> axes = bankroll_paths({"fixed": bankroll})
    >>> [text.get_text() for text in axes.get_legend().get_texts()]
    ['fixed']
    >>> axes.get_yscale()
    'log'
    """
    if isinstance(histories, Mapping):
        series = {
            str(name): _history_array(history, label=f"History for {name!r}")
            for name, history in histories.items()
        }
        if not series:
            raise ValueError("At least one bankroll history is required")
    else:
        series = {"bankroll": _history_array(histories)}
    axes = _new_axes("Period", "Bankroll")
    for index, (name, values) in enumerate(sorted(series.items())):
        axes.plot(
            np.arange(values.size, dtype=float),
            values,
            color=PALETTE[index % len(PALETTE)],
            label=name,
        )
    if log_scale:
        axes.set_yscale("log")
    axes.legend()
    return axes


def drawdown_history(history: "np.typing.ArrayLike | BankRoll") -> matplotlib.axes.Axes:
    """
    Plot the peak-to-trough drawdown curve of a bankroll history.

    The drawdown at each entry is the fractional loss from the best
    bankroll seen so far - zero at every new peak, one at a total wipeout -
    read as ``(running_peak - value) / running_peak``. Entries before the
    first positive value (an unfunded bankroll) carry a zero drawdown:
    with nothing ever banked there is nothing to lose.

    Parameters
    ----------
    history : sequence of float or object
        The bankroll history - a sequence of nonnegative values, or the
        :class:`~keeks.bankroll.BankRoll` whose ``history`` attribute holds
        one.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the curve was drawn on, with the drawdown line and a light
        fill beneath it.

    Raises
    ------
    ValueError
        If the history is not a non-empty one-dimensional sequence of
        finite nonnegative numbers.

    Examples
    --------
    >>> from keeks.allocation.plots import drawdown_history
    >>> axes = drawdown_history([1000.0, 1250.0, 1000.0])
    >>> len(axes.lines)
    1
    >>> round(float(axes.lines[0].get_ydata().max()), 4)
    0.2
    """
    values = _history_array(history)
    peaks = np.maximum.accumulate(values)
    # Masked assignment, not np.where: dividing inside np.where still
    # evaluates every element, and 0/0 on an unfunded prefix would warn.
    drawdowns = np.zeros_like(values)
    positive = peaks > 0
    drawdowns[positive] = (peaks[positive] - values[positive]) / peaks[positive]
    periods = np.arange(values.size, dtype=float)
    axes = _new_axes("Period", "Drawdown")
    axes.fill_between(periods, 0.0, drawdowns, color=PALETTE[0], alpha=0.3)
    axes.plot(periods, drawdowns, color=PALETTE[0], label="Drawdown")
    axes.legend()
    return axes


def weight_evolution(weights: np.typing.ArrayLike) -> matplotlib.axes.Axes:
    """
    Plot long-only weights through time as a stacked area.

    One band per option, stacked bottom-up in option order: the stack's top
    edge is the total invested fraction, and the gap down to one is the
    cash held at zero return. The form fits the layer's weight contract
    exactly - weights are nonnegative and sum to no more than one - which
    is why the y-axis is pinned to ``[0, 1]``.

    Parameters
    ----------
    weights : array-like
        The ``(periods, options)`` matrix of long-only weights, every entry
        in ``[0, 1]`` and every row summing to no more than one within
        ``PROBABILITY_SUM_TOLERANCE``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the areas were drawn on, one labeled band per option with
        colors cycling through :data:`PALETTE`.

    Raises
    ------
    ValueError
        If the matrix is not a non-empty two-dimensional sequence of finite
        numbers, any entry falls outside ``[0, 1]``, or any row sums above
        ``1 + PROBABILITY_SUM_TOLERANCE``.

    Examples
    --------
    >>> from keeks.allocation.plots import weight_evolution
    >>> axes = weight_evolution([[0.5, 0.25], [0.25, 0.5]])
    >>> [text.get_text() for text in axes.get_legend().get_texts()]
    ['Option 0', 'Option 1']
    >>> axes.get_ylim() == (0.0, 1.0)
    True
    """
    matrix = _validate_weight_history(weights)
    option_count = matrix.shape[1]
    periods = np.arange(matrix.shape[0], dtype=float)
    axes = _new_axes("Period", "Weight")
    axes.stackplot(
        periods,
        matrix.T,
        labels=[f"Option {index}" for index in range(option_count)],
        colors=[PALETTE[index % len(PALETTE)] for index in range(option_count)],
    )
    axes.set_ylim(0.0, 1.0)
    axes.legend(loc="upper right")
    return axes


def risk_contributions(
    weights: np.typing.ArrayLike, covariance: np.typing.ArrayLike
) -> matplotlib.axes.Axes:
    """
    Plot each option's share of the portfolio's risk as a bar chart.

    The risk contribution of option ``i`` is
    ``w_i (Sigma w)_i / (w' Sigma w)`` - the share of the portfolio variance
    the option's own exposure drives. The shares sum to one, and the equal
    risk contribution portfolio (:class:`~keeks.allocation.moments.RiskBudgeting`
    with equal budgets) is the book where every bar reads ``1 / N``.

    Parameters
    ----------
    weights : array-like
        The long-only weight vector - an
        :class:`~keeks.allocation.base.AllocationResult`'s ``weights`` or
        any equivalent sequence, validated like every allocation weight.
    covariance : array-like
        The covariance matrix of the options' simple returns, matching the
        weight vector's length.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the bars were drawn on, one bar per option in option
        order.

    Raises
    ------
    ValueError
        If the weights or the covariance are invalid, they disagree on the
        option count, or the portfolio carries no variance to attribute.

    Examples
    --------
    Equal weights on diagonal-variance options of 4% and 1% put 80% of the
    risk in the riskier option:

    >>> from keeks.allocation.plots import risk_contributions
    >>> axes = risk_contributions([0.5, 0.5], [[0.04, 0.0], [0.0, 0.01]])
    >>> len(axes.patches)
    2
    >>> round(float(sum(patch.get_height() for patch in axes.patches)), 12)
    1.0
    """
    weights = np.asarray(_validate_weights(weights))
    covariance = _validate_covariance(covariance, option_count=weights.size)
    marginal = covariance @ weights
    variance = float(weights @ marginal)
    if variance <= 0:
        raise ValueError(
            "Portfolio variance must be positive to attribute risk: the "
            "weights carry no exposure or the covariance carries no variance"
        )
    shares = weights * marginal / variance
    option_count = weights.size
    axes = _new_axes("Option", "Risk contribution share")
    axes.bar(
        range(option_count),
        shares,
        color=PALETTE[0],
        tick_label=[f"Option {index}" for index in range(option_count)],
    )
    return axes


def efficient_frontier(
    mean: np.typing.ArrayLike,
    covariance: np.typing.ArrayLike,
    grid: np.typing.ArrayLike,
) -> matplotlib.axes.Axes:
    """
    Plot the long-only mean-variance efficient frontier with the options.

    The frontier curve solves :class:`~keeks.allocation.moments.MeanVariance`
    at every risk aversion in ``grid`` and reads each optimum's
    ``(volatility, expected return)`` pair; the points sort by volatility,
    so the curve is drawn low-volatility first regardless of the grid's
    order. The individual options plot as a scatter at their own
    ``(volatility, expected return)``. Like every scipy-gated method,
    solving requires the ``keeks[allocation]`` optional extra - the first
    :class:`MeanVariance` construction raises the pointed ``ImportError``
    without it.

    Parameters
    ----------
    mean : array-like
        The expected simple return of each option.
    covariance : array-like
        The covariance matrix of the options' simple returns, matching the
        mean's length.
    grid : sequence of float
        The positive risk aversions to sweep, one frontier point each.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the frontier was drawn on: the frontier line plus one
        labeled point per option.

    Raises
    ------
    ImportError
        When scipy is not installed; install the ``keeks[allocation]``
        extra.
    ValueError
        If the mean, the covariance, or the grid is invalid.

    Examples
    --------
    >>> from keeks.allocation.plots import efficient_frontier
    >>> axes = efficient_frontier(
    ...     [0.02, 0.01], [[0.04, 0.0], [0.0, 0.01]], [0.5, 1.0, 2.0]
    ... )
    >>> [text.get_text() for text in axes.get_legend().get_texts()]
    ['Frontier', 'Option 0', 'Option 1']
    """
    mean = _validate_mean(mean)
    covariance = _validate_covariance(covariance, option_count=mean.size)
    grid = _validate_grid(grid)
    frontier = []
    for risk_aversion in grid:
        result = MeanVariance(
            mean, covariance, risk_aversion=float(risk_aversion)
        ).optimize()
        frontier.append((float(result.volatility), float(result.weights @ mean)))
    # Sweeping the risk aversion moves the optimum monotonically toward
    # lower volatility; sorting makes the drawn curve monotone too, immune
    # to the grid's order or solver noise at the rounding level.
    frontier.sort()
    axes = _new_axes("Volatility", "Expected return")
    axes.plot(
        [point[0] for point in frontier],
        [point[1] for point in frontier],
        color=PALETTE[0],
        label="Frontier",
    )
    volatilities = np.sqrt(np.diag(covariance))
    for index in range(mean.size):
        axes.scatter(
            [volatilities[index]],
            [mean[index]],
            color=PALETTE[1],
            label=f"Option {index}",
        )
    axes.legend()
    return axes


def correlation_heatmap(covariance: np.typing.ArrayLike) -> matplotlib.axes.Axes:
    """
    Plot a covariance matrix as a correlation heatmap.

    The covariance rescales to a correlation matrix through the diagonal
    volatilities (clipped to ``[-1, 1]`` against floating-point noise) and
    draws as a square image on the fixed ``"RdBu_r"`` diverging colormap,
    symmetric about zero on the ``[-1, 1]`` color scale.

    Parameters
    ----------
    covariance : array-like
        The covariance matrix of the options' simple returns. Must have
        strictly positive variances - the correlation divides by each
        option's volatility.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the heatmap was drawn on, with the colorbar attached.

    Raises
    ------
    ValueError
        If the covariance is invalid or carries a zero variance.

    Examples
    --------
    >>> from keeks.allocation.plots import correlation_heatmap
    >>> axes = correlation_heatmap([[0.04, 0.004], [0.004, 0.01]])
    >>> len(axes.images)
    1
    >>> axes.images[0].get_clim()
    (-1.0, 1.0)
    """
    covariance = _validate_covariance(covariance)
    variances = np.diag(covariance)
    if np.any(variances <= 0):
        raise ValueError(
            "Covariance must have strictly positive variances: the "
            "correlation divides by each option's volatility"
        )
    std = np.sqrt(variances)
    correlation = np.clip(covariance / np.outer(std, std), -1.0, 1.0)
    size = covariance.shape[0]
    axes = _new_axes("Option", "Option")
    image = axes.imshow(correlation, vmin=-1.0, vmax=1.0, cmap=COLORMAP)
    axes.figure.colorbar(image, ax=axes)
    axes.set_xticks(range(size))
    axes.set_yticks(range(size))
    return axes


def dendrogram(linkage: np.typing.ArrayLike) -> matplotlib.axes.Axes:
    """
    Plot the cluster tree an agglomerative linkage matrix describes.

    The input is the ``(n - 1, 4)`` linkage matrix
    :func:`keeks.allocation.hierarchical._agglomerative_linkage` produces
    and :class:`~keeks.allocation.hierarchical.HierarchicalRiskParity`
    exposes as its ``linkage`` attribute - one row per merge, leaves at
    height zero. Each merge draws its u-shaped connector between the two
    children at the merge distance; leaves sit at consecutive integer
    positions in the tree's left-to-right (quasi-diagonal) order. A single
    option has no merges - the empty ``(0, 4)`` matrix - and draws as one
    tick at height zero.

    Parameters
    ----------
    linkage : array-like
        The ``(n - 1, 4)`` linkage matrix, one row per merge:
        ``[first_id, second_id, distance, size]``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the tree was drawn on, with one tick per option labeled by
        its option index.

    Raises
    ------
    ValueError
        If the matrix is not two-dimensional with four columns, is not
        finite, or carries a negative merge distance.

    Examples
    --------
    >>> from keeks.allocation import HierarchicalRiskParity
    >>> from keeks.allocation.plots import dendrogram
    >>> strategy = HierarchicalRiskParity([[0.04, 0.004], [0.004, 0.01]])
    >>> axes = dendrogram(strategy.linkage)
    >>> len(axes.lines)
    1
    >>> [text.get_text() for text in axes.get_xticklabels()]
    ['0', '1']
    """
    linkage = _validate_linkage(linkage)
    option_count = linkage.shape[0] + 1
    positions = _leaf_positions(linkage, option_count)
    # Node heights: leaves sit at zero, each cluster at its merge distance.
    heights = dict.fromkeys(range(option_count), 0.0)
    for row_index, row in enumerate(linkage):
        heights[option_count + row_index] = float(row[2])
    axes = _new_axes("Option", "Linkage distance")
    for row_index, row in enumerate(linkage):
        first = int(row[0])
        second = int(row[1])
        height = heights[option_count + row_index]
        axes.plot(
            [
                positions[first],
                positions[first],
                positions[second],
                positions[second],
            ],
            [heights[first], height, height, heights[second]],
            color=PALETTE[0],
        )
    leaves = list(range(option_count))
    axes.set_xticks(
        [positions[node] for node in leaves],
        labels=[str(node) for node in leaves],
    )
    axes.set_xlim(-0.5, option_count - 0.5)
    top = max(heights.values())
    axes.set_ylim(0.0, top * 1.02 if top > 0 else 1.0)
    return axes


def scenario_losses(
    scenarios: np.typing.ArrayLike,
    weights: np.typing.ArrayLike,
    var: float | None = None,
    cvar: float | None = None,
    bins: int = 30,
) -> matplotlib.axes.Axes:
    """
    Plot a histogram of portfolio losses over scenarios with tail markers.

    The portfolio's simple returns read off the scenario matrix as
    ``scenarios @ weights``; losses are their negation. When given, ``var``
    and ``cvar`` draw vertical markers - the natural callers are the tail
    statistics of the same distribution, e.g.
    :attr:`keeks.allocation.scenarios.MeanCVaR.cvar` for the expected tail
    loss.

    Parameters
    ----------
    scenarios : array-like
        The ``(observations, options)`` matrix of joint simple returns.
    weights : array-like
        The long-only weight vector, one entry per scenario column.
    var : float, optional
        Where to draw the VaR marker, when given.
    cvar : float, optional
        Where to draw the CVaR marker, when given.
    bins : int, default=30
        The histogram's bin count.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the histogram was drawn on, with a legend when at least
        one marker was drawn.

    Raises
    ------
    ValueError
        If the scenarios or weights are invalid, they disagree on the
        option count, the bins are not a positive integer, or a given
        marker is not finite.

    Examples
    --------
    >>> from keeks.allocation.plots import scenario_losses
    >>> scenarios = [[0.03, 0.01], [-0.01, 0.02], [0.01, -0.01], [-0.02, -0.02]]
    >>> axes = scenario_losses(scenarios, [0.5, 0.5], var=0.005, cvar=0.015)
    >>> [text.get_text() for text in axes.get_legend().get_texts()]
    ['VaR', 'CVaR']
    """
    scenarios, _ = _validate_scenarios(scenarios)
    weights = np.asarray(_validate_weights(weights))
    if weights.size != scenarios.shape[1]:
        raise ValueError(
            f"Weights must carry exactly {scenarios.shape[1]} entries, one "
            f"per option, got {weights.size}"
        )
    try:
        bins = operator.index(bins)
    except TypeError as exc:
        raise ValueError("Bins must be a positive integer") from exc
    if bins <= 0:
        raise ValueError("Bins must be a positive integer")
    if var is not None:
        var = _require_finite(var, "VaR")
    if cvar is not None:
        cvar = _require_finite(cvar, "CVaR")
    losses = -(scenarios @ weights)
    axes = _new_axes("Portfolio loss", "Scenarios")
    axes.hist(losses, bins=bins, color=PALETTE[0])
    if var is not None:
        axes.axvline(var, color=PALETTE[1], label="VaR")
    if cvar is not None:
        axes.axvline(cvar, color=PALETTE[2], label="CVaR")
    if var is not None or cvar is not None:
        axes.legend()
    return axes
