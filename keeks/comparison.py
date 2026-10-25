import math
import operator
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields

import numpy as np

from keeks.bankroll import BankRoll
from keeks.binary_strategies.base import BaseStrategy
from keeks.metrics import HistorySummary, summarize_history
from keeks.multi_outcome.base import BaseMultiOutcomeStrategy
from keeks.multi_outcome.simulators import HistoricalMultiOutcomeSimulator
from keeks.simulators.historical_binary import HistoricalBinarySimulator

# Metrics where a smaller value is better; every other sort key ranks larger first.
_ASCENDING = frozenset({"max_drawdown", "ruined"})
_SORT_KEYS = frozenset(f.name for f in fields(HistorySummary)) - {"bets", "start"}


@dataclass(frozen=True)
class StrategyComparison:
    """
    One strategy's replay result: its name plus the run's summary.

    Attributes
    ----------
    name : str
        The key the strategy was supplied under.
    summary : HistorySummary
        :func:`~keeks.metrics.summarize_history` of the replayed bankroll.
    """

    name: str
    summary: HistorySummary


def _validate_comparison_args(strategies, sort_by):
    if not isinstance(strategies, Mapping) or not strategies:
        raise ValueError("strategies must be a non-empty mapping")
    for name, factory in strategies.items():
        if not isinstance(name, str):
            raise ValueError(f"Strategy names must be strings, got {name!r}")
        if not callable(factory):
            raise ValueError(f"Strategy {name!r} must be a zero-argument factory")
    if sort_by not in _SORT_KEYS:
        raise ValueError(
            f"sort_by must be one of {sorted(_SORT_KEYS)}, got {sort_by!r}"
        )


def _rank(rows, sort_by):
    def value(row: StrategyComparison) -> float | None:
        return getattr(row.summary, sort_by)

    defined = [row for row in rows if value(row) is not None]
    undefined = [row for row in rows if value(row) is None]
    defined.sort(key=value, reverse=sort_by not in _ASCENDING)
    return tuple(defined + undefined)


def compare_strategies(
    strategies: Mapping[str, Callable[[], BaseStrategy]],
    probabilities: Sequence[float],
    outcomes: Sequence[bool | int],
    *,
    payoff: float,
    loss: float,
    fee_per_bet: float,
    initial_funds: float,
    percent_bettable: float = 1.0,
    max_transaction_loss: float | None = None,
    sort_by: str = "geometric_growth_per_period",
) -> tuple[StrategyComparison, ...]:
    """
    Replay one recorded bet log through several strategies and rank them.

    Each strategy is built by calling its factory and runs against a fresh
    :class:`~keeks.bankroll.BankRoll`, so no state carries from one run to the
    next. Replay is by
    :class:`~keeks.simulators.historical_binary.HistoricalBinarySimulator`; a
    strategy whose odds differ from ``payoff``/``loss`` raises its
    ``ValueError`` unchanged. This is an educational look at past data, not
    advice.

    Parameters
    ----------
    strategies : mapping of str to callable
        Strategy name to a zero-argument factory returning a fresh strategy.
        Must be non-empty.
    probabilities, outcomes : sequence
        The recorded log, as for ``HistoricalBinarySimulator``.
    payoff, loss, fee_per_bet : float
        The odds and flat fee passed to the simulator.
    initial_funds, percent_bettable, max_transaction_loss : float
        Passed to each fresh ``BankRoll``.
    sort_by : str, default="geometric_growth_per_period"
        A numeric or boolean ``HistorySummary`` field: ``total_return``,
        ``geometric_growth_per_period``, ``end``, ``max_drawdown`` or
        ``ruined``. ``max_drawdown`` and ``ruined`` rank smaller first, the
        rest larger first; ``None`` values always rank last, and ties keep
        the mapping's order.

    Returns
    -------
    tuple of StrategyComparison
        One row per strategy, best first.

    Raises
    ------
    ValueError
        If the arguments are invalid; checked before any strategy runs.

    Examples
    --------
    >>> from keeks import FixedFractionStrategy
    >>> odds = {"payoff": 1.0, "loss": 1.0, "transaction_cost_rate": 0.0}
    >>> rows = compare_strategies(
    ...     {
    ...         "5%": lambda: FixedFractionStrategy(**odds, fraction=0.05),
    ...         "20%": lambda: FixedFractionStrategy(**odds, fraction=0.2),
    ...     },
    ...     [0.6, 0.6, 0.6],
    ...     [1, 0, 1],
    ...     payoff=1.0, loss=1.0, fee_per_bet=0.0, initial_funds=100.0,
    ... )
    >>> [(row.name, round(row.summary.end, 2)) for row in rows]
    [('20%', 115.2), ('5%', 104.74)]
    """
    _validate_comparison_args(strategies, sort_by)

    def fresh_bankroll() -> BankRoll:
        return BankRoll(initial_funds, percent_bettable, max_transaction_loss)

    fresh_bankroll()
    simulator = HistoricalBinarySimulator(
        payoff, loss, fee_per_bet, probabilities, outcomes
    )

    rows = []
    for name, factory in strategies.items():
        bankroll = fresh_bankroll()
        simulator.evaluate_strategy(factory(), bankroll)
        rows.append(StrategyComparison(name, summarize_history(bankroll.history)))

    return _rank(rows, sort_by)


def compare_multi_outcome_strategies(
    strategies: Mapping[str, Callable[[], BaseMultiOutcomeStrategy]],
    probabilities: Sequence[Sequence[float]],
    outcomes: Sequence[int | None],
    *,
    payoffs: Sequence[float],
    loss: float,
    fee_per_bet: float,
    initial_funds: float,
    percent_bettable: float = 1.0,
    max_transaction_loss: float | None = None,
    sort_by: str = "geometric_growth_per_period",
) -> tuple[StrategyComparison, ...]:
    """
    Replay a recorded mutually exclusive market log through several strategies.

    The multi-outcome counterpart of :func:`compare_strategies`: each factory
    is called once to build a fresh vector strategy, which runs against a fresh
    :class:`~keeks.bankroll.BankRoll` via
    :class:`~keeks.multi_outcome.simulators.HistoricalMultiOutcomeSimulator`.
    Rows are ranked as in :func:`compare_strategies`. A strategy whose
    ``payoffs`` or ``loss`` differ from the ones given here raises the
    simulator's ``ValueError`` unchanged. Odds are fixed decimal odds per leg
    and the fee is flat per settled, staked leg. A void charges no fee. This is
    an educational look at past data, not advice.

    Each input row is one market, but ``HistorySummary.bets`` counts settled
    transactions: a void, or a market the strategy declined, adds none, and a
    market staking two legs adds two.

    Parameters
    ----------
    strategies : mapping of str to callable
        Strategy name to a zero-argument factory returning a fresh strategy.
        Must be non-empty.
    probabilities : sequence of probability vectors
        One estimate vector per recorded market, one entry per payoff.
    outcomes : sequence of int or None
        Zero-based winning leg per market, or ``None`` for a void.
    payoffs, loss, fee_per_bet
        The fixed decimal odds, loss multiplier and flat fee passed to the
        simulator.
    initial_funds, percent_bettable, max_transaction_loss, sort_by
        As for :func:`compare_strategies`.

    Returns
    -------
    tuple of StrategyComparison
        One row per strategy, best first.

    Raises
    ------
    ValueError
        If the arguments or recorded log are invalid; checked before any
        factory runs.

    Examples
    --------
    >>> from keeks import MultiOutcomeKellyCriterion
    >>> rows = compare_multi_outcome_strategies(
    ...     {"Kelly": lambda: MultiOutcomeKellyCriterion((3.2, 3.4, 2.4), 1.0)},
    ...     [[0.42, 0.27, 0.28], [0.42, 0.27, 0.28]],
    ...     [0, None],
    ...     payoffs=(3.2, 3.4, 2.4), loss=1.0, fee_per_bet=0.0,
    ...     initial_funds=100.0,
    ... )
    >>> [(row.name, row.summary.bets) for row in rows]
    [('Kelly', 2)]
    """
    _validate_comparison_args(strategies, sort_by)
    BankRoll(initial_funds, percent_bettable, max_transaction_loss)
    simulator = HistoricalMultiOutcomeSimulator(
        payoffs, loss, fee_per_bet, probabilities, outcomes
    )

    rows = []
    for name, factory in strategies.items():
        bankroll = BankRoll(initial_funds, percent_bettable, max_transaction_loss)
        simulator.evaluate_strategy(factory(), bankroll)
        rows.append(StrategyComparison(name, summarize_history(bankroll.history)))
    return _rank(rows, sort_by)


@dataclass(frozen=True)
class BootstrapRow:
    """
    One strategy's bootstrap result for the ``sort_by`` metric.

    Attributes
    ----------
    name : str
        The key the strategy was supplied under.
    point : float or None
        The metric on the full log (``None`` if undefined there).
    lower, upper : float or None
        Percentile bounds over the resamples where the metric was defined
        (``None`` if it was never defined).
    undefined : int
        Resamples where the metric was undefined; excluded from the bounds.
    rank_counts : tuple of int
        ``rank_counts[k]`` is how many resamples ranked the strategy ``k``-th
        (zero-based, best first).
    """

    name: str
    point: float | None
    lower: float | None
    upper: float | None
    undefined: int
    rank_counts: tuple[int, ...]


def _require_int(value, name, minimum):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def bootstrap_comparison(
    strategies: Mapping[str, Callable[[], BaseStrategy]],
    probabilities: Sequence[float],
    outcomes: Sequence[bool | int],
    *,
    payoff: float,
    loss: float,
    fee_per_bet: float,
    initial_funds: float,
    percent_bettable: float = 1.0,
    max_transaction_loss: float | None = None,
    n_resamples: int = 200,
    block_size: int = 1,
    seed: int | None = None,
    confidence: float = 0.9,
    sort_by: str = "geometric_growth_per_period",
) -> tuple[BootstrapRow, ...]:
    """
    Put percentile intervals and rank counts on a strategy comparison.

    The log's bets are resampled with replacement (or in moving blocks of
    ``block_size``), kept in chronological order, and every strategy is
    replayed on the *same* resample via :func:`compare_strategies`, so
    differences between strategies are paired. Intervals reflect sampling of
    the recorded log only, not model error or a different future. Runtime is
    ``n_resamples`` times the cost of one :func:`compare_strategies` call.

    Parameters
    ----------
    strategies, probabilities, outcomes, payoff, loss, fee_per_bet, \
initial_funds, percent_bettable, max_transaction_loss, sort_by
        As for :func:`compare_strategies`.
    n_resamples : int, default=200
        Number of bootstrap resamples (at least 1).
    block_size : int, default=1
        Length of the moving blocks; 1 resamples single bets. Must not exceed
        the log length.
    seed : int
        Required nonnegative integer seeding a private
        ``numpy.random.Generator``; the global RNG is never used.
    confidence : float, default=0.9
        Central interval mass, strictly between 0 and 1.

    Returns
    -------
    tuple of BootstrapRow
        One row per strategy in the mapping's order.

    Raises
    ------
    ValueError
        If ``n_resamples``, ``block_size``, ``confidence`` or ``seed`` is
        invalid, or the log or strategies are; checked before any replay.

    Examples
    --------
    >>> from keeks import FixedFractionStrategy
    >>> odds = {"payoff": 1.0, "loss": 1.0, "transaction_cost_rate": 0.0}
    >>> rows = bootstrap_comparison(
    ...     {
    ...         "none": lambda: FixedFractionStrategy(**odds, fraction=0.0),
    ...         "10%": lambda: FixedFractionStrategy(**odds, fraction=0.1),
    ...     },
    ...     [0.7] * 8,
    ...     [1] * 8,
    ...     payoff=1.0, loss=1.0, fee_per_bet=0.0, initial_funds=100.0,
    ...     n_resamples=20, seed=1,
    ... )
    >>> [(row.name, row.rank_counts) for row in rows]
    [('none', (0, 20)), ('10%', (20, 0))]
    """
    n_resamples = _require_int(n_resamples, "n_resamples", 1)
    block_size = _require_int(block_size, "block_size", 1)
    if seed is None:
        raise ValueError("seed is required")
    seed = _require_int(seed, "seed", 0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence must be strictly between 0 and 1") from exc
    if not (math.isfinite(confidence) and 0.0 < confidence < 1.0):
        raise ValueError("confidence must be strictly between 0 and 1")
    n = len(probabilities)
    if block_size > n:
        raise ValueError("block_size must not exceed the number of bets")

    run = {
        "payoff": payoff,
        "loss": loss,
        "fee_per_bet": fee_per_bet,
        "initial_funds": initial_funds,
        "percent_bettable": percent_bettable,
        "max_transaction_loss": max_transaction_loss,
        "sort_by": sort_by,
    }
    full = compare_strategies(strategies, probabilities, outcomes, **run)
    names = list(strategies)
    points = {row.name: getattr(row.summary, sort_by) for row in full}

    rng = np.random.default_rng(seed)
    n_blocks = -(-n // block_size)
    values: dict[str, list[float]] = {name: [] for name in names}
    ranks = {name: [0] * len(names) for name in names}
    for _ in range(n_resamples):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        picked = (starts[:, None] + np.arange(block_size)).ravel()[:n]
        picked = np.sort(picked)
        rows = compare_strategies(
            strategies,
            [probabilities[i] for i in picked],
            [outcomes[i] for i in picked],
            **run,
        )
        for rank, row in enumerate(rows):
            ranks[row.name][rank] += 1
            value = getattr(row.summary, sort_by)
            if value is not None:
                values[row.name].append(float(value))

    tail = (1.0 - confidence) / 2.0 * 100.0
    result = []
    for name in names:
        seen = values[name]
        if seen:
            lower, upper = (float(v) for v in np.percentile(seen, [tail, 100 - tail]))
        else:
            lower = upper = None
        point = points[name]
        result.append(
            BootstrapRow(
                name,
                None if point is None else float(point),
                lower,
                upper,
                n_resamples - len(seen),
                tuple(ranks[name]),
            )
        )
    return tuple(result)
