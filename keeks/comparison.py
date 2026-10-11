from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields

from keeks.bankroll import BankRoll
from keeks.binary_strategies.base import BaseStrategy
from keeks.metrics import HistorySummary, summarize_history
from keeks.simulators.historical_binary import HistoricalBinarySimulator

# Metrics where a smaller value is better; every other sort key ranks larger first.
_ASCENDING = frozenset({"max_drawdown"})
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

    def value(row: StrategyComparison) -> float | None:
        return getattr(row.summary, sort_by)

    defined = [row for row in rows if value(row) is not None]
    undefined = [row for row in rows if value(row) is None]
    defined.sort(key=value, reverse=sort_by not in _ASCENDING)
    return tuple(defined + undefined)
