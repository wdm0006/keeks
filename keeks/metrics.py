import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class HistorySummary:
    """
    End-of-run statistics for a bankroll history.

    Attributes
    ----------
    bets : int
        Periods recorded after the starting entry (``len(history) - 1``).
    start, end : float
        The first and last history entries.
    total_return : float or None
        ``end / start - 1``; ``None`` when ``start`` is not positive.
    geometric_growth_per_period : float or None
        ``(end / start) ** (1 / bets) - 1``; ``None`` when undefined (no
        recorded periods, or a nonpositive ``start`` or ``end``).
    max_drawdown : float or None
        Largest peak-to-trough fractional loss, in ``[0, 1]``, measured
        against the running peak; ``None`` when no entry is ever positive.
    ruined : bool
        Whether the final funds are at or below zero.
    """

    bets: int
    start: float
    end: float
    total_return: float | None
    geometric_growth_per_period: float | None
    max_drawdown: float | None
    ruined: bool


def summarize_history(history: Sequence[float]) -> HistorySummary:
    """
    Summarize a :attr:`BankRoll.history <keeks.bankroll.BankRoll.history>`.

    Parameters
    ----------
    history : sequence of float
        Bankroll values in time order, starting with the initial funds.

    Returns
    -------
    HistorySummary
        Frozen dataclass of drawdown, growth and ruin statistics. Values
        that cannot be defined are ``None`` rather than ``0.0``.

    Raises
    ------
    ValueError
        If the history is empty, or holds a non-finite or negative value.

    Examples
    --------
    >>> summary = summarize_history([1000.0, 1250.0, 1000.0])
    >>> summary.bets, summary.total_return, summary.max_drawdown
    (2, 0.0, 0.2)
    """
    values = [float(value) for value in history]
    if not values:
        raise ValueError("history must contain at least one entry")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("history must contain only finite values")
    if any(value < 0 for value in values):
        raise ValueError("history must contain only nonnegative values")

    start, end = values[0], values[-1]
    bets = len(values) - 1

    total_return = end / start - 1 if start > 0 else None
    growth = None
    if bets > 0 and start > 0 and end > 0:
        growth = (end / start) ** (1 / bets) - 1

    peak = 0.0
    max_drawdown = None
    for value in values:
        peak = max(peak, value)
        if peak > 0:
            drawdown = (peak - value) / peak
            if max_drawdown is None or drawdown > max_drawdown:
                max_drawdown = drawdown

    return HistorySummary(
        bets=bets,
        start=start,
        end=end,
        total_return=total_return,
        geometric_growth_per_period=growth,
        max_drawdown=max_drawdown,
        ruined=end <= 0,
    )
