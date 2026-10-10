from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from keeks.simulators.historical_binary import _validate_outcome

LOG_LOSS_CLIP = 1e-12


@dataclass(frozen=True)
class CalibrationBin:
    """
    One non-empty equal-width probability bin.

    Attributes
    ----------
    lower, upper : float
        Bin edges. Bins include their lower edge; only the last bin also
        includes ``1.0``.
    count : int
        Observations in the bin.
    mean_predicted : float
        Mean stated probability in the bin.
    observed_frequency : float
        Fraction of the bin's outcomes that were wins.
    """

    lower: float
    upper: float
    count: int
    mean_predicted: float
    observed_frequency: float


@dataclass(frozen=True)
class CalibrationReport:
    """
    Calibration diagnostics for recorded probabilities and outcomes.

    Attributes
    ----------
    n : int
        Number of observations.
    brier_score : float
        Mean squared error between probability and outcome.
    log_loss : float
        Mean negative log-likelihood, with probabilities clipped to
        ``[1e-12, 1 - 1e-12]``.
    min_clipped, max_clipped : bool
        Whether any probability fell below / above the clip bounds, so the
        log loss reflects the bound rather than the stated value.
    base_rate : float
        Fraction of outcomes that were wins.
    bins : tuple of CalibrationBin
        Non-empty bins in ascending order; empty bins are omitted.
    """

    n: int
    brier_score: float
    log_loss: float
    min_clipped: bool
    max_clipped: bool
    base_rate: float
    bins: tuple[CalibrationBin, ...]


def calibration_report(
    probabilities: Sequence[float],
    outcomes: Sequence[bool | int],
    n_bins: int = 10,
) -> CalibrationReport:
    """
    Check whether recorded win probabilities were calibrated.

    Strategies size from a stated probability, so a miscalibrated one is the
    main way Kelly-family sizing loses money. This is an educational
    diagnostic of past data, not advice: a good score on a short record does
    not show the probabilities will stay calibrated.

    Parameters
    ----------
    probabilities : sequence of float
        Stated win probability of each event, each finite and in ``[0, 1]``.
    outcomes : sequence of bool or int
        Whether each event won: ``bool`` or ``0``/``1`` only (floats such as
        ``1.0`` are rejected), same length as ``probabilities``.
    n_bins : int, default 10
        Number of equal-width bins over ``[0, 1]``.

    Returns
    -------
    CalibrationReport
        Frozen dataclass of Brier score, log loss, base rate and per-bin
        stated-versus-observed frequencies.

    Raises
    ------
    ValueError
        If an input is invalid, the lengths differ, there are no
        observations, or ``n_bins`` is not a positive integer.

    Examples
    --------
    >>> report = calibration_report([0.2, 0.2, 0.8, 0.8], [0, 1, 1, 1])
    >>> report.n, round(report.brier_score, 6), report.base_rate
    (4, 0.19, 0.75)
    >>> [(b.lower, b.count, b.observed_frequency) for b in report.bins]
    [(0.2, 2, 0.5), (0.8, 2, 1.0)]
    """
    if isinstance(n_bins, bool) or not isinstance(n_bins, Integral) or n_bins < 1:
        raise ValueError(f"n_bins must be a positive integer, got {n_bins!r}")
    n_bins = int(n_bins)
    try:
        stated = np.asarray(probabilities, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Probabilities must be a finite sequence") from exc
    if stated.ndim != 1:
        raise ValueError("Probabilities must be one-dimensional")
    if not np.all(np.isfinite(stated)):
        raise ValueError("Probabilities must contain only finite values")
    if np.any((stated < 0) | (stated > 1)):
        raise ValueError("Probabilities must be between 0 and 1")
    if len(stated) != len(outcomes):
        raise ValueError(
            "Probabilities and outcomes must have the same length, got "
            f"{len(stated)} and {len(outcomes)}"
        )
    if len(stated) == 0:
        raise ValueError("At least one observation is required")
    won = np.array(
        [_validate_outcome(o, i) for i, o in enumerate(outcomes)], dtype=float
    )

    clipped = np.clip(stated, LOG_LOSS_CLIP, 1 - LOG_LOSS_CLIP)
    log_loss = -float(np.mean(won * np.log(clipped) + (1 - won) * np.log1p(-clipped)))
    bin_index = np.minimum(np.floor(stated * n_bins).astype(int), n_bins - 1)
    bins = tuple(
        CalibrationBin(
            lower=k / n_bins,
            upper=(k + 1) / n_bins,
            count=int(mask.sum()),
            mean_predicted=float(stated[mask].mean()),
            observed_frequency=float(won[mask].mean()),
        )
        for k in range(n_bins)
        if (mask := bin_index == k).any()
    )
    return CalibrationReport(
        n=len(stated),
        brier_score=float(np.mean((stated - won) ** 2)),
        log_loss=log_loss,
        min_clipped=bool(np.any(stated < LOG_LOSS_CLIP)),
        max_clipped=bool(np.any(stated > 1 - LOG_LOSS_CLIP)),
        base_rate=float(won.mean()),
        bins=bins,
    )


__all__ = ["CalibrationBin", "CalibrationReport", "calibration_report"]
