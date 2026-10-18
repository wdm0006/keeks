"""Load a recorded bet log from CSV."""

import csv
import math
import os
from pathlib import Path

_OUTCOMES = {"0": 0, "1": 1, "true": 1, "false": 0}


def load_bet_log(
    path: str | os.PathLike[str],
    *,
    probability_column: str = "probability",
    outcome_column: str = "outcome",
) -> tuple[list[float], list[int]]:
    """Read chronological probabilities and 0/1 outcomes from a CSV file.

    Row order is preserved. Probabilities must be finite and in ``[0, 1]``;
    outcomes may be ``0``, ``1``, ``true`` or ``false`` (case-insensitive).
    Any problem raises ``ValueError`` naming the file column and the 1-based
    data row (the header is not counted).

    Parameters
    ----------
    path : str or os.PathLike
        CSV file with a header row.
    probability_column, outcome_column : str
        Header names of the two columns to read.

    Returns
    -------
    tuple[list[float], list[int]]
        ``(probabilities, outcomes)``, ready for ``calibration_report``,
        ``HistoricalBinarySimulator`` or ``compare_strategies``.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     log = Path(folder) / "log.csv"
    ...     _ = log.write_text("probability,outcome\\n0.6,1\\n0.4,false\\n")
    ...     load_bet_log(log)
    ([0.6, 0.4], [1, 0])
    """
    probabilities = []
    outcomes = []
    with Path(path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        for column in (probability_column, outcome_column):
            if column not in (reader.fieldnames or []):
                raise ValueError(
                    f"{path}: column {column!r} not found; "
                    f"available columns: {list(reader.fieldnames or [])}"
                )
        for row_number, row in enumerate(reader, start=1):
            probabilities.append(
                _parse_probability(
                    row[probability_column], probability_column, row_number
                )
            )
            outcomes.append(
                _parse_outcome(row[outcome_column], outcome_column, row_number)
            )
    return probabilities, outcomes


def _parse_probability(raw: str | None, column: str, row_number: int) -> float:
    text = (raw or "").strip()
    where = f"column {column!r}, row {row_number}"
    if not text:
        raise ValueError(f"{where}: probability is empty")
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f"{where}: probability {text!r} is not a number") from None
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{where}: probability {text!r} must be finite in [0, 1]")
    return value


def _parse_outcome(raw: str | None, column: str, row_number: int) -> int:
    text = (raw or "").strip()
    try:
        return _OUTCOMES[text.lower()]
    except KeyError:
        raise ValueError(
            f"column {column!r}, row {row_number}: outcome {text!r} must be "
            "0, 1, true or false"
        ) from None
