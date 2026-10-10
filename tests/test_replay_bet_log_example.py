"""Value and output contracts for the recorded-log example."""

import csv
import importlib.util
from pathlib import Path

import pytest

EXAMPLE_PATH = Path(__file__).resolve().parents[1] / "examples" / "replay_bet_log.py"
spec = importlib.util.spec_from_file_location("replay_bet_log", EXAMPLE_PATH)
EXAMPLE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(EXAMPLE)


@pytest.mark.parametrize(
    "name,end,drawdown",
    [
        ("Kelly", 610.96, 0.7805768701804972),
        ("Half Kelly", 930.98, 0.48380001132866346),
        ("Fixed 5%", 927.65, 0.42332380952380955),
    ],
)
def test_committed_log_summary(name, end, drawdown):
    results, histories = EXAMPLE.replay_bet_log()
    row = next(row for row in results if row["strategy"] == name)
    assert row["input_rows"] == row["bets"] == 60
    assert row["start"] == histories[name][0] == 1000.0
    assert row["end"] == histories[name][-1] == end
    assert row["max_drawdown"] == drawdown
    assert row["total_return"] == end / 1000.0 - 1
    assert row["geometric_growth_per_period"] == (end / 1000.0) ** (1 / 60) - 1
    assert row["ruined"] is False


def test_csv_is_reproducible_and_matches_committed_output(tmp_path):
    EXAMPLE.main(output_dir=tmp_path)
    destination = tmp_path / "replay_bet_log.csv"
    first = destination.read_bytes()
    EXAMPLE.main(output_dir=tmp_path)
    assert destination.read_bytes() == first
    assert first == (EXAMPLE.OUTPUT_DIR / destination.name).read_bytes()
    assert (
        (tmp_path / "replay_bet_log.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    )


def test_custom_log_is_used(tmp_path):
    source = tmp_path / "bets.csv"
    source.write_text("probability,outcome\n0.40,1\n0.60,1\n0.60,0\n", encoding="utf-8")
    EXAMPLE.main(csv_path=source, output_dir=tmp_path / "output")
    with (tmp_path / "output" / "replay_bet_log.csv").open(newline="") as output:
        rows = list(csv.DictReader(output))
    assert [row["end"] for row in rows] == ["960.0", "990.0", "997.5"]
    assert [row["input_rows"] for row in rows] == ["3"] * 3
    assert [row["bets"] for row in rows] == ["2"] * 3
