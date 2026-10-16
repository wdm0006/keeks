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
    assert (tmp_path / "replay_bet_log.json").read_bytes() == (
        EXAMPLE.OUTPUT_DIR / "replay_bet_log.json"
    ).read_bytes()
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


def test_json_report_matches_apis_and_known_values(tmp_path):
    import json
    from dataclasses import asdict

    from keeks import bootstrap_comparison, calibration_report, compare_strategies

    source = tmp_path / "known.csv"
    source.write_text("probability,outcome\n0.25,0\n0.75,1\n", encoding="utf-8")
    EXAMPLE.main(source, tmp_path)
    destination = tmp_path / "replay_bet_log.json"
    first = destination.read_bytes()
    report = json.loads(first)
    probabilities, outcomes = EXAMPLE.load_bet_log(source)
    factories = EXAMPLE.strategy_factories()

    def normalize(value):
        return json.loads(json.dumps(value, allow_nan=False))

    assert report["settings"] == {
        "input_rows": 2,
        **EXAMPLE.ECONOMICS,
        "transaction_cost_rate": 0.0,
        "sort_by": EXAMPLE.SORT_BY,
        "n_bins": 10,
        **EXAMPLE.BOOTSTRAP,
    }
    assert report["calibration"] == normalize(
        asdict(calibration_report(probabilities, outcomes))
    )
    assert report["calibration"]["brier_score"] == 0.0625
    assert report["calibration"]["log_loss"] == pytest.approx(0.2876820724517809)
    assert report["calibration"]["base_rate"] == 0.5
    assert [b["mean_predicted"] for b in report["calibration"]["bins"]] == [0.25, 0.75]
    assert [b["observed_frequency"] for b in report["calibration"]["bins"]] == [
        0.0,
        1.0,
    ]
    assert [b["count"] for b in report["calibration"]["bins"]] == [1, 1]
    assert report["rankings"] == normalize(
        [
            asdict(row)
            for row in compare_strategies(
                factories,
                probabilities,
                outcomes,
                **EXAMPLE.ECONOMICS,
                sort_by=EXAMPLE.SORT_BY,
            )
        ]
    )
    assert [(r["name"], r["summary"]["end"]) for r in report["rankings"]] == [
        ("Kelly", 1500.0),
        ("Half Kelly", 1250.0),
        ("Fixed 5%", 1050.0),
    ]
    assert report["uncertainty"] == normalize(
        [
            asdict(row)
            for row in bootstrap_comparison(
                factories,
                probabilities,
                outcomes,
                **EXAMPLE.ECONOMICS,
                sort_by=EXAMPLE.SORT_BY,
                **EXAMPLE.BOOTSTRAP,
            )
        ]
    )
    EXAMPLE.main(source, tmp_path)
    assert destination.read_bytes() == first


def test_all_declined_report_has_nulls_and_explicit_counts(tmp_path):
    import json

    source = tmp_path / "declined.csv"
    source.write_text("probability,outcome\n0.25,0\n0.25,1\n", encoding="utf-8")
    EXAMPLE.main(source, tmp_path)
    report = json.loads((tmp_path / "replay_bet_log.json").read_text())
    for row in report["rankings"]:
        assert row["summary"]["bets"] == 0
        assert row["summary"]["end"] == 1000.0
        assert row["summary"]["geometric_growth_per_period"] is None
    for rank, row in enumerate(report["uncertainty"]):
        assert row["point"] is row["lower"] is row["upper"] is None
        assert row["undefined"] == 200
        assert row["rank_counts"] == [200 if i == rank else 0 for i in range(3)]
