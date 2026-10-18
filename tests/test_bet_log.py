import pytest

import keeks
from keeks import load_bet_log


def write(tmp_path, text):
    path = tmp_path / "log.csv"
    path.write_text(text, encoding="utf-8")
    return path


def test_known_csv_exact_values_and_order(tmp_path):
    path = write(tmp_path, "date,probability,outcome\nd1,0.55,1\nd2,0.4,0\nd3,1,1\n")
    assert load_bet_log(path) == ([0.55, 0.4, 1.0], [1, 0, 1])


def test_custom_columns(tmp_path):
    path = write(tmp_path, "p,won,probability\n0.7,1,0.1\n0.2,0,0.9\n")
    result = load_bet_log(path, probability_column="p", outcome_column="won")
    assert result == ([0.7, 0.2], [1, 0])


def test_bool_words_case_insensitive(tmp_path):
    path = write(tmp_path, "probability,outcome\n0.5,TRUE\n0.5,False\n0.5, true \n")
    assert load_bet_log(path)[1] == [1, 0, 1]


def test_header_only_is_empty(tmp_path):
    assert load_bet_log(write(tmp_path, "probability,outcome\n")) == ([], [])


def test_exported_and_listed():
    assert "load_bet_log" in keeks.__all__


@pytest.mark.parametrize(
    ("text", "kwargs", "needles"),
    [
        ("probability,result\n0.5,1\n", {}, ["'outcome'", "not found"]),
        ("p,outcome\n0.5,1\n", {}, ["'probability'", "not found"]),
        ("", {}, ["'probability'"]),
        ("probability,outcome\n0.5,1\n,0\n", {}, ["'probability'", "row 2", "empty"]),
        ("probability,outcome\n0.5,1\n0.5\n", {}, ["'outcome'", "row 2"]),
        ("probability,outcome\nnan,1\n", {}, ["'probability'", "row 1"]),
        ("probability,outcome\n0.5,1\n0.5,1\ninf,1\n", {}, ["row 3", "'inf'"]),
        ("probability,outcome\n-inf,1\n", {}, ["row 1"]),
        ("probability,outcome\n1.5,1\n", {}, ["'probability'", "row 1", "1.5"]),
        ("probability,outcome\n0.5,1\n-0.1,1\n", {}, ["row 2", "-0.1"]),
        ("probability,outcome\nabc,1\n", {}, ["'probability'", "row 1", "'abc'"]),
        ("probability,outcome\n0.5,2\n", {}, ["'outcome'", "row 1", "'2'"]),
        ("probability,outcome\n0.5,1\n0.5,1.0\n", {}, ["'outcome'", "row 2", "'1.0'"]),
        ("probability,outcome\n0.5,\n", {}, ["'outcome'", "row 1"]),
        (
            "p,won\n0.5,1\n0.5,yes\n",
            {"probability_column": "p", "outcome_column": "won"},
            ["'won'", "row 2", "'yes'"],
        ),
    ],
)
def test_errors_name_column_and_row(tmp_path, text, kwargs, needles):
    with pytest.raises(ValueError) as excinfo:
        load_bet_log(write(tmp_path, text), **kwargs)
    for needle in needles:
        assert needle in str(excinfo.value)


def test_feeds_calibration_report(tmp_path):
    path = write(tmp_path, "probability,outcome\n0.2,0\n0.2,1\n0.8,1\n0.8,1\n")
    report = keeks.calibration_report(*load_bet_log(path))
    assert round(report.brier_score, 6) == 0.19
