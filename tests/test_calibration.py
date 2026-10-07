import math

import numpy as np
import pytest

from keeks import CalibrationReport, calibration_report

P = [0.2, 0.2, 0.8, 0.8]
Y = [0, 1, 1, 1]


def test_hand_computed_fixture():
    r = calibration_report(P, Y)
    assert isinstance(r, CalibrationReport)
    assert r.n == 4
    assert r.brier_score == pytest.approx((0.04 + 0.64 + 0.04 + 0.04) / 4)
    expected = -(3 * math.log(0.8) + math.log(0.2)) / 4
    assert r.log_loss == pytest.approx(expected)
    assert r.base_rate == 0.75
    assert (r.min_clipped, r.max_clipped) == (False, False)
    assert [(b.lower, b.upper, b.count) for b in r.bins] == [
        (0.2, 0.3, 2),
        (0.8, 0.9, 2),
    ]
    assert r.bins[0].mean_predicted == pytest.approx(0.2)
    assert r.bins[0].observed_frequency == 0.5
    assert r.bins[1].mean_predicted == pytest.approx(0.8)
    assert r.bins[1].observed_frequency == 1.0


def test_perfectly_calibrated():
    p = [0.25] * 4 + [0.75] * 4
    y = [1, 0, 0, 0, 1, 1, 1, 0]
    r = calibration_report(p, y, n_bins=4)
    assert [(b.mean_predicted, b.observed_frequency) for b in r.bins] == [
        (0.25, 0.25),
        (0.75, 0.75),
    ]
    assert r.brier_score == pytest.approx(2 * (0.5625 + 3 * 0.0625) / 8)


def test_constant_half():
    r = calibration_report([0.5] * 4, [1, 0, 1, 0])
    assert r.brier_score == pytest.approx(0.25)
    assert r.log_loss == pytest.approx(math.log(2))
    assert len(r.bins) == 1
    assert r.bins[0].lower == 0.5 and r.bins[0].observed_frequency == 0.5


def test_extremes_clip_and_last_bin_includes_one():
    r = calibration_report([0.0, 1.0], [1, 0], n_bins=5)
    assert r.min_clipped and r.max_clipped
    assert r.log_loss == pytest.approx(-math.log(1e-12))
    assert [b.count for b in r.bins] == [1, 1]
    assert r.bins[-1].upper == 1.0 and r.bins[-1].mean_predicted == 1.0


def test_empty_bins_omitted_and_numpy_inputs():
    r = calibration_report(np.array([0.1, 0.9]), np.array([0, 1]), n_bins=np.int64(2))
    assert [(b.lower, b.count) for b in r.bins] == [(0.0, 1), (0.5, 1)]
    assert len(calibration_report([0.1], [True], n_bins=100).bins) == 1


@pytest.mark.parametrize(
    ("p", "y", "n_bins"),
    [
        ([float("nan")], [1], 10),
        ([float("inf")], [1], 10),
        ([1.5], [1], 10),
        ([-0.1], [1], 10),
        ([0.5, 0.5], [1], 10),
        ([0.5], [1.0], 10),
        ([0.5], [2], 10),
        ([0.5], ["a"], 10),
        ([], [], 10),
        (["x"], [1], 10),
        ([[0.5]], [1], 10),
        ([0.5], [1], 0),
        ([0.5], [1], -1),
        ([0.5], [1], 2.0),
        ([0.5], [1], True),
    ],
)
def test_invalid_inputs_raise(p, y, n_bins):
    with pytest.raises(ValueError):
        calibration_report(p, y, n_bins=n_bins)
