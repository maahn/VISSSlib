import cv2
import numpy as np
import pytest
import VISSSlib

pytestmark = pytest.mark.unit


def _ramp(E, delay, width, dark=12.0, gain=0.9):
    return dark + gain * np.clip(np.asarray(E, float) - delay, 0, width)


def test_estimateExposureDelay():
    E = np.arange(10, 200, 10)
    B = _ramp(E, 45.0, 60.0) + np.random.default_rng(1).normal(0, 0.3, len(E))
    res = VISSSlib.commissioning.estimateExposureDelay(E, B)
    assert res["delay"] == pytest.approx(45.0, abs=4)
    assert res["duration"] == pytest.approx(60.0, abs=8)
    assert res["dark"] == pytest.approx(12.0, abs=1)
    res = VISSSlib.commissioning.estimateExposureDelay(E, B, flashDuration=60)
    assert res["delay"] == pytest.approx(45.0, abs=4)


def test_estimateExposureDelay_flat():
    with pytest.raises(ValueError):
        VISSSlib.commissioning.estimateExposureDelay(
            [10, 20, 30, 40], [10, 10, 10, 10]
        )


def test_analyzeExposureScan(tmp_path):
    files = {}
    for e in range(20, 200, 20):
        fn = tmp_path / f"exp{e}.png"
        val = int(_ramp(e, 50.0, 60.0))
        cv2.imwrite(str(fn), np.full((32, 32), val, dtype=np.uint8))
        files[e] = str(fn)
    res = VISSSlib.commissioning.analyzeExposureScan(files)
    assert res["delay"] == pytest.approx(50.0, abs=6)
