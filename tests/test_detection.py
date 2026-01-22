import numpy as np
import pytest
from VISSSlib.detection import *

from helpers import get_test_data_path, get_test_path, readTestSettings


class TestRoi(object):
    def test_roi(self):
        img = np.random.random((100, 100))

        for xr in range(0, 40):
            for yr in range(0, 40):
                roi = (xr, yr, 40, 40)
                imgE, xo, yo, _ = extractRoi(roi, img)

                imgE1, xo, yo, extraROI = extractRoi(roi, img, extra=20)
                imgE2, _, _, _ = extractRoi(extraROI, imgE1)

                assert np.all(imgE2 == imgE), (xr, yr)


class TestDetection(object):
    @pytest.fixture(autouse=True)
    def setup_files(self):
        self.config = readTestSettings("test_0.6/test_0.6.yaml")
        self.testPath = get_test_data_path()
        yield

    def testL1Detect(self):
        fname = f"{self.testPath}/test_0.6/rawdata/level0/visss11gb_visss_leader_S1145792/2026/01/10/visss11gb_visss_leader_S1145792_20260110-083000_0.txt"
        dat = detectParticles(fname, self.config, writeNc=False, skipExisting=False)

        for var in [
            "Dfit",
            "Dmax",
            "Droi",
            "angle",
            "area",
            "areaConsideringHoles",
            "aspectRatio",
            "blur",
            "capture_id",
            "capture_time",
            "contourFFT",
            "contourFFTstd",
            "contourFFTsum",
            "extent",
            "extentConsideringHoles",
            "nThread",
            "perimeter",
            "perimeterConsideringHoles",
            "perimeterEroded",
            "pixCenter",
            "pixKurtosis",
            "pixMax",
            "pixMean",
            "pixMin",
            "pixPercentiles",
            "pixSkew",
            "pixStd",
            "position_centroid",
            "position_circle",
            "position_fit",
            "position_upperLeft",
            "record_id",
            "record_time",
            "solidity",
            "solidityConsideringHoles",
        ]:
            assert var in dat.data_vars
        assert np.isclose(dat.Dmax.mean(), 6.45963144, rtol=1e-3)
        assert np.isclose(dat.area.mean(), 42.46416092, rtol=1e-3)
        assert np.isclose(dat.perimeter.mean(), 18.66514397, rtol=1e-3)
        assert np.isclose(dat.contourFFT.mean(), 1.39864063, rtol=1e-3)
