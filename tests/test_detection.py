import numpy as np
import pytest
from VISSSlib.detection import *

from helpers import get_test_data_path, get_test_path, readTestSettings


class TestRoi(object):
    @pytest.mark.unit
    def test_roi(self):
        img = np.random.random((100, 100))

        for xr in range(0, 40):
            for yr in range(0, 40):
                roi = (xr, yr, 40, 40)
                imgE, xo, yo, _ = extractRoi(roi, img)

                imgE1, xo, yo, extraROI = extractRoi(roi, img, extra=20)
                imgE2, _, _, _ = extractRoi(extraROI, imgE1)

                assert np.all(imgE2 == imgE), (xr, yr)


class TestCheckMotion:
    """Unit tests for detection.checkMotion, a pure function of two frames
    and threshold values (no cv2 pipeline/video files needed).
    """

    @pytest.mark.unit
    def test_counts_pixels_above_each_threshold(self):
        sub = np.zeros((5, 5), dtype=np.uint8)
        old = np.zeros((5, 5), dtype=np.uint8)
        sub[2, 2] = 100
        result = checkMotion(sub, old, [10, 50, 150])
        assert list(result) == [1, 1, 0]

    @pytest.mark.unit
    def test_no_oldFrame_defaults_to_zeros(self):
        sub = np.zeros((5, 5), dtype=np.uint8)
        sub[2, 2] = 100
        assert list(checkMotion(sub, None, [10, 50, 150])) == list(
            checkMotion(sub, np.zeros((5, 5), dtype=np.uint8), [10, 50, 150])
        )

    @pytest.mark.unit
    def test_no_change_below_threshold(self):
        sub = np.full((4, 4), 5, dtype=np.uint8)
        old = np.zeros((4, 4), dtype=np.uint8)
        assert list(checkMotion(sub, old, [10])) == [0]


class TestJoinEdges:
    """Unit tests for detection.joinEdges, a pure function of a binary
    mask array that bridges small gaps between edge pixels.
    """

    @pytest.mark.unit
    def test_all_zero_mask_unchanged(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        result = joinEdges(mask)
        assert np.all(result == 0)

    @pytest.mark.unit
    def test_input_not_mutated(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[3, 3] = 255
        mask[3, 4] = 255
        original = mask.copy()
        joinEdges(mask)
        assert np.array_equal(mask, original)

    @pytest.mark.unit
    def test_bridges_small_gap_between_edge_endpoints(self):
        # two short diagonal edge segments whose endpoints are 3px apart
        # along the same row; joinEdges should fill the gap between them
        mask = np.zeros((15, 15), dtype=np.uint8)
        mask[3, 3] = 255
        mask[4, 4] = 255
        mask[5, 5] = 255
        mask[5, 8] = 255
        mask[6, 9] = 255
        mask[7, 10] = 255

        result = joinEdges(mask)

        before = set(map(tuple, np.argwhere(mask)))
        after = set(map(tuple, np.argwhere(result)))
        assert after - before == {(5, 6), (5, 7)}


class TestSplitUpContours:
    """Unit tests for detection.splitUpConours, a pure function that
    regroups a cv2.findContours-style (contours, hierarchy) pair into
    top-level contours and their descendants, using synthetic
    placeholder "contours" and hand-built hierarchy arrays.
    """

    @pytest.mark.unit
    def test_all_top_level_contours_pass_through_unchanged(self):
        cntsTmp = ["A", "B", "C"]
        # hierarchy columns are [next, previous, first_child, parent]
        hierarchy = np.array([[[-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]]])
        cnts, cntChildren = splitUpConours(cntsTmp, hierarchy)
        assert cnts == ["A", "B", "C"]
        assert cntChildren == [[], [], []]

    @pytest.mark.unit
    def test_single_child_grouped_under_its_parent(self):
        cntsTmp = ["A", "B"]
        # B's parent is index 0 (A)
        hierarchy = np.array([[[-1, -1, -1, -1], [-1, -1, -1, 0]]])
        cnts, cntChildren = splitUpConours(cntsTmp, hierarchy)
        assert cnts == ["A"]
        assert cntChildren == [["B"]]

    @pytest.mark.unit
    def test_grandchild_attributed_to_topmost_parent(self):
        cntsTmp = ["A", "B", "C"]
        # B's parent is A (index 0); C's parent is B (index 1)
        hierarchy = np.array(
            [[[-1, -1, -1, -1], [-1, -1, -1, 0], [-1, -1, -1, 1]]]
        )
        cnts, cntChildren = splitUpConours(cntsTmp, hierarchy)
        assert cnts == ["A"]
        assert cntChildren == [["B", "C"]]


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
