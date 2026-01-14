import numpy as np
import pytest
from VISSSlib.metadata import *

from helpers import get_test_data_path, get_test_path, readTestSettings


class TestMeta(object):
    @pytest.fixture(autouse=True)
    def setup_files(self):
        self.config = readTestSettings("test_0.6/test_0.6.yaml")
        self.testPath = get_test_data_path()
        yield

    def testEvents(self):
        case = "20260110"
        dat = createEvent(
            case, "leader", self.config, skipExisting=False, writeNc=False
        )
        assert np.isclose(dat.cameraTemperature.mean(), 35.8633728)
        for var in [
            "blocking",
            "brightnessMean",
            "brightnessStd",
            "cameraTemperature",
            "capture_firsttime",
            "capture_lasttime",
            "capture_starttime",
            "configuration",
            "event",
            "filename",
            "gitBranch",
            "gitTag",
            "hostname",
            "ptpStatus",
            "serialnumber",
            "transferMaxBlockSize",
            "transferQueueCurrentBlockCount",
        ]:
            assert var in dat.data_vars

    def testMetaFrames(self):
        case = "20260110"
        dat = createMetaFrames(
            case, "leader", self.config, skipExisting=False, writeNc=False
        )

        assert np.isclose(dat.nMovingPixel.mean(), 61.05675239)
        for var in [
            "capture_id",
            "nMovingPixel",
            "nThread",
            "queue_size",
            "record_id",
            "record_time",
        ]:
            assert var in dat.data_vars
