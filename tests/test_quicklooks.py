import numpy as np
import pytest
from VISSSlib.quicklooks import *

from helpers import get_test_data_path, get_test_path, readTestSettings


class TestQuicklooks(object):
    @pytest.fixture(autouse=True)
    def setup_files(self):
        self.config = readTestSettings("test_0.6/test_0.6.yaml")
        self.testPath = get_test_data_path()
        yield
        shutil.rmtree(self.config.tmpPath)

    def testLoop(self):
        case = "20260110"
        for level in [
            "level0",
            "level1detect",
            "metaFrames",
            "level2detect",
            "level1match",
            "level2match",
            "level2track",
            # "level3combinedRiming",
        ]:
            res = loop(level, case, self.config, skipExisting=False)
            try:
                res = res[0]
            except TypeError:
                res = res
            assert res is not None
