import os
import shutil
import urllib.request
import zipfile

import numpy as np
import pytest
import VISSSlib
from VISSSlib.products import *

from helpers import downloadData, get_test_data_path, get_test_path, readTestSettings


class TestProducts:
    @pytest.fixture(autouse=True)
    def setup_files(self):
        settings = "test_0.6/testtmp_0.6.yaml"
        self.config = readTestSettings(settings)

        try:
            shutil.rmtree(self.config.tmpPath)
        except FileNotFoundError:
            pass
        yield
        # clean up
        shutil.rmtree(self.config.tmpPath)

    def test_products(self):
        case = "20260110"

        products = [
            "metaEvents",
            "level1detect",
            "metaRotation",
            "level1match",
            "level1track",
            "level2detect",
            "level2match",
            "level2track",
            "allDone",
        ]
        followerProducts = ["metaEvents", "level1detect", "level2detect"]
        for prod in products:
            print("#" * 10, prod, "#" * 10)
            dp1 = DataProduct(prod, case, self.config, self.config.fileQueue, "leader")
            dp1.submitCommands(withParents=False)
            assert len(dp1.commands) > 0
            if prod in followerProducts:
                dp2 = DataProduct(
                    prod, case, self.config, self.config.fileQueue, "follower"
                )
                dp2.submitCommands(withParents=False)
                assert len(dp2.commands) > 0
            VISSSlib.tools.workers(self.config.fileQueue, waitTime=1, nJobs=2)
            assert len(dp1.listBroken()) == 0
            assert len(dp1.listFiles()) > 0
            if prod in followerProducts:
                assert len(dp2.listBroken()) == 0
                assert len(dp2.listFiles()) > 0
