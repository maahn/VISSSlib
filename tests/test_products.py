import os
import shutil
import urllib.request
import zipfile

import numpy as np
import pytest
import VISSSlib
from VISSSlib.products import *

from helpers import readTestSettings

# class TestProducts:
#     @pytest.fixture(autouse=True)
#     def setup_files(self):
#         settings = "test_0.6/testtmp_0.6.yaml"
#         self.config = readTestSettings(settings)

#         try:
#             shutil.rmtree(self.config.tmpPath)
#         except FileNotFoundError:
#             pass
#         yield
#         # clean up
#         shutil.rmtree(self.config.tmpPath)

#     def test_processCases(self):
#         case = "20260110"

#         processCases(
#             case,
#             self.config,
#             ignoreErrors=False,
#             nJobs=2,
#             fileQueue=self.config.fileQueue,
#         )


class TestProducts:
    @pytest.mark.parametrize(
        "config_subpath, case",
        [
            ("test_0.6/testtmp_0.6.yaml", "20260110"),
            #            ("test_0.4/testtmp_0.4.yaml", "20260111"),
        ],
    )
    def test_processCases(self, config_subpath, case):
        # Read the config for this test case
        config = readTestSettings(config_subpath)
        # Ensure tmpPath is clean
        if os.path.exists(config.tmpPath):
            shutil.rmtree(config.tmpPath)
        try:
            processCases(
                case,
                config,
                ignoreErrors=False,
                nJobs=2,
                fileQueue=config.fileQueue,
            )
        finally:
            # Clean up tmpPath after test
            if os.path.exists(config.tmpPath):
                shutil.rmtree(config.tmpPath)
