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

    def test_processCases(self):
        case = "20260110"

        processCases(
            case,
            self.config,
            ignoreErrors=False,
            nJobs=2,
            fileQueue=self.config.fileQueue,
        )
