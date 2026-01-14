import os
import shutil
import urllib.request
import zipfile

import numpy as np
import pytest
import VISSSlib
from VISSSlib.products import *

from helpers import get_test_data_path, get_test_path, readTestSettings


def processCase(case, config):
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
        dp1 = DataProduct(prod, case, config, config.fileQueue, "leader")
        dp1.submitCommands(withParents=False)
        assert len(dp1.commands) > 0
        if prod in followerProducts:
            dp2 = DataProduct(prod, case, config, config.fileQueue, "follower")
            dp2.submitCommands(withParents=False)
            assert len(dp2.commands) > 0
        VISSSlib.tools.workers(config.fileQueue, waitTime=1, nJobs=2)
        assert len(dp1.listBroken()) == 0
        assert len(dp1.listFiles()) > 0
        if prod in followerProducts:
            assert len(dp2.listBroken()) == 0
            assert len(dp2.listFiles()) > 0
    return


class TestProducts:
    @pytest.fixture(autouse=True)
    def setup_files(self):
        test_path = get_test_path()
        test_0_6_yaml = os.path.join(test_path, "data", "test_0.6", "test_0.6.yaml")
        if not os.path.exists(test_0_6_yaml):
            url = "https://speicherwolke.uni-leipzig.de/public.php/dav/files/PJ8dt77ND9tmaB2/?accept=zip"
            zip_path = os.path.join(test_path, "data.zip")
            print(f"Downloading test data from {url} to {zip_path}")
            urllib.request.urlretrieve(url, zip_path)
            print(f"Extracting test data to {test_path}")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(test_path)
            print(f"Removing zip file {zip_path}")
            os.remove(zip_path)

        self.config = readTestSettings("test_0.6/test_0.6.yaml")
        try:
            shutil.rmtree(self.config.tmpPath)
        except FileNotFoundError:
            pass
        yield
        # clean up
        shutil.rmtree(self.config.tmpPath)

    def test_products(self):
        case = "20260110"
        processCase(case, self.config)


# -m VISSSlib metadata.createEvent /Users/mmaahn/projectsSrv/VISSSlib/tests/data/test_0.6/test_0.6.yaml leader+20260110 1
# -m VISSSlib metadata.createEvent /Users/mmaahn/projectsSrv/VISSSlib/tests/data/test_0.6/test_0.6.yaml follower+20260110 1
# -m VISSSlib detection.detectParticles  /Users/mmaahn/projectsSrv/VISSSlib/tests/data/test_0.6/rawdata/level0/visss11gb_visss_follower_S1143155/2026/01/10/visss11gb_visss_follower_S1143155_20260110-083000_0.txt /Users/mmaahn/projectsSrv/VISSSlib/tests/data/test_0.6/test_0.6.yaml 1
# -m VISSSlib detection.detectParticles  /Users/mmaahn/projectsSrv/VISSSlib/tests/data/test_0.6/rawdata/level0/visss11gb_visss_leader_S1145792/2026/01/10/visss11gb_visss_leader_S1145792_20260110-083000_0.txt /Users/mmaahn/projectsSrv/VISSSlib/tests/data/test_0.6/test_0.6.yaml 1
# -m VISSSlib matching.createMetaRotation /Users/mmaahn/projectsSrv/VISSSlib/tests/data/test_0.6/test_0.6.yaml 20260110 1
# -m VISSSlib matching.matchParticles  /Users/mmaahn/projectsSrv/VISSSlib/tests/data/test_0.6/tmp_Mac.fritz.box/products/level1detect/2026/01/10/level1detect_V1.2_test_visss11gb_visss_leader_S1145792_20260110-083000.nc /Users/mmaahn/projectsSrv/VISSSlib/tests/data/test_0.6/test_0.6.yaml
# -m VISSSlib matching.matchParticles  /Users/mmaahn/projectsSrv/VISSSlib/tests/data/test_0.6/tmp_Mac.fritz.box/products/level1detect/2026/01/10/level1detect_V1.2_test_visss11gb_visss_leader_S1145792_20260110-083000.nc /Users/mmaahn/projectsSrv/VISSSlib/tests/data/test_0.6/test_0.6.yaml 1
