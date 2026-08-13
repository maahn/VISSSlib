import os
import shutil
import urllib.request
import zipfile

import numpy as np
import pytest
import VISSSlib
from VISSSlib.products import *

from helpers import makeSyntheticConfig, readTestSettings


class TestProducts:
    @pytest.mark.parametrize(
        "config_subpath, case",
        [
            ("test_0.6/testtmp_0.6.yaml", "20260110"),
            #            ("test_0.4/testtmp_0.4.yaml", "20260111"),
        ],
    )
    def test_processAll(self, config_subpath, case):
        # Read the config for this test case
        config = readTestSettings(config_subpath)
        # Ensure tmpPath is clean
        if os.path.exists(config.tmpPath):
            shutil.rmtree(config.tmpPath)
        try:
            processAll(
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


class TestDataProductDAG:
    """Unit tests for DataProduct's dependency-graph wiring (parentNames,
    the processL1match guard, isComplete). These use a fully synthetic,
    network-free config (see helpers.makeSyntheticConfig) rather than the
    downloaded sample dataset, since they only exercise control flow, not
    the actual science code.
    """

    @pytest.fixture
    def config(self, tmp_path):
        return makeSyntheticConfig(tmp_path)

    @pytest.fixture
    def queue(self, tmp_path):
        return str(tmp_path / "fileQueue")

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "level, camera, expectedParents",
        [
            ("level0", "leader", []),
            ("level0txt", "leader", []),
            ("metaEvents", "leader", ["leader_level0txt"]),
            ("metaFrames", "follower", ["follower_level0txt"]),
            ("level1detect", "leader", []),
            (
                "metaRotation",
                "leader",
                [
                    "leader_level1detect",
                    "follower_level1detect",
                    "leader_metaEvents",
                    "follower_metaEvents",
                ],
            ),
            ("level1match", "leader", ["leader_metaRotation"]),
            ("level1track", "leader", ["leader_level1match"]),
            (
                "level2detect",
                "follower",
                ["follower_level1detect", "follower_metaEvents"],
            ),
            (
                "level2match",
                "leader",
                ["leader_level1match", "leader_metaEvents", "follower_metaEvents"],
            ),
            (
                "level2track",
                "leader",
                ["leader_level1track", "leader_metaEvents", "follower_metaEvents"],
            ),
            (
                "level3combinedRiming",
                "leader",
                ["leader_level2track", "leader_metaEvents", "follower_metaEvents"],
            ),
        ],
    )
    def test_parentNames(self, config, queue, level, camera, expectedParents):
        p = DataProduct(level, "20260101", config, queue, camera, addRelatives=False)
        assert p.parentNames == expectedParents

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "processL1match, processL2detect, processRetrieval, expectedExtra",
        [
            (
                True,
                True,
                True,
                [
                    "leader_level2track",
                    "leader_level2match",
                    "leader_level2detect",
                    "follower_level2detect",
                    "leader_level3combinedRiming",
                ],
            ),
            (False, False, False, []),
            (True, False, False, ["leader_level2track", "leader_level2match"]),
        ],
    )
    def test_allDone_parentNames_respects_toggles(
        self,
        config,
        queue,
        processL1match,
        processL2detect,
        processRetrieval,
        expectedExtra,
    ):
        config.level1match.processL1match = processL1match
        config.level2.processL2detect = processL2detect
        config.level3.combinedRiming.processRetrieval = processRetrieval
        p = DataProduct("allDone", "20260101", config, queue, "leader", addRelatives=False)
        assert p.parentNames == [
            "leader_metaEvents",
            "follower_metaEvents",
        ] + expectedExtra

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "level",
        ["metaRotation", "level1match", "level1track", "level2match", "level2track"],
    )
    def test_processL1match_false_blocks_matching_levels(self, config, queue, level):
        config.level1match.processL1match = False
        with pytest.raises(ValueError, match="processL1match"):
            DataProduct(level, "20260101", config, queue, "leader", addRelatives=False)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "level, camera",
        [
            ("level1detect", "leader"),
            ("level2detect", "leader"),
            ("metaEvents", "leader"),
            ("allDone", "leader"),
        ],
    )
    def test_processL1match_false_does_not_block_other_levels(
        self, config, queue, level, camera
    ):
        config.level1match.processL1match = False
        # should not raise
        DataProduct(level, "20260101", config, queue, camera, addRelatives=False)

    @pytest.mark.unit
    def test_isComplete_false_when_no_files_exist(self, config, queue):
        p = DataProduct(
            "metaEvents", "20260101", config, queue, "leader", addRelatives=False
        )
        assert p.isComplete is False

    @pytest.mark.unit
    def test_isComplete_true_once_output_file_exists(self, config, queue):
        p = DataProduct(
            "metaEvents", "20260101", config, queue, "leader", addRelatives=False
        )
        outFile = p.fn.fnamesDaily["metaEvents"]
        os.makedirs(os.path.dirname(outFile), exist_ok=True)
        open(outFile, "w").close()
        assert p.isComplete is True

    @pytest.mark.unit
    def test_camera_must_be_leader_or_follower(self, config, queue):
        with pytest.raises(ValueError, match="camera"):
            DataProduct(
                "level1detect", "20260101", config, queue, "sideways", addRelatives=False
            )
