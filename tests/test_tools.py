import os

import numpy as np
import pytest
import VISSSlib

from helpers import get_test_data_path


def test_readConfig():
    testRoot = get_test_data_path()
    config = VISSSlib.tools.readSettings(f"{testRoot}/test_0.6/test_0.6.yaml")
    assert config.visssGen == "visss"


def test_block_archive(tmp_path):
    """Test block archive operations"""
    archive_file = tmp_path / "test.block"

    # Create and write to archive
    with VISSSlib.tools.BlockImageArchive(archive_file, mode="w") as archive:
        img1 = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
        archive.addnpy(f"image_1", img1)

    # Read from archive
    with VISSSlib.tools.BlockImageArchive(archive_file, mode="r") as archive:
        for i in range(10):
            img2 = archive.extractnpy(f"image_1")
            assert img2.shape == (10, 10)
            assert img2.dtype == np.uint8
            assert np.all(img1 == img2)
