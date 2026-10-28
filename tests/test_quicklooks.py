import os
import shutil

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
            res = generate(case, self.config, level, skipExisting=False)
            try:
                res = res[0]
            except TypeError:
                res = res
            assert res is not None


class TestTrackParticlesUnit(object):
    """Unit tests of the helpers of createLevel1trackParticles (synthetic
    data, no sample data needed).
    """

    @pytest.mark.unit
    def test_padToCommonShape(self):
        from VISSSlib import quicklooks

        a = np.zeros((4, 6), dtype=np.uint8)
        b = np.zeros((7, 3), dtype=np.uint8)
        padded = quicklooks._padToCommonShape([a, b], padding_value=150)
        assert all(p.shape == (7, 6) for p in padded)
        # original content is centered, the rest is padding
        assert padded[0][0, 0] == 150
        assert padded[0][3, 0] == 0
        assert padded[1][0, 0] == 150

    @pytest.mark.unit
    def test_trackParticlesSetting(self, tmp_path):
        from helpers import makeSyntheticConfig
        from VISSSlib import quicklooks

        config = makeSyntheticConfig(tmp_path)
        # defaults come from tools.DEFAULT_SETTINGS
        assert quicklooks._trackParticlesSetting(config, "targetGifs", "config") == 12
        assert quicklooks._trackParticlesSetting(config, "weightDmax", "config") == 3
        # explicit values are not touched
        assert quicklooks._trackParticlesSetting(config, "targetGifs", 3) == 3
        # a yaml value overrides the default
        config = makeSyntheticConfig(
            tmp_path, level1trackParticlesQuicklook=dict(targetGifs=7)
        )
        assert quicklooks._trackParticlesSetting(config, "targetGifs", "config") == 7
        assert quicklooks._trackParticlesSetting(config, "minSize", "config") == 20

    @pytest.mark.unit
    def test_trackParticlesQuicklookPath(self, tmp_path):
        from helpers import makeSyntheticConfig
        from VISSSlib import files

        config = makeSyntheticConfig(tmp_path)
        ff = files.FindFiles("20260110", config.leader, config)
        fOut = ff.quicklook.level1trackParticles
        assert "/level1trackParticles/2026/" in fOut
        assert os.path.basename(fOut).startswith("level1trackParticles_V1_test_")
        assert fOut.endswith("_20260110.html")
        assert ff.quicklookCurrent.level1trackParticles.endswith("_current.html")
        # the other quicklooks stay png
        assert ff.quicklook.level1matchParticles.endswith(".png")

    @pytest.mark.unit
    def test_routedZip(self):
        from VISSSlib import quicklooks

        class FakeArchive:
            def extractnpy(self, pidStr):
                return f"image {pidStr}"

        routed = quicklooks._RoutedZip()
        routed.add("0000001", FakeArchive())
        routed.add("0000002", None)  # archive missing/unreadable
        assert routed.extractnpy("0000001") == "image 0000001"
        # missing archives and unknown pids raise KeyError, the GIF renderer
        # skips these frames
        with pytest.raises(KeyError):
            routed.extractnpy("0000002")
        with pytest.raises(KeyError):
            routed.extractnpy("0000003")

    @pytest.mark.unit
    def test_writeGifHtml(self, tmp_path):
        from helpers import makeSyntheticConfig
        from PIL import Image
        from VISSSlib import quicklooks

        config = makeSyntheticConfig(tmp_path)
        gif = tmp_path / "a_track1.gif"
        Image.new("L", (4, 4)).save(gif)
        fOut = str(tmp_path / "out" / "grid.html")
        quicklooks._writeGifHtml([str(gif)], fOut, config, "title <b>", ncols=2)
        html = open(fOut).read()
        assert "data:image/gif;base64," in html
        assert "repeat(2, 1fr)" in html
        assert "title &lt;b&gt;" in html  # escaped


class TestTrackParticles(object):
    @pytest.fixture(autouse=True)
    def setup_files(self):
        self.config = readTestSettings("test_0.6/test_0.6.yaml")
        self.testPath = get_test_data_path()
        yield
        shutil.rmtree(self.config.tmpPath)

    def test_createLevel1trackParticles(self):
        case = "20260110"
        fOut = f"{self.config.tmpPath}/trackParticles_{{case}}.html"
        res = createLevel1trackParticles(
            case,
            self.config,
            skipExisting=False,
            minBlur=0,
            minSize=1,
            minTrackLength=2,
            minRenderedFrames=2,
            targetGifs=2,
            fOut=fOut,
        )
        fOutDay, nGifs = res[0] if isinstance(res, list) else res
        assert fOutDay == fOut.format(case=case)
        assert os.path.isfile(fOutDay)
        assert 1 <= nGifs <= 2
        assert "data:image/gif;base64," in open(fOutDay).read()

        # skipExisting: nothing to do the second time
        res = createLevel1trackParticles(
            case, self.config, skipExisting=True, fOut=fOut
        )
        res = res[0] if isinstance(res, list) else res
        assert res == (None, None)

    def test_createLevel1trackParticlesSingle(self):
        import re

        import xarray as xr
        from VISSSlib import files, quicklooks

        ff = files.FindFiles("20260110", self.config.leader, self.config)
        trackFile = ff.listFiles("level1track")[0]
        block = re.search(r"(\d{8}-\d{6})", os.path.basename(trackFile)).group(1)
        ffBlock = files.FindFiles(block, self.config.leader, self.config)

        # images are only stored for some particles: find a track with crops
        # of at least two matched particles in both cameras
        cache = {}
        archives = [
            quicklooks._archiveFor(
                np.datetime64(ffBlock.datetime64, "ns"), camera, self.config, cache
            )
            for camera in (self.config.leader, self.config.follower)
        ]
        with xr.open_dataset(trackFile) as tr:
            pids = tr.pid.values
            hasCrops = np.array(
                [
                    ("%07i" % pids[0, ii] in archives[0].index)
                    and ("%07i" % pids[1, ii] in archives[1].index)
                    for ii in range(pids.shape[1])
                ]
            )
            trackIds, counts = np.unique(
                tr.track_id.values[hasCrops], return_counts=True
            )
        quicklooks._closeArchives(cache)
        if not (counts >= 2).any():
            pytest.skip("no track with image crops in the sample data")
        trackId = int(trackIds[counts >= 2][0])

        fOut = f"{self.config.tmpPath}/single_{{track_id}}_{{view}}.html"
        for view, showLegend in [("both", True), ("leader", False)]:
            res = createLevel1trackParticlesSingle(
                block,
                trackId,
                self.config,
                view=view,
                showLegend=showLegend,
                fOut=fOut,
            )
            assert res == fOut.format(track_id=trackId, view=view)
            assert "data:image/gif;base64," in open(res).read()

        with pytest.raises(ValueError):
            createLevel1trackParticlesSingle(
                "20260110", trackId, self.config, fOut=fOut
            )  # block start time missing
