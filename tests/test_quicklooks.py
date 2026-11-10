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


class TestQuicklookHelpers(object):
    """Unit tests of the shared plotting helpers (synthetic data)."""

    @pytest.mark.unit
    def test_timeStep(self):
        from VISSSlib import quicklooks

        t = np.datetime64("2026-01-10T00:00") + np.arange(5) * np.timedelta64(5, "m")
        assert quicklooks._timeStep(t) == np.timedelta64(300, "s")
        # not defined for fewer than two stamps
        default = np.timedelta64(7, "m")
        assert quicklooks._timeStep(t[:1], default=default) == default

    @pytest.mark.unit
    def test_shadeSpansShowsIsolatedStepsAndKeepsLimits(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from VISSSlib import quicklooks

        fig, ax = plt.subplots()
        t0 = np.datetime64("2026-01-10T00:00")
        ax.plot([t0, t0 + np.timedelta64(1, "h")], [3, 5])
        ylim = ax.get_ylim()
        starts = np.array([t0 + np.timedelta64(10, "m"), np.datetime64("NaT")])
        handle = quicklooks._shadeSpans(
            ax, starts, np.timedelta64(5, "m"), "red", "flagged"
        )
        assert handle.get_label() == "flagged"
        # NaT is ignored, the one isolated step is drawn with a width
        assert len(ax.collections) == 1
        assert len(ax.collections[0].get_paths()) == 1
        assert ax.get_ylim() == ylim
        # nothing to shade, nothing drawn
        assert (
            quicklooks._shadeSpans(
                ax, np.array([np.datetime64("NaT")]), np.timedelta64(5, "m"), "r", "x"
            )
            is None
        )
        assert len(ax.collections) == 1
        plt.close(fig)

    @pytest.mark.unit
    def test_formatDayAxes(self, tmp_path):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from helpers import makeSyntheticConfig
        from VISSSlib import files, quicklooks

        config = makeSyntheticConfig(tmp_path)
        ff = files.FindFiles("20260110", config.leader, config)
        fig, axs, caxs = quicklooks._stackedFigure([1, 1], (6, 4))
        assert len(caxs) == 2
        quicklooks._formatDayAxes(axs, ff)
        fig.canvas.draw()
        labels = [t.get_text() for t in axs[-1].get_xticklabels()]
        assert labels[0] == "00:00" and labels[-1] == "24:00"
        # shared time axis: labels only below the lowest panel
        assert not axs[0].xaxis.get_tick_params()["labelbottom"]
        assert axs[-1].get_xlabel() == "time"
        plt.close(fig)

    @pytest.mark.unit
    def test_generateDispatchesLevel2track(self, tmp_path, monkeypatch):
        from helpers import makeSyntheticConfig
        from VISSSlib import quicklooks

        config = makeSyntheticConfig(tmp_path)
        called = []
        monkeypatch.setattr(
            quicklooks,
            "createLevel2trackQuicklook",
            lambda *args, **kwargs: called.append("track"),
        )
        monkeypatch.setattr(
            quicklooks,
            "createLevel2matchQuicklook",
            lambda *args, **kwargs: called.append("match"),
        )
        quicklooks.generate("20260110", config, "level2track")
        assert called == ["track"]


class TestSourceVersion(object):
    """oldestSourceVersion: version of the source files for the status line."""

    @staticmethod
    def _write(path, **attrs):
        import xarray as xr

        xr.Dataset({"a": ("x", [1.0])}, attrs=attrs).to_netcdf(path)
        return str(path)

    @pytest.mark.unit
    def test_oldestOfSeveralSources(self, tmp_path):
        from VISSSlib import tools

        f1 = self._write(
            tmp_path / "a.nc",
            level2track_version="1.2.2",
            level1detect_versions="1.0.1,1.2.0",
            history="2026-01-01: created with VISSSlib 1.2.2 and OpenCV 4.9",
        )
        # only known from the history (file predates the *_version attrs)
        f2 = self._write(
            tmp_path / "b.nc",
            history="2024-01-01: created with VISSSlib 1.0.10 and OpenCV 4.9",
        )
        # "1.0.10" is older than "1.2" and newer than "1.0.1"
        assert tools.oldestSourceVersion([f2, f1]) == "1.0.1"
        assert tools.oldestSourceVersion(f2) == "1.0.10"
        assert (
            tools.oldestSourceVersion(
                [f2, self._write(tmp_path / "c.nc", x_version="1.2")]
            )
            == "1.0.10"
        )

    @pytest.mark.unit
    def test_noInformation(self, tmp_path):
        from VISSSlib import tools

        assert tools.oldestSourceVersion(None) is None
        assert tools.oldestSourceVersion([]) is None
        # missing, not netCDF and without version metadata
        plain = self._write(tmp_path / "d.nc", title="no version here")
        txt = tmp_path / "e.txt"
        txt.write_text("1.0.0")
        assert (
            tools.oldestSourceVersion([plain, str(txt), str(tmp_path / "x.nc")]) is None
        )


class TestSaveFigLogo(object):
    @pytest.mark.unit
    def test_logoAnchoredAtPlotAndConstrainedLayoutKept(self, tmp_path):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from helpers import makeSyntheticConfig
        from PIL import Image
        from VISSSlib import quicklooks, tools

        logo = tmp_path / "logo.png"
        Image.new("RGBA", (30, 30), (255, 0, 0, 255)).save(logo)
        config = makeSyntheticConfig(tmp_path, logo=str(logo))

        fig, axs, _ = quicklooks._stackedFigure([1, 1], (6, 4))
        axs[0].plot([0, 1], [0, 1])
        tools.savefig(fig, config, str(tmp_path / "out.png"), logoAnchor=axs[0])
        # the constrained layout of the figure is not replaced by tight_layout
        assert type(fig.get_layout_engine()).__name__ == "ConstrainedLayoutEngine"
        assert len(fig.images) == 1
        # logo sits on top of the upper left corner of the plot
        box = axs[0].get_window_extent()
        x0, x1, y0, y1 = fig.images[0].get_extent()
        assert np.isclose(x0, box.x0, atol=1) and y0 >= box.y1
        plt.close(fig)
