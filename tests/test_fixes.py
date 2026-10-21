import types

import numpy as np
import pytest
import xarray as xr
from VISSSlib.fixes import (
    captureIdOverflows,
    isDroppableTrailingFrameShortfall,
    removeFlippedCaptureTimeFrames,
    revertIdOverflowFix,
)


def _makeDat(ids, fps=100):
    n = len(ids)
    times = np.datetime64("2026-01-01T00:00:00") + np.arange(n) * np.timedelta64(
        round(1e6 / fps), "us"
    )
    return xr.Dataset(
        {"capture_id": ("pid", np.asarray(ids))},
        coords={"pid": np.arange(n), "capture_time": ("pid", times)},
    )


class TestCaptureIdOverflows:
    """Unit tests for fixes.captureIdOverflows/revertIdOverflowFix, pure
    functions of an in-memory xr.Dataset with capture_id/capture_time
    (no file I/O). The exact overflow-wrap boundary value is a hardware
    detail this suite doesn't assert on; these cover the surrounding,
    well-defined behavior instead.
    """

    @pytest.mark.unit
    def test_no_overflow_leaves_ids_unchanged(self):
        config = types.SimpleNamespace(fps=100)
        dat = _makeDat(np.arange(100, 110))
        out = captureIdOverflows(dat, config)
        assert list(out.capture_id.values) == list(range(100, 110))

    @pytest.mark.unit
    def test_storeOrig_preserves_original_values(self):
        config = types.SimpleNamespace(fps=100)
        dat = _makeDat(np.arange(100, 110))
        out = captureIdOverflows(dat, config, storeOrig=True)
        assert "capture_id_orig" in out.keys()
        assert list(out.capture_id_orig.values) == list(range(100, 110))

    @pytest.mark.unit
    def test_storeOrig_false_omits_orig_variable(self):
        config = types.SimpleNamespace(fps=100)
        dat = _makeDat(np.arange(100, 110))
        out = captureIdOverflows(dat, config, storeOrig=False)
        assert "capture_id_orig" not in out.keys()

    @pytest.mark.unit
    def test_idOffset_is_added_to_all_ids(self):
        config = types.SimpleNamespace(fps=100)
        dat = _makeDat(np.arange(100, 110))
        out = captureIdOverflows(dat, config, idOffset=5)
        assert list(out.capture_id.values) == list(range(105, 115))

    @pytest.mark.unit
    def test_existing_capture_id_orig_is_reverted_before_refixing(self):
        config = types.SimpleNamespace(fps=100)
        dat = _makeDat(np.arange(100, 110))
        dat["capture_id_orig"] = dat["capture_id"].copy()
        dat["capture_id"] = dat["capture_id"] * 0  # corrupt capture_id
        out = captureIdOverflows(dat, config, storeOrig=False)
        # corrupted capture_id must have been restored from capture_id_orig first
        assert list(out.capture_id.values) == list(range(100, 110))

    @pytest.mark.unit
    def test_more_observed_jumps_than_expected_raises(self):
        config = types.SimpleNamespace(fps=100)
        # two apparent wraps within a time span that only accounts for one
        dat = _makeDat([65533, 65534, 0, 1, 65533, 0, 1])
        with pytest.raises(RuntimeError):
            captureIdOverflows(dat, config)

    @pytest.mark.unit
    def test_revertIdOverflowFix_restores_original_capture_id(self):
        config = types.SimpleNamespace(fps=100)
        dat = _makeDat(np.arange(100, 110))
        fixed = captureIdOverflows(dat, config, storeOrig=True)
        reverted = revertIdOverflowFix(fixed)
        assert list(reverted.capture_id.values) == list(range(100, 110))
        assert "capture_id_fixed" in reverted.keys()


def _makeMetaDat1(captureTimes):
    """A single source's metadata in its own original recording order,
    matching the shape fixes.removeFlippedCaptureTimeFrames expects."""
    n = len(captureTimes)
    return xr.Dataset(
        {"dummy": ("capture_time", np.arange(n))},
        coords={"capture_time": np.asarray(captureTimes, dtype="int64")},
    )


class TestRemoveFlippedCaptureTimeFrames:
    """Unit tests for fixes.removeFlippedCaptureTimeFrames, a pure function
    of a single source's in-memory xr.Dataset (no file I/O), reflecting the
    real M2050/M1280 hardware glitch shapes measured directly against
    production hyytiala2_v4 and SAIL ascii logs: one or more isolated,
    microsecond-scale backwards flips between two consecutive frames,
    scattered independently through an otherwise clean file.
    """

    @pytest.mark.unit
    def test_no_flips_leaves_data_untouched(self):
        metaDat = _makeMetaDat1([0, 1, 2, 3, 4])
        result, nDropped = removeFlippedCaptureTimeFrames(metaDat, "f.txt")
        assert nDropped == 0
        assert list(result.capture_time.values) == [0, 1, 2, 3, 4]

    @pytest.mark.unit
    def test_single_isolated_flip_drops_neighbours(self):
        # index 2 (value 5) and index 3 (value 1) are flipped, matching the
        # real single-swap shape measured on the SAIL/M1280 case
        metaDat = _makeMetaDat1([0, 3, 5, 1, 6, 7])
        result, nDropped = removeFlippedCaptureTimeFrames(metaDat, "f.txt")
        assert nDropped == 3
        assert list(result.capture_time.values) == [0, 6, 7]

    @pytest.mark.unit
    def test_multiple_isolated_flips_each_repaired_independently(self):
        # several well-separated single-sample flips in one file -- the
        # exact pattern measured on the real hyytiala2_v4 case (5 isolated
        # flips per thread, far apart from each other). The old algorithm
        # this replaces asserted all jumps had to form one contiguous
        # group and would have raised here; this must repair each
        # independently instead.
        metaDat = _makeMetaDat1([0, 5, 1, 6, 10, 15, 11, 16, 20, 25, 21, 26])
        result, nDropped = removeFlippedCaptureTimeFrames(metaDat, "f.txt")
        assert nDropped == 9  # 3 clusters x 3 dropped each
        assert np.all(np.diff(result.capture_time.values) >= 0)

    @pytest.mark.unit
    def test_grouped_adjacent_flips_dropped_together(self):
        # two adjacent backwards steps form one cluster and are repaired
        # as a single unit, not two overlapping ones
        metaDat = _makeMetaDat1([0, 3, 5, 2, 1, 6])
        result, nDropped = removeFlippedCaptureTimeFrames(metaDat, "f.txt")
        assert nDropped > 0
        assert np.all(np.diff(result.capture_time.values) >= 0)

    @pytest.mark.unit
    def test_too_many_flips_raises(self):
        metaDat = _makeMetaDat1(list(range(21, 0, -1)))
        with pytest.raises(AssertionError, match="more than 20"):
            removeFlippedCaptureTimeFrames(metaDat, "f.txt")

    @pytest.mark.unit
    def test_surviving_values_are_never_mutated(self):
        # every timestamp that isn't dropped must be bit-for-bit identical
        # to its original value -- this fix must only ever remove frames
        # whose own timestamp is already self-contradictory, never shift,
        # average, or interpolate a value for a frame that is kept
        original = [0, 5, 1, 6, 10, 15, 11, 16]
        metaDat = _makeMetaDat1(original)
        result, _ = removeFlippedCaptureTimeFrames(metaDat, "f.txt")
        assert set(result.capture_time.values.tolist()) <= set(original)


class TestIsDroppableTrailingFrameShortfall:
    """Unit tests for fixes.isDroppableTrailingFrameShortfall, reflecting
    the observed real-world shortfall sizes (1-6 frames, always at the
    tail of a file) on hyytiala2_v3 and nyaalesund.
    """

    @pytest.mark.unit
    def test_small_shortfall_is_droppable(self):
        assert isDroppableTrailingFrameShortfall(1) is True
        assert isDroppableTrailingFrameShortfall(6) is True

    @pytest.mark.unit
    def test_shortfall_at_cap_is_droppable(self):
        assert isDroppableTrailingFrameShortfall(25, maxFrames=25) is True

    @pytest.mark.unit
    def test_large_shortfall_is_not_droppable(self):
        # a shortfall spanning a large chunk of a file indicates real
        # corruption, not a benign encoder-flush tail loss, and must not
        # be silently swallowed
        assert isDroppableTrailingFrameShortfall(26, maxFrames=25) is False
        assert isDroppableTrailingFrameShortfall(5000) is False


class TestDespikeRotation:
    @staticmethod
    def _rot(ofz, phi=None, theta=None):
        import numpy as np
        import xarray as xr

        n = len(ofz)
        phi = [-1.0] * n if phi is None else phi
        theta = [-1.25] * n if theta is None else theta
        t = np.datetime64("2020-01-01T00:00") + np.arange(n) * np.timedelta64(5, "m")
        out = {}
        for k, v in [("camera_Ofz", ofz), ("camera_phi", phi), ("camera_theta", theta)]:
            vals = np.stack([np.array(v, float), np.full(n, 0.3)], axis=1)
            out[k] = xr.DataArray(
                vals,
                dims=["file_starttime", "camera_rotation"],
                coords=dict(file_starttime=t, camera_rotation=["mean", "err"]),
            )
        return xr.Dataset(out)

    def test_single_spike_and_run(self):
        from VISSSlib import fixes

        r, n = fixes.despikeRotation(self._rot([-38, -38, -15, -37, -37, -37]))
        assert n == 1
        assert abs(float(r.camera_Ofz.sel(camera_rotation="mean")[2]) + 37.5) < 1
        r, n = fixes.despikeRotation(
            self._rot([-5, -4, -60, -30, -8, -7, -6], phi=[2.1] * 7)
        )
        assert n == 2
        assert float(r.camera_Ofz.sel(camera_rotation="mean").min()) > -9

    def test_persistent_change_is_kept(self):
        from VISSSlib import fixes

        r, n = fixes.despikeRotation(self._rot([-38] * 4 + [-60] * 6))
        assert n == 0
        r, n = fixes.despikeRotation(self._rot([-38, -38, -60, -60, -60, -60, -60]))
        assert n == 0


class TestRejectRotationJump:
    @staticmethod
    def _r(ofz, phi=-1.0, theta=-1.2):
        import pandas as pd

        return pd.Series(dict(camera_Ofz=ofz, camera_phi=phi, camera_theta=theta))

    def test_spike_is_rejected(self):
        from VISSSlib import fixes

        default, pending, out = self._r(-33.0), [], []
        for x in [self._r(-34), self._r(-94, 7.8, 0.0), self._r(-86, 5.5, -0.9), self._r(-33.5)]:
            a, pending = fixes.rejectRotationJump(x, default, pending)
            out.append(None if a is None else float(a.camera_Ofz))
        assert out == [-34.0, None, None, -33.5]

    def test_persistent_level_is_accepted_after_confirmation(self):
        from VISSSlib import fixes

        default, pending, out = self._r(-33.0), [], []
        for x in [self._r(-80.0)] * 8:
            a, pending = fixes.rejectRotationJump(x, default, pending)
            out.append(None if a is None else float(a.camera_Ofz))
        assert out[:5] == [None] * 5 and out[5] == -80.0


class TestMosaicFrameMappingCache:
    def test_day_without_resolved_segment_is_not_outdated(self, tmp_path, monkeypatch):
        from VISSSlib import fixes, tools

        fname = str(tmp_path / "mapping.nc")
        xr.Dataset(
            {"resolved": ("segment", np.zeros(3, dtype=bool))},
            coords={"segment": np.arange(3)},
        ).to_netcdf(fname)
        monkeypatch.setattr(tools, "readSettings", lambda c: c)
        monkeypatch.setattr(fixes, "mosaicFrameMappingFname", lambda case, config: fname)
        out = fixes.createMosaicFrameMapping(
            "20200228", types.SimpleNamespace(), skipExisting=True, writeNc=False
        )
        assert not bool(out.resolved.any())


class TestMosaicRotatedLagTest:
    def _setup(self, lag=7, n=400, seed=1):
        from VISSSlib import matching

        rng = np.random.default_rng(seed)
        rotate = dict(camera_Ofz=120.0, camera_phi=-11.5, camera_theta=-1.3)
        fw = 1280
        lx = rng.uniform(0, fw, n)
        fx = rng.uniform(0, fw, n)
        fz = rng.uniform(0, 1000, n)
        lzTrue = np.asarray(
            matching.calc_L_z_withOffsets(lx, fw - fx, fz, **rotate)
        ) + rng.normal(0, 1.0, n)
        # leader frame i is seen by the follower in frame i + lag
        u = np.arange(n) * 3
        lPart = dict(D=np.full(n, 30.0), z=lzTrue, xc=lx)
        fPart = dict(D=np.full(n, 30.0), z=fz, xc=fx)
        seg = np.zeros(n, dtype=int)
        return lPart, fPart, seg, u, u + lag, rotate, fw

    def test_rotated_residual_finds_lag(self):
        from VISSSlib.fixes import _mosaicParticleLagTest

        lPart, fPart, seg, lU, fU, rotate, fw = self._setup()
        cand = list(range(0, 15))
        raw = _mosaicParticleLagTest(lPart, seg, lU, 0, fPart, seg, fU, 0, cand, 8)
        rot = _mosaicParticleLagTest(
            lPart, seg, lU, 0, fPart, seg, fU, 0, cand, 8, rotate=rotate, frameWidth=fw
        )
        assert rot["lag"] == 7
        assert rot["iqr"] < 3
        assert rot["iqr2nd"] > 10 * rot["iqr"]
        # the raw difference is broad even for the right lag
        assert raw["iqr"] > 20 or raw["lag"] != 7


class TestMosaicRotationSearch:
    def test_finds_lag_and_rotation_without_prior(self):
        from VISSSlib import matching
        from VISSSlib.fixes import _mosaicRotationSearch

        rng = np.random.default_rng(3)
        rotate = dict(camera_Ofz=60.0, camera_phi=-3.0, camera_theta=0.75)
        fw, n, lag = 1280, 3000, 11
        lx = rng.uniform(0, fw, n)
        fx = rng.uniform(0, fw, n)
        fz = rng.uniform(0, 1000, n)
        lz = np.asarray(
            matching.calc_L_z_withOffsets(lx, fw - fx, fz, **rotate)
        ) + rng.normal(0, 1.0, n)
        u = np.arange(n) * 3
        lPart = dict(D=np.full(n, 30.0), z=lz, xc=lx)
        fPart = dict(D=np.full(n, 30.0), z=fz, xc=fx)
        seg = np.zeros(n, dtype=int)
        out = _mosaicRotationSearch(
            lPart,
            seg,
            u,
            0,
            fPart,
            seg,
            u + lag,
            0,
            list(range(lag - 3, lag + 4)),
            fw,
            phiRange=(-5.0, 0.0),
            thetaRange=(-1.0, 2.0),
        )
        assert out is not None
        assert out["lag"] == lag
        assert abs(out["phi"] - rotate["camera_phi"]) < 0.3
        assert abs(out["theta"] - rotate["camera_theta"]) < 0.3
        assert abs(out["Ofz"] - rotate["camera_Ofz"]) < 8

    def test_no_result_for_random_pairs(self):
        from VISSSlib.fixes import _mosaicRotationSearch

        rng = np.random.default_rng(4)
        n, fw = 3000, 1280
        lPart = dict(D=np.full(n, 30.0), z=rng.uniform(0, 1000, n), xc=rng.uniform(0, fw, n))
        fPart = dict(D=np.full(n, 30.0), z=rng.uniform(0, 1000, n), xc=rng.uniform(0, fw, n))
        seg = np.zeros(n, dtype=int)
        u = np.arange(n) * 3
        out = _mosaicRotationSearch(
            lPart, seg, u, 0, fPart, seg, u + 5, 0, list(range(2, 9)), fw,
            phiRange=(-3.0, 1.0), thetaRange=(-1.0, 1.0),
        )
        assert out is None


class TestMosaicClockOffset:
    def test_offset_from_knots(self, monkeypatch):
        from VISSSlib import fixes

        t0 = np.datetime64("2025-12-10T22:00:00")
        kt = t0 + np.arange(5) * np.timedelta64(60, "s") + np.timedelta64(1739, "s")
        ds = xr.Dataset(
            {"knot_time": ("knot", kt.astype("datetime64[ns]")),
             "knot_offset": ("knot", np.full(5, 1739.0))}
        )
        monkeypatch.setattr(fixes, "createMosaicFrameMapping", lambda *a, **k: ds)
        off = fixes.mosaicClockOffsetSeconds(
            "20251210", None, t0, t0 + np.timedelta64(300, "s")
        )
        assert off == pytest.approx(1739.0)
        monkeypatch.setattr(fixes, "createMosaicFrameMapping", lambda *a, **k: None)
        assert fixes.mosaicClockOffsetSeconds("20251210", None, t0, t0) == 0.0


class TestMosaicBlockLags:
    def test_lag_steps_between_blocks(self):
        from VISSSlib import matching
        from VISSSlib.fixes import _mosaicBlockLags

        rng = np.random.default_rng(5)
        rotate = dict(camera_Ofz=-25.0, camera_phi=0.4, camera_theta=0.45)
        fw, nPer = 1280, 600
        lagTrue = [-229, -229, -228]  # one block each, lag drifts by one frame
        n = nPer * len(lagTrue)
        lx = rng.uniform(0, fw, n)
        fx = rng.uniform(0, fw, n)
        fz = rng.uniform(0, 1000, n)
        lz = np.asarray(matching.calc_L_z_withOffsets(lx, fw - fx, fz, **rotate)) + rng.normal(0, 0.8, n)
        lU = np.arange(n) * 4 + 10000
        block = np.repeat(np.arange(len(lagTrue)), nPer)
        fU = lU + np.array(lagTrue)[block]
        blockUs = int(10 * 60e6)
        fct = block * blockUs + np.tile(np.linspace(1e6, blockUs - 1e6, nPer), len(lagTrue)).astype(np.int64)
        lPart = dict(D=np.full(n, 30.0), z=lz, xc=lx, ct=fct)
        fPart = dict(D=np.full(n, 30.0), z=fz, xc=fx, ct=fct)
        seg = np.zeros(n, dtype=int)
        out = _mosaicBlockLags(
            lPart, seg, lU, 0, fPart, seg, fU, 0, -229, rotate, fw,
            0, blockUs * len(lagTrue), blockMinutes=10, halfRange=5, minPairs=40,
        )
        assert [b[2] for b in out] == lagTrue

    def test_block_without_pairs_has_no_lag(self):
        from VISSSlib.fixes import _mosaicBlockLags

        n, fw = 200, 1280
        rng = np.random.default_rng(6)
        lPart = dict(D=np.full(n, 30.0), z=rng.uniform(0, 1000, n), xc=rng.uniform(0, fw, n), ct=np.arange(n) * 1000)
        fPart = dict(D=np.full(n, 30.0), z=rng.uniform(0, 1000, n), xc=rng.uniform(0, fw, n), ct=np.arange(n) * 1000)
        seg = np.zeros(n, dtype=int)
        u = np.arange(n) * 3
        out = _mosaicBlockLags(
            lPart, seg, u, 0, fPart, seg, u + 7, 0, 7,
            dict(camera_Ofz=-25.0, camera_phi=0.4, camera_theta=0.45), fw,
            0, int(10 * 60e6), blockMinutes=10, halfRange=5, minPairs=40,
        )
        assert all(b[2] is None for b in out)


class TestMosaicDenseLagTest:
    @staticmethod
    def _dense(lag, K=20, nFrames=400, dz0=48.0, seed=5):
        rng = np.random.default_rng(seed)
        lu, lz, fu, fz = [], [], [], []
        vel = rng.uniform(80, 300, (nFrames, K))
        for u in range(nFrames):
            z = rng.uniform(0, 1000, K)
            lu += [u] * K
            lz += list(z)
            # same particles one frame earlier/later are displaced by their
            # fall speed, which differs from particle to particle
            for dl, shift in [(lag, 0.0), (lag + 1, 1.0), (lag - 1, -1.0)]:
                zz = z - dz0 + shift * vel[u] + rng.normal(0, 1.5, K)
                fu += [u + dl] * K
                fz += list(zz)
        lu, fu = np.array(lu), np.array(fu)
        order = np.argsort(fu, kind="stable")
        lPart = dict(D=np.full(len(lu), 30.0), z=np.array(lz), xc=np.zeros(len(lu)))
        fz = np.array(fz)[order]
        fPart = dict(D=np.full(len(fu), 30.0), z=fz, xc=np.zeros(len(fu)))
        return lPart, lu, fPart, fu[order]

    def test_finds_lag_without_single_particle_frames(self):
        from VISSSlib.fixes import _mosaicDenseLagTest

        lag = 65535 * 3 + 7
        lPart, lu, fPart, fu = self._dense(lag)
        out = _mosaicDenseLagTest(
            lPart,
            np.zeros(len(lu), int),
            lu,
            0,
            fPart,
            np.zeros(len(fu), int),
            fu,
            0,
            list(range(lag - 3, lag + 4)),
        )
        assert out is not None
        assert out["lag"] == lag
        assert abs(out["dz"] - 48.0) < 2
        assert out["z"] > 10 and out["z"] > 2 * out["z2nd"]

    def test_no_peak_for_unrelated_frames(self):
        from VISSSlib.fixes import _mosaicDenseLagTest

        rng = np.random.default_rng(6)
        n = 8000
        lu = np.repeat(np.arange(400), 20)
        fu = np.repeat(np.arange(400), 20)
        lPart = dict(D=np.full(n, 30.0), z=rng.uniform(0, 1000, n), xc=np.zeros(n))
        fPart = dict(D=np.full(n, 30.0), z=rng.uniform(0, 1000, n), xc=np.zeros(n))
        out = _mosaicDenseLagTest(
            lPart, np.zeros(n, int), lu, 0, fPart, np.zeros(n, int), fu, 0, list(range(-3, 4))
        )
        assert (out is None) or (out["z"] < 8)


class TestMosaicDenseBlockLags:
    def test_lag_change_inside_segment(self):
        from VISSSlib.fixes import _mosaicDenseBlockLags

        rng = np.random.default_rng(8)
        nFrames, K, lag0, dz0 = 6000, 12, 1000, 43.0
        lu, lz, fu, fz = [], [], [], []
        for u in range(nFrames):
            z = rng.uniform(0, 1000, K)
            lag = lag0 if u < nFrames // 2 else lag0 + 1
            lu += [u] * K
            lz += list(z)
            fu += [u + lag] * K
            fz += list(z - dz0 + rng.normal(0, 1.5, K))
        fu = np.array(fu)
        order = np.argsort(fu, kind="stable")
        lu = np.array(lu)
        period = 7143
        lPart = dict(D=np.full(len(lu), 4.0), z=np.array(lz), xc=np.zeros(len(lu)), ct=lu * period)
        fPart = dict(D=np.full(len(fu), 4.0), z=np.array(fz)[order], xc=np.zeros(len(fu)), ct=fu[order] * period)
        fu = fu[order]
        blocks = _mosaicDenseBlockLags(
            lPart, np.zeros(len(lu), int), lu, 0, fPart, np.zeros(len(fu), int), fu, 0,
            lag0, int(fPart["ct"].min()), int(fPart["ct"].max()),
            dzRef=dz0, blockMinutes=0.2, halfRange=3,
        )
        lags = [b[2] for b in blocks if b[2] is not None]
        assert len(lags) >= 0.7 * len(blocks)
        assert lags[0] == lag0 and lags[-1] == lag0 + 1
        assert set(lags) == {lag0, lag0 + 1}


class TestMosaicDenseBlockLagsRotated:
    def test_tracks_drifting_lag_with_rotation(self):
        from VISSSlib import matching
        from VISSSlib.fixes import _mosaicDenseBlockLags

        rng = np.random.default_rng(11)
        rotate = dict(camera_Ofz=-86.0, camera_phi=1.5, camera_theta=1.25)
        fw, nFrames, K, lag0 = 1280, 12000, 10, 5000
        lu, lx, lz, fu, fx, fz = [], [], [], [], [], []
        for u in range(nFrames):
            # lag drifts by one frame every 2000 frames
            lag = lag0 + u // 2000
            x = rng.uniform(0, fw, K)
            fxx = rng.uniform(0, fw, K)
            fzz = rng.uniform(0, 1000, K)
            z = np.asarray(matching.calc_L_z_withOffsets(x, fw - fxx, fzz, **rotate)) + rng.normal(0, 1.0, K)
            lu += [u] * K; lx += list(x); lz += list(z)
            fu += [u + lag] * K; fx += list(fxx); fz += list(fzz)
        lu = np.array(lu); fu = np.array(fu)
        order = np.argsort(fu, kind="stable")
        period = 7143
        lPart = dict(D=np.full(len(lu), 4.0), z=np.array(lz), xc=np.array(lx), ct=lu * period)
        fPart = dict(D=np.full(len(fu), 4.0), z=np.array(fz)[order], xc=np.array(fx)[order], ct=fu[order] * period)
        fu = fu[order]
        blocks = _mosaicDenseBlockLags(
            lPart, np.zeros(len(lu), int), lu, 0, fPart, np.zeros(len(fu), int), fu, 0,
            lag0, int(fPart["ct"].min()), int(fPart["ct"].max()),
            blockMinutes=0.2, halfRange=3, rotate=rotate, frameWidth=fw, track=True,
        )
        lags = [b[2] for b in blocks if b[2] is not None]
        assert len(lags) >= 0.7 * len(blocks)
        assert lags[0] == lag0 and lags[-1] >= lag0 + 4
        assert all(b >= a for a, b in zip(lags[:-1], lags[1:]))


class TestMosaicDenseBlockLagsDrift:
    @staticmethod
    def _drifting(nFrames=12000, K=10, lag0=3000, per=1000, dz0=43.0, seed=21):
        rng = np.random.default_rng(seed)
        lu, lz, fu, fz = [], [], [], []
        for u in range(nFrames):
            lag = lag0 + 2 * (u // per)  # 2 frames per block
            z = rng.uniform(0, 1000, K)
            lu += [u] * K; lz += list(z)
            fu += [u + lag] * K; fz += list(z - dz0 + rng.normal(0, 1.5, K))
        lu = np.array(lu); fu = np.array(fu)
        order = np.argsort(fu, kind="stable")
        period = 7143
        lPart = dict(D=np.full(len(lu), 4.0), z=np.array(lz), xc=np.zeros(len(lu)), ct=lu * period)
        fPart = dict(D=np.full(len(fu), 4.0), z=np.array(fz)[order], xc=np.zeros(len(fu)), ct=fu[order] * period)
        return lPart, lu, fPart, fu[order]

    def test_tracking_follows_drift_beyond_fixed_window(self):
        from VISSSlib.fixes import _mosaicDenseBlockLags

        lag0 = 3000
        lPart, lu, fPart, fu = self._drifting(lag0=lag0)
        args = (
            lPart, np.zeros(len(lu), int), lu, 0, fPart, np.zeros(len(fu), int), fu, 0,
            lag0 + 12, int(fPart["ct"].min()), int(fPart["ct"].max()),
        )
        kw = dict(dzRef=43.0, blockMinutes=1000 * 7143 / 60e6, halfRange=3)
        fixed = _mosaicDenseBlockLags(*args, **kw)
        tracked = _mosaicDenseBlockLags(*args, track=True, **kw)
        nf = sum(b[2] is not None for b in fixed)
        nt = sum(b[2] is not None for b in tracked)
        assert nt > nf
        assert nt >= 0.8 * len(tracked)
        lags = [b[2] for b in tracked if b[2] is not None]
        assert all(b >= a for a, b in zip(lags[:-1], lags[1:]))

    def test_fill_extrapolates_with_slope(self):
        from VISSSlib.fixes import _mosaicFillBlockLags

        blocks = [(i, i + 1, (100 + 2 * i) if 3 <= i <= 8 else None, 0, 0.0) for i in range(12)]
        out = _mosaicFillBlockLags(blocks)
        assert [b[2] for b in out] == [100 + 2 * i for i in range(12)]


class TestMosaicBlocksToRows:
    def test_lag_steps_inside_blocks(self):
        from VISSSlib.fixes import _mosaicBlocksToRows

        min_us = 60_000_000
        blocks = [(i * 10 * min_us, (i + 1) * 10 * min_us, 100 + 2 * i, 5, 9.0) for i in range(6)]
        r = dict(lsegment=0, fsegment=0, fct_start=blocks[0][0], fct_end=blocks[-1][1], lag=0, lag_mod=0, n_pairs=0, resolved=False, dense=False)
        rows = _mosaicBlocksToRows(blocks, r, 65535)
        lags = [x["lag"] for x in rows]
        assert lags == sorted(lags) and lags[0] == 100 and lags[-1] == 110
        assert len(rows) >= 9  # a frame step about every 5 minutes
        # contiguous coverage without overlap
        assert rows[0]["fct_start"] == blocks[0][0] and rows[-1]["fct_end"] == blocks[-1][1] - 1
        assert all(b["fct_start"] == a["fct_end"] + 1 for a, b in zip(rows[:-1], rows[1:]))

    def test_large_jump_not_interpolated(self):
        from VISSSlib.fixes import _mosaicBlocksToRows

        min_us = 60_000_000
        blocks = [(0, 10 * min_us, 100, 1, 9.0), (10 * min_us, 20 * min_us, 140, 1, 9.0)]
        r = dict(lsegment=0, fsegment=0, fct_start=0, fct_end=20 * min_us, lag=0, lag_mod=0, n_pairs=0, resolved=False, dense=False)
        rows = _mosaicBlocksToRows(blocks, r, 65535)
        assert [x["lag"] for x in rows] == [100, 140]


class TestMosaicMappingActive:
    @staticmethod
    def _cfg(fixes_, exclude=None):
        from VISSSlib import tools

        c = tools.DictNoDefault({"dataFixes": fixes_, "mosaicFrameMappingExclude": exclude})
        return c

    def test_switch_and_exclusion(self):
        from VISSSlib.fixes import mosaicMappingActive

        assert not mosaicMappingActive(self._cfg(["x"]), "20230325")
        c = self._cfg(["mosaicFrameMapping"])
        assert mosaicMappingActive(c, "20230325")
        c = self._cfg(["mosaicFrameMapping"], ["20230325", "20230401-20230410"])
        assert not mosaicMappingActive(c, "20230325")
        assert not mosaicMappingActive(c, "20230325-120000")
        assert not mosaicMappingActive(c, "20230405")
        assert mosaicMappingActive(c, "20230411")
        # file names: the last date counts
        assert not mosaicMappingActive(
            c, "/x/20230101/level1match_V1.2_sail_visss01_visss_leader_S1145792_20230405-135000.nc"
        )
        assert mosaicMappingActive(
            c, "/x/20230405/level1match_V1.2_sail_visss01_visss_leader_S1145792_20230412-135000.nc"
        )
