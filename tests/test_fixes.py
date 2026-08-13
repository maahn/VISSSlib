import types

import numpy as np
import pytest
import xarray as xr
from VISSSlib.fixes import captureIdOverflows, revertIdOverflowFix


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
