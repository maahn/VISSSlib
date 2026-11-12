import numpy as np
import pytest

from VISSSlib import blurTables, distributions


def test_default_is_generation_table():
    for gen in ("visss", "visss2", "visss3"):
        t = distributions._blurThreshold(gen)
        assert t.ndim == 1 and len(t) > 100
        assert np.array_equal(
            t, distributions._blurThreshold(gen, None), equal_nan=True
        )


def test_named_table_replaces_generation_table():
    for name, values in blurTables.BLUR_TABLES.items():
        t = distributions._blurThreshold("visss3", name)
        assert np.array_equal(t, np.array(values, dtype=float))
        # non-increasing: larger particles need less blur to count as sharp
        assert np.all(np.diff(t) <= 0)
        assert not np.array_equal(t, distributions._blurThreshold("visss3"))


def test_unknown_table_raises():
    with pytest.raises(ValueError):
        distributions._blurThreshold("visss3", "doesNotExist")
