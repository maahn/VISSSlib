import pytest
import VISSSlib


def test_import_package():
    """Test that the main package can be imported"""
    import VISSSlib

    assert VISSSlib.__name__ == "VISSSlib"
