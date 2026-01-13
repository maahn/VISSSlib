import os
import sys
import tempfile
import numpy as np
import pytest
import yaml
from VISSSlib.tools import readSettings, get_package_file_path, DictNoDefault, BlockImageArchive
from VISSSlib.files import FindFiles
from VISSSlib.products import DataProduct
from VISSSlib.av import doubleDynamicRange
from VISSSlib.fixes import captureIdOverflows

def test_import_package():
    """Test that the main package can be imported"""
    import VISSSlib
    assert VISSSlib.__name__ == 'VISSSlib'

def test_import_submodules():
    """Test that key submodules can be imported"""
    from VISSSlib import analysis, av, files, fixes, products, tools
    assert analysis.__name__ == 'VISSSlib.analysis'
    assert av.__name__ == 'VISSSlib.av'
    assert files.__name__ == 'VISSSlib.files'
    assert fixes.__name__ == 'VISSSlib.fixes'
    assert products.__name__ == 'VISSSlib.products'
    assert tools.__name__ == 'VISSSlib.tools'

def test_read_settings(tmp_path):
    """Test reading and validating settings files"""
    # Create sample settings file
    settings_file = tmp_path / "settings.yaml"
    settings_data = {
        "leader": "leader_test",
        "follower": "follower_test",
        "path": "/data/raw",
        "pathOut": "/data/products",
        "rotate": {
            "2025-01-01": 180
        },
        "computers": {
            "visss11gb": {"fps": 140, "resolution": 58.75}
        }
    }
    with open(settings_file, 'w') as f:
        yaml.dump(settings_data, f)

    # Read and validate settings
    config = readSettings(str(settings_file))
    assert config.leader == "leader_test"
    assert config.follower == "follower_test"
    assert config.path == "/data/raw"
    assert config.pathOut == "/data/products"
    assert config.rotate['2025-01-01'] == 180
    assert config.computers.visss11gb.fps == 140

    # Verify DictNoDefault behavior
    with pytest.raises(AttributeError):
        _ = config.nonexistent_key

def test_file_finding(tmp_path):
    """Test file discovery functionality"""
    # Create sample settings
    settings_file = tmp_path / "settings.yaml"
    settings_data = {
        "leader": "leader_test",
        "follower": "follower_test",
        "path": str(tmp_path / "raw"),
        "pathOut": str(tmp_path / "products"),
        "movieExtension": "mkv"
    }
    with open(settings_file, 'w') as f:
        yaml.dump(settings_data, f)
    config = readSettings(str(settings_file))

    # Create sample directory structure
    raw_dir = tmp_path / "raw"
    (raw_dir / "leader_test").mkdir(parents=True, exist_ok=True)
    (raw_dir / "follower_test").mkdir(parents=True, exist_ok=True)
    (raw_dir / "leader_test" / "20250101_000000.mkv").touch()
    (raw_dir / "follower_test" / "20250101_000000.mkv").touch()

    # Test file finding
    ff = FindFiles(config)
    files = ff.listFilesExt(level=0)
    assert len(files) == 2
    assert any("leader_test" in f for f in files)
    assert any("follower_test" in f for f in files)

def test_data_products(tmp_path):
    """Test data product handling"""
    # Create sample settings
    settings_file = tmp_path / "settings.yaml"
    settings_data = {
        "leader": "leader_test",
        "follower": "follower_test",
        "path": str(tmp_path / "raw"),
        "pathOut": str(tmp_path / "products")
    }
    with open(settings_file, 'w') as f:
        yaml.dump(settings_data, f)
    config = readSettings(str(settings_file))

    # Create sample data product
    product = DataProduct(
        level=1,
        case="20250101",
        settings=config,
        fileQueue=None,
        camera="leader_test"
    )
    assert product.level == 1
    assert product.camera == "leader_test"
    assert product.case == "20250101"

def test_image_processing():
    """Test image processing utilities"""
    # Create sample image
    img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    
    # Test dynamic range expansion
    processed = doubleDynamicRange(img)
    assert processed.shape == img.shape
    assert processed.dtype == np.uint8

    # Test capture ID overflow handling
    data = {"pid": np.array([10000, 20000, 30000])}
    fixed = captureIdOverflows(data, config=DictNoDefault({"maxCaptureId": 25000}))
    assert fixed["pid"].max() < 25000

def test_block_archive(tmp_path):
    """Test block archive operations"""
    archive_file = tmp_path / "test.block"
    
    # Create and write to archive
    with BlockImageArchive(archive_file, mode="w") as archive:
        for i in range(10):
            img = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
            archive.write(f"image_{i}", img)
    
    # Read from archive
    with BlockImageArchive(archive_file, mode="r") as archive:
        for i in range(10):
            img = archive.read(f"image_{i}")
            assert img.shape == (10, 10)
            assert img.dtype == np.uint8

def test_get_package_file_path():
    """Test package resource path resolution"""
    # Test with a file that should exist
    path = get_package_file_path('__init__.py')
    assert path.endswith('VISSSlib/__init__.py')
    assert '__init__.py' in path
    
    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        get_package_file_path('non_existent_file.txt')

if __name__ == '__main__':
    pytest.main([sys.argv[0]])
