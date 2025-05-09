import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="VISSSlib",
    use_scm_version={
        "version_scheme": "post-release",
    },
    author="Maximilian Maahn",
    description="VISSS processing library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maahn/VISSSlib",
    project_urls={
        "Bug Tracker": "https://github.com/maahn/VISSSlib/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_namespace_packages(where="src"),
    package_data={
        # Include any *.pkl files found in the "data" subdirectory of the
        # "VISSSlib" package:
        "VISSSlib": ["data/*.pkl"],
    },
    python_requires=">=3.11",
    install_requires=[
        "numpy",
        "scipy",
        "xarray",
        "xarray-extras",
        "dask[complete]",
        "pandas",
        "pyyaml",
        "trimesh",
        "flatten_dict",
        "matplotlib",
        "ipywidgets",
        "bottleneck",
        "pillow",
        "image-packer",
        "addict",
        "pyOptimalEstimation",
        "filterpy",
        "flox",
        "portalocker",
        "numba",
        "vg",
        "xarray_extras",
        "manifold3d",
        "task-queue",
        "psutil",
        "scikit-image",
        "scikit-learn==1.6.1",
        "tqdm",
        "pangaeapy",
    ],
    setup_requires=["setuptools_scm"],
)
# "opencv-python" -> conda version is not recongnized
# https://stackoverflow.com/questions/57821903/setup-py-with-dependecies-installed-by-conda-not-pip
