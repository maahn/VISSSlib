Installation
============



Install conda/mamba dependencies

.. code:: console

    conda install numpy  scipy  xarray  dask[complete]  pandas pyyaml matplotlib bottleneck pillow  addict opencv Pillow netcdf4 ipywidgets trimesh=4.0.5 scikit-image tqdm filterpy flox portalocker numba xarray-extras

Install PIP dependencies

.. code:: console

    pip install image-packer flatten_dict pyOptimalEstimation vg manifold3d==2.2.2

Clone the library with 

.. code:: console

    git clone https://github.com/maahn/VISSSlib

and install with

.. code:: console

    cd VISSSlib
    pip install -e .
