metaRotation - retrieve camera rotation
=======================================


Motivation
----------

Although alignment of both observation volumes is a priority during
installation, the cameras can be rotated or displaced, i.e., misaligned.
As a result, the same particle may be observed at different heights and
:math:`z = Y_\textrm{L} = Y_\textrm{F}` does not hold. The observed
offsets are not constant and can change due to unstable surfaces or
pressure of accumulated snow on the VISSS frame. We could simply ignore
the misalignment and continue to take :math:`z` from the leader, but
this would not allow us to generally use the vertical position to match
particles from both cameras (see above). Also, offsets in :math:`z`
reduce the common observation volume of both cameras, which could lead
to biases when calibrating the PSDs if not accounted for.

Besides a constant offset in the vertical :math:`z` dimension
:math:`O_{\textrm{f}z}`, one of the cameras can also be rotated around
the optical axis (expressed analogously to aircraft coordinate systems
with roll :math:`\varphi`), around the horizontal axis perpendicular to
the optical axis (pitch :math:`\theta`), or around the vertical axis
(yaw :math:`\psi`). As a consequence,
:math:`\Delta z = Y_\textrm{L}-Y_\textrm{F}` depends on the position of
the particle in the observation volume.

To account for the misalignment, we attach the coordinate system to the
leader (i.e., we assume that the leader is perfectly aligned
(:math:`x_\textrm{L}`,\ :math:`y_\textrm{L}`,\ :math:`z_\textrm{L}`) =
(:math:`x`,\ :math:`y`,\ :math:`z`)) and retrieve the misalignment of
the follower with respect to the leader in terms of :math:`\varphi`,
:math:`\theta` and :math:`O_{\textrm{f}z}`. We cannot derive
:math:`\psi` from the observation and we have no choice but to neglect
it by assuming :math:`\psi = 0` to reduce the number of unknowns.
Mathematically, we need to transform the follower coordinate system
(:math:`x_\textrm{F}`,\ :math:`y_\textrm{F}`,\ :math:`z_\textrm{F}`) to
our leader reference coordinate system
(:math:`x_\textrm{L}`,\ :math:`y_\textrm{L}`,\ :math:`z_\textrm{L}`)
using rotation and shear matrices. In the appendix
`4` (see VISSS paper Maahn et al. 2024), we show how the transformation matrices can be
arranged so that the follower’s vertical measure :math:`z_\textrm{F}`
can be converted to :math:`z_\textrm{L}` depending on :math:`\varphi`
and :math:`\theta` with

.. math::

   \begin{aligned}
    \label{eq:coordinates}
       z_\textrm{L} =   -& \frac{  \sin\theta }{\cos\theta } x_\textrm{L}      + \frac{\sin \varphi}{\cos\theta } y_\textrm{F}      +  \frac{\cos \varphi}{\cos\theta } (z_\textrm{F} + O_{\textrm{f}z}) .
   \end{aligned}

This equation can be considered as a forward operator that calculates
the expected leader observation :math:`z_\textrm{L}` based on a
misalignment state (:math:`O_{\textrm{f}z}`, :math:`\varphi`, and
:math:`\theta`) and additional parameters (:math:`x_\textrm{L}`,
:math:`y_\textrm{F}`, :math:`z_\textrm{F}`). While we assume that the
misalignment state is constant for each 10 minute observation period,
the other variables (:math:`x_\textrm{L}`, :math:`y_\textrm{F}`,
:math:`z_\textrm{F}`) are available on a per-particle basis, combining
observations from both cameras. Therefore, we can use a Bayesian inverse
Optimal Estimation retrieval (Rodgers, 2000)
implemented by the pyOptimalEstimation library
(Maahn et al. 2020) to retrieve the misalignment
state from the actual observed :math:`z_\textrm{L}`.

The retrieved misalignment parameters are required for matching, but
retrieving the misalignment parameters requires matched particles. To
solve this dilemma, we use an iterative method assuming that
misalignment does not change suddenly. The method starts by using the
misalignment estimates and uncertainties (inflated by a factor of 10)
from the previous time period (10 minutes) to match particles of the
current time period. These particles are used to retrieve values for
:math:`\varphi`, :math:`\theta`, and :math:`O_{\textrm{f}z}` which are
used as a priori input for the next iteration of misalignment retrieval.
The iteration is stopped when the changes in :math:`\varphi`,
:math:`\theta`, and :math:`O_{\textrm{f}z}` are less than the estimated
uncertainties. For efficiency, the iterative method is applied only to
the first 300 observed particles and the resulting coefficients are
stored in the metaRotation product. A drawback of the method is that
this processing step requires processing the 10-minute measurement
chunks in chronological order, creating a serial bottleneck in the
otherwise parallel VISSS processing chain. Obviously, this method does
not work when no information is available from the previous time step,
e.g., after the instrument was set up or adjusted. To get the starting
point for the iteration, the matching algorithm is applied for frames
where only a single, relatively large (:math:`>` 10 px) particle is
detected, so that the matching can be done based on particle height
difference (:math:`\Delta h`) alone, ignoring vertical offset
(:math:`\Delta z`). [*]_




Run metaRotation
----------------

Run metaRotation with 

.. autofunction:: VISSSlib.scripts.loopCreateMetaRotation

Or in a shell script with

.. code:: console

    python3 -m VISSSlib scripts.loopCreateMetaRotation  $config_files.yaml $nDays $skipExisiting

The script is **not** parallelized because previous results are typically required.

Manual adjustments
------------------



To apply metaRotation for new deployments or when the instrument has been moved, the following code is recommended. Load libraries and set case

.. code:: ipython3

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import VISSSlib
    import importlib
    import yaml
    
    print(VISSSlib.__version__)
    
    settings = "/projekt1//ag_maahn/VISSS_config/hyytiala2_v3.yaml"
    case = "20240217-0940" 

    print(case)

    config = VISSSlib.tools.readSettings(settings)
    fl = VISSSlib.files.FindFiles(case, config.leader, config)
    fname1L = fl.listFiles("level1detect")[0]
    fname1Match = VISSSlib.files.FilenamesFromLevel(fname1L, config).fname["level1match"]


Use generic first guess rotation values with high uncertainties

.. code:: ipython3

    rotate_default = pd.Series(
        {
            "camera_phi": 0.0,
            "camera_theta": 0.0,
            "camera_Ofz": 0,
        }
    )
    #
    rotate_err_default = pd.Series(
        {
            "camera_phi": 1,
            "camera_theta": 1,
            "camera_Ofz": 50,
        }
    )

In the first step, use only large particles and do not use Z difference at all to allow
arbitrary offsets

The higher you set minDMax4rot (minimum Dmax of particles used), the better the results.

The option singleParticleFramesOnly makes sure only frames are used where a single particle is observed.

The value nSamples4rot can be increased if necessary but makes the estimation slow. 

.. code:: ipython3

    fout, matchedDat, rot, rot_err, _, _, _, errors = VISSSlib.matching.matchParticles(
        fname1L,
        config,
        doRot=True,
        rotationOnly=True,
        rotate=rotate_default,
        rotate_err=rotate_err_default,
        maxDiffMs="config",
        testing=False,
        minSamples4rot=90,
        minDMax4rot=15,
        singleParticleFramesOnly=True,
        nSamples4rot=1000,
        sigma={
            #             "Z" : 1.7, # estimated from OE results
            "H": 1.2,  # estimated from OE results
            "I": 0.01,
        },
    )
    rot, rot_err

Resulting in

.. parsed-literal::

    (camera_phi      -0.403621
     camera_theta    -0.753237
     camera_Ofz      79.605159
     dtype: float64,
     camera_phi      0.024806
     camera_theta    0.026622
     camera_Ofz      0.323117
     dtype: float64)

The results are used to run the algorithm again, but this time with default settings



.. code:: ipython3

    
    fout, matchedDat, rot2, rot_err2, nL, nF, nM, errors = VISSSlib.matching.matchParticles(
        fname1L,
        config,
        doRot=True,
        rotationOnly=True,
        rotate=rot,
        rotate_err=rot_err,
        nPoints=500,
        minSamples4rot=40,
    )
    nL, nF, nM

And again

.. code:: ipython3

    fout, matchedDat, rot3, rot_err3, _, _, _, errors = VISSSlib.matching.matchParticles(
        fname1L,
        config,
        doRot=True,
        rotationOnly=True,
        rotate=rot2,
        rotate_err=rot_err2,
        nPoints=500,
        minSamples4rot=40,
    )

And again


.. code:: ipython3

    fout, matchedDat, rot4, rot_err4, _, _, _, errors = VISSSlib.matching.matchParticles(
        fname1L,
        config,
        doRot=True,
        rotationOnly=True,
        rotate=rot3,
        rotate_err=rot_err3,
        nPoints=500,
        minSamples4rot=5,
    )

Now format the output so that we can copy paste it in the config files

.. code:: ipython3

    print(
        yaml.dump(
            {
                "rotate": {
                    case: {
                        "transformation": rot4.round(6).to_dict(),
                        "transformation_err": rot_err4.round(6).to_dict(),
                    }
                }  #
            }
        )
    )


.. parsed-literal::

    rotate:
      20240228-0340:
        transformation:
          camera_Ofz: 76.795276
          camera_phi: 0.491272
          camera_theta: -1.000598
        transformation_err:
          camera_Ofz: 0.310487
          camera_phi: 0.044387
          camera_theta: 0.021347
    

API
---

metaRotation is handled in matching.py, see :doc:`matching`

.. [*] The former text has been copied from Maahn, M., D. Moisseev, I. Steinke, N. Maherndl, and M. D. Shupe, 2024: Introducing the Video In Situ Snowfall Sensor (VISSS). Atmospheric Measurement Techniques, 17, 899–919, doi:10.5194/amt-17-899-2024.