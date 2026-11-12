``distributions`` - calibration and Level 2 distributions
==============================================================

Motivation
----------

``distributions.py`` (the largest core module, ~3000 lines) turns the
per-particle, pixel-unit Level 1 products (``level1detect``/
``level1match``/``level1track``) into the Level 2 products described in
the paper's "Time-resolved particle properties" section: calibrated
(metric-unit), one-minute-resolved particle size distributions (PSDs) plus
aggregate statistics. It imports ``from .matching import *`` — it reuses
the rotation-transform functions (:func:`VISSSlib.matching.shiftRotate_F2L`
etc.) for the observation-volume geometry below, not just for particle
matching.

Entry points and chunking
-----------------------------

:func:`VISSSlib.distributions.createLevel2detect` (per-camera, decorated
with :func:`VISSSlib.tools.loopify_with_camera`),
:func:`VISSSlib.distributions.createLevel2match`, and
:func:`VISSSlib.distributions.createLevel2track` (both cameras combined,
:func:`VISSSlib.tools.loopify`) are thin dispatchers into
:func:`VISSSlib.distributions._createLevel2` with ``sublevel="detect"/
"match"/"track"``, followed by triggering the corresponding quicklook.

:func:`VISSSlib.distributions._createLevel2` does the completeness checks
(via :class:`VISSSlib.files.FindFiles`'s ``isCompleteL1*`` properties —
refuses to produce a Level 2 day until every expected Level 1 file for that
day exists) and then — for performance — **splits a full-day case into 24
hourly sub-cases** and calls
:func:`VISSSlib.distributions._createLevel2part` once per hour, concatenating
the results along ``time`` afterward. This chunking exists purely for
memory/runtime reasons; it is invisible in the output (a case parameter like
``"20260110-08"`` handed to ``_createLevel2part`` directly skips the
hourly split and processes just that hour, which is also how tests keep
Level 2 tests fast).

``applyFilters`` — the quality-selection DSL
-------------------------------------------------

Both ``createLevel2*`` functions accept an ``applyFilters`` list, each
entry a 4-tuple: ``(variable, operator, value, cameraSelector, extraDimSel)``.
``operator`` is one of ``>``/``<``/``>=``/``<=``/``==``
(:data:`VISSSlib.distributions._operators`); ``cameraSelector`` is
``"min"``/``"max"``/``"mean"`` (:data:`VISSSlib.distributions._select`) —
since a matched/tracked particle has one value per camera (or per track
step), filters must first decide how to reduce that to a single number;
``value`` can also be a two-element ``[intercept, slope]`` pair, in which
case the threshold is itself linear in ``Dmax`` rather than a constant
(e.g. a size-dependent aspect-ratio cutoff). This mechanism is what lets
callers implement ad-hoc quality selections without a code change.

Single-camera ("detect") volume correction
-----------------------------------------------

For ``sublevel="detect"`` (single camera, no stereo constraint on the
observation volume), :func:`VISSSlib.distributions._createLevel2part`
applies a **hardcoded, empirically derived per-``Dmax``-bin blur
threshold** (for ``visssGen == "visss"``) instead of a geometric volume
cutoff — the comment above the table explains it was derived by comparing
cumulative detect-vs-match PSDs in Hyytiälä winter 2021/22 to find, per
size bin, the blur threshold that makes the single-camera distribution
agree with the (volume-constrained) matched one. This is the concrete
implementation of the paper's aside that a single-camera product "would
also be possible... using a threshold based on particle blur to define the
observation volume, similar to the PIP" — it exists, and is size-dependent
rather than a single global blur cutoff.

The tables of the VISSS generations are tied to the camera pair they were
derived with. Cameras with a different image contrast need their own table:
``level2.blurTable`` selects one from :mod:`VISSSlib.blurTables` (so far
``hyytiala2_v3``, ``hyytiala2_v4``, ``hyytiala2_v5``). They are derived in the
same way: the ratio of volume corrected ``level1detect`` and ``level1match``
size distributions gives, per size, the fraction of detected particles that
are sharp enough to be matched, and the threshold is that percentile of the
detected blur. Leader and follower are averaged, smoothed and made
non-increasing; the module docstring lists the days used and the hold-out
check. Because the table is a stand-in for the observation volume, a
too-low threshold gives an excess of the single-camera over the matched
distribution and a too-high one a deficit.

With ``level2.blurScaleFromOtherCamera`` the threshold of the camera that
sees the particles fainter is scaled down by an hourly factor (75th percentile
of the blur of 8-16 px particles of this camera relative to the other one,
at most 1, at least 0.2, 1 if fewer than 300 particles). It was developed for
the Ny-Alesund VISSS2 and tested there only.

Creating a blur table
~~~~~~~~~~~~~~~~~~~~~

A table is needed for a new camera pair (or if leader and follower
``level2detect`` PSDs disagree systematically, with the sign pointing to the
blur threshold: single-camera excess = threshold too low). It is derived with
``scripts/derive_blur_table.py`` in two steps:

1. Select 10-15 snowfall days spread over the deployment, with complete
   ``level1detect``, ``level1match`` and ``level2match`` and without known
   problems (``badData``, missed frames, out-of-focus periods).
2. ``collect``: per group of about 3 days (slow, about 1 h per day; run the groups
   in parallel) the script stores per camera the volume corrected 2D histogram of
   detected particles (``Dmax`` in pixels x log10 blur) and the size distribution
   of the matched particles::

     python scripts/derive_blur_table.py collect settings.yaml \
         20240122,20240214,20240322 calib_b0.pkl

3. ``derive``: pools the pickles, computes per size the factor by which the volume
   corrected detections exceed the matches, takes the blur percentile
   ``1 - 1/factor`` of the detected particles as threshold, averages leader and
   follower, smooths (median over +-2 sizes) and makes the table non-increasing.
   Each pickle is left out in turn as a hold-out check; ``R_pass`` (detections
   above the threshold / matches, 8-40 px) should be 0.8-1.2::

     python scripts/derive_blur_table.py derive hyytiala2_v3 calib_b*.pkl -o table.txt

4. Paste the text into ``blurTables.BLUR_TABLES``, list the days and the hold-out
   result in the docstring of ``blurTables.py``, set ``level2.blurTable`` in the
   settings file and rebuild ``level2detect`` (``scripts/reprocess_levels.py
   --levels level2detect --skip-existing false``). Afterwards compare the leader
   and follower PSDs: the ratio of their total concentrations should scatter around 1.

Known limitations
~~~~~~~~~~~~~~~~~

Leader and follower ``level2detect`` PSDs can disagree despite these
corrections in some periods; they should be compared before the
single-camera product is used quantitatively:

* Ny-Alesund v4 Aug-Sep 2024, v6 Oct-Nov 2026, Hyytiala v5 from late Oct 2026:
  one camera is out of focus or loses detections.
* Hyytiala VISSS3 v3 May 2024: leader excess up to about 30 %.
* Remaining Ny-Alesund leader deficit of 7-13 %, of which about 7 % does not
  depend on size and is not related to blur.
* Missed frames: the PSD uses the nominal frame rate, so frames that were not
  recorded bias it low and the cameras miss different frames. This is the
  reason for the ``badData`` period at Ny-Alesund 12 Aug - 11 Sep 2025
  (about 159 instead of 250 fps).

Binning
--------

The core of ``_createLevel2part`` groups particles into one-minute time
bins (``groupby_bins`` on ``time``) and, within each, into ``Dmax``/
``Dequiv`` pixel bins (``DbinsPixel``, default 1px steps 0–300) via
``groupby_bins`` again — directly matching the paper's PSD binning
description. Both ``level2match``/``level2track`` use camera- or
track-reduced values (see ``applyFilters`` above and
:func:`VISSSlib.distributions.getPerTrackStatistics` for the track-based
min/max/mean/std reduction along a track) as the basis for statistics.

Observation volume: mesh intersection, not OpenSCAD
---------------------------------------------------------

The paper states the leader/follower observation-volume intersection is
computed "using the OpenSCAD library"; the actual implementation
(:func:`VISSSlib.distributions.createLeaderBox`,
:func:`VISSSlib.distributions.createFollowerBox`,
:func:`VISSSlib.distributions._createBox`,
:func:`VISSSlib.distributions._estimateVolume`) instead builds each
camera's observation volume as an 8-vertex box with :mod:`trimesh`
(boolean-intersection backend ``manifold3d``, an ``installation.rst``
dependency) and computes ``leader.intersection(follower).volume`` directly
— conceptually the same "rotate follower's box into the leader frame, then
intersect" approach the paper describes (follower vertices are transformed
via :func:`VISSSlib.matching.shiftRotate_F2L` using the retrieved
``camera_phi``/``camera_theta``/``camera_Ofz``), just a different concrete
library than what's written up. Worth knowing if you're trying to reproduce
the paper's numbers from the library alone. :func:`VISSSlib.distributions._estimateVolumes`
(``@functools.cache``d, since the same camera geometry is reused across many
size bins/cases) computes this per size bin, shrinking each box's edges to
account for :math:`D_\text{max}`-dependent partial-particle exclusion, per
the paper's ``effective observation volume reduced by`` :math:`D_\text{max}/2`
description.

``tests/test_distributions.py::TestVolume`` is a good sanity check to know
about if you're touching this geometry: ``test_volumeEstimate`` asserts
that with **zero** rotation/offset, the intersection volume collapses to
exactly ``width * width * height`` (the cameras observe the same cuboid
directly), and ``test_VolumeInterpolation`` asserts that
``_estimateVolumes``' size-bin interpolation (computing the expensive
mesh intersection at only ``nSteps`` bins and interpolating the rest, for
performance) stays within 1% of computing every bin exactly — i.e. the
interpolation shortcut is verified numerically, not just assumed safe.

``VISSSlib.distributions`` API
----------------------------------

.. automodule:: VISSSlib.distributions
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
