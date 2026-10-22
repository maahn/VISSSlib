``fixes`` - campaign-specific data patches
==============================================

Motivation
----------

``fixes.py`` collects targeted workarounds for known artifacts in specific
historical data periods (mostly MOSAiC-era hardware/timing issues, see the
top-of-file comment block in :doc:`metadata`). It's a lookup table more
than an algorithm module — treat it as reference material to check when a
new deployment surfaces a *new* raw-data anomaly, not something to
understand line-by-line up front.

Which fixes are actually live
-----------------------------------

Most fixes are opt-in per deployment via the settings file's ``dataFixes``
list (see :doc:`config_files`), checked with e.g. ``if "captureIdOverflows"
in config.dataFixes:`` at the call sites (in :mod:`VISSSlib.tools`'s
``open_mf*`` family and in :mod:`VISSSlib.matching`) — cross-checking which
fix names are actually referenced anywhere outside this file:

- **Live, opt-in**: :func:`VISSSlib.fixes.captureIdOverflows` (16-bit
  ``capture_id`` wraparound on M1280 cameras),
  :func:`VISSSlib.fixes.makeCaptureTimeEven` (follower-only clock drift),
  and :func:`VISSSlib.fixes.makeCaptureTimeEvenBothCameras`
  (leader-*and*-follower clock drift ahead of leader-follower capture_id
  offset estimation in :func:`VISSSlib.matching._resolveMatchingOffset`
  -- for deployments where the *leader's* own onboard clock drifts too,
  not just the follower's), all gated on ``config.dataFixes``.
  :func:`VISSSlib.fixes.revertIdOverflowFix` undoes the id-overflow
  correction again right before :func:`VISSSlib.matching.matchParticles`
  writes its output, so the *shifted* ids used internally for matching
  don't leak into the product and confuse anyone comparing against the
  raw ``capture_id``.
- **Live, unconditional**: :func:`VISSSlib.fixes.delayedClockReset` is
  called directly from :mod:`VISSSlib.metadata` without a ``dataFixes``
  check — it always runs.
- **Dead code**: :func:`VISSSlib.fixes.fixMosaicTimeL1` has zero callers
  anywhere in the codebase, and its own docstring says so explicitly:
  *"This is a poor attempt at fixing drift and is not used anymore."*
  :func:`VISSSlib.fixes.removeGhostFrames` is referenced only inside a
  commented-out block in :func:`VISSSlib.metadata.getMetaData` ("unclear
  whether it works"). Both are candidates for removal if you're doing
  cleanup, but are left as-is here since neither was in scope for a
  documentation pass.

Frame mapping (``mosaicFrameMapping``)
----------------------------------------

For deployments whose cameras are not hardware-triggered (or whose
``capture_id`` / ``capture_time`` are unreliable, e.g. MOSAiC, ``hyytiala_v1``,
``gochang_v1``, ``sail_v1``) the lag between the frames of the leader and the
follower cannot be taken from the record time alone. The dataFix
``mosaicFrameMapping`` determines it explicitly, once per day, in
:func:`VISSSlib.fixes.createMosaicFrameMapping` (built at the start of
:func:`VISSSlib.matching.createMetaRotation`, cached as ``metaFrameMapping``
next to ``metaRotation``; it is a hidden parent of level1match, so touch the
day's ``metaRotation`` with :func:`VISSSlib.tools.touchOutputFile` after
rebuilding it).

*Method.* For every overlap of a leader run and a follower run (a follower
restart starts a new run): first guess from the record time and from the
cross correlation of the moving-pixel signal; then the exact lag from the
particles: (1) frames with a single particle in both cameras (the vertical
position difference is narrow for the correct lag only), (2) for dense snow all
particles of corresponding frames are paired and the lag with a sharp peak at
the vertical offset wins (:func:`VISSSlib.fixes._mosaicDenseLagTest`), (3) the
same with the camera rotation applied if the cameras are rotated against each
other, and a joint search of lag and rotation as last resort. Long runs are
checked **block by block** (10 min): the lag drifts steadily when the cameras
do not record exactly the same frame rate (SAIL: 2 frames per 10 min); it is
tracked from the strongest block in both directions, interpolated at 1 minute
resolution and extrapolated through quiet blocks. Ghost frames of the follower
and the clock offset between the cameras (knots per minute) are stored too.
The follower ``capture_id``/``capture_time`` are converted to the leader's in
:func:`VISSSlib.fixes.mosaicFollowerCaptureIdToLeader`; follower particles
without a reliable mapping are dropped, and level2 flags such times
(``processingFailed``, :func:`VISSSlib.fixes.mosaicUnmappedTimes`).

*Settings* (see :doc:`config_files`):

- ``mosaicFrameMappingExclude``: days (``YYYYMMDD`` or ``YYYYMMDD-YYYYMMDD``)
  for which the fix is **not** applied. Use it for days that were processed
  well without the mapping (light snow, where the particle tests have too little
  to work with).
- ``mosaicFrameMappingFallback``: if the mapping variant of a file fails (no
  follower data left, no matched particles, Z residual too wide, ...), the file
  is processed once more **without** the mapping
  (:func:`VISSSlib.matching.matchParticles`). The settings
  ``mosaicFallbackFixes`` lists the dataFixes that variant needs (e.g.
  ``makeCaptureTimeEven``). The result carries the attribute
  ``mosaicFrameMapping = "fallback"`` and level2 does not flag its time as
  unmapped. Rotation estimation is never done with the fallback.
- Related opt-in fixes: ``rejectRotationJumps`` (a per-file rotation retrieval
  far from the recent ones is not adopted), ``rotationNoRetry``.

*Pitfalls.* Do not combine it with ``captureIdOverflows`` for 32-bit
``capture_id`` (``gochang_v1``, ``sail_v1``); a day without frame mapping loses
its follower data, which is why the fallback exists; the mapping tests need
particles, so the weakest files rely on the fallback and have lower match
scores. Typical results (sail_v1, hyytiala_v1 days with problems before):
2x more matched pairs and 25x fewer broken files compared to the processing
without the mapping.

``VISSSlib.fixes`` API
--------------------------

.. automodule:: VISSSlib.fixes
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
