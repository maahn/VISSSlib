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

Frame mapping (``frameMapping``)
----------------------------------------

For deployments whose cameras are not hardware-triggered (or whose
``capture_id`` / ``capture_time`` are unreliable, e.g. MOSAiC, ``hyytiala_v1``,
``gochang_v1``, ``sail_v1``) the lag between the frames of the leader and the
follower cannot be taken from the record time alone. The dataFix
``frameMapping`` determines it explicitly, once per day, in
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
tracked from the strongest block in both directions (the search window follows
the drift and grows with the distance to the last block that was found, so a
lag that drifts far from the first guess, e.g. -5 to -164 frames over a day, is
still followed), interpolated at 1 minute resolution and extrapolated through
quiet blocks. Because the single-frame steps of the interpolated lag can sit a
few minutes away from the real ones (the minutes in between then match almost
nothing), every minute is finally re-tested against the lags around the
interpolated one (``fineLags``, :func:`VISSSlib.fixes._mosaicRefineRowLags`,
+-6 frames, 3 minute follower window). A new lag is taken only where the
particles show it decisively (peak, vertical offset consistent with the day); a
change of more than one frame needs a neighbouring minute with the same change,
and minutes without a decisive result take the nearest decided changes. Rows that
were resolved by single particles are checked as well, because the lag can drift
inside a long row. Ghost frames of the follower
and the clock offset between the cameras (knots per minute) are stored too.
The follower ``capture_id``/``capture_time`` are converted to the leader's in
:func:`VISSSlib.fixes.mosaicFollowerCaptureIdToLeader`; follower particles
without a reliable mapping are dropped, and level2 flags such times
(``processingFailed``, :func:`VISSSlib.fixes.mosaicUnmappedTimes`).

*Settings* (see :doc:`config_files`):

- ``frameMappingExclude``: days (``YYYYMMDD`` or ``YYYYMMDD-YYYYMMDD``)
  for which the fix is **not** applied. Use it for days that were processed
  well without the mapping (light snow, where the particle tests have too little
  to work with).
- ``frameMappingFallback``: if the mapping variant of a file fails (no
  follower data left, no matched particles, Z residual too wide, ...), the file
  is processed once more **without** the mapping
  (:func:`VISSSlib.matching.matchParticles`). The settings
  ``frameMappingFallbackFixes`` lists the dataFixes that variant needs (e.g.
  ``makeCaptureTimeEven``). The result carries the attribute
  ``mosaicFrameMapping = "fallback"`` and level2 does not flag its time as
  unmapped. Rotation estimation is never done with the fallback.
  The fallback is also used if the mapping variant pairs less than 2% of the
  particles of the smaller camera (at least 1000 particles), i.e. if almost
  everything was dropped or mismatched.
- Related opt-in fixes: ``rejectRotationJumps`` (a per-file rotation retrieval
  far from the recent ones is not adopted), ``rotationNoRetry``.

*Pitfalls.* Do not combine it with ``captureIdOverflows`` for 32-bit
``capture_id`` (``gochang_v1``, ``sail_v1``); a day without frame mapping loses
its follower data, which is why the fallback exists; the mapping tests need
particles, so the weakest files rely on the fallback and have lower match
scores. Measured on the days that had problems without the mapping (October
2026): hyytiala_v1 2.0x the matched pairs and 27x fewer broken level1match files
(1081 to 40); sail_v1 1.16x the pairs and 6x fewer broken files (560 to 91).
The minute-wise lag refinement was applied to all days that had problems
(98 hyytiala_v1 and 48 sail_v1 days; the 65 sail days that are fine without the
mapping stay excluded). Over these days the matched pairs rose by 1.01x
(sail) and 1.04x (hyytiala) on top of the earlier gain, the minutes in which less than 0.3x the usual
fraction of the particles is matched fell from 6558 to 4550 (sail) and from
18394 to 8208 (hyytiala), the files with a median match score below 0.02
from 1305 to 1192 and from 1447 to 961. Some days stay partly unmatched, e.g.
sail 20230415 in the evening (a few frames of the lag still move inside a file)
and dense-snow periods in which no lag can be determined (hyytiala 20220405
06:20-07:00).

``VISSSlib.fixes`` API
--------------------------

.. automodule:: VISSSlib.fixes
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
