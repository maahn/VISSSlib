``quicklooks`` - plotting
============================

Motivation
----------

``quicklooks.py`` is the largest file in the library (~3900 lines) but
architecturally the simplest: one function per data level
(``level0Quicklook``, ``metaFramesQuicklook``, ``metaRotationQuicklook``,
``createLevel1detectQuicklook``, ``createLevel1matchParticlesQuicklook``,
``createLevel2detectQuicklook``, ``createLevel2matchQuicklook``,
``createLevel2trackQuicklook``, ``createLevel3RimingQuicklook``, ...),
each opening the relevant product file(s), building one or more matplotlib
figures, and saving them via :func:`VISSSlib.tools.savefig` (which handles
directory creation, permissions, the logo/status-text overlay, and
:func:`VISSSlib.tools.copyCurrentQuicklook` for a "latest" copy — see
:doc:`tools`). Each is wired into :doc:`products`/:doc:`distributions` next
to the product it visualizes (e.g.
:func:`VISSSlib.distributions.createLevel2track` calls
:func:`~VISSSlib.quicklooks.createLevel2trackQuicklook` when ``doPlot=True``)
rather than through the DAG's own dependency mechanism — quicklooks are a
side effect of running the corresponding processing function, not a
separate DAG level.

``generate()`` — the level dispatcher
-------------------------------------------

:func:`VISSSlib.quicklooks.generate` is a ``level`` → quicklook-function
dispatcher, structurally identical to :func:`VISSSlib.distributions._createLevel2`'s
``sublevel`` dispatch. It is not wired into the CLI (there's no
``quicklooks.generate`` entry in ``tools._create_parser()``), and no
processing module calls it either — only ``tests/test_quicklooks.py``
exercises it directly. Its ``level == "level2track"`` branch used to call
:func:`~VISSSlib.quicklooks.createLevel2matchQuicklook` (a copy-paste bug that
the test did not notice, because it only asserts that *a* figure is returned);
it now calls :func:`~VISSSlib.quicklooks.createLevel2trackQuicklook`, and a unit
test checks which function gets dispatched.

Common look of the time series quicklooks
-----------------------------------------------

All quicklooks except the particle image montages (level1detect,
level1matchParticles, level1trackParticles) share one look, implemented by a
few helpers at the top of ``quicklooks.py``:

* ``_styled`` runs a quicklook function in a matplotlib ``rc_context``
  (``_QL_RC``: font sizes, light grid, no top/right spines), so the plotting
  code does not set font sizes by hand and the global matplotlib settings are
  left alone.
* Panels share a 00:00–24:00 time axis (``_formatDayAxes``, ticks every 3 h,
  labels only below the lowest panel); ``_stackedFigure`` builds the stack,
  with a spacer column that keeps the labels of twin axes clear of the colorbars.
* Quality flags (``cameras off``, ``blocked``, ``blowing snow``, ``sun glint``,
  ...), idle periods and restarts are drawn by ``_shadeSpans`` /
  ``_markRestarts`` as full-height spans (``broken_barh`` in axes coordinates)
  and collected in **one legend above the top panel** (``_legendAbove``)
  instead of a legend in every panel. Every flagged time step is drawn, also
  isolated ones, and the shading does not change the y-limits.
* Sizes are shown in mm, camera colors are consistent (leader red, follower
  purple), and days without data or without raw data get a small placeholder
  figure (``_placeholderFigure``) with the same title.
* level2detect / level2match / level2track share ``_plotLevel2Core`` for the
  size distribution with N_tot, characteristic sizes, shape parameters and
  particle counts with the match score (on a second y-axis).

Particle image montages (``Packer_patched``)
--------------------------------------------------

The random-particle-pair grid figures (like Fig. 7 in
`Maahn et al. 2024 <https://amt.copernicus.org/articles/17/899/2024/>`_)
are built with :class:`VISSSlib.quicklooks.Packer_patched`, a small patch
of the third-party ``image_packer`` library's bin-packing (bottom-left-fill)
layout algorithm — patched specifically to accept in-memory ``PIL.Image``
objects directly rather than requiring each image to exist as a file on
disk first, since packing tens of thousands of already-in-memory particle
crops through a temp-file round trip would be wasteful.

Track particles (``createLevel1trackParticles``)
-----------------------------------------------------

:func:`VISSSlib.quicklooks.createLevel1trackParticles` shows the particles
that were tracked best on a given day as animated GIFs (leader | follower side
by side, one frame per track step), arranged in a single, self-contained HTML
page (the GIFs are base64-embedded). It is called by
:func:`VISSSlib.distributions.createLevel2track` when ``doPlot=True`` and
writes the HTML page ``level1trackParticles/<year>/level1trackParticles_V1_<site>_<camera>_<YYYYMMDD>.html``
into the quicklook folder (``files.quicklookLevelsSep``, but ``.html`` instead
of ``.png``).

How the tracks are chosen:

1. Every ``level1track`` file of the day is read (first 30000 particles per
   file) and particles are kept that pass ``minSize`` (both cameras) and
   ``config.quality.minMatchScore``. ``level1track`` contains all required
   properties of the matched particles together with their ``track_id`` and
   ``track_step``, so ``level1match`` is not read.
2. Tracks (unique per file) are dropped if they have fewer than
   ``minTrackLength`` particles left or if, in either camera, no single track
   step has blur larger than ``minBlur``. Blurry steps of an otherwise sharp
   track are kept.
3. Each track gets a score between 0 and 1, the weighted mean of the
   percentile ranks of its track length, mean blur and mean Dmax
   (``weightTrackLength``, ``weightBlur``, ``weightDmax``). Ranks are used
   instead of the values so that a few extreme tracks do not dominate. With
   the default weights (1, 0.01, 3) large particles are strongly preferred;
   small particles only show up if there are not enough large tracks with
   particle images.
4. Starting from the best track, GIFs are rendered until ``targetGifs`` GIFs
   exist. Tracks for which the particle images are missing (images are only
   stored for a subset of the particles, see ``imagesL1detect``) or which
   have fewer than ``minRenderedFrames`` images are skipped and replaced by
   the next best track, looking at most at
   ``targetGifs * candidateMultiplier`` tracks. A GIF shows the first 40
   particles of a track in the order of ``track_step``.

The defaults can be set in the settings file (see :doc:`config_files`):

.. code:: yaml

    level1trackParticlesQuicklook:
      minBlur: 200
      minSize: 20
      minTrackLength: 5
      minRenderedFrames: 3
      targetGifs: 12
      weightTrackLength: 1
      weightBlur: 0.01
      weightDmax: 3

If no track passes the filters, the HTML page states so instead of showing
GIFs. Like the other quicklooks, nothing is done if the output is newer than
``level1track``, ``metaEvents`` and ``level0``
(``skipExisting=True``) or if the ``level1track`` data are not complete yet.
Unlike the other quicklooks, the return value is
``(html file, number of GIFs)``.

The GIFs are drawn with PIL as grey scale images (no matplotlib), with the
title, camera labels and a scale bar in a fixed text size. Small particle
crops are enlarged by an integer factor (nearest neighbour) to a panel size of
about 320 pixels, so the scale bar and its label stay readable. The scale
bar uses ``calibration.slope`` of the settings.

A track that looks interesting in the quicklook (the GIF title contains the
site, the block start time and the ``track_id``) can be exported on its own with
:func:`VISSSlib.quicklooks.createLevel1trackParticlesSingle`, optionally for
only one camera and without any annotations:

.. code:: python

    VISSSlib.quicklooks.createLevel1trackParticlesSingle(
        "20250115-023000", 2421, config, view="both", showLegend=True
    )

``VISSSlib.quicklooks`` API
-------------------------------

.. automodule:: VISSSlib.quicklooks
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
