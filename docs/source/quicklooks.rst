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

``generate()`` — a dispatcher with a latent bug
-----------------------------------------------------

:func:`VISSSlib.quicklooks.generate` is a ``level`` → quicklook-function
dispatcher, structurally identical to :func:`VISSSlib.distributions._createLevel2`'s
``sublevel`` dispatch. It is not wired into the CLI (there's no
``quicklooks.generate`` entry in ``tools._create_parser()``), and no
processing module calls it either — only ``tests/test_quicklooks.py``
exercises it directly. It has a copy-paste bug: its ``level == "level2track"``
branch calls :func:`~VISSSlib.quicklooks.createLevel2matchQuicklook`
instead of :func:`~VISSSlib.quicklooks.createLevel2trackQuicklook` (the
correct function, and the one
:func:`VISSSlib.distributions.createLevel2track` itself calls directly).
The existing test doesn't catch this: it only asserts the returned figure
object is not ``None``, not which quicklook function actually produced it,
so the wrong-but-still-non-``None`` result passes silently — a good
reminder that "the test passes" and "the test would catch this bug" are
different claims.

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

1. Every ``level1match`` file of the day is read (first 30000 particles per
   file) and particles are kept that pass ``minBlur``, ``minSize`` (both
   cameras) and ``config.quality.minMatchScore``.
2. The ``level1match`` particles are linked to their track via the
   ``level1track`` files (``pair_id`` -> ``track_id``). Particles that are not
   part of a track and tracks shorter than ``minTrackLength`` are dropped.
3. Each track gets a score between 0 and 1, the weighted mean of the
   min-max normalized track length, mean blur and mean Dmax
   (``weightTrackLength``, ``weightBlur``, ``weightDmax``).
4. Starting from the best track, GIFs are rendered until ``targetGifs`` GIFs
   exist. Tracks for which the particle images are missing (images are only
   stored for a subset of the particles, see ``imagesL1detect``) or which
   have fewer than ``minRenderedFrames`` images are skipped and replaced by
   the next best track, looking at most at
   ``targetGifs * candidateMultiplier`` tracks.

The defaults can be set in the settings file (see :doc:`config_files`):

.. code:: yaml

    level1trackParticlesQuicklook:
      minBlur: 15
      minSize: 30
      minTrackLength: 5
      minRenderedFrames: 3
      targetGifs: 10

If no track passes the filters, the HTML page states so instead of showing
GIFs. Like the other quicklooks, nothing is done if the output is newer than
``level1match``, ``level1track``, ``metaEvents`` and ``level0``
(``skipExisting=True``) or if the ``level1match`` data are not complete yet.
Unlike the other quicklooks, the return value is
``(html file, number of GIFs)``. The scale bar in the GIFs requires
``matplotlib-scalebar``.

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
