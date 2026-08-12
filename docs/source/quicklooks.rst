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

``generate()`` — a dispatcher with a latent bug, and unreachable
-----------------------------------------------------------------------

:func:`VISSSlib.quicklooks.generate` is a ``level`` → quicklook-function
dispatcher, structurally identical to :func:`VISSSlib.distributions._createLevel2`'s
``sublevel`` dispatch. It is not wired into the CLI (there's no
``quicklooks.generate`` entry in ``tools._create_parser()``) and nothing
in the codebase calls it — a ``grep`` for ``quicklooks.generate(`` outside
this file's own definition returns nothing. It also has a copy-paste bug:
its ``level == "level2track"`` branch calls
:func:`~VISSSlib.quicklooks.createLevel2matchQuicklook` instead of
:func:`~VISSSlib.quicklooks.createLevel2trackQuicklook` (the correct
function, and the one :func:`VISSSlib.distributions.createLevel2track`
itself calls directly). Currently harmless since the function is
unreachable, but worth fixing (or removing) before anything starts calling
it.

Particle image montages (``Packer_patched``)
--------------------------------------------------

The random-particle-pair grid figures (like Fig. 7 in :doc:`visss_paper_r2`)
are built with :class:`VISSSlib.quicklooks.Packer_patched`, a small patch
of the third-party ``image_packer`` library's bin-packing (bottom-left-fill)
layout algorithm — patched specifically to accept in-memory ``PIL.Image``
objects directly rather than requiring each image to exist as a file on
disk first, since packing tens of thousands of already-in-memory particle
crops through a temp-file round trip would be wasteful.

``VISSSlib.quicklooks`` API
-------------------------------

.. automodule:: VISSSlib.quicklooks
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
