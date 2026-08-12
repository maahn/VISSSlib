``products`` - the processing DAG
=====================================

Motivation
----------

``products.py`` is where the whole per-level pipeline (:doc:`metadata`,
:doc:`detection`, :doc:`matching`, :doc:`tracking`, :doc:`distributions`,
:doc:`level3`) gets tied together into one dependency graph and driven
end-to-end, either directly or via a task queue for cluster/multi-worker
processing. :class:`VISSSlib.products.DataProduct` is the whole module's
center of gravity; everything else (the per-level subclasses,
:func:`~VISSSlib.products.processAll`,
:func:`~VISSSlib.products.submitAll`,
:func:`~VISSSlib.products.processRealtime`) is a thin layer on top of it.

The dependency graph lives in code, not config
----------------------------------------------------

:meth:`VISSSlib.products.DataProduct.__init__` contains an explicit
``if/elif`` chain (keyed on ``level``) that is the single authoritative
statement of "what depends on what" for the whole library — e.g.
``level1match``'s only parent is ``{camera}_metaRotation``,
``level2match``'s parents are ``{camera}_level1match`` plus **both**
cameras' ``metaEvents`` (added explicitly so a Level 2 product regenerates
whenever *more level0 data has been transferred*, since ``metaEvents`` is
the record of that, not just to satisfy a strict data dependency).
**Adding a new level requires editing this block** (and
:data:`VISSSlib.files.fileLevels`/``dailyLevels``, and
:meth:`VISSSlib.products.DataProduct.generateCommands`'s matching
``if/elif`` chain, and usually a small level-pinning subclass at the bottom
of the file like :class:`VISSSlib.products.level2track`) — there is no
single declarative place, three call sites have to stay in sync by hand.

Construction is recursive and eager
-----------------------------------------

Constructing a ``DataProduct`` for one level **recursively constructs a
``DataProduct`` for every ancestor**, all the way down to ``level0`` — the
whole relevant subgraph is built up front, not lazily. ``childrensRelatives``
is a dict threaded through the recursion so a parent shared by two branches
(e.g. both ``leader_metaEvents`` and ``follower_metaEvents`` are parents of
several levels) is only constructed once and then reused, rather than
rebuilt per branch.

Commands, not direct calls
------------------------------

:meth:`VISSSlib.products.DataProduct.generateCommands` does not call
e.g. :func:`VISSSlib.detection.detectParticles` directly — it builds a
**shell command string** that re-invokes the same
``python -m VISSSlib <module.function> ...`` CLI documented in
:doc:`command_line`, one command per level0 input file (via
:meth:`~VISSSlib.products.DataProduct._commandTemplateL1`) or one per case
(via :meth:`~VISSSlib.products.DataProduct._commandTemplateDaily`). This is
what makes the same DAG usable both for direct in-process execution
(:func:`VISSSlib.products.processAll`) and for distributed processing: the
commands are exactly what gets pushed onto the
:func:`VISSSlib.tools.runCommandInQueue` task queue for
:func:`VISSSlib.tools.workers` (potentially on other machines/SLURM nodes,
see ``scripts/VISSSlib_slurm.sh``) to execute.
:meth:`~VISSSlib.products.DataProduct.generateAllCommands` decides, for
*each* level in the ancestor chain independently, whether it's worth
generating commands at all — only once its own parents are already
complete and not younger than *their* parents — and otherwise just recurses
further down. This is why :func:`VISSSlib.products.submitAll` (and
:meth:`VISSSlib.products.DataProduct.process`) are documented as sometimes
needing to be called repeatedly: a single call only submits work for
whichever part of the chain is *currently* ready, not the full pipeline in
one shot, when several levels are simultaneously missing.

Entry points
-------------

- :func:`VISSSlib.products.processAll` walks a fixed, explicit list of
  levels (not the general parent-recursion above) and, for each, submits
  commands and drains the queue with :func:`VISSSlib.tools.workers` before
  moving to the next level — a synchronous, one-level-at-a-time barrier.
  Its own docstring calls this "a rather unefficient way of processing the
  data and mostly for testing".
- :func:`VISSSlib.products.submitAll` is the production path: it builds one
  ``DataProductRange("allDone", ...)`` (a
  :class:`~VISSSlib.products.DataProductRange`, the
  :class:`VISSSlib.files.FindFilesRange`-style wrapper for a whole case
  range) and pushes *all* currently-generatable commands across the whole
  DAG into the queue at once, for however many :func:`VISSSlib.tools.workers`
  (potentially remote) to drain in parallel — see ``worker`` in
  :doc:`command_line`.
- :func:`VISSSlib.products.processRealtime` is the "cheap, run-often"
  subset (metadata/quicklooks) meant to be invoked frequently on the
  acquisition/near-real-time side without the expensive Level 1/2/3
  processing.

``VISSSlib.products`` API
-----------------------------

.. automodule:: VISSSlib.products
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
