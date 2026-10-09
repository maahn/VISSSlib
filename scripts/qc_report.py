#!/usr/bin/env python
"""
Structural QC audit across level1detect -> level2track for a VISSSlib
deployment.

This is the reusable core of the QC playbook (see scripts/README_qc.md):
point it at any settings YAML and it scans every configured level/camera/day
for the problem categories that don't need visual judgment to spot --

  - missing      output files that should exist relative to level0 but don't
  - broken       ".broken.txt" sentinels (a command raised an exception).
                 For metaRotation specifically (a daily product, so this
                 stays cheap), each broken day is retried with current
                 code before being reported at all -- self-healing fixes
                 (see git history around "Self-heal low-matchScore
                 rotation failures") can already resolve a sentinel that
                 predates them, and reporting a since-fixed problem would
                 be noise. Pass --no-retry-broken-metarotation to skip
                 this and just list sentinels as found. Other, per-file
                 levels (level1detect, level1match, ...) are reported
                 as-is -- retrying every broken file inline wouldn't
                 stay cheap the way one-retry-per-day does.
  - nodata_susp  a ".nodata" sentinel that doesn't hold up. For
                 metaRotation, each one is verified directly: delete it
                 and reprocess the day with current code (it's a daily
                 product, so this is cheap) -- createMetaRotation
                 independently re-derives "no data" from the day's own
                 file counts rather than trusting the old sentinel, so a
                 day that rewrites the same nodata sentinel is confirmed
                 still genuinely gapped, and only a day that produces
                 something else gets reported. Other levels fall back to
                 a message-marker heuristic (does the message look like
                 it actually describes a raw-data/availability problem
                 rather than a confirmed "no precipitation" day -- this
                 generalizes a bug once seen in detection.py where
                 exactly that happened) since live-verifying every
                 per-file sentinel wouldn't stay cheap.
  - duplicate    more than one real output file for the same timestamp
  - corrupt      a real output file exists but fails to open with xarray,
                 or opens with zero timesteps (only checked with --integrity)
  - dag_stale    a day/level/camera that passes the plain file-count check
                 (isComplete: no missing files, no .broken.txt) but whose
                 real products.DataProduct.allComplete is False -- i.e. it
                 structurally "looks done" while the DAG considers it
                 stale: either older than a parent's newest output
                 (products.DataProduct._upToDateWithParents), a parent
                 level is itself incomplete
                 (products.DataProduct.parentsComplete), or its files
                 predate a pending code-driven reprocessing breakpoint
                 (tools.REPROCESS_AFTER -- e.g. "better QC" fixes that
                 don't touch any parent file, so the normal mtime-vs-parent
                 check can't see them). This is exactly the kind of gap a
                 pure file-count scan can never catch on its own: it was
                 how eriswil_v1's level2detect (178/178 real files, 0
                 missing/broken) was wrongly called "fully clean" in an
                 earlier QC pass despite every file predating a pending
                 REPROCESS_AFTER breakpoint. Always computed (no extra
                 file opens -- DataProduct.allComplete only stats/globs),
                 for every level/camera this scan checks.
  - reduced_coverage  a daily-aggregate level (level2detect/level2match/
                 level2track) completed (not itself missing/broken) for a
                 day where its per-file parent level (level1detect/
                 level1match/level1track respectively) has missing or
                 broken files that same day. The aggregate step has no way
                 to know its input was incomplete -- it just aggregates
                 whatever real parent files exist -- so this can silently
                 pass as "done" while representing only partial coverage.
                 Always computed (cheap: reuses missing/broken counts
                 already collected in the same scan, no extra file opens).
  - matchscore_suspect  a real level1match file whose matchScore is
                 suspiciously low even though it never raised (only
                 checked with --matchscore-check). matchParticles' own
                 quality gate only evaluates matchScore when a file has
                 more than config.newFileInt matched pairs -- a file with
                 few matched pairs (e.g. a quiet period) can have a
                 terrible matchScore (seen as low as ~1e-10) and still be
                 written as "successful" output, because the check that
                 would catch it never runs. This category flags real
                 files whose median matchScore falls below
                 --matchscore-threshold-factor times
                 config.quality.minMatchScore, and reports nPairs so it's
                 clear whether the quality gate was ever actually
                 evaluated for that file.
  - z_sigma_suspect  a real level1match file whose Z-consistency residual
                 (matching.zResidualSigma -- how much matched leader/
                 follower pairs disagree with the rotation model, in
                 pixels -- level1's native unit, unlike level2's SI/meter
                 output) exceeds config.quality.maxZSigma, checked with the same
                 --matchscore-check flag (one file open covers both
                 checks). A rotation refit can only correct a *biased*
                 residual, not shrink a *wide* one, so this is a more
                 specific signal of genuine correspondence ambiguity
                 (real wrong matches) than a low matchScore alone, which
                 several unrelated terms can also drag down -- see
                 matching.zResidualSigma's docstring for the empirical
                 basis (a confirmed-unfixable file had sigma ~5.5 with
                 ~zero bias; a file a rotation refit fully recovered had
                 sigma ~1.4).

  - matchscore_drop  a run of consecutive real level1match files whose
                 median matchScore is far below that of their neighbours
                 (only checked with --matchscore-drop). Opens the
                 matchScore of every file of a day and compares each file
                 with the median of the files around it, so it catches a
                 localized failure that a random sample or the absolute
                 threshold of --matchscore-check misses (e.g. Hyytiala
                 20241206 04:50: matchScore 0.0015 against 0.026 around
                 it, 10 instead of 100 thousand pairs). The usual cause is
                 an implausible per-file rotation in metaRotation (a
                 retrieval that locked onto false pairs, relaxing over the
                 next file or two): compare camera_Ofz/phi/theta of those
                 files with their neighbours and, if confirmed, see
                 fixes.despikeRotationWithMatches (evidence-based,
                 replaces only files whose pairs agree better with the
                 interpolated rotation) and rerun the children with
                 scripts/reprocess_levels.py. Other causes: follower
                 restart / unresolved frame mapping (MOSAiC), a timing
                 glitch.
  - level2_ratio_suspect  minutes in which level2detect shows particles
                 on BOTH cameras but level2match (and level2track) show
                 (almost) none, although no quality flag explains it
                 (only checked with --level2-compare). level2detect does
                 not depend on matching, so it is an independent
                 reference for every minute: a minute with Ntot >=
                 --level2-min-ntot in both level2detect files and a match
                 Ntot of zero or below --level2-ratio-factor (default 0.3) times the
                 day's typical match/detect ratio means that particles
                 were seen but lost in matching (found at MOSAiC: 19 %
                 of such minutes had exactly zero match Ntot because the
                 follower clock was ~1 s off the leader's, and Hyytiala
                 20241206 04:50-05:00 from a rotation outlier). The ratio
                 itself is site dependent (level2match Ntot can be above
                 or below level2detect), hence the comparison with the
                 day's own median. Minutes already carrying a recording/
                 processing/blocked/blowing-snow quality flag in
                 level2match are expected losses and not reported.

It deliberately does not attempt "does this look physically plausible"
checks (rotation-angle continuity, particle-count trends, PSD shape) --
those need a human looking at quicklooks; see scripts/README_qc.md for
which quicklook command to reach for once this script points at a
suspicious level/camera/period.

Nothing here is reimplemented from scratch: every category is built
directly on files.FindFiles's existing nMissing/listFiles/listBroken/
listNoData primitives (the same ones products.checkCompleteness and
products.DataProduct use), plus tools.isBadPeriod to exclude
already-known-and-accepted bad periods by default.

Usage
-----
    # full configured campaign range, all levels, structural checks only
    python scripts/qc_report.py settings.yaml --out qc_report.csv

    # a specific period, only two levels, with a netCDF-openability probe
    python scripts/qc_report.py settings.yaml --case 20231201-20231231 \\
        --levels level1detect,level1match --integrity sample --out qc.csv
"""

import argparse
import atexit
import os
import random
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd

from VISSSlib import files, matching, products, tools

LEVELS = [
    "level1detect",
    "metaRotation",
    "level1match",
    "level1track",
    "level2detect",
    "level2match",
    "level2track",
]

# levels only ever produced relative to the leader camera -- mirrors the
# camera-skip pattern already used in products.checkCompleteness /
# tools.reportLastFiles, kept in sync with those rather than re-derived
# from LEVEL_REGISTRY so the three stay consistent. level2detect is
# deliberately NOT here: unlike level2match/level2track (leader-only
# stereo-pair aggregates), level2detect calibrates each camera
# independently and has real per-camera output on both leader and
# follower -- skipping it for follower would silently leave half its
# output unchecked (this was a real bug here until 2026-09-09: the old
# "any level starting with level2" catch-all skipped it too).
LEADER_ONLY_LEVELS = {"metaRotation", "level1match", "level1track", "level2match", "level2track"}

# substrings that mark a .nodata message as describing a raw-data/
# availability problem rather than a confirmed no-precipitation
# determination -- the exact shape of the bug scripts/
# relabel_missing_movies_broken.py fixes retroactively for level1detect.
NODATA_SUSPECT_MARKERS = (
    "movie file",
    "not found",
    "thread",
    "transfer",
    "corrupt",
)

# daily-aggregate level -> the per-file level it's built from, used by the
# reduced_coverage check. Both sides here are leader-only products in this
# script's camera-skip convention (see LEADER_ONLY_LEVELS /
# _skip_for_camera), so there's no cross-camera merge to worry about.
# level2detect is left out here specifically because it's checked by the
# more general allComplete-based "dag_stale" check instead (see scan()) --
# that check already catches a level2detect day built from incomplete
# level1detect input via DataProduct.parentsComplete, per camera, without
# needing a same-pass parent count the way this dict's simpler
# missing/broken-count approach does.
PARENT_LEVEL = {
    "level2match": "level1match",
    "level2track": "level1track",
}


def _skip_for_camera(level, camera, config):
    return camera == config.follower and level in LEADER_ONLY_LEVELS


def _read_text(path):
    try:
        return Path(path).read_text()
    except OSError as e:
        return f"<unreadable: {e}>"


def _parse_broken(path):
    """Best-effort parse of a .broken.txt sentinel into (command, gist)."""
    lines = _read_text(path).splitlines()
    if len(lines) <= 1:
        return "n/a", (lines[0].strip() if lines else "<empty>")
    command = lines[1][9:].split(";")[-1].strip() if len(lines) > 1 else "n/a"
    gist = lines[-1].strip()
    return command, gist


def _looks_suspect(message):
    lower = message.lower()
    return any(marker in lower for marker in NODATA_SUSPECT_MARKERS)


def _retry_broken_metarotation(case, config):
    """
    Re-run createMetaRotation for a day whose metaRotation output is
    currently marked broken, and report whether it's still broken
    afterward.

    metaRotation now self-heals a large class of rotation-quality
    failures in-process (matchParticles/createMetaRotation refit the
    rotation from already-matched pairs rather than giving up -- see
    git history around "Self-heal low-matchScore rotation failures").
    A ".broken.txt" sentinel that predates that fix may already be
    stale: the same day, reprocessed with current code, can succeed
    outright. Reporting it as broken without retrying would be reporting
    a problem that's already solved. This is deliberately scoped to
    metaRotation, not every level: it's a daily product (at most one
    broken case per camera per day, unlike level1detect/level1match's
    per-10-minute-file granularity), so retrying every broken day in a
    scan is bounded and practical; the same "retry before reporting"
    idea would make a per-file-level scan impractically slow by default.

    Returns
    -------
    bool
        True if still broken after the retry, False if it's now fixed.
    """
    try:
        matching.createMetaRotation(case, config, skipExisting=False)
    except Exception as e:
        print(f"  retry of metaRotation {case} raised: {type(e).__name__}: {e}")
    ff = files.FindFiles(case, config.leader, config)
    return len(ff.listBroken("metaRotation")) > 0


def _verify_metarotation_nodata(nodataFile, case, config):
    """
    Confirm a metaRotation ".nodata" sentinel is a genuine, still-current
    data gap by deleting it and reprocessing the day with current code
    (the same "delete + rerun, see what comes back" check used to
    confirm confirmed nodata sentinels are on-purpose across VISSSlib --
    see scripts/qc_report.py's own git history and
    tools.isDataTransferPending / files.isGenuineDataGap for the same
    idea applied elsewhere). createMetaRotation independently re-derives
    "no data" from the day's actual level0/level1detect file counts
    (files.FindFiles.isGenuineDataGap) rather than trusting the old
    sentinel, so a day that rewrites the same nodata sentinel is
    confirmed still genuinely gapped; a day that instead produces real
    output (or goes broken) reveals the original sentinel was stale or
    wrong. Always run directly, never via a task queue -- metaRotation
    is inherently serial (each day's estimate carries forward from the
    previous day's), so it should never go through a queue regardless
    of context.

    Returns
    -------
    bool
        True if still confirmed genuinely nodata, False if reprocessing
        produced something else (real output, or broken).
    """
    os.remove(nodataFile)
    try:
        matching.createMetaRotation(case, config, skipExisting=False)
    except Exception as e:
        print(f"  retry of metaRotation {case} raised: {type(e).__name__}: {e}")
    ff = files.FindFiles(case, config.leader, config)
    return len(ff.listNoData("metaRotation")) > 0


def _check_match_quality(path, config, matchscore_factor):
    """
    Open a real level1match file once and check two independent quality
    signals that can both silently pass matchParticles' own gate.

    matchscore: median matchScore below matchscore_factor *
    config.quality.minMatchScore (matchParticles' own gate only runs at
    all when nPairs > config.newFileInt, so a low-nPairs file's matchScore
    can be arbitrarily bad and still "pass" untouched).

    zsigma: matching.zResidualSigma above config.quality.maxZSigma, ONLY
    checked for nPairs > config.newFileInt -- same reasoning as
    matchParticles' own gate: a std computed from a handful of pairs is
    itself noisy, and matching from few particles is inherently harder
    regardless (confirmed empirically: an early cut of this check with no
    nPairs floor flagged 21 files on one day, all with nPairs below 50,
    almost certainly small-sample noise rather than a real signal). This
    is a more specific signal than a low matchScore: a rotation refit can
    correct a *biased* Z residual (a mistuned calibration for this
    file/window) but cannot shrink an intrinsically *wide* one, and a wide
    one is what you'd expect from genuine correspondence ambiguity (real
    wrong matches) rather than a fixable calibration drift -- see
    zResidualSigma's and config.quality.maxZSigma's docstrings for the
    empirical basis (all three files that basis was validated against had
    nPairs > 1000). A file can have a "fine" matchScore (other terms
    compensating) while still showing this.

    Returns
    -------
    (matchscoreResult, zSigmaResult)
        matchscoreResult is (median, nPairs, gateChecked) or None --
        gateChecked is True if nPairs exceeded config.newFileInt, meaning
        the gate actually evaluated this file's matchScore (and let it
        through anyway), False if the gate never ran at all (the common
        case for the low tail). zSigmaResult is (sigma, nPairs) or None.
        Each is independently None if that specific check didn't trigger;
        both are None if the file has no pairs at all.
    """
    import xarray as xr

    with xr.open_dataset(path) as ds:
        nPairs = ds.sizes.get("pair_id", 0)
        if nPairs == 0:
            return None, None
        median = float(ds.matchScore.median())
        sigma = matching.zResidualSigma(ds, None, config)

    matchscoreResult = None
    threshold = matchscore_factor * config.quality.minMatchScore
    if median < threshold:
        matchscoreResult = (median, nPairs, nPairs > config.newFileInt)

    zSigmaResult = None
    if nPairs > config.newFileInt and sigma > config.quality.maxZSigma:
        zSigmaResult = (sigma, nPairs)

    return matchscoreResult, zSigmaResult


def _matchscore_drop(files, factor=0.5, window=6, minRef=None, minPairs=50):
    """
    Find runs of consecutive level1match files whose median matchScore is
    far below that of the files around them.

    Parameters
    ----------
    files : list of str
        real level1match files of one day (sorted here by name)
    factor : float
        a file is flagged if its median matchScore is below factor times
        the median of the medians of the up to `window` files before and
        after it (not counting flagged neighbours as reference)
    minRef : float or None
        only flag where the reference itself is at least this good (so a
        day that is bad everywhere is left to --matchscore-check)
    minPairs : int
        only flag files with at least this many pairs: the median matchScore
        of a handful of pairs takes discrete values (e.g. ~0.0101 at MOSAiC)
        and says nothing about the matching quality

    Returns
    -------
    list of dict(first, last, nFiles, median, reference, nPairsMedian)
    """
    import numpy as np
    import xarray as xr

    files = sorted(files)
    med = np.full(len(files), np.nan)
    npairs = np.zeros(len(files), dtype=int)
    for i, f in enumerate(files):
        try:
            with xr.open_dataset(f) as ds:
                n = ds.sizes.get("pair_id", 0)
                npairs[i] = n
                if n > 0:
                    med[i] = float(ds.matchScore.median())
        except Exception:
            pass
    ref = np.full(len(files), np.nan)
    for i in range(len(files)):
        nb = np.r_[med[max(0, i - window) : i], med[i + 1 : i + 1 + window]]
        nb = nb[np.isfinite(nb)]
        if len(nb) >= 3:
            ref[i] = np.median(nb)
    flagged = np.isfinite(med) & np.isfinite(ref) & (med < factor * ref)
    if minRef is not None:
        flagged &= ref >= minRef
    flagged &= npairs >= minPairs
    runs = []
    i = 0
    while i < len(files):
        if flagged[i]:
            j = i
            while j + 1 < len(files) and flagged[j + 1]:
                j += 1
            runs.append(
                dict(
                    first=files[i],
                    last=files[j],
                    nFiles=j - i + 1,
                    median=float(np.nanmedian(med[i : j + 1])),
                    reference=float(np.nanmedian(ref[i : j + 1])),
                    nPairsMedian=int(np.median(npairs[i : j + 1])),
                )
            )
            i = j + 1
        else:
            i += 1
    return runs


# bits of qualityFlags (np.packbits of recordingFailed, processingFailed,
# cameraBlocked, blowingSnow, obervationsDiffer, trackingIncomplete,
# zResidualTooWide): the ones that explain a legitimate lack of matches
_QF_EXPLAINS_LOSS = 128 | 64 | 32 | 16


def _level2_intercompare(
    matchFile, detLeaderFile, detFollowerFile, minNtot=100.0, ratioFactor=0.3
):
    """
    Compare level2match with level2detect minute by minute (see the
    level2_ratio_suspect category).

    Returns
    -------
    dict or None
        nGood: minutes with both level2detect Ntot >= minNtot and clean
        level2detect flags; nZero: of these, minutes with match Ntot == 0;
        nSuspect: minutes with match Ntot == 0 or below ratioFactor times
        the day's median match/detect ratio and no explaining flag in the
        match file; first/last: first/last suspect minute; medianRatio
    """
    import numpy as np
    import pandas as pd
    import xarray as xr

    def ntot(ds):
        n = ds.Ntot
        if n.ndim > 1:
            n = n.isel({k: 0 for k in n.dims if k != "time"})
        return n.to_series()

    with xr.open_dataset(matchFile) as m, xr.open_dataset(
        detLeaderFile
    ) as dl, xr.open_dataset(detFollowerFile) as df_:
        d = pd.DataFrame(
            {
                "match": ntot(m),
                "detL": ntot(dl).reindex(m.time.values),
                "detF": ntot(df_).reindex(m.time.values),
                "qm": m.qualityFlags.to_series(),
                "qL": dl.qualityFlags.to_series().reindex(m.time.values),
                "qF": df_.qualityFlags.to_series().reindex(m.time.values),
            }
        )
    good = (
        (d.detL >= minNtot)
        & (d.detF >= minNtot)
        & ((d.qL.fillna(255).astype(int) & _QF_EXPLAINS_LOSS) == 0)
        & ((d.qF.fillna(255).astype(int) & _QF_EXPLAINS_LOSS) == 0)
    )
    if good.sum() == 0:
        return None
    ratio = d.match.fillna(0) / np.minimum(d.detL, d.detF)
    medianRatio = float(ratio[good].median())
    low = (d.match.fillna(0) == 0) | (ratio < ratioFactor * medianRatio)
    explained = (d.qm.fillna(0).astype(int) & _QF_EXPLAINS_LOSS) != 0
    suspect = good & low & ~explained
    # longest run of consecutive suspect minutes
    longest = run = 0
    for v in suspect.values:
        run = run + 1 if v else 0
        longest = max(longest, run)
    out = dict(
        longestRun=longest,
        nGood=int(good.sum()),
        nZero=int((good & (d.match.fillna(0) == 0)).sum()),
        nSuspect=int(suspect.sum()),
        medianRatio=medianRatio,
        first=str(d.index[suspect][0])[11:16] if suspect.any() else None,
        last=str(d.index[suspect][-1])[11:16] if suspect.any() else None,
    )
    return out


def _find_reduced_coverage(rows, cases, config):
    """
    Flag daily-aggregate levels (level2match/level2track, see PARENT_LEVEL)
    that completed for a day whose per-file parent level has missing or
    broken files that same day -- see the "reduced_coverage" category in
    this module's docstring. Reuses missing/broken rows already collected
    by scan(), no extra file opens.
    """
    badPairs = {
        (r["case"], r["level"]) for r in rows if r["category"] in ("missing", "broken")
    }
    nBadByPair = Counter(
        (r["case"], r["level"]) for r in rows if r["category"] in ("missing", "broken")
    )

    extra = []
    for case in cases:
        for child, parent in PARENT_LEVEL.items():
            nBad = nBadByPair.get((case, parent), 0)
            if nBad == 0 or (case, child) in badPairs:
                continue
            ff = files.FindFiles(case, config.leader, config)
            if not ff.listFiles(child):
                continue
            extra.append(
                dict(
                    level=child,
                    camera=config.leader,
                    case=case,
                    category="reduced_coverage",
                    detail=f"{child} completed but its parent {parent} has "
                    f"{nBad} missing/broken finding(s) this day -- {child} "
                    "may silently be built from partial coverage",
                )
            )
    return extra


def _probe_netcdf(path):
    """Try to open a file with xarray; return None if fine, else a reason."""
    import xarray as xr

    try:
        with xr.open_dataset(path) as ds:
            if ds.sizes.get("time", 1) == 0 or len(ds.dims) == 0:
                return "opens but has no data"
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    return None


def scan(
    config, cases, levels, integrity="none", sample_size=30,
    include_known_bad=False, retry_broken_metarotation=True,
    matchscore_check="none", matchscore_threshold_factor=10.0,
    matchscore_drop="none", drop_day_fraction=0.2, drop_factor=0.5,
    level2_compare=False, level2_min_ntot=100.0, level2_ratio_factor=0.3,
    level2_min_minutes=5, level2_min_run=3,
):
    """
    Run the structural QC scan and return a findings DataFrame.

    One row per finding, columns: level, camera, case, category, detail.
    """
    rows = []
    cameras = [config.leader, config.follower]

    # products.DataProduct(..., None, ...) below builds a throwaway
    # taskqueue.TaskQueue whose FileQueueAPI eagerly mkdir()s real
    # directories on disk (movement/, queue/) even though this read-only
    # dag_stale check never submits/polls anything -- passing None makes
    # every call invent its own randomly-named /tmp/visss_<random> queue
    # that's never cleaned up, so scanning day-by-day/level-by-level across
    # a whole deployment leaves thousands of orphaned directories behind.
    # Use one shared queue dir for this entire scan instead, and register
    # it for cleanup on process exit.
    sharedQueueDir = tempfile.mkdtemp(prefix="visss_qc_report_")
    atexit.register(shutil.rmtree, sharedQueueDir, ignore_errors=True)

    for case in cases:
        for camera in cameras:
            ff = files.FindFiles(case, camera, config)
            for level in levels:
                if _skip_for_camera(level, camera, config):
                    continue

                isBad, badReason = tools.isBadPeriod(case, config, product=level)
                if isBad and not include_known_bad:
                    continue

                # ignoreBrokenFiles=False: a .broken.txt sentinel counts as
                # "accounted for" here (it's reported separately, below, with
                # its own traceback) -- ignoreBrokenFiles=True would count
                # every broken file as *also* missing (files.FindFiles.nMissing
                # treats "ignore broken" as "don't count them as existing"),
                # which double-reports the same file under both categories.
                nMissing = ff.nMissing(level, ignoreBrokenFiles=False)
                if nMissing > 0:
                    rows.append(
                        dict(
                            level=level,
                            camera=camera,
                            case=case,
                            category="missing",
                            detail=f"{nMissing} file(s) missing"
                            + (f" [known bad: {badReason}]" if isBad else ""),
                        )
                    )

                brokenFiles = ff.listBroken(level)
                if (
                    level == "metaRotation"
                    and retry_broken_metarotation
                    and brokenFiles
                    and camera == config.leader  # metaRotation is leader-only
                ):
                    print(f"  retrying broken metaRotation {case} with current code...")
                    if _retry_broken_metarotation(case, config):
                        brokenFiles = ff.listBroken(level)  # still broken, re-read for a fresh message
                    else:
                        print(f"    {case} now fixed, not reporting")
                        brokenFiles = []

                for brokenFile in brokenFiles:
                    command, gist = _parse_broken(brokenFile)
                    rows.append(
                        dict(
                            level=level,
                            camera=camera,
                            case=case,
                            category="broken",
                            detail=f"{brokenFile} | cmd={command} | {gist}"
                            + (f" [known bad: {badReason}]" if isBad else "")
                            + (
                                " [retried with current code, still fails]"
                                if level == "metaRotation" and retry_broken_metarotation
                                else ""
                            ),
                        )
                    )

                for nodataFile in ff.listNoData(level):
                    message = _read_text(nodataFile).strip()
                    if level == "metaRotation" and retry_broken_metarotation:
                        print(f"  verifying metaRotation nodata {case} with current code...")
                        if _verify_metarotation_nodata(nodataFile, case, config):
                            print(f"    {case} confirmed genuinely nodata")
                            continue
                        message = _read_text(f"{nodataFile}").strip() if os.path.exists(nodataFile) else "<reprocessing produced real output or went broken instead of nodata>"
                        rows.append(
                            dict(
                                level=level,
                                camera=camera,
                                case=case,
                                category="nodata_suspect",
                                detail=f"{nodataFile} | reprocessed with current code and did NOT "
                                f"come back nodata: {message}"
                                + (f" [known bad: {badReason}]" if isBad else ""),
                            )
                        )
                    elif _looks_suspect(message):
                        rows.append(
                            dict(
                                level=level,
                                camera=camera,
                                case=case,
                                category="nodata_suspect",
                                detail=f"{nodataFile} | {message}"
                                + (f" [known bad: {badReason}]" if isBad else ""),
                            )
                        )

                realFiles = ff.listFiles(level)
                if len(realFiles) > 1:
                    seen = Counter()
                    stamped = []
                    for f in realFiles:
                        try:
                            dt = files.FilenamesFromLevel(f, config).datetime64
                        except Exception:
                            continue
                        stamped.append((dt, f))
                        seen[dt] += 1
                    dupTimes = {t for t, n in seen.items() if n > 1}
                    if dupTimes:
                        dupFiles = [f for t, f in stamped if t in dupTimes]
                        rows.append(
                            dict(
                                level=level,
                                camera=camera,
                                case=case,
                                category="duplicate",
                                detail=f"{len(dupFiles)} file(s) sharing a timestamp: "
                                + ", ".join(dupFiles),
                            )
                        )

                # allComplete check: catches staleness a plain file-count
                # scan structurally cannot see (see "dag_stale" in this
                # module's docstring). Only meaningful once isComplete is
                # already True -- an incomplete day is already reported
                # above as "missing"/"broken", and allComplete would just
                # be False for that same, already-explained reason.
                cameraShort = "leader" if camera == config.leader else "follower"
                dp = products.DataProduct(level, str(case), config, sharedQueueDir, cameraShort)
                if dp.isComplete and not dp.allComplete:
                    reasons = []
                    if not dp._pastReprocessBreakpoint:
                        reasons.append(
                            "files predate a pending tools.REPROCESS_AFTER "
                            "breakpoint for this level"
                        )
                    if not dp.parentsComplete:
                        reasons.append("a parent level is itself incomplete")
                    elif not dp._upToDateWithParents:
                        reasons.append("older than a parent's newest output")
                    if not reasons:
                        reasons.append("allComplete is False")
                    rows.append(
                        dict(
                            level=level,
                            camera=camera,
                            case=case,
                            category="dag_stale",
                            detail="isComplete but not allComplete: "
                            + "; ".join(reasons)
                            + (f" [known bad: {badReason}]" if isBad else ""),
                        )
                    )

                if integrity != "none" and realFiles:
                    toCheck = realFiles
                    if integrity == "sample" and len(realFiles) > sample_size:
                        toCheck = random.sample(realFiles, sample_size)
                    for f in toCheck:
                        reason = _probe_netcdf(f)
                        if reason is not None:
                            rows.append(
                                dict(
                                    level=level,
                                    camera=camera,
                                    case=case,
                                    category="corrupt",
                                    detail=f"{f} | {reason}",
                                )
                            )

                if (
                    matchscore_check != "none"
                    and level == "level1match"
                    and realFiles
                ):
                    toCheck = realFiles
                    if matchscore_check == "sample" and len(realFiles) > sample_size:
                        toCheck = random.sample(realFiles, sample_size)
                    for f in toCheck:
                        matchscoreResult, zSigmaResult = _check_match_quality(
                            f, config, matchscore_threshold_factor
                        )
                        if matchscoreResult is not None:
                            median, nPairs, gateChecked = matchscoreResult
                            rows.append(
                                dict(
                                    level=level,
                                    camera=camera,
                                    case=case,
                                    category="matchscore_suspect",
                                    detail=f"{f} | median matchScore={median:.2e} "
                                    f"nPairs={nPairs} | "
                                    + (
                                        "quality gate DID evaluate this file and let it "
                                        "through anyway"
                                        if gateChecked
                                        else f"nPairs <= config.newFileInt="
                                        f"{config.newFileInt}, quality gate never ran"
                                    ),
                                )
                            )
                        if zSigmaResult is not None:
                            sigma, nPairs = zSigmaResult
                            rows.append(
                                dict(
                                    level=level,
                                    camera=camera,
                                    case=case,
                                    category="z_sigma_suspect",
                                    detail=f"{f} | Z-residual sigma={sigma:.2f} "
                                    f"(config.quality.maxZSigma="
                                    f"{config.quality.maxZSigma}) nPairs={nPairs} | "
                                    "matched pairs disagree with each other more than "
                                    "a rotation refit could fix -- likely genuine "
                                    "correspondence ambiguity, not a calibration issue",
                                )
                            )

    # per-day checks that need the files of a whole day / several levels
    for case in cases:
        if not include_known_bad:
            isBad, _ = tools.isBadPeriod(case, config, product="level1match")
            if isBad:
                continue
        leader = files.FindFiles(case, config.leader, config)
        follower = files.FindFiles(case, config.follower, config)

        if (
            matchscore_drop != "none"
            and "level1match" in levels
            and (matchscore_drop == "all" or random.random() < drop_day_fraction)
        ):
            realFiles = leader.listFiles("level1match")
            if len(realFiles) >= 8:
                for run in _matchscore_drop(
                    realFiles,
                    factor=drop_factor,
                    minRef=config.quality.minMatchScore,
                ):
                    rows.append(
                        dict(
                            level="level1match",
                            camera=config.leader,
                            case=case,
                            category="matchscore_drop",
                            detail=f"{run['first']} .. {os.path.basename(run['last'])} | "
                            f"{run['nFiles']} file(s) median matchScore="
                            f"{run['median']:.2e} vs {run['reference']:.2e} in the "
                            f"neighbouring files, median nPairs={run['nPairsMedian']} | "
                            "check metaRotation of these files for an implausible "
                            "rotation (fixes.despikeRotationWithMatches), a follower "
                            "restart or a timing glitch",
                        )
                    )

        if level2_compare:
            detL = leader.listFiles("level2detect")
            detF = follower.listFiles("level2detect")
            for level2 in ["level2match", "level2track"]:
                if level2 not in levels:
                    continue
                matchFiles = leader.listFiles(level2)
                if not (matchFiles and detL and detF):
                    continue
                try:
                    res = _level2_intercompare(
                        matchFiles[0], detL[0], detF[0],
                        minNtot=level2_min_ntot, ratioFactor=level2_ratio_factor,
                    )
                except Exception as e:
                    rows.append(
                        dict(
                            level=level2, camera=config.leader, case=case,
                            category="corrupt", detail=f"{matchFiles[0]} | {e}",
                        )
                    )
                    continue
                if (
                    res is not None
                    and res["nSuspect"] >= level2_min_minutes
                    and res["longestRun"] >= level2_min_run
                ):
                    rows.append(
                        dict(
                            level=level2,
                            camera=config.leader,
                            case=case,
                            category="level2_ratio_suspect",
                            detail=f"{res['nSuspect']} of {res['nGood']} minute(s) with "
                            f"level2detect Ntot >= {level2_min_ntot:g} on both cameras "
                            f"have no/little {level2} Ntot ({res['nZero']} exactly zero, "
                            f"longest run {res['longestRun']} min), "
                            f"no explaining quality flag; first {res['first']} last "
                            f"{res['last']} UTC; median {level2}/detect ratio of the day "
                            f"{res['medianRatio']:.2f}",
                        )
                    )

    rows.extend(_find_reduced_coverage(rows, cases, config))

    columns = ["level", "camera", "case", "category", "detail"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("settings", help="VISSSlib settings YAML file")
    parser.add_argument(
        "--case",
        default="0",
        help="Number of days back, 'YYYYMMDD', or 'YYYYMMDD-YYYYMMDD' "
        "(default: '0', which getCaseRange resolves to the full "
        "configured start..end deployment range)",
    )
    parser.add_argument(
        "--levels",
        default=",".join(LEVELS),
        help=f"Comma-separated levels to check (default: {','.join(LEVELS)})",
    )
    parser.add_argument(
        "--integrity",
        choices=["none", "sample", "all"],
        default="none",
        help="Probe real output files by opening them with xarray: 'none' "
        "(default, fast, structural checks only), 'sample' (random "
        "subset per level/camera/day), or 'all' (slow -- every file)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=30,
        help="Files sampled per level/camera/day when --integrity sample (default: 30)",
    )
    parser.add_argument(
        "--no-retry-broken-metarotation",
        dest="retry_broken_metarotation",
        action="store_false",
        default=True,
        help="Skip live retrying/verifying of metaRotation broken and "
        "nodata sentinels before reporting them (default: retry each "
        "broken day with current code first, since self-healing fixes "
        "may already resolve it, and reprocess each nodata day to "
        "confirm it's still genuinely nodata rather than trusting the "
        "old sentinel -- only report if it still fails / doesn't come "
        "back nodata; see scripts/README_qc.md). metaRotation is a "
        "daily product, so this stays fast even by default; the same "
        "isn't done for per-file levels like level1detect/level1match, "
        "where retrying every file inline wouldn't be.",
    )
    parser.add_argument(
        "--matchscore-check",
        choices=["none", "sample", "all"],
        default="none",
        help="Open real level1match files and check two independent "
        "signals even when the file never raised: matchScore suspiciously "
        "low (matchscore_suspect) and Z-consistency residual suspiciously "
        "wide (z_sigma_suspect, a more specific sign of genuine wrong "
        "matches -- see that category's docstring). 'none' (default, "
        "fast), 'sample' (random subset per level/camera/day, size "
        "--sample-size), or 'all' (slow -- opens every level1match file).",
    )
    parser.add_argument(
        "--matchscore-threshold-factor",
        type=float,
        default=10.0,
        help="Flag a level1match file's median matchScore as suspect when "
        "it falls below this factor times config.quality.minMatchScore "
        "(default: 10.0)",
    )
    parser.add_argument(
        "--matchscore-drop",
        choices=["none", "sample", "all"],
        default="none",
        help="Find runs of consecutive level1match files whose median "
        "matchScore is far below their neighbours' (matchscore_drop; see "
        "its description). Reads matchScore of every file of a day: "
        "'sample' checks a random fraction of the days (--drop-day-"
        "fraction), 'all' every day.",
    )
    parser.add_argument("--drop-day-fraction", type=float, default=0.2,
                        help="fraction of days checked by --matchscore-drop sample (default 0.2)")
    parser.add_argument("--drop-factor", type=float, default=0.5,
                        help="flag a file below this factor times its neighbours' median matchScore (default 0.5)")
    parser.add_argument(
        "--level2-compare",
        action="store_true",
        help="Compare level2match/level2track with level2detect minute by "
        "minute (level2_ratio_suspect): minutes with particles on both "
        "cameras in level2detect but (almost) none in the matched product "
        "and no explaining quality flag. level2detect must exist.",
    )
    parser.add_argument("--level2-min-ntot", type=float, default=100.0,
                        help="minimum level2detect Ntot (1/m3) on both cameras for a minute to count (default 100)")
    parser.add_argument("--level2-min-minutes", type=int, default=5,
                        help="report a day only with at least this many suspect minutes (default 5) ...")
    parser.add_argument("--level2-min-run", type=int, default=3,
                        help="... including a run of at least this many consecutive minutes (default 3); isolated minutes are scatter")
    parser.add_argument("--level2-ratio-factor", type=float, default=0.3,
                        help="flag minutes whose match/detect ratio is below this factor times the day's median (default 0.1)")
    parser.add_argument(
        "--include-known-bad",
        action="store_true",
        help="Also report findings inside periods listed in config.badData "
        "(default: excluded, since those are already known/accepted)",
    )
    parser.add_argument(
        "--out",
        default="qc_report.csv",
        help="Path to write the full findings CSV (default: qc_report.csv)",
    )
    args = parser.parse_args()

    config = tools.readSettings(args.settings)
    cases = tools.getCaseRange(args.case, config)
    levels = [l.strip() for l in args.levels.split(",") if l.strip()]

    print(f"QC scan: {config.site} | {len(cases)} day(s) "
          f"[{cases[0]} .. {cases[-1]}] | levels={levels} | integrity={args.integrity}")

    df = scan(
        config,
        cases,
        levels,
        integrity=args.integrity,
        sample_size=args.sample_size,
        include_known_bad=args.include_known_bad,
        retry_broken_metarotation=args.retry_broken_metarotation,
        matchscore_check=args.matchscore_check,
        matchscore_threshold_factor=args.matchscore_threshold_factor,
        matchscore_drop=args.matchscore_drop,
        drop_day_fraction=args.drop_day_fraction,
        drop_factor=args.drop_factor,
        level2_compare=args.level2_compare,
        level2_min_ntot=args.level2_min_ntot,
        level2_ratio_factor=args.level2_ratio_factor,
        level2_min_minutes=args.level2_min_minutes,
        level2_min_run=args.level2_min_run,
    )

    df.to_csv(args.out, index=False)
    print(f"\nWrote {len(df)} finding(s) to {args.out}")

    if len(df) == 0:
        print("No findings.")
        return 0

    print("\nSummary (count per level x category):")
    summary = df.groupby(["level", "category"]).size().unstack(fill_value=0)
    print(summary.to_string())

    print("\nSummary (count per camera x category):")
    summary = df.groupby(["camera", "category"]).size().unstack(fill_value=0)
    print(summary.to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
