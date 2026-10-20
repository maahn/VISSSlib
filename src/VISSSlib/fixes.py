# -*- coding: utf-8 -*-


import warnings

# import matplotlib.pyplot as plt
from copy import deepcopy
from itertools import groupby

import numpy as np
import xarray as xr
from loguru import logger as log

# various tools to fix bugs in the data


def fixMosaicTimeL1(dat1, config):
    """
    Attempt to fix drift of capture time with record_time.

    This function attempts to correct timing drift between capture_time and
    record_time by estimating and interpolating drift patterns over time.

    Parameters
    ----------
    dat1 : xarray.Dataset
        Input dataset containing capture_time and record_time variables
    config : object
        Configuration object containing fps parameter for frame rate

    Returns
    -------
    xarray.Dataset
        Dataset with corrected capture_time values

    Notes
    -----
    This is a poor attempt at fixing drift and is not used anymore.
    The function groups data into time chunks and estimates drift patterns
    to interpolate and correct the timing issues.
    """
    datS = dat1[["capture_time", "record_time"]]
    datS = datS.isel(capture_time=slice(None, None, config["fps"]))
    diff = datS.capture_time - datS.record_time

    # no estiamte the drift
    drifts1 = []
    # group netcdf into 1 minute chunks
    index1min = (
        diff.capture_time.resample(capture_time="1T", label="right")
        .first()
        .capture_time.values
    )
    if len(index1min) <= 2:
        index1min = (
            diff.capture_time.resample(capture_time="30s", label="right")
            .first()
            .capture_time.values
        )
        if len(index1min) <= 2:
            index1min = (
                diff.capture_time.resample(capture_time="10s", label="right")
                .first()
                .capture_time.values
            )
            if len(index1min) <= 2:
                index1min = (
                    diff.capture_time.resample(capture_time="1s", label="right")
                    .first()
                    .capture_time.values
                )

    grps = diff.groupby_bins("capture_time", bins=index1min)

    # find max. difference in each chunk
    # this is the one were we assume it is the true dirft
    # also time stamp or max.  is needed, this is why resample cannot be used directly
    for ii, grp in grps:
        drifts1.append(grp.isel(capture_time=grp.argmax()))
    drifts = xr.concat(drifts1, dim="capture_time")

    # interpolate to original resolution
    # extrapolation required for beginning or end - works usually very good!
    driftsInt = (
        drifts.astype(int)
        .interp_like(dat1.capture_time, kwargs={"fill_value": "extrapolate"})
        .astype("timedelta64[ns]")
    )

    # get best time estimate
    bestestimate = dat1.capture_time.values - driftsInt.values

    #                 plt.figure()
    #                 driftsInt.plot(marker="x")
    #                 diff.plot()

    # replace time in nc file
    dat1["capture_time_orig"] = deepcopy(dat1["capture_time"])
    dat1 = dat1.assign_coords(capture_time=bestestimate)

    # the difference between bestestimate and capture time must jump more than 1% of the measurement interval
    timeDiff = np.abs(
        (
            (dat1.capture_time - dat1.capture_time_orig).diff("capture_time")
            / dat1.capture_time_orig.diff("capture_time")
        )
    )
    assert np.all(timeDiff < 0.01), timeDiff.max()

    return dat1


def captureIdOverflows(dat, config, storeOrig=True, idOffset=0, dim="pid"):
    """
    Fix capture_id overflows for M1280 devices.

    For M1280 devices, capture_id is a 16-bit integer that overflows every few minutes.
    This function detects and fixes overflow conditions by applying appropriate offsets.

    Parameters
    ----------
    dat : xarray.Dataset
        Input dataset containing capture_id and capture_time variables
    config : object
        Configuration object containing fps parameter for frame rate
    storeOrig : bool, optional
        Whether to store original capture_id values, default is True
    idOffset : int, optional
        Constant offset to add to capture_id, default is 0
    dim : str, optional
        Dimension name for diff operations, default is "pid"

    Returns
    -------
    xarray.Dataset
        Dataset with fixed capture_id values

    Notes
    -----
    This function handles the specific case where capture_id overflows due to
    being a 16-bit integer. It detects overflow points and applies corrections
    to maintain proper sequential numbering.
    """
    log.info("fixing captureIdOverflows")
    maxInt = 65535

    # if someone already messed with the data, revert it
    if "capture_id_orig" in dat.keys():
        dat["capture_id"] = deepcopy(dat["capture_id_orig"])

    if storeOrig:
        dat["capture_id_orig"] = deepcopy(dat["capture_id"])

    # constant offset
    if idOffset != 0:
        dat["capture_id"] += idOffset

    idDiffObserved = dat.capture_id.diff(dim)
    idDiffEstimated = np.round(
        dat.capture_time.diff(dim) / np.timedelta64(round(1 / config.fps * 1e6), "us")
    ).astype(int)

    stepsObserved = (idDiffObserved < 0) | (idDiffEstimated >= maxInt)
    nStepsObserved = stepsObserved.sum()

    # estimate expected steps
    firstII = dat.capture_id.values[0]
    firstCaptureT = dat.capture_time.values[0]
    lastCaptureT = dat.capture_time.values[-1]

    deltaT = (lastCaptureT - firstCaptureT) / np.timedelta64(1, "s")
    nFrames = np.ceil(deltaT * config.fps).astype(int)
    nStepsExpected = int((firstII + nFrames) / maxInt)

    if nStepsObserved == nStepsExpected == 0:
        # nothing to do
        return dat

    if (nStepsExpected == nStepsObserved) or ((nStepsExpected - 1) == nStepsObserved):
        jumpIIs = np.where(stepsObserved)[0] + 1

        for jumpII in jumpIIs:
            dat["capture_id"][jumpII:] += maxInt

    else:
        raise RuntimeError("was einfallen lassen...")

    assert np.all(dat.capture_id.diff(dim) >= 0)
    log.info(
        f"expecting {nStepsExpected} jumps, found and fixed {(stepsObserved).sum().values} jumps"
    )

    return dat


def revertIdOverflowFix(dat):
    """
    Revert capture_id overflow fix by restoring original values.

    This function restores the original capture_id values by renaming
    the fixed and original variables back to their original names.

    Parameters
    ----------
    dat : xarray.Dataset
        Input dataset with fixed capture_id and capture_id_orig variables

    Returns
    -------
    xarray.Dataset
        Dataset with original capture_id restored

    Notes
    -----
    This function is used to undo the effects of captureIdOverflows when
    needed for data recovery or analysis consistency.
    """
    log.info("reverting revertIdOverflowFix")
    dat = dat.rename({"capture_id": "capture_id_fixed"})
    dat = dat.rename({"capture_id_orig": "capture_id"})
    return dat


def removeGhostFrames(metaDat, config, intOverflow=True, idOffset=0, fixIteration=3):
    """
    Remove ghost frames from MOSAiC follower data.

    For MOSAiC follower devices, additional ghost frames are occasionally added
    to the dataset. These can be identified by their spacing being less than
    1/fps apart. This function identifies and removes such frames.

    Parameters
    ----------
    metaDat : xarray.Dataset
        Input dataset containing capture_time and capture_id variables
    config : object
        Configuration object containing fps parameter for frame rate
    intOverflow : bool, optional
        Whether to handle integer overflows, default is True
    idOffset : int, optional
        Offset to add to capture_id, default is 0
    fixIteration : int, optional
        Number of iterations to attempt ghost frame removal, default is 3

    Returns
    -------
    tuple
        A tuple containing (fixed_dataset, dropped_frames, beyond_repair_flag)
        where:
        - fixed_dataset is the dataset with ghost frames removed
        - dropped_frames is the count of removed frames
        - beyond_repair_flag indicates if data is beyond repair

    Notes
    -----
    Ghost frames are typically identified by their spacing being significantly
    different from the expected 1/fps interval. The function performs multiple
    iterations to handle complex cases where ghost frames might be in data gaps.
    """
    log.info("fixing removeGhostFrames")

    beyondRepair = False
    metaDat["capture_id_orig"] = deepcopy(metaDat["capture_id"])

    metaDat["capture_id"] = metaDat["capture_id"] + idOffset

    if intOverflow:
        metaDat = captureIdOverflows(
            metaDat, config, dim="capture_time", storeOrig=False
        )

    # ns are assumed
    assert metaDat["capture_time"].dtype == "<M8[ns]"

    droppedFrames = 0
    for nn in range(fixIteration + 1):
        slope = (
            (
                metaDat["capture_time"].diff("capture_time")
                / metaDat["capture_id"].diff("capture_time")
            )
        ).astype(int)
        configSlope = 1e9 / config.fps
        # we find them because dat is not 1/fps apart
        jumps = ((slope / configSlope).values > 1.03) | (
            (slope / configSlope).values < 0.97
        )
        jumpsII = np.where(jumps)[0]
        nGroups = sum(k for k, v in groupby(jumps))

        # the last loop is only for testng
        if nn == fixIteration:
            if nGroups != 0:
                log.error("FILE BROKEN BEYOND REPAIR")
                droppedFrames += len(metaDat.capture_time) - jumpsII[0]
                # remove fishy data and everything after
                metaDat = metaDat.isel(capture_time=slice(0, jumpsII[0]))
                beyondRepair = True
            break

        lastII = np.concatenate((jumpsII[:-1][np.diff(jumpsII) != 1], jumpsII[-1:])) + 1
        assert nGroups == len(lastII)

        for lastI in lastII:
            metaDat["capture_id"][lastI:] = metaDat["capture_id"][lastI:] - 1

        # remove all fishy frames
        metaDat = metaDat.drop_isel(capture_time=jumpsII)
        droppedFrames += len(jumpsII)

        if nGroups > 0:
            log.warn(
                f"ghost iteration {nn}: found {nGroups} ghost frames at {lastII.tolist()}"
            )
        else:
            break

    return metaDat, droppedFrames, beyondRepair


def delayedClockReset(metaDat, config):
    """
    Check for and fix delayed clock reset issues.

    This function detects delayed clock resets in the data and attempts to
    correct them by adjusting timestamps accordingly.

    Parameters
    ----------
    metaDat : xarray.Dataset
        Input dataset containing capture_time and capture_id variables
    config : object
        Configuration object containing fps parameter for frame rate

    Returns
    -------
    xarray.Dataset
        Dataset with corrected timestamps if reset was detected

    Notes
    -----
    Delayed clock resets are identified by large negative time differences
    (>10 seconds). The function handles both cases where integer overflows
    and timestamp issues coexist, and attempts to fix the timing problems
    by recalculating timestamps based on known good values.
    """
    if (metaDat.capture_time.diff() <= -10e6).any():
        log.info("fixing detected delayedClockReset")

        resetII = np.where((metaDat.capture_time.diff() < -10e6))[0]
        assert len(resetII) == 1, "len(resetII) %i" % len(resetII)
        resetII = resetII[0]  # +1 already applied by pandas!
        assert resetII < 20, (
            "time jump usually occures within first few frames %i" % resetII
        )

        if (metaDat.capture_id.diff()[1 : resetII + 1] < 0).any():
            # we cannot handle int overflows in capture id AND wrong timestamps,
            # cut data
            metaDat = metaDat.iloc[resetII:]
        else:
            # attempt to fix it!
            firstGoodTime = metaDat.capture_time.iat[resetII]
            firstGoodID = metaDat.capture_id.iat[resetII]
            deltaT = round(1 / config.fps * 1e6)
            offsets = (metaDat.capture_id.iloc[:resetII] - firstGoodID) * deltaT
            metaDat.iloc[:resetII, metaDat.columns.get_loc("capture_time")] = (
                firstGoodTime + offsets
            )

    return metaDat


def makeCaptureTimeEven(datF, config, dim="capture_time"):
    """
    Make capture time even for M1280 follower devices.

    For M1280 follower devices, significant drift can occur causing clocks to
    drift more than 1 frame apart within 10 minutes. This function creates
    a new time vector with even spacing based on a trusted capture_id.

    Parameters
    ----------
    datF : xarray.Dataset
        Input dataset containing capture_time and capture_id variables
    config : object
        Configuration object containing fps parameter for frame rate
    dim : str, optional
        Dimension name for operations, default is "capture_time"

    Returns
    -------
    xarray.Dataset
        Dataset with new evenly spaced capture_time_even variable

    Notes
    -----
    This function is specifically designed for capture_id offset estimation
    and creates a new time vector that maintains even spacing regardless
    of timing drift issues. It validates that the calculated slopes are
    within acceptable ranges.
    """
    log.info("making follower times even")

    if len(datF[dim]) <= 1:
        print("makeCaptureTimeEven: too short, nothing to do")
        return datF

    if dim in ["fpid", "pid"]:
        unqiue, uniqueII = np.unique(datF.capture_time, return_index=True)
        datF4slope = datF.isel(**{dim: uniqueII})
    else:
        datF4slope = datF

    assert len(datF4slope.capture_id) > 1, "need at least two samples to do derivative"

    assert np.all(
        datF4slope.capture_id.diff(dim) >= 0
    ), "capture_id must increase monotonically "
    assert np.all(
        datF4slope.capture_time.diff(dim).astype(int) > 0
    ), "capture_time must increase monotonically "

    slopeF = datF4slope["capture_time"].diff(dim).astype(int) // datF4slope[
        "capture_id"
    ].diff(dim).astype(int)

    configSlope = int(round(1e9 / config.fps, -3))
    deltaSlope = 1000  # =1us

    # make sure we do not have ghost frames in the data
    if dim == "pid":
        # we can have slope 0 in level1detect
        slopeF = slopeF.isel(pid=(datF["capture_id"].diff(dim) != 0))

    assert slopeF.min() >= (
        configSlope - deltaSlope
    ), f"min slope {slopeF.min()} too small {(configSlope+deltaSlope)}"
    assert slopeF.max() <= (
        configSlope + deltaSlope
    ), f"max slope {slopeF.max()} too large {(configSlope+deltaSlope)}"

    offset = datF.capture_time.values[0]
    fixedTime = ((datF.capture_id - datF.capture_id[0]) * configSlope) + offset

    # datF["capture_time_orig"] = deepcopy(datF["capture_time"])
    datF["capture_time_even"] = fixedTime

    return datF


def makeCaptureTimeEvenBothCameras(leaderDat, followerDat, config):
    """
    Reconstruct an evenly-spaced ``capture_time_even`` from ``capture_id``
    for *both* leader and follower, each anchored at its own first sample.

    Unlike :func:`makeCaptureTimeEven`, this does not validate the
    reconstructed slope against ``config.fps`` (no assertions) -- some
    M2050 deployments' actual frame interval differs from the nominal
    ``1/config.fps`` by a few microseconds, more than
    :func:`makeCaptureTimeEven`'s tolerance allows, which would otherwise
    reject perfectly usable segments. It also does not deduplicate by
    ``capture_time`` first, since callers here already pass 1D
    (``fpid``-indexed) arrays without the multiple-particles-per-frame
    complication :func:`makeCaptureTimeEven` guards against for
    ``pid``-indexed level1detect data.

    Grounded in the hardware sync: the follower is pulse-triggered by the
    leader at capture time, so both cameras' true capture instants are
    tied to a shared, evenly-spaced ``capture_id`` sequence -- only each
    camera's *own* onboard clock (``capture_time``) can drift from that
    sequence independently. Reconstructing time from ``capture_id``
    removes that per-camera clock drift from the offset estimate.

    Parameters
    ----------
    leaderDat, followerDat : xarray.Dataset
        Datasets with ``capture_time`` and ``capture_id`` variables along
        their (matching) leading dimension.
    config : object
        Configuration object containing the ``fps`` parameter.

    Returns
    -------
    tuple(xarray.Dataset, xarray.Dataset)
        (leaderDat, followerDat), each with a new ``capture_time_even``
        variable.
    """
    configSlope = int(round(1e9 / config.fps, -3))

    def _evenTime(dat):
        offset = dat.capture_time.values[0]
        return ((dat.capture_id - dat.capture_id[0]) * configSlope) + offset

    leaderDat = leaderDat.copy()
    leaderDat["capture_time_even"] = _evenTime(leaderDat)
    followerDat = followerDat.copy()
    followerDat["capture_time_even"] = _evenTime(followerDat)

    return leaderDat, followerDat


# def revertMakeCaptureTimeEven(dat):
#     dat = dat.rename({"capture_time": "capture_time_even"})
#     dat = dat.rename({"capture_time_orig": "capture_time"})
#     return dat


def detectCaptureIdDropTimes(
    leaderDat,
    followerDat,
    dim="fpid",
    nPoints=500,
    maxDiffMs=1,
    timeDim="capture_time",
    minRunLength=5,
    maxJump=2,
):
    """
    Find times within a window where the leader-follower capture_id
    offset makes a clean, sustained jump of a few frames -- the
    signature of one camera silently dropping (or duplicating) a
    frame index, as opposed to genuine timing ambiguity between the
    two cameras.

    Pre-PTP hardware occasionally lost a captured frame's index
    without otherwise disturbing the capture_id sequence. That leaves
    the true offset well-defined and constant on either side of the
    drop, but `tools.estimateCaptureIdDiffCore` averages the whole
    window into one offset and fails its >70% consistency check
    whenever a drop happens to fall inside it. The offset itself is
    not ambiguous -- only the single-offset-per-window assumption is
    wrong. Returns candidate leader `capture_time` values to split the
    window at (feed into `timeBlocks` alongside genuine follower
    restarts) so each side can be resolved independently; not meant
    as a replacement for `estimateCaptureIdDiffCore` itself.

    Parameters
    ----------
    leaderDat, followerDat : xarray.Dataset
        Same inputs as `tools.estimateCaptureIdDiffCore`.
    dim : str, optional
        Dimension to sample leader points from, by default "fpid".
    nPoints : int, optional
        Number of leader points to sample, by default 500.
    maxDiffMs : float, optional
        Matching window in ms, by default 1.
    timeDim : str, optional
        Time coordinate to use, by default "capture_time".
    minRunLength : int, optional
        Minimum number of samples for a run to be trusted as a real
        offset regime rather than noise, by default 5.
    maxJump : int, optional
        Only trust a jump between neighbouring runs up to this many
        frames (a dropped/duplicated frame is normally exactly 1), by
        default 2.

    Returns
    -------
    list of numpy.datetime64
        Leader capture_times to split the window at, in time order.
    """
    if (len(leaderDat[dim]) == 0) or (len(followerDat[dim]) == 0):
        return []

    # Both leader and follower have their own onboard clock that can drift
    # independently (see makeCaptureTimeEvenBothCameras's docstring) --
    # over a ~10-minute window that drift can accumulate past half a frame
    # period, at which point this function's own nearest-time matching
    # silently latches onto the *next* frame instead of the true
    # corresponding one, producing a spurious, sustained idDiff step that
    # looks exactly like a genuine dropped frame (confirmed on real data:
    # hyytiala2_v3 20240216-125000 falsely "detected" a drop at 12:57:47
    # this way -- V1.0 output, from before this function existed, used a
    # single constant offset for the whole file with no matchScore
    # degradation anywhere, proving no real drop occurred). Prefer
    # capture_time_even (drift-removed) for *both* sides symmetrically if
    # the caller has already reconstructed it -- previously only the
    # follower side was checked, which happened to be enough to hide this
    # exact false positive in testing but left the leader side vulnerable
    # to the same failure mode.
    timeDimLeader = timeDim
    if (timeDim == "capture_time") and ("capture_time_even" in leaderDat.data_vars):
        timeDimLeader = "capture_time_even"
    timeDimFollower = timeDim
    if (timeDim == "capture_time") and ("capture_time_even" in followerDat.data_vars):
        timeDimFollower = "capture_time_even"

    if len(leaderDat[dim]) > nPoints:
        points = np.linspace(0, len(leaderDat[dim]), nPoints, dtype=int, endpoint=False)
    else:
        points = range(len(leaderDat[dim]))

    times = []
    idDiffs = []
    for point in points:
        absDiff = np.abs(
            leaderDat[timeDimLeader].isel(**{dim: point}).values
            - followerDat[timeDimFollower]
        )
        pMin = np.min(absDiff).values
        if pMin < np.timedelta64(int(maxDiffMs), "ms"):
            pII = absDiff.argmin().values
            idDiffs.append(
                followerDat.capture_id.values[pII]
                - leaderDat.capture_id.isel(**{dim: point}).values
            )
            times.append(leaderDat[timeDimLeader].isel(**{dim: point}).values)

    if len(idDiffs) < 2 * minRunLength:
        return []

    times = np.array(times)
    idDiffs = np.array(idDiffs)
    order = np.argsort(times)
    times, idDiffs = times[order], idDiffs[order]

    changeAt = np.where(np.diff(idDiffs) != 0)[0] + 1
    runBounds = np.concatenate(([0], changeAt, [len(idDiffs)]))
    runs = [
        (idDiffs[a], a, b)
        for a, b in zip(runBounds[:-1], runBounds[1:])
        if (b - a) >= minRunLength
    ]
    if len(runs) < 2:
        return []

    breakTimes = []
    for (valA, _, _), (valB, sB, _) in zip(runs[:-1], runs[1:]):
        if 0 < abs(int(valB) - int(valA)) <= maxJump:
            breakTimes.append(times[sB])
    return breakTimes


def detectPhaseJumpTimes(
    leaderDat,
    followerDat,
    config,
    dim="fpid",
    binSeconds=10,
    minJumpFrac=0.5,
    recoverBins=3,
    maxOnsetsPerCamera=5,
):
    """
    Find times where one camera's raw ``capture_time`` makes a brief,
    self-correcting jump away from its own steady, ``capture_id``-implied
    schedule -- roughly one frame period, lasting minutes, then decaying
    back -- as opposed to a genuine ``capture_id``-level drop (handled by
    `detectCaptureIdDropTimes`) or ordinary independent per-camera clock
    drift (removed by `makeCaptureTimeEvenBothCameras`).

    Confirmed on real data (hyytiala2_v3 20240213-215000): the follower's
    raw capture_time jumped ~4.25ms (~1 frame period at this deployment's
    fps) relative to its own capture_id-implied schedule at 21:51:07,
    then decayed back over the following ~8.5 minutes -- capture_id
    numbering itself never skipped or repeated a value there (confirmed
    by inspection), so `detectCaptureIdDropTimes` (which works in
    capture_id space) cannot see it, and resolving the matching offset
    from `capture_time_even` (reconstructed purely from capture_id) is
    blind to it by construction, since that reconstruction cannot
    represent a discrepancy between capture_id and the camera's own
    recorded timestamp. The single segment-wide offset
    `_resolveMatchingOffset` picked for the rest of that file (correct
    for the bulk of it) was wrongly applied to the ~70s window right
    after the jump too, degrading Z-consistency there even though the
    file's aggregate quality looked fine.

    This is deliberately a different signal from `detectCaptureIdDropTimes`:
    ordinary drift accumulates *smoothly* -- exactly what
    `capture_time_even` already removes correctly -- and never produces a
    single large bin-to-bin jump the way this transient does. Watching
    the *rate of change* of (raw capture_time minus its own
    capture_id-reconstructed value), rather than its absolute size, is
    what tells the two apart without reintroducing the false-positive
    `detectCaptureIdDropTimes` had before it started preferring
    `capture_time_even` (see that function's docstring).

    Parameters
    ----------
    leaderDat, followerDat : xarray.Dataset
        RAW (not drift-corrected) leader/follower data with
        ``capture_time`` and ``capture_id``, as read from level1detect.
    config : dict
        Configuration settings (for ``config.fps``).
    dim : str, optional
        Dimension to operate along, by default "fpid".
    binSeconds : float, optional
        Bin width for the raw-vs-reconstructed deviation series, by
        default 10.
    minJumpFrac : float, optional
        Minimum bin-to-bin jump, as a fraction of one nominal frame
        period (``1000/config.fps`` ms), to flag as a glitch, by
        default 0.5.
    recoverBins : int, optional
        Number of consecutive bins the deviation must stay back within
        tolerance of its pre-jump level to mark the glitch as over, by
        default 3.
    maxOnsetsPerCamera : int, optional
        A genuine transient should be rare (at most a couple of times
        in a 10-minute file) -- confirmed on real data
        (hyytiala2_v3 20240112-030001, an extreme-density file with
        2.6M leader particles): the raw-vs-reconstructed deviation
        itself becomes noisy enough at that density to cross the
        threshold in nearly every bin, which produced 179 "onsets"
        spaced exactly one bin apart rather than a real repeating
        hardware event, and would have fragmented the file into ~180
        useless few-second segments. If a camera's raw onset count
        exceeds this, the signal for that camera is untrustworthy at
        this operating point -- discard all its onsets rather than
        acting on noise, by default 5.

    Returns
    -------
    list of numpy.datetime64
        capture_time values (paired onset/recovery per glitch found) to
        add as extra segment-split points, in time order. Each value is
        in whichever camera's own capture_time it was detected from --
        consistent with how `timeBlocks` already mixes leader- and
        follower-timeline boundaries as approximate wall-clock cut
        points.
    """
    import pandas as pd

    framePeriodMs = 1000 / config.fps
    threshold = minJumpFrac * framePeriodMs
    configSlope = int(round(1e9 / config.fps, -3))

    breakTimes = []
    for dat in (leaderDat, followerDat):
        if len(dat[dim]) < 2 * recoverBins:
            continue
        ct = dat.capture_time.values
        cid = dat.capture_id.values
        even = (
            (cid.astype("int64") - cid[0]) * configSlope + ct[0].astype("int64")
        ).astype("datetime64[ns]")
        diffMs = (ct - even) / np.timedelta64(1, "ms")

        s = (
            pd.Series(diffMs, index=pd.DatetimeIndex(ct))
            .resample(f"{int(binSeconds)}s")
            .median()
            .dropna()
        )
        if len(s) < 2 * recoverBins:
            continue

        vals = s.values
        times = s.index.values
        jumps = np.abs(np.diff(vals))
        onsets = np.where(jumps > threshold)[0] + 1

        if len(onsets) > maxOnsetsPerCamera:
            log.warning(
                f"detectPhaseJumpTimes found {len(onsets)} onsets in one "
                f"camera, exceeding maxOnsetsPerCamera={maxOnsetsPerCamera} "
                "-- treating the raw-vs-reconstructed deviation as too "
                "noisy to trust (e.g. extreme particle density) rather "
                "than a real repeating transient, discarding all of this "
                "camera's onsets"
            )
            continue

        for onsetIdx in onsets:
            baseline = vals[onsetIdx - 1]
            recovered = None
            for j in range(onsetIdx, len(vals) - recoverBins + 1):
                window = vals[j : j + recoverBins]
                if np.all(np.abs(window - baseline) <= threshold):
                    recovered = j
                    break
            breakTimes.append(times[onsetIdx])
            if recovered is not None:
                breakTimes.append(times[recovered])

    if len(breakTimes) == 0:
        return []
    return sorted(np.unique(np.array(breakTimes, dtype="datetime64[ns]")).tolist())


def removeFlippedCaptureTimeFrames(metaDat1, fname):
    """
    Drop frames around isolated backwards jumps in one source's capture_time.

    A camera's own onboard clock occasionally stamps two consecutive frames
    with flipped capture_time (frame k+1 gets a capture_time a few
    microseconds *earlier* than frame k, even though k was physically
    written first) -- see the "flipped capture_time" note in metadata.py's
    module docstring. Left alone, this is not just a cosmetic timestamp
    error: getMetaData() later concatenates and sorts all camera-thread
    data by capture_time to interleave threads chronologically, and sorting
    by a locally-flipped value reorders that one thread's own record_id
    sequence out of order at exactly that point. detection.py's frame
    reader walks each thread strictly by increasing record_id and has no
    way to seek backwards, so this then surfaces downstream as a hard
    "Cannot go back!" crash for the whole 10 minute file.

    This must be called on a single source's data (e.g. one camera
    thread's ascii file) while it is still in its own original, unsorted
    recording order -- calling it after data from multiple sources has
    already been concatenated and sorted by capture_time is a no-op, since
    sorted data cannot contain a backwards step by construction (that is
    exactly what makes the flip invisible again once threads are merged).

    Only rows whose own capture_time is already self-contradictory (stamped
    earlier than a frame recorded before it, in the same camera's write
    order) are dropped. No surviving frame's capture_time is ever changed,
    shifted, or interpolated -- this only removes evidence that was already
    unusable, it never invents a value that downstream stereo matching
    could be misled by.

    Parameters
    ----------
    metaDat1 : xarray.Dataset
        Single-source metadata in its original (not capture_time-sorted)
        recording order, as returned by _getMetaData1().
    fname : str
        Source filename, used only for the diagnostic print.

    Returns
    -------
    tuple
        (metaDat1, nDropped) with the frames around each backwards jump
        removed and the count of dropped frames.
    """
    jumps = np.diff(metaDat1.capture_time.astype(int)) < 0
    nJumps = np.sum(jumps)
    droppedIndices = []
    if nJumps > 0:
        ss = np.where(jumps)[0]
        assert nJumps < 20, "more than 20 is very fishy..."
        # Consecutive jump indices belong to one glitch and are repaired
        # together; separate (non-adjacent) glitches elsewhere in the same
        # file are independent and each get their own neighbours dropped.
        # This matters in practice: a handful of isolated single-sample
        # capture_time flips scattered through one 10 minute file is the
        # commonly observed pattern, not one contiguous bad patch.
        groups = np.split(ss, np.where(np.diff(ss) != 1)[0] + 1)
        for group in groups:
            log.warning(
                "%s: capture_time flip, DROPPING FRAMES around %i-%i"
                % (fname, group[0], group[-1])
            )
            droppedIndices.append(group[0] - 1)
            droppedIndices.extend(group.tolist())
            droppedIndices.append(group[-1] + 1)
        droppedIndices = np.unique(droppedIndices)
        metaDat1 = metaDat1.drop_isel(capture_time=droppedIndices)

    return metaDat1, len(droppedIndices)


# a movie file occasionally ends a handful of frames before the ascii log
# does (e.g. the video encoder's last buffered frames never got flushed
# before the file was closed/rotated). Once the video has genuinely run
# out of frames, treat up to this many orphaned trailing ascii rows as an
# acceptable, unrecoverable tail loss rather than aborting the whole file.
# Observed shortfalls in production (hyytiala2_v3, nyaalesund) are 1-6
# frames; this cap is kept well below that to avoid ever masking a real
# mid-file corruption instead.
_MAX_TRAILING_FRAMES_TO_DROP = 25


def isDroppableTrailingFrameShortfall(
    rowsRemainingForThread, maxFrames=_MAX_TRAILING_FRAMES_TO_DROP
):
    """
    Is an end-of-video frame shortfall small enough to treat as benign?

    detection.py calls this once a camera thread's video has genuinely run
    out of decodable frames while the ascii log still has more rows for
    that thread. A shortfall of a handful of frames right at the tail of a
    10 minute recording is expected/benign (e.g. the video encoder's last
    buffered frames never got flushed before the file was closed/
    rotated); a shortfall spanning a large fraction of the file instead
    indicates real, unrelated corruption that should still fail loudly
    rather than be silently swallowed.

    Parameters
    ----------
    rowsRemainingForThread : int
        Number of ascii rows for this thread, from the current position to
        the end of the file, that have no corresponding video frame.
    maxFrames : int, optional
        Upper bound on what counts as a benign trailing shortfall.

    Returns
    -------
    bool
    """
    return rowsRemainingForThread <= maxFrames


# ---------------------------------------------------------------------------
# MOSAiC (first VISSS, M1280 cameras, two computers, no PTP): leader/follower
# frame mapping via capture_id
#
# What the data show (Oct/Nov 2019, see project notes):
#
# * Both cameras are effectively hardware triggered: within one follower run
#   the follower-leader capture_id lag is constant for many hours while the
#   two camera clocks (capture_time) drift 25-30 ppm apart. Pairing frames by
#   capture_id is therefore exact, capture_time is not needed at all.
# * capture_id counts 1..65535 and wraps 65535 -> 1 (period 65535).
# * Every frame is recorded (also frames without moving particles), so
#   capture_id is gap free, even when the follower drops ~274 frame blocks.
# * Ghost frames (follower only): 6 consecutive frame intervals of only
#   ~0.81-0.93 frame periods (summing up to 5 periods) while capture_id
#   advances by 6, i.e. one extra capture_id is inserted. From then on, the
#   lag is permanently off by one. 0-31 events per day.
# * The follower is restarted every few hours, resetting capture_id.
# * record_time (computer clock, set in the processing queue) gives the lag
#   only to +-1 frame for a 5 min file, the median over a whole restart
#   segment is usually exact. The exact frame is determined from the
#   vertical position of single particles seen by both cameras, which is
#   only consistent for the correct lag (a one frame error corresponds to
#   ~120 px for a 1 m/s particle).
# ---------------------------------------------------------------------------

MOSAIC_CAPTURE_ID_PERIOD = 65535


def _mosaicLoadMetaFrames(case, camera, config):
    """Load capture_time/record_time (us), raw capture_id and moving pixel
    counts of all metaFrames files of a day, in recording order."""
    from . import files

    fnames = files.FindFiles(case, camera, config).listFiles("metaFrames")
    parts = []
    for fname in fnames:
        with xr.open_dataset(fname) as ds:
            if len(ds.capture_time) == 0:
                continue
            parts.append(
                {
                    "ct": ds.capture_time.values.astype("datetime64[us]").astype(
                        np.int64
                    ),
                    "rt": ds.record_time.values.astype("datetime64[us]").astype(
                        np.int64
                    ),
                    "cid": ds.capture_id.values.astype(np.int64),
                    "nmp": np.nan_to_num(
                        ds.nMovingPixel.isel(nMovingPixelThresh=0).values
                    ),
                }
            )
    if len(parts) == 0:
        return None
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def _mosaicFramePeriod(dat, config):
    """Frame period in us as measured by the camera's own clock."""
    dct = np.diff(dat["ct"])
    dc = np.diff(dat["cid"]) % MOSAIC_CAPTURE_ID_PERIOD
    good = (dc == 1) & (dct > 0)
    if good.sum() < 100:
        return 1e6 / config.fps
    return float(np.median(dct[good]))


def _mosaicSegmentsAndGhosts(dat, period, correctGhosts=True):
    """
    Split frames into camera runs (restart = capture_id step not consistent
    with capture_time step), unwrap capture_id per run, and detect/correct
    ghost frames.

    Returns
    -------
    seg : array of int
        run index per frame
    u : array of int
        unwrapped, ghost corrected capture_id per frame (congruent to the
        raw capture_id minus the number of preceding ghost frames of the run)
    drop : array of bool
        frames inside a ghost frame sequence (timing ambiguous)
    ghosts : list of tuples
        (index of first frame, index of first corrected frame, extra ids)
    """
    P = MOSAIC_CAPTURE_ID_PERIOD
    cid = dat["cid"]
    dc = np.diff(cid) % P
    dt = np.diff(dat["ct"]) / period
    reset = (dt <= 0) | (np.abs(dc - (np.round(dt) % P)) > 3)
    seg = np.concatenate(([0], np.cumsum(reset)))

    starts = np.concatenate(([0], np.flatnonzero(reset) + 1))
    ends = np.concatenate((starts[1:], [len(cid)]))
    u = np.empty(len(cid), dtype=np.int64)
    for a, b in zip(starts, ends):
        u[a:b] = cid[a] + np.concatenate(([0], np.cumsum(dc[a : b - 1])))

    # ghost frames: compressed frame intervals with regular capture_id steps
    short = (dt > 0.5) & (dt < 0.97) & (dc == 1)
    idx = np.flatnonzero(short)
    drop = np.zeros(len(cid), dtype=bool)
    ghosts = []
    if len(idx) > 0:
        for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            extra = int(round(len(run) - dt[run].sum()))
            if extra < 1:
                continue
            i0, i1 = run[0], run[-1] + 1
            if (seg[i0] != seg[i1]) or not correctGhosts:
                continue
            segEnd = ends[seg[i1]]
            u[i1:segEnd] -= extra
            drop[i0 + 1 : i1] = True
            ghosts.append((i0, i1, extra))

    # ghost frames hidden in a block of dropped frames: capture_id advances
    # by 1-3 more than the elapsed frame periods (e.g. 275 ids for 274.007
    # periods, 20200106 07:54). Observed rarely, but each one shifts the lag
    # by one for the rest of the run.
    if correctGhosts:
        excess = dc - np.round(dt)
        hidden = np.flatnonzero((dt > 1.5) & (excess >= 1) & (excess <= 3))
        for i0 in hidden:
            i1 = i0 + 1
            if seg[i0] != seg[i1]:
                continue
            extra = int(excess[i0])
            u[i1 : ends[seg[i1]]] -= extra
            ghosts.append((i0, i1, extra))
        ghosts.sort()
    return seg, u, drop, ghosts


def _mosaicActivityLag(lu, lact, fu, fact, guess, searchRange=60):
    """
    Lag (follower u - leader u) maximizing the cross correlation of the
    per-frame "anything moving" signal of both cameras. Independent of any
    clock, but the peak is a few frames broad, so only good to +-1-2 frames.

    Returns (lag, z-score of peak) or (None, nan)
    """
    from scipy.signal import fftconvolve

    if (len(lu) < 1000) or (len(fu) < 1000):
        return None, np.nan
    if (lu.max() - lu.min() > 2e7) or (fu.max() - fu.min() > 2e7):
        return None, np.nan

    def series(u, act):
        s = np.full(u.max() - u.min() + 1, np.nan)
        s[u - u.min()] = act > 0
        valid = np.isfinite(s)
        s = np.where(valid, s - np.nanmean(s), 0.0)
        return s, valid.astype(float)

    a, va = series(lu, lact)
    b, vb = series(fu, fact)
    if (np.sum(a * a) == 0) or (np.sum(b * b) == 0):
        return None, np.nan
    cc = fftconvolve(b, a[::-1], "full")
    nn = fftconvolve(vb, va[::-1], "full")
    lags = np.arange(-(len(a) - 1), len(b)) + fu.min() - lu.min()
    sel = (np.abs(lags - guess) <= searchRange) & (nn > 1000)
    if sel.sum() < 10:
        return None, np.nan
    cc = cc[sel] / nn[sel]
    lags = lags[sel]
    k = np.argmax(cc)
    bg = np.median(cc)
    sd = 1.4826 * np.median(np.abs(cc - bg))
    if sd == 0:
        return None, np.nan
    return int(lags[k]), float((cc[k] - bg) / sd)


def _mosaicLoadParticles(case, camera, config):
    """capture_time (us), Dmax and vertical center position of all
    level1detect particles of a day"""
    from . import files

    fnames = files.FindFiles(case, camera, config).listFiles("level1detect")
    parts = []
    for fname in fnames:
        with xr.open_dataset(fname) as ds:
            if len(ds.pid) == 0:
                continue
            z = ds.position_upperLeft.sel(dim2D="y") + ds.Droi.sel(dim2D="y") / 2.0
            xc = ds.position_upperLeft.sel(dim2D="x") + ds.Droi.sel(dim2D="x") / 2.0
            parts.append(
                {
                    "ct": ds.capture_time.values.astype("datetime64[us]").astype(
                        np.int64
                    ),
                    "D": ds.Dmax.values.astype(float),
                    "z": z.values.astype(float),
                    "xc": xc.values.astype(float),
                }
            )
    if len(parts) == 0:
        return None
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def _mosaicParticleFrames(part, dat, seg, u, drop, period):
    """assign each particle the (seg, u) of its frame; -1 if unknown/dropped"""
    pSeg = np.full(len(part["ct"]), -1, dtype=np.int64)
    pU = np.full(len(part["ct"]), -1, dtype=np.int64)
    order = np.argsort(dat["ct"], kind="stable")
    ctSorted = dat["ct"][order]
    ii = np.searchsorted(ctSorted, part["ct"])
    ii = np.clip(ii, 1, len(ctSorted) - 1)
    left = np.abs(part["ct"] - ctSorted[ii - 1]) < np.abs(part["ct"] - ctSorted[ii])
    ii = np.where(left, ii - 1, ii)
    ok = np.abs(ctSorted[ii] - part["ct"]) < period / 4
    frame = order[ii]
    ok &= ~drop[frame]
    pSeg[ok] = seg[frame[ok]]
    pU[ok] = u[frame[ok]]
    return pSeg, pU


def _mosaicSingleParticles(part, pSeg, pU, segId, Dmin):
    """particles which are the only particle in their frame"""
    m = pSeg == segId
    uu, inv, cnt = np.unique(pU[m], return_inverse=True, return_counts=True)
    single = (cnt[inv] == 1) & (part["D"][m] >= Dmin)
    idx = np.flatnonzero(m)[single]
    order = np.argsort(pU[idx])
    return idx[order]


def _mosaicParticleLagTest(
    lPart,
    lSeg,
    lU,
    ls,
    fPart,
    fSeg,
    fU,
    fs,
    candidates,
    Dmin,
    minPairs=20,
    rotate=None,
    frameWidth=None,
):
    """
    For each candidate lag, pair single-particle frames of both cameras and
    measure the spread (IQR) of the vertical position difference. Only the
    correct lag gives a narrow distribution.

    Returns dict(lag, iqr, iqr2nd, n, dz) of the best candidate or None.
    """
    li = _mosaicSingleParticles(lPart, lSeg, lU, ls, Dmin)
    fi = _mosaicSingleParticles(fPart, fSeg, fU, fs, Dmin)
    if (len(li) < minPairs) or (len(fi) < minPairs):
        return None
    fUs = fU[fi]
    res = []
    for lag in candidates:
        target = lU[li] + lag
        jj = np.clip(np.searchsorted(fUs, target), 0, len(fUs) - 1)
        hit = fUs[jj] == target
        if hit.sum() < minPairs:
            continue
        if rotate is None:
            dz = lPart["z"][li[hit]] - fPart["z"][fi[jj[hit]]]
        else:
            # residual after applying the camera rotation: independent of
            # the position in the image, i.e. narrow for the correct lag even
            # if the cameras are rotated against each other (MOSAiC after
            # 5 May 2020: IQR of the raw difference > 120 px)
            from . import matching

            fIdx = fi[jj[hit]]
            dz = lPart["z"][li[hit]] - np.asarray(
                matching.calc_L_z_withOffsets(
                    lPart["xc"][li[hit]],
                    frameWidth - fPart["xc"][fIdx],
                    fPart["z"][fIdx],
                    **rotate,
                )
            )
        q75, q25 = np.percentile(dz, [75, 25])
        # fraction of pairs in a narrow peak around the median: robust
        # during heavy snowfall, when many "single particle" frames pair
        # different particles and broaden the IQR even for the correct lag
        peakFrac = np.mean(np.abs(dz - np.median(dz)) < 10)
        res.append((q75 - q25, lag, hit.sum(), np.median(dz), peakFrac))
    if len(res) < 2:
        return None
    res.sort()
    best = res[0]
    # In dense snowfall almost all pairs of single-particle frames are random
    # at every lag, the IQR is ~300 px for all of them and its minimum is
    # arbitrary, while the correct lag still shows a clear peak (fraction of
    # pairs within +-10 px of the median; 20200117 00:00: 0.23 at the right lag,
    # 0.04 at the lag with the smallest IQR). A decisive peak wins.
    peakBest = max(res, key=lambda r: r[4])
    others = [r[4] for r in res if r is not peakBest]
    if (
        (peakBest is not best)
        and (peakBest[4] >= 0.15)
        and (peakBest[4] >= 3.0 * max(others))
    ):
        best = peakBest
    rest = [r for r in res if r is not best]
    return {
        "lag": int(best[1]),
        "iqr": float(best[0]),
        "iqr2nd": float(min(r[0] for r in rest)),
        "n": int(best[2]),
        "dz": float(best[3]),
        "peakFrac": float(best[4]),
        "peakFrac2nd": float(max(r[4] for r in rest)),
    }


def _mosaicFramePairs(idxA, uA, idxB, uB, lag, maxFrames, maxPairs, rng):
    """all (A, B) particle index pairs of frames which are `lag` frames apart
    (frame of B = frame of A + lag); idxA/idxB sorted by frame number uA/uB,
    only frames with at most Kmax particles are expected here"""
    fa, sa, ca = np.unique(uA[idxA], return_index=True, return_counts=True)
    fb, sb, cb = np.unique(uB[idxB], return_index=True, return_counts=True)
    jj = np.clip(np.searchsorted(fb, fa + lag), 0, len(fb) - 1)
    hit = fb[jj] == fa + lag
    fa, sa, ca, jj = fa[hit], sa[hit], ca[hit], jj[hit]
    # random subset of frames within a budget of particle pairs
    perm = rng.permutation(len(fa))
    nPairs = np.cumsum((ca * cb[jj])[perm])
    perm = perm[nPairs <= maxPairs]
    if len(perm) > maxFrames:
        perm = perm[:maxFrames]
    sel = np.sort(perm)
    fa, sa, ca, jj = fa[sel], sa[sel], ca[sel], jj[sel]
    if len(fa) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    # every particle of A frame x every particle of the matching B frame
    pA = np.repeat(np.arange(len(fa)), ca)
    posA = np.concatenate([sa[i] + np.arange(ca[i]) for i in range(len(fa))])
    nB = cb[jj][pA]
    A = np.repeat(idxA[posA], nB)
    startB = np.repeat(sb[jj][pA], nB)
    offs = np.arange(nB.sum()) - np.repeat(np.cumsum(nB) - nB, nB)
    B = idxB[startB + offs]
    return A, B


def _mosaicDenseLagTest(
    lPart,
    lSeg,
    lU,
    ls,
    fPart,
    fSeg,
    fU,
    fs,
    candidates,
    Dmin=0,
    Kmax=200,
    maxFrames=3000,
    maxPairs=4_000_000,
    halfWidth=3,
    zRange=400,
    rotate=None,
    frameWidth=None,
    seed=0,
    fMask=None,
):
    """
    Lag test for dense snowfall, when (almost) no frame contains a single
    particle. All particles of a leader frame are paired with all particles
    of the follower frame at the candidate lag. The pairs of the particles
    seen by both cameras pile up at the vertical offset between the cameras
    (a sharp peak of a few pixels), all others are spread over the image.
    For a wrong lag the same particles are displaced by their fall speed
    (tens to hundreds of pixels, differing from particle to particle), i.e.
    the peak is smeared out.

    fMask restricts the follower particles (boolean per particle), e.g. to a
    time block.

    Returns dict(lag, z, z2nd, dz, excess, n) of the best candidate or None.
    z is the peak height over the background in units of its Poisson noise,
    excess the number of pairs in the peak above the background.
    """
    rng = np.random.default_rng(seed)

    def frames(part, seg, U, s):
        m = np.flatnonzero((seg == s) & (part["D"] >= Dmin))
        if len(m) == 0:
            return m
        m = m[np.argsort(U[m], kind="stable")]
        _, inv, cnt = np.unique(U[m], return_inverse=True, return_counts=True)
        return m[cnt[inv] <= Kmax]

    li = frames(lPart, lSeg, lU, ls)
    fi = frames(fPart, fSeg, fU, fs)
    if fMask is not None:
        fi = fi[fMask[fi]]
    if (len(li) < 200) or (len(fi) < 200):
        return None
    res = []
    for lag in candidates:
        A, B = _mosaicFramePairs(li, lU, fi, fU, lag, maxFrames, maxPairs, rng)
        if len(A) < 500:
            continue
        if rotate is None:
            dz = lPart["z"][A] - fPart["z"][B]
        else:
            from . import matching

            dz = lPart["z"][A] - np.asarray(
                matching.calc_L_z_withOffsets(
                    lPart["xc"][A],
                    frameWidth - fPart["xc"][B],
                    fPart["z"][B],
                    **rotate,
                )
            )
        h, edges = np.histogram(dz, bins=2 * zRange, range=(-zRange, zRange))
        w = 2 * halfWidth + 1
        win = np.convolve(h, np.ones(w), mode="same")
        ip = int(np.argmax(win[halfWidth:-halfWidth])) + halfWidth
        # background from the windows next to the peak (the difference of
        # two random positions is not flat but triangular)
        dist = np.abs(np.arange(len(win)) - ip)
        near = (dist > 2 * w) & (dist <= 12 * w)
        bg = float(np.mean(win[near]))
        excess = float(win[ip] - bg)
        z = excess / np.sqrt(bg + 1.0)
        res.append((z, int(lag), excess, float(edges[ip] + 0.5), len(A)))
    if len(res) < 2:
        return None
    res.sort(reverse=True)
    best = res[0]
    return {
        "lag": best[1],
        "z": best[0],
        "z2nd": res[1][0],
        "excess": best[2],
        "dz": best[3],
        "n": int(best[4]),
    }


def _mosaicDenseBlockLags(
    lPart,
    lSeg,
    lU,
    ls,
    fPart,
    fSeg,
    fU,
    fs,
    lagCenter,
    tStart,
    tEnd,
    dzRef=None,
    blockMinutes=10,
    halfRange=8,
    minZ=8.0,
    minRatio=1.5,
    maxDzDiff=10.0,
    rotate=None,
    frameWidth=None,
    track=False,
):
    """
    Lag per time block for long segments in which the lag changes (frames
    lost at the start of a recording, the leader/follower lag then drifts by
    one frame every few hours). The dense lag test (all particles of
    corresponding frames) is applied to blocks of the follower capture_time.

    With rotate (dict camera_Ofz/phi/theta) the vertical difference is
    corrected for the camera rotation (cameras rotated against each other, no
    constant raw offset); track=True moves the candidate window along with the
    lag found in the previous block.

    Returns a list of (t0, t1, lag or None, n, z) like _mosaicBlockLags.
    """
    out = []
    edges = np.arange(tStart, tEnd + 1, int(blockMinutes * 60_000_000))
    if edges[-1] < tEnd:
        edges = np.append(edges, tEnd + 1)
    center = lagCenter
    for t0, t1 in zip(edges[:-1], edges[1:]):
        cands = list(range(center - halfRange, center + halfRange + 1))
        mask = (fPart["ct"] >= t0) & (fPart["ct"] < t1)
        found = _mosaicDenseLagTest(
            lPart,
            lSeg,
            lU,
            ls,
            fPart,
            fSeg,
            fU,
            fs,
            cands,
            fMask=mask,
            rotate=rotate,
            frameWidth=frameWidth,
        )
        ok = (
            (found is not None)
            and (found["z"] >= minZ)
            and (found["z"] >= minRatio * max(found["z2nd"], 1.0))
            and ((dzRef is None) or (abs(found["dz"] - dzRef) < maxDzDiff))
        )
        if ok and track:
            # the lag drifts steadily: follow it from block to block
            center = int(found["lag"])
        out.append(
            (
                int(t0),
                int(t1),
                int(found["lag"]) if ok else None,
                int(found["n"]) if found is not None else 0,
                float(found["z"]) if found is not None else np.nan,
            )
        )
    return out


def _mosaicRotationGridSearch(lx, lz, fy, fz, phis, thetas, binw=1.0, win=2):
    """
    Camera rotation that makes most pairs agree: for each (phi, theta) of a
    grid the residual leader z - estimated leader z (Ofz = 0) of all pairs is
    histogrammed, the fullest bin (+-win bins) counts the consistent pairs and
    its position is the Ofz-like vertical shift. Returns (count, phi, theta,
    shift) of the best grid point.
    """
    from . import matching

    edges = np.arange(-1500, 1500 + binw, binw)
    best = (-1, np.nan, np.nan, np.nan)
    for ph in phis:
        for th in thetas:
            est = np.asarray(
                matching.calc_L_z_withOffsets(
                    lx, fy, fz, camera_phi=ph, camera_theta=th, camera_Ofz=0.0
                )
            )
            h, _ = np.histogram(lz - est, bins=edges)
            hs = np.convolve(h, np.ones(2 * win + 1), mode="same")
            k = int(np.argmax(hs))
            if hs[k] > best[0]:
                best = (int(hs[k]), ph, th, edges[k] + binw / 2)
    return best


def _mosaicRotationSearch(
    lPart,
    lSeg,
    lU,
    ls,
    fPart,
    fSeg,
    fU,
    fs,
    candidates,
    frameWidth,
    Dmin=8,
    maxPairs=30000,
    minPairs=100,
    phiRange=(-14.0, 2.0),
    thetaRange=(-4.0, 3.0),
    step=0.25,
    minRatio=3.0,
):
    """
    Frame lag AND camera rotation of a segment from single particle frames,
    without any a priori rotation: for each candidate lag the pairs of
    single-particle frames are scanned over a grid of camera angles, and the
    number of pairs that become consistent in height is counted. Only the
    correct lag has a rotation that makes a large fraction of the pairs agree
    (20200506: 36000 consistent pairs at the right lag, <1600 at the others).
    Useful where the cameras were moved (floe motion) and the rotation of the
    neighbouring periods does not apply.

    Returns a dict (lag, count, count2nd, n, Ofz, phi, theta) or None.
    """
    from . import matching

    li = _mosaicSingleParticles(lPart, lSeg, lU, ls, Dmin)
    fi = _mosaicSingleParticles(fPart, fSeg, fU, fs, Dmin)
    if (len(li) < minPairs) or (len(fi) < minPairs):
        return None
    fUs = fU[fi]
    rng = np.random.default_rng(0)
    phis = np.arange(phiRange[0], phiRange[1] + 1e-9, step)
    thetas = np.arange(thetaRange[0], thetaRange[1] + 1e-9, step)
    res = []
    for lag in candidates:
        target = lU[li] + lag
        jj = np.clip(np.searchsorted(fUs, target), 0, len(fUs) - 1)
        hit = fUs[jj] == target
        if hit.sum() < minPairs:
            continue
        a = li[hit]
        b = fi[jj[hit]]
        if len(a) > maxPairs:
            sel = rng.choice(len(a), maxPairs, replace=False)
            a, b = a[sel], b[sel]
        lx = lPart["xc"][a]
        lz = lPart["z"][a]
        fy = frameWidth - fPart["xc"][b]
        fz = fPart["z"][b]
        cnt, ph, th, shift = _mosaicRotationGridSearch(lx, lz, fy, fz, phis, thetas)
        res.append((cnt, lag, len(a), ph, th, shift, (lx, lz, fy, fz)))
    if len(res) < 2:
        return None
    res.sort(key=lambda r: -r[0])
    best = res[0]
    second = res[1][0]
    if best[0] < minRatio * max(second, 1):
        return None
    cnt, lag, n, ph, th, shift, (lx, lz, fy, fz) = best
    # refine: finer angles, then the vertical offset where the residual of the
    # consistent pairs is zero
    phis2 = np.arange(ph - 0.3, ph + 0.3001, 0.05)
    thetas2 = np.arange(th - 0.3, th + 0.3001, 0.05)
    cnt2, ph, th, shift = _mosaicRotationGridSearch(
        lx, lz, fy, fz, phis2, thetas2, binw=0.5, win=3
    )
    Ofz = float(shift)
    for _ in range(3):
        est = np.asarray(
            matching.calc_L_z_withOffsets(
                lx, fy, fz, camera_phi=ph, camera_theta=th, camera_Ofz=Ofz
            )
        )
        d = lz - est
        inl = np.abs(d) < 5
        if inl.sum() < 20:
            break
        Ofz += float(np.median(d[inl]))
    return {
        "lag": int(lag),
        "count": int(cnt2),
        "count2nd": int(second),
        "n": int(n),
        "Ofz": float(Ofz),
        "phi": float(ph),
        "theta": float(th),
    }


def _mosaicBlockLags(
    lPart,
    lSeg,
    lU,
    ls,
    fPart,
    fSeg,
    fU,
    fs,
    lagCenter,
    rotate,
    frameWidth,
    tStart,
    tEnd,
    blockMinutes=10,
    halfRange=60,
    minPairs=40,
    minFrac=0.25,
    minRatio=2.0,
    Dmin=8,
):
    """
    Frame lag in consecutive time blocks of one leader/follower run, for runs
    in which the lag is not constant. After the camera time synchronisation
    failed in gochang_v1 (20251205-20251221) the lag drifted by one frame
    every ~1.5 h (20251214: -229 at 00 UTC, -216 at 20 UTC), and a single lag
    per run resolves nothing. The correct lag of a block is the one for which
    most pairs of single-particle frames agree in height with the known
    camera rotation (60-90 % of the pairs, against ~5 % at the other lags).

    Returns list of (t0, t1, lag, n, fraction) with t0, t1 in follower
    capture_time (us); blocks without a decisive lag are returned with
    lag None.
    """
    from . import matching

    li = _mosaicSingleParticles(lPart, lSeg, lU, ls, Dmin)
    fi = _mosaicSingleParticles(fPart, fSeg, fU, fs, Dmin)
    if (len(li) < minPairs) or (len(fi) < minPairs):
        return []
    lUs = lU[li]
    lx = lPart["xc"][li]
    lz = lPart["z"][li]
    fUsAll = fU[fi]
    fct = fPart["ct"][fi]
    blockUs = int(blockMinutes * 60e6)
    edges = np.arange(tStart, tEnd + blockUs, blockUs)
    out = []
    for t0, t1 in zip(edges[:-1], edges[1:]):
        sel = (fct >= t0) & (fct < t1)
        if sel.sum() < minPairs:
            out.append((int(t0), int(t1), None, int(sel.sum()), np.nan))
            continue
        fIdx = fi[sel]
        fUb = fUsAll[sel]
        order = np.argsort(fUb)
        fIdx, fUb = fIdx[order], fUb[order]
        lo = np.searchsorted(lUs, fUb.min() - lagCenter - halfRange - 5)
        hi = np.searchsorted(lUs, fUb.max() - lagCenter + halfRange + 5)
        if hi - lo < minPairs:
            out.append((int(t0), int(t1), None, int(hi - lo), np.nan))
            continue
        lUb, lxb, lzb = lUs[lo:hi], lx[lo:hi], lz[lo:hi]
        scores = []
        for lag in range(lagCenter - halfRange, lagCenter + halfRange + 1):
            tgt = lUb + lag
            jj = np.clip(np.searchsorted(fUb, tgt), 0, len(fUb) - 1)
            hit = fUb[jj] == tgt
            n = int(hit.sum())
            if n < minPairs:
                continue
            b = fIdx[jj[hit]]
            est = np.asarray(
                matching.calc_L_z_withOffsets(
                    lxb[hit], frameWidth - fPart["xc"][b], fPart["z"][b], **rotate
                )
            )
            d = lzb[hit] - est
            cnt = int((np.abs(d - np.median(d)) < 3).sum())
            scores.append((cnt / n, cnt, n, lag))
        if len(scores) < 2:
            out.append((int(t0), int(t1), None, 0, np.nan))
            continue
        scores.sort(reverse=True)
        best, second = scores[0], scores[1]
        if (best[0] >= minFrac) and (best[0] >= minRatio * second[0]):
            out.append((int(t0), int(t1), int(best[3]), int(best[2]), float(best[0])))
        else:
            out.append((int(t0), int(t1), None, int(best[2]), float(best[0])))
    return out


def _mosaicDayRotation(case, config):
    """function (rt_start, rt_end in us) -> median rotation of the daily
    metaRotation within that period, or None if there is no metaRotation"""
    import os

    from . import files

    fn = files.FindFiles(case, config.leader, config).fnamesDaily["metaRotation"]
    if not os.path.isfile(fn):
        return None
    with xr.open_dataset(fn) as ds:
        ds = ds.sel(camera_rotation="mean").load()
    keys = ["camera_Ofz", "camera_phi", "camera_theta"]
    t = np.atleast_1d(ds.file_starttime.values).astype("datetime64[us]").astype(np.int64)
    vals = {k: np.atleast_1d(ds[k].values).astype(float) for k in keys}
    good = np.all([np.isfinite(v) for v in vals.values()], axis=0)
    if good.sum() == 0:
        return None
    t = t[good]
    vals = {k: v[good] for k, v in vals.items()}

    def get(t0, t1):
        sel = (t >= t0 - 600e6) & (t <= t1)
        if not sel.any():
            sel = np.ones(len(t), dtype=bool)
        return {k: float(np.nanmedian(v[sel])) for k, v in vals.items()}

    return get


def mosaicFrameMappingFname(case, config):
    """daily cache file next to metaRotation"""
    from . import files

    fl = files.FindFiles(case, config.leader, config)
    return fl.fnamesDaily["metaRotation"].replace("metaRotation", "metaFrameMapping")


def createMosaicFrameMapping(
    case,
    config,
    skipExisting=True,
    writeNc=True,
    minOverlapS=60,
    maxIqr=60.0,
    minIqrRatio=1.5,
    activitySearchRange=20000,
    minPeakFrac=0.15,
    minPeakFracRatio=3.0,
    minConsensusIqrRatio=1.1,
    minConsensusActivityZ=5.0,
    useRotation=False,
    maxIqrRotated=15.0,
    minIqrRatioRotated=2.0,
    rotation=None,
    searchRotation=False,
    searchWide=True,
    blockLags=False,
    denseLags=False,
    denseBlockLags=True,
    minDenseZ=10.0,
    minDenseZRatio=2.0,
):
    """
    Determine, for one day, the exact follower -> leader capture_id mapping
    for every combination of leader and follower camera run.

    For every overlap of a leader run and a follower run:
    1. first guess of the lag from record_time (median over the overlap),
    2. clock-independent check by cross correlating the per-frame "anything
       moving" signal of both cameras,
    3. exact lag from the vertical position of single particles seen by
       both cameras (tested for small and for large particles, which are
       rarer and thus less ambiguous during heavy snowfall).

    Ghost frames are removed from the follower before (see module notes).
    The mapping is cached as metaFrameMapping netCDF next to metaRotation.

    Parameters
    ----------
    case : str
        Day YYYYMMDD
    config : dict or str
        Settings
    skipExisting : bool
        Read cached file if present
    writeNc : bool
        Write cache file
    minOverlapS : float
        Minimum overlap of leader and follower run in seconds
    maxIqr : float
        Maximum IQR (px) of the vertical position difference for the best lag
    minIqrRatio : float
        Minimum ratio IQR(2nd best lag)/IQR(best lag)
    minConsensusActivityZ : float
        Minimum z-score of the activity cross correlation peak for the
        consensus route (the activity lag must agree within 1 frame with
        the record_time and the particle lag)
    minConsensusIqrRatio : float
        Alternative acceptance if the best lag agrees within 1 frame with
        both the activity (z >= 8) and the record_time estimate: minimum
        IQR ratio of 2nd best/best lag
    minPeakFrac, minPeakFracRatio : float
        Alternative acceptance when the IQR is broad (heavy snowfall): the
        fraction of pairs within +-10 px of the median must be at least
        minPeakFrac and minPeakFracRatio times that of every other lag
    useRotation : bool
        Second chance for segments the other criteria could not resolve:
        repeat the particle test with the vertical position difference
        corrected by the camera rotation of the day's metaRotation (needs
        that file, i.e. only possible after createMetaRotation). Accepted
        if the IQR of the residual is <= maxIqrRotated and the next best
        lag has at least minIqrRatioRotated times the IQR. Necessary when
        the cameras are rotated against each other (raw IQR > 100 px).
    denseLags : bool
        Second chance for segments without enough single-particle frames
        (dense snowfall): pair all particles of corresponding frames and look
        for a sharp peak of the vertical offset (see _mosaicDenseLagTest).
        Accepted if the peak is minDenseZ noise sigmas high, minDenseZRatio
        times higher than for any other candidate lag and (if other
        segments of the day are resolved) within 10 px of their offset.
    denseBlockLags : bool
        With denseLags: segments of >= 20 min in which the dense test finds a
        lag with the right vertical offset but not decisively (the lag drifts
        within the segment) are split into 10 min blocks, see
        _mosaicDenseBlockLags
    blockLags : bool
        Last resort for long unresolved segments: lag per time block with the
        rotation of the day (or the config prior), see _mosaicBlockLags
    searchWide : bool
        Also use the peak of the activity correlation over all lags as a
        candidate (clocks far apart)
    searchRotation : bool
        Last resort for segments that are still unresolved (also after the
        rotation-corrected test): search lag and camera rotation together
        (see _mosaicRotationSearch). The rotation found is stored as rot_Ofz,
        rot_phi and rot_theta of the segment.
    rotation : dict, optional
        Use this rotation (camera_Ofz, camera_phi, camera_theta) instead of
        the one of the metaRotation file. The residual median of the
        result tells how far Ofz is off, the IQR whether phi and theta are
        right.
    activitySearchRange : int
        Frames around the record_time guess searched by the activity cross
        correlation. Must be wide because the clocks of the two computers
        were not always synchronized (e.g. ~5 s off end of Nov 2019).

    Returns
    -------
    xarray.Dataset or None
        Dataset with dimension "segment" (one per leader/follower run
        overlap; lag, quality flags, follower capture_time range) and
        "ghost" (follower ghost frame sequences)
    """
    import datetime
    import os
    import uuid

    from . import __version__, tools

    config = tools.readSettings(config)
    case = case.split("-")[0]
    fname = mosaicFrameMappingFname(case, config)
    if skipExisting and os.path.isfile(fname):
        with xr.open_dataset(fname) as ds:
            cached = ds.load()
        # files from before the clock offset knots were added must be rebuilt;
        # a day without any resolved segment has no knots by construction
        if (
            ("segment" not in cached.dims)
            or ("knot_time" in cached)
            or (not bool(cached["resolved"].any()))
        ):
            return cached
        if not writeNc:
            raise RuntimeError(
                f"{fname} is outdated (no clock offset knots), rebuild with "
                "VISSSlib.fixes.createMosaicFrameMapping(case, config, "
                "skipExisting=False)"
            )
        log.warning(f"{fname} is outdated (no clock offset knots), rebuilding")

    log.info(f"createMosaicFrameMapping {case}")
    P = MOSAIC_CAPTURE_ID_PERIOD
    L = _mosaicLoadMetaFrames(case, config.leader, config)
    F = _mosaicLoadMetaFrames(case, config.follower, config)

    rows = []
    ghostRows = []
    knotRows = []
    candidateSets = {}
    if (L is not None) and (F is not None):
        lPeriod = _mosaicFramePeriod(L, config)
        fPeriod = _mosaicFramePeriod(F, config)
        lSegF, lu, lDrop, lGhosts = _mosaicSegmentsAndGhosts(L, lPeriod)
        fSegF, fu, fDrop, fGhosts = _mosaicSegmentsAndGhosts(F, fPeriod)
        if len(lGhosts) > 0:
            # never observed so far; leader capture_id is used as is
            log.warning(f"{len(lGhosts)} ghost frame sequences in LEADER data")
            lSegF, lu, lDrop, _ = _mosaicSegmentsAndGhosts(
                L, lPeriod, correctGhosts=False
            )
        log.info(f"{len(fGhosts)} follower ghost frame sequences")
        for i0, i1, extra in fGhosts:
            ghostRows.append(
                {
                    "ghost_fsegment": fSegF[i0],
                    "ghost_ct_start": F["ct"][i0],
                    "ghost_ct_end": F["ct"][i1],
                    "ghost_extra": extra,
                }
            )

        lPart = _mosaicLoadParticles(case, config.leader, config)
        fPart = _mosaicLoadParticles(case, config.follower, config)
        if (lPart is not None) and (fPart is not None):
            lPSeg, lPU = _mosaicParticleFrames(lPart, L, lSegF, lu, lDrop, lPeriod)
            fPSeg, fPU = _mosaicParticleFrames(fPart, F, fSegF, fu, fDrop, fPeriod)

        fKeep = ~fDrop
        for ls in np.unique(lSegF):
            lm = lSegF == ls
            for fs in np.unique(fSegF):
                fm = (fSegF == fs) & fKeep
                if fm.sum() == 0:
                    continue
                t0 = max(L["rt"][lm].min(), F["rt"][fm].min())
                t1 = min(L["rt"][lm].max(), F["rt"][fm].max())
                if (t1 - t0) < minOverlapS * 1e6:
                    continue
                lmm = lm & (L["rt"] >= t0) & (L["rt"] <= t1)
                fmm = fm & (F["rt"] >= t0) & (F["rt"] <= t1)
                if (lmm.sum() < 1000) or (fmm.sum() < 1000):
                    continue
                row = {
                    "lsegment": ls,
                    "fsegment": fs,
                    "rt_start": t0,
                    "rt_end": t1,
                    "fct_start": F["ct"][fmm].min(),
                    "fct_end": F["ct"][fmm].max(),
                    "lag_recordTime": -1,
                    "lag_activity": -1,
                    "activity_z": np.nan,
                    "lag": -1,
                    "lag_mod": -1,
                    "dz_iqr": np.nan,
                    "dz_iqr_2nd": np.nan,
                    "dz_median": np.nan,
                    "n_pairs": 0,
                    "dmin": 0,
                    "peak_frac": np.nan,
                    "peak_frac_2nd": np.nan,
                    "resolved": False,
                    "rotated": False,
                    "dense": False,
                    "rot_Ofz": np.nan,
                    "rot_phi": np.nan,
                    "rot_theta": np.nan,
                }

                # 1. record_time first guess
                lIdx = np.flatnonzero(lmm)
                lIdx = lIdx[
                    np.linspace(0, len(lIdx) - 1, min(20000, len(lIdx))).astype(int)
                ]
                fIdx = np.flatnonzero(fmm)
                fOrder = np.argsort(F["rt"][fIdx], kind="stable")
                fRt = F["rt"][fIdx][fOrder]
                jj = np.clip(np.searchsorted(fRt, L["rt"][lIdx]), 1, len(fRt) - 1)
                jj = np.where(
                    np.abs(fRt[jj - 1] - L["rt"][lIdx])
                    < np.abs(fRt[jj] - L["rt"][lIdx]),
                    jj - 1,
                    jj,
                )
                close = np.abs(fRt[jj] - L["rt"][lIdx]) < 20e3
                if close.sum() < 100:
                    rows.append(row)
                    continue
                guess = int(np.median(fu[fIdx][fOrder][jj[close]] - lu[lIdx[close]]))
                row["lag_recordTime"] = guess

                # 2. activity cross correlation
                lagAct, zAct = _mosaicActivityLag(
                    lu[lmm],
                    L["nmp"][lmm],
                    fu[fmm],
                    F["nmp"][fmm],
                    guess,
                    searchRange=activitySearchRange,
                )
                candidates = set(range(guess - 3, guess + 4))
                if lagAct is not None:
                    row["lag_activity"] = lagAct
                    row["activity_z"] = zAct
                    if zAct > 8:
                        candidates |= set(range(lagAct - 3, lagAct + 4))
                # the record_time guess is off by many minutes if the cameras'
                # clocks are (gochang_v1 20251214: 29 min, i.e. 240000 frames,
                # far outside the search range): the unrestricted peak of the
                # activity correlation is a further candidate
                if searchWide:
                    lagWide, zWide = _mosaicActivityLag(
                        lu[lmm],
                        L["nmp"][lmm],
                        fu[fmm],
                        F["nmp"][fmm],
                        guess,
                        searchRange=10**9,
                    )
                    if (lagWide is not None) and (zWide > 8):
                        candidates |= set(range(lagWide - 3, lagWide + 4))

                # 3. exact lag from particles
                if (lPart is None) or (fPart is None):
                    rows.append(row)
                    continue
                candidateSets[id(row)] = (ls, fs, sorted(candidates))
                best = None
                for Dmin in [8, 20]:
                    res = _mosaicParticleLagTest(
                        lPart,
                        lPSeg,
                        lPU,
                        ls,
                        fPart,
                        fPSeg,
                        fPU,
                        fs,
                        sorted(candidates),
                        Dmin,
                    )
                    if res is None:
                        continue
                    res["dmin"] = Dmin
                    res["ratio"] = res["iqr2nd"] / max(res["iqr"], 1e-3)
                    if (best is None) or (res["ratio"] > best["ratio"]):
                        if (best is not None) and (best["lag"] != res["lag"]):
                            # both particle sizes are decisive but disagree
                            if min(best["ratio"], res["ratio"]) >= minIqrRatio:
                                best = None
                                break
                        best = res
                if best is not None:
                    row.update(
                        {
                            "lag": best["lag"],
                            "lag_mod": best["lag"] % P,
                            "dz_iqr": best["iqr"],
                            "dz_iqr_2nd": best["iqr2nd"],
                            "dz_median": best["dz"],
                            "n_pairs": best["n"],
                            "dmin": best["dmin"],
                            "resolved": (best["iqr"] <= maxIqr)
                            and (best["ratio"] >= minIqrRatio),
                            "resolved_peak": (
                                (best["peakFrac"] >= minPeakFrac)
                                and (
                                    best["peakFrac"]
                                    >= minPeakFracRatio * best["peakFrac2nd"]
                                )
                            )
                            # consensus: three independent estimates agree
                            # (only possible if the computer clocks are in
                            # sync); the particle test is then just not
                            # decisive, typically because particles move
                            # little within one frame
                            or (
                                (best["ratio"] >= minConsensusIqrRatio)
                                and (row["activity_z"] >= minConsensusActivityZ)
                                and (abs(best["lag"] - row["lag_activity"]) <= 1)
                                and (abs(best["lag"] - row["lag_recordTime"]) <= 1)
                            ),
                            "peak_frac": best["peakFrac"],
                            "peak_frac_2nd": best["peakFrac2nd"],
                        }
                    )
                rows.append(row)

        # alternative acceptance (peak fraction or consensus), but only if the vertical
        # offset agrees with the segments of the same day resolved by the IQR
        # criterion (camera geometry does not change within a day)
        # (if none, the peak-qualified segments themselves are the reference)
        dzRef = [r["dz_median"] for r in rows if r["resolved"]]
        if len(dzRef) == 0:
            dzRef = [r["dz_median"] for r in rows if r.get("resolved_peak", False)]
        for r in rows:
            peakOk = r.pop("resolved_peak", False)
            if (not r["resolved"]) and peakOk and (len(dzRef) > 0):
                r["resolved"] = bool(abs(r["dz_median"] - np.median(dzRef)) < 10)

        if denseLags and (lPart is not None) and (fPart is not None):
            # reference vertical offset of the day from well populated
            # segments only (a few dozen pairs can be a false detection and
            # would make the dense test reject every correct peak)
            dzDay = [
                r["dz_median"] for r in rows if r["resolved"] and r["n_pairs"] >= 300
            ]
            dzDay = float(np.median(dzDay)) if len(dzDay) > 0 else None
            denseDrift = {}
            for r in rows:
                if r["resolved"] or (id(r) not in candidateSets):
                    continue
                ls, fs, cc = candidateSets[id(r)]
                found = _mosaicDenseLagTest(
                    lPart, lPSeg, lPU, ls, fPart, fPSeg, fPU, fs, cc
                )
                log.info(
                    f"dense lag test segment {r['lsegment']}/{r['fsegment']}: "
                    + (
                        "no result"
                        if found is None
                        else f"lag {found['lag']} z {found['z']:.1f} "
                        f"2nd {found['z2nd']:.1f} dz {found['dz']:.1f} "
                        f"excess {found['excess']:.0f} (day dz {dzDay})"
                    )
                )
                if (
                    (found is not None)
                    and (found["z"] >= minDenseZ)
                    and (found["z"] >= minDenseZRatio * max(found["z2nd"], 1.0))
                    and ((dzDay is None) or (abs(found["dz"] - dzDay) < 10))
                ):
                    r.update(
                        {
                            "lag": found["lag"],
                            "lag_mod": found["lag"] % P,
                            "dz_median": found["dz"],
                            "n_pairs": int(found["excess"]),
                            "resolved": True,
                            "dense": True,
                        }
                    )
                elif (
                    denseBlockLags
                    and (found is not None)
                    and (found["z"] >= minDenseZ)
                    and ((dzDay is None) or (abs(found["dz"] - dzDay) < 10))
                ):
                    denseDrift[id(r)] = (ls, fs, found["lag"])

            newRows = []
            for r in rows:
                if id(r) not in denseDrift:
                    newRows.append(r)
                    continue
                ls, fs, lagC = denseDrift[id(r)]
                durMin = (r["fct_end"] - r["fct_start"]) / 60e6
                if durMin < 20:
                    newRows.append(r)
                    continue
                blocks = _mosaicDenseBlockLags(
                    lPart,
                    lPSeg,
                    lPU,
                    ls,
                    fPart,
                    fPSeg,
                    fPU,
                    fs,
                    int(lagC),
                    int(r["fct_start"]),
                    int(r["fct_end"]),
                    dzRef=dzDay,
                )
                goodIdx = [i for i, b in enumerate(blocks) if b[2] is not None]
                log.info(
                    f"dense block lags segment {r['lsegment']}/{r['fsegment']}: "
                    f"{len(goodIdx)} of {len(blocks)} blocks, lags "
                    + str(sorted({blocks[i][2] for i in goodIdx}))[:120]
                )
                if len(goodIdx) < 3:
                    newRows.append(r)
                    continue
                # blocks without a decisive peak (little snow): the lag
                # drifts steadily, so interpolate between the neighbouring
                # blocks (flat before the first / after the last one)
                goodLag = np.array([blocks[i][2] for i in goodIdx], dtype=float)
                filled = []
                for i, b in enumerate(blocks):
                    if b[2] is None:
                        nearest = min(goodIdx, key=lambda j: abs(j - i))
                        if abs(nearest - i) <= 18:
                            lagI = int(np.round(np.interp(i, goodIdx, goodLag)))
                            b = (b[0], b[1], lagI, b[3], np.nan)
                    filled.append(b)
                cur = None
                for b in filled:
                    if (b[2] is None) and (cur is None):
                        continue
                    if (cur is not None) and (b[2] == cur["lag"]):
                        cur["fct_end"] = b[1] - 1
                        cur["n_pairs"] += b[3]
                        continue
                    if cur is not None:
                        newRows.append(cur)
                        cur = None
                    if b[2] is not None:
                        cur = dict(r)
                        cur.update(
                            {
                                "fct_start": b[0],
                                "fct_end": b[1] - 1,
                                "lag": b[2],
                                "lag_mod": b[2] % P,
                                "n_pairs": max(b[3], 1),
                                "resolved": True,
                                "dense": True,
                            }
                        )
                if cur is not None:
                    newRows.append(cur)
            rows[:] = newRows

        if useRotation and (lPart is not None) and (fPart is not None):
            if rotation is not None:
                getRotation = lambda t0, t1: dict(rotation)  # noqa: E731
            else:
                getRotation = _mosaicDayRotation(case, config)
            for r in rows:
                if r["resolved"] or (id(r) not in candidateSets) or (getRotation is None):
                    continue
                ls, fs, cc = candidateSets[id(r)]
                rotate = getRotation(r["rt_start"], r["rt_end"])
                best = None
                for Dmin in [8, 20]:
                    res = _mosaicParticleLagTest(
                        lPart,
                        lPSeg,
                        lPU,
                        ls,
                        fPart,
                        fPSeg,
                        fPU,
                        fs,
                        cc,
                        Dmin,
                        rotate=rotate,
                        frameWidth=config.frame_width,
                    )
                    if res is None:
                        continue
                    res["dmin"] = Dmin
                    res["ratio"] = res["iqr2nd"] / max(res["iqr"], 1e-3)
                    if (best is None) or (res["ratio"] > best["ratio"]):
                        if (best is not None) and (best["lag"] != res["lag"]):
                            if min(best["ratio"], res["ratio"]) >= minIqrRatioRotated:
                                best = None
                                break
                        best = res
                log.info(
                    f"rotated lag test segment {r['lsegment']}/{r['fsegment']} "
                    f"rotation {rotate}: "
                    + (
                        "no result"
                        if best is None
                        else f"lag {best['lag']} iqr {best['iqr']:.1f} "
                        f"2nd {best['iqr2nd']:.1f} n {best['n']} "
                        f"residual median {best['dz']:.1f}"
                    )
                )
                if (
                    (best is not None)
                    and (best["iqr"] <= maxIqrRotated)
                    and (best["ratio"] >= minIqrRatioRotated)
                ):
                    r.update(
                        {
                            "lag": best["lag"],
                            "lag_mod": best["lag"] % P,
                            "dz_iqr": best["iqr"],
                            "dz_iqr_2nd": best["iqr2nd"],
                            "dz_median": best["dz"],
                            "n_pairs": best["n"],
                            "dmin": best["dmin"],
                            "peak_frac": best["peakFrac"],
                            "peak_frac_2nd": best["peakFrac2nd"],
                            "resolved": True,
                            "rotated": True,
                        }
                    )

        if searchRotation and (lPart is not None) and (fPart is not None):
            for r in rows:
                if r["resolved"] or (id(r) not in candidateSets):
                    continue
                ls, fs, cc = candidateSets[id(r)]
                if r["lag_activity"] != -1:
                    # the activity lag is a candidate however weak the peak
                    cc = sorted(set(cc) | set(range(r["lag_activity"] - 3, r["lag_activity"] + 4)))
                found = _mosaicRotationSearch(
                    lPart,
                    lPSeg,
                    lPU,
                    ls,
                    fPart,
                    fPSeg,
                    fPU,
                    fs,
                    cc,
                    config.frame_width,
                )
                log.info(
                    f"rotation search segment {r['lsegment']}/{r['fsegment']}: "
                    + ("no result" if found is None else str(found))
                )
                if found is not None:
                    r.update(
                        {
                            "lag": found["lag"],
                            "lag_mod": found["lag"] % P,
                            "n_pairs": found["n"],
                            "dz_iqr": np.nan,
                            "dz_median": np.nan,
                            "resolved": True,
                            "rotated": True,
                            "rot_Ofz": found["Ofz"],
                            "rot_phi": found["phi"],
                            "rot_theta": found["theta"],
                        }
                    )

        if denseBlockLags and (lPart is not None) and (fPart is not None):
            # long segments resolved by the rotation search got ONE lag for
            # the whole run, but the lag drifts steadily (SAIL 20230209: 14 h
            # run, wrong lag for hours): track it block by block
            newRows = []
            for r in rows:
                durMin = (r["fct_end"] - r["fct_start"]) / 60e6
                if not (r["resolved"] and r["rotated"] and np.isfinite(r["rot_Ofz"]) and durMin >= 20):
                    newRows.append(r)
                    continue
                rot = dict(
                    camera_Ofz=r["rot_Ofz"],
                    camera_phi=r["rot_phi"],
                    camera_theta=r["rot_theta"],
                )
                blocks = _mosaicDenseBlockLags(
                    lPart,
                    lPSeg,
                    lPU,
                    r["lsegment"],
                    fPart,
                    fPSeg,
                    fPU,
                    r["fsegment"],
                    int(r["lag"]),
                    int(r["fct_start"]),
                    int(r["fct_end"]),
                    rotate=rot,
                    frameWidth=config.frame_width,
                    track=True,
                    halfRange=6,
                )
                goodIdx = [i for i, b in enumerate(blocks) if b[2] is not None]
                log.info(
                    f"rotated dense block lags segment {r['lsegment']}/{r['fsegment']}: "
                    f"{len(goodIdx)} of {len(blocks)} blocks, lags "
                    + str(sorted({blocks[i][2] for i in goodIdx}))[:120]
                )
                if len(goodIdx) < max(3, len(blocks) // 4):
                    newRows.append(r)
                    continue
                goodLag = np.array([blocks[i][2] for i in goodIdx], dtype=float)
                filled = []
                for i, b in enumerate(blocks):
                    if b[2] is None:
                        lagI = int(np.round(np.interp(i, goodIdx, goodLag)))
                        b = (b[0], b[1], lagI, b[3], np.nan)
                    filled.append(b)
                cur = None
                for b in filled:
                    if (cur is not None) and (b[2] == cur["lag"]):
                        cur["fct_end"] = b[1] - 1
                        cur["n_pairs"] += b[3]
                        continue
                    if cur is not None:
                        newRows.append(cur)
                    cur = dict(r)
                    cur.update(
                        {
                            "fct_start": b[0],
                            "fct_end": b[1] - 1,
                            "lag": b[2],
                            "lag_mod": b[2] % P,
                            "n_pairs": max(b[3], 1),
                            "dense": True,
                        }
                    )
                if cur is not None:
                    newRows.append(cur)
            rows[:] = newRows

        if blockLags and (lPart is not None) and (fPart is not None):
            newRows = []
            getRot = _mosaicDayRotation(case, config)
            for r in rows:
                durMin = (r["rt_end"] - r["rt_start"]) / 60e6
                if r["resolved"] or (r["lag_recordTime"] == -1) or (durMin < 20):
                    newRows.append(r)
                    continue
                if getRot is not None:
                    rotate = getRot(r["rt_start"], r["rt_end"])
                else:
                    rotate, _, _ = tools.getPrevRotationEstimates(
                        np.datetime64(
                            f"{case[:4]}-{case[4:6]}-{case[6:8]}"
                        ),
                        config,
                    )
                    rotate = dict(rotate)
                blocks = _mosaicBlockLags(
                    lPart,
                    lPSeg,
                    lPU,
                    r["lsegment"],
                    fPart,
                    fPSeg,
                    fPU,
                    r["fsegment"],
                    int(r["lag_recordTime"]),
                    rotate,
                    config.frame_width,
                    int(r["fct_start"]),
                    int(r["fct_end"]),
                )
                good = [b for b in blocks if b[2] is not None]
                # blocks without enough single-particle frames take the lag of
                # the nearest block with one (the lag only changes by one frame
                # every ~1.5 h; a wrong lag cannot create false matches because
                # it displaces the particles by ~200 px)
                goodIdx = [i for i, b in enumerate(blocks) if b[2] is not None]
                if len(goodIdx) >= 3:
                    filled = []
                    for i, b in enumerate(blocks):
                        if b[2] is None:
                            nearest = min(goodIdx, key=lambda j: abs(j - i))
                            if abs(nearest - i) <= 18:
                                b = (b[0], b[1], blocks[nearest][2], b[3], np.nan)
                        filled.append(b)
                    blocks = filled
                log.info(
                    f"block lags segment {r['lsegment']}/{r['fsegment']}: "
                    f"{len(good)} of {len(blocks)} blocks, lags "
                    + str(sorted({b[2] for b in good}))[:120]
                )
                if len(good) < 3:
                    newRows.append(r)
                    continue
                # merge neighbouring blocks with the same lag into one row
                cur = None
                for b in blocks:
                    if (b[2] is None) and (cur is None):
                        continue
                    if (cur is not None) and (b[2] == cur["lag"]):
                        cur["fct_end"] = b[1] - 1
                        cur["n_pairs"] += b[3]
                        continue
                    if cur is not None:
                        newRows.append(cur)
                        cur = None
                    if b[2] is not None:
                        cur = dict(r)
                        cur.update(
                            {
                                "fct_start": b[0],
                                "fct_end": b[1] - 1,
                                "lag": b[2],
                                "lag_mod": b[2] % P,
                                "n_pairs": b[3],
                                "resolved": True,
                                "rotated": True,
                            }
                        )
                if cur is not None:
                    newRows.append(cur)
            rows[:] = newRows

        # clock offset between the cameras: capture_time of a follower frame
        # minus capture_time of the same frame of the leader. The clocks of
        # the cameras are ~1-2 s apart and drift ~28 ppm, so follower
        # capture_time cannot be compared with the leader's directly (the
        # matching cuts the follower to the time range of leader slices,
        # which fails for short slices, i.e. during dense snowfall).
        for ir, r in enumerate(rows):
            if r["n_pairs"] <= 0:
                continue
            lm = lSegF == r["lsegment"]
            fm = (
                (fSegF == r["fsegment"])
                & fKeep
                & (F["ct"] >= r["fct_start"])
                & (F["ct"] <= r["fct_end"])
            )
            fUs = fu[fm]
            fct = F["ct"][fm]
            target = lu[lm] + r["lag"]
            jj = np.clip(np.searchsorted(fUs, target), 0, len(fUs) - 1)
            ok = fUs[jj] == target
            if ok.sum() < 100:
                continue
            tF = fct[jj[ok]]
            offs = (tF - L["ct"][lm][ok]) * 1e-6
            bins = ((tF - tF[0]) // 60_000_000).astype(int)
            for b in np.unique(bins):
                sel = bins == b
                if sel.sum() < 50:
                    continue
                knotRows.append(
                    {
                        "knot_segment": ir,
                        "knot_time": int(np.median(tF[sel])),
                        "knot_offset": float(np.median(offs[sel])),
                    }
                )

    def asArray(key, rr, dim, dtype=None):
        return xr.DataArray(np.array([r[key] for r in rr], dtype=dtype), dims=[dim])

    ds = xr.Dataset()
    if len(rows) > 0:
        for key in rows[0].keys():
            if key in ["rt_start", "rt_end", "fct_start", "fct_end"]:
                ds[key] = (
                    asArray(key, rows, "segment", np.int64)
                    .astype("datetime64[us]")
                    .astype("datetime64[ns]")
                )
            elif key in ["resolved", "rotated", "dense"]:
                ds[key] = asArray(key, rows, "segment", np.int8)
            else:
                ds[key] = asArray(key, rows, "segment")
    if len(ghostRows) > 0:
        for key in ghostRows[0].keys():
            if key in ["ghost_ct_start", "ghost_ct_end"]:
                ds[key] = (
                    asArray(key, ghostRows, "ghost", np.int64)
                    .astype("datetime64[us]")
                    .astype("datetime64[ns]")
                )
            else:
                ds[key] = asArray(key, ghostRows, "ghost", np.int64)
    if len(knotRows) > 0:
        for key in knotRows[0].keys():
            if key == "knot_time":
                ds[key] = (
                    asArray(key, knotRows, "knot", np.int64)
                    .astype("datetime64[us]")
                    .astype("datetime64[ns]")
                )
            else:
                ds[key] = asArray(key, knotRows, "knot")
        ds["knot_offset"].attrs["units"] = "s"
        ds["knot_offset"].attrs[
            "long_name"
        ] = "follower minus leader capture_time of the same frame"
    ds.attrs["description"] = (
        "MOSAiC leader/follower frame mapping: leader capture_id = "
        "((follower capture_id - ghost_extra of preceding ghosts in the same "
        "fsegment - lag_mod - 1) mod 65535) + 1 for follower frames with "
        "fct_start <= capture_time <= fct_end of a resolved segment"
    )
    ds.attrs[
        "history"
    ] = f"{datetime.datetime.utcnow()}: created with VISSSlib {__version__}"
    nRes = int(ds.resolved.sum()) if "resolved" in ds else 0
    log.info(
        f"createMosaicFrameMapping {case}: {nRes} of {len(rows)} segments resolved"
    )

    if writeNc:
        tools.createParentDir(fname, mode=config.dirMode)
        tmpFile = f"{fname}.{os.getpid()}.{uuid.uuid4().hex}.tmp.cdf"
        ds.to_netcdf(tmpFile)
        os.chmod(tmpFile, config.fileMode)
        os.replace(tmpFile, fname)
        log.info(f"saved {fname}")
    return ds


def mosaicFollowerCaptureIdToLeader(follower1D, config, dim="fpid"):
    """
    Replace the (raw, not yet overflow corrected) follower capture_id by the
    capture_id the leader assigned to the same trigger, using the daily
    metaFrameMapping. If missing (e.g. the next day around midnight), it is
    computed on the fly but not cached, because level1detect of that day
    might not be complete yet; the cache is written by createMetaRotation.
    Particles in ghost frame sequences or in periods without reliable
    mapping are removed.

    Parameters
    ----------
    follower1D : xarray.Dataset
        follower level1detect data with raw capture_id
    config : dict
        Settings
    dim : str
        particle dimension

    Returns
    -------
    xarray.Dataset or None
    """
    import pandas as pd

    P = MOSAIC_CAPTURE_ID_PERIOD
    ct = follower1D.capture_time.values.astype("datetime64[ns]")
    raw = follower1D.capture_id.values.astype(np.int64)
    days = np.unique(
        np.concatenate(
            [
                pd.to_datetime(ct - np.timedelta64(10, "s")).strftime("%Y%m%d"),
                pd.to_datetime(ct + np.timedelta64(10, "s")).strftime("%Y%m%d"),
            ]
        )
    )
    linearIds = bool(raw.max() > P)
    newId = np.full(len(raw), -1, dtype=np.int64)
    ghostDrop = np.zeros(len(raw), dtype=bool)
    ctNs = ct.astype(np.int64)
    shiftNs = np.zeros(len(raw))
    noClock = np.zeros(len(raw), dtype=bool)
    for day in days:
        mapping = createMosaicFrameMapping(
            day,
            config,
            skipExisting=True,
            writeNc=False,
        )
        if (mapping is None) or ("segment" not in mapping.dims):
            continue
        for ss in range(len(mapping.segment)):
            seg = mapping.isel(segment=ss)
            if not bool(seg.resolved):
                continue
            m = (ct >= seg.fct_start.values) & (ct <= seg.fct_end.values) & (newId < 0)
            if m.sum() == 0:
                continue
            nBefore = np.zeros(m.sum(), dtype=np.int64)
            if "ghost" in mapping.dims:
                g = mapping.isel(ghost=(mapping.ghost_fsegment == seg.fsegment).values)
                for gg in range(len(g.ghost)):
                    gStart = g.ghost_ct_start.values[gg]
                    gEnd = g.ghost_ct_end.values[gg]
                    ghostDrop[m] |= (ct[m] > gStart) & (ct[m] < gEnd)
                    nBefore += (ct[m] >= gEnd) * int(g.ghost_extra.values[gg])
            if linearIds:
                # capture_id does not wrap (32 bit counter, e.g. gochang_v1):
                # the lag is the plain difference of the ids
                newId[m] = raw[m] - nBefore - int(seg.lag)
            else:
                newId[m] = ((raw[m] - nBefore - int(seg.lag_mod) - 1) % P) + 1

            # follower capture_time -> leader clock
            kk = (
                (mapping.knot_segment.values == ss)
                if "knot_time" in mapping
                else np.zeros(0, dtype=bool)
            )
            if kk.any():
                kt = (
                    mapping.knot_time.values[kk]
                    .astype("datetime64[ns]")
                    .astype(np.int64)
                )
                ko = mapping.knot_offset.values[kk] * 1e9
                shiftNs[m] = np.interp(ctNs[m], kt, ko)
            else:
                noClock[m] = True

    keep = (newId > 0) & ~ghostDrop & ~noClock
    log.info(
        f"mosaicFollowerCaptureIdToLeader: kept {keep.sum()} of {len(keep)} "
        f"follower particles ({ghostDrop.sum()} in ghost frames, "
        f"{(newId < 0).sum()} without reliable mapping)"
    )
    if keep.sum() == 0:
        return None
    follower1D = follower1D.isel({dim: keep})
    follower1D["capture_id"] = xr.DataArray(
        newId[keep].astype(follower1D.capture_id.dtype), dims=[dim]
    )
    # the follower's own clock is needed to look up its events (blocking,
    # ptpStatus); removed again in matchParticles
    follower1D["capture_time_followerClock"] = xr.DataArray(
        ctNs[keep].astype(np.int64).astype("datetime64[ns]"), dims=[dim]
    )
    follower1D["capture_time"] = xr.DataArray(
        (ctNs[keep] - shiftNs[keep]).astype(np.int64).astype("datetime64[ns]"),
        dims=[dim],
    )
    return follower1D


def mosaicCaptureIdOffset(leader1D, follower1D, dim="fpid"):
    """
    Offset between follower and leader capture_id after
    mosaicFollowerCaptureIdToLeader and captureIdOverflows. Both cameras then
    use the same ids, but captureIdOverflows unwraps each data set relative
    to its own start, so the offset is a multiple of 65535, resolved with
    record_time (good to a few seconds, i.e. << 65535 frames).
    """
    P = MOSAIC_CAPTURE_ID_PERIOD

    def medianDiff(timeName, maxDtS):
        lT = leader1D[timeName].values.astype("datetime64[ns]").astype(np.int64)
        fT = follower1D[timeName].values.astype("datetime64[ns]").astype(np.int64)
        order = np.argsort(fT)
        fT = fT[order]
        fId = follower1D.capture_id.values.astype(np.int64)[order]
        jj = np.clip(np.searchsorted(fT, lT), 0, len(fT) - 1)
        close = np.abs(fT[jj] - lT) < maxDtS * 1e9
        if close.sum() < 50:
            return None
        diff = fId[jj] - leader1D.capture_id.values.astype(np.int64)
        return np.median(diff[close])

    # capture_time of the follower has been moved to the leader clock
    # (mosaicFollowerCaptureIdToLeader), so it is good to milliseconds; the
    # record_time is only good to seconds, and it is not even reliable after
    # a follower restart (20191201 10:51: wrong multiple of 65535, i.e. not a
    # single match for 4 minutes of heavy snowfall)
    med = medianDiff("capture_time", 0.05)
    if med is None:
        med = medianDiff("record_time", 20.0)
    if med is None:
        return 0
    return int(np.round(med / P)) * P


def mosaicUnmappedTimes(case, config, timeIndex, maxUnmappedS=10):
    """
    Flag time bins in which follower frames could not be mapped to leader
    frames (no resolved metaFrameMapping segment, e.g. around follower
    restarts or for unresolved segments). Follower particles of such periods
    are dropped by mosaicFollowerCaptureIdToLeader, so level2match/track
    would otherwise silently report too few particles.

    Parameters
    ----------
    case : str
        Day YYYYMMDD
    config : dict
        Settings
    timeIndex : pandas.DatetimeIndex
        Start times of the level2 time bins (with freq)
    maxUnmappedS : float
        Bins with more than this many seconds not covered by a resolved
        segment are flagged

    Returns
    -------
    xarray.DataArray of bool with dimension time
    """
    import os

    import pandas as pd

    freqS = timeIndex.freq.nanos * 1e-9
    starts = []
    ends = []
    day = pd.Timestamp(case[:8])
    for dd in [day - pd.Timedelta("1D"), day, day + pd.Timedelta("1D")]:
        fname = mosaicFrameMappingFname(dd.strftime("%Y%m%d"), config)
        if not os.path.isfile(fname):
            continue
        with xr.open_dataset(fname) as ds:
            if "segment" not in ds.dims:
                continue
            ok = ds.resolved.values.astype(bool)
            starts.append(ds.fct_start.values[ok])
            ends.append(ds.fct_end.values[ok])
    if len(starts) > 0:
        starts = np.concatenate(starts)
        ends = np.concatenate(ends)
    binStart = timeIndex.values.astype("datetime64[ns]")
    binEnd = binStart + np.timedelta64(int(freqS * 1e9), "ns")
    covered = np.zeros(len(binStart))
    for s, e in zip(starts, ends):
        overlap = (np.minimum(binEnd, e) - np.maximum(binStart, s)) / np.timedelta64(
            1, "s"
        )
        covered += np.clip(overlap, 0, None)
    unmapped = (freqS - covered) > maxUnmappedS
    log.info(f"mosaicUnmappedTimes: {unmapped.sum()} of {len(unmapped)} bins flagged")
    return xr.DataArray(unmapped, dims=["time"], coords=[timeIndex])


def _rotationOutlierRuns(metaRotation, ofzTol, angleTol, maxRun):
    """
    Find runs of up to maxRun consecutive files whose rotation differs from
    the last good file by more than the tolerances, followed by a file that
    agrees with that last good file again.

    Returns
    -------
    list of tuple
        (index of last good file, index of first outlier, index of first
        file that is good again)
    """
    tol = {"camera_Ofz": ofzTol, "camera_phi": angleTol, "camera_theta": angleTol}
    vals = {
        k: metaRotation[k].sel(camera_rotation="mean").values.astype(float) for k in tol
    }
    n = len(metaRotation.file_starttime)
    if n < 3:
        return []

    def far(i, j):
        return np.any([abs(vals[k][i] - vals[k][j]) > tol[k] for k in tol])

    def valid(i):
        return not np.any([np.isnan(vals[k][i]) for k in tol])

    runs = []
    i = 1
    while i < n - 1:
        if valid(i - 1) and valid(i) and far(i, i - 1):
            for j in range(i + 1, min(i + maxRun, n - 1) + 1):
                if not valid(j):
                    break
                if not far(j, i - 1):
                    runs.append((i - 1, i, j))
                    i = j
                    break
        i += 1
    return runs


def _interpolateRotation(metaRotation, before, first, after, only=None):
    """replace files first..after-1 by linear interpolation between files
    before and after (only the indices in `only`, if given)"""
    for i in range(first, after):
        if (only is not None) and (i not in only):
            continue
        w = (i - before) / (after - before)
        for k in ["camera_Ofz", "camera_phi", "camera_theta"]:
            for c in [0, 1]:
                metaRotation[k].values[i, c] = (1 - w) * metaRotation[k].values[
                    before, c
                ] + w * metaRotation[k].values[after, c]


def despikeRotation(metaRotation, ofzTol=8.0, angleTol=0.4, maxRun=3):
    """
    Replace short runs of implausible per-file rotation estimates.

    A retrieval can lock onto false pairs (typically during dense snowfall)
    and return an implausible rotation with a small formal uncertainty (e.g.
    MOSAiC 20200203 15:55: Ofz -15 px instead of -38, phi -1.9 instead of
    -0.97), and the following files, which start from this value, can need a
    file or two to relax (Hyytiala 20241206 04:50-05:00: Ofz -61 and -30
    instead of -5). A run of up to maxRun files that all differ from the
    last good file by more than the tolerances, followed by a file that
    agrees with that last good file again, is replaced by a linear
    interpolation between the two. Persistent changes (e.g. camera moved)
    are kept because they do not return to the old values.

    Without further evidence this also replaces correct estimates (about
    every second candidate with the default tolerances, since the rotation
    parameters are partly degenerate), so createMetaRotation only uses it
    with large tolerances. For existing data use
    despikeRotationWithMatches, which checks every candidate against the
    pairs of its level1match file.

    Parameters
    ----------
    metaRotation : xarray.Dataset
        Rotation estimates with dimension file_starttime and variables
        camera_Ofz, camera_phi, camera_theta (dimension camera_rotation
        with mean and err)
    ofzTol, angleTol : float
        Tolerances for camera_Ofz (px) and camera_phi/camera_theta (deg)
    maxRun : int
        Maximum number of consecutive files to replace

    Returns
    -------
    xarray.Dataset, int
        Corrected dataset and number of replaced files
    """
    runs = _rotationOutlierRuns(metaRotation, ofzTol, angleTol, maxRun)
    if len(runs) == 0:
        return metaRotation, 0
    metaRotation = metaRotation.copy(deep=True)
    nReplaced = 0
    for before, first, after in runs:
        _interpolateRotation(metaRotation, before, first, after)
        nReplaced += after - first
        log.warning(
            "despikeRotation: replaced rotation of "
            f"{metaRotation.file_starttime.values[first]} to "
            f"{metaRotation.file_starttime.values[after - 1]} by interpolation"
        )
    return metaRotation, nReplaced


def zResidualBias(matchedDat, rotate, config):
    """
    Median and standard deviation (px) of the per-pair Z residual
    L_z - L_z_estimated of matched pairs for a given rotation (recomputed,
    not taken from the file).
    """
    import copy

    from . import matching

    matchedDat = matchedDat.drop_vars("position3D_center", errors="ignore")
    config = copy.deepcopy(config)
    cams = [str(c) for c in matchedDat.camera.values]
    config["leader"], config["follower"] = cams[0], cams[1]
    L_x, L_z, F_y, F_z = matching.get3DPosition(
        matchedDat.sel(camera=cams[0]), matchedDat.sel(camera=cams[1]), config
    )
    diffZ = np.asarray(L_z) - np.asarray(
        matching.calc_L_z_withOffsets(L_x, F_y, F_z, **dict(rotate))
    )
    return float(np.nanmedian(diffZ)), float(np.nanstd(diffZ))


def level1matchFnameFromMetaRotation(fnameMetaRotation, file_starttime):
    """name of the level1match file of the file starting at file_starttime
    that belongs to a metaRotation file"""
    import os

    root = fnameMetaRotation.split("/metaRotation/")[0]
    base = os.path.basename(fnameMetaRotation)
    ts = (
        str(np.datetime64(file_starttime, "s"))
        .replace("-", "")
        .replace("T", "-")
        .replace(":", "")
    )
    fname = base.replace("metaRotation", "level1match").replace(
        "_" + base.split("_")[-1], f"_{ts}.nc"
    )
    return f"{root}/level1match/{ts[:4]}/{ts[4:6]}/{ts[6:8]}/{fname}"


def despikeRotationWithMatches(
    metaRotation,
    fnameMetaRotation,
    config,
    ofzTol=8.0,
    angleTol=0.4,
    maxRun=12,
    minPairs=50,
    minBias=2.0,
    improvement=0.7,
):
    """
    Replace implausible per-file rotation estimates of existing data, but
    only where the replacement is supported by the data.

    Candidates are found as in despikeRotation (default: up to 12 files).
    For each candidate file, the pairs of its level1match file are
    evaluated with the original and with the interpolated rotation, and the
    file is only replaced if the original leaves a median Z residual of at
    least minBias pixels and the interpolated rotation reduces it clearly
    (the original rotation is favoured by construction, because it was used
    to find the pairs). Candidates without level1match
    file (or fewer than minPairs pairs) are only replaced if the geometric
    deviation is large (more than 3 times the tolerances) or if they belong
    to a run in which at least one file was replaced on the evidence of its
    pairs (a bad value carried over as prior over many files, mostly during
    periods without matched data). A file with pairs that fell short of the
    criterion is taken along only if its rotation is identical to that of a
    replaced file.

    Parameters
    ----------
    metaRotation : xarray.Dataset
        Rotation estimates (see despikeRotation)
    fnameMetaRotation : str
        Name of the metaRotation file (to find the level1match files)
    config : dict
        Settings of the deployment (frame size is needed)
    ofzTol, angleTol, maxRun : float
        See despikeRotation
    minPairs : int
        Minimum number of pairs for the evidence check
    minBias : float
        The original rotation must leave a median Z residual of at least
        this many pixels (matching tolerates ~1.7 px) to be replaced
    improvement : float
        The replacement must reduce the absolute median Z residual to below
        this fraction of the original one

    Returns
    -------
    xarray.Dataset, list
        Corrected dataset and list of dicts describing every replaced file
        (file_starttime, old and new values, bias of the pairs with the old
        and new rotation)
    """
    import os

    keys = ["camera_Ofz", "camera_phi", "camera_theta"]
    runs = _rotationOutlierRuns(metaRotation, ofzTol, angleTol, maxRun)
    if len(runs) == 0:
        return metaRotation, []
    new = metaRotation.copy(deep=True)
    replaced = []
    for before, first, after in runs:
        trial = metaRotation.copy(deep=True)
        _interpolateRotation(trial, before, first, after)
        accept = set()
        withEvidence = set()
        runTimes = {str(t) for t in metaRotation.file_starttime.values[first:after]}
        for i in range(first, after):
            t = metaRotation.file_starttime.values[i]
            old = {
                k: float(metaRotation[k].sel(camera_rotation="mean")[i]) for k in keys
            }
            interp = {k: float(trial[k].sel(camera_rotation="mean")[i]) for k in keys}
            fname = level1matchFnameFromMetaRotation(fnameMetaRotation, t)
            info = dict(
                file_starttime=str(t),
                old=old,
                new=interp,
                biasOld=np.nan,
                biasNew=np.nan,
                nPairs=0,
                evidence=False,
            )
            try:
                if os.path.isfile(fname):
                    with xr.open_dataset(fname) as ds:
                        m = ds.load()
                    if m.sizes["pair_id"] >= minPairs:
                        info["nPairs"] = int(m.sizes["pair_id"])
                        info["biasOld"], _ = zResidualBias(m, old, config)
                        info["biasNew"], _ = zResidualBias(m, interp, config)
                        info["evidence"] = True
                        withEvidence.add(i)
            except Exception as e:
                log.warning(f"despikeRotationWithMatches: {fname}: {e}")
            if info["evidence"]:
                ok = (abs(info["biasOld"]) >= minBias) and (
                    abs(info["biasNew"]) < improvement * abs(info["biasOld"])
                )
            else:
                ok = any(
                    abs(old[k] - interp[k]) > 3 * tol
                    for k, tol in zip(keys, [ofzTol, angleTol, angleTol])
                )
            if ok:
                accept.add(i)
                replaced.append(info)
        # files of the same run without usable pairs (typically a bad value
        # that was carried over as prior over many files, mostly during
        # periods with no matched data) follow the decision for the run
        if any(r["evidence"] for r in replaced if r["file_starttime"] in runTimes):
            acceptedOld = {
                tuple(round(r["old"][k], 4) for k in keys)
                for r in replaced
                if r["file_starttime"] in runTimes
            }
            for i in range(first, after):
                if i in accept:
                    continue
                same = (
                    tuple(
                        round(float(metaRotation[k].sel(camera_rotation="mean")[i]), 4)
                        for k in keys
                    )
                    in acceptedOld
                )
                # files with pairs that fell short of the criterion are only
                # taken along if their rotation is identical to an accepted
                # one (same carried-over value, same cause)
                if (i in withEvidence) and not same:
                    continue
                accept.add(i)
                replaced.append(
                    dict(
                        file_starttime=str(metaRotation.file_starttime.values[i]),
                        old={
                            k: float(metaRotation[k].sel(camera_rotation="mean")[i])
                            for k in keys
                        },
                        new={
                            k: float(trial[k].sel(camera_rotation="mean")[i])
                            for k in keys
                        },
                        biasOld=np.nan,
                        biasNew=np.nan,
                        nPairs=0,
                        evidence=False,
                        followsRun=True,
                    )
                )
        _interpolateRotation(new, before, first, after, only=accept)
    return new, replaced


def medianRotation(rotations):
    """component-wise median of a list of rotation dicts/Series"""
    import pandas as pd

    return pd.DataFrame([dict(r) for r in rotations]).median()


def rejectRotationJump(
    rot,
    rotDefault,
    pending,
    ofzTol=8.0,
    angleTol=0.4,
    confirmN=6,
    confirmTol=(5.0, 0.3, 0.3),
):
    """
    Do not let a single implausible rotation retrieval become the new
    default (prior) of the following files.

    In dense snowfall the retrieval can diverge (MOSAiC 20200212 17:20: Ofz
    -94 px instead of -33, phi +7.8 deg instead of -1.0) while reporting a
    small formal error, and since every retrieval is the first guess of the
    next one, the garbage is carried on for hours or days (20200210 06:25 to
    20200211 19:10). The geometry of a deployment is stable for long periods and
    real changes are announced in the settings file (`rotate` seeds). A
    retrieval that is farther from the reference (rotDefault) than the
    tolerances is therefore rejected, unless confirmN consecutive retrievals
    agree with each other within confirmTol (then it is a real new level).
    The reference should be robust, e.g. the median of the last 12 accepted
    retrievals (medianRotation): 20200213 ended with a retrieval only ~10 px
    off that was adopted, carried for 11 hours and biased the matching.

    Parameters
    ----------
    rot, rotDefault : dict or pandas.Series
        current retrieval and current default with camera_Ofz,
        camera_phi, camera_theta
    pending : list
        rejected retrievals since the default was last updated
        (returned, pass it back in for the next file)

    Returns
    -------
    (rot or None, pending)
        the retrieval to accept, or None if it is rejected (use the
        default instead)
    """
    keys = ["camera_Ofz", "camera_phi", "camera_theta"]
    tol = [ofzTol, angleTol, angleTol]
    jump = any(abs(float(rot[k]) - float(rotDefault[k])) > t for k, t in zip(keys, tol))
    if not jump:
        return rot, []
    pending = list(pending) + [rot]
    if len(pending) >= confirmN:
        last = pending[-confirmN:]
        if all(
            max(float(r[k]) for r in last) - min(float(r[k]) for r in last) <= t
            for k, t in zip(keys, confirmTol)
        ):
            return rot, []
    log.warning(
        f"rejectRotationJump: retrieval {dict((k, round(float(rot[k]), 3)) for k in keys)} "
        f"is far from the current default {dict((k, round(float(rotDefault[k]), 3)) for k in keys)}, "
        "keeping the default"
    )
    return None, pending


def mosaicClockOffsetSeconds(case, config, tStart, tEnd, marginS=600):
    """
    Follower minus leader clock (s) around a leader time span, from the
    clock offset knots of the day's frame mapping. The cameras' clocks are
    seconds apart in MOSAiC, but were up to ~30 min apart in gochang_v1 when
    the camera time synchronisation failed (20251205-20251221), so the
    follower files and events must be looked up with that offset.
    Returns 0 if there is no mapping or no knot.
    """
    try:
        m = createMosaicFrameMapping(case, config, skipExisting=True, writeNc=False)
    except Exception:
        return 0.0
    if (m is None) or ("knot_time" not in m):
        return 0.0
    kt = m.knot_time.values.astype("datetime64[ns]").astype(np.int64)
    ko = m.knot_offset.values.astype(float)
    leaderT = kt - (ko * 1e9).astype(np.int64)
    t0 = np.datetime64(tStart, "ns").astype(np.int64) - int(marginS * 1e9)
    t1 = np.datetime64(tEnd, "ns").astype(np.int64) + int(marginS * 1e9)
    sel = (leaderT >= t0) & (leaderT <= t1)
    if sel.sum() == 0:
        return float(np.median(ko))
    return float(np.median(ko[sel]))


def mosaicFollowerFiles(
    ffl1,
    config,
    leaderStart,
    leaderEnd,
    level="level1detect",
    marginSeconds=5,
    offsetSeconds=0.0,
):
    """
    Files of the other camera that cover the time span of one file.

    MOSAiC files are not always newFileInt long: if a new-file event is
    skipped, one file covers 10 minutes. Looking only for files of the
    other camera that START within the nominal window of this file then
    misses (a) the other camera's long file that started before the window
    and (b) the other camera's files in the second half of this file when it
    is long itself (20191201 01:45-01:55: no 01:50 file, the follower
    files at 01:50 were never read and nothing matched in the second half).
    A file of the other camera covers the time from its start to the start
    of the next file.

    Parameters
    ----------
    ffl1 : files.FilenamesFromLevel
        the file (of one camera) to find the other camera's files for
    leaderStart, leaderEnd : numpy.datetime64
        time span of this file (capture_time, i.e. of its own clock)
    level : str
        level of the files to return, e.g. level1detect or level0txt
    marginSeconds : float
        safety margin for the offset between the clocks of the cameras
    offsetSeconds : float
        follower minus leader clock, see mosaicClockOffsetSeconds

    Returns
    -------
    list of str
        file names (as filenamesOtherCamera, may include sentinels)
    """
    import pandas as pd

    margin = np.timedelta64(int(marginSeconds), "s")
    shift = np.timedelta64(int(round(offsetSeconds * 1e3)), "ms")
    leaderStart = leaderStart + shift
    leaderEnd = leaderEnd + shift
    cand = ffl1.filenamesOtherCamera(
        graceInterval=1200 + int(abs(offsetSeconds)), level=level
    )
    if len(cand) == 0:
        return []
    if (config.nThreads is not None) and level in ["level0", "level0txt"]:
        ts = [f.split("_")[-2] for f in cand]
    else:
        ts = [f.split("_")[-1].split(".")[0] for f in cand]
    try:
        starts = pd.to_datetime(ts, format="%Y%m%d-%H%M%S").values
    except ValueError:
        return list(cand)
    order = np.argsort(starts)
    starts = starts[order]
    cand = [cand[i] for i in order]
    # a sentinel (.nodata/.broken) and the real file of the same start must not
    # count as "next file"
    uniqueStarts = np.unique(starts)
    nextStart = {}
    for a, b in zip(uniqueStarts[:-1], uniqueStarts[1:]):
        nextStart[a] = b
    keep = []
    for f, s in zip(cand, starts):
        end = nextStart.get(s, s + np.timedelta64(int(config.newFileInt) + 300, "s"))
        if (s <= leaderEnd + margin) and (end > leaderStart - margin):
            keep.append(f)
    return keep
