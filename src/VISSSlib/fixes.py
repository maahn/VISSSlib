# -*- coding: utf-8 -*-


import logging
import warnings

# import matplotlib.pyplot as plt
from copy import deepcopy
from itertools import groupby

import numpy as np
import xarray as xr

log = logging.getLogger(__name__)


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


# def revertMakeCaptureTimeEven(dat):
#     dat = dat.rename({"capture_time": "capture_time_even"})
#     dat = dat.rename({"capture_time_orig": "capture_time"})
#     return dat
