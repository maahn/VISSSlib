# -*- coding: utf-8 -*-

import datetime
import io
import json
import logging
import multiprocessing
import os
import shutil
import socket
import struct
import subprocess
import tarfile
import time
import warnings
import zipfile
import zlib
from copy import deepcopy
from functools import wraps

import numba
import numpy as np
import taskqueue
import xarray as xr
from addict import Dict

from . import __version__, __versionFull__, files, fixes

log = logging.getLogger(__name__)


# settings that stay mostly constant
DEFAULT_SETTINGS = {
    "correctForSmallOnes": False,
    "height_offset": 64,
    "minMovingPixels": [20, 10, 5, 2, 2, 2, 2],
    "threshs": [20, 30, 40, 60, 80, 100, 120],
    "goodFiles": ["None", "None"],
    "level1detectQuicklook": {
        "minBlur": 500,
        "minSize": 10,
        "omitLabel4small": True,
    },
    "rotate": {},
    "fileMode": 0o664,  # 436
    "dirMode": 0o775,  # 509
    "level1detect": {
        "maxMovingObjects": 1000,  # 60 until 18.9.24
        "minAspectRatio": None,  # testing only
        "minBlur4picturewrite": 250,
        "minSize4picturewrite": 8,
        "trainingSize": 100,
        "minContrast": 20,
        "minDmax": 0,
        "minBlur": 10,
        "minArea": 0,
        "erosionTestThreshold": 0.06,
        "height_offset": 64,
        "maskCorners": None,
        "cropImage": None,  # (offsetX, offsetY)
        "backSubKW": {
            "dist2Threshold": 400,
            "detectShadows": False,
            "history": 100,
        },  #       dist2Threshold of 100 was extensively tested, but this makes small particles larger even though it helps with wings etc.
        #  this is compensated by the canny filter, but not for holes in the particles (the default is {"dist2Threshold": 400,
        #                               "detectShadows": False, "history": 100})
        "backSub": "cv2.createBackgroundSubtractorKNN",
        "applyCanny2Particle": True,  # canny filter gets edges better
        "dilateIterations": 1,  # to close gaps in canny edges (the default is 1 whic is sufficient)
        "blurSigma": 1,
        "minAspectRatio": None,
        "dilateErodeFgMask": False,  # turns out to be not so smart because it makes holes insides particles smaller
        "check4childCntLength": True,  # discard short child contours instead of dilate/erose
        "doubleDynamicRange": True,
        "dilateFgMask4Contours": True,
        "testMovieFile": True,
        "writeImg": True,
    },
    "level1match": {
        "maxMovingObjects": 300,  # 60 until 18.9.24
    },
    "level1track": {
        "maxMovingObjects": 300,  # 60 until 18.9.24
    },
    "level1shape": {},
    "level2": {
        "freq": "1min",
    },
    "quality": {
        "obsRatioThreshold": 0.7,
        "blowingSnowFrameThresh": 0.05,
        "blockedPixThresh": 0.1,
        "minMatchScore": 1e-3,
        "minSize4insituM": 10,
        "trackLengthThreshold": 2,
    },
    "matchData": True,
    "processL2detect": True,
    "aux": {
        "meteo": {
            # "source": "cloudnetMeteo",
            "downloadData": True,
        },
        "radar": {
            # "source": "cloudnetCategorize",
            "elevation": 90,
            "downloadData": True,
            "heightRange": (120, 360),
            "minHeightBins": 4,
            "timeOffset": 120,
            "calibrationOffset": {
                "2000-01-01": 0,  # values that need to be added to radar Z. always previos value is used
            },
        },
        "cloudnet": {},
        "arm": {},
        "pangaea": {},
    },
    "level3": {
        "combinedRiming": {
            "processRetrieval": False,
            "habit": "mean",  # SSRG particle habit
            "Zvar": "Ze_ground",  # extrapolated to surface using aux.radar.heightIndices
            "maxTemp": 275.15,  # +2°C
            "minZe": -10,
            "minNParticles": 100,
            "extraFileStr": "",
        }
    },
}


niceNames = (
    ("master", "leader"),
    ("trigger", "leader"),
    ("slave", "follower"),
)


def loopify_with_camera(func):
    """
    Decorator to make function loop over cases and cameras.

    Parameters
    ----------
    func : callable
        Function to be decorated.

    Returns
    -------
    callable
        Wrapped function that loops over cameras.
    """
    @wraps(func)
    def myinner(case, camera, config, *args, **kwargs):
        config = readSettings(config)
        if camera == "all":
            cameras = [config.leader, config.follower]
        elif camera == "leader":
            cameras = [config.leader]
        elif camera == "follower":
            cameras = [config.follower]
        else:
            cameras = [camera]

        cases = getCaseRange(case, config)

        returns = list()
        for case1 in cases:
            for camera1 in cameras:
                returns.append(func(case1, camera1, config, *args, **kwargs))
        if len(returns) == 1:
            return returns[0]
        else:
            return returns

    return myinner


def loopify(func):
    """
    Decorator to make function loop over cases.

    Parameters
    ----------
    func : callable
        Function to be decorated.

    Returns
    -------
    callable
        Wrapped function that loops over cases.
    """
    @wraps(func)
    def myinner(case, config, *args, **kwargs):
        config = readSettings(config)
        cases = getCaseRange(case, config)

        returns = list()
        for case1 in cases:
            returns.append(func(case1, config, *args, **kwargs))
        if len(returns) == 1:
            return returns[0]
        else:
            return returns

    return myinner


class DictNoDefault(Dict):
    """
    Dictionary class that raises KeyError when accessing missing keys.

    Inherits from addict.Dict.
    """
    def __missing__(self, key):
        raise KeyError(key)


def nicerNames(string):
    """
    Replace VISSS names with up-to-date equivalents.

    Parameters
    ----------
    string : str
        Input string to process.

    Returns
    -------
    str
        String with replaced names.
    """
    for i in range(len(niceNames)):
        string = string.replace(*niceNames[i])
    return string


def readSettings(fname):
    """
    Read configuration settings from a YAML file. Default is taken from DEFAULT_SETTINGS

    Parameters
    ----------
    fname : str or dict
        Path to YAML file or already parsed config dict.

    Returns
    -------
    addict.Dict
        Configuration settings.
    """
    import flatten_dict
    import yaml

    if type(fname) is str:
        # we have to flatten the dictionary so that update works
        config = flatten_dict.flatten(DEFAULT_SETTINGS)
        with open(fname, "r") as stream:
            loadedSettings = flatten_dict.flatten(yaml.load(stream, Loader=yaml.Loader))
            config.update(loadedSettings)
        # unflatten again and convert to addict.Dict
        config = DictNoDefault(flatten_dict.unflatten(config))
        config["filename"] = fname
        return config
    else:  # is already config
        return fname


def getCaseRange(nDays, config, endYesterday=True):
    """
    Get list of case identifiers from a date range.

    Parameters
    ----------
    nDays : int, str
        Number of days going back or date string "YYYYMMDD" or "YYYYMMDD-YYYYMMDD"
        or "YYYYMMDD,YYYYMMDD,YYYYMMDD".
    config : dict
        Configuration settings.
    endYesterday : bool, optional
        Whether to end yesterday, by default True.

    Returns
    -------
    list of str
        List of case identifiers.
    """
    # shortcut to detect timestamps of YYYYMMDD-HH or YYYYMMDD-HHMMSS
    if (nDays[-3] == "-") or (nDays[-7] == "-"):
        return [nDays]
    days = getDateRange(nDays, config, endYesterday=endYesterday)
    cases = []
    for dd in days:
        year = str(dd.year)
        month = "%02i" % dd.month
        day = "%02i" % dd.day
        case = f"{year}{month}{day}"
        cases.append(case)
    return cases


def getDateRange(nDays, config, endYesterday=True):
    """
    Generate date range based on input parameters.

    Parameters
    ----------
    nDays : int, str
        Number of days going back or date string "YYYYMMDD" or "YYYYMMDD-YYYYMMDD"
        or "YYYYMMDD,YYYYMMDD,YYYYMMDD".
    config : dict
        Configuration settings.
    endYesterday : bool, optional
        Whether to end yesterday, by default True.

    Returns
    -------
    pandas.DatetimeIndex
        Date range.
    """
    import pandas as pd

    if type(nDays) is np.str_:
        nDays = str(nDays)

    config = readSettings(config)
    if config["end"] == "today":
        end = datetime.datetime.utcnow()
        if endYesterday:
            end2 = datetime.datetime.utcnow() - datetime.timedelta(days=1)
        else:
            end2 = datetime.datetime.utcnow() - datetime.timedelta(hours=1)
    else:
        end = end2 = config["end"]

    if (type(nDays) is int) or type(nDays) is float:
        if nDays > 1000:
            nDays = str(nDays)
    elif (type(nDays) is str) and (len(nDays) < 6):
        nDays = int(nDays)

    if nDays == 0:
        days = pd.date_range(
            start=config["start"],
            end=end2,
            freq="1D",
            tz=None,
            normalize=True,
            name=None,
            inclusive="both",
        )

    elif type(nDays) is str:
        days = nDays.split(",")
        if len(days) == 1:
            days = nDays.split("-")
            if len(days) == 1:
                days = pd.date_range(
                    start=nDays,
                    periods=1,
                    freq="1D",
                    tz=None,
                    normalize=True,
                    name=None,
                    inclusive="both",
                )
            else:
                days = pd.date_range(
                    start=days[0],
                    end=days[1],
                    freq="1D",
                    tz=None,
                    normalize=True,
                    name=None,
                    inclusive="both",
                )
        else:
            days = pd.DatetimeIndex(days)

    else:
        days = pd.date_range(
            end=end2,
            periods=nDays,
            freq="1D",
            tz=None,
            normalize=True,
            name=None,
            inclusive="both",
        )

    # double check to make sure we did not add too much
    if np.any(days < pd.Timestamp(config.start)):
        log.warning(
            f"Date range {nDays} includes cases that are before the specified start {config.start}"
        )
        days = days[days >= pd.Timestamp(config.start)]

    if config.end == "today":
        end = datetime.datetime.utcnow()
    else:
        end = config.end

    if np.any(days > pd.Timestamp(end)):
        log.warning(
            f"Date range {nDays} includes cases that are after the specified end {end}"
        )
        days = days[days <= pd.Timestamp(config.end)]

    return days


def getPreviousKey(thisDict, case):
    """
    Get the previous key-value pair in a dictionary based on timestamp.

    Parameters
    ----------
    thisDict : dict
        Dictionary with datetime keys.
    case : str
        Case identifier.

    Returns
    -------
    tuple
        Previous value and timestamp, or None if not found.
    """
    thisDict1 = {}
    for k, v in thisDict.items():
        thisDict1[np.datetime64(k)] = v

    dates = np.array(list(thisDict1.keys()))

    # 1. Filter for dates before the case
    previous_dates = dates[dates < case2timestamp(case)]

    # 2. Get the maximum of the remaining dates
    result = np.max(previous_dates) if previous_dates.size > 0 else None

    if result is None:
        return None
    else:
        return thisDict1[result], result


def getPreviousCalibrationOffset(case, config):
    """
    Get the previous calibration offset for a given case.

    Parameters
    ----------
    case : str
        Case identifier.
    config : dict
        Configuration settings.

    Returns
    -------
    tuple
        Calibration offset and timestamp, or None if not found.
    """
    return getPreviousKey(config.aux.radar.calibrationOffset, case)



def getMode(a):
    """
    Compute the mode of an array.

    Parameters
    ----------
    a : array_like
        Input array.

    Returns
    -------
    ndarray
        Mode of the array.
    """
    (_, idx, counts) = np.unique(a, return_index=True, return_counts=True)
    index = idx[np.argmax(counts)]
    mode = a[index]
    return a


def open_mfmetaFrames(fnames, config, start=None, end=None, skipFixes=[]):
    """
    Open multiple metaFrame files.

    Parameters
    ----------
    fnames : list of str
        File names.
    config : dict
        Configuration settings.
    start : datetime, optional
        Start time, by default None.
    end : datetime, optional
        End time, by default None.
    skipFixes : list of str, optional
        Fixes to skip, by default [].

    Returns
    -------
    xarray.Dataset
        Combined dataset.
    """

    def preprocess(dat):
        # keep track of file start time
        fname = dat.encoding["source"]
        ffl1 = files.FilenamesFromLevel(fname, config)
        dat["file_starttime"] = xr.DataArray(
            [ffl1.datetime64] * len(dat.capture_time), coords=[dat.capture_time]
        )

        return dat

    dat = xr.open_mfdataset(
        fnames, combine="nested", concat_dim="capture_time", preprocess=preprocess
    ).load()

    if skipFixes != "all":
        # fix potential integer overflows if necessary
        if ("captureIdOverflows" in config.dataFixes) and (
            "captureIdOverflows" not in skipFixes
        ):
            print("apply fix", "captureIdOverflows")
            dat = fixes.captureIdOverflows(dat, config, dim="capture_time")

        # # beware of unintended implications down the processing chain becuase capture_time is used
        # # in analysis module
        # # fix potential integer overflows if necessary
        # if ("makeCaptureTimeEven" in config.dataFixes) and ("makeCaptureTimeEven" not in skipFixes):
        #     print("apply fix", "makeCaptureTimeEven")
        #     #does not make sense for leader
        #     if "follower" in dat.encoding["source"]:
        #         dat = fixes.makeCaptureTimeEven(dat, config, dim="capture_time")
        #     else:
        #         #make sure follower and leader data are consistent
        #         dat["capture_time_orig"] = dat["capture_time"]

    if start is not None:
        dat = dat.isel(capture_time=(dat.capture_time >= start))
        if len(dat.capture_time) == 0:
            return None

    if end is not None:
        dat = dat.isel(capture_time=(dat.capture_time <= end))
        if len(dat.capture_time) == 0:
            return None

    if len(dat.capture_time) == 0:
        return None

    return dat


def open_mflevel1detect(
    fnamesExt, config, start=None, end=None, skipFixes=[], datVars="all"
):
    """
    Open multiple level1detect files.

    Parameters
    ----------
    fnamesExt : list of str
        File names.
    config : dict
        Configuration settings.
    start : datetime, optional
        Start time, by default None.
    end : datetime, optional
        End time, by default None.
    skipFixes : list of str, optional
        Fixes to skip, by default [].
    datVars : str or list of str, optional
        Variables to include, by default "all".

    Returns
    -------
    xarray.Dataset
        Combined dataset.
    """

    def preprocess(dat):
        # keep trqack of file start time
        fname = dat.encoding["source"]
        # print("open_mflevel1detect",fname)
        ffl1 = files.FilenamesFromLevel(fname, config)
        dat["file_starttime"] = xr.DataArray(
            [ffl1.datetime64] * len(dat.pid), coords=[dat.pid]
        )
        if "nThread" not in dat.keys():
            dat["nThread"] = xr.DataArray([-99] * len(dat.pid), coords=[dat.pid])

        if datVars != "all":
            dat = dat[datVars]
        return dat

    if type(fnamesExt) is not list:
        fnamesExt = [fnamesExt]

    fnames = []
    for fname in fnamesExt:
        if fname.endswith("nodata"):
            pass
        elif fname.endswith("broken.txt"):
            pass
        elif fname.endswith("notenoughframes"):
            pass
        else:
            fnames.append(fname)
    if len(fnames) == 0:
        return None

    dat = xr.open_mfdataset(
        fnames, combine="nested", concat_dim="pid", preprocess=preprocess
    ).load()

    if start is not None:
        dat = dat.isel(pid=(dat.capture_time >= start))
        if len(dat.pid) == 0:
            return None

    if end is not None:
        dat = dat.isel(pid=(dat.capture_time <= end))
        if len(dat.pid) == 0:
            return None

    if skipFixes != "all":
        # fix potential integer overflows if necessary
        if ("captureIdOverflows" in config.dataFixes) and (
            "captureIdOverflows" not in skipFixes
        ):
            dat = fixes.captureIdOverflows(dat, config)

        # moved to meta data
        # # fix potential integer overflows if necessary
        # if( "makeCaptureTimeEven" in config.dataFixes) and ( "makeCaptureTimeEven" not in skipFixes):

        #     #does not make sense for leader
        #     if "follower" in dat.encoding["source"]:
        #         dat = fixes.makeCaptureTimeEven(dat, config, dim="pid")
        #     else:
        #         #make sure follower and leader data are consistent
        #         dat["capture_time_orig"] = dat["capture_time"]

    # replace pid by empty dimesnion to allow concatenating files without jumps in dimension pid
    dat = dat.swap_dims({"pid": "fpid"})

    if len(dat.fpid) == 0:
        return None

    if len(dat.fpid) == 0:
        return None

    dat.load()
    return dat


def open_mflevel1match(fnamesExt, config, datVars="all"):
    """
    Open multiple level1match files.

    Parameters
    ----------
    fnamesExt : list of str
        File names.
    config : dict
        Configuration settings.
    datVars : str or list of str, optional
        Variables to include, by default "all".

    Returns
    -------
    xarray.Dataset
        Combined dataset.
    """
    if type(fnamesExt) is not list:
        fnamesExt = [fnamesExt]

    fnames = []
    for fname in fnamesExt:
        if fname.endswith("nodata"):
            pass
        elif fname.endswith("broken.txt"):
            pass
        elif fname.endswith("notenoughframes"):
            pass
        else:
            fnames.append(fname)
    if len(fnames) == 0:
        return None

    def preprocess(dat):
        # keep trqack of file start time
        fname = dat.encoding["source"]
        ffl1 = files.FilenamesFromLevel(fname, config)
        dat["file_starttime"] = xr.DataArray(
            [ffl1.datetime64] * len(dat.pair_id), coords=[dat.pair_id]
        )
        if datVars != "all":
            dat = dat[datVars]
        return dat

    dat = xr.open_mfdataset(
        fnames, combine="nested", concat_dim="pair_id", preprocess=preprocess
    ).load()
    # replace pid by empty dimesnion to allow concatenating files without jumps in dimension pid
    dat = dat.swap_dims({"pair_id": "fpair_id"})

    return dat


def identifyBlockedBlowingSnowData(fnames, config, timeIndex1, sublevel):
    """
    Identify blocked blowing snow data.

    Parameters
    ----------
    fnames : list of str
        File names.
    config : dict
        Configuration settings.
    timeIndex1 : array_like
        Time bins.
    sublevel : str
        Sublevel identifier.

    Returns
    -------
    tuple of xarray.DataArray
        Blowing snow ratio and detected particles.
    """
    # handle blowing snow, estimate ratio of skipped frames

    # print("starting identifyBlowingSnowData", cam)
    movingObjects = []
    for fna in fnames:
        movingObjects.append(xr.open_dataset(fna).movingObjects)
    movingObjects = xr.concat(movingObjects, dim="capture_time")
    # movingObjects = xr.open_mfdataset(fnames,  combine='nested', preprocess=preprocess).movingObjects.load()
    movingObjects = movingObjects.sortby("capture_time")

    tooManyMove = movingObjects > config[f"level1{sublevel}"].maxMovingObjects
    tooManyMove = tooManyMove.groupby_bins(
        "capture_time", timeIndex1, labels=timeIndex1[:-1]
    )

    nFrames = tooManyMove.count()
    # does not make sense for small number of frames
    nFrames = nFrames.where(nFrames > 100)
    blowingSnowRatio = tooManyMove.sum() / nFrames  # now a ratio
    # nan means nothing recorded, so no blowing snow either
    blowingSnowRatio = blowingSnowRatio.fillna(0)
    blowingSnowRatio = blowingSnowRatio.rename(capture_time_bins="time")

    nDetected = (
        movingObjects.groupby_bins("capture_time", timeIndex1, labels=timeIndex1[:-1])
        .sum()
        .fillna(0)
    )
    nDetected = nDetected.rename(capture_time_bins="time")

    # print("done identifyBlowingSnowData")
    return blowingSnowRatio, nDetected


def compareNDetected(nDetectedL, nDetectedF):
    """
    Compare detected particles between two cameras.

    Parameters
    ----------
    nDetectedL : xarray.DataArray
        Detected particles for leader.
    nDetectedF : xarray.DataArray
        Detected particles for follower.

    Returns
    -------
    xarray.DataArray
        Ratio of minimum to maximum detections.
    """
    minParticles = 1000
    nDetected = xr.concat([nDetectedL, nDetectedF], dim="camera")
    ratio = nDetected.min("camera") / nDetected.max("camera")
    ratio = ratio.where((nDetectedL > minParticles) & (nDetectedF > minParticles))
    # ratio.values[(nDetectedL < minParticles) | (nDetectedF < minParticles)] = np.nan
    return ratio


def removeBlockedBlowingData(dat1D, events, config, threshold=0.1):
    """
    Remove blocked or blowing snow data.

    Parameters
    ----------
    dat1D : xarray.Dataset
        Input dataset.
    events : str or xarray.Dataset
        Events file or dataset.
    config : dict
        Configuration settings.
    threshold : float, optional
        Blocking threshold, by default 0.1 i.e. 10%

    Returns
    -------
    xarray.Dataset
        Filtered dataset or None if empty.
    """
    # shortcut
    if dat1D is None:
        return None

    ts, counts = np.unique(dat1D.capture_time, return_counts=True)
    if np.any(counts > config.level1match.maxMovingObjects):
        tsBlowingSnow = ts[counts > config.level1match.maxMovingObjects]
        dat1D = dat1D.isel(fpid=~dat1D.capture_time.isin(tsBlowingSnow))
    if len(dat1D.fpid) == 0:
        print("no data after removing blowing snow data")
        return None

    if type(events) is str:
        events = xr.open_dataset(events)
    blocked = events.blocking.sel(blockingThreshold=50) > threshold

    # interpolate blocking status to observed particles
    isBlocked = blocked.sel(file_starttime=dat1D.capture_time, method="nearest").values
    dat1D = dat1D.isel(fpid=(~isBlocked))
    events.close()

    if len(dat1D.fpid) > 0:
        print(f"{np.sum(isBlocked)/ len(dat1D.fpid)*100}% blocked data removed")

    if len(dat1D.fpid) == 0:
        print("no data after removing blocked data")
        return None
    else:
        return dat1D


def estimateCaptureIdDiff(
    leaderFile,
    followerFiles,
    config,
    dim,
    concat_dim="capture_time",
    nPoints=500,
    maxDiffMs=1,
):
    """
    Estimate capture ID difference between cameras.

    Parameters
    ----------
    leaderFile : str
        Leader camera file.
    followerFiles : list of str
        Follower camera files.
    config : dict
        Configuration settings.
    dim : str
        Dimension to use.
    concat_dim : str, optional
        Concatenation dimension, by default "capture_time".
    nPoints : int, optional
        Number of points to sample, by default 500.
    maxDiffMs : int, optional
        Maximum difference in milliseconds, by default 1.

    Returns
    -------
    tuple
        Estimated ID difference and number of samples.
    """
    leaderDat = xr.open_dataset(leaderFile)
    followerDat = xr.open_mfdataset(
        followerFiles, combine="nested", concat_dim=concat_dim
    ).load()

    followerDat = cutFollowerToLeader(
        leaderDat, followerDat, gracePeriod=0, dim=concat_dim
    )

    if "captureIdOverflows" in config.dataFixes:
        leaderDat = fixes.captureIdOverflows(
            leaderDat, config, storeOrig=True, idOffset=0, dim=dim
        )
        followerDat = fixes.captureIdOverflows(
            followerDat, config, storeOrig=True, idOffset=0, dim=dim
        )
        # fix potential integer overflows if necessary

    if "makeCaptureTimeEven" in config.dataFixes:
        # does not make sense for leader
        # redo capture_time based on first time stamp...
        # requires trust in capture_id
        followerDat = fixes.makeCaptureTimeEven(followerDat, config, dim)

    idDiff = estimateCaptureIdDiffCore(
        leaderDat, followerDat, dim, nPoints=nPoints, maxDiffMs=maxDiffMs
    )
    leaderDat.close()
    followerDat.close()

    return idDiff


def estimateCaptureIdDiffCore(
    leaderDat, followerDat, dim, nPoints=500, maxDiffMs=1, timeDim="capture_time"
):
    """
    Core function to estimate capture ID difference.

    Parameters
    ----------
    leaderDat : xarray.Dataset
        Leader data.
    followerDat : xarray.Dataset
        Follower data.
    dim : str
        Dimension to use.
    nPoints : int, optional
        Number of points to sample, by default 500.
    maxDiffMs : int, optional
        Maximum difference in milliseconds, by default 1.
    timeDim : str, optional
        Time dimension, by default "capture_time".

    Returns
    -------
    tuple
        Estimated ID difference and number of samples.

    Raises
    ------
    RuntimeError
        If capture ID varies too much.
    """
    if len(leaderDat[dim]) == 0:
        raise RuntimeError(f"leaderDat has zero length")
    if len(followerDat[dim]) == 0:
        raise RuntimeError(f"followerDat has zero length")

    # check whether correction has been applied and use if present
    if (timeDim == "capture_time") and ("capture_time_even" in followerDat.data_vars):
        timeDimFollower = "capture_time_even"
    else:
        timeDimFollower = timeDim

    # cut number of investigated points in time if required
    if len(leaderDat[dim]) > nPoints:
        points = np.linspace(0, len(leaderDat[dim]), nPoints, dtype=int, endpoint=False)
    else:
        points = range(len(leaderDat[dim]))

    # loop through all points
    idDiffs = []
    for point in points:
        absDiff = np.abs(
            leaderDat[timeDim].isel(**{dim: point}).values
            - followerDat[timeDimFollower]
        )
        pMin = np.min(absDiff).values
        if pMin < np.timedelta64(int(maxDiffMs), "ms"):
            pII = absDiff.argmin().values
            idDiff = (
                followerDat.capture_id.values[pII]
                - leaderDat.capture_id.isel(**{dim: point}).values
            )
            idDiffs.append(idDiff)

    nIdDiffs = len(idDiffs)
    print(f"using {nIdDiffs} of {len(points)}")

    if nIdDiffs > 0:
        (vals, idx, counts) = np.unique(idDiffs, return_index=True, return_counts=True)
        idDiff = vals[np.argmax(counts)]
        ratioSame = np.sum(idDiffs == idDiff) / nIdDiffs
        print(
            "estimateCaptureIdDiff statistic:",
            dict(zip(vals[np.argsort(counts)[::-1]], counts[np.argsort(counts)[::-1]])),
            timeDim,
        )

        if ratioSame > 0.7:
            print(
                f"capture_id determined {idDiff}, {ratioSame*100}% have the same value"
            )
            return idDiff, nIdDiffs
        else:
            raise RuntimeError(
                f"capture_id varies too much, only {ratioSame*100}% of {nIdDiffs} samples have the same value {idDiff}, 2n place: {vals[np.argsort(counts)[-2]]}"
            )

    else:
        print(f"nIdDiffs {nIdDiffs} is too short")
        return None, nIdDiffs


def otherCamera(camera, config):
    """
    Get the other VISSS camera based on configuration.

    Parameters
    ----------
    camera : str
        Camera identifier.
    config : dict
        Configuration settings.

    Returns
    -------
    str
        Other camera identifier.

    Raises
    ------
    ValueError
        If camera is not in instruments list.
    """
    if camera == config["instruments"][0]:
        return config["instruments"][1]
    elif camera == config["instruments"][1]:
        return config["instruments"][0]
    else:
        raise ValueError



def cutFollowerToLeader(leader, follower, gracePeriod=1, dim="fpid"):
    """
    Cut follower data to match leader data with respect to time.

    Parameters
    ----------
    leader : xarray.Dataset
        Leader data.
    follower : xarray.Dataset
        Follower data.
    gracePeriod : int, optional
        Grace period in seconds, by default 1.
    dim : str, optional
        Dimension to use, by default "fpid".

    Returns
    -------
    xarray.Dataset
        Cut follower data.
    """
    start = leader.capture_time[0].values - np.timedelta64(
        int(gracePeriod * 1000), "ms"
    )
    end = leader.capture_time[-1].values + np.timedelta64(int(gracePeriod * 1000), "ms")

    if start is not None:
        follower = follower.isel({dim: (follower.capture_time >= start)})
    if end is not None:
        follower = follower.isel({dim: (follower.capture_time <= end)})

    return follower


def nextCase(case):
    """
    Get the next case identifier.

    Parameters
    ----------
    case : str
        Current case identifier.

    Returns
    -------
    str
        Next case identifier.
    """
    return str(case2timestamp(case) + np.timedelta64(1, "D")).replace("-", "")


def prevCase(case):
    """
    Get the previous case identifier.

    Parameters
    ----------
    case : str
        Current case identifier.

    Returns
    -------
    str
        Previous case identifier.
    """
    return str(case2timestamp(case) - np.timedelta64(1, "D")).replace("-", "")


def timestamp2case(dd):
    """
    Convert timestamp to case identifier.

    Parameters
    ----------
    dd : datetime
        Timestamp.

    Returns
    -------
    str
        Case identifier.
    """
    year = str(dd.year)
    month = "%02i" % dd.month
    day = "%02i" % dd.day
    case = f"{year}{month}{day}"
    return case


def case2timestamp(case):
    """
    Convert case identifier to timestamp.

    Parameters
    ----------
    case : str
        Case identifier.

    Returns
    -------
    numpy.datetime64
        Timestamp.

    Raises
    ------
    NotImplementedError
        If long cases are not implemented.
    """
    if len(case) == 8:
        return np.datetime64(f"{case[:4]}-{case[4:6]}-{case[6:8]}")
    else:
        raise NotImplementedError("long cases not implemented yet")


def rescaleImage(
    frame1, rescale, anti_aliasing=False, anti_aliasing_sigma=None, mode="edge"
):
    """
    Rescale an image.

    Parameters
    ----------
    frame1 : array_like
        Input image.
    rescale : float
        Scaling factor.
    anti_aliasing : bool, optional
        Apply anti-aliasing, by default False.
    anti_aliasing_sigma : float, optional
        Anti-aliasing sigma, by default None.
    mode : str, optional
        Resizing mode, by default "edge".

    Returns
    -------
    array_like
        Rescaled image.
    """
    import skimage

    if len(frame1.shape) == 3:
        newShape = np.array(
            (frame1.shape[0] * rescale, frame1.shape[1] * rescale, frame1.shape[2])
        )
    else:
        newShape = np.array(frame1.shape) * rescale
    frame1 = skimage.transform.resize(
        frame1,
        newShape,
        mode=mode,
        anti_aliasing=anti_aliasing,
        anti_aliasing_sigma=anti_aliasing_sigma,
        preserve_range=True,
        order=0,
    )
    return frame1


def displayImage(frame, doDisplay=True, rescale=None):
    """
    Display an image.

    Parameters
    ----------
    frame : array_like
        Input image.
    doDisplay : bool, optional
        Whether to display, by default True.
    rescale : float, optional
        Scaling factor, by default None.

    Returns
    -------
    IPython.display.Image or None
        Image object if doDisplay=False, otherwise None.
    """
    # opencv cannot handle grayscale png with alpha channel
    import cv2
    import IPython.display

    if (len(frame.shape) == 3) and (frame.shape[2] == 2):
        fill_color = 0
        frameAlpha = frame[:, :, 1]
        frameData = frame[:, :, 0]
        frameCropped = deepcopy(frameData)
        frameCropped[frameAlpha == 0] = fill_color
        frame1 = np.hstack((frameData, frameCropped))
    else:
        frame1 = frame

    if rescale is not None:
        frame1 = rescaleImage(frame1, rescale)

    _, frame1 = cv2.imencode(".png", frame1)

    if doDisplay:
        IPython.display.display(IPython.display.Image(data=frame1.tobytes()))
    else:
        return IPython.display.Image(data=frame1.tobytes())


class ZipFile(zipfile.ZipFile):
    """
    Extended zip file class with automatic parent directory creation. 
    Extra functions to store and retrieve images as numpy arrays
    """
    def __init__(self, file, **kwargs):
        createParentDir(file)
        super().__init__(file, **kwargs)

    def addnpy(self, fname, array):
        # encode
        buf1 = io.BytesIO()
        np.save(buf1, array)
        return self.writestr(fname, buf1.getbuffer())

    def extractnpy(self, fname):
        array = np.load(io.BytesIO(self.read(fname)))
        return array


def imageZipFile(fname, **kwargs):
    """
    Create appropriate archive file handler.

    Parameters
    ----------
    fname : str
        File name.
    **kwargs : dict
        Additional arguments for archive creation.

    Returns
    -------
    Archive handler
        ZipFile or BlockImageArchive instance.
    """
    createParentDir(fname)

    if fname.endswith("zip"):
        return ZipFile(fname, **kwargs)
    else:
        return BlockImageArchive(fname, **kwargs)


class BlockImageArchive:
    """
    A high-efficiency binary archive for storing thousands of small numpy arrays.

    This class provides a single-file storage solution specifically optimized for
    small, variable-sized uint8 arrays. It uses block compression to maximize the
    compression ratio and maintains a JSON index for O(1) random access.

    Parameters
    ----------
    filepath : str
        Path to the archive file.
    mode : {'a', 'w', 'r'}, optional
        'a' : Append/Read mode. Loads existing index and allows adding more data.
        'w' : Write mode. Overwrites any existing file.
        'r' : Read-only mode. Prevents any modifications to the file.
        Default is 'a'.
    block_size : int, optional
        Number of images to group into a single compressed block.
        Default is 256.
    level : int, optional
        Zip compression level.
        Default is 8.
    """

    def __init__(self, filepath, mode="r", block_size=256, level=8):
        self.filepath = filepath
        self.mode = mode
        self.level = level
        self.block_size = block_size
        self.index = {}
        self.buffer = []
        self._f = None

        file_exists = os.path.exists(filepath)

        if mode == "r":
            if not file_exists:
                raise FileNotFoundError(f"Archive not found: {filepath}")
            self._f = open(self.filepath, "rb")
        elif mode == "w":
            if file_exists:
                os.remove(filepath)
            self._f = open(self.filepath, "wb+")
        elif mode == "a":
            self._f = open(self.filepath, "rb+" if file_exists else "wb+")
        else:
            raise ValueError("Mode must be 'r', 'w', or 'a'")

        # Load index if the file is not new
        if file_exists and os.path.getsize(self.filepath) > 8:
            self._read_index_from_disk()

    def _read_index_from_disk(self):
        """Reads the index pointer and loads the index dictionary."""

        self._f.seek(-8, os.SEEK_END)
        index_ptr = struct.unpack("<Q", self._f.read(8))[0]
        self._f.seek(index_ptr)
        data = self._f.read(os.path.getsize(self.filepath) - index_ptr - 8)
        self.index = json.loads(data.decode("utf-8"))

        # If appending, seek to the start of the old index to overwrite it.
        # If reading, this seek doesn't hurt.
        self._f.seek(index_ptr)

    def _write_current_block(self):
        """Compresses buffered images and appends them to the file."""
        if self.mode == "r":
            raise IOError("Cannot write to an archive opened in read-only mode.")

        if not self.buffer:
            return

        raw_bytes_list = [img.tobytes() for _, img, _ in self.buffer]
        raw_block = b"".join(raw_bytes_list)
        compressed_block = zlib.compress(raw_block, level=self.level)

        self._f.seek(0, os.SEEK_END)
        block_offset = self._f.tell()
        self._f.write(compressed_block)
        block_len = self._f.tell() - block_offset

        current_inner_offset = 0
        for i, (image_id, array, shape) in enumerate(self.buffer):
            img_len = len(raw_bytes_list[i])
            self.index[str(image_id)] = [
                block_offset,
                block_len,
                current_inner_offset,
                img_len,
                shape,
            ]
            current_inner_offset += img_len

        self.buffer = []

    def addnpy(self, image_id, array):
        """
        Add a numpy array to the archive.

        The array is added to an internal buffer. When the buffer reaches
        `block_size`, the images are compressed and written to disk.

        Parameters
        ----------
        image_id : str or int
            Unique identifier for the image.
        array : numpy.ndarray
            The uint8 array to store.
        """
        if self.mode == "r":
            raise IOError("Archive is in read-only mode.")
        self.buffer.append((image_id, array, list(array.shape)))
        if len(self.buffer) >= self.block_size:
            self._write_current_block()

    def extractnpy(self, image_id):
        """
        Retrieve a numpy array by its ID.

        This method reads only the required compressed block from disk,
        decompresses it in memory, and returns the specific slice.

        Parameters
        ----------
        image_id : str or int
            The identifier of the image to retrieve.

        Returns
        -------
        numpy.ndarray
            The reconstructed uint8 array.

        Raises
        ------
        KeyError
            If the image_id is not found in the archive index.
        """
        if str(image_id) not in self.index:
            raise KeyError(f"ID {image_id} not found.")

        b_offset, b_len, inner_off, img_len, shape = self.index[str(image_id)]

        self._f.seek(b_offset)
        compressed_block = self._f.read(b_len)
        raw_block = zlib.decompress(compressed_block)

        img_bytes = raw_block[inner_off : inner_off + img_len]
        return np.frombuffer(img_bytes, dtype=np.uint8).reshape(shape)

    def close(self):
        """
        Closes the file handle. In 'w' or 'a' mode, finalizes the index.
        In 'r' mode, simply closes the file.
        """
        if self._f and not self._f.closed:
            # Only write metadata if we were in a writing mode
            if self.mode in ("w", "a"):
                self._write_current_block()
                self._f.seek(0, os.SEEK_END)
                ptr = self._f.tell()
                self._f.write(json.dumps(self.index).encode("utf-8"))
                self._f.write(struct.pack("<Q", ptr))

            self._f.close()
            self._f = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        try:
            if hasattr(self, "_f") and self._f is not None:
                if not self._f.closed:
                    self.close()
        except:
            pass


def createParentDir(file, mode=None):
    """
    Create parent directory if it doesn't exist.

    Parameters
    ----------
    file : str
        File path.
    mode : int, optional
        Directory mode, by default None.
    """
    parent_dir = os.path.dirname(os.path.abspath(file))
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
        if mode is not None:
            os.chmod(parent_dir, mode)
    return


def savefig(fig, config, filename, **kwargs):
    """
    Save a matplotlib Figure to `filename` with proper permissions.

    - Creates parent directories if they do not exist.
    - Directories: owner rwx, group rwx, others rx (0o775)
    - File: owner rw, group rw, others r (0o664)

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    config : dict
        Configuration settings.
    filename : str
        Output file name.
    **kwargs : dict
        Additional arguments for saving.
    """
    # Ensure parent directory exists
    createParentDir(filename, config.dirMode)

    # sometimes exisiting files make problems
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass

    # Save the figure
    fig.savefig(filename, **kwargs)

    # Set file permissions
    try:
        os.chmod(filename, config.fileMode)
    except PermissionError:
        log.warning(f"chmod {config.fileMode} {filename} failed")


def ncAttrs(site, visssGen, extra={}):
    """
    Generate NetCDF attributes.

    Parameters
    ----------
    site : str
        Site identifier.
    visssGen : str
        Generator information.
    extra : dict, optional
        Extra attributes, by default {}.

    Returns
    -------
    dict
        NetCDF attributes.
    """
    import cv2
    import psutil

    my_process = psutil.Process(os.getpid())
    myCommand = " ".join(my_process.cmdline())

    if os.environ.get("USER") is not None:
        user = f" by user {os.environ.get('USER')}"
    else:
        user = ""
    attrs = {
        "title": f"Video In Situ Snowfall Sensor (VISSS) observations at {site}",
        "source": f"{visssGen} observations at {site}",
        "history": f"{str(datetime.datetime.utcnow())}: created with VISSSlib {__versionFull__} and OpenCV {cv2.__version__} on {socket.getfqdn()}{user}",
        "command": myCommand,
        "references": "Maahn, M., D. Moisseev, I. Steinke, N. Maherndl, and M. D. Shupe, 2024: Introducing the Video In Situ Snowfall Sensor (VISSS). Atmospheric Measurement Techniques, 17, 899–919, https://doi.org/10.5194/amt-17-899-2024.",
    }

    attrs.update(extra)
    return attrs


def finishNc(dat, site, visssGen, extra={}):
    """
    Finalize NetCDF dataset with attributes and encoding.

    Parameters
    ----------
    dat : xarray.Dataset
        Dataset to finalize.
    site : str
        Site identifier.
    visssGen : str
        Generator information.
    extra : dict, optional
        Extra attributes, by default {}.

    Returns
    -------
    xarray.Dataset
        Finalized dataset.
    """
    # todo: add yaml dump of config file

    extra = deepcopy(extra)
    for k in extra.keys():
        extra[k] = str(extra[k])

    dat.attrs.update(ncAttrs(site, visssGen, extra=extra))

    for k in list(dat.data_vars) + list(dat.coords):
        if dat[k].dtype == np.float64:
            dat[k] = dat[k].astype(np.float32)

        # #newest netcdf4 version doe snot like strings or objects:
        if (dat[k].dtype == object) or (dat[k].dtype == str):
            dat[k] = dat[k].astype("U")

        if not str(dat[k].dtype).startswith("<U"):
            dat[k].encoding = {}
            dat[k].encoding["zlib"] = True
            dat[k].encoding["complevel"] = 5
        # need to overwrite units becuase leting xarray handle that might lead to inconsistiencies
        # due to mixing of milli and micro seconds
        if k.endswith("time") or k.endswith("time_orig") or k.endswith("time_even"):
            dat[k].encoding["units"] = "microseconds since 1970-01-01 00:00:00"

    # sort variabels alphabetically
    dat = dat[sorted(dat.data_vars)]
    return dat


def getPrevRotationEstimates(datetime64, config):
    """
    Extract rotation estimates from config.

    Parameters
    ----------
    datetime64 : numpy.datetime64
        Datetime to query.
    config : dict
        Configuration settings.

    Returns
    -------
    tuple
        Rotation matrix, error matrix, and timestamp.
    """
    rotate, rotate_time = getPrevRotationEstimate(datetime64, "transformation", config)
    assert len(rotate) != 0
    rotate_err, rotate_err_time = getPrevRotationEstimate(
        datetime64, "transformation_err", config
    )
    assert len(rotate_err) != 0
    assert rotate_time == rotate_err_time

    return rotate, rotate_err, rotate_time


def getPrevRotationEstimate(datetime64, key, config):
    """
    Extract single rotation estimate from config.

    Parameters
    ----------
    datetime64 : numpy.datetime64
        Datetime to query.
    key : str
        Key for rotation estimate.
    config : dict
        Configuration settings.

    Returns
    -------
    tuple
        Rotation estimate and timestamp.

    Raises
    ------
    RuntimeError
        If no rotation estimate found.
    """

    rotate_all = {
        np.datetime64(datetime.datetime.strptime(d.ljust(15, "0"), "%Y%m%d-%H%M%S")): r[
            key
        ]
        for d, r in config.rotate.items()
    }
    rotTimes = np.array(list(rotate_all.keys()))
    rotDiff = datetime64 - rotTimes
    rotTimes = rotTimes[rotDiff >= np.timedelta64(0)]
    rotDiff = rotDiff[rotDiff >= np.timedelta64(0)]

    try:
        prevTime = rotTimes[np.argmin(rotDiff)]
    except ValueError:
        raise RuntimeError(
            f"datetime64 {datetime64} before earliest rotation estimate {np.min(np.array(list(rotate_all.keys())))}"
        )
    return rotate_all[prevTime], prevTime


def rotXr2dict(dat, config=None):
    """
    Convert rotation xarray data to dictionary.

    Parameters
    ----------
    dat : xarray.Dataset
        Rotation data.
    config : dict, optional
        Configuration settings, by default None.

    Returns
    -------
    dict
        Configuration dictionary with rotation info.
    """
    import pandas as pd

    if config is None:
        config = {}
        config["rotate"] = {}
    for ii, tt in enumerate(dat.file_starttime):
        t1 = pd.to_datetime(str(tt.values)).strftime("%Y%m%d-%H%M%S")
        config["rotate"][t1] = {
            "transformation": dat.isel(file_starttime=ii)
            .sel(camera_rotation="mean")
            .to_pandas()
            .to_dict(),
            "transformation_err": dat.isel(file_starttime=ii)
            .sel(camera_rotation="err")
            .to_pandas()
            .to_dict(),
        }

    return config


def rotDict2Xr(rotate, rotate_err, prevTime):
    """
    Convert rotation dictionary to xarray dataset.

    Parameters
    ----------
    rotate : dict
        Rotation matrix.
    rotate_err : dict
        Rotation error matrix.
    prevTime : numpy.datetime64
        Previous timestamp.

    Returns
    -------
    xarray.Dataset
        Rotation dataset.
    """
    if rotate is np.nan:
        rotate = {"camera_Ofz": np.nan, "camera_phi": np.nan, "camera_theta": np.nan}
    if rotate_err is np.nan:
        rotate_err = {
            "camera_Ofz": np.nan,
            "camera_phi": np.nan,
            "camera_theta": np.nan,
        }

    metaRotationDf = {}
    for k in rotate.keys():
        metaRotationDf[k] = xr.DataArray(
            np.ones((1, 2)) * np.array([rotate[k], rotate_err[k]]),
            dims=["file_starttime", "camera_rotation"],
            coords=[[prevTime], np.array(["mean", "err"])],
        )
    return xr.Dataset(metaRotationDf)


def execute_stdout(command):
    """
    Execute command and print output continuously.

    Parameters
    ----------
    command : str
        Command to execute.

    Returns
    -------
    tuple
        Exit code and output.
    """
    # launch application as subprocess and print output constantly:
    # http://stackoverflow.com/questions/4417546/constantly-print-subprocess-output-while-process-is-running
    print(command)
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    output = ""

    # Poll process for new output until finished
    for line in process.stdout:
        line = line.decode()
        print(line, end="")
        output += line
        time.sleep(0.1)
    for line in process.stderr:
        line = line.decode()
        print(line, end=" ")
        output += line
        time.sleep(0.1)

    process.wait()
    exitCode = process.returncode

    return exitCode, output


def concat(*strs):
    """
    Concatenate strings with spaces.

    Parameters
    ----------
    *strs : str
        Strings to concatenate.

    Returns
    -------
    str
        Concatenated string.
    """
    concat = " ".join([str(s) for s in strs])
    return concat


def concatImgY(im1, im2, background=0):
    """
    Concatenate images vertically.

    Parameters
    ----------
    im1 : array_like
        Upper image.
    im2 : array_like
        Lower image.
    background : int, optional
        Background color, by default 0.

    Returns
    -------
    array_like
        Concatenated image.
    """
    y1, x1 = im1.shape
    y2, x2 = im2.shape

    y3 = y1 + y2
    x3 = max(x1, x2)
    imT = np.full((y3, x3), background, dtype=np.uint8)
    imT[:y1, :x1] = im1
    imT[y1:, :x2] = im2
    # print("Y",im1.shape, im2.shape, imT.shape)

    return imT


def concatImgX(im1, im2, background=0):
    """
    Concatenate images horizontally.

    Parameters
    ----------
    im1 : array_like
        Left image.
    im2 : array_like
        Right image.
    background : int, optional
        Background color, by default 0.

    Returns
    -------
    array_like
        Concatenated image.
    """
    y1, x1 = im1.shape
    y2, x2 = im2.shape

    y3 = max(y1, y2)
    x3 = x1 + x2
    imT = np.full((y3, x3), background, dtype=np.uint8)
    imT[:y1, :x1] = im1
    imT[:y2, x1:] = im2

    # print("X",im1.shape, im2.shape, imT.shape)

    return imT


def open2(file, config, mode="r", cleanUp=True, **kwargs):
    """
    Open file with directory creation and permissions.

    Parameters
    ----------
    file : str
        File path.
    config : dict
        Configuration settings.
    mode : str, optional
        File mode, by default "r".
    cleanUp : bool, optional
        Clean up temporary files, by default True.
    **kwargs : dict
        Additional arguments for opening.

    Returns
    -------
    file handle
        Opened file handle.
    """
    createParentDir(file, mode=config.dirMode)

    if cleanUp:
        origFile = file.replace(".nc.nodata", ".nc").replace(".nc.broken.txt", ".nc")
        tryRemovingFile(origFile)
        tryRemovingFile(f"{origFile}.nodata")
        tryRemovingFile(f"{origFile}.broken.txt")
        pass
    f = open(file, mode, **kwargs)
    os.chmod(file, config.fileMode)

    return f


def tryRemovingFile(file):
    """
    Try to remove a file.

    Parameters
    ----------
    file : str
        File path.
    """
    try:
        os.remove(file)
    except:
        pass
    else:
        if not file.endswith("processing.txt"):
            log.warning(f"removed {file}")
    return


def to_netcdf2(dat, config, file, **kwargs):
    """
    Save dataset to NetCDF with directory creation.
    Write to random file and move to final file to 
    avoid errors due to race conditions or exisiting files
    
    Parameters
    ----------
    dat : xarray.Dataset
        Dataset to save.
    config : dict
        Configuration settings.
    file : str
        Output file name.
    **kwargs : dict
        Additional arguments for saving.

    Returns
    -------
    None
    """

    from dask.diagnostics import ProgressBar

    print(f"saving {file}")

    createParentDir(file, mode=config.dirMode)
    if os.path.isfile(file):
        tryRemovingFile(file)

    tmpFile = f"{file}.{np.random.randint(0, 99999 + 1)}.tmp.cdf"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        if dat[list(dat.data_vars)[-1]].chunks is not None:
            with ProgressBar():
                res = dat.to_netcdf(tmpFile, **kwargs)
        else:
            res = dat.to_netcdf(tmpFile, **kwargs)
    os.chmod(tmpFile, config.fileMode)
    os.rename(tmpFile, file)
    log.info(f"saved {file}")

    tryRemovingFile(f"{file}.nodata")
    tryRemovingFile(f"{file}.broken.txt")

    return res


@numba.jit(nopython=True)
def linreg(x, y):
    """
    Perform linear regression.
    Turns out to be a lot faster than scipy and 
    numpy code when using numba (1.7 us vs 25 us)


    Parameters
    ----------
    x : array_like
        Independent variable.
    y : array_like
        Dependent variable.

    Returns
    -------
    tuple
        Slope and intercept of regression line.
    """
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    n = len(x)
    # formula to calculate m : x-mean_x*y-mean_y/ (x-mean_x)^2
    num = 0
    denom = 0
    for i in range(n):
        num += (x[i] - mean_x) * (y[i] - mean_y)
        denom += (x[i] - mean_x) ** 2
    slope = num / denom

    # calculate c = y_mean - m * mean_x / n
    intercept = mean_y - slope * mean_x
    return slope, intercept


def cart2pol(x, y):
    """
    Convert Cartesian coordinates to polar.

    Parameters
    ----------
    x : array_like
        X coordinates.
    y : array_like
        Y coordinates.

    Returns
    -------
    tuple
        Polar coordinates (rho, phi).
    """

    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)


@taskqueue.queueable
def runCommandInQueue(IN, stdout=subprocess.DEVNULL):
    """
    Run command in queue with file locking.

    Parameters
    ----------
    IN : tuple
        Command and output file.
    stdout : file handle, optional
        Standard output, by default subprocess.DEVNULL.

    Returns
    -------
    bool
        Success status.
    """
    import portalocker

    command, fOut = IN
    tmpFile = os.path.basename("%s.processing.txt" % fOut)

    success = True
    running = False
    # with statement extended to avoid race conditions
    try:
        with portalocker.Lock(tmpFile, timeout=0) as f:
            f.write("PID & Host: %i %s\n" % (os.getpid(), socket.gethostname()))
            f.write("Command: %s\n" % command)
            f.write("Outfile: %s\n" % fOut)
            f.write("#########################\n")
            f.flush()
            log.info(f"written {tmpFile} in {os.getcwd()}")
            log.info(command)

            # proc = subprocess.Popen(shlex.split(f'bash -c "{command}"'), stderr=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
            proc = subprocess.Popen(
                command, shell=True, stdout=stdout, stderr=subprocess.PIPE
            )

            # Poll process for new output until finished
            if proc.stdout is not None:
                for line in proc.stdout:
                    line = line.decode()
                    log.info(line)
                    f.write(line)
                    f.flush()
            for line in proc.stderr:
                line = line.decode()
                log.error(line)
                f.write(line)
                f.flush()

            proc.wait()
            exitCode = proc.returncode
            if exitCode != 0:
                success = False
                log.error(f"BROKEN {fOut} {exitCode}")
            else:
                log.info(f"SUCCESS {fOut} {exitCode}")

            # flush and sync to filesystem
            f.flush()
            os.fsync(f.fileno())

    except portalocker.LockException:
        log.info(f"{fOut} RUNNING")
        success = True
        running = True

    if not success:
        shutil.copy(tmpFile, "%s.broken.txt" % tmpFile)
        try:
            createParentDir(fOut)
            shutil.copy(tmpFile, "%s.broken.txt" % fOut)
        except:
            pass
    if not running:
        tryRemovingFile(tmpFile)

    return success


def create_TaskQueuePatched():
    """
    Create patched TaskQueue class with custom is_empty_wait function.

    Returns
    -------
    class
        Patched TaskQueue class.
    """
    import taskqueue

    class TaskQueuePatched(taskqueue.TaskQueue):
        def is_empty_wait(self):
            import psutil

            waitTime = 5
            # first delay everything if there are no jobs
            for i in range(4):
                if not self.is_empty():
                    break
                print(f"waiting for jobs... {i}", flush=True)
                time.sleep(waitTime)
            # if there are really no jobs, nothing to do
            if self.is_empty():
                return True
            print("jobs present")

            # if there are jobs, check for killwitch file
            if os.path.isfile("VISSS_KILLSWITCH"):
                print(f"{ii}, found file VISSS_KILLSWITCH, stopping", flush=True)
                return True
            print("no VISSS_KILLSWITCH")

            # if there are jobsm check for memory and wait otherwise
            while True:
                if psutil.virtual_memory().percent < 95:
                    break
                print(f"waiting for available memory...", flush=True)
                time.sleep(waitTime)
            print("sufficient memory")
            return self.is_empty()

    return TaskQueuePatched


def worker1(queue, ww=0, status=None, waitTime=5):
    """
    Worker function for processing queue items.

    Parameters
    ----------
    queue : str
        Queue name.
    ww : int, optional
        Worker number, by default 0.
    status : array, optional
        Status array, by default None.
    waitTime : int, optional
        Wait time between checks, by default 5.

    Returns
    -------
    None
    """
    print(f"starting worker {ww} for {queue}", flush=True)
    time.sleep(ww / 5.0)  # to avoid race conditions
    TaskQueuePatched = create_TaskQueuePatched()  # generator for lazy import
    tq = TaskQueuePatched(f"fq://{queue}")
    out = None
    while True:
        if not tq.is_empty():
            if status is not None:
                status[ww] = 1
            try:
                out = tq.poll(
                    verbose=True,
                    tally=True,
                    stop_fn=tq.is_empty,
                    lease_seconds=2,
                    backoff_exceptions=[BlockingIOError],
                )
            except:
                pass
            finally:
                if status is not None:
                    status[ww] = 0
        else:
            print(f"worker {ww} queue {queue} empty", flush=True)
        if status is not None:
            if np.all([ss == 0 for ss in status]):
                print(
                    f"do not restart worker {ww} because all empty {[status[i] for i in range(len(status))]}",
                    flush=True,
                )
                break
            summary = [status[i] for i in range(len(status))]
        else:
            summary = ""
        print(
            f"restart worker {ww} {summary}",
            flush=True,
        )
        time.sleep(waitTime)

    return out


def workers(queue, nJobs=os.cpu_count(), waitTime=60, join=True):
    """
    Start multiple worker processes.

    Parameters
    ----------
    queue : str
        Queue name.
    nJobs : int, optional
        Number of jobs, by default os.cpu_count().
    waitTime : int, optional
        Wait time between checks, by default 60.
    join : bool, optional
        Join processes, by default True.

    Returns
    -------
    list
        Worker processes.
    """
    # for communication between subprocesses
    print(f"starting {nJobs} workers")
    status = multiprocessing.Array("i", [0] * nJobs)
    workerList = []
    for ww in range(nJobs):
        x = multiprocessing.Process(
            target=worker1,
            args=(queue,),
            kwargs={
                "ww": ww,
                "status": status,
                "waitTime": waitTime,
            },
        )
        x.start()
        workerList.append(x)
    if join:
        [x.join() for x in workerList]
    return workerList


def checkForExisting(ffOut, level0=None, events=None, parents=None):
    """
    Check if file exists and is up-to-date including potential parents.
    
    Parameters
    ----------
    ffOut : str
        Output file path.
    level0 : list, optional
        Level 0 data files, by default None.
    events : list, optional
        Event files, by default None.
    parents : list, optional
        Parent files, by default None.

    Returns
    -------
    bool
        True if file should be regenerated.
    """
    # checks whether file exists and checks age of parents and meta files
    if not os.path.isfile(ffOut):
        return False
    if level0 is not None:
        if len(level0) == 0:
            print("no level0 data", ffOut)
            return True
    if events is not None:
        if np.any(
            os.path.getmtime(ffOut)
            < np.array([0] + [os.path.getmtime(f) for f in events])
        ):
            print("file exists but older than event file, redoing", ffOut)
            return False
    if parents is not None:
        if np.any(
            os.path.getmtime(ffOut)
            < np.array([0] + [os.path.getmtime(f) for f in parents])
        ):
            print(
                "file exists but older than parents files, redoing",
                ffOut,
            )
            return False

    return True


def unpackQualityFlags(quality, doubleTimestamps=False):
    """
    Unpack quality flags into boolean array.

    Parameters
    ----------
    quality : xarray.DataArray
        Quality flags.
    doubleTimestamps : bool, optional
        Double timestamps for plotting, by default False.

    Returns
    -------
    xarray.DataArray
        Expanded quality flags.
    """
    flags = xr.DataArray(
        [
            "recordingFailed",
            "processingFailed",
            "cameraBlocked",
            "blowingSnow",
            "obervationsDiffer",
            "tracksTooShort",
        ],
        dims=["flag"],
        name="flag",
    )
    qualityExpanded = np.unpackbits(
        quality.values[..., np.newaxis], axis=-1, count=len(flags)
    )
    qualityExpanded = xr.DataArray(qualityExpanded, coords=[quality.time, flags])

    # trick for plotting
    if doubleTimestamps:
        qualityExpanded2 = xr.DataArray(
            qualityExpanded.values,
            coords=[qualityExpanded.time + np.timedelta64(5, "m"), flags],
        )
        qualityExpanded = xr.concat(
            (qualityExpanded, qualityExpanded2), dim="time"
        ).sortby("time")
    return qualityExpanded


def timestamp2str(ts):
    """
    Convert timestamp to string.

    Parameters
    ----------
    ts : int
        Timestamp.

    Returns
    -------
    str
        Formatted timestamp string.
    """
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
