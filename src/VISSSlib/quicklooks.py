# -*- coding: utf-8 -*-

import datetime
import functools
import glob
import os
import shutil
import sys
import uuid
import warnings
from copy import deepcopy

import numpy as np
import xarray as xr
from image_packer import packer
from loguru import logger as log

from . import *
from . import __version__, av, files, tools


@tools.loopify
def generate(
    case,
    config,
    level,
    camera="all",
    version=__version__,
    skipExisting=True,
    endYesterday=False,
):
    """
    Generate quicklooks for a specified level over a range of days.

    Parameters
    ----------
    case : str
        case identifier. e.g. string in format YYYYMMDD
    config : dict
        Configuration dictionary or settings file
    camera : str, optional
        Camera identifier. Default is "all"".
    level : str
        The data level for which to generate quicklooks. Options include:
        'level0', 'level1detect', 'metaFrames', 'level2detect', 'level1match',
        'level2match', 'level2track', 'level3combinedRiming'
    version : str, optional
        Version identifier for the processing, by default __version__
    skipExisting : bool, optional
        Whether to skip generation if output already exists, by default True
    endYesterday : bool, optional
        Whether to end the case range at yesterday. Default is True.
        This parameter is passed to getCaseRange.

    Returns
    -------
    None
        Function processes files but doesn't return anything directly

    Raises
    ------
    ValueError
        If the specified level is not recognized
    """

    if camera == "all":
        if level in ["level0", "level1detect", "metaFrames", "level2detect"]:
            cameras = config["instruments"]
        else:
            cameras = [config.leader]
    else:
        cameras = [camera]
    for camera in cameras:
        if level == "level0":
            res = level0Quicklook(
                case, camera, config, version=version, skipExisting=skipExisting
            )
        elif level == "metaFrames":
            res = metaFramesQuicklook(
                case, camera, config, version=version, skipExisting=skipExisting
            )
        elif level == "metaRotation":
            res = metaRotationQuicklook(
                case, config, version=version, skipExisting=skipExisting
            )
        elif level == "level1detect":
            res = createLevel1detectQuicklook(
                case, camera, config, version=version, skipExisting=skipExisting
            )
        elif level == "level1match":
            res = createLevel1matchParticlesQuicklook(
                case, config, version=version, skipExisting=skipExisting
            )
        elif level == "level2detect":
            res = createLevel2detectQuicklook(
                case, camera, config, version=version, skipExisting=skipExisting
            )
        elif level == "level2match":
            res = createLevel2matchQuicklook(
                case, config, version=version, skipExisting=skipExisting
            )
        elif level == "level2track":
            res = createLevel2trackQuicklook(
                case, config, version=version, skipExisting=skipExisting
            )
        elif level == "level3combinedRiming":
            res = createLevel3RimingQuicklook(
                case, config, version=version, skipExisting=skipExisting
            )
        else:
            raise ValueError(f"Do not know level {level}")
    return res


def _plotVar(
    pVar,
    capture_time,
    ax,
    ylabel=None,
    axhline=None,
    xlabel=None,
    resample="5min",
    func="mean",
    color="C1",
    label=None,
    ratiovar=None,
):
    """
    Plot variable data with error bars or statistics.

    Parameters
    ----------
    pVar : array-like
        Variable data to plot
    capture_time : xarray.DataArray
        Time coordinates for the data
    ax : matplotlib.axes.Axes
        Axes object to plot on
    ylabel : str, optional
        Label for y-axis, by default None
    axhline : float, optional
        Horizontal line to draw, by default None
    xlabel : str, optional
        Label for x-axis, by default None
    resample : str, optional
        Resampling frequency, by default "5min"
    func : str, optional
        Function to apply for aggregation ('mean', 'count', 'first', 'ratio'),
        by default "mean"
    color : str, optional
        Color for the plot, by default "C1"
    label : str, optional
        Label for legend, by default None
    ratiovar : xarray.DataArray, optional
        Variable for ratio calculation, by default None

    Returns
    -------
    tuple
        (axes, resampled_data)
    """
    if axhline is not None:
        ax.axhline(axhline, color="magenta", lw=0.5)

    try:
        capture_time = capture_time.isel(camera=0)
    except ValueError:
        pass

    pVar = xr.DataArray(pVar, coords=[capture_time.values], dims=["time"])
    pVar = pVar.sortby("time")  # this is to avoid making a fuzz about jumping indices
    pVar = pVar.resample(time=resample)

    if func == "mean":
        pMean = pVar.mean()
        pStd = pVar.std()

        try:
            pMean.plot(ax=ax, marker=".", color=color, label=label)
        except TypeError:
            log.warning("no data to plot pMean")
        ax.fill_between(pMean.time, pMean - pStd, pMean + pStd, alpha=0.3, color=color)
    elif func == "count":
        pCount = pVar.count()
        try:
            pCount.plot(ax=ax, marker=".", color=color, label=label)
        except TypeError:
            log.warning("no data to plot pCount")
    elif func == "first":
        pFirst = pVar.first()
        try:
            pFirst.plot(ax=ax, marker=".", color=color, label=label)
        except TypeError:
            log.warning("no data to plot pFirst")
    elif func == "ratio":
        try:
            (ratiovar.count() / pVar.count()).plot(
                ax=ax, marker=".", color=color, label=label
            )
        except TypeError:
            log.warning("no data to plot ratiovar/pvar")
    else:
        raise ValueError(f"Do not know {func}")

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    return ax, pVar


def _plot2dhist(
    pVar,
    capture_time,
    ax,
    cax,
    bins,
    ylabel=None,
    logScale=True,
    resample="5min",
    cbarlabel=None,
):
    """
    Plot 2D histogram data.

    Parameters
    ----------
    pVar : array-like
        Variable data to plot
    capture_time : xarray.DataArray
        Time coordinates for the data
    ax : matplotlib.axes.Axes
        Axes object to plot on
    cax : matplotlib.axes.Axes
        Axes object for colorbar
    bins : array-like
        Bin edges for histogram
    ylabel : str, optional
        Label for y-axis, by default None
    logScale : bool, optional
        Whether to use logarithmic scale, by default True
    resample : str, optional
        Resampling frequency, by default "5min"
    cbarlabel : str, optional
        Label for colorbar, by default None

    Returns
    -------
    tuple
        (axes, resampled_data)
    """
    import matplotlib.pyplot as plt

    pVar = xr.DataArray(
        pVar, coords=[capture_time.isel(camera=0).values], dims=["time"]
    )
    pVar = pVar.sortby("time")  # this is to avoid making a fuzz about jumping indices
    pVar = pVar.resample(time=resample)

    binMeans = (bins[1:] + bins[:-1]) / 2.0
    hists = []
    labels = []
    for l, p in pVar:
        labels.append(l)
        hists.append(
            100 * np.histogram(p, bins=bins)[0] / np.sum(np.isfinite(p.values))
        )

    hists = xr.DataArray(hists, coords=[labels, binMeans], dims=["time", "bins"])
    hists = hists.where(hists != 0)

    hists = hists.resample(
        time=resample
    ).first()  # little trick to fill up missing values
    # import pdb; pdb.set_trace()

    if np.any(np.array(hists.shape) <= 1):
        log.warning("no data to plot hists")
    else:
        pc = hists.T.plot.pcolormesh(
            ax=ax, cbar_ax=cax, cbar_kwargs={"label": cbarlabel}
        )

    if logScale:
        ax.set_yscale("log")

    ax.set_ylabel(ylabel)
    ax.set_xlabel(None)
    ax.set_ylim(bins[0], bins[-1])

    return ax, pVar


def _crop(image):
    """
    Crop black image parts from an image.

    Parameters
    ----------
    image : numpy.ndarray
        Input image array

    Returns
    -------
    numpy.ndarray
        Cropped image array
    """
    y_nonzero, x_nonzero = np.nonzero(image)
    return image[
        np.min(y_nonzero) : np.max(y_nonzero) + 1,
        np.min(x_nonzero) : np.max(x_nonzero) + 1,
    ]


################################################################################
# Shared look & feel of the (non-particle-image) quicklooks: one rc style, flag
# shading with a single legend, a common time axis and stacked-panel figures.

_QL_RC = {
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "legend.frameon": True,
    "legend.framealpha": 0.85,
    "legend.edgecolor": "none",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "0.85",
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.4,
    "lines.markersize": 4,
}

# colors identifying the two cameras in plots showing both of them
_LEADER_COLOR = "tab:red"
_FOLLOWER_COLOR = "tab:purple"


def _styled(func):
    """Run a quicklook function with the common quicklook rc style."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import matplotlib as mpl

        with mpl.rc_context(_QL_RC):
            return func(*args, **kwargs)

    return wrapper


def _dayLimits(ff):
    """Start and end of the day of a FindFiles object as datetime64."""
    t0 = np.datetime64(f"{ff.year}-{ff.month}-{ff.day}T00:00")
    return t0, t0 + np.timedelta64(1, "D")


def _timeStep(times, default=np.timedelta64(10, "m")):
    """Typical (median) spacing of a time array, `default` if undefined."""
    times = np.sort(np.asarray(times).astype("datetime64[ns]"))
    times = times[~np.isnat(times)]
    dt = np.diff(times).astype("timedelta64[s]").astype(float)
    dt = dt[dt > 0]
    if len(dt) == 0:
        return default
    return np.timedelta64(int(np.median(dt)), "s")


def _suptitle(fig, what, ff, config):
    fig.suptitle(
        f"VISSS {what}  ·  {ff.year}-{ff.month}-{ff.day}  ·  {config['name']}",
        fontsize=15,
        fontweight="bold",
    )


def _panelTitle(ax, text):
    # inside the axes: tight_layout does not reserve room for titles between
    # stacked panels
    ax.text(
        0.005,
        0.96,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 2},
        zorder=5,
    )


def _placeholderFigure(what, ff, config, message):
    """Small figure for days without (raw) data, so the gallery has no gaps."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 3), layout="constrained")
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=20,
        color="0.45",
    )
    _suptitle(fig, what, ff, config)
    return fig


def _shadeSpans(axs, starts, width, color, label, alpha=0.25, hatch=None):
    """
    Shade full-height spans [start, start+width) on all axes.

    Unlike `fill_between` between flagged time stamps, every flagged time
    step is shown (also isolated ones) and the y-limits are not touched.

    Returns
    -------
    matplotlib.patches.Patch or None
        Legend handle, None if there is nothing to shade.
    """
    import matplotlib.dates as mdates
    from matplotlib.patches import Patch

    starts = np.asarray(starts).astype("datetime64[ns]")
    starts = starts[~np.isnat(starts)]
    if len(starts) == 0:
        return None
    x = mdates.date2num(starts)
    w = width / np.timedelta64(1, "D")
    for ax in np.atleast_1d(axs):
        ylim = ax.get_ylim()
        ax.broken_barh(
            [(xi, w) for xi in x],
            (0, 1),
            transform=ax.get_xaxis_transform(),
            facecolors=color,
            edgecolors="none",
            alpha=alpha,
            hatch=hatch,
            zorder=1,
        )
        ax.set_ylim(ylim)  # shading must not change the autoscaled limits
    return Patch(facecolor=color, alpha=alpha, hatch=hatch, label=label)


def _flagStyles(config):
    """Color/label of the level2 quality flags (in the order of the legend)."""
    q = config.quality
    return [
        ("recordingFailed", "0.55", 0.2, "cameras off"),
        ("processingFailed", "purple", 0.25, "processing failed"),
        ("cameraBlocked", "red", 0.25, f"camera blocked > {q.blockedPixThresh*100:g}%"),
        (
            "blowingSnow",
            "orange",
            0.25,
            f"blowing snow > {q.blowingSnowFrameThresh*100:g}%",
        ),
        ("sunGlint", "gold", 0.35, f"sun glint > {q.sunGlintRatioThresh*100:g}%"),
        ("obervationsDiffer", "magenta", 0.25, "observed # of particles differ"),
        (
            "trackingIncomplete",
            "blue",
            0.2,
            f"tracking completeness < {q.minTrackCompleteness}",
        ),
        ("zResidualTooWide", "cyan", 0.25, f"Z-residual sigma > {q.maxZSigma} px"),
    ]


def _shadeQualityFlags(axs, dat2, config, flagNames):
    """Shade the level2 quality flags `flagNames`, return the legend handles."""
    quality = tools.unpackQualityFlags(dat2.qualityFlags)
    time = quality.time.values
    width = _timeStep(time, default=np.timedelta64(5, "m"))
    handles = []
    for name, color, alpha, label in _flagStyles(config):
        if name not in flagNames:
            continue
        flag = quality.sel(flag=name).values.astype(bool)
        handle = _shadeSpans(axs, time[flag], width, color, label, alpha=alpha)
        if handle is not None:
            handles.append(handle)
    return handles


def _restartTimes(events):
    """Times at which the acquisition software was (re)started."""
    isRestart = np.array(
        [str(e).startswith(("start", "launch")) for e in events.event.values],
        dtype=bool,
    )
    return events.file_starttime.values[isRestart]


def _markRestarts(axs, times, label="restarted", color="red", ls=":"):
    """Draw vertical lines at `times`, return the legend handle (or None)."""
    from matplotlib.lines import Line2D

    times = np.asarray(times).astype("datetime64[ns]")
    times = times[~np.isnat(times)]
    if len(times) == 0:
        return None
    for ax in np.atleast_1d(axs):
        ax.vlines(
            times,
            0,
            1,
            transform=ax.get_xaxis_transform(),
            colors=color,
            linestyles=ls,
            linewidth=1,
            zorder=3,
        )
    return Line2D([], [], color=color, ls=ls, lw=1, label=label)


def _formatDayAxes(axs, ff, xlabel="time"):
    """
    Common time axis for the panels of a day: 00:00-24:00, ticks every 3 h
    (minor every hour), labels only below the lowest panel.
    """
    import matplotlib.dates as mdates
    from matplotlib.ticker import FuncFormatter

    axs = list(axs)
    t0, t1 = _dayLimits(ff)
    end = mdates.date2num(t1)

    def formatter(x, pos=None):
        if np.isclose(x, end):
            return "24:00"
        return mdates.num2date(x).strftime("%H:%M")

    for ax in axs:
        ax.set_xlim(t0, t1)
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 3)))
        ax.xaxis.set_minor_locator(mdates.HourLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(formatter))
        ax.set_xlabel("")
        ax.tick_params(axis="x", which="minor", length=2)
    for ax in axs[:-1]:
        ax.tick_params(axis="x", labelbottom=False)
    axs[-1].set_xlabel(xlabel)


def _legendAbove(ax, handles, ncol=4):
    """One legend above the top panel (does not hide any data)."""
    if len(handles) == 0:
        return
    ax.legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(1, 1.0),
        ncol=ncol,
        borderaxespad=0.2,
    )


def _line(ax, da, **kwargs):
    """Plot a 1D DataArray against its coordinate without xarray's labels."""
    da = da.squeeze(drop=True)
    return ax.plot(da[da.dims[0]].values, da.values, **kwargs)


def _stackedFigure(heights, figsize, cbar=True, hspace=0.04):
    """
    Panels sharing the time axis in a constrained layout; with `cbar` a
    narrow second column holds the color bars (unused ones are switched off
    by the caller).

    Returns
    -------
    fig, axs, caxs
        `caxs` is None without `cbar`.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize, layout="constrained")
    fig.get_layout_engine().set(
        h_pad=0.08,
        w_pad=0.05,
        hspace=hspace,
        wspace=0.02,
        rect=(0, 0.015, 1, 0.985),  # keep the status line at the bottom free
    )
    if cbar:
        gs = fig.add_gridspec(
            len(heights), 2, width_ratios=[1, 0.015], height_ratios=heights
        )
        axs = [fig.add_subplot(gs[ii, 0]) for ii in range(len(heights))]
        caxs = [fig.add_subplot(gs[ii, 1]) for ii in range(len(heights))]
        for cax in caxs:
            cax.axis("off")
    else:
        gs = fig.add_gridspec(len(heights), 1, height_ratios=heights)
        axs = [fig.add_subplot(gs[ii, 0]) for ii in range(len(heights))]
        caxs = None
    for ax in axs[1:]:
        ax.sharex(axs[0])
    return fig, axs, caxs


def _plotSizeDist(ax, cax, da, cbarLabel, **kwargs):
    """
    Size distribution (time x D_bins) as color plot, sizes converted to mm.

    Returns None without plotting if there are no data.
    """
    import matplotlib as mpl

    da = da.squeeze(drop=True)
    if not np.any(da.notnull()):
        return None
    if "norm" not in kwargs and "vmin" not in kwargs:
        # PSD: logarithmic
        vmax = float(da.max())
        kwargs["norm"] = mpl.colors.LogNorm(vmin=1, vmax=max(vmax, 10))
    da = da.assign_coords(D_bins=da.D_bins * 1e3)
    cax.axis("on")
    mesh = da.transpose("D_bins", "time").plot.pcolormesh(
        ax=ax,
        x="time",
        y="D_bins",
        cbar_ax=cax,
        cbar_kwargs={"label": cbarLabel},
        rasterized=True,
        **kwargs,
    )
    ax.set_ylim(bottom=0)
    ax.set_title("")
    ax.set_xlabel("")
    return mesh


def _plotLevel2Core(
    axPSD, caxPSD, axSize, axShape, axPerf, dat2, sel, aspectSel=None, matchScore=False
):
    """
    Panels shared by the level2 quicklooks: size distribution with N_tot,
    characteristic sizes, shape parameters and counts (with match score).

    Parameters
    ----------
    sel : dict
        Selection for the camera dimension (empty for single camera data).
    aspectSel : dict, optional
        Different selection for the aspect ratio (default `sel`).
    matchScore : bool
        Also show the mean match score.
    """
    if aspectSel is None:
        aspectSel = sel
    Dmax = dict(size_definition="Dmax", **sel)

    _plotSizeDist(
        axPSD,
        caxPSD,
        dat2.PSD.sel(**Dmax),
        "Particle size distribution [m$^{-4}$]",
    )
    axPSD.set_ylabel("Dmax [mm]")

    axN = axPSD.twinx()
    axN.spines["right"].set_visible(True)
    axN.grid(False)
    _line(axN, dat2.Ntot.sel(**Dmax), color="k", alpha=0.7, lw=1.2)
    axN.set_yscale("log")
    axN.set_ylim(1e0, 1e11)
    axN.set_ylabel("N$_{tot}$ [m$^{-3}$]")

    _line(axSize, dat2.Dmax_mean.sel(**sel) * 1e3, label="mean Dmax")
    _line(axSize, dat2.D32.sel(**Dmax) * 1e3, label="D32")
    _line(axSize, dat2.D43.sel(**Dmax) * 1e3, label="D43")
    axSize.set_ylabel("Size [mm]")
    axSize.legend(loc="upper right", ncol=3)

    _line(
        axShape,
        dat2.normalizedRimeMass_dist.sel(**Dmax).mean("D_bins"),
        label="normalized rime mass [-]",
    )
    _line(
        axShape,
        dat2.aspectRatio_mean.sel(fitMethod="cv2.fitEllipseDirect", **aspectSel),
        label="aspect ratio [-]",
    )
    axShape.set_ylabel("Shape [-]")
    axShape.legend(loc="upper right", ncol=2)

    _line(
        axPerf,
        dat2.counts.sel(**Dmax).sum("D_bins"),
        label="observed particles [1/min]",
    )
    axPerf.set_yscale("log")
    axPerf.set_ylabel("Particles [1/min]")
    handles = list(axPerf.get_lines())
    if matchScore:
        axM = axPerf.twinx()
        axM.spines["right"].set_visible(True)
        axM.grid(False)
        _line(axM, dat2.matchScore_mean.sel(**sel), color="C1", label="match score [-]")
        axM.margins(y=0.2)
        axM.set_ylabel("Match score [-]", color="C1")
        axM.tick_params(axis="y", colors="C1")
        handles += list(axM.get_lines())
    axPerf.legend(handles=handles, loc="upper right", ncol=2)

    for ax in [axSize, axShape, axPerf]:
        ax.margins(y=0.2)  # room for the legend above the curves
    for ax in [axPSD, axN, axSize, axShape, axPerf]:
        ax.set_title("")
        ax.set_xlabel("")


def _concatRotationFiles(rotFiles):
    """Concatenate metaRotation files, dropping duplicate file_starttimes."""
    # open_mfdataset does not work due to duplicate file_starttimes...
    rotDat = [xr.open_dataset(rotFile) for rotFile in rotFiles]
    if len(rotDat) == 0:
        return None
    rotDat = xr.concat(rotDat, "file_starttime")
    return rotDat.isel(
        file_starttime=np.unique(rotDat.file_starttime, return_index=True)[1]
    ).sortby("file_starttime")


@tools.loopify_with_camera(endYesterday=False)
def createLevel1detectQuicklookHourly(
    case,
    camera,
    config,
    hours=range(1, 24),
    version=__version__,
    container_width=200,
    container_height_max=600,
    nTiles=120,
    nRows=4,
    extra=1,
    readParticlesFromFiles=True,
    skipExisting=True,
    ffOut="default",
    timeStep="fixed",
    minBlur="config",
    minSize=8,
    omitLabel4small="config",
    timedelta=np.timedelta64(1, "h"),
):
    """
    Create hourly version of particle quicklook.

    Parameters
    ----------
    case : str
        Date string in format YYYYMMDD
    camera : str
        Camera identifier
    config : dict
        Configuration dictionary
    hours : range, optional
        Hours to process, by default range(1, 24)
    version : str, optional
        Version identifier, by default __version__
    container_width : int, optional
        Width of each tile container, by default 200
    container_height_max : int, optional
        Maximum height of each tile container, by default 600
    nTiles : int, optional
        Number of tiles per row, by default 120
    nRows : int, optional
        Number of rows in output, by default 4
    extra : int, optional
        Extra spacing between tiles, by default 1
    readParticlesFromFiles : bool, optional
        Whether to read particles from files, by default True
    skipExisting : bool, optional
        Whether to skip if output exists, by default True
    ffOut : str, optional
        Output file path, by default "default"
    timeStep : str, optional
        Time step method ('fixed' or 'variable'), by default "fixed"
    minBlur : float or str, optional
        Minimum blur threshold, by default "config"
    minSize : int or str, optional
        Minimum particle size, by default 8
    omitLabel4small : bool or str, optional
        Whether to omit labels for small particles, by default "config"
    timedelta : numpy.timedelta64, optional
        Time window for data selection, by default np.timedelta64(1, "h")

    Returns
    -------
    None
        Function processes files but doesn't return anything directly
    """
    import matplotlib.pyplot as plt

    for hh in hours:
        timestamp = f"{case}-{hh:02d}"
        f = createLevel1detectQuicklook(
            timestamp,
            camera,
            config,
            version=version,
            container_width=container_width,
            container_height_max=container_height_max,
            nTiles=nTiles,
            nRows=nRows,
            extra=extra,
            readParticlesFromFiles=readParticlesFromFiles,
            skipExisting=skipExisting,
            ffOut=ffOut,
            timeStep=timeStep,  # attempt to fill plot equally
            minBlur=minBlur,
            minSize=minSize,
            omitLabel4small=omitLabel4small,
            timedelta=timedelta,
            returnFig=False,
        )
        print(f)

    return


@tools.loopify_with_camera(endYesterday=False)
def createLevel1detectQuicklook(
    timestamp,
    camera,
    config,
    version=__version__,
    container_width=200,
    container_height_max=300,
    nTiles=60,
    nRows=4,
    extra=1,
    readParticlesFromFiles=True,
    skipExisting=True,
    ffOut="default",
    timeStep="variable",  # attempt to fill plot equally
    minBlur="config",
    minSize="config",
    omitLabel4small="config",
    timedelta=np.timedelta64(1, "D"),
    returnFig=True,
):
    """
    Create level1detect quicklook for a given timestamp.

    Parameters
    ----------
    timestamp : str
        Timestamp string in format YYYYMMDDHH or YYYYMMDD
    camera : str
        Camera identifier
    config : dict
        Configuration dictionary
    version : str, optional
        Version identifier, by default __version__
    container_width : int, optional
        Width of each tile container, by default 200
    container_height_max : int, optional
        Maximum height of each tile container, by default 300
    nTiles : int, optional
        Number of tiles per row, by default 60
    nRows : int, optional
        Number of rows in output, by default 4
    extra : int, optional
        Extra spacing between tiles, by default 1
    readParticlesFromFiles : bool, optional
        Whether to read particles from files, by default True
    skipExisting : bool, optional
        Whether to skip if output exists, by default True
    ffOut : str, optional
        Output file path, by default "default"
    timeStep : str, optional
        Time step method ('fixed' or 'variable'), by default "variable"
    minBlur : float or str, optional
        Minimum blur threshold, by default "config"
    minSize : int or str, optional
        Minimum particle size, by default "config"
    omitLabel4small : bool or str, optional
        Whether to omit labels for small particles, by default "config"
    timedelta : numpy.timedelta64, optional
        Time window for data selection, by default np.timedelta64(1, "D")
    returnFig : bool, optional
        Whether to return the figure, by default True

    Returns
    -------
    tuple
        (output_file_path, figure) if returnFig=True, otherwise just output_file_path
    """
    import cv2
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import pandas as pd
    from PIL import Image, ImageDraw, ImageFont
    from tqdm import tqdm

    if minBlur == "config":
        minBlur = config["level1detectQuicklook"]["minBlur"]
    if minSize == "config":
        minSize = config["level1detectQuicklook"]["minSize"]
    if omitLabel4small == "config":
        omitLabel4small = config["level1detectQuicklook"]["omitLabel4small"]

    ff = files.FindFiles(timestamp, camera, config, version)
    if ffOut == "default":
        ffOut = ff.quicklook.level1detect

    site = config["site"]

    particlesPloted = 0

    if skipExisting and tools.checkForExisting(
        ffOut,
        level0=ff.listFiles("level0"),
        events=ff.listFiles("metaEvents"),
        parents=ff.listFilesExt("level1detect"),
    ):
        return None, None

    if site != "mosaic":
        noLevel0Yet = (len(ff.listFiles("level0")) == 0) and (
            len(ff.listFiles("level0status")) == 0
        )
        if noLevel0Yet and ff.isDataTransferPending("level0"):
            print("NO DATA YET (TRANSFERRED?)", ffOut)
            return None, None

    if len(ff.listFiles("metaFrames")) > len(ff.listFiles("level0")):
        print("DATA TRANSFER INCOMPLETE ", ffOut)
        print(
            len(ff.listFiles("level0")),
            "of",
            len(ff.listFiles("metaFrames")),
            "transmitted",
        )
        return None, None

    if (len(ff.listFilesExt("level1detect")) == 0) and (
        len(ff.listFiles("level0")) > 0
    ):
        print("NO DATA YET ", ffOut)
        return None, None

    if not ff.isCompleteL1detect:
        print(
            "NOT COMPLETE YET %i of %i L1detect %s "
            % (
                len(ff.listFilesExt("level1detect")),
                len(ff.listFiles("level0txt")),
                ffOut,
            )
        )

        #         if (len(ff.listFilesExt("level1detect")) == len(ff.listFiles("level0"))):
        #             afshgsa
        return None, None

    #     else:

    total_width = (container_width + extra) * nTiles // nRows
    max_height = (20 + extra + container_height_max) * nRows + 60

    # let use a matplotlib font becuase we can be sure it is there
    mpl_data_dir = os.path.dirname(mpl.matplotlib_fname())
    mpl_ttf_dir = os.path.join(mpl_data_dir, "fonts", "ttf")
    font = ImageFont.truetype(f"{mpl_ttf_dir}/DejaVuSans.ttf", 35)
    fontL = ImageFont.truetype(f"{mpl_ttf_dir}/DejaVuSans.ttf", 16)

    print("RUNNING open files ", ffOut, len(ff.listFiles("metaFrames")))

    dats2 = []
    l1Files = ff.listFilesWithNeighbors("level1detect")

    nParticles = 0

    for fname2 in tqdm(l1Files):
        fname1 = fname2.replace("level1detect", "metaFrames")

        try:
            dat2 = xr.open_dataset(fname2)
        except FileNotFoundError:
            if os.path.isfile(f"{fname2}.nodata"):
                continue
            elif os.path.isfile(f"{fname2}.broken.txt"):
                continue
            elif os.path.isfile(f"{fname2}.notenoughframes"):
                continue
            else:
                raise FileNotFoundError(fname2)

        nParticles += len(dat2.pid)

        dat2 = dat2[
            ["Dmax", "blur", "record_time", "record_id", "position_upperLeft", "Droi"]
        ]
        # it is more efficient to load the data now in comparison to after isel
        dat2 = dat2.load()

        dat2 = dat2.isel(pid=((dat2.blur > minBlur) & (dat2.Dmax > minSize)))

        if len(dat2.pid) == 0:
            continue
        dat2 = dat2[["record_time", "record_id", "Droi", "position_upperLeft"]]

        dat2 = dat2.expand_dims(dict(file=[fname1]))
        #     dat2 = dat2.set_coords(dict(file = fname2))
        dat2 = dat2.stack(fpid=("file", "pid"))

        dats2.append(dat2)

    print("opened")
    new_im = Image.new("RGB", (total_width, max_height), color=(255, 255, 255))

    if len(dats2) == 0:
        draw = ImageDraw.Draw(new_im)
        draw.text(
            (total_width // 3, max_height // 3), "no raw data", (0, 0, 0), font=font
        )

        nParticles = 0

    else:
        limDat = xr.concat(dats2, dim="fpid")
        limDat = limDat.isel(
            fpid=(limDat.record_time >= ff.datetime64)
            & (limDat.record_time < (ff.datetime64 + timedelta))
        )
        #         limDat = dats2
        print("merged")

        if len(limDat.fpid) == 0:
            print("TOO FEW DATA ", ffOut)

            draw = ImageDraw.Draw(new_im)
            draw.text(
                (total_width // 3, max_height // 3),
                "no data recorded",
                (0, 0, 0),
                font=font,
            )

            nParticles = 0

        else:
            print("Total number of particles for plotting %i" % len(limDat.fpid))

            if timeStep == "variable":
                timeSteps = np.percentile(
                    limDat.record_time, np.linspace(0, 100, nTiles + 1)
                )
            elif timeStep == "fixed":
                timeSteps = np.array(
                    pd.date_range(
                        start=ff.datetime64,
                        periods=nTiles + 1,
                        end=ff.datetime64 + timedelta,
                    )
                )
            else:
                raise ValueError("do not understand timeStep")
            mosaics = []

            videos = {}
            for tt, (t1, t2) in enumerate(zip(timeSteps[:-1], timeSteps[1:])):
                if tt == len(timeSteps) - 2:
                    whereCond = limDat.record_time >= t1
                else:
                    whereCond = (limDat.record_time >= t1) & (limDat.record_time < t2)

                thisDat = limDat.where(whereCond).dropna("fpid")
                totalArea = 0

                # select pids randomly, figure out how much we need, and sort them again
                pids = deepcopy(thisDat.fpid.values)
                fnames = thisDat.fpid.file.values
                particleImages = {}
                for fname in np.unique(fnames):
                    fn = files.FilenamesFromLevel(fname, config)
                    # tarRoot = fn.fname.imagesL1detect.split("/")[-1].replace(".tar.bz2","")
                    # particleImages[fname] = (tools.imageTarFile.open(fn.fname.imagesL1detect, "r:bz2"), tarRoot)
                    # print(fn.fname.imagesL1detect)

                    for imageFile in [
                        fn.fname.imagesL1detect,
                        fn.fname.imagesL1detect.replace(".bin", ".zip"),
                    ]:
                        try:
                            particleImages[fname] = tools.imageZipFile(
                                imageFile, mode="r"
                            )
                        except FileNotFoundError:
                            log.warning(f"did not find {imageFile}")
                        except tools.zipfile.BadZipFile:
                            log.warning(f"is broken {imageFile}")
                        else:
                            break
                nPids = len(pids)
                rng = np.random.default_rng(tt)
                rng.shuffle(pids)

                containerSize = container_width * container_height_max

                if nPids < 5:  # fo very few particles I want to see them all!
                    nParticlesNeeded = len(pids)
                else:
                    try:
                        nParticlesNeeded = (
                            np.where(
                                (thisDat.sel(fpid=pids).Droi + 1)
                                .prod("dim2D")
                                .cumsum("fpid")
                                / containerSize
                                > 0.95
                            )[0][0]
                            + 1
                        )
                    except IndexError:
                        nParticlesNeeded = nPids

                pids = np.sort(pids[:nParticlesNeeded])
                print(tt, "/", nTiles, t1, t2, nParticlesNeeded, "of", nPids)
                particlesPloted += nParticlesNeeded
                ims = []

                videos = {}

                for fname, pid in pids:
                    # basenameImg = fname.split('/')[-1]

                    if not readParticlesFromFiles:
                        f1 = files.FilenamesFromLevel(fname, config)
                        thisfname_lv0 = f1.fname.level0

                        if thisfname_lv0 not in videos.keys():
                            for k in videos.keys():
                                videos[k].release()

                            videos[thisfname_lv0] = av.VideoReaderMeta(
                                thisfname_lv0, fname, safeMode=True
                            )
                    #             print('opened %s'%thisfname_lv0)

                    particle = thisDat.sel(fpid=(fname, pid))
                    kk = int(particle.record_id.values)
                    background = 0
                    if not readParticlesFromFiles:
                        _, frame1, _ = videos[thisfname_lv0].getFrameByIndex(kk)

                        if frame1 is None:
                            continue

                        w, h = particle.Droi.values.astype(int)
                        x, y = particle.position_upperLeft.values.astype(int)
                        if len(frame1.shape) == 3:
                            frame1 = frame1[:, :, 0]
                        im = frame1[
                            y
                            + config.level1detect.height_offset : y
                            + config.level1detect.height_offset
                            + h,
                            x : x + w,
                        ]
                    else:
                        pidStr = "%07i" % pid
                        # imName = '%s.png' % (pidStr)
                        # imfname = '%s/%s/%s' % (particleImages[fname][1],pidStr[:4], imName)
                        try:
                            # im = particleImages[fname][0].extractimage(imfname)
                            im = particleImages[fname].extractnpy(pidStr)
                        except KeyError:
                            log.error(f"Particle image {pidStr} NOT FOUND ")
                            continue
                        # apply alpha channel
                        # im[...,0][im[...,1] == 0] = background
                        # drop alpha channel
                        im = im[..., 0]

                    # im = av.doubleDynamicRange(im, offset=2)

                    im = np.pad(im, [(0, 1), (0, 1)])
                    try:
                        fid = np.where(fname == np.array(ff.listFiles("metaFrames")))[
                            0
                        ][0]
                    except:
                        fid = -1
                    text = np.full((100, 100), background, dtype=np.uint8)

                    textStr = "%i.%i" % (fid, pid)
                    # print(textStr)
                    text = cv2.putText(
                        text,
                        textStr,
                        (0, 50),
                        cv2.FONT_HERSHEY_PLAIN,
                        0.75,
                        255,
                        1,
                    )
                    text = _crop(text)

                    y1, x1 = im.shape
                    y2, x2 = text.shape

                    # only add label if large enough
                    if (omitLabel4small == "all") or (
                        (omitLabel4small == True) and (x1 < x2)
                    ):
                        imT = im
                    else:
                        y3 = y1 + y2
                        x3 = max(x1, x2)
                        imT = np.full((y3, x3), background, dtype=np.uint8)
                        imT[:y1, :x1] = im
                        imT[y1:, :x2] = text
                    ims.append(imT)
                    totalArea += np.prod(imT.shape)

                for fname in fnames:
                    try:
                        particleImages[fname].close()
                    except KeyError:
                        pass

                # make tile
                images = [Image.fromarray(im) for im in ims]
                if len(images) == 0:
                    mosaic = (
                        np.ones(
                            (container_height_max, container_width, 3), dtype=np.uint8
                        )
                        * 0
                    )
                else:
                    mosaic = Packer_patched(images).pack(
                        container_width=container_width,
                        container_height_max=container_height_max,
                    )
                    mosaic = np.array(mosaic)

                    if container_width > mosaic.shape[1]:
                        mosaic = np.pad(
                            mosaic,
                            [(0, 0), (0, container_width - mosaic.shape[1]), (0, 0)],
                        )

                    # sometimes container is too large...
                    mosaic = mosaic[:container_height_max, :container_width]

                label = Image.fromarray(
                    np.ones((20, mosaic.shape[1], 3), dtype=np.uint8) * 255
                )
                drawL = ImageDraw.Draw(label)
                textStr = "%s-%s" % (
                    str(t1).split(".")[0].split("T")[1],
                    str(t2).split(".")[0].split("T")[1],
                )
                if nParticlesNeeded != nPids:
                    textStr += " (R)"
                drawL.text((0, 0), textStr, (0, 0, 0), font=fontL)

                mosaic = Image.fromarray(np.vstack((label, mosaic)))
                #             display(mosaic)
                mosaics.append(mosaic)

            nCols = nTiles // nRows

            widths, heights = zip(*(i.size for i in mosaics))

            for nRow in range(nRows):
                x_offset = 0

                for im in mosaics[nCols * (nRow) : nCols * (nRow + 1)]:
                    new_im.paste(im, (x_offset, max(heights) * nRow + 50))
                    x_offset += im.size[0] + extra

                # x_offset = 0
            # for im in mosaics[len(mosaics)//nRows:]:
            #   new_im.paste(im, (x_offset,max(heights) +50))
            #   x_offset += im.size[0] + extra

    tenmm = 1e6 / (1 / config.calibration.slope) / 100

    if ff.hour == "":
        title = (
            "%s-%s-%s %s %s, size threshold for plotting: %i px (%.2f mm), %i of %i larger detections plotted, 10 mm = %.1f px ="
            % (
                ff.year,
                ff.month,
                ff.day,
                tools.nicerNames(camera),
                config["name"],
                minSize,
                minSize * (1 / config.calibration.slope) * 1e-6 * 1000,
                particlesPloted,
                nParticles,
                tenmm,
            )
        )
    else:
        title = (
            "%s-%s-%sT%s %s %s, size threshold for plotting: %i px (%.2f mm), %i of %i larger detections plotted, 10 mm = %.1f px ="
            % (
                ff.year,
                ff.month,
                ff.day,
                ff.hour,
                tools.nicerNames(camera),
                config["name"],
                minSize,
                minSize * (1 / config.calibration.slope) * 1e-6 * 1000,
                particlesPloted,
                nParticles,
                tenmm,
            )
        )

    # new_im = cv2.putText(np.array(new_im), title,
    #                      (0, 45), FONT, FONT_SCALE, (0, 0, 0), FONT_THICKNESS,)
    # (label_width, label_height), baseline = cv2.getTextSize(
    #     title, FONT, FONT_SCALE, FONT_THICKNESS)

    draw = ImageDraw.Draw(new_im)
    draw.text((0, 0), title, (0, 0, 0), font=font)
    width = draw.textlength(title, font=font)
    # Call textbbox to get the bounding box
    bbox = draw.textbbox((0, 0), title, font=font)

    # Extract top and bottom values from the returned tuple
    left, top, right, bottom = bbox
    height = top - bottom
    # width, height = draw.textsize(title, font=font)##Depricated
    draw.line((width + 15, 30, width + 15 + round(tenmm), 30), fill=0, width=5)

    tools.createParentDir(ffOut, mode=config.dirMode)
    new_im.save(ffOut)
    print("SAVED ", ffOut)

    if returnFig:
        return ffOut, new_im
    else:
        return ffOut


class Packer_patched(packer.Packer):
    """
    Patched image_packer routine that works without files.

    This class patches the image_packer library to work with in-memory images
    rather than requiring temporary files.

    Attributes
    ----------
    _uid_to_filepath : dict
        Mapping of unique identifiers to image objects
    _pieces : list
        List of image pieces to pack
    _has_alpha : bool
        Flag indicating whether images have alpha channels
    """

    def __init__(self, images):
        # Ensure plugins are fully loaded so that Image.EXTENSION is populated.
        from PIL import Image, ImageDraw, ImageFont

        Image.init()

        self._uid_to_filepath = dict()
        self._pieces = list()
        self._has_alpha = False

        for im in images:
            width = im.width
            height = im.height
            uid = uuid.uuid4()
            self._uid_to_filepath[uid] = deepcopy(im)
            self._pieces.append(
                packer.blf.Piece(uid=uid, size=packer.blf.Size(width, height))
            )
            if im.mode in ("RGBA", "LA") or (
                im.mode == "P" and "transparency" in im.info
            ):
                self._has_alpha = True

    def pack(self, container_width, options=None, container_height_max=100):
        """Packs multiple images of different sizes or formats into one image.

        Parameters
        ----------
        container_width : int
            Width of the container image
        options : dict, optional
            Packing options, by default None
        container_height_max : int, optional
            Maximum height of the container, by default 100

        Returns
        -------
        PIL.Image.Image
            Packed image
        """
        if options is None:
            options = self._DEFAULT_OPTIONS
        else:
            options = {
                key: options[key] if key in options else self._DEFAULT_OPTIONS[key]
                for key in self._DEFAULT_OPTIONS.keys()
            }

        margin_ = options["margin"]
        assert isinstance(margin_, tuple) and len(margin_) == 4

        if options["enable_vertical_flip"]:
            margin = packer.blf.Thickness(
                top=margin_[2], right=margin_[1], bottom=margin_[0], left=margin_[3]
            )
        else:
            margin = packer.blf.Thickness(
                top=margin_[0], right=margin_[1], bottom=margin_[2], left=margin_[3]
            )

        blf_options = {
            "margin": margin,
            "collapse_margin": options["collapse_margin"],
            "enable_auto_size": options["enable_auto_size"],
            "force_pow2": options["force_pow2"],
        }

        container_width, container_height, regions = packer.blf_solver.solve(
            pieces=self._pieces, container_width=container_width, options=blf_options
        )

        compImage = self._save_image(
            container_width=container_width,
            container_height=container_height_max,
            regions=regions,
            options=options,
        )
        return compImage

    def _save_image(self, container_width, container_height, regions, options):
        from PIL import Image, ImageDraw, ImageFont

        bg_color_ = options["bg_color"]
        assert isinstance(bg_color_, tuple) and (3 <= len(bg_color_) <= 4)
        bg_color = tuple(int(channel * 255.0) for channel in bg_color_)
        if len(bg_color) == 3:
            bg_color += (255,)

        if self._has_alpha:
            blank_image = Image.new(
                mode="RGBA", size=(container_width, container_height), color=bg_color
            )
        else:
            blank_image = Image.new(
                mode="RGB",
                size=(container_width, container_height),
                color=bg_color[0:3],
            )

        enable_vertical_flip = options["enable_vertical_flip"]

        for region in regions:
            x = region.left
            if enable_vertical_flip:
                y = region.bottom
            else:
                y = container_height - region.top

            im = self._uid_to_filepath.get(region.uid)

            blank_image.paste(im=im, box=(x, y))

        return blank_image


@tools.loopify_with_camera(endYesterday=False)
def level0Quicklook(case, camera, config, version=__version__, skipExisting=True):
    """
    Create level0 quicklook for a given case and camera.

    Parameters
    ----------
    case : str
        Date string in format YYYYMMDD
    camera : str
        Camera identifier
    config : dict
        Configuration dictionary
    version : str, optional
        Version identifier, by default __version__
    skipExisting : bool, optional
        Whether to skip if output exists, by default True

    Returns
    -------
    tuple
        (output_file_path, None) if successful, (None, None) otherwise
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    global metaDats
    metaDats = []

    # get level 0 file names
    ff = files.FindFiles(case, camera, config, version)
    fOut = ff.quicklook.level0
    tools.createParentDir(fOut, mode=config.dirMode)

    if skipExisting and tools.checkForExisting(
        fOut,
        events=ff.listFiles("metaEvents"),
    ):
        return None, None

    print(case, camera, fOut)

    if len(ff.listFiles("level0txt")) == 0 and len(ff.listFiles("level0status")) == 0:
        print(case, "no data")

        return None, None

    fnames = []
    cases = []
    for fname in ff.listFiles("level0jpg"):
        fn = files.Filenames(fname, config, version)
        if fn.case.endswith("0000") or fn.case.endswith("5959"):
            fnames.append(fname)
            cases.append(fn.case)
    fnames = fnames[:24]
    cases = cases[:24]
    execStr = ""
    for f, c in zip(fnames, cases):
        execStr += f"-label {c} {f} "
    if len(fnames) > 0:
        tools.execute_stdout(
            f"montage {execStr} -geometry 225x180+1+1 -tile 6x4 -quality 70% -title '{config.name} {camera} {case}' {fOut}"
        )
    else:
        print("no input files")
        return None
    tools.copyCurrentQuicklook("level0", ff)
    return fOut


@tools.loopify_with_camera(endYesterday=False)
@_styled
def metaFramesQuicklook(
    case, camera, config, version=__version__, skipExisting=True, plotCompleteOnly=True
):
    """
    Create metaFrames quicklook for a given case and camera.

    Parameters
    ----------
    case : str
        Date string in format YYYYMMDD
    camera : str
        Camera identifier
    config : dict
        Configuration dictionary
    version : str, optional
        Version identifier, by default __version__
    skipExisting : bool, optional
        Whether to skip if output exists, by default True
    plotCompleteOnly : bool, optional
        Whether to only plot if data is complete, by default True

    Returns
    -------
    tuple
        (output_file_path, matplotlib.figure.Figure) if successful, (None, None) otherwise
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    global metaDats
    metaDats = []

    # get level 0 file names
    ff = files.FindFiles(case, camera, config, version)
    fOut = ff.quicklook.metaFrames

    if skipExisting and tools.checkForExisting(
        fOut,
        events=ff.listFiles("metaEvents"),
        parents=ff.listFiles("metaFrames"),
    ):
        return None, None

    if plotCompleteOnly and not ff.isCompleteMetaFrames:
        print(
            "NOT COMPLETE YET %i of %i MetaFrames %s"
            % (
                len(ff.listFilesExt("metaFrames")),
                len(ff.listFiles("level0txt")),
                ff.fnamesPattern.metaFrames,
            )
        )
        return None, None

    print(case, camera, fOut)

    if len(ff.listFiles("level0txt")) == 0 and len(ff.listFiles("level0status")) == 0:
        print(case, "no data")

        return None, None

    try:
        level0status = ff.listFiles("level0status")[0]
    except IndexError:
        level0status = None

    print("reading events")
    if len(ff.listFiles("metaEvents")) == 0:
        if len(ff.listFilesExt("metaEvents")) == 0:
            log.error(f"event data not found")
        else:
            log.warning(f"{case} metaEvents is nodata/broken, skipping quicklook")
        return None, None
    try:
        events = xr.open_dataset(ff.listFiles("metaEvents")[0])
    except:
        log.error(f'{ff.listFiles("metaEvents")[0]} broken')
        return None, None

    if len(events.data_vars) == 0:
        log.error(f'{ff.listFiles("metaEvents")[0]} empty')
        return None, None

    if not ff.isCompleteMetaFrames:
        log.warning(f"meta frames not complete")
        return None, None

    # iterate to get meta data
    # if len(ff.listFiles("metaFrames")) > 0:
    #     print("reading metaFrames")
    # try: #faster?
    #     metaDats = xr.open_mfdataset(ff.listFiles("metaFrames"), combine="by_coords")
    # except ValueError:
    #     metaDats = xr.open_mfdataset(ff.listFiles("metaFrames"), combine="nested")

    # else:
    #     print(f'metaFrames not found')
    #     return None, None
    print("reading metaFrames")
    metaDats = []
    for fname1 in ff.listFiles("metaFrames"):
        metaDat = xr.open_dataset(fname1)
        keys = ["capture_time"]
        if "queue_size" in metaDat.variables:
            keys.append("queue_size")
        else:
            keys.append("record_time")
        metaDat = metaDat[keys]
        metaDats.append(deepcopy(metaDat))
        metaDat.close()
    if len(metaDats) == 0:
        metaDats = None
    else:
        metaDats = xr.concat(metaDats, dim="capture_time")
        metaDats = metaDats.sortby("capture_time")

    isIdle = (
        (events.event == "sleep-trigger") | (events.event == "stop-trigger")
    ).values
    idleStarts = events.file_starttime.values[isIdle]
    fileStep = _timeStep(events.file_starttime.values)
    isBlocked = events.blocking.dropna("file_starttime").sel(blockingThreshold=50) > 0.1
    isBlocked = isBlocked.file_starttime.values[isBlocked.values]

    print("plotting")
    # plotting
    fig, axs, _ = _stackedFigure([1, 1, 1], figsize=(13, 9), cbar=False, hspace=0.12)
    ax1, ax2, ax3 = axs
    _suptitle(fig, f"metaFrames  ·  {camera.replace('_', ' ')}", ff, config)

    if metaDats is not None:
        if "queue_size" in keys:
            ax1.plot(metaDats.capture_time.values, metaDats.queue_size.values, lw=1)
            ax1.set_ylabel("frames in queue")
            _panelTitle(ax1, "Queue size")
        else:
            delay = (metaDats.record_time - metaDats.capture_time).astype(int) / 1e9
            ax1.plot(metaDats.capture_time.values, delay.values, lw=1)
            ax1.set_ylabel("record delay [s]")
            _panelTitle(ax1, "Record delay")
    ax1.set_ylim(bottom=0)

    nominalFrames = config.fps * 5 * 60
    if metaDats is not None:
        frames = metaDats.capture_time.notnull().resample(capture_time="5min").sum()
        ax2.plot(frames.capture_time.values, frames.values, lw=1)
    ax2.axhline(nominalFrames, color="0.4", ls="--", lw=0.8)
    ax2.set_ylim(0, nominalFrames * 1.05)
    _panelTitle(ax2, "Total frames / 5 min")
    ax2.set_ylabel("frames")

    blocking = events.blocking.isel(
        blockingThreshold=range(0, len(events.blockingThreshold), 2)
    ).dropna("file_starttime")
    if len(blocking) != 0:
        cmap = plt.get_cmap("viridis")
        for ii, threshold in enumerate(blocking.blockingThreshold.values):
            ax3.plot(
                blocking.file_starttime.values,
                blocking.sel(blockingThreshold=threshold).values,
                color=cmap(ii / max(len(blocking.blockingThreshold) - 1, 1)),
                lw=1.2,
                label=f"{threshold:g}",
            )
        ax3.legend(
            title="threshold", title_fontsize=9, loc="upper right", ncol=4, fontsize=8.5
        )
    ax3.set_ylim(0, 1)
    _panelTitle(ax3, "Blocking")
    ax3.set_ylabel("fraction of pixels\nbelow threshold")

    handles = [
        _shadeSpans(axs, isBlocked, fileStep, "red", "blocked", alpha=0.25),
        _shadeSpans(axs, idleStarts, fileStep, "orange", "idle", alpha=0.35),
        _markRestarts(axs, _restartTimes(events)),
    ]
    _formatDayAxes(axs, ff)
    _legendAbove(ax1, [h for h in handles if h is not None], ncol=3)
    fig.align_ylabels(axs)

    # _statusText(fig, ff.listFiles("metaFrames"), config)
    tools.savefig(fig, config, fOut, fnames=ff.listFiles("metaFrames"), logoAnchor=ax1)
    tools.copyCurrentQuicklook("metaFrames", ff)

    if metaDats is not None:
        metaDats.close()
    events.close()

    return fOut, fig


@tools.loopify
@_styled
def createLevel1matchQuicklook(
    case,
    config,
    skipExisting=True,
    version=__version__,
    plotCompleteOnly=True,
    returnFig=True,
):
    """
    Create level1match quicklook for a given case.

    Parameters
    ----------
    case : str
        Date string in format YYYYMMDD
    config : dict
        Configuration dictionary
    skipExisting : bool, optional
        Whether to skip if output exists, by default True
    version : str, optional
        Version identifier, by default __version__
    plotCompleteOnly : bool, optional
        Whether to only plot if data is complete, by default True
    returnFig : bool, optional
        Whether to return the figure, by default True

    Returns
    -------
    tuple
        (output_file_path, matplotlib.figure.Figure) if returnFig=True,
        otherwise (output_file_path, None)
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import pandas as pd

    resample = "5min"  # 5 mins

    # find files
    fl = files.FindFiles(case, config["leader"], config, version)
    ff = files.FindFiles(case, config["follower"], config, version)
    # get level 0 file names
    fOut = fl.quicklook.level1match

    if skipExisting and tools.checkForExisting(
        fOut,
        level0=fl.listFiles("level0") + ff.listFiles("level0"),
        events=fl.listFiles("metaEvents") + ff.listFiles("metaEvents"),
        parents=fl.listFilesExt("level1match"),
    ):
        return None, None

    noLevel0Yet = (len(fl.listFiles("level0")) == 0) and (
        len(fl.listFiles("level0status")) == 0
    )
    if noLevel0Yet and fl.isDataTransferPending("level0"):
        print(
            "NO DATA YET (TRANSFERRED?)",
            fl.fnamesPattern["level0"],
            fl.quicklook.level1match,
        )
        return None, None

    if (len(fl.listFilesExt("level1match")) == 0) and (len(fl.listFiles("level0")) > 0):
        print(
            "NO DATA YET ", fl.fnamesPatternExt["level1match"], fl.quicklook.level1match
        )
        return None, None

    if plotCompleteOnly and not fl.isCompleteL1match:
        print(
            "NOT COMPLETE YET %i of %i L1match %s"
            % (
                len(fl.listFilesExt("level1match")),
                len(fl.listFiles("level0txt")),
                fl.fnamesPattern.level1match,
            )
        )
        return None, None

    fnames1M = fl.listFiles("level1match")
    if len(fnames1M) == 0:
        print("No precipitation", case, fl.fnamesPattern.level1match)
        fig = _placeholderFigure("level1match", fl, config, "no precipitation")
        tools.savefig(fig, config, fOut)
        return fOut, fig

    print("Running", fOut)

    fnames1DL = fl.listFiles("level1detect")
    ff = files.FindFiles(case, config.follower, config, version)
    fnames1DF = ff.listFiles("level1detect")

    assert len(fnames1DL) > 0
    assert len(fnames1DF) > 0

    datMfull = tools.open_mflevel1match(
        fnames1M,
        config,
        datVars=[
            "Dmax",
            "capture_time",
            "matchScore",
            "position3D_centroid",
            "position_upperLeft",
            "Droi",
            "camera_theta",
            "camera_phi",
            "camera_Ofz",
        ],
    )
    datM = datMfull.isel(fpair_id=(datMfull.matchScore >= config.quality.minMatchScore))

    if len(datM.fpair_id) <= 1:
        print("No precipitation (2)", case, fl.fnamesPattern.level1match)
        fig = _placeholderFigure("level1match", fl, config, "no precipitation (2)")
        tools.savefig(fig, config, fOut)
        return fOut, fig

    datDL = tools.open_mflevel1detect(
        fnames1DL, config, skipFixes="all", datVars=["Dmax", "capture_time"]
    )
    datDF = tools.open_mflevel1detect(
        fnames1DF, config, skipFixes="all", datVars=["Dmax", "capture_time"]
    )

    fig, axs, caxs = _stackedFigure(
        [2, 2, 2, 1, 1, 1, 1, 1, 1],
        figsize=(13, 17),
        hspace=0.12,
    )
    ax = axs
    cax = caxs
    cax[1].axis("on")
    cax[2].axis("on")

    fig.suptitle(
        f"VISSS level1match  ·  {case}  ·  {config.name}",
        fontsize=15,
        fontweight="bold",
    )

    Dmax = datM.Dmax.mean("camera").values
    _, rs = _plotVar(
        Dmax,
        datM.capture_time,
        ax[0],
        "Counts [-]",
        func="count",
        label="matched",
        color="C1",
        resample=resample,
    )
    _, rs1 = _plotVar(
        datDL.Dmax,
        datDL.capture_time,
        ax[0],
        "Counts [-]",
        func="ratio",
        label="ratio leader",
        color=_LEADER_COLOR,
        ratiovar=rs,
        resample=resample,
    )
    _, rs1 = _plotVar(
        datDF.Dmax,
        datDF.capture_time,
        ax[0],
        "Counts [-]",
        func="ratio",
        label="ratio follower",
        color=_FOLLOWER_COLOR,
        ratiovar=rs,
        resample=resample,
    )
    ax[0].set_yscale("log")
    ax[0].margins(y=0.2)
    ax[0].add_artist(ax[0].legend(loc="upper left", ncol=3))

    bins = np.logspace(0, 2.5, 21)
    _, _ = _plot2dhist(
        Dmax, datM.capture_time, ax[1], cax[1], bins, ylabel="Dmax [px]", cbarlabel="%"
    )

    matchScore = datMfull.matchScore.values
    bins = np.logspace(-10, 0, 41)
    _, _ = _plot2dhist(
        matchScore,
        datMfull.capture_time,
        ax[2],
        cax[2],
        bins,
        ylabel="match score [-]",
        cbarlabel="%",
    )
    ax[2].axhline(config.quality.minMatchScore, color="k", lw=1, ls="--")

    zDiff = datM.position_upperLeft.sel(dim2D="y").diff("camera").values.squeeze()
    _, rs = _plotVar(
        zDiff,
        datM.capture_time,
        ax[3],
        "z difference [px]",
        axhline=0,
        resample=resample,
    )

    hDiff = datM.Droi.sel(dim2D="y").diff("camera").values.squeeze()
    _, _ = _plotVar(
        hDiff,
        datM.capture_time,
        ax[4],
        "h difference [px]",
        axhline=0,
        resample=resample,
    )

    tDiff = datM.capture_time.diff("camera").values.squeeze().astype(int) * 1e-9
    _, _ = _plotVar(
        tDiff,
        datM.capture_time,
        ax[5],
        "t difference [s]",
        axhline=0,
        resample=resample,
    )

    try:
        defaultRotation, prevTime = tools.getPrevRotationEstimate(
            ff.datetime64, "transformation", config
        )
    except RuntimeError as e:
        # e.g. case predates every entry in config.rotate (a legitimate
        # deployment date with no rotation prior configured yet) --
        # the reference axhlines below are a plotting nicety, not
        # essential, but the gap itself should stay visible in the log
        # rather than silently vanishing.
        log.warning(
            f"{case}: no rotation prior available for the theta/phi/Ofz "
            f"reference lines ({e}); plotting without them"
        )
        defaultRotation = None

    theta = datM.camera_theta.sel(camera_rotation="mean").values.squeeze()
    _, _ = _plotVar(
        theta,
        datM.capture_time.isel(camera=0),
        ax[6],
        "theta [°]",
        axhline=defaultRotation["camera_theta"] if defaultRotation else None,
        resample=resample,
    )
    phi = datM.camera_phi.sel(camera_rotation="mean").values.squeeze()
    _, _ = _plotVar(
        phi,
        datM.capture_time.isel(camera=0),
        ax[7],
        "phi [°]",
        axhline=defaultRotation["camera_phi"] if defaultRotation else None,
        resample=resample,
    )
    Ofz = datM.camera_Ofz.sel(camera_rotation="mean").values.squeeze()
    _, _ = _plotVar(
        Ofz,
        datM.capture_time.isel(camera=0),
        ax[8],
        "Ofz [px]",
        axhline=defaultRotation["camera_Ofz"] if defaultRotation else None,
        resample=resample,
    )

    timeIndex1 = pd.date_range(
        start=case,
        end=fl.datetime64 + np.timedelta64(1, "D"),
        freq=resample,
        inclusive="both",
    )

    blowingSnowL, nDetectedL = tools.identifyBlockedBlowingSnowData(
        fl.listFilesWithNeighbors("metaDetection"), config, timeIndex1, "match"
    )
    blowingSnowF, nDetectedF = tools.identifyBlockedBlowingSnowData(
        ff.listFilesWithNeighbors("metaDetection"), config, timeIndex1, "match"
    )
    blowingSnowL = blowingSnowL.time.values[
        (blowingSnowL > config.quality.blowingSnowFrameThresh).values
    ]
    blowingSnowF = blowingSnowF.time.values[
        (blowingSnowF > config.quality.blowingSnowFrameThresh).values
    ]

    observationsRatio = tools.compareNDetected(nDetectedL, nDetectedF)
    obervationsDiffer = observationsRatio < config.quality.obsRatioThreshold
    obervationsDiffer = obervationsDiffer.time.values[obervationsDiffer.values]

    eventDatL = xr.open_dataset(fl.listFiles("metaEvents")[0])
    lBlocked = (
        eventDatL.blocking.sel(blockingThreshold=50) > config.quality.blockedPixThresh
    )
    lBlocked = lBlocked.file_starttime.values[lBlocked.values]
    eventDatF = xr.open_dataset(ff.listFiles("metaEvents")[0])
    fBlocked = (
        eventDatF.blocking.sel(blockingThreshold=50) > config.quality.blockedPixThresh
    )
    fBlocked = fBlocked.file_starttime.values[fBlocked.values]

    fileStep = _timeStep(eventDatL.file_starttime.values)
    resampleStep = np.timedelta64(5, "m")
    handles = [
        _shadeSpans(
            ax,
            obervationsDiffer,
            resampleStep,
            "magenta",
            "observed # of particles differ",
        ),
        _shadeSpans(ax, lBlocked, fileStep, _LEADER_COLOR, "blocked leader"),
        _shadeSpans(
            ax,
            blowingSnowL,
            resampleStep,
            "orange",
            "blow. snow leader",
            hatch="///",
        ),
        _shadeSpans(ax, fBlocked, fileStep, _FOLLOWER_COLOR, "blocked follower"),
        _shadeSpans(
            ax,
            blowingSnowF,
            resampleStep,
            "gold",
            "blow. snow follower",
            hatch="///",
        ),
        _markRestarts(
            ax, _restartTimes(eventDatL), "leader restarted", _LEADER_COLOR, ls="-"
        ),
        _markRestarts(
            ax,
            _restartTimes(eventDatF),
            "follower restarted",
            _FOLLOWER_COLOR,
            ls="-",
        ),
    ]
    for bx in ax:
        bx.set_title("")
    # short two-line labels fit into the small panels; plain tick labels
    # instead of an offset ("1e-8+4.7e-1") in the corner of each panel
    for bx, label in zip(
        ax[3:],
        ["z diff.\n[px]", "h diff.\n[px]", "t diff.\n[s]"]
        + ["theta\n[°]", "phi\n[°]", "Ofz\n[px]"],
    ):
        bx.set_ylabel(label)
        bx.ticklabel_format(axis="y", useOffset=False)
    _formatDayAxes(ax, fl)
    _legendAbove(ax[0], [h for h in handles if h is not None], ncol=3)
    fig.align_ylabels(ax)
    # do not keep the files open: they are re-opened for the status line
    eventDatL.close()
    eventDatF.close()

    print("DONE", fOut)
    tools.savefig(fig, config, fOut, fnames=fnames1M, logoAnchor=ax[0])
    if returnFig:
        return fOut, fig
    else:
        return fOut


@_styled
def metaRotationYearlyQuicklook(year, config, version=__version__, skipExisting=True):
    """
    Create yearly meta rotation quicklook for a given year.

    Parameters
    ----------
    year : str
        Year string in format YYYY
    config : dict or str
        Configuration dictionary or path to config file
    version : str, optional
        Version identifier, by default __version__
    skipExisting : bool, optional
        Whether to skip if output exists, by default True

    Returns
    -------
    tuple
        (output_file_path, matplotlib.figure.Figure)
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    config = tools.readSettings(config)

    ff = files.FindFiles(f"{year}0101", config.leader, config, version)
    ff1 = files.FindFiles(f"{int(year)-1}0101", config.leader, config, version)
    fOut = ff.quicklook.metaRotation
    fOut = fOut.replace("0101.png", ".png")

    fPattern = ff.fnamesPattern.metaRotation.replace("0101.nc", "*.nc")
    rotFiles = sorted(glob.glob(fPattern))

    if (
        skipExisting
        and tools.checkForExisting(fOut, parents=rotFiles)
        # and (
        #     int(year)
        #     < int((datetime.datetime.utcnow() - datetime.timedelta(days=60)).year)
        # )
    ):
        print(f"{year} skip exisiting {fOut}")
        return None, None

    rotDat = _concatRotationFiles(rotFiles)
    if rotDat is None:
        log.error(f"{fPattern} not found")
        return None, None

    # handle current year a bit differently: show the last 365 days
    if int(year) == int(datetime.datetime.utcnow().year):
        rotFiles1 = ff1.fnamesPattern.metaRotation.replace("0101.nc", "*.nc")
        rotFiles1 = sorted(glob.glob(rotFiles1))
        rotDat1 = _concatRotationFiles(rotFiles1)
        if rotDat1 is None:
            log.warning(f"{rotFiles1} not found, plotting the current year only")
        else:
            rotDat = xr.concat((rotDat1, rotDat), dim="file_starttime")
        lastYear = np.datetime64(
            datetime.datetime.utcnow() - datetime.timedelta(days=365)
        )
        rotDat = rotDat.isel(file_starttime=(rotDat.file_starttime > lastYear))
        rotDat = rotDat.sortby("file_starttime")

    import matplotlib.dates as mdates

    fig, axs, _ = _stackedFigure([1, 1, 1], figsize=(13, 9), cbar=False, hspace=0.08)
    ax1, ax2, ax3 = axs
    fig.suptitle(
        f"VISSS rotation  ·  {year}  ·  {config['name']}",
        fontsize=15,
        fontweight="bold",
    )

    for var, ax, color, ylabel in [
        ("camera_phi", ax1, "C0", "phi rotation [°]"),
        ("camera_theta", ax2, "C1", "theta rotation [°]"),
        ("camera_Ofz", ax3, "C2", "z offset [px]"),
    ]:
        mean = rotDat[var].sel(camera_rotation="mean")
        err = rotDat[var].sel(camera_rotation="err")
        ax.plot(mean.file_starttime.values, mean.values, color=color, lw=1)
        ax.fill_between(
            mean.file_starttime.values,
            (mean - err).values,
            (mean + err).values,
            color=color,
            alpha=0.3,
            lw=0,
        )
        ax.set_ylabel(ylabel)

    handles = []
    ofz = rotDat.camera_Ofz.sel(camera_rotation="mean")
    handles.append(
        _shadeSpans(
            axs,
            rotDat.file_starttime.values[np.isnan(ofz.values)],
            _timeStep(rotDat.file_starttime.values, default=np.timedelta64(1, "D")),
            "red",
            "rotation failed",
            alpha=0.25,
        )
    )
    resets = [
        np.datetime64(datetime.datetime.strptime(d.ljust(15, "0"), "%Y%m%d-%H%M%S"))
        for d in config.rotate.keys()
    ]
    resets = [
        r
        for r in resets
        if rotDat.file_starttime.min().values <= r <= rotDat.file_starttime.max().values
    ]
    handles.append(
        _markRestarts(axs, resets, "rotation from config", color="k", ls="-")
    )

    ax1.set_xlim(rotDat.file_starttime.min().values, rotDat.file_starttime.max().values)
    locator = mdates.AutoDateLocator()
    ax3.xaxis.set_major_locator(locator)
    ax3.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    for ax in axs[:-1]:
        ax.tick_params(axis="x", labelbottom=False)
    ax3.set_xlabel("time")
    _legendAbove(ax1, [h for h in handles if h is not None], ncol=3)
    fig.align_ylabels(axs)

    # _statusText(fig, rotFiles, config)
    tools.savefig(fig, config, fOut, fnames=rotFiles, logoAnchor=ax1)

    rotDat.close()

    if year == str(datetime.datetime.today().year):
        try:
            shutil.copy(fOut, ff.quicklookCurrent.metaRotation)
        except PermissionError:
            log.error(f"No permission to write {ff.quicklookCurrent.metaRotation}")

    return fOut, fig


@tools.loopify
@_styled
def metaRotationQuicklook(case, config, version=__version__, skipExisting=True):
    """
    Create meta rotation quicklook for a given case.

    Parameters
    ----------
    case : str
        Date string in format YYYYMMDD
    config : dict
        Configuration dictionary
    version : str, optional
        Version identifier, by default __version__
    skipExisting : bool, optional
        Whether to skip if output exists, by default True

    Returns
    -------
    tuple
        (output_file_path, matplotlib.figure.Figure)
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import pandas as pd

    camera = config.leader

    # get level 0 file names
    ff = files.FindFiles(case, camera, config, version)
    fOut = ff.quicklook.metaRotation

    if skipExisting and tools.checkForExisting(
        fOut,
        events=ff.listFiles("metaEvents"),
        parents=ff.listFiles("metaRotation"),
    ):
        return None, None

    fnames1D = {}
    fnames1D["leader"] = ff.listFiles("level1detect")
    ff2 = files.FindFiles(case, config.follower, config, version)
    fnames1D["follower"] = ff2.listFiles("level1detect")

    nParticles = {}
    for camera in fnames1D.keys():
        nParticles[camera] = {}
        for fname in fnames1D[camera]:
            d1 = xr.open_dataset(fname)
            nP = len(d1.pid)
            d1.close()
            tt = files.FilenamesFromLevel(fname, config).datetime64
            nParticles[camera][tt] = nP
        nParticles[camera] = pd.Series(nParticles[camera])

    print(case, camera, fOut)

    if len(ff.listFiles("level0txt")) == 0 and len(ff.listFiles("level0status")) == 0:
        print(case, "no data")

        return None, None

    print("reading events")
    if len(ff.listFiles("metaEvents")) == 0:
        if len(ff.listFilesExt("metaEvents")) == 0:
            log.error(f"event data not found")
        else:
            log.warning(f"{case} metaEvents is nodata/broken, skipping quicklook")
        return None, None

    try:
        eventFile = ff.listFiles("metaEvents")[0]
    except IndexError:
        print(f"no leader event file")
        return None, None

    try:
        with xr.open_dataset(eventFile) as ds:
            events = ds.load()
    except:
        print(f"{eventFile} broken")
        return None, None

    try:
        eventFile = ff2.listFiles("metaEvents")[0]
    except IndexError:
        print(f"no follower event file")
        return None, None

    try:
        with xr.open_dataset(eventFile) as ds:
            events2 = ds.load()
    except:
        print(f"{eventFile} broken")
        return None, None

    if len(events.data_vars) == 0:
        print(f'{ff.listFiles("metaEvents")[0]} empty')
        return None, None

    if len(events2.data_vars) == 0:
        print(f'{ff.listFiles("metaEvents")[0]} empty')
        return None, None

    if len(ff.listFiles("metaRotation")) == 0:
        print(f"no metaRotation data yet")
        return None, None

    try:
        with xr.open_dataset(ff.listFiles("metaRotation")[0]) as ds:
            rotDat = ds.load()
    except:
        print(f'{ff.listFiles("metaRotation")[0]} broken')
        return None, None

    if len(rotDat.file_starttime) <= 1:
        print(f'{ff.listFiles("metaRotation")[0]} empty')
        return None, None

    ts = events.file_starttime.where(
        (events.event == "sleep-trigger") | (events.event == "stop-trigger")
    )
    ts2 = events2.file_starttime.where(
        (events2.event == "sleep-trigger") | (events2.event == "stop-trigger")
    )

    print("plotting")
    # plotting
    fig, axs, _ = _stackedFigure([1, 1, 1], figsize=(13, 9), cbar=False, hspace=0.08)
    ax1, ax2, ax3 = axs
    _suptitle(fig, "rotation", ff, config)

    # (variable, panel, color, label): value with its uncertainty as band
    for var, ax, color, label in [
        ("camera_phi", ax1, "C0", "phi"),
        ("camera_theta", ax1, "C1", "theta"),
        ("camera_Ofz", ax2, "C2", "z offset"),
    ]:
        mean = rotDat[var].sel(camera_rotation="mean")
        err = rotDat[var].sel(camera_rotation="err")
        ax.plot(
            mean.file_starttime.values,
            mean.values,
            marker="x",
            color=color,
            label=label,
        )
        ax.fill_between(
            mean.file_starttime.values,
            (mean - err).values,
            (mean + err).values,
            color=color,
            alpha=0.3,
            lw=0,
        )
    ax1.set_ylabel("angular rotation [°]")
    ax2.set_ylabel("z offset [px]")
    seriesHandles = list(ax1.get_lines())

    for camera, color in [("leader", _LEADER_COLOR), ("follower", _FOLLOWER_COLOR)]:
        ax3.plot(
            nParticles[camera].index.values,
            nParticles[camera].values,
            marker="x",
            color=color,
            label=camera,
        )
    ax3.set_yscale("log")
    ax3.set_ylabel("# particles in lv1detect")
    ax3.legend(loc="upper right", ncol=2)

    handles = seriesHandles

    ofz = rotDat.camera_Ofz.sel(camera_rotation="mean")
    handle = _shadeSpans(
        axs,
        rotDat.file_starttime.values[np.isnan(ofz.values)],
        _timeStep(rotDat.file_starttime.values),
        "red",
        "rotation failed",
        alpha=0.3,
    )
    handles.append(handle)

    for ev, color, name in [
        (events, _LEADER_COLOR, "leader"),
        (events2, _FOLLOWER_COLOR, "follower"),
    ]:
        width = _timeStep(ev.file_starttime.values)
        blocked = ev.blocking.dropna("file_starttime").sel(blockingThreshold=50) > 0.1
        handles.append(
            _shadeSpans(
                axs,
                blocked.file_starttime.values[blocked.values],
                width,
                color,
                f"{name} blocked",
                alpha=0.25,
            )
        )
        isIdle = ((ev.event == "sleep-trigger") | (ev.event == "stop-trigger")).values
        handles.append(
            _shadeSpans(
                axs,
                ev.file_starttime.values[isIdle],
                width,
                "orange",
                f"{name} idle",
                alpha=0.35,
            )
        )

    t0, t1 = _dayLimits(ff)
    resets = [
        np.datetime64(datetime.datetime.strptime(d.ljust(15, "0"), "%Y%m%d-%H%M%S"))
        for d in config.rotate.keys()
    ]
    resets = [r for r in resets if t0 <= r <= t1]
    handles.append(
        _markRestarts(axs, resets, "rotation from config", color="k", ls="-")
    )
    handles.append(
        _markRestarts(
            axs, _restartTimes(events), "leader restarted", _LEADER_COLOR, ls=":"
        )
    )
    handles.append(
        _markRestarts(
            axs, _restartTimes(events2), "follower restarted", _FOLLOWER_COLOR, ls="--"
        )
    )

    _formatDayAxes(axs, ff)
    _legendAbove(ax1, [h for h in handles if h is not None], ncol=6)
    fig.align_ylabels(axs)
    # do not keep the files open: they are re-opened for the status line
    events2.close()

    # _statusText(fig, ff.listFiles("metaRotation"), config)
    tools.savefig(
        fig, config, fOut, fnames=ff.listFiles("metaRotation"), logoAnchor=ax1
    )
    rotDat.close()
    events.close()

    return fOut, fig


@tools.loopify_with_camera(endYesterday=False)
@_styled
def createLevel2detectQuicklook(
    case, camera, config, version=__version__, skipExisting=True, returnFig=True
):
    """
    Create level2detect quicklook for a given case and camera.

    Parameters
    ----------
    case : str
        Date string in format YYYYMMDD
    camera : str
        Camera identifier
    config : dict
        Configuration dictionary
    version : str, optional
        Version identifier, by default __version__
    skipExisting : bool, optional
        Whether to skip if output exists, by default True
    returnFig : bool, optional
        Whether to return the figure, by default True

    Returns
    -------
    tuple
        (output_file_path, matplotlib.figure.Figure) if returnFig=True,
        otherwise (output_file_path, None)
    """
    version = __version__

    nodata = False
    noRawData = False
    # get level 0 file names
    ff = files.FindFiles(case, camera, config, version)
    fOut = ff.quicklook.level2detect

    if skipExisting and tools.checkForExisting(
        fOut,
        events=ff.listFiles("metaEvents"),
        parents=ff.listFiles("level2detect"),
    ):
        return None, None

    lv2 = ff.listFiles("level2detect")
    if len(lv2) == 0:
        if len(ff.listFilesExt("level2detect")) == 0:
            log.error(f"{case} level2detect data not found")
            return None, None
        else:
            nodata = True
    else:
        lv2 = lv2[0]

    if len(ff.listFiles("metaEvents")) == 0:
        if len(ff.listFilesExt("metaEvents")) == 0:
            log.error(f"{case} event data not found")
            return None, None
        else:
            # confirmed sentinel (nodata/broken.txt), not just "not
            # processed yet" -- still worth a placeholder quicklook so the
            # gallery has one image per day instead of a silent gap
            log.warning(f"{case} metaEvents is nodata/broken, plotting as no raw data")
            noRawData = True

    log.info(f"running {case} {fOut}")

    what = f"level2detect {camera.split('_')[0]}"
    logoAnchor = None
    if noRawData:
        fig = _placeholderFigure(what, ff, config, "no raw data")
    elif nodata:
        fig = _placeholderFigure(what, ff, config, "no data")
    else:
        dat2 = xr.open_dataset(lv2)
        fEvents1 = ff.listFiles("metaEvents")[0]

        try:
            events = xr.open_dataset(fEvents1)
        except:
            print(f"{fEvents1} broken")
            return None, None

        fig, axs, caxs = _stackedFigure([2.5, 1, 1, 1], figsize=(13, 12))
        _suptitle(fig, what, ff, config)
        logoAnchor = axs[0]
        _plotLevel2Core(axs[0], caxs[0], axs[1], axs[2], axs[3], dat2, sel={})

        handles = _shadeQualityFlags(
            axs,
            dat2,
            config,
            ["recordingFailed", "obervationsDiffer", "processingFailed"]
            + ["cameraBlocked", "blowingSnow", "sunGlint"],
        )
        handle = _markRestarts(axs, _restartTimes(events))
        if handle is not None:
            handles.append(handle)
        _formatDayAxes(axs, ff)
        _legendAbove(axs[0], handles)
        fig.align_ylabels(axs)
        # do not keep the files open: they are re-opened for the status line
        dat2.close()
        events.close()

    tools.savefig(
        fig,
        config,
        fOut,
        fnames=ff.listFiles("level2detect"),
        logoAnchor=logoAnchor,
    )
    if returnFig:
        return fOut, fig
    else:
        return fOut


@tools.loopify
@_styled
def createLevel2matchQuicklook(
    case, config, version=__version__, skipExisting=True, returnFig=True
):
    """
    Create level2match quicklook for a given case.

    Parameters
    ----------
    case : str
        Date string in format YYYYMMDD
    config : dict
        Configuration dictionary
    version : str, optional
        Version identifier, by default __version__
    skipExisting : bool, optional
        Whether to skip if output exists, by default True
    returnFig : bool, optional
        Whether to return the figure, by default True

    Returns
    -------
    tuple
        (output_file_path, matplotlib.figure.Figure) if returnFig=True,
        otherwise (output_file_path, None)
    """
    camera = config.leader
    nodata = False
    noRawData = False
    # get level 0 file names
    ff = files.FindFiles(case, camera, config, version)
    fOut = ff.quicklook.level2match

    if skipExisting and tools.checkForExisting(
        fOut,
        events=ff.listFiles("metaEvents"),
        parents=ff.listFiles("level2match"),
    ):
        return None, None

    lv2match = ff.listFiles("level2match")
    if len(lv2match) == 0:
        if len(ff.listFilesExt("level2match")) == 0:
            log.error(
                f"{case} level2match data not found {ff.fnamesPattern.level2match}"
            )
            return None, None
        else:
            nodata = True
    else:
        lv2match = lv2match[0]

    if len(ff.listFiles("metaEvents")) == 0:
        if len(ff.listFilesExt("metaEvents")) == 0:
            log.error(f"{case} event data not found")
            return None, None
        else:
            log.warning(f"{case} metaEvents is nodata/broken, plotting as no raw data")
            noRawData = True

    log.info(f"running {case} {fOut}")

    what = "level2match"
    logoAnchor = None
    if noRawData:
        fig = _placeholderFigure(what, ff, config, "no raw data")
    elif nodata:
        fig = _placeholderFigure(what, ff, config, "no data")
    else:
        dat2 = xr.open_dataset(lv2match)
        fEvents1 = ff.listFiles("metaEvents")[0]
        fEvents2 = files.FilenamesFromLevel(fEvents1, config).filenamesOtherCamera(
            level="metaEvents"
        )[0]

        try:
            events1 = xr.open_dataset(fEvents1)
        except:
            print(f"{fEvents1} broken")
            return None, None
        try:
            events2 = xr.open_dataset(fEvents2)
        except:
            print(f"{fEvents2} broken")
            return None, None

        events = xr.concat((events1, events2), dim="file_starttime")

        fig, axs, caxs = _stackedFigure([2.5, 1, 1, 1], figsize=(13, 12))
        _suptitle(fig, what, ff, config)
        logoAnchor = axs[0]
        _plotLevel2Core(
            axs[0],
            caxs[0],
            axs[1],
            axs[2],
            axs[3],
            dat2,
            sel=dict(camera="max"),
            matchScore=True,
        )

        handles = _shadeQualityFlags(
            axs,
            dat2,
            config,
            ["recordingFailed", "obervationsDiffer", "processingFailed"]
            + ["cameraBlocked", "blowingSnow", "sunGlint", "zResidualTooWide"],
        )
        handle = _markRestarts(axs, _restartTimes(events))
        if handle is not None:
            handles.append(handle)
        _formatDayAxes(axs, ff)
        _legendAbove(axs[0], handles)
        fig.align_ylabels(axs)
        # do not keep the files open: they are re-opened for the status line
        dat2.close()
        events1.close()
        events2.close()

    tools.savefig(
        fig,
        config,
        fOut,
        fnames=ff.listFiles("level2match"),
        logoAnchor=logoAnchor,
    )
    if returnFig:
        return fOut, fig
    else:
        return fOut


@tools.loopify
@_styled
def createLevel2trackQuicklook(
    case, config, version=__version__, skipExisting=True, returnFig=True
):
    """
    Create level2track quicklook for a given case.

    Parameters
    ----------
    case : str
        Date string in format YYYYMMDD
    config : dict
        Configuration dictionary
    version : str, optional
        Version identifier, by default __version__
    skipExisting : bool, optional
        Whether to skip if output exists, by default True
    returnFig : bool, optional
        Whether to return the figure, by default True

    Returns
    -------
    tuple
        (output_file_path, matplotlib.figure.Figure) if returnFig=True,
        otherwise (output_file_path, None)
    """
    camera = config.leader
    nodata = False
    noRawData = False
    # get level 0 file names
    ff = files.FindFiles(case, camera, config, version)
    fOut = ff.quicklook.level2track
    if skipExisting and tools.checkForExisting(
        fOut,
        events=ff.listFiles("metaEvents"),
        parents=ff.listFiles("level2track"),
    ):
        return None, None

    lv2track = ff.listFiles("level2track")
    if len(lv2track) == 0:
        if len(ff.listFilesExt("level2track")) == 0:
            log.error(f"{case} lv2track data not found")
            return None, None
        else:
            nodata = True
    else:
        lv2track = lv2track[0]

    if len(ff.listFiles("metaEvents")) == 0:
        if len(ff.listFilesExt("metaEvents")) == 0:
            log.error(f"{case} event data not found")
            return None, None
        else:
            log.warning(f"{case} metaEvents is nodata/broken, plotting as no raw data")
            noRawData = True

    log.info(f"running {case} {fOut}")

    what = "level2track"
    logoAnchor = None
    if noRawData:
        fig = _placeholderFigure(what, ff, config, "VISSS did not operate, no raw data")
    elif nodata:
        fig = _placeholderFigure(what, ff, config, "no data")
    else:
        dat2 = xr.open_dataset(lv2track)

        fEvents1 = ff.listFiles("metaEvents")[0]
        fEvents2 = files.FilenamesFromLevel(fEvents1, config).filenamesOtherCamera(
            level="metaEvents"
        )[0]

        try:
            events1 = xr.open_dataset(fEvents1)
        except:
            print(f"{fEvents1} broken")
            return None, None
        try:
            events2 = xr.open_dataset(fEvents2)
        except:
            print(f"{fEvents2} broken")
            return None, None

        events = xr.concat((events1, events2), dim="file_starttime")

        fig, axs, caxs = _stackedFigure([1.25, 1.25, 1, 1, 1], figsize=(13, 13.5))
        _suptitle(fig, what, ff, config)
        logoAnchor = axs[0]

        # PSD / N_tot, sizes, shape and performance as for level2match; the
        # velocity distribution takes the second panel
        _plotLevel2Core(
            axs[0],
            caxs[0],
            axs[2],
            axs[3],
            axs[4],
            dat2,
            sel=dict(cameratrack="max"),
            aspectSel=dict(cameratrack="min"),
            matchScore=True,
        )
        _plotSizeDist(
            axs[1],
            caxs[1],
            dat2.velocity_dist.sel(
                cameratrack="mean", size_definition="Dmax", dim3D="z"
            ),
            "Sedimentation velocity [m/s]",
            vmin=0,
            vmax=3,
        )
        axs[1].set_ylabel("Dmax [mm]")

        handles = _shadeQualityFlags(
            axs,
            dat2,
            config,
            ["recordingFailed", "obervationsDiffer", "processingFailed"]
            + ["cameraBlocked", "blowingSnow", "sunGlint"]
            + ["trackingIncomplete", "zResidualTooWide"],
        )
        handle = _markRestarts(axs, _restartTimes(events))
        if handle is not None:
            handles.append(handle)
        _formatDayAxes(axs, ff)
        _legendAbove(axs[0], handles)
        fig.align_ylabels(axs)
        # do not keep the files open: they are re-opened for the status line
        dat2.close()
        events1.close()
        events2.close()

    tools.savefig(
        fig,
        config,
        fOut,
        fnames=ff.listFiles("level2track"),
        logoAnchor=logoAnchor,
    )
    if returnFig:
        return fOut, fig
    else:
        return fOut


@tools.loopify
def createLevel1matchParticlesQuicklook(
    timestamp,
    config,
    version=__version__,
    container_width=200,
    container_height_max=300,
    nTiles=60,
    nRows=4,
    extra=1,
    readParticlesFromFiles=True,
    skipExisting=True,
    ffOut="default",
    timeStep="variable",  # attempt to fill plot equally
    minBlur="config",
    minSize="config",
    omitLabel4small="config",
    timedelta=np.timedelta64(1, "D"),
    returnFig=True,
    doLevel1matchQuicklook=True,
):
    """
    Create level1match particles quicklook for a given timestamp.

    Parameters
    ----------
    timestamp : str
        Timestamp string in format YYYYMMDDHH or YYYYMMDD
    config : dict
        Configuration dictionary
    version : str, optional
        Version identifier, by default __version__
    container_width : int, optional
        Width of each tile container, by default 200
    container_height_max : int, optional
        Maximum height of each tile container, by default 300
    nTiles : int, optional
        Number of tiles per row, by default 60
    nRows : int, optional
        Number of rows in output, by default 4
    extra : int, optional
        Extra spacing between tiles, by default 1
    readParticlesFromFiles : bool, optional
        Whether to read particles from files, by default True
    skipExisting : bool, optional
        Whether to skip if output exists, by default True
    ffOut : str, optional
        Output file path, by default "default"
    timeStep : str, optional
        Time step method ('fixed' or 'variable'), by default "variable"
    minBlur : float or str, optional
        Minimum blur threshold, by default "config"
    minSize : int or str, optional
        Minimum particle size, by default "config"
    omitLabel4small : bool or str, optional
        Whether to omit labels for small particles, by default "config"
    timedelta : numpy.timedelta64, optional
        Time window for data selection, by default np.timedelta64(1, "D")
    returnFig : bool, optional
        Whether to return the figure, by default True
    doLevel1matchQuicklook : bool, optional
        Whether to also run level1match quicklook, by default True

    Returns
    -------
    tuple
        (output_file_path, matplotlib.figure.Figure) if returnFig=True,
        otherwise (output_file_path, None)
    """
    import cv2
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import pandas as pd
    from PIL import Image, ImageDraw, ImageFont
    from tqdm import tqdm

    camera = config["leader"]

    # for convinience, do the other L1match quicklook as well
    if doLevel1matchQuicklook:
        createLevel1matchQuicklook(timestamp, config, skipExisting=skipExisting)

    if minBlur == "config":
        minBlur = config["level1detectQuicklook"]["minBlur"]
    if minSize == "config":
        minSize = config["level1detectQuicklook"]["minSize"]
    if omitLabel4small == "config":
        omitLabel4small = config["level1detectQuicklook"]["omitLabel4small"]

    ff = files.FindFiles(timestamp, camera, config, version)
    if ffOut == "default":
        ffOut = ff.quicklook.level1matchParticles

    site = config["site"]

    particlesPloted = 0

    if skipExisting and tools.checkForExisting(
        ffOut,
        level0=ff.listFiles("level0"),
        events=ff.listFiles("metaEvents"),
        parents=ff.listFilesExt("level1match"),
    ):
        return None, None

    if site != "mosaic":
        noLevel0Yet = (len(ff.listFiles("level0")) == 0) and (
            len(ff.listFiles("level0status")) == 0
        )
        if noLevel0Yet and ff.isDataTransferPending("level0"):
            print("NO DATA YET (TRANSFERRED?)", ffOut)
            return None, None

    if len(ff.listFiles("metaFrames")) > len(ff.listFiles("level0")):
        print("DATA TRANSFER INCOMPLETE ", ffOut)
        print(
            len(ff.listFiles("level0")),
            "of",
            len(ff.listFiles("metaFrames")),
            "transmitted",
        )
        return None, None

    if (len(ff.listFilesExt("level1match")) == 0) and (len(ff.listFiles("level0")) > 0):
        print("NO DATA YET ", ffOut)
        return None, None

    if not ff.isCompleteL1match:
        print(
            "NOT COMPLETE YET %i of %i L1match %s "
            % (
                len(ff.listFilesExt("level1match")),
                len(ff.listFiles("level0txt")),
                ffOut,
            )
        )

        #         if (len(ff.listFilesExt("level1match")) == len(ff.listFiles("level0"))):
        #             afshgsa
        return None, None

    #     else:

    total_width = (container_width + extra) * nTiles // nRows
    max_height = (20 + extra + container_height_max) * nRows + 60

    # let use a matplotlib font becuase we can be sure it is there
    mpl_data_dir = os.path.dirname(mpl.matplotlib_fname())
    mpl_ttf_dir = os.path.join(mpl_data_dir, "fonts", "ttf")
    font = ImageFont.truetype(f"{mpl_ttf_dir}/DejaVuSans.ttf", 35)
    fontL = ImageFont.truetype(f"{mpl_ttf_dir}/DejaVuSans.ttf", 16)

    print("RUNNING open files ", ffOut, len(ff.listFiles("metaFrames")))

    dats2 = []
    l1Files = ff.listFilesWithNeighbors("level1match")

    nParticles = 0
    for fname2 in tqdm(l1Files):
        fname1 = fname2.replace("level1match", "metaFrames")
        try:
            dat2 = xr.open_dataset(fname2)

        except FileNotFoundError:
            if os.path.isfile(f"{fname2}.nodata"):
                continue
            elif os.path.isfile(f"{fname2}.broken.txt"):
                continue
            elif os.path.isfile(f"{fname2}.notenoughframes"):
                continue
            else:
                raise FileNotFoundError(fname2)

        nParticles += len(dat2.pair_id)

        # to speed things up, lets look only at the first 30000 particles. this avoids crazily large files
        dat2 = dat2.isel(pair_id=slice(30000))

        dat2 = dat2[
            [
                "Dmax",
                "blur",
                "record_time",
                "record_id",
                "Droi",
                "file_starttime",
                "matchScore",
            ]
        ]

        # it is more efficient to load the data now in comparison to after isel
        dat2 = dat2.load()

        # print(fname2, len(dat2.pair_id))
        dat2 = dat2.isel(
            pair_id=(
                (dat2.blur > minBlur).all("camera")
                & (dat2.Dmax > minSize).all("camera")
                & (dat2.matchScore > config.quality.minMatchScore)
            )
        )
        # print(fname2, len(dat2.pair_id))

        if len(dat2.pair_id) == 0:
            continue
        dat2 = dat2[["record_time", "record_id", "Droi", "file_starttime"]]

        dat2 = dat2.expand_dims(dict(file=[fname1]))
        #     dat2 = dat2.set_coords(dict(file = fname2))
        dat2 = dat2.stack(fpair_id=("file", "pair_id"))
        dats2.append(dat2)

    print("opened")
    new_im = Image.new("RGB", (total_width, max_height), color=(255, 255, 255))

    if len(dats2) == 0:
        draw = ImageDraw.Draw(new_im)
        draw.text(
            (total_width // 3, max_height // 3), "no raw data", (0, 0, 0), font=font
        )

    else:
        limDat = xr.concat(dats2, dim="fpair_id")
        limDat = limDat.isel(
            fpair_id=(limDat.record_time.isel(camera=0).values >= ff.datetime64)
            & (limDat.record_time.isel(camera=0).values < (ff.datetime64 + timedelta))
        )
        #         limDat = dats2
        print("merged")

        if len(limDat.fpair_id) == 0:
            print("TOO FEW DATA ", ffOut)

            draw = ImageDraw.Draw(new_im)
            draw.text(
                (total_width // 3, max_height // 3),
                "no data recorded",
                (0, 0, 0),
                font=font,
            )

        else:
            print("Total number of particles for plotting %i" % len(limDat.fpair_id))

            if timeStep == "variable":
                timeSteps = np.percentile(
                    limDat.record_time, np.linspace(0, 100, nTiles + 1)
                )
            elif timeStep == "fixed":
                timeSteps = np.array(
                    pd.date_range(
                        start=ff.datetime64,
                        periods=nTiles + 1,
                        end=ff.datetime64 + timedelta,
                    )
                )
            else:
                raise ValueError("do not understand timeStep")
            mosaics = []

            videos = {}
            for tt, (t1, t2) in enumerate(zip(timeSteps[:-1], timeSteps[1:])):
                if tt == len(timeSteps) - 2:
                    whereCond = limDat.record_time >= t1
                else:
                    whereCond = (limDat.record_time >= t1) & (limDat.record_time < t2)

                thisDat = limDat.isel(fpair_id=whereCond.any("camera").values)
                totalArea = 0

                # select pids randomly, figure out how much we need, and sort them again
                starttimes = deepcopy(thisDat.file_starttime.values).T
                fpair_ids = deepcopy(thisDat.fpair_id.values)

                fnames = []
                time2Fname = {}
                for cc, camera in enumerate(config.instruments):
                    time2Fname[camera] = {}
                    starttimes1 = np.unique(starttimes[:, cc])
                    for starttime1 in starttimes1:
                        case = pd.to_datetime(str(starttime1)).strftime("%Y%m%d-%H%M%S")
                        ff1 = files.FindFiles(case, camera, config, version=version)
                        zipF = ff1.listFiles("imagesL1detect")
                        if len(zipF) > 0:
                            fnames.append(zipF[0])
                            time2Fname[camera][starttime1] = zipF[0]
                        else:
                            log.error(f"no zip file for {case} {camera}")
                particleImages = {}
                for fname in np.unique(fnames):
                    fn = files.FilenamesFromLevel(fname, config)
                    # tarRoot = fn.fname.imagesL1detect.split("/")[-1].replace(".tar.bz2","")
                    # particleImages[fname] = (tools.imageTarFile.open(fn.fname.imagesL1detect, "r:bz2"), tarRoot)
                    # print(fn.fname.imagesL1detect)
                    particleImages[fname] = tools.imageZipFile(
                        fn.fname.imagesL1detect, mode="r"
                    )

                nPids = fpair_ids.shape[0]
                # import pdb

                # pdb.set_trace()
                rng = np.random.default_rng(tt)
                rng.shuffle(fpair_ids)
                # apply shuffling
                thisDat = thisDat.sel(fpair_id=fpair_ids)

                containerSize = container_width * container_height_max

                if nPids < 5:  # fo very few particles I want to see them all!
                    nParticlesNeeded = len(fpair_ids)
                else:
                    try:
                        nParticlesNeeded = (
                            np.where(
                                (thisDat.Droi + 1)
                                .prod("dim2D")
                                .sum("camera")
                                .cumsum("fpair_id")
                                / containerSize
                                > 0.95
                            )[0][0]
                            + 1
                        )
                    except IndexError:
                        nParticlesNeeded = nPids

                fpair_ids = fpair_ids[:nParticlesNeeded]
                print(tt, "/", nTiles, t1, t2, nParticlesNeeded, "of", nPids)
                particlesPloted += nParticlesNeeded
                ims = []

                for fp, fpair_id in enumerate(fpair_ids):
                    im = [None, None]
                    fname2, pair_id = fpair_id
                    particle_pair = thisDat.sel(fpair_id=fpair_id)
                    for cc, camera in enumerate(config.instruments):
                        particle = particle_pair.sel(camera=camera)
                        pid = particle.pid.values
                        background = 0

                        pidStr = "%07i" % pid
                        # imName = '%s.png' % (pidStr)
                        # imfname = '%s/%s/%s' % (particleImages[fname][1],pidStr[:4], imName)

                        starttime1 = particle.file_starttime
                        fname1 = time2Fname[camera][starttime1.values]
                        try:
                            # im = particleImages[fname][0].extractimage(imfname)
                            im[cc] = particleImages[fname1].extractnpy(pidStr)
                        except KeyError:
                            print("NOT FOUND ", pidStr)
                            im[cc] = np.array([[0]])
                            continue
                        # apply alpha channel
                        # im[...,0][im[...,1] == 0] = background
                        # drop alpha channel
                        im[cc] = im[cc][..., 0]

                    im = tools.concatImgX(*im, background=50)

                    # im = av.doubleDynamicRange(im, offset=2)

                    im = np.pad(im, [(0, 1), (0, 1)], constant_values=0)
                    try:
                        fid = np.where(fname2 == np.array(ff.listFiles("metaFrames")))[
                            0
                        ][0]
                    except:
                        fid = -1
                    text = np.full((100, 100), background, dtype=np.uint8)

                    textStr = "%i.%i" % (fid, pid)

                    text = cv2.putText(
                        text,
                        textStr,
                        (0, 50),
                        cv2.FONT_HERSHEY_PLAIN,
                        0.75,
                        255,
                        1,
                    )
                    text = _crop(text)

                    y1, x1 = im.shape
                    y2, x2 = text.shape

                    # only add label if large enough
                    if (omitLabel4small == "all") or (
                        (omitLabel4small == True) and (x1 < x2)
                    ):
                        imT = im
                    else:
                        imT = tools.concatImgY(im, text)
                        # import pdb;pdb.set_trace()
                    ims.append(imT)
                    totalArea += np.prod(imT.shape)

                for fname in fnames:
                    particleImages[fname].close()

                # make tile
                images = [Image.fromarray(im) for im in ims]
                if len(images) == 0:
                    mosaic = (
                        np.ones(
                            (container_height_max, container_width, 3), dtype=np.uint8
                        )
                        * 0
                    )
                else:
                    mosaic = Packer_patched(images).pack(
                        container_width=container_width,
                        container_height_max=container_height_max,
                    )
                    mosaic = np.array(mosaic)

                    if container_width > mosaic.shape[1]:
                        mosaic = np.pad(
                            mosaic,
                            [(0, 0), (0, container_width - mosaic.shape[1]), (0, 0)],
                        )

                    # sometimes container is too large...
                    mosaic = mosaic[:container_height_max, :container_width]

                label = Image.fromarray(
                    np.ones((20, mosaic.shape[1], 3), dtype=np.uint8) * 255
                )
                drawL = ImageDraw.Draw(label)
                textStr = "%s-%s" % (
                    str(t1).split(".")[0].split("T")[1],
                    str(t2).split(".")[0].split("T")[1],
                )
                if nParticlesNeeded != nPids:
                    textStr += " (R)"
                drawL.text((0, 0), textStr, (0, 0, 0), font=fontL)

                mosaic = Image.fromarray(np.vstack((label, mosaic)))
                #             display(mosaic)
                mosaics.append(mosaic)

            nCols = nTiles // nRows

            widths, heights = zip(*(i.size for i in mosaics))

            for nRow in range(nRows):
                x_offset = 0

                for im in mosaics[nCols * (nRow) : nCols * (nRow + 1)]:
                    new_im.paste(im, (x_offset, max(heights) * nRow + 50))
                    x_offset += im.size[0] + extra

                # x_offset = 0
            # for im in mosaics[len(mosaics)//nRows:]:
            #   new_im.paste(im, (x_offset,max(heights) +50))
            #   x_offset += im.size[0] + extra

    tenmm = 1e6 / (1 / config.calibration.slope) / 100

    if ff.hour == "":
        title = (
            "%s-%s-%s %s, size threshold for plotting: %i px (%.2f mm), %i of %i larger matched particles plotted, 10 mm = %.1f px ="
            % (
                ff.year,
                ff.month,
                ff.day,
                config["name"],
                minSize,
                minSize * (1 / config.calibration.slope) * 1e-6 * 1000,
                particlesPloted,
                nParticles,
                tenmm,
            )
        )
    else:
        title = (
            "%s-%s-%sT%s %s, size threshold for plotting: %i px (%.2f mm), %i of %i larger matched particles plotted, 10 mm = %.1f px ="
            % (
                ff.year,
                ff.month,
                ff.day,
                ff.hour,
                config["name"],
                minSize,
                minSize * (1 / config.calibration.slope) * 1e-6 * 1000,
                particlesPloted,
                nParticles,
                tenmm,
            )
        )

    # new_im = cv2.putText(np.array(new_im), title,
    #                      (0, 45), FONT, FONT_SCALE, (0, 0, 0), FONT_THICKNESS,)
    # (label_width, label_height), baseline = cv2.getTextSize(
    #     title, FONT, FONT_SCALE, FONT_THICKNESS)

    draw = ImageDraw.Draw(new_im)
    draw.text((0, 0), title, (0, 0, 0), font=font)
    width = draw.textlength(title, font=font)

    draw.line((width + 15, 30, width + 15 + round(tenmm), 30), fill=0, width=5)

    tools.createParentDir(ffOut, mode=config.dirMode)
    new_im.save(ffOut)
    print("SAVED ", ffOut)

    if returnFig:
        return ffOut, new_im
    else:
        return ffOut


@tools.loopify
@_styled
def createLevel3RimingQuicklook(
    case,
    config,
    skipExisting=True,
    version=__version__,
    returnFig=True,
):
    """
    Create level3 combined rime quicklook for a given case.

    Parameters
    ----------
    case : str
        Date string in format YYYYMMDD
    config : dict
        Configuration dictionary
    skipExisting : bool, optional
        Whether to skip if output exists, by default True
    version : str, optional
        Version identifier, by default __version__
    returnFig : bool, optional
        Whether to return the figure, by default True

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    camera = config.leader
    nodata = False
    noRawData = False
    # get level 0 file names
    ff = files.FindFiles(case, camera, config, version)
    fOut = ff.quicklook.level3combinedRiming

    if skipExisting and tools.checkForExisting(
        fOut,
        events=ff.listFiles("metaEvents"),
        parents=ff.listFiles("level3combinedRiming"),
    ):
        return None, None

    lv3 = ff.listFiles("level3combinedRiming")
    if len(lv3) == 0:
        if len(ff.listFilesExt("level3combinedRiming")) == 0:
            log.error(
                f"{case} level3combinedRiming data not found {ff.fnamesPattern.level3combinedRiming}"
            )
            return None, None
        else:
            nodata = True
    else:
        lv3 = lv3[0]

    if len(ff.listFiles("metaEvents")) == 0:
        if len(ff.listFilesExt("metaEvents")) == 0:
            log.error(f"{case} event data not found")
            return None, None
        else:
            log.warning(f"{case} metaEvents is nodata/broken, plotting as no raw data")
            noRawData = True

    log.info(f"running {case} {fOut}")

    what = f"level3combinedRiming {config.level3.combinedRiming.extraFileStr}"
    logoAnchor = None
    if noRawData:
        fig = _placeholderFigure(what, ff, config, "no raw data")
    elif nodata:
        fig = _placeholderFigure(what, ff, config, "no data")
    else:
        dat3 = xr.open_dataset(lv3)  # .sel(size_definition="Dmax", drop=True)

        dat2 = xr.open_dataset(ff.listFiles("level2track")[0]).sel(
            size_definition="Dmax", cameratrack="max", drop=True
        )

        fEvents1 = ff.listFiles("metaEvents")[0]
        fEvents2 = files.FilenamesFromLevel(fEvents1, config).filenamesOtherCamera(
            level="metaEvents"
        )[0]

        try:
            events1 = xr.open_dataset(fEvents1)
        except:
            print(f"{fEvents1} broken")
            return None, None
        try:
            events2 = xr.open_dataset(fEvents2)
        except:
            print(f"{fEvents2} broken")
            return None, None

        events = xr.concat((events1, events2), dim="file_starttime")

        fig, axs, _ = _stackedFigure([1, 1, 1, 1], figsize=(13, 12), cbar=False)
        _suptitle(fig, what, ff, config)
        logoAnchor = axs[0]
        ax1, ax2, ax3, ax4 = axs

        shapes = ["column", "dendrite", "needle", "plate", "rosette"]

        _line(ax1, dat3.Ze_0, label="Ze_0")
        _line(ax1, dat3.Ze_ground, label="Ze_ground")
        _line(
            ax1,
            dat3.Ze_combinedRetrieval.sel(shape="mean"),
            label="Ze_combinedRetrieval",
        )
        _line(
            ax1,
            dat3.Ze_ground_fitResidual.where(dat3.Ze_0 > -10),
            label="Ze_ground_fitResidual",
        )
        _line(ax1, dat3.Ze_std.where(dat3.Ze_0 > -10), label="Ze_std")
        ax1.set_ylabel("Ze [dBZ]")
        ax1.set_ylim(-20, 40)
        ax1.add_artist(
            ax1.legend(loc="upper left", ncol=3)
        )  # kept next to the flag legend

        _line(
            ax2,
            dat3.combinedNormalizedRimeMass.sel(shape="mean"),
            label="M (combined, shape: mean)",
        )
        for ii, shape in enumerate(shapes):
            _line(
                ax2,
                dat3.combinedNormalizedRimeMass.sel(shape=shape),
                c="C0",
                alpha=0.3,
                lw=1,
                label="M (combined, individual shapes)" if ii == 0 else None,
            )
        _line(ax2, dat2.normalizedRimeMass_mean, label="M (in situ weighted mean)")
        _line(
            ax2,
            dat2.normalizedRimeMass_dist.quantile(0.9, dim="D_bins"),
            label="M (in situ 90th percentile)",
        )
        ax2.set_ylabel("M [-]")
        ax2.margins(y=0.2)
        ax2.set_yscale("log")
        ax2.set_ylim(1e-3, 10)
        ax2.legend(loc="upper left", ncol=2)

        _line(ax3, dat3.IWC.sel(shape="mean"), label="IWC (combined, shape: mean)")
        for ii, shape in enumerate(shapes):
            _line(
                ax3,
                dat3.IWC.sel(shape=shape),
                c="C0",
                alpha=0.3,
                lw=1,
                label="individual shapes" if ii == 0 else None,
            )
        ax3.set_ylabel("IWC [kg m$^{-3}$]")
        ax3.margins(y=0.2)
        ax3.set_yscale("log")
        ax3.legend(loc="upper left", ncol=2)

        _line(
            ax4,
            dat3.SR_M.sel(shape="mean"),
            label="SR (combined with meas. fall vel., shape: mean)",
        )
        _line(
            ax4,
            dat3.SR_M_heymsfield10.sel(shape="mean"),
            label="SR (combined with param. fall vel., shape: mean)",
        )
        for ii, shape in enumerate(shapes):
            _line(
                ax4,
                dat3.SR_M.sel(shape=shape),
                c="C0",
                alpha=0.3,
                lw=1,
                label="individual shapes" if ii == 0 else None,
            )
            _line(ax4, dat3.SR_M_heymsfield10.sel(shape=shape), c="C1", alpha=0.3, lw=1)
        ax4.set_ylabel("SR [mm h$^{-1}$ w.e.]")
        ax4.margins(y=0.2)
        ax4.set_yscale("log")
        ax4.legend(loc="upper left", ncol=1)

        handles = _shadeQualityFlags(
            axs,
            dat2,
            config,
            ["recordingFailed", "obervationsDiffer", "processingFailed"]
            + ["cameraBlocked", "blowingSnow", "sunGlint", "trackingIncomplete"],
        )
        handle = _markRestarts(axs, _restartTimes(events))
        if handle is not None:
            handles.append(handle)
        _formatDayAxes(axs, ff)
        _legendAbove(ax1, handles)
        fig.align_ylabels(axs)
        # do not keep the files open: they are re-opened for the status line
        events1.close()
        events2.close()
        dat2.close()
        dat3.close()
    tools.savefig(
        fig,
        config,
        fOut,
        fnames=ff.listFiles("level3combinedRiming"),
        logoAnchor=logoAnchor,
    )
    return fig


################################################################################
# Track particles quicklook: animated GIFs of the best-scored particle tracks of
# a day (leader | follower), laid out in an HTML grid.


def _trackParticlesSetting(config, key, value):
    """Resolve a "config" parameter value for the track particles quicklook."""
    if isinstance(value, str) and value == "config":
        return config["level1trackParticlesQuicklook"][key]
    return value


class _RoutedZip:
    """Routes every pid to its own block's archive."""

    def __init__(self):
        self.map = {}

    def add(self, pidStr, handle):
        self.map[pidStr] = handle

    def extractnpy(self, pidStr):
        handle = self.map.get(pidStr)
        if handle is None:
            raise KeyError(pidStr)
        return handle.extractnpy(pidStr)


def _archiveFor(st, camera, config, cache, version=__version__):
    """Single image archive whose block start == st for `camera`, cached."""
    import pandas as pd

    key = (camera, np.datetime64(st, "ns"))
    if key in cache:
        return cache[key]
    caseB = pd.to_datetime(str(st)).strftime("%Y%m%d-%H%M%S")
    ff1 = files.FindFiles(caseB, camera, config, version=version)
    znames = ff1.listFiles("imagesL1detect")  # EXACT block
    if len(znames) == 0:
        cache[key] = None
    else:
        # as in createLevel1matchParticlesQuicklook: several versions of the
        # archive can exist, use the canonical name for the current version
        zname = files.FilenamesFromLevel(znames[0], config).fname.imagesL1detect
        try:
            cache[key] = tools.imageZipFile(zname, mode="r")
        except (FileNotFoundError, tools.zipfile.BadZipFile):
            log.warning(f"cannot open {zname}")
            cache[key] = None
    return cache[key]


def _closeArchives(cache):
    """Close all image archives opened by _archiveFor."""
    for handle in cache.values():
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass


def _routedZips(trackDat, config, cache, version=__version__):
    """Leader & follower _RoutedZip: every pid -> its own block's archive."""
    zLeader, zFollower = _RoutedZip(), _RoutedZip()
    for ii in range(trackDat.sizes["fpair_id"]):
        d = trackDat.isel(fpair_id=ii)
        lst = np.datetime64(d.file_starttime.isel(camera=0).values, "ns")
        fst = np.datetime64(d.file_starttime.isel(camera=1).values, "ns")
        zLeader.add(
            "%07i" % int(d.pid.isel(camera=0).values),
            _archiveFor(lst, config.leader, config, cache, version),
        )
        zFollower.add(
            "%07i" % int(d.pid.isel(camera=1).values),
            _archiveFor(fst, config.follower, config, cache, version),
        )
    return zLeader, zFollower


def _padToCommonShape(images, padding_value=150):
    """Center-pad all 2D images to the largest height/width among them."""
    pad_to_length = max(img.shape[0] for img in images)
    pad_to_width = max(img.shape[1] for img in images)
    padded = []
    for img in images:
        len_diff = pad_to_length - img.shape[0]
        width_diff = pad_to_width - img.shape[1]
        pad_width_2d = (
            (len_diff // 2, len_diff - len_diff // 2),
            (width_diff // 2, width_diff - width_diff // 2),
        )
        padded.append(
            np.pad(
                img,
                pad_width=pad_width_2d,
                mode="constant",
                constant_values=padding_value,
            )
        )
    return padded


# crops are enlarged (integer factor, nearest neighbour) so that a panel is about
# this many pixels large, which keeps text and scale bar readable for small
# particles
_PANEL_TARGET = 320


@functools.lru_cache(maxsize=None)
def _font(size):
    """DejaVu Sans, shipped with matplotlib, so it is always there."""
    import matplotlib as mpl
    from PIL import ImageFont

    ttfDir = os.path.join(os.path.dirname(mpl.matplotlib_fname()), "fonts", "ttf")
    return ImageFont.truetype(f"{ttfDir}/DejaVuSans.ttf", size)


def _niceLength(maxLength):
    """Largest 1, 2 or 5 times a power of ten that is <= maxLength."""
    exponent = np.floor(np.log10(maxLength))
    for mantissa in (5, 2, 1):
        if mantissa * 10**exponent <= maxLength:
            return mantissa * 10**exponent


def _saveTrackGif(
    frames,
    labels,
    title,
    fOut,
    umPerPx,
    fps=4,
    showLegend=True,
    upscale=None,
):
    """
    Save the frames of a track as an animated grey scale GIF.

    Parameters
    ----------
    frames : list of list of numpy.ndarray
        For every frame one 2D uint8 image per panel (all of the same shape).
    labels : list of str
        Label of each panel, only used with showLegend.
    title : str
        Title, only used with showLegend.
    fOut : str
        Output file.
    umPerPx : float
        Size of a pixel of the crops in micrometer for the scale bar.
    fps : int, optional
        Frames per second.
    showLegend : bool, optional
        Add title, panel labels and a scale bar (below the first panel). The
        text always has the same size, small crops are enlarged for this. If
        False, only the crops are saved (separated by a small gap).
    upscale : int, optional
        Enlargement factor of the crops. Default: automatic with showLegend,
        1 without.
    """
    from PIL import Image, ImageDraw

    nPanels = len(frames[0])
    H, W = frames[0][0].shape
    if upscale is None:
        upscale = max(1, int(round(_PANEL_TARGET / max(H, W)))) if showLegend else 1
    panelH, panelW = H * upscale, W * upscale
    gap = 4
    panelsW = nPanels * panelW + (nPanels - 1) * gap

    # static part of the frames
    if showLegend:
        fontTitle, fontLabel = _font(16), _font(15)
        draw = ImageDraw.Draw(Image.new("L", (1, 1)))
        titleW = int(draw.textlength(title, font=fontTitle))
        titleH, labelH, footerH = 30, 24, 34
        canvasW = max(panelsW, titleW + 20)
        canvasH = titleH + labelH + panelH + footerH
        x0, y0 = (canvasW - panelsW) // 2, titleH + labelH
    else:
        canvasW, canvasH, x0, y0 = panelsW, panelH, 0, 0
    xPanels = [x0 + i * (panelW + gap) for i in range(nPanels)]

    template = Image.new("L", (canvasW, canvasH), 255)
    if showLegend:
        draw = ImageDraw.Draw(template)
        draw.text((canvasW / 2, 6), title, fill=0, font=fontTitle, anchor="ma")
        for x, label in zip(xPanels, labels):
            draw.text(
                (x + panelW / 2, titleH + 2), label, fill=0, font=fontLabel, anchor="ma"
            )
        # scale bar below the first panel
        length = _niceLength(0.3 * panelW * umPerPx / upscale)
        barPx = int(round(length / umPerPx * upscale))
        text = f"{length / 1000:g} mm" if length >= 1000 else f"{length:g} \u00b5m"
        xb, yb = xPanels[0] + 6, y0 + panelH + footerH // 2
        draw.line((xb, yb, xb + barPx, yb), fill=0, width=3)
        draw.line((xb, yb - 5, xb, yb + 5), fill=0, width=2)
        draw.line((xb + barPx, yb - 5, xb + barPx, yb + 5), fill=0, width=2)
        draw.text((xb + barPx + 10, yb), text, fill=0, font=fontLabel, anchor="lm")

    images = []
    for imgs in frames:
        im = template.copy()
        for x, img in zip(xPanels, imgs):
            pil = Image.fromarray(np.asarray(img, dtype=np.uint8), "L")
            if upscale > 1:
                pil = pil.resize((panelW, panelH), Image.Resampling.NEAREST)
            im.paste(pil, (x, y0))
        images.append(im)
    images[0].save(
        fOut,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),
        loop=0,
    )


def _renderTrackGif(
    trackDat,
    zLeader,
    zFollower,
    title,
    fOut,
    config,
    fps=4,
    minRenderedFrames=3,
):
    """
    Animate one track (leader | follower side by side) and save it as GIF.

    Returns the number of rendered frames; 0 if nothing was saved (missing
    crops, or fewer than minRenderedFrames frames).
    """
    img1_list, img2_list = [], []
    for ii in range(trackDat.sizes["fpair_id"]):
        pid = trackDat.pid.isel(fpair_id=ii).values
        try:
            img1 = zLeader.extractnpy("%07i" % pid[0])
            img2 = zFollower.extractnpy("%07i" % pid[1])
        except KeyError:
            continue
        img1_list.append(img1[:, :, 0])
        img2_list.append(img2[:, :, 0])

    nFrames = len(img1_list)
    if nFrames < max(minRenderedFrames, 1):
        return 0

    padded = _padToCommonShape(img1_list + img2_list)
    frames = list(zip(padded[:nFrames], padded[nFrames:]))
    _saveTrackGif(
        frames,
        ["Leader", "Follower"],
        title,
        fOut,
        umPerPx=1 / config.calibration.slope,
        fps=fps,
    )
    return nFrames


def _makeTrackGifs(
    limDat,
    ranked,
    config,
    outFolder,
    version=__version__,
    maxGifs=None,
    minRenderedFrames=3,
    maxFrames=40,
):
    """
    Write one GIF per track into outFolder, walking `ranked` (per-track table,
    best score first) and stopping once maxGifs GIFs rendered successfully.
    Tracks with missing crops / too few frames are skipped and don't count.
    The frames of a track are ordered by track_step, at most maxFrames
    (first) particles per track are animated.

    Returns the list of written GIF paths, best first.
    """
    import pandas as pd

    dat = limDat
    if dat.sizes.get("fpair_id", 0) == 0:
        log.info("no particles to animate")
        return []

    files_arr = dat["file"].values
    tid_arr = dat.track_id.values
    order = list(ranked[["file", "track_id"]].itertuples(index=False, name=None))

    cache = {}
    gifs = []
    try:
        for fkey, tid in order:
            trackDat = dat.isel(fpair_id=(files_arr == fkey) & (tid_arr == tid))
            if trackDat.sizes.get("fpair_id", 0) == 0:
                continue
            order1 = np.argsort(trackDat.track_step.values, kind="stable")
            trackDat = trackDat.isel(fpair_id=order1[:maxFrames])

            st = trackDat.file_starttime.sel(camera=config.leader).values[0]
            caseB = pd.to_datetime(str(st)).strftime("%Y%m%d-%H%M%S")
            title = f"site: {config.site} | case: {caseB} | track_id: {int(tid)}"
            fpath = os.path.join(outFolder, f"{caseB}_track{int(tid)}.gif")

            zLeader, zFollower = _routedZips(trackDat, config, cache, version)
            nFrames = _renderTrackGif(
                trackDat,
                zLeader,
                zFollower,
                title,
                fpath,
                config,
                minRenderedFrames=minRenderedFrames,
            )
            if nFrames == 0:
                log.info(f"skip track {int(tid)} ({caseB}): too few frames rendered")
                continue

            gifs.append(fpath)
            log.info(f"saved {fpath} ({nFrames} frames)")
            if maxGifs is not None and len(gifs) >= maxGifs:
                break
    finally:
        _closeArchives(cache)
    return gifs


def _writeHtml(fOut, body, config, title):
    """Small standalone html page with VISSSlib file permissions."""
    import html as _html

    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>{_html.escape(title)}</title>
<style>
  body {{ margin: 0; padding: 16px; background: #ffffff;
         font-family: system-ui, -apple-system, sans-serif; }}
  h1 {{ font-size: 15px; font-weight: 600; margin: 0 0 12px; }}
  .grid {{ display: grid; gap: 6px; }}
  .cell {{ margin: 0; text-align: center; }}
  .cell img {{ width: 100%; height: auto; background: #fff; border: 1px solid #eee; }}
</style></head>
<body>
<h1>{_html.escape(title)}</h1>
{body}
</body></html>"""

    tools.createParentDir(fOut, mode=config.dirMode)
    try:
        os.remove(fOut)
    except FileNotFoundError:
        pass
    with open(fOut, "w", encoding="utf-8") as f:
        f.write(doc)
    try:
        os.chmod(fOut, config.fileMode)
    except PermissionError:
        log.warning(f"chmod {config.fileMode} {fOut} failed")
    return fOut


def _writeGifHtml(gifs, fOut, config, title, ncols=4):
    """HTML grid of GIFs, base64-embedded so the page is self-contained."""
    import base64
    import html as _html

    cells = []
    for p in gifs:
        with open(p, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("ascii")
        label = _html.escape(os.path.splitext(os.path.basename(p))[0])
        cells.append(
            f'<figure class="cell"><img loading="lazy" '
            f'src="data:image/gif;base64,{b64}" alt="{label}"></figure>'
        )
    body = (
        f'<div class="grid" style="grid-template-columns: repeat({ncols}, 1fr);">\n'
        + "\n".join(cells)
        + "\n</div>"
    )
    return _writeHtml(fOut, body, config, title)


@tools.loopify
def createLevel1trackParticles(
    case,
    config,
    version=__version__,
    skipExisting=True,
    minBlur="config",
    minSize="config",
    minTrackLength="config",  # drop tracks shorter than this (detections)
    minRenderedFrames="config",  # drop GIFs that render fewer frames than this
    targetGifs="config",  # how many GIFs end up in the html
    candidateMultiplier=20,  # try up to targetGifs * candidateMultiplier tracks
    weightTrackLength="config",
    weightBlur="config",
    weightDmax="config",
    ncols=4,  # columns of the html grid (4 x 3 GIFs with the default targetGifs)
    particleLimitPerFile=30000,
    timedelta=np.timedelta64(1, "D"),
    fOut="default",
):
    """
    Create the track particles quicklook for a given case: every
    matched particle track of the day is scored (track length, blur, Dmax; large
    particles are preferred),
    the best tracks are animated as GIFs (leader | follower) and written into
    a HTML grid.

    Parameters
    ----------
    case : str
        Case identifier, e.g. "YYYYMMDD", or a range "YYYYMMDD-YYYYMMDD"
        (see tools.getCaseRange).
    config : dict or str
        Configuration dictionary or settings file.
    version : str, optional
        Version identifier, by default __version__
    skipExisting : bool, optional
        Whether to skip if output exists and is newer than its parents,
        by default True
    minBlur : float or str, optional
        A track is only used if, in each camera, at least one track step has
        blur > minBlur (the other steps may be blurrier). "config" reads
        config["level1trackParticlesQuicklook"].
    minSize : float or str, optional
        Only particles with Dmax > minSize in both cameras are used. "config"
        as above.
    minTrackLength : int or str, optional
        Minimum number of detections per track. "config" as above.
    minRenderedFrames : int or str, optional
        Minimum number of frames that actually load per GIF. "config" as above.
    targetGifs : int or str, optional
        Number of GIFs in the final html. "config" as above.
    candidateMultiplier : int, optional
        Size of the candidate pool relative to targetGifs, so tracks with
        missing crops can be replaced by the next best ones.
    weightTrackLength, weightBlur, weightDmax : float or str, optional
        Weights of the per-track score, the weighted mean of the percentile
        ranks of track length, mean blur and mean Dmax. "config" as above.
    ncols : int, optional
        Number of columns in the html grid.
    particleLimitPerFile : int or None, optional
        Only look at the first n particles per level1track file.
    timedelta : numpy.timedelta64, optional
        Time window for data selection, by default np.timedelta64(1, "D")
    fOut : str, optional
        Output html path, by default "default" (the quicklook folder). May
        contain "{case}", e.g. "/tmp/test/{case}.html", so day ranges don't
        overwrite each other.

    Returns
    -------
    tuple
        (output_file_path, number_of_gifs), or (None, None) if skipped or
        the data is not complete yet.
    """
    import tempfile

    import pandas as pd
    from tqdm import tqdm

    minBlur = _trackParticlesSetting(config, "minBlur", minBlur)
    minSize = _trackParticlesSetting(config, "minSize", minSize)
    minTrackLength = _trackParticlesSetting(config, "minTrackLength", minTrackLength)
    minRenderedFrames = _trackParticlesSetting(
        config, "minRenderedFrames", minRenderedFrames
    )
    targetGifs = _trackParticlesSetting(config, "targetGifs", targetGifs)
    weightTrackLength = _trackParticlesSetting(
        config, "weightTrackLength", weightTrackLength
    )
    weightBlur = _trackParticlesSetting(config, "weightBlur", weightBlur)
    weightDmax = _trackParticlesSetting(config, "weightDmax", weightDmax)

    camera = config["leader"]
    site = config["site"]
    ff = files.FindFiles(case, camera, config, version)
    if fOut == "default":
        fOut = ff.quicklook.level1trackParticles
    else:
        fOut = fOut.format(case=case)
    date = f"{ff.year}-{ff.month}-{ff.day}"
    title = f"VISSS track particles {config['name']} {date}"

    # ---- guard clauses -----------------------------------------------------
    if skipExisting and tools.checkForExisting(
        fOut,
        level0=ff.listFiles("level0"),
        events=[(ff, "metaEvents")],
        parents=[(ff, "level1track")],
    ):
        return None, None

    has_raw = (len(ff.listFiles("level0")) > 0) or (
        len(ff.listFiles("level0status")) > 0
    )
    if has_raw:
        if len(ff.listFiles("metaFrames")) > len(ff.listFiles("level0")):
            print("DATA TRANSFER INCOMPLETE ", fOut)
            print(
                len(ff.listFiles("level0")),
                "of",
                len(ff.listFiles("metaFrames")),
                "transmitted",
            )
            return None, None

        if (len(ff.listFilesExt("level1track")) == 0) and (
            len(ff.listFiles("level0")) > 0
        ):
            print("NO DATA YET ", fOut)
            return None, None

        if not ff.isCompleteL1track:
            print(
                "NOT COMPLETE YET %i of %i L1track %s "
                % (
                    len(ff.listFilesExt("level1track")),
                    len(ff.listFiles("level0txt")),
                    fOut,
                )
            )
            return None, None
    else:
        # raw level0 is gone (e.g. Ny-Ålesund): rely on the processed files only
        if site != "mosaic" and len(ff.listFilesExt("level1track")) == 0:
            print("NO DATA (no level0 and no level1track)", fOut)
            return None, None
        log.info("no level0 on disk - using processed level1track only")

    def _noFigures(reason):
        log.info(f"{case}: {reason}")
        _writeHtml(fOut, f"<p>no figures for day {date} ({reason})</p>", config, title)
        return fOut, 0

    print("RUNNING open files ", fOut, len(ff.listFiles("metaFrames")))

    # ---- load + filter each level1track file -------------------------------
    dats2 = []
    for fname2 in tqdm(ff.listFilesWithNeighbors("level1track")):
        fname1 = fname2.replace("level1track", "metaFrames")
        try:
            with xr.open_dataset(fname2) as dat2:
                if particleLimitPerFile is not None:
                    dat2 = dat2.isel(pair_id=slice(particleLimitPerFile))
                dat2 = dat2[
                    [
                        "Dmax",
                        "blur",
                        "record_time",
                        "record_id",
                        "Droi",
                        "file_starttime",
                        "matchScore",
                        "track_id",
                        "track_step",
                    ]
                ].load()
        except FileNotFoundError:
            if os.path.isfile(f"{fname2}.nodata"):
                continue
            elif os.path.isfile(f"{fname2}.broken.txt"):
                continue
            elif os.path.isfile(f"{fname2}.notenoughframes"):
                continue
            else:
                raise FileNotFoundError(fname2)

        dat2 = dat2.isel(
            pair_id=(
                (dat2.Dmax > minSize).all("camera")
                & (dat2.matchScore > config.quality.minMatchScore)
            )
        )
        if len(dat2.pair_id) == 0:
            continue

        dat2 = dat2[
            [
                "record_time",
                "record_id",
                "Droi",
                "file_starttime",
                "Dmax",
                "blur",
                "track_id",
                "track_step",
            ]
        ]
        dat2 = dat2.expand_dims(dict(file=[fname1]))
        dat2 = dat2.stack(fpair_id=("file", "pair_id"))
        dats2.append(dat2)

    if len(dats2) == 0:
        return _noFigures("no particles passed the filters")

    limDat = xr.concat(dats2, dim="fpair_id")
    limDat = limDat.isel(
        fpair_id=(limDat.record_time.isel(camera=0).values >= ff.datetime64)
        & (limDat.record_time.isel(camera=0).values < (ff.datetime64 + timedelta))
    )
    if len(limDat.fpair_id) == 0:
        return _noFigures("no particles on this day")

    # very rarely there are duplicate (track_id, track_step) entries
    isDuplicate = pd.DataFrame(
        {
            "file": limDat["file"].values,
            "track_id": limDat["track_id"].values,
            "track_step": limDat["track_step"].values,
        }
    ).duplicated()
    limDat = limDat.isel(fpair_id=~isDuplicate.values)

    # ---- blur criterion per track ------------------------------------------
    # not every particle has to be sharp: a track is kept if each camera has
    # at least one track step with blur > minBlur
    blurCam = limDat["blur"].transpose("camera", "fpair_id").values
    blurMax = (
        pd.DataFrame(
            {
                "file": limDat["file"].values,
                "tid": limDat["track_id"].values,
                "leader": blurCam[0],
                "follower": blurCam[1],
            }
        )
        .groupby(["file", "tid"])[["leader", "follower"]]
        .transform("max")
    )
    sharpTrack = (
        (blurMax["leader"] > minBlur) & (blurMax["follower"] > minBlur)
    ).values
    limDat = limDat.isel(fpair_id=sharpTrack)
    if len(limDat.fpair_id) == 0:
        return _noFigures(f"no tracks with blur > {minBlur} in both cameras")

    # ---- per-track scoring -------------------------------------------------
    tracks = (
        pd.DataFrame(
            {
                "file": limDat["file"].values,
                "track_id": limDat["track_id"].values,
                "blur": limDat["blur"].mean("camera").values,
                "Dmax": limDat["Dmax"].mean("camera").values,
            }
        )
        .groupby(["file", "track_id"])
        .agg(
            length_track=("track_id", "size"),
            blur_mean_track=("blur", "mean"),
            Dmax_mean_track=("Dmax", "mean"),
        )
        .reset_index()
    )
    tracks = tracks[tracks["length_track"] >= minTrackLength]
    if len(tracks) == 0:
        return _noFigures(f"no tracks with length >= {minTrackLength}")

    # Score = weighted mean of the percentile ranks (0..1) of the tracks, so a
    # few extreme tracks do not dominate the scale
    wSum = weightTrackLength + weightBlur + weightDmax
    tracks["score"] = (
        weightTrackLength * tracks["length_track"].rank(pct=True)
        + weightBlur * tracks["blur_mean_track"].rank(pct=True)
        + weightDmax * tracks["Dmax_mean_track"].rank(pct=True)
    ) / wSum

    # best first; candidate pool so failures can be backfilled
    ranked = tracks.sort_values("score", ascending=False).head(
        targetGifs * candidateMultiplier
    )
    isCandidate = pd.MultiIndex.from_arrays(
        [limDat["file"].values, limDat["track_id"].values]
    ).isin(pd.MultiIndex.from_frame(ranked[["file", "track_id"]]))
    candidates = limDat.isel(fpair_id=isCandidate)

    # ---- GIFs (temporary) -> html grid --------------------------------------
    with tempfile.TemporaryDirectory() as tmpDir:
        gifs = _makeTrackGifs(
            candidates,
            ranked,
            config,
            tmpDir,
            version=version,
            maxGifs=targetGifs,
            minRenderedFrames=minRenderedFrames,
        )
        if len(gifs) == 0:
            return _noFigures("no track rendered enough frames")
        if len(gifs) < targetGifs:
            log.warning(f"only {len(gifs)} of {targetGifs} GIF(s) could be made")
        _writeGifHtml(gifs, fOut, config, title, ncols=ncols)

    print("SAVED ", fOut)
    return fOut, len(gifs)


#####################################################################################
# if you have found a nice track in the quicklook, retrieve the individual one


def createLevel1trackParticlesSingle(
    case,
    track_id,
    config,
    version=__version__,
    view="both",  # "both", "leader" or "follower"
    showLegend=True,  # False -> only the image(s), no title/labels/scale bar
    filterLikeQuicklook=False,  # True -> exactly the frames of the quicklook GIF
    fps=4,
    upscale=2,  # only for showLegend=False
    keepGif=False,  # also save the .gif next to the html
    fOut="default",
):
    """
    Save a single, hand-picked particle track (e.g. one found in the
    createLevel1trackParticles quicklook) as its own html page.

    Parameters
    ----------
    case : str
        start time as shown in the GIF titles of the quicklook
    track_id : int
        track_id within that block.
    config : dict or str
        Configuration dictionary or settings file.
    view : {"both", "leader", "follower"}, optional
        Which camera(s) to show, by default "both".
    showLegend : bool, optional
        If False, the GIF contains only the crop(s) and the html page only the
        GIF - no title, camera labels or scale bar. By default True.
    filterLikeQuicklook : bool, optional
        By default all particles of the track are shown. True applies the same
        filters as the quicklook (minBlur, minSize, minMatchScore)
    fps : int, optional
        Frames per second of the GIF.
    upscale : int, optional
        Nearest-neighbour upscaling factor for showLegend=False, by default 2.
    keepGif : bool, optional
        Also save the GIF itself next to the html, by default False.
    fOut : str, optional
        Output html path. "default" puts it into the level1trackParticles
        quicklook folder. May contain "{case}", "{track_id}" and "{view}".

    Returns
    -------
    str
        Path of the written html.
    """
    import base64
    import html as _html
    import re
    import shutil
    import tempfile

    import pandas as pd
    from PIL import Image

    # ---- arguments -------------------------------------------------------------
    if isinstance(config, str):
        config = tools.readSettings(config)

    digits = re.sub(r"\D", "", str(case))
    if len(digits) != 14:
        raise ValueError(
            f"case needs date AND block start time (YYYYMMDD-HHMMSS), got {case!r}"
        )
    caseB = f"{digits[:8]}-{digits[8:]}"
    track_id = int(track_id)
    view = view.lower()
    panels = []  # (label, camera name, camera index)
    if view in ("both", "leader"):
        panels.append(("Leader", config.leader, 0))
    if view in ("both", "follower"):
        panels.append(("Follower", config.follower, 1))
    if len(panels) == 0:
        raise ValueError(f"view must be 'both', 'leader' or 'follower', got {view!r}")

    ff1 = files.FindFiles(caseB, config.leader, config, version=version)
    if fOut == "default":
        folder = os.path.dirname(ff1.quicklook.level1trackParticles)
        suffix = "" if view == "both" else f"_{view}"
        if not showLegend:
            suffix += "_clean"
        fOut = os.path.join(
            folder, f"{config.site}_{caseB}_track{track_id}{suffix}.html"
        )
    else:
        fOut = fOut.format(case=caseB, track_id=track_id, view=view)
    title = f"site: {config.site} | case: {caseB} | track_id: {track_id}"

    # ---- the particles of the track (level1track), ordered by track_step -------
    trackFiles = ff1.listFiles("level1track")
    if len(trackFiles) == 0:
        raise FileNotFoundError(f"no level1track file for {caseB}")
    with xr.open_dataset(trackFiles[0]) as ds:
        inTrack = ds["track_id"].values == track_id
        if not inTrack.any():
            raise ValueError(f"track_id {track_id} not found in {trackFiles[0]}")
        dat = ds[["file_starttime", "Dmax", "blur", "matchScore", "track_step"]].isel(
            pair_id=inTrack
        )
        dat = dat.load()
    dat = dat.isel(pair_id=np.argsort(dat.track_step.values, kind="stable"))
    # (rare) duplicates of track_step
    _, firstII = np.unique(dat.track_step.values, return_index=True)
    dat = dat.isel(pair_id=np.sort(firstII))

    if filterLikeQuicklook:
        minBlur = _trackParticlesSetting(config, "minBlur", "config")
        minSize = _trackParticlesSetting(config, "minSize", "config")
        good = (dat.Dmax > minSize).all("camera") & (
            dat.matchScore > config.quality.minMatchScore
        )
        dat = dat.isel(pair_id=good.values)
        if dat.sizes["pair_id"] > 0 and not (dat.blur.max("pair_id") > minBlur).all():
            log.warning(
                f"track {track_id} would not be used by the quicklook: blur is not "
                f"> {minBlur} in both cameras at any step"
            )
        # the quicklook animates the first 40 particles only
        dat = dat.isel(pair_id=slice(40))
    if dat.sizes["pair_id"] == 0:
        raise ValueError(f"no particles left for track {track_id} in {caseB}")
    log.info(f"track {track_id} ({caseB}): {dat.sizes['pair_id']} particles")

    # ---- load crops: every particle from its own block's image archive ---------
    archives = {}  # (camera, block start) -> imageZipFile or None

    frames = []  # list of [img per panel]
    try:
        for ii in range(dat.sizes["pair_id"]):
            d = dat.isel(pair_id=ii)
            imgs = []
            for _, camera, cam in panels:
                z = _archiveFor(
                    np.datetime64(d.file_starttime.isel(camera=cam).values, "ns"),
                    camera,
                    config,
                    archives,
                    version,
                )
                if z is None:
                    break
                try:
                    imgs.append(
                        z.extractnpy("%07i" % int(d.pid.isel(camera=cam).values))[
                            :, :, 0
                        ]
                    )
                except KeyError:
                    break
            if len(imgs) == len(panels):
                frames.append(imgs)
    finally:
        _closeArchives(archives)

    nFrames = len(frames)
    if nFrames == 0:
        raise RuntimeError(f"no crops found for track {track_id} ({caseB})")

    # center-pad all crops to a common shape (grey, like the quicklook)
    padded = _padToCommonShape([img for f in frames for img in f])
    frames = [padded[i : i + len(panels)] for i in range(0, len(padded), len(panels))]
    with tempfile.TemporaryDirectory() as tmpDir:
        gif = os.path.join(tmpDir, "track.gif")

        # ---- GIF ---------------------------------------------------------------
        _saveTrackGif(
            frames,
            [label for label, _, _ in panels],
            title,
            gif,
            umPerPx=1 / config.calibration.slope,
            fps=fps,
            showLegend=showLegend,
            upscale=None if showLegend else upscale,
        )

        # ---- html (GIF base64-embedded -> self-contained) ----------------------
        with open(gif, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("ascii")
        img = f'<img src="data:image/gif;base64,{b64}" alt="{_html.escape(title)}">'
        if showLegend:
            body = f"<h1>{_html.escape(title)}</h1>\n{img}"
            css = (
                "body { margin: 0; padding: 16px; font-family: system-ui, sans-serif; }"
                " h1 { font-size: 15px; font-weight: 600; margin: 0 0 12px; }"
                " img { display: block; width: 100%; max-width: 1000px; height: auto; }"
            )
        else:
            body = img
            css = (
                "body { margin: 0; }"
                " img { display: block; max-width: 100%; height: auto;"
                " image-rendering: pixelated; }"
            )
        doc = (
            '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">\n'
            f"<title>{_html.escape(title)}</title>\n"
            f"<style> {css} body {{ background: #ffffff; }} </style></head>\n"
            f"<body>\n{body}\n</body></html>"
        )

        tools.createParentDir(fOut, mode=config.dirMode)
        written = [fOut]
        try:
            os.remove(fOut)
        except FileNotFoundError:
            pass
        with open(fOut, "w", encoding="utf-8") as f:
            f.write(doc)
        if keepGif:
            gifOut = os.path.splitext(fOut)[0] + ".gif"
            shutil.copyfile(gif, gifOut)
            written.append(gifOut)
        for p in written:
            try:
                os.chmod(p, config.fileMode)
            except PermissionError:
                log.warning(f"chmod {config.fileMode} {p} failed")

    print("SAVED ", fOut, f"({nFrames} frames)")
    return fOut
