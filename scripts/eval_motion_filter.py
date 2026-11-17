#!/usr/bin/env python
"""
Evaluate the acquisition-side motion filter (storage_worker_cv.h) with the
unconditional 10 s status frames.

Since VISSS data-acquisition 84b779a (2022-11-08) storage thread 0 writes the
first frame of every full 10 s second (capture time % 10 == 0) regardless of
the motion filter. These status frames are an unbiased sample of *all* frames.
Frames among them that fail the motion criterion (the "status-only" frames)
would not have been stored otherwise, and level1detect skips them as well.

For a 10 min file this script
  1. selects the status frames from metaFrames (thread 0, first frame of a
     second divisible by 10),
  2. classifies them as moving / not moving with the acquisition criterion
     (cumulative changed-pixel histogram >= max(minMovingPixel / 2**l, 2)),
  3. runs VISSSlib particle detection on exactly these frames (and only
     these, so moving and non-moving status frames are treated identically),
  4. stores one row per status frame and one row per detected particle.

The fraction of particle sightings on non-moving status frames among all
status frame sightings estimates the fraction of particle sightings lost to
the motion filter.

usage:
  eval_motion_filter.py sample SETTINGS NFILES OUTDIR [SEED [MINSIZE]]
  eval_motion_filter.py run SETTINGS LEVEL0TXT OUTDIR
"""

import glob
import os
import random
import sys
import types

import numpy as np
import pandas as pd
import xarray as xr

import VISSSlib
from VISSSlib import detection, distributions, files, tools

# acquisition defaults: minMovingPixel = 20, threshold of histogram bin l is
# minMovingPixel / 2**l but at least 2 (float compare against int counts)
MIN_MOVING_PIXEL = 20


def acqThresholds(nBins):
    return np.maximum(MIN_MOVING_PIXEL / 2.0 ** np.arange(nBins), 2)


def statusFrames(meta, fps, nThreads):
    """Index (into meta.capture_time) of the 10 s status frames of thread 0."""
    t = pd.to_datetime(meta.capture_time.values)
    thread = meta.nThread.values
    sec = t.floor("s")
    frac = (t - sec).total_seconds()
    isT0 = thread == 0
    cand = np.where(isT0 & (sec.second % 10 == 0))[0]
    out = []
    seen = set()
    # frame period of thread 0, allow one dropped frame
    maxFrac = 2.5 * max(nThreads, 1) / fps
    for ii in cand:
        if sec[ii] in seen:
            continue
        seen.add(sec[ii])
        if frac[ii] <= maxFrac:
            out.append(ii)
    return np.array(out, dtype=int)


def runFile(settings, fnameL0, outDir):
    config = tools.readSettings(settings)
    fn = files.Filenames(fnameL0, config)
    outBase = os.path.join(outDir, os.path.basename(fnameL0).replace(".txt", ""))
    if os.path.isfile(outBase + "_frames.csv"):
        print("exists", outBase)
        return

    nThreads = config.nThreads or 1
    meta = xr.open_dataset(fn.fname.metaFrames).load()
    idx = statusFrames(meta, config.fps, nThreads)
    if len(idx) == 0:
        print("no status frames", fnameL0)
        pd.DataFrame().to_csv(outBase + "_frames.csv")
        return
    nmp = meta.nMovingPixel.values[idx]
    thr = acqThresholds(nmp.shape[1])
    moving = (nmp >= thr).any(1)
    # first frame of a file (or of a thread file) is always stored, flag it
    firstOfFile = meta.capture_id.values[idx] == meta.capture_id.values[0]

    sub = meta.isel(capture_time=idx)

    # detectParticles opens metaFrames itself -> hand it our subset
    realXr = detection.xr
    fake = types.SimpleNamespace(**{k: getattr(realXr, k) for k in dir(realXr) if not k.startswith("__")})

    def open_dataset(path, *args, **kwargs):
        if os.path.basename(path) == os.path.basename(fn.fname.metaFrames):
            return sub.copy()
        return realXr.open_dataset(path, *args, **kwargs)

    fake.open_dataset = open_dataset
    config.level1detect.minMovingPixels = [0] * nmp.shape[1]
    config.level1detect.writeImg = False
    # badData periods are irrelevant for this evaluation (and a date-only
    # start currently breaks tools.isBadPeriod for per-file levels)
    config.badData = None
    detection.xr = fake
    try:
        res = detection.detectParticles(
            fnameL0, config, writeNc=False, skipExisting=False
        )
    finally:
        detection.xr = realXr
    if isinstance(res, tuple):
        res = res[0]

    frames = pd.DataFrame(
        {
            "capture_time": sub.capture_time.values,
            "capture_id": sub.capture_id.values,
            "moving": moving,
            "firstOfFile": firstOfFile,
        }
    )
    for ii, th in enumerate(meta.nMovingPixelThresh.values):
        frames[f"nmp{th}"] = nmp[:, ii]

    if isinstance(res, xr.Dataset) and len(res.pid) > 0:
        parts = pd.DataFrame(
            {
                "capture_time": res.capture_time.values,
                "capture_id": res.capture_id.values,
                "Dmax": res.Dmax.values,
                "blur": res.blur.values,
                "area": res.area.values,
                "x": res.position_centroid.sel(dim2D="x").values,
                "y": res.position_centroid.sel(dim2D="y").values,
            }
        )
        for v in ["pixMin", "pixMean", "pixMax", "pixStd"]:
            if v in res:
                parts[v] = res[v].values
        try:
            blurTable = config.level2.blurTable
        except KeyError:
            blurTable = None
        blurThresh = distributions._blurThreshold(config.visssGen, blurTable)
        D = np.clip(np.around(parts.Dmax.values).astype(int), 0, len(blurThresh) - 1)
        parts["passBlur"] = (parts.blur.values >= blurThresh[D]) & (parts.Dmax <= 350)
        parts = parts.merge(
            frames[["capture_id", "moving"]], on="capture_id", how="left"
        )
    else:
        parts = pd.DataFrame(
            columns=["capture_time", "capture_id", "Dmax", "blur", "area", "x", "y",
                     "passBlur", "moving"]
        )
    parts.to_csv(outBase + "_parts.csv", index=False)
    frames.to_csv(outBase + "_frames.csv", index=False)
    print("done", outBase, len(frames), moving.sum(), len(parts))


def sample(settings, nFiles, outDir, seed=0, minSize=50e3):
    """Random sample of leader level0 txt files that have a level1detect file
    larger than minSize bytes (i.e. some precipitation)."""
    config = tools.readSettings(settings)
    start = pd.Timestamp(max(str(config.start), "2022-11-15"))
    end = pd.Timestamp.now().normalize() - pd.Timedelta(days=30) if config.end == "today" else pd.Timestamp(str(config.end))
    cands = []
    for day in pd.date_range(start, end, freq="D"):
        case = day.strftime("%Y%m%d")
        ff = files.FindFiles(case, config.leader, config)
        for f in ff.listFiles("level1detect"):
            try:
                if os.path.getsize(f) > minSize:
                    cands.append(f)
            except OSError:
                pass
    print(len(cands), "candidate files")
    rng = random.Random(seed)
    rng.shuffle(cands)
    out = []
    for f in cands:
        fl = files.FilenamesFromLevel(f, config)
        l0 = fl.fname.level0txt
        if os.path.isfile(l0) and os.path.isfile(fl.fname.metaFrames):
            out.append(l0)
        if len(out) >= nFiles:
            break
    os.makedirs(outDir, exist_ok=True)
    with open(os.path.join(outDir, "files.txt"), "w") as fh:
        fh.write("\n".join(out) + "\n")
    print(len(out), "files sampled")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "sample":
        sample(sys.argv[2], int(sys.argv[3]), sys.argv[4],
               int(sys.argv[5]) if len(sys.argv) > 5 else 0,
               float(sys.argv[6]) if len(sys.argv) > 6 else 50e3)
    elif mode == "run":
        runFile(sys.argv[2], sys.argv[3], sys.argv[4])
