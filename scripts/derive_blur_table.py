#!/usr/bin/env python
"""
Derive a deployment specific blur threshold table for ``level2detect``.

The table (see ``VISSSlib.blurTables`` and ``level2.blurTable``) is a stand-in for the
observation volume of the single camera product: a particle is only counted if its
blur (Laplacian variance) is above a size dependent threshold. The threshold is chosen
such that the volume corrected ``level1detect`` size distribution agrees with the
``level1match`` size distribution, which is limited by the stereo observation volume.

Two steps, because the first one is slow (about 1 h per day of data):

1. ``collect``: for a list of snowfall days, build per camera the 2D histogram
   (Dmax in pixels x log10 blur) of all detected particles, corrected for the
   observation volume loss of the single camera, and the size distribution of the matched
   particles. Run it several times with different days (in parallel, 3 days per run is a
   good size) and write each result to its own pickle.

2. ``derive``: pool the pickles, derive the table, check it by leaving each pickle out
   in turn, and write the table as text to paste into ``blurTables.BLUR_TABLES``.

Example::

    python scripts/derive_blur_table.py collect settings.yaml 20240122,20240214,20240322 calib_b0.pkl &
    python scripts/derive_blur_table.py collect settings.yaml 20240420,20240221,20240316 calib_b1.pkl &
    ...
    python scripts/derive_blur_table.py derive hyytiala2_v3 calib_b*.pkl -o table.txt

Choose 10-15 days with plenty of snow spread over the whole deployment (the table
should not depend on one weather situation), without badData or known problems, and
with level1detect, level1match and level2match complete. The result is checked by
the hold-out ratio ``R_pass`` (volume corrected detections above the threshold divided
by the matched particles, 8-40 px, per camera) when one pickle is left out. It should
be close to 1 (0.8-1.2); a clearly different value for one pickle means that its days
differ too much from the others.

The raw blur is used, the table set with ``level2.blurTable`` in the settings is not
involved. Blocked and blowing snow periods are removed from level1detect as in
level2detect, otherwise detections and matches are not comparable.
"""

import argparse
import pickle
import sys
import warnings

import numpy as np

NS = 299  # number of sizes (Dmax in pixels, 0 ... NS-1) of the table
D0 = 8  # sizes below D0 take the value at D0
NMIN = 30  # minimum number of matched particles per size to constrain the table
LB = np.linspace(0, 6, 241)  # bin edges of log10(blur)


def collect(settings, days, output):
    import xarray as xr

    import VISSSlib

    warnings.filterwarnings("ignore")
    c = VISSSlib.tools.readSettings(settings)
    fm, fd, fo, fe, fe2, fL2 = [], [], [], [], [], []
    for d in days:
        fl = VISSSlib.files.FindFiles(d, c.leader, c)
        fm += fl.listFiles("level1match")
        fe += fl.listFiles("metaEvents")
        fL2 += fl.listFiles("level2match")
        fe2 += VISSSlib.files.FindFiles(d, c.follower, c).listFiles("metaEvents")
    for f in fm:
        f1 = VISSSlib.files.FilenamesFromLevel(f, c)
        fd.append(f1.fname.level1detect)
        fo += f1.filenamesOtherCamera(graceInterval=-10)
    print(settings, len(fm), len(fd), len(fo), flush=True)

    match = VISSSlib.tools.open_mflevel1match(
        fm, c, datVars=["Dmax", "blur", "matchScore"]
    )
    det = {
        "leader": VISSSlib.tools.open_mflevel1detect(
            fd, c, datVars=["Dmax", "blur", "capture_time"], skipFixes="all"
        ),
        "follower": VISSSlib.tools.open_mflevel1detect(
            fo, c, datVars=["Dmax", "blur", "capture_time"], skipFixes="all"
        ),
    }
    meta = {"leader": xr.open_mfdataset(fe), "follower": xr.open_mfdataset(fe2)}
    L2 = xr.open_mfdataset(fL2)
    for cam in det:
        for m in ("follower", "leader"):
            det[cam] = VISSSlib.tools.removeBlockedBlowingData(det[cam], meta[m], c)
    match = match.isel(fpair_id=(match.matchScore > 1e-3))

    # observation volume of the single camera relative to the ideal one: the
    # detected particles are weighted with it, the matched ones are not
    Db, vol = VISSSlib.distributions._estimateVolumes(
        c.frame_width,
        c.frame_height,
        c.level2.correctForSmallOnes,
        0.0,
        0.0,
        0.0,
        tuple(np.arange(300)),
        tuple(),
        tuple(),
        tuple(),
        nSteps=150,
    )
    ideal = vol / c.calibration.slope**3 / 1e6**3
    vloss = (L2.obs_volume.mean("time") / ideal).values

    out = {"vloss": vloss}
    bins = np.arange(300)
    edgesD = np.arange(301) - 0.5
    for cam in ("leader", "follower"):
        mD = match.sel(camera=c[cam]).Dmax.values
        mB = match.sel(camera=c[cam]).blur.values
        dD = det[cam].Dmax.values
        dB = det[cam].blur.values
        mN, _ = np.histogram(mD, bins=bins)
        Hdet, _, _ = np.histogram2d(
            np.clip(np.around(dD), 0, 299),
            np.log10(np.clip(dB, 1, None)),
            bins=[edgesD, LB],
        )
        Hmatch, _, _ = np.histogram2d(
            np.clip(np.around(mD), 0, 299),
            np.log10(np.clip(mB, 1, None)),
            bins=[edgesD, LB],
        )
        out[cam] = {"match": mN, "Hdet": Hdet, "Hmatch": Hmatch, "lb": LB}
        # without any blur threshold: detections / matches (8-40 px)
        aN, _ = np.histogram(dD, bins=bins)
        print(
            cam,
            "detections/matches without threshold (8-40 px):",
            aN[8:40].sum() * np.nanmean(vloss[8:40]) / mN[8:40].sum(),
            flush=True,
        )
    pickle.dump(out, open(output, "wb"))


def pool(batches):
    """sum volume corrected detect histograms and match counts per camera"""
    P = {}
    for cam in ("leader", "follower"):
        Hd = sum(b[cam]["Hdet"][:NS] * b["vloss"][:NS, None] for b in batches)
        m = sum(b[cam]["match"] for b in batches)
        P[cam] = dict(Hd=Hd, m=m, lb=batches[0][cam]["lb"])
    return P


def thresholdPerSize(Hd, m, lb):
    """
    Threshold per size: the volume corrected number of detected particles divided by
    the number of matched ones is the factor by which the detections are too many; the
    fraction 1 - 1/factor of the detected particles with the lowest blur is removed.
    """
    thr = np.full(NS, np.nan)
    n = Hd.sum(1)
    for D in range(NS):
        if n[D] <= 0 or m[D] < 5:
            continue
        fac = n[D] / m[D]
        q = max(0.0, 1 - 1 / fac) if fac > 1 else 0.0
        c = np.cumsum(Hd[D]) / n[D]
        i = min(np.searchsorted(c, q), len(lb) - 2)
        lo = c[i - 1] if i > 0 else 0
        hi = c[i]
        f = 0 if hi <= lo else (q - lo) / (hi - lo)
        # interpolate within the bin on the log scale
        thr[D] = 10 ** (lb[i] + f * (lb[i + 1] - lb[i]))
    return thr


def smooth(t, valid):
    """
    Median over +-2 sizes, non-increasing from D0 on, sizes below D0 take the value at
    D0 and sizes beyond the last size with enough matches keep the last value.
    """
    idx = np.arange(NS)
    ok = np.isfinite(t) & valid
    last = idx[ok].max()
    ts = np.full(NS, np.nan)
    for D in range(D0, last + 1):
        w = t[max(D0, D - 2) : D + 3]
        w = w[np.isfinite(w)]
        ts[D] = np.median(w) if len(w) else np.nan
    g = np.isfinite(ts)
    ts[D0 : last + 1] = np.interp(idx[D0 : last + 1], idx[g], ts[g])
    ts[D0 : last + 1] = np.minimum.accumulate(ts[D0 : last + 1])
    ts[:D0] = ts[D0]
    ts[last + 1 :] = ts[last]
    return ts


def makeTable(P):
    """mean of leader and follower threshold, smoothed"""
    per = {cam: thresholdPerSize(P[cam]["Hd"], P[cam]["m"], P[cam]["lb"]) for cam in P}
    valid = (P["leader"]["m"] >= NMIN) & (P["follower"]["m"] >= NMIN)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean = np.nanmean([per["leader"], per["follower"]], axis=0)
    return smooth(mean, valid)


def rPass(batch, tab):
    """volume corrected detections above the threshold / matches (8-40 px)"""
    r = {}
    for cam in ("leader", "follower"):
        o = batch[cam]
        H = o["Hdet"][:NS] * batch["vloss"][:NS, None]
        edges = 10 ** o["lb"][:-1]  # lower edge of each blur bin
        tp = sum(H[D][edges >= tab[D]].sum() for D in range(8, 40))
        r[cam] = tp / o["match"][8:40].sum()
    return r


def formatTable(name, tab, width=88):
    """text to paste into blurTables.BLUR_TABLES"""
    lines = [f'    "{name}": [']
    line = "       "
    for x in np.round(tab).astype(int):
        item = f" {x},"
        if len(line) + len(item) > width:
            lines.append(line.rstrip())
            line = "       "
        line += item
    lines.append(line.rstrip())
    lines.append("    ],")
    return "\n".join(lines) + "\n"


def derive(name, files, output):
    B = [pickle.load(open(f, "rb")) for f in files]
    print(name, len(B), "pickles")
    if len(B) > 1:
        for k in range(len(B)):
            tab = makeTable(pool([b for j, b in enumerate(B) if j != k]))
            r = {c: round(float(v), 2) for c, v in rPass(B[k], tab).items()}
            print("hold out", files[k], "R_pass", r)
    tab = makeTable(pool(B))
    sizes = (8, 10, 12, 15, 20, 25, 30, 40, 50, 70, 100, 150)
    print("table (Dmax px: threshold)", {D: int(round(tab[D])) for D in sizes})
    text = formatTable(name, tab)
    if output:
        open(output, "w").write(text)
        print("written to", output)
    else:
        print(text)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect", help="histograms for some days (slow)")
    c.add_argument("settings", help="settings yaml of the deployment")
    c.add_argument("days", help="comma separated days, YYYYMMDD")
    c.add_argument("output", help="pickle to write")
    d = sub.add_parser("derive", help="table from the pickles of collect")
    d.add_argument("name", help="name of the table, e.g. hyytiala2_v3")
    d.add_argument("pickles", nargs="+", help="pickles written by collect")
    d.add_argument("-o", "--output", help="text file for the table (default: stdout)")
    a = p.parse_args(argv)
    if a.command == "collect":
        collect(a.settings, a.days.split(","), a.output)
    else:
        derive(a.name, a.pickles, a.output)


if __name__ == "__main__":
    sys.exit(main())
