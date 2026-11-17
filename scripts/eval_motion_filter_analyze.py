#!/usr/bin/env python
"""Summarize eval_motion_filter.py output per VISSS generation.

usage: eval_motion_filter_analyze.py RUNSDIR
RUNSDIR contains one sub directory per settings file (named like the yaml).
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

from VISSSlib import files, tools

GEN = {
    "sail_v1": "VISSS1", "eriswil_v1": "VISSS1", "lim24_v1": "VISSS1",
    "gochang_v1": "VISSS1", "incheon_v1": "VISSS1",
    "nyaalesund_v4": "VISSS2", "nyaalesund_v5": "VISSS2", "nyaalesund_v6": "VISSS2",
    "hyytiala2_v3": "VISSS3", "hyytiala2_v4": "VISSS3", "hyytiala2_v5": "VISSS3",
}
CONFDIR = "/projekt1/ag_maahn/VISSS_config"


def load(runs):
    F, P = [], []
    for dep in sorted(os.listdir(runs)):
        d = os.path.join(runs, dep)
        if not os.path.isdir(d):
            continue
        for ff in sorted(glob.glob(f"{d}/*_frames.csv")):
            f = pd.read_csv(ff)
            if len(f) == 0:
                continue
            key = os.path.basename(ff).replace("_frames.csv", "")
            f["file"] = key
            f["dep"] = dep
            F.append(f)
            p = pd.read_csv(ff.replace("_frames.csv", "_parts.csv"))
            p["file"] = key
            p["dep"] = dep
            P.append(p)
    F = pd.concat(F, ignore_index=True)
    P = pd.concat(P, ignore_index=True)
    F["gen"] = F.dep.map(GEN)
    P["gen"] = P.dep.map(GEN)
    return F, P


def neighbourDistance(runs, F):
    """capture_id distance from each non-moving status frame to the next
    recorded frame (any thread) before and after."""
    out = {}
    for dep in F.dep.unique():
        config = tools.readSettings(f"{CONFDIR}/{dep}.yaml")
        for l0 in open(f"{runs}/{dep}/files.txt").read().split():
            key = os.path.basename(l0).replace(".txt", "")
            sub = F[(F.file == key) & (~F.moving)]
            if len(sub) == 0:
                continue
            meta = xr.open_dataset(files.Filenames(l0, config).fname.metaFrames)
            cid = np.sort(meta.capture_id.values)
            for i, c in zip(sub.index, sub.capture_id.values):
                j = np.searchsorted(cid, c)
                before = c - cid[j - 1] if j > 0 else 9999
                after = cid[j + 1] - c if j + 1 < len(cid) else 9999
                out[i] = min(before, after)
    return pd.Series(out)


def lossTable(F, P, by, cond=None, nBoot=1000, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    PP = P if cond is None else P[cond]
    for g, f in F.groupby(by):
        p = PP[PP[by] == g]
        nFrames = f.groupby("file").moving.agg(["sum", "size"])
        nPart = p.groupby(["file", "moving"]).size().unstack(fill_value=0)
        nPart = nPart.reindex(index=nFrames.index, columns=[False, True], fill_value=0)
        tot = nPart.sum()
        loss = tot[False] / max(tot.sum(), 1)
        # bootstrap over files
        bs = []
        arr = nPart.values
        for _ in range(nBoot):
            ii = rng.integers(0, len(arr), len(arr))
            s = arr[ii].sum(0)
            bs.append(s[0] / max(s.sum(), 1))
        nNon = (~f.moving).sum()
        nMov = f.moving.sum()
        rows.append(
            {
                by: g,
                "files": len(nFrames),
                "statusFrames": len(f),
                "fracNotMoving": nNon / len(f),
                "partPerMovingFrame": tot[True] / max(nMov, 1),
                "partPerNonMovingFrame": tot[False] / max(nNon, 1),
                "nPartNonMoving": int(tot[False]),
                "nPartMoving": int(tot[True]),
                "lostFraction": loss,
                "lost_p05": np.percentile(bs, 5),
                "lost_p95": np.percentile(bs, 95),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    runs = sys.argv[1]
    F, P = load(runs)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.precision", 4)
    for name, cond in [
        ("all detections", None),
        ("level2 quality (blur threshold passed)", P.passBlur.astype(bool)),
        ("level2 quality and Dmax >= 10 px", P.passBlur.astype(bool) & (P.Dmax >= 10)),
    ]:
        print("=" * 30, name)
        print(lossTable(F, P, "gen", cond).to_string(index=False))
        print(lossTable(F, P, "dep", cond).to_string(index=False))
    print("=" * 30, "Dmax distribution [px] by moving flag (all detections)")
    print(P.groupby(["gen", "moving"]).Dmax.describe(percentiles=[0.1, 0.5, 0.9]))
    print(P.groupby(["gen", "moving"]).pixMin.describe(percentiles=[0.1, 0.5, 0.9]))
    print(P.groupby(["gen", "moving"]).blur.describe(percentiles=[0.1, 0.5, 0.9]))
    P.to_csv(os.path.join(runs, "all_parts.csv"), index=False)
    F.to_csv(os.path.join(runs, "all_frames.csv"), index=False)
