"""delete .broken.txt sentinels created by import-time environment errors (e.g. while the conda env was
updated), so that the normal reprocessing regenerates them. Only files whose content matches an
environment-error pattern are removed; real failures are left alone.
usage: clear_env_broken.py <minutes> [root-glob ...]"""
import glob, re, sys, os
mins = int(sys.argv[1]); roots = sys.argv[2:] or ["/projekt1/ag_maahn/data_obs_nobackup/*/visss*_1.2"]
pat = re.compile(r"ImportError|has no attribute '__version__'|undefined symbol|libtiff|cannot open shared object|ModuleNotFoundError")
import subprocess, time
n = 0; files = []
for r in roots:
    for lv in ["level1match", "level1track", "level2match", "level2track", "level3combinedRiming", "level2detect", "level1detect"]:
        out = subprocess.run(f"find {r}/{lv} -name '*.nc.broken.txt' -mmin -{mins} 2>/dev/null", shell=True, capture_output=True, text=True).stdout.split()
        files += out
for f in files:
    try:
        if pat.search(open(f, errors="ignore").read()):
            os.remove(f); n += 1; print("removed", f.split("/data_obs_nobackup/")[1])
    except FileNotFoundError:
        pass
print("removed", n, "of", len(files), "recent broken files")
