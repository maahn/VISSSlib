---
name: qc-playbook
description: Run the VISSS quality-control playbook against a deployment settings YAML — structural scan, resubmit broken files via the task queue, re-scan, sample-then-escalate matchScore/Z-residual checks, classify remaining findings as accepted-unfixable vs. genuinely new, confirm the level1track/level2match/level2track DAG has settled, and deliver a broken-periods table. Use this whenever the user asks to "run QC", "QC scan", "quality control", "check for broken/missing files", or "reprocess broken" against a VISSSlib config/settings YAML, or names a deployment (e.g. "run the QC playbook on hyytiala2_v3") — even if they don't spell out the individual steps, since this skill already encodes the ordering and the repo-specific idioms (task queue, DataProductRange, memory-keyed accepted-unfixable list) that make doing it ad hoc error-prone.
---

# VISSS QC playbook

A repeatable audit-and-reprocess pass over one deployment's settings YAML. The
goal is to separate three outcomes for every finding: fixed by reprocessing,
already-known-and-accepted as unfixable, or genuinely new (needs a human to
look at it). Nothing here is a black box — every step calls into
`scripts/qc_report.py` / `scripts/reprocess_broken.py` / `products.py`, which
are the actual source of truth and may have grown flags or categories since
this was written. Skim `python scripts/qc_report.py --help` and that script's
own module docstring before a run if it's been a while — don't rely on this
file's paraphrase of it.

## Step 0 — load prior status from memory, don't re-derive it

Before touching any scripts, look in the user's memory directory
(`/Users/mmaahn/.claude/projects/-Users-mmaahn-projectsSrv-VISSSlib/memory/`)
for `project_<config-basename>_qc_status.md`, where `<config-basename>` is the
settings YAML's filename without `.yaml` (e.g. `hyytiala2_v3.yaml` ->
`project_hyytiala2_v3_qc_status.md`). If it exists, it already records this
deployment's list of accepted-unfixable periods and any deployment-specific
reprocessing quirks — read it and use it as ground truth for Step 5's
classification instead of re-deriving what "accepted unfixable" means from
scratch.

If it doesn't exist, this is the first QC pass for this config: proceed with
the general playbook below, and write that memory file at the end of the run
(Step 7) so future passes don't repeat this analysis.

Also check `vissslib_production_deployment.md` in memory for the current task
queue directory and conda env — these vary and shouldn't be hardcoded.

## Step 1 — structural scan

```bash
python scripts/qc_report.py <config>.yaml --out /tmp/qc_<config-basename>.csv
```

This is fast (no file-opening) and reports: `missing`, `broken`,
`nodata_suspect`, `duplicate`, `reduced_coverage` (see the script's docstring
for exactly what each means — it's detailed and deliberately not repeated
here). Known-bad periods already listed in `config.badData` are excluded by
default; don't pass `--include-known-bad` unless the user specifically wants
those re-surfaced.

## Step 2 — resubmit everything broken via the task queue

For each affected `<camera>_<product>` pair the scan turned up (e.g.
`leader_level1match`, `follower_level1detect`):

```bash
python scripts/reprocess_broken.py \
    --settings <config>.yaml \
    --products <camera>_<product> [<camera>_<product> ...] \
    --case <case-range-or-0-for-full-range> \
    --queue <task-queue-dir-from-memory>
```

This clears `.broken.txt` sentinels and resubmits via
`products.DataProductRange(...).cleanUpBroken()` +
`.submitCommands()` — cheap and safe because `skipExisting=True` means only
what's actually missing gets regenerated. It deliberately does **not** touch
SLURM directly (no `sbatch`/`squeue`/`scancel`) — existing workers drain the
queue on their own.

**2-worker SLURM cap**: during a QC pass, keep concurrent SLURM worker jobs
draining this queue capped at 2 (not the full batch-processing worker count)
— this is a manual constraint on however many `worker` jobs are running
against the queue, since QC reprocessing is exploratory rather than a full
production backfill and shouldn't compete hard with other cluster load. If
it's unclear how many workers are currently running or how to cap them on
this cluster, ask the user rather than guessing at `sbatch`/`squeue` usage.

Don't wait inline for the queue to drain by polling tightly — check back
after a reasonable interval, or ask the user to confirm workers have caught
up, before moving to Step 3.

## Step 3 — re-scan

Once the queue has drained, re-run the Step 1 scan. Anything that dropped out
of the `broken`/`missing` categories was fixed by reprocessing. Anything
still there survived a fresh reprocessing attempt with current code — that's
the real remainder to classify.

## Step 4 — matchScore / Z-residual check: sample first, escalate only if non-trivial

```bash
python scripts/qc_report.py <config>.yaml --matchscore-check sample --out /tmp/qc_<config-basename>_matchscore.csv
```

This opens a random sample of real `level1match` files per level/camera/day
and flags `matchscore_suspect` (median matchScore below threshold, often
because the file had too few pairs for matchParticles' own quality gate to
ever run) and `z_sigma_suspect` (Z-consistency residual too wide — see Step 5).

Only escalate to `--matchscore-check all` (which opens every level1match
file — slow) if the sample result looks non-trivial: findings spread across
many distinct days/periods rather than one or two isolated files, or a rate
that suggests the sample undercounts a real systemic problem. A sample
turning up nothing, or one or two isolated one-off files, does not warrant
the full scan.

## Step 5 — classify the remainder

For everything still outstanding after Steps 3–4, sort into:

- **Accepted-unfixable**: a `z_sigma_suspect` finding — a real, non-broken
  `level1match` file whose Z-consistency residual is *wide* rather than
  *biased*. Per `matching.zResidualSigma`'s empirical basis, a rotation
  refit can correct a biased residual but cannot shrink a genuinely wide
  one, so a wide sigma that survives reprocessing is the expected signature
  of real correspondence ambiguity, not a new bug — it does not need further
  investigation. Cross-check against the prior periods already listed in
  `project_<config-basename>_qc_status.md` (Step 0) if that file existed.

- **Genuinely new — investigate**: anything else that didn't resolve:
  persistent `broken`/`missing`/`matchscore_suspect`/`nodata_suspect`
  findings, or a `z_sigma_suspect` in a period/pattern not already covered
  by the memory file. Before treating it as a mystery, check whether it
  matches a known bug class these narrower scripts already handle:
  - `scripts/relabel_missing_movies_broken.py` — a `.nodata` sentinel that
    actually describes a raw-data-availability problem ("movie file... not
    found", transfer/corruption language) rather than confirmed
    no-precipitation.
  - `scripts/fix_level1match_rotation_seeds.py` — level1match failing on
    "matchScore smaller than minMatchScore" because metaRotation never got
    a good seed for that day.
  Only escalate to genuinely open-ended investigation if it matches neither.

## Step 6 — confirm the dependent levels settle

After reprocessing, confirm `level1track`/`level2match`/`level2track` have
caught up with whatever level1detect/level1match/metaRotation changes were
made (level2track additionally depends on level2match — see AI.md):

```python
from VISSSlib import products
for level in ("level1track", "level2match", "level2track"):
    p = products.DataProductRange(level, case, "<config>.yaml", queue, camera=leader)
    print(level, p.generateAllCommands())  # empty list == settled
```

An empty command list for a level means it's fully caught up with its
parents for that case range. A non-empty list means something upstream still
needs to finish draining through the queue — don't call the pass done until
these settle.

## Deliverable — broken-periods table

Report a concise table, not the raw CSV dump:

| level | camera | case/period | category | status | note |
|---|---|---|---|---|---|

Where `status` is one of `reprocessed-fixed`, `accepted-unfixable`, or
`new-investigated`, and `note` is a one-line reason (e.g. "z-sigma 5.8 after
refit, matches known-unfixable pattern" or "seed missing, see
fix_level1match_rotation_seeds.py output").

## Step 7 — memory update policy

Only write or update `project_<config-basename>_qc_status.md` if this pass
found something **not already recorded there** — a newly-confirmed
accepted-unfixable period, a reprocessing quirk specific to this deployment,
or (if the file didn't exist yet) the full first-pass summary. Don't restate
findings that memory file already covers; that just adds noise for the next
read.
