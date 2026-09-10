# Existing-loop enrolled roster

`loop.serial_run` can derive its targets from an existing resolved campaign. It
uses the same serial child owner, finite batches, STOP and original continuation
checks as `--target-args`; that existing CLI remains supported.

```bash
PYTHONPATH=. python3 -m scripts.kernel_rnd.autokernel.loop.serial_run \
  --resolved-campaign /absolute/resolved-campaign.json \
  --owned-targets /absolute/owned-targets.json \
  --state-dir /absolute/serial-state \
  --batch-iterations 5 --rounds 1 --dry-run
```

Remove `--dry-run` to execute at the owning session's authorized boundary. The
dry-run invokes each selected target's existing `run.main --dry-run`: it checks
actual source/build startup and workload metadata, but does not call providers,
build, claim resources or measure. It does not create the state/store directories.
It is not a lightweight assertion that missing original artifacts are available.

The one owner map supplies facts the campaign does **not** establish:

```json
{
  "enrolled-target-alias": {
    "worktree": "/absolute/owned-experimental-source",
    "anchor_build": "/absolute/original-build",
    "branch": "ak/experimental/original-candidate",
    "frozen_prompts": "/absolute/original-prompts.json",
    "calibrate_serving": 5
  }
}
```

`calibrate_serving` is optional: omit it to reopen an existing request-bound floor
or remain explicitly uncalibrated. An optional `launch` path selects the actual
canonical resolved experimental launch; otherwise the exact enrolled recipe file
is used and checked against its original digest. A template-only recipe is not a
resolved launch. An optional `store` preserves an existing target's history.

Model, backend, target identity and planner/critic defaults come from the resolved
campaign. Store/worker/build roots otherwise derive under
`state-dir/targets/<original-target-digest>/`; `--target-root` changes that parent.
The recipe retains its original CPU affinity and request settings. Generated
children use `taskset` within declared host CPUs; this is confinement, not a new
resource grant. Existing child owners still acquire their original claims.

`--common-args` optionally names one bounded JSON argv list for shared actor model/
effort, workers, pair counts, `--belief-root-repo` or explicit recall-ranking options.
It is not required and cannot override target, model, launch or ownership paths.
For this host's belief reader, pass
`--belief-root-repo /mnt/raid0/llm/worktrees/mains/autokernel-unified-20260908` there.

Only ready targets with explicit owned inputs execute. Unowned/non-ready targets
are reported by alias and reason; this is partial coverage, not completion of the
campaign. Byte-identical aliases share one execution entry; conflicting ownership
for aliases refuses. Production and candidate enrollment provenance remains in
the original target and every child continuation. No source tree/branch is created,
no missing model is fetched, and no canonical promotion is inferred.

Scheduling remains finite round-robin, not evidence-ranked or concurrent. Selected
CPU and GPU entries use their exact serving launch/request, not an implicit cheap
GPU screen. GPU serving requires the matching installed direct-serving owner;
legacy `--target-args` screens remain explicitly distinct. This does not implement
mechanism-aware partition search, arbitrary GPU mapping, automatic actor fallback,
or interrupted-active-child recovery. Those semantics are not invented from a
resolved manifest or a successful dry-run.
