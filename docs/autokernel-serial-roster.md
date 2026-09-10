# Existing-loop enrolled roster

`loop.serial_run` can derive its targets from an existing resolved campaign. It
uses the same serial child owner, durable STOP/recovery, held-resource accounting,
and original continuation checks as `--target-args`; that existing CLI remains
supported.

```bash
cd /mnt/raid0/llm/worktrees/mains/autokernel-unified-research-20260908
PYTHONPATH=. python3 -m scripts.kernel_rnd.autokernel.loop.serial_run \
  --resolved-campaign /absolute/resolved-campaign.json \
  --owned-targets /absolute/owned-targets.json \
  --state-dir /absolute/serial-state \
  --batch-iterations 1 --rounds 1 --dry-run
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

Owned-roster scheduling derives proposal templates and resource vectors from the
resolved campaign. It reserves production-frontier coverage, then uses the
existing cost/seed policy; one selected stage runs one research iteration. With
`--rounds 0`, the derived campaign is continuous until STOP, a declared budget is
exhausted, or the recorded default cap of 1000 attempts is reached. CPU stages can
screen an evidence-supported quarter/half-machine scope, but a reduced keep is
provisional and must complete its recorded full-target confirmation before it can
become full-surface evidence. Selected CPU and GPU entries always retain their
exact serving launch/request and original held-resource receipts.

Targets may share one exact owned source worktree and branch: execution stays
serial, each target retains its own store/request/history, and the next child may
consume a digest-bound completed source continuation without treating another
target's measurement as its own. Shared-candidate folding is a separate consumer;
this continuation path does not stage keeps or move the canonical champion.
GPU serving still requires its installed direct-serving owner, and legacy
`--target-args` screens remain explicitly distinct. A dry-run proves this wiring,
not scientific validity or canonical admission.
