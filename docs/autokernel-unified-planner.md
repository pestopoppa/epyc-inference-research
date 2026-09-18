# Unified AutoKernel planner boundary

`autokernel.loop.unified_planner` is the offline planning boundary shared by enrolled
CPU and GPU targets. It consumes a validated `ResolvedCampaign`, exact target profiles,
the accepted scoped `EvidenceIndex`, canonical resolved recipes, and the persistent
`SchedulerEngine`. It does not launch a process, build a tree, acquire a grant, or keep a
candidate.

## Typed flow

1. `TargetProfile` binds freshness, quant scope, distinct observation states, hotspots,
   resource cost, and `Opportunity` records to the canonical target-revision digest.
2. `prepare_runtime_anchors()` validates each unique production export and environment
   policy once at campaign/config load, verifies its sealed recipe sidecars, and returns
   an immutable campaign-bound capability. Repeated `plan_iteration()` calls consume the
   prepared recipes without rereading the export or filesystem. A changed campaign,
   export, or policy requires a fresh preparation; preparation does not promise that the
   backing files remain unchanged until execution, so the worker still performs its
   dependency/current-byte checks before measurement.
3. `enumerate_runtime_dimensions()` implements the routine runtime adapter. It supports
   canonical thread/background-thread, taskset CPU-list, canonical numactl policy, batch,
   ubatch, and allowlisted environment set/unset changes. Every enrolled anchor is
   rederived with the accepted production-export validator/resolver from the exact
   exported target and sealed recipe JSON whose byte digest Campaign pins. Real export
   source and full/quarter-mode provenance is retained; caller-supplied hash labels are
   not accepted. Every arm is then rebuilt through
   `resolve_canonical_launch()` and revalidated. Both arms retain the exact model,
   executable, DSO set, build path, backend, and port. Environment dimensions preserve
   unrelated allowlisted state and unknown policy keys are scoped refusals.
4. A runtime opportunity is dispatchable only after a real `ExperimentPlan` is supplied
   for that proposal. Its campaign/target, arm execution digests, metric, direction,
   estimand, unit, changed factor, instrument, and witnesses must match. Source/build
   proposals are emitted only as `pending_plan_preparation` prerequisite work because the
   final candidate recipe identity is not known yet.
5. The persistent scheduler selects a typed `StageProposal`. Production coverage uses the
   canonical grouped target-revision digest as its frontier identity. A selected item may
   be handed to a recorder as `DispatchRequest`; this record always says
   `execution_authorized=false`.
6. The dispatch carries an `experiment_intent` that binds proposal digest, exact
   `ClaimKey`, effect question, and final ExperimentPlan digest before measurement. Arm
   scalar records remain arm-level observations and are explicitly not gain evidence.

Mandatory conflicts are supplied to an actor separately from the bounded JSON prompt, so
prompt truncation cannot hide conflict 41. Cross-epoch values are removed from actor
ranking input while the attempted mechanism and stale status remain. Projection outage
disables evidence reuse but does not prevent an independently specified fresh hypothesis.

## Deliberate limits

- Multi-factor observations may be valid under a separately registered design, but this
  slice only emits single-factor runtime A2 comparisons. It does not globally prohibit
  multi-factor research.
- No transfer is inferred. Q4_K findings do not rank Q8_0 work, and a missing target
  profile is a prerequisite rather than permission to guess a hotspot.
- The lifecycle/worker owner must persist the intent, obtain actual execution authority,
  run the frozen plan, and route native artifacts. Scheduler advice is not that authority.
- Legacy `ResolvedRecipe` records do not embed a reconstructable template and are refused
  by runtime enumeration. Canonical production CPU and GPU records are supported.
- Generic build metadata is not treated as executable-byte identity. Runtime enrollment
  currently requires the production export convention whose build artifact is the exact
  executable digest.

The legacy GPU-only loop remains unchanged. Integration into `actors.py`, `loop.py`, or
`run.py` is intentionally deferred to a separately reviewed driver seam; no default path
changes in this slice.
