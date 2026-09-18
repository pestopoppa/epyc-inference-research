# Standalone serving calibration preparation

Standalone input v4 adds prospective raw calibration preparation to the existing
driver, scheduler and owned worker. Prepared v3 is a preparation dispatch with the
existing native observation/capture v2 capability. Input v1–v3 and Prepared v1–v2
retain their closed schemas; adding preparation fields to an older schema is
refused. There is no additional executor, provider grant or scoring ladder.

## Original configuration and runnable entry point

The startup factory request `epyc.autokernel.startup_factory_request.v4` extends
the native v3 request with `serving_preparation: {path, sha256}`, pointing to an
original compact `epyc.autokernel.serving_preparation_startup.v1` artifact.
Each `entries` element contains:

- `declaration`: the closed `ServingPreparationDeclaration.to_dict()` document;
- `protocol`: original `protocol_ref`, `protocol_status`, `policy_snapshot`,
  `required_witnesses`, `comparison_kind` and `estimand` fields;
- `submitted_at` and `estimated_duration_seconds`: explicit scheduler inputs.

The declaration retains the actual campaign seed, controls, committed stopping
rule, split/rotation rule, statistical construction, effect scale, hypothesis,
margin and owning repetition rule. It includes exact canonical A/A recipes,
optional genuinely authored byte-identical neutral-copy material, original
prompts/frame/source identities, resource limits and a finite retry policy.
Neither a floor-only calibration receipt nor an ordinary execution success can
substitute for that original statistical material. Supplied protocol labels do
not create ratification or control-bootstrap permission.

The factory still requires matching sealed campaign CLI v2 and production export
pins, explicit installed providers, native model preparation, observation/feed
configuration, and ordinary target execution inputs. It computes independent
pair/material/process identities, all original retry attempts and scheduler
enrollment from these inputs before selection. The total expansion is checked
against the campaign attempt cap before pair allocation; compact configuration
and expanded request bytes have explicit bounds. Prompt count, target, campaign,
epoch, metric/direction, canonical anchor, available profile quant, resources and
execution budgets must agree. Missing profiles stay unavailable.

The public driver constructor repeats the enrolled metric/direction and exact
installed execution-input joins before creating an owner. Per-unit prompt
selection uses the existing serving contract: each unit selects exactly its
arm's `np` requests; additional unselected manifest prompts are not rejected.
The compact startup expansion explicitly selects its full manifest for each unit.

Run against the intended installed research source:

```sh
PYTHONPATH=/absolute/research/scripts/kernel_rnd python3 -B -m autokernel.loop.startup_factory \
  --request /absolute/original-factory-request.json --out-dir /absolute/new-startup-bundle
PYTHONPATH=/absolute/research/scripts/kernel_rnd python3 -B -m autokernel.loop.unified_driver \
  --config /absolute/new-startup-bundle/startup.json --dry-run
```

`factory-receipt.json` gives the exact configured executable, Python path and
output command. Dry-run performs no ArtifactStore, feed/SQLite, model-byte,
provider acquisition or worker I/O. It labels the instrument/catalog unpublished,
preparation uncollected and scientific qualification unavailable.

## Execution, retention and retries

The installed runtime passes concrete preparation requests into the public
driver constructor; compose and tick enable calibration only when these requests
are installed. The ordinary scheduler selects each fixed independent pair.
Allocation, owned child launch, model hashing, native observation, teardown and
provider-authored held accounting use the existing lifecycle. Hashing remains
before the measurement window. Sampling prompts are not independent process
repetitions.

Neutral copies can share an execution digest with their anchor while having
different snapshot digests and physical paths. V4 retains every exact snapshot in
the native artifact catalog. Execution-keyed model/observation settings are shared
only after exact relevant model, workload, placement, environment and DSO checks;
the alternate executable is not invented as a runtime dimension.

Each collected chunk binds the original plan, request, process/block membership,
result artifact and actually committed native captures. Original witness failures
mark contamination. Unknown witnesses stay unknown, not fabricated passes. Only
an exact durable successful settlement makes a chunk available to the numerical
pool. A CAS chunk without settlement cannot enter the solve. Contamination or
failure enables only the next predeclared reversed-order attempt; successful
blocks and their original process identities are never replaced. Exhausted retry
budgets stay explicit.

After both original A/A and neutral pools settle, `solve_collected` constructs the
owning `CalibrationInputs` and calls `controls.run_calibration_block`, which
delegates to the existing statistical solver. `reopen_solve` reopens the original
native material and rederives the entire numeric result and qualification debt;
self-consistent edited CAS output is insufficient. A fresh manifest/runtime
restart recovers the same settled chunks and solve reference without recollection
or repeated accounting.

## Deliberate limits

All collected chunks and numerical solves remain diagnostic-only:
`qualification=unavailable`, `ranking_authorized=false`. Original window
admissibility and a genuinely qualified same-frame control panel are still
required. Canonical serving recipes do not supply scientific phase/cell-class
scope, so those declared labels carry explicit unverified-scope debt. No
provisional control PASS, bootstrap policy exception, production validation or
search-ranking authority is introduced. The native scientific final-trial owner
remains v2-only; raw preparation is not forced through it.

`OP-AKU-HELD` remains a distinct recovery dependency. If the process exits after
raw completion but before durable settlement, the current lifecycle does not
persist the original provider-held accounting receipt. The restart therefore
returns `recovery_required` with unknown ownership; it must not silently fail,
rerun, or promote the completed block. The strict expected-failure regression
records this gap. Ordinary settled restart and same-owner exact settlement retry
are covered, but unfinished-settlement restart is not claimed as complete.

The end-to-end tests run the real exporter/CLI/factory, scheduler, controller,
native child transport, capture and numerical reducers using explicitly synthetic
production bytes, provider and observations. They do not constitute a real-host
campaign deployment or AKU-12a live-input acceptance.
