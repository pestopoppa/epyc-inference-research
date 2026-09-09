# AutoKernel campaign control service

This v1 service is an offline control and snapshot producer. It does not launch workers,
allocate resources, grant execution authority, run inference, or proxy through the dashboard.
OP-41 remains the authority boundary.

The service consumes an immutable resolved campaign and an explicit positive configuration
generation. Its configuration digest hashes the complete normalized `ResolvedCampaign`, including
resolved source/artifact/target identities; the requested manifest digest remains a separate field.
A nonblocking `.supervisor.lock` permits one controller for the store. Every restart
appends a fsynced `CAMPAIGN_SUPERVISOR_EVENT/START`, advances numeric supervisor incarnation and
stream epoch, and preserves the prior desired state, control revision, and command results.
Corrupt or unsupported history refuses startup. Changing campaign identity or configuration
requires a new explicit generation/store operation; the constructor never migrates it implicitly.
The controller pins the private store, named lock object, and live journal directory and refuses
replacement or symlinked journal-critical objects at its public boundaries. Its `_journal` member is
test/internal-only; direct external appends would not be serialized by the controller lock.

Commands use `epyc.autokernel.campaign_command.v1` with exact fields, an empty v1 payload, a
semantic digest independent of expected revision, and optimistic `expected_control_revision`.
Duplicate request IDs with identical semantics return their durable prior result; conflicting reuse
and new stale revisions refuse. Acceptance is fsynced before acknowledgment. Acceptance and
completion are distinct. Initial pause is quiescent, resume waits for an injected prerequisite
checker, and drain is terminal for this v1 generation—further work needs a new campaign/generation.

Snapshots are full ordered projections. `(stream_epoch, sequence)` is the ordering key; sequence
also advances for health-only publication while `journal_cursor` and
`last_scientific_result_at: null` remain honest. `execution_authorized` is always false and
`active_worker` is null in v1. Once the producer exits, its last snapshot is historical, not proof
of a live service.

`producer_build` identifies an exact subset of the loaded `autokernel.loop.campaign_control` module:
module-function and class-method bytecode, property accessors, static/classmethod wrappers, and the
listed scalar/frozen-set constants. Generated method bytecode is labelled separately in the included
symbol manifest. It deliberately does not hash source files or mtimes at heartbeat time: changing
bytes on disk does not imply that the running interpreter reloaded them. Other descriptors, service
transport, journal validation, and whole-package launch attestation are explicitly outside this
scoped identifier; supervisor launch integration must provide the latter before describing it as a
complete package ID.

The pure `may_start_stage` seam only consumes a typed trusted grant. It refuses closed controls,
stale control/supervisor tokens, revoked grants, failed renewal for future admission, unknown or
failed dependencies, and deadlines that cannot contain both the bounded stage and teardown. It
does not create or mutate grants.

CLI:

```text
PYTHONPATH=scripts/kernel_rnd python3 -m autokernel.loop.campaign_service \
  --resolved-campaign resolved.json --store /private/store
```

With neither `--once` nor `--listen`, this is non-mutating inspection. `--once` opens the controller,
publishes one durable snapshot, and closes it. `--listen 127.0.0.1:PORT` requires a bearer secret in
`AUTOKERNEL_CONTROL_TOKEN`; `/health` is transport-only, while `/snapshot` and `/commands` require
authentication. `/health` is transport-only and performs no journal read, fsync, or snapshot
publication. The service-owned publisher writes an initial snapshot and refreshes it at the finite
configured interval without observer traffic; it stops before the controller lease is released, and
a publication failure is retained and makes transport health unhealthy. These health-only snapshots
advance publication sequence and heartbeat time, but never journal cursor or scientific-result time.
V1 accepts numeric
loopback binds only, bounded JSON headers/bodies and read time, and no filesystem or shell fields.

Remaining AKU-07/09 work includes durable launch intent, owned-child/container reconciliation,
the real trusted grant/provider adapter, bounded worker execution, service-manager deployment, and
the existing hub’s direct producer consumer. These tests and seams do not complete AKU-07.
