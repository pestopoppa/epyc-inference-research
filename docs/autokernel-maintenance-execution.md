# Bounded maintenance execution owner integration

`loop/maintenance_execution.py` and the controller now implement the durable exclusion state machine
around the existing native retention consumer. Live fresh admission remains unavailable by default:
the current release has neither a complete external artifact/dependency catalog nor a concrete
provider that can issue a storage-maintenance hold and settle its accounting.

## Ownership and mutex lifecycle

The controller exclusion outlives every short controller transaction but does not keep the controller
mutex locked. Once the missing native catalog is connected, fresh admission will atomically check
current ownership, reserve one exclusion token, and append `INTENT`. Today it explicitly refuses.
Restart recovery of an already durable intent may append a fresh current token chained to the old
token digest; it never reuses the old provider hold. Provider acquisition,
filesystem measurement/hash/removal, and provider accounting run with no controller mutex held.

Before each tombstone mutation, the adapter refreshes the provider hold outside the controller and
then performs a short controller revalidation followed by a short native-Journal append. After I/O it
performs a short `IO_COMPLETE` transaction, settles exact provider accounting outside the mutex, and
performs a short `COMPLETED` transaction that releases the exclusion. A failure before a tombstone
intent settles the provider hold with an exact zero-cost `aborted` accounting receipt before `ABORTED`
releases the exclusion. A false, missing, or misbound settlement receipt, any acquisition exception or
reply loss, and any ambiguity after validated completion accounting or a tombstone intent cause
`UNRESOLVED` to be recorded and retain the exclusion for replay. Only a typed, request-bound provider `NoHoldReceipt`
proves that acquisition was refused without a hold and permits the controller exclusion to abort.
Thus `snapshot()` and command/drain handling remain responsive during a blocked filesystem call.

There is no nested lock order: controller transactions and provider calls never overlap. The temporal
order is controller admission, provider acquire, controller revalidation, provider refresh/controller
mutation pairs, controller I/O-complete, provider settlement, and controller completion. Candidate
integration, worker launch, new dependency publication, and competing maintenance must all refuse
while the durable exclusion is owned. Shutdown must preserve an unresolved exclusion rather than
convert it to completion.

`close()` refuses while maintenance remains live or unresolved, independent of snapshot schema.
Bounded shutdown drain cannot report quiescence while that exclusion is owned. Exact `COMPLETED` or
receipt-backed `ABORTED` transitions re-run the existing command settlement and notify shutdown
waiters; a deadline never grants force-close or cleanup authority. Replay tests use real subprocess
owner death rather than weakening the production close fence.

## Exact identities and events

The exclusion token binds `token_id`, `campaign_id`, `config_digest`, `config_generation`,
`supervisor_id`, `supervisor_incarnation`, `snapshot_id`, `snapshot_generation`, `snapshot_digest`,
`plan_digest`, `policy_digest`, ordered `selected_artifact_ids`, `admitted_at`, optional
`predecessor_token_digest`, and `token_digest`. The provider
receipt binds `provider_id`, `hold_id`, exact request digest, `provider_generation`,
`accounting_epoch`, `deadline`, `current`, `revoked`, and its content digest. Refresh may not silently
change provider ID, hold ID, generation, or accounting epoch. The controller owner checks `deadline`
against its own clock during each short revalidation; `current=true` alone is only a provider assertion.
Completion accounting binds both receipt digests, disposition, artifact count, reclaimed bytes,
deleted artifact count, and bytes deleted in this attempt.

Primary persistence uses the single Journal kind `MAINTENANCE_EXECUTION` and schema
`epyc.autokernel.maintenance_execution_event.v2`. Every closed event has exactly `schema`, `event`,
`token`, `hold`, `cost`, `accounting_receipt_digest`, `abort_receipt`, `reason`, and `occurred_at`.
Legacy v1 non-abort rows remain replayable, while receiptless v1 `ABORTED` rows fail closed because they
cannot prove provider settlement. Nullability is:

Replay validates every supplied hold against its exact token, including the first provider-held row.
`COMPLETED` must preserve the exact `IO_COMPLETE` cost and carry the digest reconstructable from that
cost, token, and hold. A new job after a terminal event begins with a distinct token and no predecessor;
an owned recovery instead requires a fresh token chained to the exact prior token. Either kind of new
intent clears the prior hold and cost, so settled provider state cannot leak into another job.

| Event | Non-null event fields beyond token/time |
|---|---|
| `INTENT` | none; the token carries the snapshot/root and plan binding |
| `PROVIDER_HELD` | `hold` |
| `MUTATION_REVALIDATED` | `hold` |
| `IO_COMPLETE` | `hold`, `cost` |
| `COMPLETED` | `hold`, `cost`, `accounting_receipt_digest` |
| `ABORTED` | `reason` plus either exact no-hold refusal or exact zero-cost aborted accounting receipt; the latter also requires its current `hold`, `cost`, and matching accounting digest |
| `UNRESOLVED` | `reason`, with optional current `hold`/`cost` |

## Required primary release hooks

The bounded controller backend maps to seven public methods:

- `maintenance_admit(job)` refuses fresh work until native catalog coverage exists; for an exact
  replayed unresolved job it emits a new current token chained to the predecessor without clearing the
  exclusion or restoring stale provider authority.
- `maintenance_revalidate(token, hold)` checks the indexed current token, supervisor/config binding,
  immutable plan/root digest, provider identity/generation/accounting epoch, and deadline against the
  controller clock.
- `maintenance_append_tombstone(token, kind, payload, campaign_id)` appends to the existing Journal.
- `maintenance_io_complete(token, cost)` records measured results without releasing ownership.
- `maintenance_complete(token, accounting)` records settlement and releases the exclusion.
- `maintenance_abort(token, reason, receipt, hold=None)` releases only for an exact request-bound
  no-hold refusal or exact zero-cost aborted accounting bound to the supplied latest valid hold.
- `maintenance_unresolved(token, reason)` preserves visible owned recovery state.

Admission also refuses incomplete/uncertain dependencies, missing native catalog entries, protected
production or rollback targets, active/unresolved worker/acquisition/integration mutations, a competing
maintenance exclusion, unhealthy Journal/runtime identity, invalid policy, a selection outside the
complete plan or outside 1..64 artifacts, and tombstone history above the bounded 192 records.
`CandidateTransactions.retention_view()` replays candidate state and verifies the actual immutable
ArtifactStore manifests, then explicitly refuses because candidate objects alone do not cover worker,
evidence, DSO/RUNPATH, and physical artifact roots. Those catalog records must come from their existing
owner write hooks; callers cannot supply roots or membership.

The narrow race gates are at `run_worker_stage` admission, new candidate `INTENT` append, and new
native-capture publication after the exact-retry return. Candidate inspection and exact native capture
retry remain available while maintenance owns the exclusion. The maintenance Journal append uncertainty
path poisons the controller and requires replay.

## Provider finding

No existing concrete adapter satisfies the provider seam. `TrustedGrantProvider` in
`worker_lifecycle.py` is only a protocol for process/container stages, and its repository uses are test
fakes. Treating `stage="maintenance"` as authority would launch a subprocess and still would not supply
an in-process maintenance hold. `execution/provider.py` owns source-build isolation, not campaign
resource grants or retained-cost accounting. Primary must release a real broker/provider method that
returns the typed maintenance hold and accounting receipt; until then the default executor refuses.
