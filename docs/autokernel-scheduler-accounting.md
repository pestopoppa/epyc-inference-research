# AutoKernel bounded scheduler/accounting v1

This slice is a deterministic offline projection. It selects a service opportunity and accounts
supplied held-claim receipts; it does not acquire a lock, contact a provider, admit execution, certify
overlap, or create measurement/claim authority. Every selection serializes
`execution_authorized: false`.

`SchedulerConfig` freezes the provisional numerical configuration: maximum stage-plus-teardown `D`,
noncoverage `K`, reservation slots, campaign and seed attempt/charged-time caps, resource capacity,
fixed weight source, and shared-work apportionment rule. V1 fixes normal weight 1, seed weight 2, and
ends boost at three valid comparisons or the finite attempt/charged cap, whichever occurs first.
These are scheduling parameters, not statistical policy. V1 has no adaptive rule: configurations with
one are refused and `adaptive_weights` reports `adaptation_not_configured`. Adaptation belongs at a
future verified batch boundary, not in this offline selector.

At a new round, `select_stage` freezes the currently eligible production-frontier identities. Each
gets one coverage opportunity before at most `K` noncoverage opportunities. An unused optional slot
does not stall the next production round, and an unavailable or infeasible optional proposal does not
block an otherwise ready production frontier. Arrivals do not mutate the active round. Every expensive
class—search, prerequisite, calibration, validation, reject audit,
maintenance, build, and teardown—uses one slot when its native receipt is accounted. An eligible seed
requires `K >= 1` and gets a reserved FIFO opportunity within finite caps. Alias-equivalent revisions
of the same backend and target revision share arrival, boost, attempt, and charge history. Completing a
boost leaves the ordinary noncoverage budget available. The displayed conservative serial opportunity
bound is `(N + K) * D`, plus separately displayed union wall time for supplied outages and an
already-running bounded stage; it is not a promise of valid research under noise or unavailable
authority.

If a frozen frontier remains scoped-unavailable after the round's `K` opportunities are consumed—or
unused `K` cannot be consumed because no required runnable noncoverage work exists—the engine records
that exact unserved obligation as coverage debt and starts the next round from the continuously ready
production set. The debt is not a served stage or measurement and is cleared only when that frontier is
actually accounted. It remains visible across export/replay; no wall-time bound is claimed through an
authority or resource outage. Frozen seed and other reservations are not erased to create this progress.

`HeldClaimReceipt` accounts the time integral of claims. Physical-region cost is declared fraction
times held seconds; affinity does not discount it. Each GPU is charged device-seconds and must include
host CPU fraction. Memory reservations retain byte-seconds. Setup/load/warmup/idle/failure are charged
because accounting uses the complete held interval. Shared work is charged once and attributed using
explicit normalized beneficiary shares. Exact duplicate receipts are idempotent; conflicting receipt
IDs, outcomes, selection bindings, or overlapping reuse of a physical claim are refused. An actual
duration or capacity overrun remains charged and creates a successor fence; spent work is never erased.
Capacity changes with the same scheduling policy begin a usable new accounting epoch without clearing
prior deficits, attempts, receipts, or seed history.

`SchedulerEngine` is the operational owner. It validates and indexes complete history once at startup,
then selection reads compact totals and accounting touches a receipt-ID lookup plus only the affected
physical-claim interval lists. Replay and `export_state()` may scan or serialize full history. The pure
`select_stage` and `account_stage` functions remain bounded reference transitions that construct an
engine from a complete immutable state. Seed identity/ID lookups, updates, and FIFO promotion also use
startup-built indices and per-backend heaps; operational transitions do not scan historical seed
accounts.

The real consumer must supply trusted eligibility, actual ownership/allocation generations, physical
claim IDs, complete intervals, and authority/resource outage events. It must durably journal round and
accounting transitions before treating them as current.

The current campaign service, controller Journal producer/replayer, provider, region locks, broker,
overlap/coexistence verifier, evidence grader, and measurement pipeline are not wired to this module.
A future controller seam must journal the exported state, issued selection identity, and corresponding
account transition, reconcile outstanding selections with native held-claim receipts after restart,
and inject trusted eligibility. Selection remains advice with `execution_authorized: false`; it is not
an allocation, grant, admission, compatibility decision, or claim grade. In particular, a compatibility
reference is retained as provenance but never interpreted here as permission for concurrent backfill.
V1 has no trusted compatibility consumer, so a future due `full_region` reservation causes only earlier
unreserved optional backfill slots to be explicitly skipped and persisted; earlier frozen seed or other
reserved obligations retain their slots, followed by the exclusive proposal in its reserved slot.
Caller labels and `compatibility_authority_refs` cannot relax that conservative rule.

## AKU-07k / AKU-06l additive slice (2026-11-18)

Operator approval: OP-AKU-BIND, OP-AKU-A2 (2026-11-18). Hermetic code only; no hardware claim.

- **Native accounting binding (AKU-07k).** `worker_lifecycle.AccountingBinding`
  (`epyc.autokernel.stage_accounting_binding.v1`: catalog/transition/selection/proposal digests,
  proposal ID, backend, scheduler stage class) is an optional `StageRequest.accounting_binding`.
  When present it joins the prospective request identity, the `OWNED_LAUNCH_INTENT` contract and the
  held-receipt event (`accounting_binding_digest`). The provider's typed receipt is verified against it
  before it is retained or indexed, and recovery re-verifies the durable held row before appending any
  event. A request without a binding is byte-identical to v1 (digests, contract, events); an old reader's
  closed contract/event validators refuse a bound record. Identity is never inferred from argv, the
  `setup` stage name or a digest. The profile producer derives the binding from the exact
  `SelectedProfileWork`; the former strict xfail on the actual-driver profile settlement now passes.
- **Typed admission denial (AKU-06l).** `StageAdmission.denial_class` (`retryable` | `terminal` |
  `None`=legacy untyped) surfaces as `AdmissionDenied` (a `WaitingAuthority`); a denial has no held
  interval and no receipt.
- **Original multi-child settlement (AKU-06l).** `SchedulerEngine.account_operation` accounts one selected
  operation from separate original `OperationChild` receipts (ordered planner/critic/build prefix, unique
  receipt IDs, no cross-role reuse): one attempt, one slot, every receipt recorded, charged seconds are
  occupancy (sum of held intervals). Overlap is admitted only as provably disjoint CPU affinity partitions
  within aggregate capacity; GPU/shared-claim overlap refuses. Validation precedes all mutation.
- **Explicit no-launch cancellation.** `cancel_operation_before_launch` (terminal denial only): one failed
  attempt, zero seconds, no receipt, recorded as `cancelled_operations` (additive operational-projection
  key, absent for legacy state); idempotent per denial reference.
- **Not done:** the Journal child-record kind, `UNIFIED_DRIVER_SETTLED.v2`, durable persistence of
  `cancelled_operations` in `SchedulerState`, the actor/source-build/standalone wiring, and the live
  startup-to-settlement proof through the installed entrypoint.
