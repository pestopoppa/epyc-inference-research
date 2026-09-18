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

Example offline inspection:

```bash
PYTHONPATH=scripts/kernel_rnd python3 -m autokernel.loop.scheduling_cli \
  --resolved-campaign resolved-campaign.json \
  --config scheduler-config.json \
  --proposals eligible-proposals.json \
  --state scheduler-state.json \
  --receipts held-claim-receipts.json \
  --outages scheduler-outages.json \
  --now 1788926400 --out scheduler-inspection.json
```

The real consumer must supply trusted eligibility, actual ownership/allocation generations, physical
claim IDs, complete intervals, and authority/resource outage events. It must durably journal round and
accounting transitions before treating them as current. CLI scheduler identity binds the campaign ID to
the digest of the complete normalized `ResolvedCampaign`, so a changed resolved snapshot cannot reuse
old state under the same label. Input target revisions use `target_id@revision`; the CLI resolves every
alias through the grouped `TargetRevision`, requires `alias_identity` to equal its
`workload_signature`, and replaces both target and production-frontier identity with one canonical
group digest. Backend and production/seed enrollment must match the resolved group. A non-ready target
cannot become serving work through `eligible:true`; only separately typed non-frontier `prerequisite`
or `build` work is inspectable until it is ready. Optional receipts use the closed
`epyc.autokernel.scheduler_receipt_input.v1` wrapper containing exactly `schema`, `receipt`, and
`outcome`, and may account only the transition's selected declared proposal.

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
