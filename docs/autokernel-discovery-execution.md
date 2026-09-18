# Controller-owned A2 discovery execution

`autokernel.loop.discovery_execution` connects the published `A2RuntimeScreen` phase
consumer to the actual `CampaignController` and primary `Journal`. It introduces no
second log, grader, threshold, grant authority, artifact store, or promotion path.

Each logical execution derives a stable SHA-256 identity from campaign/config, caller
logical ID, plan digest, and complete runtime-frame digest. Supervisor incarnation is
deliberately excluded from that logical identity. Every physical write still records
the supervisor incarnation that owned it. A clean same-config restart therefore
re-attests original events without relabeling their worker/supervisor identities, while
new events bind the new current owner. Reusing a logical ID with changed plan/frame
refuses.

The one new native Journal kind is `A2_RUNTIME_EXECUTION`. Its closed v1 payload wraps
either one existing `a2_runtime_phase_event.v1` or one separately closed
`a2_bank_reference.v1`, plus campaign, config, logical-execution, and write-owner
identity. An execution still contains at most fourteen actual phase events and may
contain at most one bank reference. `CampaignController` builds indexes once during
startup. Append, exact retry, and replay use only selected bounded executions; they do
not rescan the Journal. A returned append is fsynced. An uncertain
append poisons that controller lifetime; restart replay determines whether the exact
event exists. The append call also receives the concrete frozen `ExperimentPlan` and
rechecks every INTENT/TERMINAL against its exact unit, arm, process, order, prompts and
plan digest before writing.

Serialized rows alone do not mint reuse authority. The narrow
`attest_controller_phase_history()` function accepts only the concrete active
`CampaignController`, compares the requested events against its indexed Journal state,
and then mints `RegisteredPhaseVerifier` internally. No verifier token or generic JSON
constructor is exposed. A stale/closed controller callback refuses.

## Durable cached-bank reuse

A second logical screen does not synthesize local anchor events. Before its first
candidate INTENT, it appends one bank-reference transition. The controller resolves the
bank through its bank-digest index, requires exactly one byte-for-byte matching original
sealed source, and records the source execution/logical/plan/frame identities, all seven
original anchor event digests and Journal entry IDs, the exact anchor-history digest,
seal digest, and bank digest. The source and target frames must be identical. The
reference points back to the original INTENT/TERMINAL records, whose units, producers,
native invocation proofs, and write supervisors remain unchanged.

Replay reopens that original source from the controller's execution index and recomputes
the complete reference. A missing source, changed source history, stale frame, forged
bank, or reference ordered after a local phase event refuses. Only then does
`attest_controller_bank_reference()` mint a `RegisteredBankVerifier`. Caller JSON, a
matching digest, or even a verifier minted by a different controller store cannot create
durable reuse authority. Generic `A2RuntimeScreen` bank-verifier behavior is unchanged;
the controller-backed wrapper supplies the newly re-attested verifier after restart.
Candidate screening still performs exactly three candidate launches and zero anchors.

Pending INTENT rows remain explicit in replay and `A2RuntimeScreen` refuses to rerun
them. The native bridge must reconcile the exact worker request through controller
ownership APIs and either recover its durable terminal result or prove that acquisition
never occurred before continuing that already-declared unit. Caller-authored statuses
and an empty current-worker projection are not recovery proof.

## Native invocation seam still required

The next bridge revision should stream one contained planned-serving invocation for
each unit only after that unit's phase INTENT has been fsynced. It returns one complete
`InvocationResult` derived from the parent-validated native result, capture, held-claim
receipt, and registered witness references. This is smaller and safer than lazily
executing a three-unit batch behind unit 0's intent: units 1 and 2 never run before
their own durable scientific membership declarations.

The adapter must call `CampaignController.run_worker_stage`, retain exact native IDs on
retry, and refuse while acquisition, terminal, observer, or append ownership is
unresolved. An unseen request's `worker_attempt_status == "unknown"` is not permission to
launch. The next bridge version therefore also needs a current-owner, single-use
first-issue reservation tied to the freshly persisted INTENT; replayed unknown intents
remain fenced. Ordinary A2 foreign load remains diagnostic noise; witnessed competing
inference remains non-admissible. This persistence release performs no native execution.
