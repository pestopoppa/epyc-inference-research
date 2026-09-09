# Native parent evidence for unified AutoKernel

The native v2 path derives factual unit witnesses from original parent-owned lifecycle
evidence and reopened artifacts. Child observations are inputs, not witness authority.
This path does not introduce scientific acceptance policy or a ClaimTuple grading ladder.

## Integration

`UnifiedDriverExecution` accepts optional concrete `NativeFactualEvidenceConfiguration`.
It requires an already-issued v2 plan, the existing parent observation configuration,
and an owned lifecycle provider capable of describing its active claim. Without this
configuration the existing parent producer retains unknown evidence semantics.

The implementation lives under `scripts/kernel_rnd/autokernel/loop/`:

- `native_parent_service.py`: bounded parent evidence thread and explicit configuration.
- `native_parent_evidence.py`: exact original context joins and factual unit receipts.
- `native_parent_receipt_replay.py`: parent issuance registry and deterministic replay.
- `native_producer_source.py`: prospective loaded producer code/configuration identities.

The child seals its native observation and sends an artifact-only v2 completion request:
`schema`, `nonce`, `sequence`, `fence_id`, `native_observation`, `request_digest`.
The artifact reference contains exactly `locator`, `sha256`, `verified`; the parent
reopens bytes and verifies the namespace, digest and identity joins regardless of that bit.

A health phase notice binds the same active unit, process generation, fence, observation
binding, descendant receipt and phase boundary. It requests a parent-side readback while
the target is alive. Its acknowledgement reports only `captured` or `unavailable`, never
a witness verdict. Exact duplicates reuse one result; conflicting retries are refused.
Completion cannot overtake a pending health notice.

All inherited socket exchanges are serialized. Waiting is bounded by the original
provider deadline, the unit fence and the existing phase acknowledgement budget; it
never extends ownership. The lifecycle watchdog only queues/polls bounded notices:
artifact IO and readback evaluation run in the separate parent service.

## Original issuance and replay

The service resolves the full original `owned_descendant_bound` event through the
parent-only `WorkerLifecycle.owned_descendant_event` lookup. It joins that event to the
actual held claim, worker incarnation, unit/process generation, binding and frozen plan.

`IssuedNativeEvidenceRegistry` records the actual evaluated result before completion
publication, retaining original request/context/receipt bytes and loaded validation source
pins. It is bounded to the prepared attempt. A child cannot create registry authority by
writing a plausible receipt or setting a verified flag.

Before native capture, `NativeParentReceiptReplayer` reopens the original native,
lifecycle, receipt and readback artifacts and rederives their factual findings. It compares
the entire witness map and provider screen to original issuance. Replay does not resample,
reissue a result, or accept a newly supplied witness reference.

Replay runs during outside-lock prevalidation; the existing serialized current-owner check
still controls Journal capture. The driver retains the original registry through an exact
attempt retry and removes it after durable settlement. Restart does not reconstruct a lost
uncommitted issuance from child files. An exact already-committed Journal duplicate retains
the existing early idempotent path and does not require a newly manufactured receipt.

V1 completion request/response/chain behavior is unchanged. V2 retains the existing completion
response and native carrier grammar; its completion chain binds the artifact request digest.

## Prospective producer identity

The original loaded-instrument artifact contains
`used_constants.producer_source_closure`, schema
`epyc.autokernel.native_capture_producer_source.v1`, before plan issue. The carrier contains
an instrument reference, so consumers must reopen and verify the referenced identity bytes.

The closed record names actual inherited deferred capture methods, observation seal/reopen
helpers, v2 native validator methods, and the four selected parent verifier slots:
`observation`, `purpose`, `runtime`, `gpu`. A slot is null or a loaded callable identity.
Capture checks the selected validator instance's configuration as well as its class methods.
This is explicit named-method scope, not an inferred transitive call graph.

`validate_producer_source_closure` validates the closed record;
`producer_source_closure_complete` reports loaded-code/configuration pin completeness;
`verify_capture_producer_source` reopens original bytes and compares the selected implementation.
None of these functions grants scientific validity or eligibility. Missing historical closure
remains missing: today's code cannot supply original-run provenance for an older v2 record.

Loaded-code identity supports exact immutable string frozenset constants up to 4096 members,
sorted deterministically with their type retained. Mutable, subclassed, oversized or non-string
sets remain unproven; generic object/repr serialization is not accepted.

## Factual scope and limitations

- Request completeness requires exact prompt/request joins and each predicted count equal
  to its corresponding frozen `n_predict`.
- Placement requires an observed owned target during load, placement, health, warmup and
  measurement. Factual mismatch fails; missing or cross-boundary evidence remains unknown.
- Runtime readback supports only explicit existing recipe expectations and original
  parent readback facts. A phase acknowledgement alone proves no runtime setting.
- Correctness, purpose, contention and GPU validity are not inferred from process names,
  generic success, absence of a late sample, or a fully pinned implementation. Unsupported
  witnesses remain unknown and records remain diagnostic where existing policy requires it.

The focused tests include real tiny-child/socket execution, original-receipt mutations,
exact restart duplicates, bounded concurrent exchanges, frozen count/placement changes,
configured-verifier changes and cross-hash-seed identity reproducibility. Their hermetic
resource/proc fixtures are not evidence of live production CPU/GPU measurement validity.
