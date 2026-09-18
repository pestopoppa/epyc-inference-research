# Prospective unified AutoKernel arm capture

`loop.measurement_capture.NativeMeasurementSink` is the optional evidence sink for
`run_planned_comparison`. It seals every native observation and completed attempt before
publishing a per-arm `epyc.autokernel.unified_arm_capture.v1` carrier. The carrier records the
frozen plan and prompt manifest, exact candidate/control execution and source identities,
controller/worker/grant/container incarnation, producer-observed interval, raw slot timings,
typed contention/placement/residency witnesses, and the canonical admissible-unit view.

The metric remains **the sum of per-request `predicted_per_second` values for one independent
process launch**. It is not common-window or wall-clock throughput. This slice supports exactly a
serving instrument with `unit=process`, `estimand=level`, and `estimator_id=median.v1`;
other combinations are retained diagnostically rather than assigned invented semantics.
The scalar's unit is `t/s`; `independent_unit=process` records the distinct sampling unit.
`independent_n` counts scored process launches, never slots or tokens. A failed launch, incomplete
slots, zero scored launches, unknown placement
or contention, or missing GPU in-window residency is retained as a diagnostic carrier with
`measurement: null`.

The artifact store is an owned mode-0700 `RuntimeRoot`. It stages and fsyncs bytes, publishes with
no-overwrite link semantics, verifies the pinned parent plus single-link mode-0600 target, and
returns the exact relative locator and file SHA-256. Artifact existence is not replay authority.
`ArtifactStore.verify(namespace, body)` is the non-creating exact-byte check for a known object; it
refuses missing objects and unresolved staged publications. Creating a new store also requires its
immediate parent to exist so that creation can be fsynced explicitly.
The injected `capture_transaction(measurement_id, payload)` must perform exact-payload lookup,
idempotent append, cursor update, uncertainty poisoning and closed-capability fencing under the
controller's one serialization boundary. Until that controller extension and the closed journal
kind are reviewed, constructing a sink does not connect live capture.

Example construction (inside an already admitted and contained worker):

```python
sink = NativeMeasurementSink(
    context=CaptureContext.from_dict(frozen_context),
    store=ArtifactStore(capture_root),
    capture_transaction=controller_capture_transaction,
)
result = run_planned_comparison(..., artifact_sink=sink, wall_clock=trusted_utc_clock)
```

The schema is a native producer record, not a protocol ratification, eligibility rule, resource
grant, or promotion authority. Discovery and observation records retain their intended-use limits;
the Vidya adapter projects the carrier and the existing shared `ClaimTuple` ladder alone grades it.
An `unknown` protocol status is preserved as such and supplies no protocol citation.
Historical pre-hook runs cannot be backfilled because they lack in-window contention, placement and
residency facts.

An accepted continuation writes the `continued_unit` artifact, preserving the prior lineage and
artifact digest, but it does not pretend the new lineage performed another launch. Until the caller
also supplies and verifies the prior native observation bytes, a reuse-only arm is journaled as a
diagnostic carrier with no measurement interval or scalar and therefore projects no `ClaimTuple`.
