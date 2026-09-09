# Native capture controller boundary

`NativeMeasurementSink` publishes prospective per-arm carriers through a callback
issued by the active `CampaignController`. The callback is scoped to its creating
thread and controller lifetime. Under the controller mutex it performs exact-ID
lookup, conflict detection, validation, journal append/fsync, cursor update, and
index update. It does not create a second journal or lock namespace.

New capture is closed by default. The lifecycle owner must install a
`NativeCaptureValidator` whose `NativeCaptureBinding` names the current campaign,
resolved-config digest and generation, supervisor ID, and supervisor incarnation.
Its `TrustedWorkerResultFence` provider must independently confirm the exact worker,
worker incarnation, grant, container, and lineage and explicitly accept the result.
Carrier JSON is provenance input, never grant or worker authority. The controller
does not currently own those supervisor/worker identities, so wiring this provider
is an explicit integration prerequisite rather than inferred eligibility.

Before a new event is accepted, the validator checks the frozen plan, prompt and
admissible-view bindings, both execution/source identities, scalar or diagnostic
shape, interval, carrier digest, and producer schema. It then uses the accepted
`ArtifactStore.verify` API to non-creatively verify the exact carrier and every raw
artifact. A `verified: true` field is only a receipt claim until this byte check
succeeds.

Current scalar use is limited to the producer's explicit planned-serving contract:
`aggregate_tok_s`, higher-is-better, process units, `median.v1`, and the `level`
estimand under `planned-serving/v1`. Each launch value is rederived from the frozen
prompt request hashes and retained native slot rates, through its exact completed
attempt, process, worker/fence, witness, and admissible-row links. Unsupported plan
semantics remain explicit diagnostics when the producer classifies them that way;
an unsupported carrier claiming a measurement is refused.

Startup replays the journal once into a measurement-ID index. An exact retry returns
the original event, including after restart or after later records, without asking a
current worker fence to recreate historical authority. The same ID with different
canonical payload bytes is refused. Malformed, duplicate, foreign-campaign, wrong
generation, or wrong-incarnation history refuses startup. Append or in-memory index
uncertainty poisons the active controller; replay is required to learn whether the
fsynced event landed. Artifact files that precede a failed append are orphans and do
not authorize reconstruction.

Diagnostic carriers are retained by the same path but explicitly contribute zero
scientific progress. This boundary does not ratify protocols, grade measurements,
advance candidates, allocate resources, launch workers, or authorize serving.

The intended scoped use is:

```python
controller.register_native_capture(validator)
with controller.native_capture_callback() as capture_transaction:
    sink = NativeMeasurementSink(
        context=context,
        store=artifact_store,
        capture_transaction=capture_transaction,
    )
    run_planned_comparison(..., artifact_sink=sink)
```

The callback must not be retained beyond the `with` block or moved to another
thread. Close the artifact store separately after the producer is finished.
