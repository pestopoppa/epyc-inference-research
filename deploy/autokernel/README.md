# AutoKernel source-only service template

This directory is deliberately not an installable service package. The unit has no
`[Install]` section and its condition names an approval marker that is not shipped.
Do not create that marker until the core service has one reviewed SIGTERM-to-durable-
drain bridge and the contained workload launcher places workers in broker-owned,
separate cgroups.

Substitute every `@...@` field with a literal, reviewed value. `@PINNED_CHECKOUT@`
and `@PYTHON@` must identify immutable deployed source and interpreter locations;
the campaign, store, generation, snapshot version, port, user, group, and secret environment path are
not shell-expanded. Install the environment file mode 0600, owned by the service
account. The token must never be placed in the unit, URL, argv, logs, or status.

The snapshot-version placeholder must name a version for which that exact checkout
ships a closed validator used by both producer and standalone client. This delivery
integration includes the published closed v2 and v3 validators. Its client dispatches
the unified v3 producer to `validate_snapshot_v3`; it never parses v3 through a
permissive v2 field union.

The literal unit still starts the management-only `campaign_service`; a command-line
flag cannot construct runtime authority. The source runner now accepts one typed
`runtime_factory(resolved, args)` hook from an embedding owner. That factory must return
an already entered/replayed snapshot-v3 `CampaignController` and the exact composed
`StandaloneRuntime`; the service validates their campaign, store, generation, scheduler,
and object identity. With no factory, runtime is reported/noted as not connected and a
v3 listening service refuses before creating a store.

The service calls `recover()` before starting HTTP or its sole non-daemon execution
thread. The only successful recovery statuses are `recovered` and `settled`; the latter
means an exact replayed intent was durably completed before admission. Every unknown or
`recovery_required` status refuses. HTTP handlers never tick the runtime. Shutdown durably drains first, requests
runtime stop through the shared event, joins to the one absolute deadline, closes the
runtime executor, then closes HTTP and the controller. A live thread, typed
`recovery_required`, unavailable held receipt, or `shutdown_incomplete` retains the
entered owner and keeps authenticated health/status available; elapsed manager time is
not cleanup evidence.

The factory retains cleanup if it returns an untyped or internally split owner pair.
For a coherent typed pair, including one rejected on service identity, the service closes the runtime before
the controller on pipe, signal-handler, listener, or initial publication failure. It
restores the prior SIGTERM handler before closing the self-pipe. If a runtime/thread or
handler cannot be proved closed, the corresponding owner/descriptors remain retained
and the service reports an unresolved failure rather than guessing cleanup.

The installed `unified_driver` entrypoint now recognizes the closed, self-hashed
`epyc.autokernel.standalone_inputs.v1` startup manifest. It reconstructs the existing
`DriverConfig`, scheduler state, prepared anchors, dimensions, profiles, plans,
profile requests, frozen prompt/execution inputs, and complete `EvidenceIndex`
projection with their recorded identities. `--dry-run` performs that bounded static
validation without creating the store, consulting a registry, calling readiness, or
activating a provider. Its unavailable provider/verifier result is expected.

The manifest hash is file integrity only. It grants no resource, evidence, claim, or
serving authority. Lifecycle/readiness provider IDs and the evidence-verifier ID can
resolve only through an application-injected `ProviderRegistry`; the manifest cannot
name an import, callable, class, receipt, or grant. Listen-time evidence reconstruction
requires the existing scope/use/result verifier callbacks and the exact recorded
support-rule identity. A declared rule ID without those callbacks stays unavailable.
The source entrypoint delegates the resulting typed owner to the existing
`campaign_service.main(..., runtime_factory=...)`; it does not create another daemon,
controller, transport, scheduler, or WAL.

No production provider/verifier registry or closed provider-profile constructor is
installed yet, so the literal service-manager unit remains management-only. Do not
encode runtime objects in environment variables or fabricate a provider. The template
therefore does not establish resource/provider authority,
scientific liveness, deployment readiness, or production acceptance. `KillMode`
does not authorize killing eventual broker-owned workload cgroups, and
`SendSIGKILL=no` prevents a manager timeout from being presented as clean teardown.
Legacy v1 controls have no atomic supervisor-incarnation field. The standalone client
uses the closed v2 command envelope, whose config digest and supervisor incarnation are
validated under the controller mutex; deployment must not attribute that fence to v1.

A ready target without a `TargetProfile` is not automatically a global startup error.
When its exact typed `ProfilePreparationRequest` exists, dry-run lists the target under
`pending_profile_targets` and names the request's registered adapter ID. Execution stays
unavailable until team 2's concrete `TargetProfileExecution` is installed and the
runtime consumes the published `SelectedProfileWork` returned by
`UnifiedCampaignDriver.materialize_profile`. The current team-2 source is not published
or frozen: the required seam is `target_profile_execution.py` (`ProfileMechanism`,
`TargetProfileExecution.prepare/verified_target_profile`) plus the controller's
`register/reserve/cancel/finish/record/current` target-profile methods. The startup
package neither synthesizes a profile nor silently drops the selected prerequisite.
The registry's current profile entry is locator-only and grants nothing: this package
never invokes it, and it can only become usable after validation against the concrete
`TargetProfileExecution` class and connection to the runtime consumer.

Native v2 observation startup is also not connected by this packet. The current
checkout already contains `observation_binding.ParentObservationConfiguration`,
`ParentObservationVerifiers`, additive `UnifiedDriverExecution` constructor and
native-validator wiring, plus lifecycle/provider
`describe_active_observation_claim`. This packet deliberately preserves those
published implementations. Provider preflight already requires the active-claim
method, but the materializer remains unavailable until the main-owned startup
composition supplies the typed observation configuration/verifiers to
`StandaloneRuntime.compose`, and that method passes them to
`UnifiedDriverExecution`. A provider method alone is not native evidence authority.
