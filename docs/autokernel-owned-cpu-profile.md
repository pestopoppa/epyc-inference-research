# Owned CPU profile producer

`autokernel.loop.cpu_profile` implements `owned-cpu-perf-v1` through the existing
installed `ProfileMechanism` and target-profile execution route. It does not
provide grants, alter target-profile admission, or introduce a grading rule.

## Installation and ownership

Author a closed `epyc.autokernel.cpu_profile_config.v1` document and pin its
absolute path and SHA-256. Call
`build_installed_cpu_profile_binding(config_ref, valid_for_seconds=...)`, then
install that concrete mechanism binding in the existing
`InstalledProfilePreparationBinding` / `ProviderRegistry.profile_bindings` map.
The ordinary predeclared `ProfilePreparationRequest` must select its exact
mechanism ID and adapter digest. Validity duration is original parent policy,
not a value the producer invents.

The config contains exactly:

- `schema`, `mode`, and six-field `loaded_identity`;
- original `resolved_recipe`, `prompt_manifest`, and `model_preparation`;
- `profiler` (canonical executable path, SHA-256, exact reported version, and
  optional pinned script interpreter);
- complete prospective `source_closure`, canonical `storage`, and closed `budgets`.

The installed entry point is `scripts/benchmark/run_autokernel_cpu_profile.py`.
The owning profile lifecycle launches it with the selected request as its single
JSON argument. `EPYC_AUTOKERNEL_CPU_PROFILE_CONFIG` carries only the pinned
config reference. It is not a general-purpose PID attachment or shell interface.

The worker verifies the complete original model inventory inside the held stage,
retaining the inventory identity and explicit entry-file digest bridge. Stable
device/inode/size/mtime/ctime observations surround verification and serving;
they are continuity evidence, never a substitute for a model hash. The existing
model validator remains the authority for exact inventory verification. No
full-model verification was executed by the hermetic acceptance tests.

The server must be the producer's actual child, and perf children must share the
producer's actual container. The controller itself may be outside that worker
container. Initial attachment records PID/start/boot/container before launcher
exec completes. After health, and before each enable, readback checks the actual
executable/argv and required executable DSO mappings. Mapping joins include both
device and inode, exact path, and refusal of deleted mappings. Each round retains
before/after loaded-process readbacks. An external PID cannot be supplied.

## Supported observation and output

The sealed mode remains `independent_full_request_v1`: CPU backend, `np=1`, one
original frozen prompt, and separate full warmup and measurement requests. The
frozen prompt parser now supports v1 and explicit v2 token/cache requests; older
documentation claiming an unconditional GLM request refusal was stale. The existing
serving request bytes and throughput arithmetic are unchanged. A successful
natural EOS is supported: original `stop=true` and observed positive completion
length no greater than the requested maximum are retained. No tokens or seed are
manufactured. HTTP/JSON/truncation errors cannot publish a verified profile.

The optional concrete serving hook is absent from ordinary runs. Only an exact
CPU capture can use the no-GPU sampling context; it reports zero samples and
unproven CPU placement/contention. The existing CPU residency result remains
`not_applicable`. GPU and ordinary absent-hook behavior are unchanged.

The raw `epyc.autokernel.cpu_profile_capture.v1` receipt contains:

- original request/target/loaded identities and prospective source closure;
- original model-verification receipt and bounded perf-version artifact;
- producer, ancestor and attached-server process identities;
- ordered warmup/measurement records, original HTTP bytes and request intervals;
- owned perf commands/processes, enable/disable command-and-ACK bounds, terminal
  status, bounded diagnostics, raw-file identities and parser process identity;
- exact per-symbol/per-TID sampled-period sums and separate counter reductions;
- explicit completion and scope limitations.

Samples use monotonic perf timestamps and are filtered to each exact HTTP request
interval. Samples outside it remain counted separately. Lost-record support is
deliberately not invented: malformed/lost-event output refuses, and the receipt
does not assert a zero independent loss count. Counters retain their own
enable-to-disable transition brackets, including ACK and post-request skew;
they are not exact request-window counts. No IPC is derived across those windows.
Unavailable/multiplexed counters remain explicit states.

The output retains the existing `epyc.autokernel.target_profile_output.v1` and
profile measurement carrier grammar. Hotspots are ordered by descending observed
sampled period with deterministic ties. `opportunities` is empty and observation
state is unknown. The numeric sampled-period carrier is not a comparable
performance objective, exact CPU cost, wall-time share, or speedup claim; these
limitations also survive in `TargetProfile.kept_scope` and carrier extras.
The validation proposition is original receipt integrity, not model correctness
or trial validity. Existing source classes and owning graders remain unchanged.

## Bounds, replay and failure

The config fixes stage/teardown/control/parser deadlines, raw/aggregate/parser/
metadata byte limits, and row/symbol limits before launch. Aggregate admission
includes both record files, both parser files, bounded request/response material,
diagnostics and metadata; actual retained artifacts plus canonical receipt bytes
are checked again before success. Storage and model preparation require the
original scheduled resource policy; the helper is not permission to run perf or
hash models in the background. Perf permission denial is failed evidence, never
a request to elevate capabilities or change host policy.

Owned subprocesses are enrolled before fallible identity reads. Both output
pipes and control ACKs are drained with finite deadlines and byte caps; missing
EOF cannot make a read unbounded. Failed disable ACKs still run teardown.
SIGTERM/SIGINT enter the same failure cleanup. The serving owner always handles
its own server teardown; the original parent lifecycle owns final containment,
terminal evidence, incurred cost and settlement. Failure leaves bounded
diagnostics/raw artifacts but no fabricated successful profile. Existing BIND
and held-recovery refusals are not changed.

`reopen_capture(reference, store=..., config=..., request=...)` reopens retained
facts and deterministically recomputes the same sample/counter reductions.
Regular-file reads use bounded no-follow/nonblocking descriptors. Raw reductions
hash and parse the same descriptor and verify its stable identity. Replay checks
original source, request, response, model, process/container, DSO and window joins;
it does not attach to departed PIDs or restore a grant. This is original capture
plus reducer replay, not independently recollected perf data. Planner feedback
still requires the original successful driver settlement joined to the original
`PROFILE_VERIFIED`. Restart rematerializes the original immutable startup
manifest, then replays those durable events without launching a new child.

## Prospective evidence-feed projection

The pinned ROOT feed closure v3 loads `adapters.autokernel_profile`. Its two
registered projectors retain the producer-authored measurement and receipt-integrity
tuples; canonical `claim_tuple.grade()` remains the only grader. This is not
production validation or a comparable speedup observation.

The feed durably retains the original accepted worker terminal before acknowledging
it. An exact `PROFILE_VERIFIED` join supplies its request, generation, plan,
lineage, campaign and controller identities. Only the bounded compact capture is
reopened by the adapter, not raw perf/model payloads. Acknowledgment follows durable
pair projection; restart and retraction retain both measurement associations even
with a one-entry cache. Sampled TIDs may be a nonempty subset of pinned threads;
unsampled threads receive no invented zero-period measurements.

An exact failed driver settlement releases its otherwise unjoinable terminal.
After an exact publication join, a rejected compact capture or conflicting profile
has a durable quarantine disposition and releases only its consumed terminal.
A forged identity join cannot release it. Retryable storage failures leave the
source cursor unchanged. Legacy closures do not accumulate terminals they cannot
consume; historical profiles without original terminals produce diagnostic zero
tuples, never retrospective warrant.

## Direct existing-loop observation

The ordinary CPU serving route in `loop.run` now calls `profile_loop` at startup
and after an actual source keep, while its existing CPU claim is held. Nulls do
not reprofile or recalibrate. This is a separate instrumented server launch:
its throughput is discarded, and acceptance pairs/floor are not instrumented.
`--cpu-profiler /usr/bin/perf` selects the executable (that path is the default).
The bounded stage is at most 1800 seconds, reduced to the selected campaign's
stage timeout when supplied. Perf permission denial/unavailability is visible to
the planner; an owned-child cleanup uncertainty propagates and stops the run.

`CpuProfileCapture.for_loop` reuses the same process checks, perf lifecycle and
sample/counter reducers using the actual current resolved launch and unmodified
`FrozenPromptManifest`. It creates no sealed config, model-verification receipt,
loaded-target issuance, `TargetProfile` or `PROFILE_VERIFIED` event. The original
2029-token GLM request is exercised byte-exactly by a tiny synthetic HTTP fixture;
this is transport/ownership evidence, not a GLM model measurement.

The store's `cpu-profiles/` directory retains a direct capture
(`epyc.autokernel.loop_cpu_profile_capture.v1`) and a small producer-authored
measurement carrier (`epyc.autokernel.loop_cpu_profile.v1`). The latter binds the
capture's exact private-store locator/SHA. `reopen_loop_profile` uses the same raw
replay as the sealed producer. `run.build_context` and `actors.render_context`
carry descending symbol periods, fractions of the observed period total, original
execution/request digests, record pointer and limitations into the actual actor
prompt. They do not populate GPU duration/share fields.

The existing ROOT `autokernel_profile` measurement projector and corpus ingestion
accept this exact direct carrier; no direct integrity/verifier row is emitted.
The prospective current source closure is explicitly pinned, with the historical
sealed source pin preserved. The adapter reopens bounded compact bytes and checks
their original joins; it does not independently reread the raw perf files. The
numeric carrier's lower-better field is not permission to compare totals across
windows: exposure, duration and unknown sample loss remain explicit limitations.

## Acceptance scope

The acceptance chain uses a tiny HTTP child, a harmless loaded fixture DSO, tiny
synthetic model bytes, and a synthetic perf executable. It proves installed
ownership, artifact joins, settlement, feedback, restart and refusal behavior,
plus direct loop observation/actor/corpus wiring, not real hardware profiling,
scientific qualification or GLM performance. No host permissions or production
kernel state are changed by this implementation.
