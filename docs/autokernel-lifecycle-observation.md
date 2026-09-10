# AutoKernel lifecycle observation API

`loop/lifecycle_observation.py` records one serving unit from setup through owned
teardown. It is an observation producer, not an admission provider, inference detector,
protocol evaluator, or grader. Its output always carries `verdict.status=not_evaluated`.
Annex K A2 remains upstream: ordinary service/build/agent/filesystem load is recorded
noise, while only an owning consumer with a verified competing-inference witness may
apply the Annex rule. The sampler emits no `policy_effect`.

## Stable bridge API

Construct exactly one `ObservationSession` per fixed measurement unit:

```python
ObservationSession(
    context,
    probe=bounded_probe,
    owned_identity_resolver=trusted_provider_resolver,
    foreign_verifier=trusted_foreign_verifier_or_none,
    runtime_verifier=trusted_runtime_verifier_or_none,
    monotonic=clock,
    wall_clock=utc_clock,
    record_callback=artifact_store_callback,
)
```

The required resolver is a trusted provider capability, not JSON supplied by the
candidate. For the PID captured after `Popen`, it returns exactly `pid`, `start_ticks`,
`boot_id`, `binding_ref`, and the full worker binding. That worker binding contains
integer worker incarnation, grant identity/generation, and the provider-captured
container identity (`path`, `dev`, `ino`, `uid`, `nlink`, `mode`). The collector compares
the complete binding, current PID-start/boot, exact unified-cgroup path, and current
container inode identity before attributing target data. A future PID is never declared
in the pre-launch context.

Call order is:

1. `start("setup")` before `Popen`;
2. `phase("load")` immediately before `Popen`;
3. `attach_target(popen.pid)` and `phase("placement")` immediately after launch;
4. `phase("health")` after readiness, before effective readback;
5. `phase("warmup")`, then `phase("measurement")` before measured requests and
   `checkpoint("measurement_end")` after they complete;
6. `phase("teardown")` before the first owned terminate action; and
7. `finish()` only after owned `wait()` completes, including exception paths.

`load_window` spans the load marker through the health marker; placement is explicitly
an overlapping launcher marker, not a claim that load stopped. Skipped exception-path
markers are retained as gaps. The two measurement markers enclose active generation and
supply persistence for an absence observation. Each marker gets a bounded acknowledgement
wait; it never starts a second reader.

`finish()` returns a detached, content-sealed record and invokes the optional artifact
callback with another detached copy. `shutdown_resolved` and `successor_permitted` expose
reader ownership. If a syscall remains blocked past the bounded join, the result is
frozen once with `shutdown.status=unresolved`; late returns cannot append or invoke the
callback, and the containing owner must refuse a successor unit. The existing serving
hook raises `ObserverShutdownUnresolved` as a local signal on an otherwise successful
unit. Because higher-level fixed-unit runners may catch measurement exceptions, the
bridge admission/guard path must also retain this session and refuse its next unit while
`successor_permitted` is false. An original launch/readback/request/residency exception
remains the primary exception.
After the reader actually returns, the owner may call `reconcile_shutdown()`. This can
lift the in-memory successor fence but never rewrites the frozen historical record.

## Bounded evidence

The closed context schema is
`epyc.autokernel.lifecycle_observation_context.v1`; the sample and final schemas are
`epyc.autokernel.lifecycle_observation_sample.v1` and
`epyc.autokernel.lifecycle_observation.v1`. Unknown fields and versions are refused.
The context binds boot/clock, loaded-instrument and recipe digests, worker/grant/container,
held CPU/GPU claims, requested CPU/NUMA/THP state, required prepared DSO identities, and
all budgets.

Budgets cover samples, pending markers, all proc-root entries, numeric processes, bytes
per file, total retained sample bytes, maps, FDs, CPU-range expansion, NUMA rows,
DSO/device rows, phase acknowledgement, and shutdown join. Exhaustion is explicit
unknown evidence with counts. The sole reader records the
actual read begin and end in monotonic and UTC clocks, queue delay, reader duration,
sample count/budget, and reader-seconds divided by the whole observation duration.
Reader seconds and completed/failed/oversized counts are accumulated before retention,
so dropping an oversized sample cannot erase incurred observer cost. Any unjoined
reader at the frozen shutdown boundary makes reader-cost status unknown. Its exact
unresolved-read count may be zero when the live reader is between probes; a late exit
or completion cannot rewrite the sealed record. A
syscall itself is not falsely advertised cancellable.

Topology comes from bounded `thread_siblings_list` and per-CPU NUMA nodes. Sibling rows
must form symmetric physical partitions. Held logical CPUs expand to their physical
sibling footprint; there is no `c+96` assumption. Foreign intervals use process-total
`/proc/<pid>/stat` tick deltas for one unchanged PID-start identity. Affinity intersection
is named `potential_physical_claim_overlap`; it is not attributed per-claim CPU time.
Born/disappeared identities, reuse, counter reset, census read gaps, and permission errors
are retained and make coverage unknown instead of disappearing.

Target multi-file capture checks PID-start before and after. It uses process-total
`smaps_rollup`, and NUMA rows retain `kernel_page_size_kb` alongside per-node page counts.
Memory availability, swap/vmstat, PSI, CPU/NUMA affinity, current processor, and effective
THP are sampled in-window and bound to requested recipe/instrument identities.

## Verifiers and GPU facts

When `foreign_verifier` is absent, process purpose is `unknown`; names never infer model
inference. When `runtime_verifier` is absent, every requested knob witness is `unknown`.
Only trusted verifier callbacks can emit verified `ordinary`/`model_inference` or
`unsupported`/`compiled_unproven`/`fired_under_target` facts. A fired result requires its
verifier evidence reference; caller-authored `authenticated:true` is not a contract.

`prepare_artifact_identity()` runs before the observation window, hashes within a byte
bound, checks metadata before/after hashing, and pins device, inode, size, mtime, ctime,
path, and digest. In-window DSO evidence matches the mapped device/inode to that prepared
identity; it never hashes whichever file later occupies the pathname.
Cheap stat readback compares device/inode/size/mtime/ctime during capture, so replacement
and same-inode modification become unknown. A deleted pathname remains mapped-inode
evidence but cannot be upgraded to a current prepared-identity match.

GPU evidence keeps global device VRAM separate from target attribution. Positive
measurement residency requires the prepared DSO mapping, target `/dev/kfd` FD, and a
target PID-start/boot-specific allocation from an explicitly injected trusted GPU
adapter. The collector rechecks `/proc/<pid>/stat` around that adapter call. Global VRAM
from another process cannot make the target positive. Missing target attribution is
`unknown`; observed zero target allocation is `not_observed`. CPU residency is
`not_applicable`.

## Loaded instrument identity

`loaded_instrument_identity()` produces
`epyc.autokernel.loaded_serving_instrument.v1`. It separately names the selected loaded
measurement callable and clock callable, supporting callables, explicitly used constants,
and dependency versions. Python implementation projections cover bytecode, behavior
constants/nested code, exception tables, flags, names, arguments, and free/cell variables.
Defaults, keyword defaults, closure values, and supported bound-instance state are pinned
when stably serializable. Bound state includes both `__dict__` and readable `__slots__`
declared across the instance MRO; an unrepresented or unstable part makes configuration
explicitly `unproven`. Missing
dependency versions also make completeness false. Builtin/extension callables are
version-scoped and explicitly unproven rather than claimed as bytecode-covered.

This utility is intentionally not wired into strict plan/native carrier schemas yet.
The coordinated migration must introduce an explicit arm/plan version, recompute the
identity from the loaded serving path, carry its digest in native evidence, and treat
legacy absence as `instrument_identity_unknown`; it must not infer identity from current
files, labels, old records, or mtimes.

No default or invented KFD/debugfs file format is assumed. If the provider cannot supply
a demonstrated native adapter with PID-start/boot-specific allocation evidence, GPU
attribution remains unknown. Tests use an explicitly fake adapter only. The observer does
not authenticate callbacks; that authority belongs to the lifecycle/provider bridge that
constructs the session.

## Existing direct CPU loop: factual noise and bounded arm reschedule

The separate, unsealed `residency.CpuLifecycleSampler` now retains aggregate CPU
ticks, memory/swap counters, memory PSI, and bounded non-target process CPU-tick
intervals during its original launch lifetime, including host reads before target
attachment. Each positive process interval retains both PID/start identities and
allowed CPU lists. These are observations, not a model-inference classifier or a
physical held-region overlap proof. Missing reads, partial censuses, phase crossings,
gaps and count/time/byte exhaustion remain explicit. No endpoint substitutes for a
missing during-phase observation.

P-AK-SEARCH-1-A2 ordinary host/build activity and PSI remain diagnostic noise. They
never block, abort, or trigger a retry. No process-name classifier, pressure cutoff,
new noise floor, or foreign process signal was added. Contention and actual NUMA
page placement remain unproven; competing-inference classification is unavailable
in this direct sampler.

After readiness, the same PID/start identity changing is an instrument contradiction.
A task affinity outside the original recipe invalidates only after **two distinct,
non-crossing samples of the same TID/start identity**; a single sample does not act.
Setup/exec and teardown are excluded from that decision. Missing facts never confer
a positive placement or quiet-window warrant.

An invalid server launch finishes owned teardown before raising `MeasurementInvalid`.
Its original resolved recipe/build artifacts, request bytes/digest, failed condition,
whole retained lifecycle record and inadmissible raw rate survive in the existing
experiment payload as `measurement_invalid`, without a comparison/effect or null.
The loop may charge one additional existing iteration-budget draw and archive this
invalid outcome before rescheduling that whole original server launch once. The
same serialized tail retains the hypothesis, archived patch and build: no reset,
reauthoring, rebuild, profiling or recalibration occurs. Completed valid launches
are not repeated. The final comparison identifies the archived invalid attempt by
digest and still uses the existing serving reducer and belief export.

STOP, exhausted budget, a second invalid launch, or an unresolved server cleanup
prevents rescheduling. This is an in-process continuation, **not restart recovery**;
stored JSON cannot recreate it. The recipe-only path retains its original arm pair
and remains subject to its owning runtime admission, not a borrowed source floor.
