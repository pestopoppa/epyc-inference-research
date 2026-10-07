# GPU local resource phases

For a selected GPU serving target, `--gpu-cpu-region-claim off` avoids a CPU
region hold across the batch. The original `mi210_0` device claim remains held
throughout. With `--cpu-measurement-gpu-quiet lock`, local work uses separate
original acquired owners:

- Every compiler invocation takes a `build` CPU-region claim covering its declared
  CPU affinity, with SMT siblings folded onto physical cores. It releases that
  claim after the build and its cookie-owned compiler-descendant cleanup. Nested
  compiler wrappers reuse that same owner. This hold takes no gpu-quiet lock.
- GPU correctness, profiling, calibration and measurement take gpu-quiet
  EXCLUSIVE for their compute body. Nested compute calls reuse the original
  owner. They release it before hosted planning or the next build.
- An author's `ak-check` inherits a per-batch capture directory, acquires its own
  narrow `build` owner, and retains the release receipt there. A pending child
  capture or failed owner release refuses accounting rather than becoming a gap.

Both local providers latch their original native entry before constructing the
receipt or observing its open state. Any later entry, observation, release or
retention failure, including `BaseException`, poisons publication and preserves
the child's pending refusal. Only a known busy refusal before native entry is
retryable; an unexpected entry failure or omitted callback cannot prove an idle gap.

Every `local_compiles` caller captures its configure/build descendants under the
original native cookie guard and verifies cleanup before the build owner closes,
on success, failure and timeout. Supplied active owned environments keep their
original cookie and temp paths, with a verified sweep inside the native owner.
Author `ak-check` builds carry their own captured cookie through subprocesses.
Marked source/build paths receive durable exact-cookie fences before launch;
uncertain cleanup poisons phase publication and retains those paths and the
author's pending marker. Parent retention preserves a child's durable disk fence.

`direct_held_intervals.v2` retains the original device component and a bounded
list of original local components. Reopening checks each owner's lock identity,
open/close observations, clock domain and release, and refuses overlapping local
owners or intervals outside the original device interval. It partitions device
ownership at every local boundary into `held_claim_receipt.v2` records.

Each partition charges the device throughout and physical CPU regions only while
a real CPU owner held them. Quiet exclusion contributes its physical lock ID and
zero physical CPU fraction. Hosted gaps retain only device ownership. Affinity
and the proposal's predicted CPU fraction never substitute for an actual hold.
The proposal retains its original nonzero host estimate; estimates and actual
receipts have distinct roles.

The old `--cpu-measurement-gpu-quiet off` path still requires the launcher's
continuous EXCLUSIVE quiet hold. Its v1 intervals and v1 receipts remain unchanged,
including rejection of zero-host GPU v1 receipts. Historical journals are not
rewritten. Measurement grading and numerical gates are unchanged.

The EXCLUSIVE quiet owner takes no CPU regions. It excludes clients enrolled in
the native quiet protocol; a plain `region-lock run --role build` without
`--gpu-quiet shared` acquires CPU regions alone and is not excluded by quiet.
Heavy CPU test/build clients scheduled during a GPU measurement period must
acquire the original native SHARED quiet owner before their CPU regions (the
combined provider / CLI `--gpu-quiet shared`). Existing enrolled CPU lanes use
that ordering for their whole acquired build/measurement tail. Own queued
validation must finish before live GPU timing. A one-time preflight observation
does not reserve isolation or authorize overlapping heavy work.
