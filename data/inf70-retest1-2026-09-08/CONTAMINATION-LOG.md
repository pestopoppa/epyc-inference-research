# RETEST-1 — contamination events and their disposition

## E1 — 2026-09-08T09:30:30Z–09:32:00Z, peer-disclosed `jobs=64` compile on `96-183`

**Source of the report: the GPU session's own disclosure, relayed by the coordinator — NOT my
contention screen.** Their draining loop handed a serialized tail to another lane, which started a
64-way compile pinned to `96-183`, the SMT siblings of bench cores 0-87, with no announced gap.
They killed it at the build stage (~1 minute), verified the process group empty and the GPU claim
free.

**Disposition: NO ARM DROPPED — none overlapped.** My campaign took the region at **09:33:12Z**,
1 minute 12 seconds after the window closed. Sequence from `campaign1.nohup.log` and
`runs/S1_OFF.timeline`:

| time | event |
|---|---|
| 09:30:11Z | PREREGISTRATION.md frozen (sha256 recorded) |
| 09:30:30–09:32:00Z | **contamination window (peer-disclosed)** |
| ~09:32:50Z | region-lock dry-run — acquired and released cleanly |
| 09:33:12Z | region acquired, campaign 1 starts |
| 09:33:19Z | targeted eviction done in 7 s; server launched |
| ~09:33:5xZ | first arm `AA1` begins |

**What my screen read across that window: nothing, and I will not claim otherwise.** The
`foreign_load.py` sampler runs *per arm*, and no arm was running between 09:30:30Z and 09:32:00Z,
so I have **no independent reading of the burst**. This is neither a validation nor a failure of
the sampler — it is a gap in coverage, and it is the honest statement.

That gap is itself worth recording for MEAS-1/OP-41: **the in-window sampler only covers windows
that contain an arm.** A burst that lands between arms, or between the eviction and the first
arm, is invisible to it. On this host the only instrument that would have caught E1 is the peer's
own disclosure, which is exactly what happened. The contract worked; the instrument would not have.

**The one number this window could have destroyed is the A/A floor**, and it did not touch it: the
calibration block began after the window closed. Per the coordinator's instruction, a
known-contaminated arm is not rehabilitated by a screen that failed to see it — that rule is
recorded here and is binding on any later event, but it does not fire for E1.
