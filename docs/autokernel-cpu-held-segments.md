# Original CPU ownership generations

`--cpu-held-intervals-v2 on` is an opt-in scheduled CPU writer. The default `off`
preserves the v1 envelope and reader behavior. Activation requires an
owner-controlled completed-batch boundary and a new compatible serial parent
and worker epoch, coordinated with native owner identity v2. Updating files
does not update an already imported parent. No live CPU writer is activated
by this change. Historical journals are unchanged.

`claim.hold_cpu(capture_segments=True)` retains a fresh original `HeldCpuClaim`
only after its own native provider has acquired the enrolled CPU regions.
`CpuClaimLease.close_owner` observes that owner before its exit and records
completed release only after the original exit returns. Every reacquisition
has a distinct original context, monotonically increasing ownership generation,
the same physical lock identities, fraction, affinity and process/clock domain.
Failed observation, uncertain exit, repeated/nonsequential generation, changed
allocation or overlap refuses v2 publication/reopening. The separate CPU-window
hint ledger never supplies an interval. The run writes a durable refusal record
when v2 accounting capture fails. Normal and failed settlement read that
selection-bound refusal before evidence or preclaim fallback. A separate original
native-provider entry event is written before observer construction; it can
refuse an uncertain post-acquisition failure but cannot charge or imply zero CPU.

`claim.publish_intervals(cpu_segments=True)` writes `direct_held_intervals.v2`
with original CPU components, an empty GPU phase list, and strict
`stage_elapsed`: original outer context ID, same boot/process/monotonic domain,
original scope start/end and completed outer context. This separate clock frame
contains no fraction or ownership warrant. Every component must fit inside it. The compatible
`serial_scheduling.reopen_held_receipts` creates one `held_claim_receipt.v2` per
proved original CPU generation. No component covers a yielded or hosted gap.
A v2 scope already yielded at teardown retains only its completed original
generations and does not reacquire for an envelope endpoint. CPU v1 still
reacquires at teardown and creates the historical one-envelope receipt. The GPU v2 carrier
and device-held partition path remain unchanged.

The units and prospective consumers are explicit:

- `scheduling.charge_receipts` / `SchedulerEngine._index_receipt` accumulate
  physical-region seconds as sum of original held seconds times actual region
  fraction. `account_stage_components` uses the same sum divided by configured
  physical capacity for the fairness deficit. No yielded gap contributes CPU.
- `serial_scheduling.retain_cost_sample` uses summed original CPU v2 seconds
  for the operational CPU duration forecast. `serial_run._cost_body_scope`
  separates native-generation v2 from v1 envelope samples; prospective lookup
  follows the next original input argv. Absent compatible samples retains the
  declared admission estimate, never a reconstructed past discount.
- `SchedulerEngine.account_stage_components` retains full
  original outer scope wall seconds (including trailing yielded/actor gaps) via
  the explicit CPU-v2-only `cpu_stage_elapsed` parameter for
  `campaign_charged_seconds`, seed wall controls and
  failed-stage max-duration fences. These controls are distinct from actual
  physical CPU cost. GPU forecasts continue to describe its original continuous
  device-held stage, and cannot turn that time into CPU ownership.

CPU-window `held` state remains the original lease state; this change supplies
original settlement evidence, not new scheduling permission. Quiet ordering,
CPU measurement isolation, heavy-work isolation, policy and grading are unchanged.

`serial_scheduling.reopen_cpu_stage_elapsed` reopens the same sealed SHA/selection/
target bundle and verifies domain/bounds/components. Success and failure serial
settlement pass that validated frame explicitly. CPU receipt IDs bind its digest,
so a replay cannot change wall scope while retaining the same receipt group.
CPU v1 and GPU readers/shapes/contiguity remain unchanged.

The legacy `recover_abstained_overrun_fence` migration cannot reopen the original
CPU-v2 wall frame from receipt spans alone. It retains failed/invalid CPU-v2
fences rather than clearing a true wall overrun from a short held sum. Its
existing v1/GPU recovery behavior remains unchanged.

CPU-v2 `runtime_recovery._verify` first reopens the original native segment
evidence and checks authoritative refusal. It joins runtime interruption holder
identity and next-holder nonoverlap to the validated original outer clock frame,
which supplies no region fraction or acquisition permission. Replacement still
requires an actual new native holder observed held. CPU v1/GPU holder joins retain
their existing component path.
