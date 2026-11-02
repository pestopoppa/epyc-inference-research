# KPF-50: shared-trunk attribution

The scheduled KPF-50 trial profiles the 4 × 80k / 60k-fork workload. Its
rocprof trace alone does not identify the time spent reading shared-trunk cells:
flash attention fuses the reduction over trunk and private cells. In particular,
whole-run `fattn` totals include prompt prefill and cannot be divided by overall
wall time to claim the proposed 20% gate.

`analyze_profile.py` is a fail-closed **diagnostic reader** for a future trace
where all of the following are recorded before the profile is interpreted:

1. Seal model, server, HIP DSO, profiler, prompt and native fork-witness hashes;
   record GPU process residency and exclusive claim, DFlash2, exactly four live
   sequences at 80k context and a 60k trunk, and three actual child-fork events.
2. Record complete decode and verify call boundaries synchronized to GPU work.
   P3's `KPF30_BEGIN/END` alone has no timestamps or clock calibration and is
   insufficient. Capture host monotonic and GPU profiler timestamps with a
   bounded calibration before and after each observation window; convert the
   boundaries to the rocprof `BeginNs`/`EndNs` clock. Refuse ambiguous clock
   conversions or kernels crossing a boundary. Preserve unknown phases.
3. Export native rocprof per-dispatch `KernelName`, `BeginNs`, `EndNs` CSV and
   seal its sha256. Predeclare exact kernel-name patterns. Keep raw kernel rows,
   including non-attention work, and per-step spans, including missing attempts.
   Union overlapping GPU intervals to avoid counting concurrency twice.
4. The diagnostic `attention_busy_ns / step_ns` is the fraction of the measured
   synchronous step occupied by any identified attention dispatch, not trunk
   time. The analyzer always emits `trunk_attention_ns: null` and
   `gate_verdict: UNRESOLVED`. An empty attention set, missing residency,
   missing witness, mixed/unknown phase, crossed boundary, malformed trace or
   absent clock calibration refuses output; it is never zero or a pass.

## Prospective trunk-marginal experiment

To obtain a useful attribution rather than guessing from token counts, capture
representative query/K/V/mask and stride/layout metadata from actual verified
four-sibling decode and verify steps. On the same full-candidate GPU build,
replay each sampled attention operation in balanced interleaved full-context and
private-only matched pairs, using the same query, kernel dispatch family,
physical K/V layout and corresponding private mask entries; remove trunk only
by a validated diagnostic view/mask and preserve live private-cell positions.
Collect enough repeats for the A/A noise floor, record per-dispatch native device
intervals and use `full_ns - private_only_ns` as a **counterfactual marginal**.
This is not a timed segment of the original fused kernel: a change in kernel
selection, memory locality, launch geometry, or graph scheduling invalidates the
comparison. Also separately profile non-attention and FA kernels inside the
matched original step; never sum independently overlapping dispatch times. Check
clock calibration, actual fork witness, shape and per-path coverage. Run an
A/A control, a no-sharing control and a deliberate perturbation that must be
rejected before a model run is treated as decision material. Decide whether the
result qualifies under the existing measurement authority; the proposed 20%
threshold is not ratified by this tool.

Experimental llama.cpp branch `experimental/rtg58-kpf50-iteration-20261102`
adds opt-in synchronized CLOCK_MONOTONIC step markers and records the input
sequence tokens/positions and a physical common-trunk layout signature. The
`LLAMA_KPF50_MASK_SHARED_TRUNK=1` arm removes the shared prefix only for an
eligible four-sequence, >=80k full-attention batch after all rows verify exactly
60k visible cells owned by all four sequences. Full and mask arms have the
same instrumentation. The mask arm deliberately changes the answer and cannot
support a quality, performance or serving claim. Compare only the **first**
verified affected step in independent fresh full/mask/mask/full windows. The
later masked steps have diverged K/V state and are invalid for pairing. Identical
input token IDs and layout signatures are necessary but do not themselves prove
K/V value identity; record that residual uncertainty.

`hsa_clock.py` samples raw HSA system ticks in host CLOCK_MONOTONIC brackets
before and after each profiler window; `offset_interval` rejects a frequency
change or unstable cross-clock offsets. Validate with an actual GPU calibration
kernel that rocprof `BeginNs` shares this HSA timestamp epoch before accepting
the transform. The reader refuses absent clock calibration; no current run has
established this alignment. The queued original profile may supply whole-run
descriptive kernel distributions, but cannot close KPF-50. Never invent
calibration or fork witnesses after the run.

The successor packet uses the exact retained `kpf18_gate.py` and
`kpf18_arms.py` hashes in `runner-source.json`. Reconstruct the runner with
`prepare_runner.py --source-dir <retained-runner> --out <new-dir>`; it refuses a
source or prepared-image hash mismatch. The patch enables the profiler and
clock probes and changes item-7 to four **fork-on** windows in full/mask/mask/full
order. Its records explicitly label the synchronized/masked run diagnostic and
carry no performance or quality decision. Use `select_step.py` to refuse
nonidentical inputs, phase, row count or layout before comparing the first
eligible step. It does not claim that matching signatures establish equal K/V.
`verify_clock_smoke.py` requires an actual rocprof-observed HIP witness to fit
between its host markers under the HSA clock-offset interval; simulator tests
alone never establish GPU clock alignment.

The target-step markers bracket `llama_decode`; separate synchronized iteration markers span `pre_decode` (including drafter), target decode and post-decode. The latter omit inter-callback scheduling gaps and both markers perturb real scheduling. Never report them as production throughput.
