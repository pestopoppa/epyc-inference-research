# KVU-12.6: prospective occupancy and length comparison

**Draft, not an authorized run or completed predeclaration.** The handoff asks why the np8
decode rate fell from 26.71 to 10.18 tok/s between L32k and L64k, separating pool occupancy
from np/length and accounting for per-stream resume depth. The old values are historical
observations, not a baseline that can be mixed with a new placement. This plan introduces no
new scientific threshold and does not reopen the separately declined 128k bank/pool-full work.

## Two comparisons on one pool

Use unified KV, pool 524,288 cells and the same model, candidate binary, MTP configuration,
cache policy, CPU placement and thread count for every arm:

| Arm | Concurrent streams | Length | Nominal aggregate | Purpose |
| --- | ---: | ---: | ---: | --- |
| `np8_32k` | 8 | 32,768 | 262,144 | Short-depth reference |
| `np8_64k` | 8 | 65,536 | 524,288 | Longer-depth / higher-fill arm |
| `np8_48k` | 8 | 49,152 | 393,216 | Equal-nominal-token control, np8 |
| `np4_96k` | 4 | 98,304 | 393,216 | Equal-nominal-token control, np4 |

Compare `np8_32k` with `np8_64k` to reproduce the length-and-fill contrast under the new
instrument. Compare `np8_48k` with `np4_96k` to examine np/length at similar occupancy.
The first comparison intentionally changes occupancy and must not be labelled an
occupancy-matched pair. The second is matched only if **observed** physical-cell occupancy
meets the tolerance declared before collection; nominal totals do not establish a match.
Report actual prompt and cached-token counts per stream and wave. `cache_n` is the server's
reported cached-token count, not by itself proof of the recurrent checkpoint's native position;
capture the actual checkpoint position if the instrument exposes it, otherwise mark it unknown.

## Pinning and preparation

- Target is the frontdoor `Qwen3.6-35B-A3B-MTP-Q8_0.gguf`, SHA-256
  `c1283d8b80c3e38b2735ddbc9766d3b3126f44d6c484be419d4e101d09a76131`, from the research
  master at `7ac65aecf`. Revalidate the authoritative registry and the full file digest before
  any later run; do not silently substitute a model.
- Experimental native-gauge source is `cdf1532b993fe58bd57ba53d91659f4bb4ee508f`, based on
  `ffc1bac82eeca6f9099e1ccd9ba49703c460a115`. A separately validated champion successor may be
  used only after its actual source and build are re-pinned in the prospective plan. The
  model-free CPU build under `/dev/shm/rtg57-kvu126-unique-cells-build-20261102` was compiled
  with `GGML_NATIVE=OFF`; it proves compilation and symbol presence, not production-shape
  performance eligibility. Build the timing instrument with the approved recipe and pin its
  executable and every mapped DSO by digest; verify source/manifest-to-binary binding.
- All four arms must use the same declared prompt mode, draft depth, generation length, bank
  or source corpus, CPU mask, threads, relevant environment, and cache-idle policy. Build and
  seal the extra 48k/96k bank states under a new path; verify state-layout compatibility, all
  per-doc token counts, hashes, no missing files and request fit. The original driver refuses
  short existing token files; never extend a frozen bank silently.
- Predeclare the exact arm order and counterbalanced independent repeats, not just waves in one
  server lifetime. Bound RAM, disk, duration, stop/cleanup and CPU/GPU-quiet requirements using
  measured qualification and the active calendar owner. Recheck original exposures and use
  disjoint IDs where the applicable no-redraw rules require them. No old grant survives the
  operator's stop.

## Native observations and analysis

For each original decode wave, retain its hashed `waves_capture.jsonl` row, actual per-stream
`prompt_n`, `cache_n`, generated-token count and decode rates. Sample the native
`llamacpp:kv_cache_used_cells` gauge at least twice while **every** requested stream is decoding:
the interval starts at the latest first-token monotonic timestamp and ends at the earliest
completion timestamp. Preserve the full raw `/metrics` responses, sample request start/end,
monotonic times, process birth identity, binary and mapped-library identity. Samples outside
the overlap, an unsupported metric, missing stream, missing record, partial wave, interrupted
request, or identity mismatch are invalid/unknown; never substitute nominal totals, restored
token sums or `used_cells_est`.

Before generating observations, choose and freeze a numeric acceptable occupancy-fraction gap,
minimum waves and native samples, repeat/anchor design, and a justified noise rule. These are
**not supplied by the KVU-12.6 task**; the owner must use the applicable measurement protocol
and actual qualification evidence. An unmatched equal-token pair cannot support an
occupancy-controlled conclusion. On a matched pair, report effects and the predeclared noise
assessment as scoped evidence, not a universal causal claim; retain errors and abstentions in
the denominator. The fixed-np pair answers the declared reproduction question, not the
equal-occupancy comparison. Decide any deployment only through its own stack gate.

The analyzer accepts a pinned **pre-run declaration** separately from a pinned **post-run
evidence index**. Its `epyc.kvu126.prospective_plan.v1` requires exact four arm geometries,
the two comparisons above, expected source/binary/model identities, placement, minimum counts,
occupancy tolerance and a written noise rule. The later `epyc.kvu126.evidence_index.v1`
contains build and wave-bound runtime identity receipts plus original wave/occupancy file
hashes. This separation is essential: a post-run evidence hash cannot be part of a declaration
that was frozen before the run. The tool verifies declared receipt consistency, not independent
provenance, repeat independence, statistical admission or the noise rule's validity.

Wire the producer's native write-side belief capture and validate its planning-context canary
before a result is used for a decision. No tuple or run warrant may be invented from a later
reader. Keep the handoff open until the original comparison, missing-data and result rules have
been satisfied and reviewed.
