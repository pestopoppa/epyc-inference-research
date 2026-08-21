# MI210 power-sensor characterisation (RVP-PWR-2 / PWR-5) + PC-sampling capability probe (RVP-C4-10)

**2026-08-21, gfx90a MI210, ROCm 6.2.** Observations only per MEASUREMENT.md — no protocol id,
gates nothing. GPU idle-verified before every run (zero KFD clients, host-wide fd scan).

## Measured results (two-run persistence on the long-wave numbers)

| Quantity | run1 | run2 | Meaning |
|---|---|---|---|
| idle draw (derived dE/dt) | 59.8 W | 60.8 W | cross-checks `--showpower` |
| active plateau (1024² fp16 mm loop) | 224.1 W | 224.5 W | realized load level |
| **averaged field t_d** (delay to 10%) | **192.7 ms** | **184.1 ms** | `power_average` responds ~0.19 s late |
| **averaged field t_r** (10→90%) | **4418 ms** | **3913 ms** | ~4 s to track a step up |
| **averaged field t_f** (90→10%) | **3566 ms** | **3479 ms** | ~3.5 s to decay after off |
| derived (energy-counter) t_d / t_r | ≤12 ms / ≤2 ms | 1.2 ms / — | ~100-1000× faster than the averaged field |
| sampler cadence (rsmi via ctypes) | 107 µs/sample sustained | — | 1700× faster than the rocm-smi CLI |
| energy-counter refresh | 236 distinct values in 236 ms | — | **the 1 ms cadence CONFIRMED on-die** |

**Aliasing (FFT of derived power, commanded-period ground truth):** 10 Hz → peak 9.9 Hz,
34.3 dB over floor; **250 Hz (the paper's aliased case) → peak 249.94 Hz, 32.2 dB — CLEAN** (n=1).
500 Hz analysed at a 2 ms bin (Nyquist 250) → peak shifts to 187.7 Hz at 2.8 dB — the textbook
signature (peak shift + collapsed prominence), **validating the FFT detection test itself**.

## Three findings beyond the numbers

1. **arXiv:2604.06056's "~4 ms aliasing knee" is an INSTRUMENTATION-LAYER cost, not a sensor
   limit** — consistent with the paper's own wording ("widened by the costs of sampling and
   logging"). With a 107 µs sampler the 1 kHz counter resolves 250 Hz cleanly; the hard limit is
   the counter's own Nyquist (500 Hz). Our attribution floor is therefore set by OUR tooling,
   and this tooling is ~40× faster than the one in the paper.
2. **Consequence for W_conf on the averaged field: a phase must exceed ~8 s** (0.19 s delay +
   ~4.2 s rise on entry, ~3.5 s fall contaminating the tail) **before ANY attributable interior
   exists.** For shorter phases the averaged field is unusable and dE/dt is the only instrument.
3. **API TRAP (cost us one wrong analysis pass): `rsmi_dev_energy_count_get` returns the RAW
   COUNTER; energy_uJ = counter × counter_resolution (15.3).** The CLI pre-multiplies; the API
   does not. A naive dE/dt under-reads by exactly 15.3× and produces a PLAUSIBLE-LOOKING wrong
   number (3.9 "W" idle). Cross-check derived power against `--showpower` at idle before
   trusting any pipeline built on this API.

Also in this directory: `pcs_capability_probe.c` (RVP-C4-10) — proves PC sampling on ROCm 6.2 is
a STUB: the API is exported but returns status 16 "defined but not implemented" against the live
gfx90a agent. The CLI flag does not exist. GPA's input class is unobtainable at this ROCm version.

Load-generator lesson: submit-and-sync EVERY op — an async backlog past the phase edge halved the
realized wave frequency (measured, exactly −50% FFT peak) before the fix.

## Follow-on 2026-08-21 (late): token-cadence phase-lock on REAL decode — BOUNDED NEGATIVE

Two llama-bench tg-only decode runs (production b10125 binary, own libs), sampler concurrent:
gemma-4-e2b Q8 @ **122.4 tok/s** and Qwen3.8-27B Q8 @ **30.5 tok/s**. In the settled steady window
(>6 s past load start), averaged field vs energy-counter ground truth: **−0.86% and −0.84%** —
**unchanged across a 4× cadence change, which is the decisive test: a phase effect must move with
cadence.** The token cadence is essentially invisible in the dE/dt spectrum (0.4 dB at 122 Hz,
−1.1 dB at 30 Hz): steady decode is back-to-back compute, near-DC at the token scale, so the
modulation that could phase-lock barely exists. **Scope of the all-clear: STEADY, SETTLED decode
only.** The measured hazards stand where the square waves put them — transitions (~190 ms delay,
~4 s rise) and duty-cycled/bursty loads (request gaps, prefill/decode alternation), where dE/dt
remains the only instrument. Ops note: this build's llama-cli is a chat REPL that ignores `-no-cnv`
and floods stdout (0.9 GB of redraw in one hung run, orphaned llama-cli killed + verified);
llama-bench is the right decode-load generator.
