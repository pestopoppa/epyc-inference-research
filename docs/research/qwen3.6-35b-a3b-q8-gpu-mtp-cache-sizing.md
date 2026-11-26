# Qwen3.6-35B-A3B-Q8 GPU MTP -- `--cache-ram` Sizing A3B Recipe Notes

**Model:** Qwen3.6-35B-A3B-MTP Q8_0
**Probe geometry:** 32k context, np=1, f16 KV (ctk=ctv=f16), no unified KV
**Recipe artifact:** [qwen3.6-35b-a3b-q8-gpu-mtp.json](../../artifacts/serving-recipes/qwen3.6-35b-a3b-q8-gpu-mtp.json) (bytes preserved; not modified by this A3)

---

## Source rows

| Row ID | Purpose | SHA-256 | Verdict |
|---|---|---|---|
| `taskflow-stage1-canary-r2-20261120` (server.log) | 10+ concurrent worktree eviction log (41 eviction events) | `85533337680e2dee4307f67fb0fda5e08beae1b122adc0f53cacf15a08c4e377` | Reference |
| `ak-pfxc-probe-gpu-20261120-r2` (RESULT.json) | 8-task probe, 0 evictions at 8192 MiB cache | `1512c7b77500b66635c3ac23f46c4a3b6ff28c50f69d8066f5532bd65246f32d` | KEEP (PASS) |
| `prefix-cache-brief-20261120` (BRIEF-autokernel.md) | Brief entry PFXC-A3 context | `63c5843f7c524efcd216ad359a1589530da278d4f132a8954214ab4f0934ee96` | Reference |

**Extraction command** (from server.log):

```bash
grep 'making room for prompt cache entry' server.log | \
  sed -n 's/.*size = \([0-9.]*\) MiB).*/\1/p'
```

---

## Entry sizes

Source: `taskflow-stage1-canary-r2-20261120` server.log -- all 41 evicted entries (MiB, source order):

```
329.659  208.063 1044.502  536.559 1066.741  868.183  899.782  883.998 1026.461  679.251
1123.136  977.217  978.184  354.989  344.592 1189.547  733.401  519.842 1348.877  812.094
 973.457 1107.180 1096.003  413.289  838.923  457.959  745.443  485.260  317.898  697.916
 925.704 1207.946  878.716  992.326  410.660  462.359  884.114  629.443  147.505  388.420
 714.588
```

- Range: 147.505 -- 1348.877 MiB
- Nearest-rank p95: 1189.547 MiB

**Selection bias note:** These 41 values are evicted entries -- they represent the oldest entries at eviction time, not the full population of entries across all worktrees. The direction of sample-eviction bias is unknown. The eviction-only sample does not capture entries that survived without eviction.

---

## Cache sizing

**Sizing rule:**

```
cache-ram (MiB) = entries x p95_entry_size
```

For 10 live entries (controlled 10-worktree check):

```
10 x 1189.547 = 11895.47 MiB -> ceil = 11896 MiB
```

Deliberate margin over the 11896 MiB baseline (not nearest-256 rounding; the next 256 multiple is 12032 MiB): **12288 MiB** (12 GiB) starting host RAM budget. 12288 MiB is 3.295% above the 11896 MiB ceil.

Scaled for N live entries: `ceil(N x 1189.547)` with a deliberate margin for headroom.

**Prospective expectation from the brief, not measured in this note:** 0-3 hit-rate points (only where evictions hit a live conversation; zero kernel work).

**Falsification:** If the probe replay logs zero evictions at 8192 MiB for a 10-task trace, the sizing is not a factor.

---

## VRAM budget

**np1 brief baseline (from `prefix-cache-brief-20261120/BRIEF-autokernel.md`):**
- Peak VRAM: ~37.9 GiB (brief)

**`ak-pfxc-probe-gpu-20261120-r2` RESULT.json:**
- 0 evictions at 8192 MiB cache RAM across 8 tasks
- Peak VRAM: 37.68 GiB (c14 arm), 37.67 GiB (pfxc arm)
- Verdict: KEEP (PASS)

**np2 planning note:**
- VRAM budget np2 should be set against the distinct 37.9 GiB np1 brief baseline.
- Verify incremental KV/recurrent memory before raising np.
- Host cache RAM (`--cache-ram`) is separate from VRAM.
- The np2 memory estimate is unvalidated; do not transfer these sizing numbers to the differently shaped canonical recipe.

**Review provenance:** A3B produced and corrected the draft. Independent main review repaired the executable extraction command and the brief-baseline source attribution. This note reuses the named historical records and makes no new performance measurement.
