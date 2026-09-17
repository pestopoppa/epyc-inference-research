# Champion `ef81196d5` — every launch that measured it today, as DATA

**This is a between-launch precision estimate for the champion, not a caveat.** It is the `n` and
the `sd` anyone needs to size a proper final characterisation, and it is reusable directly.

Champion configuration = shim OFF, champion knob state. `bin-h1` at its defaults and `bin-r1` with
`GGML_SOLO_YIELD_ROWCOL=0, GGML_SCALE_SPLIT=0, GGML_TINY_SOLO_CLAMP=1, GGML_GET_ROWS_SOLO=0` are the
same configuration — proven **24/24 byte-identical**, and their `ggml/src/ggml-cpu` git tree object
hashes are identical (`040d43aa…`). Shim-ON sessions are excluded: they are not the champion default.

| launch | start (UTC) | binary | champion arms | **mean t/s** | arm spread | screens | host conditions |
|---|---|---|---:|---:|---:|---|---|
| `S1_OFF` | 09:33:12 | bin-h1 10242 | 5 | 25.3452 | 7.14% | 5 kept | morning, **contended** (undetected bandwidth confound) |
| `RA_AA` | 09:53:41 | bin-r1 10303 | 5 | 25.3211 | 2.15% | 5 kept | **concurrent GPU-chain** work |
| `QA_AA` | 11:18:57 | bin-r1 10303 | 5 | 25.6953 | 1.05% | 5 kept, **1 dropped** | quiet host (3rd-party 8-core python hit Q2) |
| `S11_OFF` | 11:40:09 | bin-r1 10303 | 2 | 24.3820 | 1.66% | 2 kept | quiet host |
| `S13_OFF` | 11:53:32 | bin-r1 10303 | 2 | 24.8187 | 0.28% | 2 kept | quiet host |
| `S15_OFF` | 12:06:30 | bin-r1 10303 | 2 | 24.6616 | 2.52% | 2 kept | quiet host |
| `S17_OFF` | 12:19:31 | bin-r1 10303 | 2 | 25.2876 | 0.30% | 2 kept | quiet host |
| `S20_FIX1` | 12:33:16 | bin-r1 10303 | 4 (C arms) | **27.6058** | 0.43% | 4 kept | quiet host |
| `S30_CP` | 13:15:03 | bin-r1 10303 | 5 | **27.3826** | 0.48% | 5 kept, 1 flagged | quiet host |

All: hot harness, 24-prompt production mix, token-weighted decode, `-np 1 -c 8192 -t 48 --no-mmap`,
`taskset 0-95`, `numactl --interleave=all`, `GGML_IQK=1`, `-fa on -ctk f16 -ctv f16`, GPU loop down.

## Between-launch precision — the number to reuse

| set | n launches | mean | **sd** | range |
|---|---:|---:|---:|---:|
| all launches | 9 | 25.6111 | **4.457%** | 24.382 – 27.606 (**12.59%**) |
| **quiet host only** | 7 | 25.6905 | **5.081%** | 24.382 – 27.606 (**12.55%**) |

**Restricting to a quiet host does not shrink the spread — it slightly widens it.** Contention is
not the driver. This is host/process state that a clean window does not remove.

**Note this is LARGER than the 2.793% between-session sd measured inside the THP block.** Both are
correct and they answer different questions: 2.793% is between launches **within one contiguous
~52-minute block**; 5.081% is between launches **spread across ~3.7 hours**. A characterisation
whose launches are spread over a working day faces the larger figure.

### Launches required for a champion headline

```
  +/-0.5% : n = 397 launches = 25.14 h
  +/-1.0% : n = 100 launches = 6.33 h
  +/-2.0% : n =  25 launches = 1.58 h
  +/-3.0% : n =  12 launches = 0.76 h
```
(one launch = 228 s: 7 s evict + 44 s model load + ~2 s verification + ~165 s arm + ~10 s teardown)

**A ±1% champion headline costs ~100 launches / 6.3 h.** A ±3% headline costs 12 launches / 0.76 h.
That is the real price of the number, and it was invisible while single-session precision (0.4–0.5%)
was being quoted.

## An unexplained step, reported as observation

The series is **not** smooth drift. It sits at 24.4–25.7 from 09:33 through 12:20, then steps to
**27.61 at 12:33 and 27.38 at 13:15** — a **+9% jump between 12:20 and 12:33** — and the two
post-step launches agree with each other to 0.8%. Nothing I recorded explains it: both screens were
clean, NUMA placement was constant, `AnonHugePages` was flat, and the orchestrator API stop (12:05Z,
operator-ruled idle and a non-event) is 15–28 minutes earlier and on the wrong side of the step.

**Reported as an observation, not a cause.** It is the single largest term in the table's spread and
it is unattributed.

## Cross-campaign conclusion

**Any single-launch headline for `ef81196d5` is inadmissible on either surface — CPU or GPU — until
several launches are quoted with between-launch precision.** Within-launch tightness of 0.4–0.5% is
real and is measuring the *launch*, not the champion; quoting it as the champion's precision
overstates it by roughly **tenfold**.

The GPU session's `tg128` A/B **alternates processes and therefore already samples the session
unit** — their methodology was correct on this axis where the CPU side's was not. This is recorded
as a cross-campaign methodological conclusion, not an INF-70 caveat: **the replication unit is the
process launch whenever the quantity varies by launch**, which the table shows it does, by 12.5%.
