# M-3: MTP draft depth 4 vs 8 on production `:8083` traffic, from the server log only

**Question.** The olympiad bench in this study (README §6) showed depth 4 and depth 8 within about
±2.5% per request. Does production traffic on `:8083` (Qwen3.8-27B Q8_0, MI210) confirm that? No
requests were sent for this analysis. It reads only
`/mnt/raid0/llm/epyc-orchestrator/logs/llama-server-8083.log`, a single un-rotated file with 9 server
instances appended to it. The log was read at 2026-09-24 19:22Z.

**Verdict: depth 4 is NOT confirmed on production traffic. The depth-4 arm has 0 organic requests.**
The log-only projection points slightly *against* depth 4 for the typical request. It is a model,
not a measurement (see *Projection*).

## Arms and how they were anchored

The llama.cpp log stamps are process-relative (`min.sec.ms.us`). The two arms were pinned to
wall-clock time like this:

| Arm | Log instance (lines) | Recipe evidence in log | Wall-clock anchor |
|---|---|---|---|
| **depth 8** | #7 (21434–35038) | `n_max=8`, `n_slots = 2, n_ctx_slot = 98304, kv_unified = 'false'` | It exited at +1912.7 min, just before the study's first driver start (15:37:37Z), which needed `:8083` stopped. That puts its launch at ≈ 2026-09-23 07:44Z. All 240 requests fall between +1495 and +1788 min, **≈ 08:39Z–13:32Z on 2026-09-24**, before the sweeps. |
| **depth 4** | #9 (35063–end) | `n_max=4`, `n_slots = 4, n_ctx_slot = 196608, kv_unified = 'true'` | `orchestrator_state.json` gives `started_at 2026-09-24T18:44:15`. |

Excluded:

- **Instance #8.** It ran from ≈18:20Z to 18:44Z at `-lv 3`, so it did not log its draft depth. It
  served one request, inside the stated 15:36–18:42Z window.
- **Bench servers on `:18072`.** They log elsewhere.
- **Older Qwen3.8 instances #4 and #6** (`n_ctx_slot` 32768 and 98304). They do not log `n_max`
  either. #4's maximum mean len of 9.00 proves depth 8 there. They are shown below only as a
  sensitivity check, not as the arm.

**Traffic composition caveat.** The depth-8 window is dominated by agentic long-context sessions. The
DS41 opencode planner overflowed the 98,304-token slot at 09:42Z (`progress/2026-09/2026-09-24-main-dsv41.md`),
and 3 requests in this arm end `truncated = 1` at 98,303 tokens. The KV context at request end
has median 63.5k and IQR 34.1k–78.9k.

## Per-request distributions

Regex: `draft acceptance = a ( acc accepted / gen generated), mean len = m` together with the
`eval time … tokens per second` line of the same task. "mean len" is tokens per target pass
(1 + accepted drafts). Decode tok/s is the server's per-request eval rate.

| Arm | n | mean len median [IQR] | acceptance median [IQR] | decode tok/s median [IQR] |
|---|---|---|---|---|
| depth 8, all requests | 240 | 4.86 [3.96–5.83] | 0.483 [0.370–0.604] (token-weighted 0.350) | 38.2 [30.6–47.7] |
| depth 8, solo (no overlapping request) | 199 | 4.87 [4.03–5.84] | 0.484 [0.378–0.605] | 40.8 [33.9–49.3] |
| depth 8, solo, ≥ 100 generated tokens | 175 | 4.86 [3.97–5.78] | 0.483 [0.372–0.597] | 39.6 [33.8–48.5] |
| (sensitivity) instance #4, depth 8, 32k slots | 281 | 5.00 [4.43–5.50] | 0.500 [0.429–0.562] | 48.6 [22.3–61.8] |
| **depth 4, all requests** | 22 | 2.96 [2.25–3.10] | 0.492 [0.314–0.524] | 28.8 [25.8–34.9] |
| depth 4, organic | **0** | — | — | — |

Depth-8 decode by context band (solo, ≥ 100 tokens):

| Context at end | n | tok/s |
|---|---|---|
| 8–32k | 24 | 45.8 |
| 32–64k | 46 | 41.7 |
| ≥ 64k | 104 | 38.3 |

So any comparison must be context-matched.

**None of the 22 depth-4 requests is production traffic:**

- **1 post-reload probe**, 20 s after start: 192-token prompt, 939 tokens, 35.9 tok/s, mean len 2.65.
- **1 long-prompt probe**: 124,173 prompt tokens, 16 generated.
- **6 identical synthetic requests**: 21-token prompt, 100 tokens. Five of them logged the same
  52/185 acceptance.
- **14 M-4 requests**: fixed length, `ignore_eos`, temperature 0, 1024 tokens, about 1k context.
  At concurrency 1 they ran at 40.5 and 40.2 tok/s with mean len 2.97 and 2.95.

These run at about 1k context against production's roughly 60k, on different content, so they are
not comparable to the depth-8 arm. M-4's aggregate numbers are in `m4_np4_concurrent_live.json`.

## Projection from the depth-8 log (a model, not evidence of depth 4)

`-lv 4` logs `acc per pos` for every depth-8 request. Every depth-8 draft was full length:
#gen tokens / #gen drafts = 652,582 / 81,579 = 8.0. The MTP draft chain is autoregressive with
`p_min = 0`, so positions 1–4 should not depend on `n_max`. Truncating the chain at 4 then predicts
depth-4 behaviour on the *same* production requests:

- **Draft-weighted acceptance by position:** .739, .550, .416, .324, .255, .205, .168, .141.
  Positions 5–8 carry **27.5%** of all accepted tokens.
- **Predicted mean len at depth 4:** 3.03 against 3.80 observed at depth 8 (draft-weighted), a
  ratio of 0.797. The per-request ratio has median 0.732 and IQR 0.672–0.795.
- **Target-pass rate from the bench (README §6 cells).** Depth 4 is about 1.27× depth 8:
  11.3 vs 8.9 passes/s at np 2, and 8.1 vs 6.4 at np 4.
- **Projected depth-4/depth-8 decode tok/s:** 1.01 aggregate, but **0.93 for the median request**
  (IQR 0.85–1.01).

Production traffic is more draftable than the olympiad bench: the median depth-8 mean len is 4.86
here against 4.35–4.82 on the bench. Depth 8 therefore earns more on production than on the bench.
Two caveats limit the projection:

- The pass-cost ratio comes from olympiad cells at 8k context or less, and may differ at 60k context.
- The truncation assumption has not been validated against a matched depth-4 run.

## What it would take to confirm

The depth-8 production arm is **closed**: no more depth-8 traffic will arrive. It holds 175
comparable requests (solo, ≥ 100 generated tokens), with a robust CV of decode tok/s of 0.275.
The power figures below use a two-sample Mann-Whitney test, α 0.05, power 0.8:

| Organic depth-4 requests (same stratum) | Smallest resolvable median tok/s difference |
|---|---|
| 125 | ≈ 9% |
| 175 | ≈ 8% |
| 500 | ≈ 7% |
| unlimited | ≈ 6% |

Context-band matching reduces the effective n further.

**A ±2.5% effect, the size the bench suggests, can never be resolved from these logs.** That needs
an interleaved same-window A/B on production-shaped prompts. At the depth-8 arm's pace (240 requests
in about 5 active hours), roughly 175 organic depth-4 requests will take one to two comparable working
days. Nothing was set up here.
