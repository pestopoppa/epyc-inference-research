# K11 — gemma-4-26B-A4B serving determinism on MI210 and CPU

Provenance documentation written **2026-09-15** (NIB2-73a). This is the largest campaign
directory in the repo — **524 tracked files, 23,181,080 bytes** across 38 run directories and
one loose roll-up file — accumulated by **18 commits between 2026-07-16 19:48:39 and
2026-07-20 13:20:02 UTC**, and untouched since. It had no campaign-level README, so the
evidence-durability gate warned on it and a reader arriving from one of the five registry
citations could not tell what the other 519 files were for. Nothing was moved, copied or
re-measured to produce this file and the `SHA256SUMS` beside it.

| | |
|---|---|
| scratch origin | the first run was staged at `/mnt/raid0/llm/tmp/k11-gemma4-determinism-20260716T194501Z/` and its `summary.json` still points there; every later run wrote straight into the repo |
| measured (UTC) | **2026-07-16 → 2026-07-20**, 5 calendar days |
| dates taken from | the 18 commit dates, the `YYYYMMDDTHHMMSSZ` stamps in run-directory names, and each `summary.json`'s own `created_at` (earliest `2026-07-16T19:45:21.851539+00:00`, latest `2026-07-20T13:08:58.416236+00:00`) — the three agree. **Not** file mtimes, which in a fresh worktree are checkout times |
| documented | 2026-09-15 |
| carried | **524 files, 23,181,080 bytes** — the whole campaign, nothing withheld |
| models | `gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf` and `gemma-4-26B-A4B-it-UD-IQ4_XS.gguf` (targets); `gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf` (external MTP draft head) |
| binary | `/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server` in **every** arm — experimental v7, never a frozen production tree. Builds named in the record span `d1e5a20eb`, `6ad45fa3ff` and post-candidate `12a292f0c21d` |

**Name collision, read this before linking anything:** `K11` here is the gemma-challenge
kernel-technique row. The unrelated `BULK-kbrag-autowiki-k11` in epyc-root
`progress/2026-07/2026-07-20.md:16` and `progress/2026-07/2026-07-21.md:13` is a KBRAG/AutoWiki
recall measurement and has nothing to do with this campaign.

## What "determinism" means here — say it exactly

Determinism runs assert bit-identity of *something*, and the something matters. In this
campaign it is defined by `scripts/benchmark/k11_gemma4_determinism_runner.py`:

- **The comparison key is a SHA-256 over the canonical JSON of the pair
  `{content, reasoning_content}`** — `output_hash = sha256_text(canonical_json(semantic_output))`
  (`:967`). It is **not** `sha256` of the completion text, and not the raw HTTP bytes. A
  separate `response_sha256` over the whole response is recorded at `:968` but never used for
  the verdict.
- **The predicate** (`:1041-1043`): `deterministic = len(unique_hashes) <= 1 and
  len(hashes) == args.runs and all(status == "ok")`. So `deterministic: true` means *all N
  runs succeeded and collapsed to exactly one semantic-output hash* — not token-level
  identity.
- **The topology is N fresh sequential servers, one request each** — not N requests against
  one server. Every run record carries its own `server_pid`, `port`, `server_argv` and
  `server_log`, and they differ per run. This is cross-process reproducibility across cold
  starts, which is a stronger claim than same-server repetition, and it is why the `slots: 4`
  arms are meaningful (4 slots configured, 1 request in flight).
- **Token-ID identity is a separate, opt-in second key**, present only in the `*_trace_*`
  directories as `token_divergence.all_token_sequences_identical` with a
  `common_prefix_length` and `first_divergence`. In the whole campaign it is asserted **true
  exactly once** — for the CPU control.
- Sampling is greedy throughout: `temperature 0.0`, `top_p 1.0`, `seed 42`. The
  `request_sampler_mode` field says how: `current` keeps the historical `top_k=1` payload;
  `explicit-greedy` sends `top_k=0`, `min_p=0.0`, `backend_sampling=false`,
  `samplers=["temperature"]`; `cpu-top-k` sends CPU-side `samplers=["top_k","temperature"]`.

Prompts are synthetic throughout: a compact-JSON smoke, a "200 x the word `benchmark`" exact-count
task, a `word-array-200` schema task, and a 160-word lowercase essay *"about why deterministic
cleanup matters in an inference service"* ending in the token `END`.

## What was found — non-determinism, reproduced, root cause NOT identified

The arc, and every step of it is in the artifacts:

1. **2026-07-16/17 — the short JSON smoke passes.** `k11_gemma4_determinism_20260716T194501Z`
   and `k11_gemma4_determinism_20260717Tquiet_glm_done`: 3 runs each, `deterministic: true`,
   one hash, `18/18` drafts accepted. Both produced **the same** hash
   `5ebb69a089cca8fd860d6d709e8e15babb00c794e9c655e3551899d44a412972`, so the quiet-host
   repeat reproduced the earlier output, not merely its own internal consistency. (The 07-16
   run's own README notes it was taken while GLM-5.2 was downloading; the
   `…Tquiet_glm_done` suffix is the follow-up it asked for, and that README's caveat is
   therefore stale.)
2. **2026-07-18 — extended to a long natural-language exact-count task, and it fails
   intermittently.** `k11_gemma4_long_mtp_np4_n10_default_backend_current_20260718T142203Z`
   (2 hashes / 10 runs), `…_backend_off_current_20260718T142614Z` (2 / 10),
   `k11_gemma4_long_nospec_np4_n10_current_20260718T143112Z` (3 / 10).
3. **Every proposed mechanism was then eliminated, in order** — request-sampler shape, the
   ROCm `TOP_K` backend-sampler warning, server-side stop strings, external-head MTP,
   multi-slot scheduling, HIP graph replay (`GGML_CUDA_DISABLE_GRAPHS=1`), flash-attention
   (`-fa off`), and TopK-MoE fusion (`GGML_CUDA_DISABLE_TOPK_MOE_FUSION=1`). None repaired it.
4. **The one deterministic arm is the CPU path.**
   `k11_natural_freeform_orig_q4_cpu_nospec_np1_explicit_greedy_trace_20260720T114120Z`:
   1 hash over 10 cold servers, `all_token_sequences_identical: true`,
   `common_prefix_length: 182`, `first_divergence: null`. Its paired GPU arm diverged.

The campaign's own final verdict, `k11_gpu_backend_path_ab_20260720T1204Z/summary.md:7`:

> "K11 natural-prose nondeterminism persists on MI210 no-spec single-slot with pre-sampling
> probabilities captured. CPU no-spec single-slot is deterministic on the same prompt, while
> GPU diverges with graphs on, graphs off, and flash-attention off. The remaining root cause
> is therefore broader GPU backend numerical nondeterminism or logits handoff, not
> external-head MTP, multi-slot scheduling, HIP graph replay, or flash-attention alone."

and its instruction for whoever picks it up (`:29`): *"Inspect GPU backend matmul/reduction
determinism and logits copy/handoff. Do not spend more K11 time repeating MTP, slot-count,
graph, or flash-attention toggles unless the new run changes the actual GPU math path."*

**Two separate findings are easy to conflate, so state both.** Determinism *is* recoverable
under structural constraint and is not recoverable for natural prose:
`k11_long_nospec_explicit_greedy_stop_np4_n10_20260718Tcodex/k11_explicit_greedy_stop_report.md:31-33`
records determinism **without** correctness — *"Adding the `stop` payload made the output
deterministic in this no-spec control, but did not solve task compliance. The model never
emitted the `END` marker, so the server-side stop mechanism had nothing to intercept."* (10
runs, 1 hash, 0/10 task passes, 512 words every run.) Meanwhile
`k11_natural_freeform_explicit_greedy_compare_20260720T1059Z/summary.json` `verdict` records
`explicit_greedy_repairs_natural_prose_determinism: false` and
`explicit_greedy_repairs_exact_160_word_contract: false`, and
`k11_natural_freeform_compare_20260720T0937Z.summary.json` `verdict` reads *"Both ORIG-Q4 and
UD-IQ4_XS fail the natural free-form deterministic task gate; ORIG is faster, UD is not a
replacement on this slice."*

**This campaign also narrowed an earlier, wrong framing** — worth knowing, because the earlier
text is still in the same handoff. `handoffs/active/gemma-challenge-kernel-techniques-v7.md:247`
and `:300` describe the defect as *"specific to gemma4's external-head + shared-KV path"* and
consistent with *"a load-sensitive race"*. K11.1n/o/p later reproduced it with **no** MTP,
**one** slot, on a quiet host. Both of those characterisations are too narrow.

## Owning handoff

epyc-root `handoffs/active/gemma-challenge-kernel-techniques-v7.md`, lines **134-153**. The
parent row `:134` is checked — *"K11 — gemma4 external-head MTP short-smoke determinism ✅
2026-07-17; broad free-form gate reopened 2026-07-18"* — and the eighteen sub-rows `:135-152`
are all checked, several explicitly as **CLOSED NEGATIVE** (K11.1l, K11.1n, K11.1p, K11.1r).

**`:153` is the one unchecked row and this directory is its evidence base**: *"K11.1 —
free-form sampler/stop-condition root cause for broad Gemma4 worker lane"*, whose promotion
criterion is a fresh-server multi-slot worker lane passing a longer task-level determinism
gate *"in the intended natural free-form serving shape, not just a short JSON hash smoke,
repeated-token exact-count prompt, or schema-constrained task."* Nothing here clears it.

The matching progress record is epyc-root `progress/2026-07/2026-07-20.md:1594-1597`, which
states the same verdict independently: *"graph replay and flash-attention are not sufficient
explanations for K11 natural-prose nondeterminism. With the CPU no-spec control deterministic,
remaining work should inspect broader GPU backend numerical nondeterminism or logits
copy/handoff…"*

## Registry claims this backs

`orchestration/model_registry.yaml`. A case-insensitive grep for `k11` returns exactly five
lines. Key paths are the stable reference; line numbers are as of 2026-09-15. Note that
**none of the five cites the determinism result** — the registry draws on this campaign only
for throughput and stop-string observations taken with the same harness.

- **L5293** &nbsp;`roles.worker_general.performance.mi210_ngram_mtp_repetition_observation` →
  `k11_gemma4_mi210_modes_20260719T004400Z/summary.json`. Strict repetitive JSON/tool-shaped
  fixture: no-spec `74.15 t/s`, draft-mtp `113.45 t/s` with `465/474` accepted, and
  `ngram-mod,draft-mtp` `183.50 t/s` with `625/1454` accepted.
- **L5294** &nbsp;`roles.worker_general.performance.mi210_ngram_mtp_prompt_diverse_observation` →
  `k11_gemma4_mi210_prompt_diverse_20260719T004626Z/compact_summary.json`. The follow-up that
  contains the caveat: `ngram-mod,draft-mtp` reproduced the repetitive-output speedup
  (`330.57` vs `143.58` vs `85.77 t/s`) but *"did not improve non-repetitive coding
  explanation speed and failed the required-heading contract … Treat `ngram-mod,draft-mtp` as
  a task-specific token-reuse lane, not the broad worker default."*
- **L9830** &nbsp;`roles.gemma4_26b_a4b_ud_iq4xs_local.performance.current_v7_mi210_stop_string_observation`
  → the two `k11_stop_end_*_concurrent_pc4o_20260720T08*Z/summary.json` runs. UD-IQ4_XS passed
  `10/10` with one hash, exact 200 words, mean decode `113.738 t/s`, `1333/1340` drafts
  accepted; the matched ORIG-Q4 comparison also passed but decoded **faster** (`141.788 t/s`),
  *"so UD remains compression/residency candidate rather than faster worker replacement."*
  The registry marks it `Observation-grade because CPU PC-4o overlapped` — a contended host,
  by its own admission.
- **L9836, L9837** &nbsp;`roles.gemma4_26b_a4b_ud_iq4xs_local.performance.evidence[3]` and
  `[4]` — absolute paths to the same two `summary.json` files.

All five cited artifacts are tracked and resolve. Worth recording because the two `evidence`
entries look suspicious and are not: their directories were added by `22527a43` *"Add PC4o K11
K28 evidence artifacts"*. Both arms in fact produced the **identical** output hash
`42bf7027010486d0b335d5ada36b23ef30cf25c59b72d96fecc14fe47d4d9d21`.

## Provenance gaps — read these before citing anything here

1. **Six `*_20260718Tcodex` directories carry a report and no run data.** Each holds exactly
   one tracked file, a `.md` report or a `plan.json`. Worse,
   `k11_long_stop_condition_20260718Tcodex/k11_long_stop_condition_report.md:19-24` tabulates
   six evidence artifacts (`k11_gemma4_long_mtp_np1_20260718Tcodex`,
   `…_np4_scored_…`, `…_np4_n10_…`, `k11_gemma4_long_nospec_np4_…`, `…_repeat_…`,
   `…_np1_…`) and **none of the six is tracked anywhere in this repo.** The 2026-07-18 Codex
   lane's numbers are therefore **prose-only and not verifiable from this checkout.** This is
   the largest hole in the campaign. The reports' *conclusions* are corroborated by the later
   tracked runs; their *metrics* are not.
2. **The owning handoff has two dangling citations into this directory.**
   `gemma-challenge-kernel-techniques-v7.md:141` (K11.1g) cites
   `k11_schema_word_array_ud_iq4xs_mtp_np4_n10_currentv7_20260719Tmain_1024/summary.json` and
   `…_20260719Tmain/summary.json`. Neither is tracked; the only UD schema directory here is
   the **n=2** `k11_schema_word_array_ud_iq4xs_mtp_np4_n2_currentv7_20260719T000822Z`. The
   same pair is repeated at `progress/2026-07/2026-07-19.md:612,614`. Do not read K11.1g as an
   n=10 result backed by this tree.
3. **One run was deliberately excluded and the exclusion was executed, not just declared.**
   `k11_gpu_backend_path_ab_20260720T1204Z/summary.md:9`: the first pre-sampling attempt
   `…_presampling_trace_20260720T115054Z` *"used the default compact JSON prompt instead of the
   natural 160-word prompt"* and was moved to `/tmp/`. Confirmed: it is not tracked. Good
   hygiene, recorded so nobody re-derives it as missing evidence.
4. **Commit `c20beae5` was made with `--no-verify`**, bypassing the PII pre-commit hook
   (epyc-root `progress/2026-07/2026-07-20.md:1577-1579`), and its three directories
   deliberately omit `logs/` and `responses/`. An independent scan on 2026-09-15 found nothing
   requiring redaction, but the bypass belongs in the record rather than in nobody's memory.
5. **Two incompatible summary schemas coexist.** Most directories use the runner's
   (`deterministic`, `unique_output_hashes` as a *list*, `runs` as a *list of records*);
   `k11_gemma4_long_nospec_cpu_device_none_np4_n10_20260718T143903Z` uses a hand-rolled one-off
   (`runs` as an *integer*, `unique_hashes`, `task_passes`/`task_failures`, `results`). The
   handoff at `:139-140` explains why — the runner had no CPU device surface yet — and records
   the remediation: *"Future K11 CPU controls should use the runner, not hand-written one-off
   harnesses."* **Any parser over this directory must handle both shapes.**
6. **Minor numeric drift between handoff and artifact.** `:148` (K11.1n) reports no-spec+np1
   mean decode `69.86 t/s`; the artifact says `69.435`
   (`k11_natural_freeform_orig_q4_nospec_np1_explicit_greedy_trace_20260720T113634Z/analysis.md:13`).
   Prefer the artifact.
7. **The only in-tree README before this one describes one run out of 38**
   (`k11_gemma4_determinism_20260716T194501Z/README.md`) and its closing caveat — that a
   quiet-host repeat is still needed — was satisfied the next day by
   `k11_gemma4_determinism_20260717Tquiet_glm_done`. It is kept as written; it is a dated
   artifact, not a live index.

## Integrity

`SHA256SUMS` seals all 524 tracked files, including the per-run `README.md` in
`k11_gemma4_determinism_20260716T194501Z/` and the twelve other `.md` reports — those are dated
artifacts of the campaign, not descriptions of it, and a reader needs to know they are the
versions the verdicts were written from. Only this top-level `README.md` is excluded:
documentation is not evidence, and hashing it would make every doc edit break the seal,
following the 2026-08-02 precedent in
`data/paddleocr_vl_receipt_extract_20260717T194415Z/`. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/k11_gemma4_determinism/SHA256SUMS
```

No PII and no credentials: zero email addresses, zero credential-shaped strings, zero
`/home/<user>` paths, and zero non-loopback IP addresses. Every prompt is synthetic. Nothing
here is a binary or a model weight — the census is 356 `.json`, 96 `.log`, 35 `.txt`, 21
`.sh`, 13 `.md` and one each of `.stdout`, `.stderr` and `.exit_code`, of which exactly two
are zero bytes; the dozen largest files are all ~520 KB
`token_traces/run_NN.tokens.json` pre-sampling probability traces, which are the campaign's
most load-bearing evidence.

The files do carry first-party operational detail: `/mnt/raid0/llm/` absolute paths in 415 of
524 files, loopback binds with ephemeral ports in 297, and **338 `server_pid` values** — those
PIDs are not incidental leakage, they are the proof that each run got its own fresh process,
which is the whole topology claim above. `k11_gemma4_mi210_prompt_diverse_20260719T004626Z/compact_summary.json`
additionally embeds the host's `earlyoom` command line in its cleanup proof and its
`kernel_commit` provenance pair
`["ed4091266d286045510e498ceb059c209a65aff9", "experimental-v7-refresh-20260716"]`.
