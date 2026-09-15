# GLM-5.2 UD-IQ2_M — native MTP A/B and the NEXTN row-selection repair

Provenance documentation written **2026-09-15** (NIB2-73a). The tree itself was committed
on **2026-07-19** and has never been touched since: a single commit, `8fd06369` *"Record
GLM native MTP repair evidence"*, added all 15 files and no later commit modifies any of
them (`git log --follow -- data/glm52_native_mtp_ab`). The registry cites this campaign
from three places, so it needed a README and a `SHA256SUMS` to satisfy the
evidence-durability gate; nothing was moved, copied or re-measured to produce them.

Two provenance details a reader will not find by grepping commit subjects. The write-up was
published **before** its evidence: `a98b01f5` *"Record GLM native MTP A/B failure"*
(19:13:07) added the companion doc alone, no data files, and `8fd06369` sixty-one minutes
later added all 15 files **and rewrote that doc's verdict from failure to repair**. And the
registry citations did not arrive with the data: they were added 26 minutes afterwards by
`171be783` (20:40:41), whose subject is *"Add DR-0 self-spec accounting scaffold"* and names
neither GLM nor native MTP.

> **The subject model no longer exists.** GLM-5.2 UD-IQ2_M was ruled **KILL** by operator
> ruling **OP-8** and its 223 GB artifact deleted; the owning handoff
> `handoffs/active/glm52-reviewer-capability-gates.md:3-6` was retargeted to GLM-5.3-Flash on
> 2026-09-01. That handoff's own instruction (`:16-22`) is to *"treat as historical evidence
> about a deleted model, never as carry-forward state"* every GLM-5.2 measurement and serving
> constant, the next-power-of-two `indexer_top_k` schedule included. **This directory is a
> record of what was measured, not a source of constants to inherit.** No open backlog row
> anywhere cites it; every gate it fed is checked off.

| | |
|---|---|
| scratch origin | none — committed straight into the repo by `8fd06369` |
| measured (UTC) | 2026-07-19 18:58:37 → 19:52:56 |
| dates taken from | `plan.json` `generated_at` fields (18:58:37.268, 19:04:28.229, 19:50:37.385) and the streaming timestamps inside the response artifacts — **not** file mtimes, which in a fresh worktree are checkout times |
| committed | 2026-07-19 (`8fd06369`) |
| documented | 2026-09-15 |
| carried | **15 files, 359,969 bytes** — the whole campaign, nothing withheld |
| model | `GLM-5.2-UD-IQ2_M-00001-of-00006.gguf` (6 shards, `general.architecture = glm-dsa`, 79 blocks, 256 experts / 8 used, quantized_by Unsloth) |
| binary | `/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server`, **`build 10098 (6ad45fa3f)`** in all three runs — see *What the binary does not prove* |

## What was measured

One probe, `long_context_dsa_probe`, from `scripts/benchmark/glm52_dsa_probe_runner.py`:
a 15,860-character synthetic prompt asking the model to emit `READY` and then the word
`tokenstream` 620 times, streamed through `/v1/chat/completions` at `temperature 0.0`,
`seed 42`, `max_tokens 512`, `min_completion_tokens 384`, `-c 4096`,
`glm-dsa.attention.indexer.top_k` overridden to `4096`. CPU-only: `--device none -ngl 0
-t 96 -ub 512`, `OMP_NUM_THREADS=1`, `numactl --interleave=all`, `env -i`. The prompt is
generated filler, not anybody's document.

The three arms are flag-identical apart from `--spec-type` and the port
(`plan.json` → `stages[].server.server_command`), so this is a matched A/B and not a
comparison across configurations. **n = 1 per arm.** Nothing here was repeated.

## The A/B — both arms named, and why the first pair settled nothing

| arm | directory | status | prompt tok (client / server) | prompt t/s | eval tok | decode t/s |
|---|---|---|---|---|---|---|
| `--spec-type none` (control) | `glm52-native-mtp-ab-20260719T185837Z/nospec/` | `ok` | 2919 / 2931 | 22.56 | 512 | **2.49** |
| `--spec-type draft-mtp` | `glm52-native-mtp-ab-20260719T185837Z/draft_mtp/` | **failed** — `completion_token_min_passed: false` | 2919 / — | — | **0** | — |
| `--spec-type draft-mtp`, after the repair | `glm52-native-mtp-draft-long-repair-20260719T195037Z/` | `ok` | 2919 / 2931 | 22.77 | 512 | **5.33** |

Numbers come from `plan.json` → `execution.stages[<long_context_dsa_probe>]`: the client
count is `prompt_token_count`, the server count and both rates are
`server_log.prompt_eval_tokens` / `prompt_eval_tps` / `decode_tps`. The two token counts
differ by 12 in every arm — 2919 is the runner's own count, 2931 is llama.cpp's
`prompt eval` line and the server's `usage.prompt_tokens`. **The prompt is byte-identical in
all three arms** (`md5sum` of the three `long_context_dsa_probe_4096_prompt.txt` files agree),
so the two numbers are two meters on one prompt, not two prompt shapes. The companion doc's
result table mixes the sources — 2931 for the first two rows, 2919 for the third — which
makes the repaired arm look as though it ran a slightly different prompt, and that reading
propagated into the registry (`prompt_tokens: 2919` at L8115) and into the handoff prose.
Nothing downstream is wrong by more than 12 tokens; it is a labelling artifact, recorded here
so the next reader does not chase it.

**The original A/B is not a completed comparison.** The `draft-mtp` arm returned nothing:
`draft_mtp/artifacts/long_context_dsa_probe_4096_response.json` is 411 bytes of
`"content": ""`, `"chunk_count": 0`, `"first_chunk_at": null`, empty `timings` and empty
`usage`, and its `server_log` block records `decode_tokens: null`,
`prompt_eval_tokens: null`. The server had reached the intended path — the log shows
`common_speculative_init_result: creating MTP draft context`,
`common_specu: adding speculative implementation 'draft-mtp'`, `n_max=3, n_min=0,
p_min=0.00, n_embd=6144, backend_sampling=1` and `speculative decoding context
initialized` — and then produced no decode checkpoints at all. So the 18:58 pair reads as
*control served, treatment crashed*, which is a defect report, not a measured A/B.

**The comparison that does resolve** is the control against the 19:50 matched retry:
**5.33 vs 2.49 t/s, a 2.14x decode win for native draft-MTP**, with draft acceptance
`0.93300 (376 accepted / 403 generated)`, mean accepted length `3.79`, per-position
acceptance `(0.970, 0.926, 0.889)`
(`glm52-native-mtp-draft-long-repair-20260719T195037Z/logs/long_context_dsa_probe.server.log:358`
and `:360`). Prefill is unchanged (22.56 → 22.77 t/s), as expected.

## The recorded verdict

[docs/data/glm52_native_mtp_ab_20260719.md](../../docs/data/glm52_native_mtp_ab_20260719.md) is the campaign's own write-up and is itself
cited by the registry. Its verdict, verbatim:

- `:36-39` — *"The repaired arm is a `2.14x` decode-speed improvement over the no-spec
  baseline on this long-output serving shape (`5.33 / 2.49`). This is an
  acceleration/serving result only: the prompt intentionally asked for a long repeated
  sequence, and the GLM output still entered reasoning text, so this row does not change
  the separate reviewer-quality verdict."*
- `:80-82` — *"Native GLM-MTP is now functionally repaired on the matched long-context
  serving gate and shows a measured decode-speed win on this long repeated-output shape.
  It is no longer correct to describe B6 as a zero-chunk serving failure."*
- `:84-87` — *"This does **not** admit GLM as the production patch reviewer. The
  decision-grade C-CRAB P-REV-1 reviewer gate remains failed…"*

Quality is genuinely not attested here. `content_preview` in the two successful arms
begins *"I need to output \"READY\" first, then output the word \"tokenstream\" repeated
exactly 620 times…"* — the model emitted its own reasoning text, `finish_reason` is
`length`, and `expected_substring_passed` is `null` in every arm because no content
assertion was configured. **Speed only.**

## What the binary does not prove

`docs/data/glm52_native_mtp_ab_20260719.md:55-59` attributes the repair to a source change
that "changed the shared DeepSeek32/GLM-DSA main graph to preserve full token rows for
`res->t_h_nextn`". That change is not visible in this evidence. All three server logs
report the same `build 10098 (6ad45fa3f)`, none carries a dirty marker, and `plan.json`
records no binary digest — so nothing in this directory distinguishes the binary that
failed from the binary that succeeded.

**UNVERIFIED — no artifact in this campaign identifies the repaired binary.** The repair
is credible from the log signature (the failed arm initialised the MTP context and then
emitted zero chunks; the retry with byte-identical flags emitted 515) and from the source
hardening claimed at `:67-76` (`test-llama-archs --arch glm-dsa` / `--arch deepseek32`),
but the build-identity evidence that would close it was not captured. Do not cite this
directory as proof of *which* commit fixed it.

**And the repair run was not a single-variable change.** `diff -u` of the failed arm's
`request.json` against the repair's shows two changes, not one: the port, and a new
`"sse_ping_interval": 5` in the request payload. The failure being diagnosed *was a
zero-chunk SSE stream*, and the co-change *is an SSE keep-alive knob* — added to
`scripts/benchmark/glm52_dsa_probe_runner.py` in the same commit that landed this evidence.
The companion doc attributes the repair entirely to the NEXTN graph fix and does not mention
the payload change at all. This is not a refutation of the graph fix; it is a statement that
**these artifacts cannot separate the two causes.** Anyone re-opening this should vary one at
a time.

## Registry claims this backs

`orchestration/model_registry.yaml`, all under `roles.glm_52_ud_iq2m.performance`.
Key paths are the stable reference; line numbers are as of 2026-09-15.

- **L8026-8034** &nbsp;`native_mtp_repair_observation`
  > 2026-07-19 DeepSeek32/GLM-DSA NEXTN row-selection repair made the matched
  > long-context draft-mtp retry stream successfully. Evidence:
  > data/glm52_native_mtp_ab/glm52-native-mtp-draft-long-repair-20260719T195037Z/plan.json.
  > Repaired draft-mtp processed 2919 prompt tokens at 22.77 t/s and decoded 512 tokens at
  > 5.33 t/s with alpha=0.933 (376/403 accepted), versus matched no-spec decode 2.49 t/s.
  > This is acceleration evidence for the single-NextN repaired path only; it does not
  > reverse the P-REV-1 reviewer-admission failure.
- **L8111-8125** &nbsp;`measured[1]` — the structured row whose `protocol` is
  `native_mtp_repair_observation`, `role: acceleration`, `path: single_nextn_draft_mtp`,
  `prompt_tps: 22.77`, `decode_tps: 5.33`, `no_spec_decode_tps: 2.49`,
  `draft_generated_tokens: 403`, `draft_accepted_tokens: 376`, `alpha: 0.933`,
  `role_ready: false`, and `limitation: single-NextN repaired path; acceleration evidence
  only, not reviewer admission`. Its `evidence` (L8125) is the repair `plan.json`.
- **L8253-8255** &nbsp;`evidence.native_mtp_repair` — the repair `plan.json` and
  [docs/data/glm52_native_mtp_ab_20260719.md](../../docs/data/glm52_native_mtp_ab_20260719.md).

Note what the registry cites and what it does not. Every citation points at the **repair**
run's `plan.json`; **no registry line cites the `20260719T185837Z` A/B pair at all.** The
control that makes 5.33 t/s a 2.14x number (`nospec/plan.json`, `no_spec_decode_tps: 2.49`)
and the zero-chunk failure that motivated the repair are therefore uncited but
load-bearing — they are carried for that reason, the same reasoning as the CPU control in
`data/gemma4_iq4_residency/`. This campaign has **no `summary.json`**; `plan.json` is the
result file, because the runner writes measurements back into the plan's `execution` block.

## Integrity

`SHA256SUMS` seals all 15 tracked files. `README.md` is deliberately not in it —
documentation is not evidence, and hashing it would make every doc edit break the seal —
following the 2026-08-02 precedent in
`data/paddleocr_vl_receipt_extract_20260717T194415Z/`. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/glm52_native_mtp_ab/SHA256SUMS
```

No PII and no credentials: the prompt and response bodies are synthetic
`tokenstream`-repetition filler and the model's own reasoning about it. The files do carry
first-party operational detail — local absolute paths under `/mnt/raid0/llm/`, the
experimental build path and commit `6ad45fa3f`, listening ports, and the model's HuggingFace
cache manifest path.
