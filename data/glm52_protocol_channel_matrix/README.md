# GLM-5.2 UD-IQ2_M — reviewer-serving protocol/channel matrix (gate GC-0d)

Provenance documentation written **2026-09-15** (NIB2-73a). The tree was committed on
**2026-07-18** by a single commit, `0ba62b8a` *"Record GLM chat matrix and Nemotron protocol
gates"*, which added all 14 files together with the runner that produced them; no later
commit modifies any of them (`git log --follow -- data/glm52_protocol_channel_matrix`). The
registry cites this campaign from two places, so it needed a README and a `SHA256SUMS` to
satisfy the evidence-durability gate; nothing was moved, copied or re-measured to produce
them.

> **The subject model no longer exists, and these prompt bands explicitly do not carry
> forward.** GLM-5.2 UD-IQ2_M was ruled **KILL** by operator ruling **OP-8** and its 223 GB
> artifact deleted; the owning handoff
> `handoffs/active/glm52-reviewer-capability-gates.md:3-6` was retargeted to GLM-5.3-Flash on
> 2026-09-01. That handoff names this campaign among the things that must **not** transfer
> (`:16-22`): *"treat as historical evidence about a deleted model, never as carry-forward
> state … every GLM-5.2 serving constant — the next-power-of-two `indexer_top_k` schedule
> (`2048`/`4096`/`16384`) … and **the protocol/channel matrix's prompt bands**. GLM-5.3-Flash
> is arch `glm5next` (288x10B), not `glm-dsa` … those constants must be re-derived, not
> inherited."* Read the numbers below as a record of one afternoon on one deleted artifact.

| | |
|---|---|
| scratch origin | none — committed straight into the repo by `0ba62b8a` |
| measured (UTC) | **2026-07-18**, plan generated 01:11:16, `execution.elapsed_s: 1756.929` (≈29 min) |
| dates taken from | `plan.json`/`summary.json` `generated_at` (`2026-07-18T01:11:16.287976+00:00`) and the directory-name stamp `20260718T0120Z`. The two differ by ~9 min: the name is the runner's output-directory argument, chosen before the plan was written. **Not** file mtimes, which in a fresh worktree are checkout times |
| committed | 2026-07-18 01:52:50 (`0ba62b8a`) |
| documented | 2026-09-15 |
| carried | **14 files, 378,520 bytes** — every file the run emitted, nothing withheld |
| model | `GLM-5.2-UD-IQ2_M-00001-of-00006.gguf`, 6 shards, `total_shard_gib: 222.19`, HF tree manifest `complete` (`plan.json` → `inventory`) |
| binary | `/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server`, invoked CPU-only (`--device none -ngl 0`) |
| kernel build | **UNVERIFIED from this campaign's own artifacts** — see *No server logs* below |

## What was measured

`scripts/benchmark/glm52_protocol_channel_matrix_runner.py`, invoked exactly as recorded in
`summary.json` → `preexisting_processes`:

```
python3 scripts/benchmark/glm52_protocol_channel_matrix_runner.py --execute \
  --bands p2168_tk4096,p12000_tk16384 --modes free_reasoning_off,json_reasoning_off \
  --endpoints chat --trace-logs --metrics --request-timeout 3600 \
  --output-dir data/glm52_protocol_channel_matrix/glm52-gc0d-chat-p2168-p12000-20260718T0120Z
```

Four cells, one request each, on the `/v1/chat/completions` channel only, at
`temperature 0.0`, `seed 42`, `max_tokens 64`. Server shape per cell: `env -i`,
`OMP_NUM_THREADS=1`, `numactl --interleave=all`, `--device none -ngl 0 -t 96 -ub 512`,
`--reasoning-format deepseek --reasoning off --reasoning-budget 0`, and
`--override-kv glm-dsa.attention.indexer.top_k` set per band.

The matrix is **prompt band × response format**, and both axes are more than their names
suggest:

- **Band** co-varies three things, not just prompt length. `p2168_tk4096` =
  `min_prompt_tokens 2168`, `context_length 4096`, `indexer_top_k 4096`,
  `prompt_context_guard_tokens 128`; `p12000_tk16384` = `12000`, `16384`, `16384`, `512`
  (`plan.json` → `cells[].band`). The `tk` in the name is the DSA `indexer_top_k` cap, not a
  token count — this is the "next power-of-two top-k schedule" under test.
- **Format** is a server-side constraint, not just a differently worded prompt.
  `json_reasoning_off` adds `--json-schema
  {"type":"object","additionalProperties":false,"required":["decision"],"properties":{"decision":{"enum":["allow"]}}}`
  and switches the validator from `exact_ready` to `json_decision_allow`.

Prompts are synthetic: one sentence — *"The GLM DSA probe keeps the context deterministic
while the runner checks shard integrity, load behavior, and KV scaling under a fixed indexer
configuration."* — repeated to the band's token floor, then a `--- TASK ---` block asking for
`READY` or `{"decision":"allow"}` and nothing else. No document, no third-party corpus.

**n = 1 per cell.** Nothing here was repeated.

## Results — four cells, four passes

| cell | band / format | prompt tok | prompt t/s | eval tok | decode t/s | validator | passed |
|---|---|---|---|---|---|---|---|
| `execution.cells[0]` | p2168_tk4096 / free | 2894 | **24.709** | 2 | 4.640 | `exact_ready` | **true** |
| `execution.cells[1]` | p2168_tk4096 / json | 2898 | **24.609** | 6 | 2.766 | `json_decision_allow` | **true** |
| `execution.cells[2]` | p12000_tk16384 / free | 12044 | **16.683** | 2 | 3.452 | `exact_ready` | **true** |
| `execution.cells[3]` | p12000_tk16384 / json | 12045 | **16.416** | 6 | 2.050 | `json_decision_allow` | **true** |

jq paths, per row: `.execution.cells[N].endpoints[0].usage.prompt_tokens`,
`.timings.prompt_per_second`, `.usage.completion_tokens`, `.timings.predicted_per_second`,
`.validation.validator`, `.validation.passed`. `.execution.status` is `"ok"` and
`.refusal_reasons` is `[]`.

**Read the prefill column, not the decode column.** Completions are 2 and 6 tokens long, so
`predicted_per_second` is dominated by first-token latency and is not a decode-throughput
measurement of anything. The registry quotes only the prefill rates, correctly. The
`p2168` → `p12000` prefill fall (24.7 → 16.7 t/s) is the substantive rate observation.

**What the four cells actually establish** is a *channel* property, and it is visible in a
field the summary table above does not show: in all four cells
`.endpoints[0].channels.content` holds the answer and
`.endpoints[0].channels.reasoning_content` is **empty**. With `--reasoning off
--reasoning-budget 0`, GLM put its answer where a client reads it. That is the finding — and
its significance is sharpened by the fact that the *same commit* recorded Nemotron-Nano runs
where the answer landed in `reasoning_content` with `content` empty.

Also note `.execution.cells[*].prompt_token_count` (2888 / 2892 / 12038 / 12039) sits a few
tokens below the server's `usage.prompt_tokens`; the runner and llama.cpp count differently.
Every downstream document here quotes the **server** numbers, which is the right choice.

## The recorded verdict

There is **no `verdict` or `disposition` key in `summary.json`** — the words do not appear in
it. Every recorded verdict is prose, in four places, and all four agree:

- epyc-root `progress/2026-07/2026-07-18.md:63` — *"Disposition: GC-0d is closed for the
  reviewer-serving chat/free+schema channel. Raw `/completion` and `/v1/completions` remain
  unvalidated because the first all-endpoint attempt hit raw endpoint cost/pathology and was
  stopped."*
- epyc-root `handoffs/active/glm52-reviewer-capability-gates.md:144` — the gate row, `[x] …
  ✅ 2026-07-18`, adding that the raw endpoints *"must be probed only narrowly if a future
  route needs them"*; and `:202`, the dependency graph: *"→ GC-0d protocol-channel matrix ✅
  (chat/free+schema reviewer-serving channel; raw endpoints unvalidated)"*.
- This repo, `docs/reference/models/model-admission-2026-07-16.md:211` — *"Disposition:
  GC-0d is closed for the chat/free+schema reviewer-serving channel at the tested prompt
  bands. This is not broad GLM quality."* Same file, `:173`, is blunter: the matrix
  *"validates only the narrow chat/free+schema reviewer-serving lane at the tested prompt
  bands, not broad GLM quality."*
- Same repo, `docs/reference/models/model-smoke-queue-2026-07-16.md:163`, item 16.

`GC-0d` is the fourth sub-gate of the `GC-0` series in the owning handoff (`:140` GC-0
evidence hygiene, `:141` GC-0a 32K needle/coherence, `:142` GC-0b top-k cap diagnosis, `:143`
GC-0c the top-k schedule, `:144` GC-0d this matrix); the lowercase `gc0d` in the directory
name is that gate id. **UNVERIFIED — what the letters "GC" abbreviate is not written down
anywhere in either repo.** Do not expand it.

So: a **narrow channel gate passed**. Two prompt bands, one endpoint, two response formats,
one request each. It is not a quality result, not a throughput result, and per the handoff's
own scope note it never became one.

## No server logs — and what that costs

All four cells record `.execution.cells[N].server_log.status = "missing"`. The plan asked for
them (`--trace-logs`, `trace_logs: true`, and a `log_file` path per cell), but no `logs/`
directory was ever written and none is tracked. Consequences, stated plainly:

- **The kernel build cannot be established from this campaign's own evidence.** The only
  attribution is prose: *"current-source experimental v7 (server version d1e5a20eb)"* in the
  registry citation at L7959, repeated at
  `docs/reference/models/model-admission-2026-07-16.md:202`. `d1e5a20eb` is corroborated as
  the 2026-07-18-era experimental v7 by sibling campaigns of that date (for example
  `data/gemma4_iq4_residency/README.md`), but not by anything in here.
- Nothing corroborates the timings independently. The `timings` blocks come from the server's
  own response objects, which is a sound source, but there is no second meter.

The sibling campaign `data/glm52_native_mtp_ab/` has the opposite shape — full trace logs,
no `summary.json`. The two ran **different builds a day apart** (`6ad45fa3f` / build 10098
there, `d1e5a20eb` asserted here), so **do not cross-compare their throughput numbers.**

## Registry claims this backs

`orchestration/model_registry.yaml`, both under `roles.glm_52_ud_iq2m.performance`. A
repo-wide grep for `glm52_protocol_channel_matrix` returns exactly these two lines. Key paths
are the stable reference; line numbers are as of 2026-09-15.

- **L7957-7970** &nbsp;`protocol_channel_matrix_observation` — the citation is at L7961:
  > 2026-07-18 chat-only protocol/channel matrix passed on current-source experimental v7
  > (server version d1e5a20eb) with CPU-only GLM and the next-power-of-two top-k schedule.
  > Evidence:
  > data/glm52_protocol_channel_matrix/glm52-gc0d-chat-p2168-p12000-20260718T0120Z/summary.json.
  > Free-text chat returned exact READY at 2894 prompt tokens with indexer_top_k=4096
  > (prompt 24.71 t/s) and at 12044 prompt tokens with indexer_top_k=16384 (prompt
  > 16.68 t/s). JSON-schema chat returned exact {"decision":"allow"} at 2898 prompt tokens
  > (prompt 24.61 t/s) and at 12045 prompt tokens (prompt 16.42 t/s). This closes the
  > reviewer-serving chat/free+schema channel gate for these bands, not broad task quality.
  > The first all-endpoint attempt was aborted after raw completion endpoints proved too
  > costly/pathological for this model; raw /completion and /v1/completions remain
  > unvalidated.

  Every figure in that paragraph reconciles to the table above, to the digit.
- **L8234-8235** &nbsp;`protocol_channel_matrix` — a one-item evidence list holding the
  absolute path to the same `summary.json`. Note the registry shape: under `performance:`,
  `evidence:` (L8188) is a **sibling** of `protocol_channel_matrix:` (L8234) and the other
  per-gate evidence lists, not their parent.

The role also carries `constraints.forbid: [production_stack_registration,
production_role_claim_without_dsa_quality_gate]`, whose stated reason (L7800) leans on this
campaign — *"Chat free-text and JSON-schema channels pass at ~2.9K/~12.0K … but
decision-grade C-CRAB P-REV-1 patch-review admission failed"* — and concludes that GLM was
rejected as production patch reviewer. **Nothing here ever licensed a production role**, and
the model is now deleted.

## Integrity

`SHA256SUMS` seals all 14 tracked files. `README.md` is deliberately not in it —
documentation is not evidence, and hashing it would make every doc edit break the seal —
following the 2026-08-02 precedent in
`data/paddleocr_vl_receipt_extract_20260717T194415Z/`. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/glm52_protocol_channel_matrix/SHA256SUMS
```

No PII and no credentials: zero email addresses, zero credential-shaped strings, no external
URLs at all, and `127.0.0.1` is the only address that appears. Prompts and completions are
synthetic. The files do carry first-party operational detail — local absolute paths under
`/mnt/raid0/llm/`, loopback ports 19420-19423, the host's thread count, and, in
`summary.json`, two live process entries with their PIDs and full argv (the runner itself and
the host's `earlyoom` guard).
