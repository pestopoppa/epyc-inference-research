# Qwable-v1 reasoning economics — strict-output, task-quality and verifier/selector gates

Provenance documentation added **2026-09-15** (NIB2-73a residual: this campaign directory carried
no `README.md` and no `SHA256SUMS`, so the evidence-durability gate warned on it and a reader could
not tell what was measured or which claim it backs). The measurement files themselves were committed
between 2026-07-17 and 2026-07-19 and are unchanged; nothing was copied or migrated by this commit.

The campaign asks a **cost/quality routing question**, not a speed question: can a GPU-resident
Qwen3.6-35B-A3B/Fable-5 reasoning distillation (Qwable-v1) answer directly, emit strict JSON, and act
as a best-of-N *verifier/selector* for a separate beneficiary model? The tracked artifacts are the
gate bundles for those questions.

| | |
|---|---|
| scratch origin | **none** — the runner wrote directly into this path. Each `summary.json` records its own output paths as `data/qwable_reasoning_economics/<run>/…` (e.g. `qwable_schema_sourcehead_repeat_20260719T000517Z/summary.json:2`), and `qwable_cpu_verifier_standalone_20260719T021216Z/summary.compact.json` records `artifact_dir` as `/mnt/raid0/llm/epyc-inference-research/data/qwable_reasoning_economics/…`. |
| measured (UTC) | 2026-07-17 06:45 → 2026-07-19 02:13. **Source**: the `created_at` field inside each `summary.json` (e.g. `2026-07-19T00:05:23.541834+00:00`), corroborated by the `…T<HHMMSS>Z` suffix in each run-directory name. File mtimes are NOT used — in a fresh worktree they are checkout times (OBS-13). |
| committed | five commits: `852c0dcf` 2026-07-17 "Close Qwable schema smoke gate", `560c4098` 2026-07-17 "Add Qwable task-quality gate", `7bf910ea` 2026-07-17 "Close Qwable expanded routing gate", `89013bcc` 2026-07-18 "Capture inference checkpoint evidence and verifier tooling", `9682f2c9` + `dacb6776` 2026-07-19 "Record v7 model admission checkpoint" / "Record MI210 image and CPU verifier probes" |
| carried (tracked) | **168 files, 973,567 bytes** (0.93 MiB) in 11 run directories |
| also on disk, gitignored | 1,347 further files across 19 more run directories at the same path in the shared clone `/mnt/raid0/llm/epyc-inference-research` — see "Not carried" |
| model under test | `Qwable-v1.IQ4_XS.gguf` (18,939,313,056 B, HF revision `f35ea1502056a2886dd88fb8a29272f8f3c9c3a5`, AGPL-3.0) and a Q8_0 sibling, on experimental-v7 (`llama.cpp-experimental/build-hip/bin/llama-server`, per the `command` field in each `summary.json`) |

## Registry claims this backs

`orchestration/model_registry.yaml`. The YAML key path is the stable reference; line numbers are as
of 2026-09-15. Two roles cite this directory: `roles.qwable_v1_iq4xs` (tier C,
`benchmark_status: expanded_task_quality_pass_iq4_plain_preferred_ngram_neutral`) and
`roles.qwable_v1_q8_0` (tier B,
`benchmark_status: strict_output_and_small_task_quality_slice_pass_routing_open`).

Nine of the 11 tracked run directories are cited. Each figure below carries its unit and its n as
the artifact or the registry states it; where n is not stated, it is marked.

- **L9114, L9115** &nbsp;`roles.qwable_v1_iq4xs.performance.evidence`, described at **L8945**
  `…performance.quiet_host_repeat_observation`; **L9196, L9234** the same run under
  `roles.qwable_v1_q8_0` → `qwable_quality_quiet_20260717T0645Z/` (21 files, 63,733 B).
  Six sequential arms on a quiet host. MI210 `standalone_iq4_gpu` returned valid fenced JSON at
  **prompt 317.96 t/s, generation 99.27 t/s**; `standalone_q8_gpu` valid fenced JSON at
  **prompt 298.08 t/s, generation 103.04 t/s** over **45 prompt / 41 completion tokens**; the CPU
  IQ4 baseline returned strict JSON at **13.82 t/s** decode. n = **1 request per arm** (this is a
  bounded smoke, one call per arm). The registry itself records the caveat: *"The scaffold and
  selector stubs were parseable but semantically placeholder/arbitrary, so they are not deployment
  evidence."*
- **L9116, L9117** &nbsp;`…iq4xs.performance.evidence`, described at **L8963**
  `…performance.schema_mode_gate` → `qwable_schema_fixed_quiet_20260717T0718Z/` (6 files, 32,478 B).
  The corrected top-level `json_schema` arm returned exactly
  `{"arm":"strict_iq4_schema_gpu","quant":"IQ4_XS","role":"reasoner"}` at **prompt 241.73 t/s
  (31 tokens), generation 64.55 t/s (51 completion tokens)**, n = 1 request. The registry's own
  words: *"This is schema acceptance, not task-quality acceptance."*
- **L9118, L9119** &nbsp;`…iq4xs.performance.evidence`; **L9235, L9236** the same two runs under
  `…q8_0.performance.evidence` → `qwable_task_quality_20260717T113232Z/` (18 files, 56,054 B, MI210)
  and `qwable_task_quality_cpu_20260717T113317Z/` (18 files, 56,173 B, CPU). Six deterministic
  tasks, IQ4_XS vs Q8_0. **n = 6 tasks per arm.** MI210: IQ4_XS **6/6** at mean prompt 371.24 /
  mean decode 112.15 t/s; Q8_0 **6/6** at 333.49 / 113.62 t/s. CPU: IQ4_XS **6/6** at 89.78 /
  17.11 t/s; Q8_0 **6/6** at 70.72 / 13.66 t/s. Verdict recorded in the registry: Q8_0 showed **no**
  quality advantage in this slice and was slower on CPU.
- **L9120, L9121, L9122** &nbsp;`…iq4xs.performance.evidence`, described at
  `…performance.expanded_task_quality_observation` → the three `…expanded_final_20260717T1841…`
  runs (22 files each; plain 125,689 B, ngram 126,271 B, cpu 125,862 B). Expanded gate adding
  routing, format, code, architecture, long-context-needle and exact-JSON tasks. **n = 18 tasks per
  lane.** MI210 plain reasoning-off **18/18** at mean prompt 396.21 / mean decode 106.65 t/s;
  MI210 `ngram-mod` **18/18** at 395.31 / 106.66 t/s; CPU plain **18/18** at 90.51 / 15.96 t/s.
  Verdict: `ngram-mod` is **safe but not a speed lever** on this slice (106.66 vs 106.65 t/s).
- **L9123** &nbsp;`…iq4xs.performance.evidence` → `qwable_task_quality_20260718T160400Z/` (2 files,
  79,679 B — `plan.json` + `summary.json` only). A 2026-07-18 current-v7 paired GPU repeat of the
  `default+expanded` task set (`spec_type: none`). **n = 18 tasks per arm.** IQ4_XS **14/18** at mean
  decode 108.38 t/s; Q8_0 **13/18** at 110.44 t/s. The registry's conclusion is negative: it
  *"did not support upgrading to Q8_0"*. The artifact's own `classification` field says
  *"deterministic task-quality slice; compare selected quant/device/spec lanes, but do not promote a
  production role from this alone."*
- **L9124 – L9131** &nbsp;`…iq4xs.performance.evidence`, described at **L9005 – L9036**
  `…performance.verifier_selector_observation` — eight verifier/selector run directories. **These
  are cited but NOT tracked** (`verifier_selector_scaled_no_solve_first_*`,
  `verifier_selector_replay_full96_20260718T203101Z`, `…replay_known_misses_20260718T_main`, etc.
  live on disk, gitignored, at the same path). The combined scaled result the registry records is
  **pass@1 45/96, oracle pass@5 53/96, verifier-selected 49/96, selection accuracy 49/53 (92.45%)**
  over the 96-row CruxEval-output slice, with the verdict *"still positive but observation-grade …
  supports miss analysis and a repeated/decision-grade gate, **not a deployment decision yet**"*.

### Two tracked runs the registry does not cite

- `qwable_schema_sourcehead_repeat_20260719T000517Z/` (12 files, 41,309 B) — a source-head repeat of
  the strict `json_schema` arm. Its `summary.json` records the exact invocation
  (`llama-server … --device ROCm0 -ngl 99 -t 96 -c 8192 -fa on -rea off`, port 19170, seed 42,
  `temperature 0.0`, `top_k 1`), the identical strict object, and **prompt 174.31 t/s over 31 tokens,
  generation 61.59 t/s over 47 completion tokens** with `finish_reason: stop`, n = 1 request.
  Note the decode rate is **lower** than the 64.55 t/s of the 2026-07-17 run it repeats, on a
  different completion-token count (47 vs 51); this is a repeat, not a speedup.
- `qwable_verifier_replay_known_misses_sourcehead_20260718T235425Z/` (5 files, 206,841 B) — a
  verifier-only replay of the two known selector misses with no beneficiary regeneration. Its
  `summary.json` `metrics` block records **n_cases 10, ok_cases 10, selection_accuracy 1.0
  (10/10)**, `known_miss_recovered: [cruxeval_output_0057, cruxeval_output_0081]`, and — the part
  that matters — `order_sensitive_qids: [cruxeval_output_0057, cruxeval_output_0081]`, i.e. **both
  recovered rows are order-sensitive**.

Both are recorded in epyc-root `progress/2026-07/2026-07-19.md:72`: *"Qwable verifier | Replayed
known misses and repeated the schema source-head lane, with committed summaries under
`data/qwable_reasoning_economics/`."*

- `qwable_cpu_verifier_standalone_20260719T021216Z/` (20 files, 59,478 B) is likewise **not cited by
  the registry** but has a recorded verdict in epyc-root
  `handoffs/active/scaffold-autopilot-cost-lever-deployment.md:91`: execute `exit_code=0`, both arms
  `status=ok` on `--device none -ngl 0`, decode **13.9607 / 14.0744 t/s** (n = 1 request per arm,
  35 and 39 completion tokens), `finish_reason=stop`, and post-run ROCm `No KFD PIDs currently
  running`. Its `verifier_selector_stub` arm returned `{"arm":"x86_64","selector":"any",
  "verifier":"default"}` in fenced mode — i.e. parseable but semantically a placeholder, the same
  caveat the registry attaches to the stub arms above.

## What this proved

The routing decision this campaign produced is recorded in epyc-root, not here:

- `handoffs/active/design-backlog-triage-2026-07-23.md:250` — *"Qwable standalone primary route …
  **DECIDED 2026-07-17**: plain reasoning-off standalone (77%) beats scaffold (73%); scaffold is the
  beneficiary-must-answer fallback only."* (The 77%/73% pair is quoted from that row; the
  artifact backing those two percentages is not identified there and is **UNVERIFIED — which run
  directory yields 77%/73% is not stated in the row**.)
- `handoffs/active/scaffold-autopilot-cost-lever-deployment.md:88-91` records the same three gates
  closing: quiet-host server/chat evidence, schema-mode acceptance "for the harness boundary", and
  the expanded standalone-routing quality gate, with *"Treat plain reasoning-off IQ4_XS …"* as the
  routing recommendation.
- The registry's own `routing_recommendation` (under `roles.qwable_v1_iq4xs.performance`):
  *"Prefer Qwable-v1 IQ4_XS as a standalone reasoning-heavy route when it can answer directly. Use
  the plain reasoning-off server lane by default; keep `ngram-mod` available only as a measured
  per-task experiment because it was neutral on the expanded gate."*

**Nothing here is a production admission.** Every cited observation is qualified in the registry as
bounded-smoke, slice, or observation-grade; the verifier/selector line is explicitly
*"not production stack admission"*.

**The model under test no longer exists on disk.** `roles.qwable_v1_iq4xs.artifact_status` records
`download: deleted`, `deleted: 2026-07-26`, reason *"operator-approved disk cleanup; failed all
reviewer slates (RM-2.fast), 0 active-registry refs; recorded results in handoffs survive
deletion"*. A later restore for a different campaign is recorded at
`handoffs/active/gpu-cot-scaffold-sidecar.md:9`. These artifacts are therefore the *only* remaining
first-party record of these runs; they cannot be re-measured from the same weights without
re-fetching the pinned HF revision.

`UNVERIFIED — a handoff row naming this directory as its artifact root.` No handoff row does.
The nearest owners are `handoffs/active/scaffold-autopilot-cost-lever-deployment.md` (deployment
design, cites four of these runs) and `handoffs/active/gpu-cot-scaffold-sidecar.md` (the original
study scope, which does **not** reference `data/qwable_reasoning_economics` at all).

## Not carried

1,347 files across 19 further run directories remain on disk, gitignored, at
`/mnt/raid0/llm/epyc-inference-research/data/qwable_reasoning_economics/` — including every
`verifier_selector_*` bundle the registry cites at L9124–L9131, the
`codex_verifier_selector_followup_20260718T1918Z/` per-candidate response trees, and the
`qwable_gpu_verifier_selector_iq4_expanded_*` runs. Their citations resolve as `OK` on this host
(`durable_untracked`), per `scripts/validate/check_evidence_durability.py`. Nothing was deleted or
moved; this commit adds documentation only and does not change what is tracked.

The tracked content includes CruxEval prompts and model completions — a public code-reasoning
benchmark, not third-party personal data. No PII and no credential-shaped strings were found
(scanned: zero email addresses, zero credential assignments). The files do carry first-party
operational detail: local absolute paths under `/mnt/raid0/llm/`, local ports, and OS PIDs
(`server_pid`) in the `summary.json` files.

## Integrity

`SHA256SUMS` seals all 168 tracked files. `README.md` and `SHA256SUMS` are deliberately not in it
(documentation is not evidence, and hashing the README would make every doc edit break the seal),
following the 2026-08-02 precedent in `data/paddleocr_vl_receipt_extract_20260717T194415Z/`.
Verify from the repository root:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/qwable_reasoning_economics/SHA256SUMS
```
