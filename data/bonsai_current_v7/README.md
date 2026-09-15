# Bonsai / Ternary-Bonsai sub-2-bit probes on experimental-v7 — MI210 speed rows

Provenance documentation added **2026-09-15** (NIB2-73a residual: this campaign directory carried
no `README.md` and no `SHA256SUMS`, so the evidence-durability gate warned on it and a reader could
not tell what was measured or which claim it backs). The measurement files themselves were
committed in July 2026 and are unchanged; nothing was copied or migrated by this commit.

**This directory is NEGATIVE / non-load-bearing-for-quality evidence.** Every tracked artifact in
it is a *speed-only* llama-bench row (one of them a failed run). None of it establishes model
quality, and none of it licenses a role. Read the "What this proved" section before quoting a
number from it.

| | |
|---|---|
| scratch origin | **none** — the runners wrote directly into this path. Each `summary.json` records its own `artifact_dir` as `/mnt/raid0/llm/epyc-inference-research/data/bonsai_current_v7/<run>` (e.g. `bonsai27_q1_mi210_llama_bench_20260718T150243Z/summary.json:2`). |
| measured (UTC) | 2026-07-18 15:02 – 15:07 and 2026-07-19 05:26 – 05:30. **Source**: the `…T<HHMMSS>Z` suffix in each run-directory name, corroborated for the 8B run by its own `start_utc.txt` (`2026-07-19T05:26:24Z`) and `end_utc.txt` (`2026-07-19T05:30:15Z`). File mtimes are NOT used — in a fresh worktree they are checkout times (OBS-13). |
| committed | `89013bcc` 2026-07-18 "Capture inference checkpoint evidence and verifier tooling" (the two 27B runs) and `71a853ac` 2026-07-19 "Record Bonsai-8B current-build MI210 row" (the 8B run). |
| carried (tracked) | **17 files, 22,244 bytes**, in 3 run directories |
| also on disk, untracked | 158 files (7 further run directories + `ternary_bonsai_q2_layout_contract_20260718Tcodex.json`, 13,752 B) at the same path in the shared clone `/mnt/raid0/llm/epyc-inference-research` — see "Not carried" below |
| kernels | experimental-v7 `d1e5a20eb` (build 10088) for the 27B Q1_0 row; `6a8dd5ea6` (build 10097) for the 8B row; **`null`** in the ternary Q2_g64 artifact, which never reached a build line |

## Registry claims this backs

`orchestration/model_registry.yaml`. The YAML key path is the stable reference; line numbers are as
of 2026-09-15. Four different roles cite this directory.

- **L8640** &nbsp;`roles.bonsai_27b_q1_0.performance.evidence` →
  `bonsai27_q1_mi210_llama_bench_20260718T150243Z/summary.json` (1 file, 921 B). MI210 llama-bench,
  `qwen35 27B Q1_0`, 3.53 GiB / 26.90 B params, ROCm backend, `ngl 99`, `fa 1`, `dev ROCm0`,
  `exit_code 0`: `pp2048 798.59 ± 0.40 t/s`, `pp8192 759.19 ± 1.72 t/s`, `tg1024 11.24 ± 0.00 t/s`
  (n given by the artifact only as the bench's own `± stddev`; the run command `-r 2` is recorded in
  epyc-root, not in this file). The role's `benchmark_status` is
  `cpu_mi210_load_smoke_pass_quality_gate_partial_fail` and `constraints.forbid` carries
  `production_stack_registration_without_quality_gate`.
- **L8833** &nbsp;`roles.ternary_bonsai_27b_q2_g64.performance.evidence`, described at **L8780**
  `…performance.current_v7_llama_bench_observation` →
  `ternary_bonsai_q2_g64_mi210_llama_bench_20260718T150707Z/summary.json` (1 file, 436 B).
  **This is a FAILED run, and the registry says so**: `exit_code 125`, `build: null`, and exactly
  one row — `pp2048 25.68 ± 0.01 t/s` — with no decode row. The registry's wording is
  *"emitted only `pp2048 25.68 t/s`, then remained CPU-bound with `0%` GPU and no decode row until
  manual termination. Treat as a bench-path or acceleration-path failure and partial speed
  observation, not clean decode evidence."* Role `benchmark_status`:
  `experimental_v7_quality_throughput_partial_not_role_ready`.
- **L8887, L8888** &nbsp;`roles.bonsai_8b_local_orphan.performance.evidence`, described at
  `…performance.current_v7_mi210_context_observation` →
  `bonsai8b_mi210_context_6a8dd5ea68_20260719T052624Z/summary.json` plus the sibling
  `bonsai8b_mi210_context_6a8dd5ea68_20260719T052624Z/summary.md` (15 files,
  20,887 B for the whole run directory). MI210, `qwen3 8B Q1_0`, q8_0 KV, flash attention, `ngl 99`,
  prompt-only `pp512 2414.00 t/s` (σ 62.88), `pp4096 2062.67` (σ 2.29), `pp16384 1284.44` (σ 2.14),
  decode-only `tg1024 36.74 t/s` (σ 0.09). Role tier **D**, `benchmark_status`
  `cpu_mi210_load_smoke_pass_observation_orphan`; `constraints.forbid` carries
  `production_stack_registration_without_provenance` — the GGUF has no HF sidecar, so this row
  refreshes throughput hypotheses for a **catalogue-only** artifact.
- **L8715** &nbsp;`roles.ternary_bonsai_27b_q2_0.performance.evidence` →
  `ternary_bonsai_q2_layout_contract_20260718Tcodex.json`. **Cited but NOT tracked** (it is on disk,
  untracked, 13,752 B). It is the raw-GGUF layout verifier behind the Q2_0 load failure.
- **L8626, L8627, L8641, L8642** cite two further run directories in this campaign
  (`bonsai27_q1_cpu_prompt_repair_20260718T191220Z/`,
  `bonsai27_q1_cpu_prompt_control_n6_20260718T200059Z/`) that are likewise **on disk but untracked**.
  Both are CPU prompt-repair probes; the registry records the control as *"direct prompts 0/6,
  repaired prompts 1/6, overall 1/12 … This weakens the prompt-repair path"*.

## What this proved — and what it did not

The recorded verdicts are in epyc-root, not here. Cited by file:line as of 2026-09-15:

- `handoffs/active/tq3-quantization-evaluation.md:196` — the 27B Q1_0 row: *"confirms the local 27B
  Q1 path remains decode-slow on MI210 and is **speed-only evidence**; the 6/8 quality gate still
  blocks role readiness."*
- `…:197` — the ternary Q2_g64 attempt: *"Treat as a current bench-path/acceleration failure and
  partial speed observation, **not quality evidence**."*
- `…:195` — the 8B row: *"remains speed-only and **does not make Bonsai role-ready**."*
- `…:199` — the Q2_0 layout verifier (the untracked JSON above): Q2_0 has `498/498` tensors short
  under standard 18-byte/block `Q2_0`; sibling Q2_g64 has `0/498` mismatches.

**Whether this directory is the evidence behind the sub-2-bit rejection — verified, with a
correction.** `handoffs/active/tq3-quantization-evaluation.md:8-16` carries the standing
**⚠ STEERING (2026-07-19) — deprioritize the sub-2-bit breadth probes**, which names four findings:
Q1_0 6/8 instruction-format, Q2_g64 6/8 + empty `<think>`, Q2_0 won't load (498/498 tensors short),
and *"Do NOT keep running speed reruns on these."*

Only ONE of those four comes from this directory: the Q2_0 layout finding, and that artifact is
the untracked JSON at L8715. The two **quality** failures came from sibling campaign directories
(`data/bonsai_q1_quality_gate/bonsai_q1_quality_clean_20260717T0755Z/summary.json` and
`data/ternary_q2_g64_quality_gate/…`), not from here. What this directory contributes to the
rejection is the *fourth* clause: it is the set of speed reruns that were run on a refreshed v7
binary and changed nothing, which is why the steering forbids more of them. That is negative
evidence and it is load-bearing as such — it is the reason the track is closed to speed churn — but
it is **not** the quality verdict, and quoting `2414 t/s` or `798 t/s` from here as a Bonsai result
misrepresents what was measured. `…:200` keeps the whole family a parked operator-review candidate,
reopenable "only on a named prompt/template/protocol fix", and notes the public quality claim is
"self-reported **and independently contested**".

`UNVERIFIED — n (repetition count) for the two 27B rows.` Those two artifacts are a bare
`summary.json` with no command record; they carry only the bench's own `± stddev`. The `-r 2` figure
appears in epyc-root `handoffs/active/tq3-quantization-evaluation.md:196` and `:197` — prose about
the run, not the run bundle.

The 8B row is the one run here with its own invocation on record, in its tracked `command.sh`:
`llama-bench -m /mnt/raid0/llm/models/Bonsai-8B.gguf -dev ROCm0 -ngl 99 -fa on -t 8 -b 512 -ub 512
-ctk q8_0 -ctv q8_0 -p 512,4096,16384 -n 1024 -o json`, with `GGML_HIP_GRAPHS=1`,
`ROCR_VISIBLE_DEVICES=0`, `HIP_VISIBLE_DEVICES=0`, and `LD_LIBRARY_PATH` pinned to
`llama.cpp-experimental/build-hip/bin`. It passes **no `-r`**, so llama-bench's own default
repetition count applied; the artifact never states that number, so `n` for the 8B row is
`UNVERIFIED — not recorded in the bundle` rather than inferred.

`UNVERIFIED — a dedicated owning handoff for THIS directory.` The tq3 handoff owns the sub-2-bit
watch and records these runs, but no handoff row names `data/bonsai_current_v7` as its artifact
root.

## Not carried

158 files remain untracked at the same path in the shared clone
`/mnt/raid0/llm/epyc-inference-research/data/bonsai_current_v7/`, including three `…cpu_context…`
run directories (2026-07-19 01:28–01:34), a `ternary_bonsai_q2_g64_mi210_context_currentv7…` run,
the two registry-cited CPU prompt-repair bundles and their per-prompt `arms/` subtrees, and the
Q2_0 layout-contract JSON. Nothing was deleted or moved; this commit adds documentation only and
does not change what is tracked.

No PII and no credential-shaped strings were found in the tracked files (scanned: zero email
addresses, zero credential assignments). The tracked files do carry first-party operational
detail — local absolute paths under `/mnt/raid0/llm/`, local build SHAs, and the model path
`/mnt/raid0/llm/models/Bonsai-8B.gguf`.

## Integrity

`SHA256SUMS` seals all 17 tracked files. `README.md` and `SHA256SUMS` are deliberately not in it
(documentation is not evidence, and hashing the README would make every doc edit break the seal),
following the 2026-08-02 precedent in `data/paddleocr_vl_receipt_extract_20260717T194415Z/`.
Verify from the repository root:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/bonsai_current_v7/SHA256SUMS
```
