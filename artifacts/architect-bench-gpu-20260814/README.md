# architect-bench GPU — Qwen3.8-27B quality + throughput campaign, 2026-08-14 → 08-20

Measurement evidence made durable on **2026-09-15** (NIB2-73a). The master registry cited
this directory twice as a bare path and cited one file inside it explicitly, and all of it
existed only as UNTRACKED files in the shared clone
`/mnt/raid0/llm/epyc-inference-research`. The evidence gate therefore passed in that one
checkout and reported `MISSING` in every worktree, and a `git clean` would have taken the
evidence behind the `qwen38_27b_q8_local` production-throughput attestation with it.
Copied byte-for-byte within the same clone (sha256-verified, all 167 files); the untracked
originals were left in place, as the 2026-08-02 migration did.

| | |
|---|---|
| origin | `/mnt/raid0/llm/epyc-inference-research/artifacts/architect-bench-gpu-20260814/` (untracked) |
| measured (UTC) | 2026-08-14 → 2026-08-20 |
| made durable | 2026-09-15 |
| carried | **167 files, 362,078 bytes** — of 432 files / 23,439,000 bytes (1.54%) |

## Registry claims this backs

`orchestration/model_registry.yaml` — line numbers as of 2026-09-15; the YAML key path is
the stable reference.

- **L1306** &nbsp;`server_mode.architect_general.acceleration` (comment above
  `spec_type: draft-mtp` / `draft_max: 8`) → names `mtp_nmax_sweep_20260819/` and
  `mtp_ab_20260819/`. Justifies `draft_max: 8` as a **re-measured**, not inherited,
  optimum for Qwen3.8-27B: plain 27.78, n2 39.77, n3 46.61, n4 51.03, n6 55.22, n8 55.46,
  n12 51.14 t/s, with acceptance falling 0.842 → 0.482 across depth 2 → 8. The curve turns
  over between 6 and 8 — the reason the claim is "2.00x over plain at n-max 8" and not
  "deeper is better".
- **L10025** &nbsp;`roles.qwen38_27b_q8_local.production_throughput.attest` → names
  `mtp_ab_20260819/` (plain-vs-MTP), `mtp_nmax_sweep_20260819/` (depth sweep) and
  `q38_vram_shape_20260820/` (VRAM at production shape). Attests the whole
  `production_throughput` block: `optimized_tps: 55.46`, `baseline_tps: 27.78` (spec OFF,
  same 12 prompts), `vram_gib: 37.22` (39,963,078,656 B sampled DURING residency at
  n_slots=4, n_ctx 262144, q8_0 KV, kv_unified=true), and — importantly —
  `optimized_tps_long_context: null` and `contended_tps: null`, both *not measured*.
- **L10075** &nbsp;`roles.qwen38_27b_q8_local.acceleration.challenger_under_evaluation.evidence`
  → `dflash2_np1_20260820/campaign-summary.json`. The `dflash2-block8` challenger at np=1:
  decode **70.0 t/s** vs a same-campaign matched MTP n-max-8 arm at **55.2** (reproducing
  the historical 55.46 to within 0.5%), `gain_vs_matched_mtp_pct: 26.81`, mean draft
  acceptance 0.62804 vs 0.48246. Paired identity is proved inside the summary itself (same
  12 question ids AND prompt fingerprints in order, all seed 42 rep 0, 36/36 request rows
  exact, zero errors). Status is `np1_only_NOT_SELECTABLE`; the summary carries its own
  scope limit, *"DF2-4 np=1 speed/acceptance evidence; no np=8 or promotion conclusion"*.
  It ran on `llama.cpp-experimental ak/dflash2-qwen38-20260820 @ 2046c64e` (build 10131),
  so it **cannot** run on the frozen v9 production binary.

## What was carried, and why each class earns its bytes

| class | files | purpose |
|---|---|---|
| `result.json`, `regen_result.json` | 30 | **The distilled per-suite scores.** Schema `v7_quality_gate_capture.v4`: accuracy, n, correct, errors, truncated, per-tier and per-item breakdowns, a `capture{}` completeness block, and `throughput{}` (concurrency, wall_s, tokens, `aggregate_decode_tok_s`). These *are* the claims. |
| `campaign-summary.json`, `summary.json` | 4 | The sealed campaign record (`epyc.df2.matched_np1_campaign.v1`) plus the three per-arm digests. |
| `preflight.json`, `binary-version.stderr`, `linkage-report.txt` | 3 | The provenance chain: binary path + `binary_sha256`, target/draft model paths + sha256 + sizes, runner sha256, questions file sha256 + count, `source_commit`, `source_clean`. `linkage-report.txt` is an `ldd` audit proving every `libggml*`/`libllama*` resolved **inside** the dflash2 build dir — the one thing that distinguishes a real experimental-binary run from a silent fallback to another tree's ggml. |
| `server_command.txt`, `runner_command.txt`, `commands.json`, `geometry.txt` | 31 | Exact argv and the grepped `n_slots = 4, n_ctx_slot = 262144, kv_unified = 'true'` line that the `vram_gib` claim depends on. Highest provenance value per byte in the tree. |
| `rocm_during.txt`, `kfd_line.txt` | 6 | GPU state sampled **during** the run, not after — the only form of VRAM evidence that is admissible at all. |
| `accept.txt`, `acceptance.txt` | 10 | The raw `slot print_timing: … draft acceptance = …` lines behind every acceptance fraction quoted above. The `plain` arm's copy is empty **by design** (no speculation, so no acceptance lines) — absent, not zero. |
| `*.live-status.json` (suite-level) | 30 | The runner's fail-closed completeness markers: `completed_draws`/`expected_draws`, `complete`, `request_error_rows`, `length_cap_rows`, `provisional`, `artifact_integrity_fail_closed`. This is how a reader tells a finished capture from a truncated one. |
| `convert_diag*.json`, `waiver_manifest.json` | 7 | SWE-bench search/replace→patch conversion gate, and the **explicit scoring waiver** naming which instance/block was skipped and what the status would have been without it. A reader must see the waiver to know a gate was waived. |
| `capture-status.json` | 4 | Per-instance agentic capture integrity: `evidence_complete`, `model_patch_sha256`, `trajectory_sha256`, and an `anomalies[]` list that appears nowhere else. 50 KB of the 362 KB, kept because dropping the only record of capture anomalies to save 50 KB is a bad trade. |
| `transport.json`, `claim-*.json`, `claim-journal.jsonl`, `processes`-adjacent | 12 | Arm lifecycle and the MI210 device-claim lease receipts (acquire/release), i.e. the proof that nothing else was co-resident. |
| `harness.stdout`, `runner.stderr`, `q38_merged_provenance.json`, `*.exit_code` | 30 | Compact human-readable run outcomes and the instance→run mapping. |

## What was NOT carried — 265 files, 23,076,922 bytes

Excluded deliberately, by class. Nothing here is cited, and everything a reader needs from
it is already distilled into a file above.

1. **Raw capture substrate — 15,357,553 B (65.5%).** 12 × `per_question.jsonl`, 11 ×
   `pq.jsonl`, 7 × `regen_pq.jsonl` (full prompts and full model completions) and 61
   agentic `trajectories/<instance>.jsonl` (turn-by-turn tool transcripts with embedded
   `/testbed` repository source). `result.json` is the distillation of exactly these.
2. **Verbose server logs — 5,306,589 B (22.6%).** `server.log`, `server_code.log`,
   `server_regen.log`, `agentic*/server.log` and 23 × `server.stderr`, all `-lv 3` dumps.
   Every line anyone cites has already been grepped out into `accept.txt`, `geometry.txt`,
   `kfd_line.txt` and `binary-version.stderr`.
3. **VRAM telemetry — 1,941,705 B (8.3%).** 3 × `resource-samples.json`, roughly 6,500
   quarter-second polls, distilled to four integers per arm in `campaign-summary.json`.
4. **SWE-bench submission patches — 454,664 B.** 5 × `predictions.json`,
   `q38_merged_predictions.json` and 4 × `predictions.json.diagnostics.jsonl`.
   `convert_diag.json` carries the conversion summary; the patch bodies are upstream repo
   diffs and the suite scored 0.0 accuracy.
5. **Noise — 63 files, ~15 KB.** 23 empty `server.stdout`, 20 empty `runner.stdout`,
   20 × `server.pid` (a bare PID is not evidence), 61 per-instance
   `<instance>.jsonl.live-status.json` markers subsumed by `transport.json`.

**This exclusion also removes the entire third-party-PII surface, which is an independent
reason for it and not merely a size argument.** The bulk files contain real email
addresses — scikit-learn author headers (`olivier.grisel@ensta.org`,
`gael.varoquaux@normalesup.org`, `amueller@ais.uni-bonn.de` and others), django git-log
credits (`timograham@gmail.com`, `scott@staplefish.com`) surfaced by an agent running
`git log --all` inside `/testbed`, and django/sphinx synthetic test fixtures. All of it is
already-public OSS commit metadata quoted into a benchmark prompt, but none of it belongs
in this repository and none of it is in the carry set. The 167 carried files scan clean:
zero email addresses, zero credential-shaped strings, zero `/home/<user>` paths.

## Three integrity caveats — recorded, not repaired

1. **`dflash2_np1_20260820/sha256sums.txt` intentionally over-covers.** It is a 60-line
   seal that hashes *every* file in that campaign directory, including `pq.jsonl` and
   `resource-samples.json`, which are NOT carried. It is kept because it is the campaign's
   own seal and its hashes are the record of what the seal covered; it is therefore not
   independently checkable from this checkout alone. Check it against the origin path
   above. `SHA256SUMS` in this directory is the seal for what *is* carried.
2. **`agentic4/mtp_evidence.txt` is 0 bytes at the origin** — a file named as evidence
   with no content in it. It is not carried, because committing an empty file dressed as
   evidence is worse than recording the fact here. Any claim that leans on MTP evidence
   for the `agentic4` run is unbacked by that file.
3. **`swebench_oracle*` scored 0.0 accuracy across all four capture attempts, and the
   `dflash2`/`mtp_nmax_sweep`/`mtp_ab` arms all scored 0.0 on `olympiadbench_hard` with
   `truncated: 12`** — every draw hit the 2048-token cap. Those zeros are a length
   artifact of a throughput-shaped run, **not** a quality result, and the registry cites
   these arms for decode rate only. Do not read an accuracy number off them.

Also note `meta.sampling_fields_are_requested_not_effective: true` in every `result.json`:
the recorded temperature/top_p/top_k are what was *requested*, and the file says so rather
than implying they were confirmed effective.

## Integrity

`SHA256SUMS` seals all 167 carried files, hashed after the copy and compared against the
origin (all 167 verified `OK`). `README.md` is deliberately not in it (documentation is
not evidence, and hashing it would make every doc edit break the seal), following the
2026-08-02 precedent in `data/paddleocr_vl_receipt_extract_20260717T194415Z/`. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c artifacts/architect-bench-gpu-20260814/SHA256SUMS
```
