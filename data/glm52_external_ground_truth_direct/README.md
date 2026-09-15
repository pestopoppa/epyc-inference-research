# GLM-5.2 UD-IQ2_M — external ground-truth P-REV-1 gates (JudgeBench-GPT, SWE-bench-Verified)

Measurement evidence sealed on **2026-09-15** (NIB2-73a residual). These files were committed
to git the same day they were produced (2026-07-19); what was missing was the durability pair
— a README saying what was measured and a `SHA256SUMS` proving the bytes have not drifted.
Nothing was copied or moved; this commit adds documentation and hashes only.

This directory holds the only two **decision-grade** (`observation_only: false`,
`measurement_protocol: p_rev1`) external-benchmark runs in the GLM-5.2 campaign, both under
operator attestation `MEASUREMENT-P-REV1-OPERATOR-APPROVED-20260719`, era `p_rev1_attested`.
The measured model no longer exists: the `GLM-5.2-UD-IQ2_M` artifact was deleted on 2026-08-31
under the operator KILL ruling
(`epyc-root handoffs/completed/glm51-reap-cpu-evaluation.md:112`), so these runs can never be
repeated.

| | |
|---|---|
| origin | produced in place under `data/glm52_external_ground_truth_direct/` and committed from the shared clone |
| measured (UTC) | 2026-07-19 15:45 – 17:56, from each run's `generated_at` / `run_manifest.json` |
| commits that added it | `4af1ebe2` "Record GLM external JudgeBench gate" (2026-07-19 17:07Z), `9dd669d0` "Prepare GLM SWE external direct gate", `8482b66e` "Record GLM SWE external live gate" (2026-07-19 18:03Z) |
| sealed | 2026-09-15 |
| carried | **16 files, 407,448 bytes** — 4 run directories, all tracked, nothing untracked |
| serving | CPU-only GLM, `build-hip/bin/llama-server`, chat endpoint, band `p12000_tk16384` (`context_length 16384`, `indexer_top_k 16384`), `--reasoning off --reasoning-budget 0`, `temperature 0.0`, `seed 52`, `max_tokens 128`, server-side `--json-schema` |
| kernel | UNVERIFIED — no build id is recoverable from the carried files. `summary.json` records the binary *path* `/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server`; the server logs that would carry `build NNNNN (sha)` were never committed (see below). The sibling 2026-07-18 runs in `data/glm52_dsa_probe/` and `data/glm52_reviewer_capability_direct/` record **build 10088 (`d1e5a20eb`)**, but that is one day earlier and is not evidence about these runs. |

## Run inventory

| Run | mode | n | rows source | what it produced |
|---|---|---:|---|---|
| `glm52-external-judgebench-gpt-n24-p-rev1-20260719T154517Z` | `execute` | 24 | `docs/data/glm52_external_judgebench_gpt_n24_rows_20260719.jsonl` | live pairwise A/B; `plan.json`, `progress.jsonl`, `decisions.jsonl`, `run_manifest.json`, `summary.json`, plus `choice_rescore_response_rows_20260719.jsonl` (the saved responses the rescore replays) |
| `glm52-external-judgebench-gpt-n24-p-rev1-choice-rescore-20260719` | `score-responses` | 24 | same rows, responses from the run above | no inference (`server.not_started: true`, `server.log_file: null`); rescores the saved responses under the exact-choice scorer |
| `glm52-external-swebench-verified-n24-p-rev1-20260719Tlive` | `execute` | 24 | `docs/data/glm52_external_swebench_verified_n24_rows_20260719.jsonl` | live accept-control patch review |
| `glm52-external-swebench-verified-n24-p-rev1-dryrun-20260719` | `dry-run` | — | — | `plan.json` only, no inference |

Both `rows_jsonl` inputs live **outside** this directory, at tracked paths under `docs/data/`,
together with the plan JSONs and the two write-ups. That is where the row selection is
reproducible from; this directory holds the outputs.

## Results as recorded in the artifacts

JudgeBench-GPT (12 gold A / 12 gold B), from
`…-20260719T154517Z/summary.json` `execution.score` and
`…-choice-rescore-20260719/summary.json` `score`:

| Scoring view | Correct | Accuracy | Parse failures |
|---|---:|---:|---:|
| Original strict schema | 15/24 | 62.5% | 7/24 |
| Exact-choice P-REV-1 rescore | 22/24 | 91.7% | 0/24 |

The seven "failures" were valid A/B decisions that expressed confidence on a 0–100 scale; the
rescore normalises that to a warning (`confidence_warning_counts:
{"confidence_scale_0_100": 7}`) instead of a failed decision. Final decisions were balanced
12 A / 12 B. Live run elapsed 1604.367 s.

SWE-bench-Verified accept controls (24 known-good patches, **no hard negatives**), from
`…-20260719Tlive/summary.json` `execution.score`: 22/24 correct approvals,
`false_rejects: 2`, `fr_rate: 0.0833`, `parse_failure_rate: 0.0`, `fa_rate: null` and
`hard_negative_n: 0` — the slice structurally cannot measure false accepts. Elapsed
1965.612 s. The two false rejects are named in the write-up:
`django__django-12663` and `pylint-dev__pylint-8898`, both rejected at confidence 0.9
(`../../docs/data/glm52_external_swebench_verified_p_rev1_live_20260719.md:33-34`).

## Verdict

Recorded, and it is a **bounded positive that does not clear the role**.

- **epyc-root `handoffs/active/glm52-reviewer-capability-gates.md:176`** (GC-external-1) —
  > Pairwise adapter/scoring/live path is complete, JudgeBench-GPT live evidence is positive,
  > and SWE-bench-Verified accept controls are live-scored positive. **Verdict: external
  > evidence improves the GLM picture but does not override the failed decision-grade C-CRAB
  > hard-negative gate.**

  Per-run rows at `:181` (GC-external-1d.1, JudgeBench) and `:184` (GC-external-1d.2b, SWE).
- **The route-away decision, GC-external-1e**, epyc-root
  `handoffs/active/glm52-reviewer-capability-gates.md:185` — "Verdict: GLM is not admitted as
  production patch reviewer; scope it to research/judge-preference/accept-control diagnostics
  unless a concrete new repair hypothesis exists", with the narrative at `:109-120` and the
  progress record at `epyc-root progress/2026-07/2026-07-19.md:1795-1802`.
- **In this repo**, each run has its own write-up carrying the disposition:
  `docs/data/glm52_external_judgebench_gpt_p_rev1_live_20260719.md:23` ("it does not clear
  patch-review admission because the same model already failed decision-grade C-CRAB P-REV-1")
  and `docs/data/glm52_external_swebench_verified_p_rev1_live_20260719.md:36` ("GLM remains
  research-only pending a new repair hypothesis or a policy decision that scopes GLM away from
  hard-negative patch-review"). The dry-run has
  `../../docs/data/glm52_external_swebench_verified_p_rev1_dryrun_20260719.md`.

The failing gate these positives do not override is
`data/glm52_reviewer_corpus_direct/gc-shadow-repair4b-p-rev1-20260719T132459Z/` — FA 41.7%,
FR 25.0%, AUC 0.509. Read that directory's README alongside this one; quoting 91.7% or 22/24
without it misrepresents the campaign.

Owning handoff: epyc-root `handoffs/active/glm52-reviewer-capability-gates.md` (series H6),
row `REV-02` in `handoffs/active/reviewer-control-plane-index.md:14` — now RETARGETED to
GLM-5.3-Flash, so this evidence is historical, never carry-forward state.

## Registry claims this backs

`orchestration/model_registry.yaml`, all under `roles.glm_52_ud_iq2m`. Line numbers are as of
the commit that adds this README; the YAML key paths are the stable reference.

- **L8035** &nbsp;`performance.external_judgebench_gpt_p_rev1_observation` — names both
  JudgeBench run directories (L8042–L8043) and reports 15/24 strict, 22/24 (91.7%) exact-choice
  rescore, parse 0/24, 7/24 confidence warnings. Closes: "This is positive judge-native
  pairwise preference evidence, not patch-review admission."
- **L8047** &nbsp;`performance.external_swebench_verified_p_rev1_observation` — 22/24 approved,
  2/24 false-rejected (FR 8.3%), parse 0/24. Closes: "This is positive accept-control evidence
  only; the slice has no hard negatives and does not clear the failed C-CRAB patch-review gate."
- **L8090** &nbsp;`performance.measured[]` — two `protocol: p_rev1` rows, both
  `role_ready: false`: **L8139** `role: judge_pairwise_preference`, corpus `judgebench-gpt`,
  `exact_choice_accuracy: 0.917`, evidence the choice-rescore `summary.json`; **L8153**
  `role: patch_review_accept_control`, corpus `swe-bench-verified`,
  `false_reject_rate: 0.083`, evidence the live `summary.json`. Each carries an explicit
  `limitation`.
- **L8260** &nbsp;`performance.external_ground_truth` — an evidence list of six absolute paths
  into this directory (`summary.json` + `decisions.jsonl` for the three scored runs) plus the
  two `docs/data/*_live_20260719.md` write-ups.

The `dryrun-20260719` directory is not cited by any registry key. It is kept because its
`plan.json` is the only record of the pre-flight row selection and refusal checks for the live
SWE run, and it is discussed at
`../../docs/data/glm52_external_swebench_verified_p_rev1_dryrun_20260719.md`.

## What was NOT committed — and why

Each live run's `logs/` and expanded `artifacts/` (raw prompts, requests, responses, and the
full `llama-server` log named in `summary.json` `server.log_file`) are **not in this
directory**. This was a deliberate decision at the time, recorded in
`docs/data/glm52_external_judgebench_gpt_p_rev1_live_20260719.md:29`:

> Expanded raw live `artifacts/` and `logs/` remain local/untracked because the repository PII
> hook flags long digit runs in saved prompts/server logs; the committed decision-grade
> evidence is the summary, manifest, decisions, and plan.

Consequences a reader must know:

- The **kernel build id is not recoverable** from this directory (see the table above).
- Per-row model outputs *are* durable — `decisions.jsonl` carries them — but the raw request
  payloads and the server-side timing log are not. `docs/data/…swebench…live_20260719.md:14-24`
  is the only surviving record of median/max row latency (69.236 s / 196.198 s), median/max
  prompt tokens (1194.5 / 3431) and the server prompt/decode tail (20.21 / 2.79 t/s).
- The originals remain untracked in the shared clone at
  `/mnt/raid0/llm/epyc-inference-research/data/glm52_external_ground_truth_direct/<run>/logs/`
  and `…/artifacts/`. They are **not** hash-recorded here, so unlike the WITHHELD precedent in
  `data/paddleocr_vl_receipt_extract_20260717T194415Z/` there is no way to prove a
  scratch-side copy is the one that was measured. That is a real gap, not a policy choice, and
  it cannot be repaired retroactively.

## Third-party content

The carried `decisions.jsonl` and `summary.json` embed excerpts of two public benchmark
corpora — JudgeBench-GPT prompt/response pairs and SWE-bench-Verified issue text and patch
diffs from open-source repositories (Django, pylint and others). A credential-shaped-string
scan hits once, on the literal `password` inside a quoted Django `help_text` diff
(`UserChangeForm.__init__`); there are no real credentials and no private-individual PII. No
files are withheld.

## Integrity

`SHA256SUMS` seals all 16 tracked files. `README.md` is deliberately not in it (documentation
is not evidence, and hashing it would make every doc edit break the seal), following the
2026-08-02 precedent in `data/paddleocr_vl_receipt_extract_20260717T194415Z/`. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/glm52_external_ground_truth_direct/SHA256SUMS
```
