# GLM-5.2 UD-IQ2_M — near-miss reviewer corpus: shadow observations and the failed C-CRAB P-REV-1 gate

Measurement evidence sealed on **2026-09-15** (NIB2-73a residual). These files were committed
to git as they were produced (2026-07-18 → 2026-07-19, across 14 commits); what was missing
was the durability pair — a README saying what was measured and a `SHA256SUMS` proving the
bytes have not drifted. Nothing was copied or moved; this commit adds documentation and hashes
only.

**This is the directory that carries the campaign's decision.** One of its 17 run directories,
`gc-shadow-repair4b-p-rev1-20260719T132459Z`, is the only decision-grade
(`observation_only: false`, `measurement_protocol: p_rev1`, attestation
`MEASUREMENT-P-REV1-OPERATOR-APPROVED-20260719`) hard-negative patch-review gate GLM-5.2 ever
ran, and it **failed**. Every other run here is explicitly stamped
`measurement_note: "pre-P-REV-1 observation; non-decision-gating"`. The model no longer exists
— the `GLM-5.2-UD-IQ2_M` artifact was deleted on 2026-08-31 under the operator KILL ruling
(`epyc-root handoffs/completed/glm51-reap-cpu-evaluation.md:112`) — so nothing here can be
re-run.

| | |
|---|---|
| origin | produced in place under `data/glm52_reviewer_corpus_direct/` and committed from the shared clone |
| measured (UTC) | 2026-07-18 03:14 – 2026-07-19 13:24, from each run's `generated_at` |
| commits that added it | `b2f53f2f` "Add GLM near-miss corpus shadow runner" (2026-07-18 03:32Z) → `b9e2b558` "Run RM-2 reviewer ablation slate" (2026-07-19 16:47Z); 14 commits |
| sealed | 2026-09-15 |
| carried | **180 files, 1,660,134 bytes** — 17 run directories plus a `.gitignore`, all tracked, nothing untracked |
| kernel | `llama.cpp-experimental` **build 10088 (`d1e5a20eb`)** for the 2026-07-18 runs, read from the server logs inside `glm52-nearmiss-code-n12-20260718Tcheckpoint/raw_prompt_and_server_log_artifacts.tar.gz` and `glm52-ccrab-patch-review-rowid-v5-notes-n12-20260718Tcodex/raw_prompt_and_server_log_artifacts.tar.gz`. **UNVERIFIED for the 2026-07-19 P-REV-1 run** — no server log was committed for it, so its build id is not recoverable from this directory. |
| serving | CPU-only GLM, `build-hip/bin/llama-server`, chat/completions, JSON schema, `--reasoning off --reasoning-budget 0`, recovered band `p12000_tk16384` |
| corpus | `/mnt/raid0/llm/datasets/nearmiss-corpus-v1/rows.jsonl` — **outside this repository.** Row ids are recorded, the rows are not. |

## The corpus is not here

Every run selects rows from `nearmiss-corpus-v1`, a local dataset that is not in this repo and
is not hash-recorded here. What *is* durable: each `summary.json` `corpus` block records the
selection filters, `n_judgeable_available`, the full `selected_row_ids` list and the
representation counts, and `decisions.jsonl` records the per-row decision. What is **not**
durable: the rows themselves. A reader can verify *what GLM decided about row
`nearmiss-v1:c-crab:00710c9c18cd10fb`* but cannot re-derive *what that row was* from this
repository. Treat the FA/FR numbers as re-checkable arithmetic over the committed decisions,
not as a reproducible experiment.

## Run inventory

Seven runs carry a full `summary.json` with a score block:

| Run | `generated_at` (UTC) | slice | n | FA | FR | parse |
|---|---|---|---:|---:|---:|---:|
| `glm52-nearmiss-code-n12-20260718Tcheckpoint` | 07-18 03:15 | mixed `seeded-mutation` + one `c-crab`, 6/6 | 12 | 16.7% | 66.7% | 0.0% |
| `glm52-nearmiss-code-n12-calibrated-20260718T034916Z` | 07-18 03:49 | same 12 rows, explicit `--gold-confidence multi_oracle` | 12 | 16.7% | 66.7% | 0.0% |
| `glm52-nearmiss-code-n24-binaryschema-20260718Tcontinuation` | 07-18 04:22 | `seeded-mutation`, binary approve/reject schema | 24 | 50.0% | 75.0% | 0.0% |
| `glm52-nearmiss-code-n24-cruxeval-exactmatch-20260718Tglmrepair2` | 07-18 05:02 | homogeneous `seeded-mutation|cruxeval|exact_match` | 24 | 0.0% | 16.7% | 0.0% |
| `glm52-nearmiss-code-n24-ccrab-patchdiff-20260718Tglmrepair3` | 07-18 05:17 | matched `c-crab|python` patch diffs | 24 | **91.7%** | 16.7% | 0.0% |
| `glm52-ccrab-patch-review-rowid-v5-notes-n12-20260718Tcodex` | 07-18 10:16 | pinned 12-row screen with curated oracle notes | 12 | 0.0% | 16.7% | 0.0% |
| **`gc-shadow-repair4b-p-rev1-20260719T132459Z`** | **07-19 13:24** | **GC-shadow-repair4b.2c, 24 hard accept controls + 24 matched hard negatives** | **48** | **41.7%** | **25.0%** | **0.0%** |

Five are dry-run plans, `mode: dry-run`, no inference:
`glm52-nearmiss-code-n12-dryrun-20260718Tcheckpoint` (07-18 03:14),
`glm52-cruxeval-exactmatch-dryrun-20260718Tcheckpoint` and
`glm52-mixed-dryrun-after-representation-guard-20260718Tcheckpoint` (both 07-18 05:01 —
the second is the one that matters: `refusal_reasons` has one entry, the representation guard
refusing the old unfiltered n=24 shape because its rows span eight representation buckets),
`glm52-ccrab-patchdiff-dryrun-20260718Tcheckpoint` (07-18 05:16) and
`glm52-gc4b-acceptobs-plus-hardneg-dryrun-20260719Tcodex` (07-19 09:13, the pre-flight for the
P-REV-1 run).

Five carry **only a report markdown**, no `summary.json`, no decisions:
`glm52-ccrab-patch-review-rowid-n6-20260718Tcodex`,
`glm52-ccrab-patch-review-rowid-v4-n6-20260718Tcodex`,
`glm52-ccrab-patch-review-rowid-v5-notes-n6-20260718Tcodex` (the three pinned n=6 screens) and
`glm52-ccrab-patchdiff-negative-evidence-v3c-n4-20260718Tcodex` (n=4, FA 0.0% / FR 50.0%) and
`glm52-seeded-debugbench-substring-multioracle-n12-20260718Tcodex` (n=12, FA 0.0% /
FR 100.0%) — both calibration reports stamped
`protocol: P-REV-1 (DRAFT — pre-amendment; observation-grade, non-decision-gating)`.

## The P-REV-1 gate — what it says

`gc-shadow-repair4b-p-rev1-20260719T132459Z/` carries `plan.json`, `summary.json`,
`decisions.jsonl`, `progress.jsonl`, `run_manifest.json`, both
`reviewer_calibration_report.json` and `.md`, and 48 `artifacts/*.response.json` — one per row.
`summary.json` `execution.score`: `n: 48`, `n_good: 24`, `n_bad: 24`, approve/reject 28/20,
`false_accepts: 10` (`fa_rate: 0.4167`), `false_rejects: 6` (`fr_rate: 0.25`),
`parse_failures: 0`. `gc-shadow-repair4b-p-rev1-20260719T132459Z/reviewer_calibration_report.md` adds the calibration view — ECE 0.239,
**AUC 0.509**, Brier 0.278, FA/FR ratio 1.67 — under the rubric
`glm52_direct_nearmiss_review_v5+binary_schema+task_test_alignment+oracle_notes`, and states
its own grade: "P-REV-1: metrics are decision-grade for the material inputs and attestation
recorded in the supplied run manifest."

AUC 0.509 is the number that decides it: on a balanced 24/24 slice that is chance.

## Verdict

Recorded, verbatim, in several places.

- **epyc-root `handoffs/active/glm52-reviewer-capability-gates.md:175`** (GC-shadow-repair4b.2d)
  — "Result: `FA 41.7%` (`10/24`), `FR 25.0%` (`6/24`), accept `58.3%`, parse `0.0%`,
  ECE/AUC/Brier `0.239/0.509/0.278`, elapsed `6443.405s`… **Verdict: GLM-5.2 is not
  patch-reviewer role-ready.**" The parent row at `:161` adds "Do not rerun this same C-CRAB
  policy unchanged."
- **The campaign decision, GC-external-1e**, same file `:185` — "Verdict: GLM is not admitted
  as production patch reviewer; scope it to research/judge-preference/accept-control
  diagnostics unless a concrete new repair hypothesis exists", narrative at `:109-120`.
- **`orchestration/model_registry.yaml:8109`** — `verdict: route_away_on_current_policy`, in
  the `measured[]` row for this artifact (`role: patch_reviewer`, `slice:
  GC-shadow-repair4b.2c` at L8097, `median_latency_seconds: 124.0` at L8107,
  `role_ready: false` at L8108).
- **epyc-root `progress/2026-07/2026-07-19.md:1593-1604`** and
  `progress/2026-07/2026-07-19-p-gpu-1-glm-quality.md:64-66` ("First claim-grade verdict:
  GLM-5.2 is not a usable patch reviewer").
- **In this repo**, the pre-P-REV-1 dispositions are in
  `../../docs/reference/models/model-admission-2026-07-16.md`: §230 near-miss shadow with the
  Disposition at :271, §273 representation repair at :304, §306 patch-diff observation at :331
  ("GLM is too permissive on patch diffs under the current task-grounded prompt"), §333
  oracle-note repair at :359 and §365 n=12 confirmation at :390. The summary row at `:28`
  carries the whole arc. There is **no** dedicated P-REV-1 section in that doc — the
  decision-grade run is recorded in the registry, the handoff and
  `../../docs/data/reviewer_model_ablations_rm2_fast_ccrab_p_rev1_20260719.md`.

**Do not read the pre-P-REV-1 wins as the story.** `cruxeval|exact_match` n=24 at FA 0.0% and
the pinned n=6/n=12 oracle-note screens at FA 0.0% are real, and they are why the campaign kept
going; the admission doc's own disposition at `:304` is explicit that "GLM is not globally
failed as a reviewer… However, the improved result is for exact-answer review only." The
matched hard-negative gate is the one that counts, and it failed.

## Comparator context

The same 48-row slice was replayed against three other reviewer arms the same day:
`../../docs/data/reviewer_model_ablations_rm2_fast_ccrab_p_rev1_20260719.md` lists this directory's
GLM run as the CPU baseline (median row wall 121.7 s) beside Qwen3.6-27B Q8 (FA 54.2%,
AUC 0.503, 6.2 s), Qwable IQ4_XS (AUC 0.438) and a Qwen+Qwable scaffold (FA 33.3%, AUC 0.659,
FR 41.7%). Its verdict: "No tested small/fast arm cleanly beats GLM as a production reviewer."
Those arms' artifacts live in `data/reviewer_model_ablations/`, not here.

## Registry claims this backs

`orchestration/model_registry.yaml`, all under `roles.glm_52_ud_iq2m`. Line numbers are as of
the commit that adds this README; the YAML key paths are the stable reference.

- **L7988** &nbsp;`performance.reviewer_corpus_shadow_observation` — the pre-P-REV-1 arc. Names
  `glm52-nearmiss-code-n12-20260718Tcheckpoint` with its four files (L7993), the calibrated
  replay directory (L8004) and the binary-schema continuation (L8011), and reports FA 16.7% /
  FR 66.7% / ECE 0.392 / AUC 0.414 / Brier 0.402 for the first two and FA 50.0% / FR 75.0% /
  AUC 0.663 for the third. Closes: "Any next corpus run must change reviewer policy, prompting,
  thresholding, or calibration first."
- **L8017** &nbsp;`performance.reviewer_corpus_p_rev1_failure` — the decision. Names
  `gc-shadow-repair4b-p-rev1-20260719T132459Z/` (L8022) and states "Verdict: not
  patch-reviewer role-ready under this policy; do not rerun unchanged."
- **L8090** &nbsp;`performance.measured[]`, first row — the structured claim tuple, evidence at
  **L8110**.
- **L8236** &nbsp;`performance.reviewer_corpus_shadow` — 12 absolute paths (`summary.json`,
  `decisions.jsonl`, the per-run `reviewer_calibration_report` markdown, `raw_prompt*.tar.gz` for three runs).
- **L8249** &nbsp;`performance.reviewer_corpus_p_rev1` — 3 absolute paths into the P-REV-1 run.

Uncited but kept: the five dry-run plans, the five report-only directories, the
`cruxeval-exactmatch` and `ccrab-patchdiff` n=24 runs and the `rowid-v5-notes-n12` screen are
named in `../../docs/reference/models/model-admission-2026-07-16.md` but not by any registry key.
They are the falsification trail — the over-approval failure (FA 91.7%) and the
representation-guard refusal are what establish that the surviving positives are narrow, so
dropping them would make the cited claims unfalsifiable.

## What was NOT committed — and why

The tracked `.gitignore` in this directory excludes two classes:

```
*/artifacts/*.prompt.txt
*/logs/
```

Per `../../docs/reference/models/model-admission-2026-07-16.md:271`, expanded `artifacts/*.prompt.txt`
and `logs/` copies were treated as "local scratch and ignored to avoid whitespace-only artifact
churn". Two runs escape the exclusion by archiving their raw prompts and server log into a
tarball that *is* committed
(`glm52-nearmiss-code-n12-20260718Tcheckpoint/raw_prompt_and_server_log_artifacts.tar.gz`,
`glm52-ccrab-patch-review-rowid-v5-notes-n12-20260718Tcodex/…`); four more commit
`raw_prompt_artifacts.tar.gz` (prompts only, no server log — the calibrated replay used
`--no-trace-logs`, per `../../docs/reference/models/model-admission-2026-07-16.md:257`).

The **P-REV-1 run itself has no archive**: its `logs/` are excluded and no tarball was made, so
the server-side timing and the kernel build id for the decision-grade run exist only in the
untracked shared clone at
`/mnt/raid0/llm/epyc-inference-research/data/glm52_reviewer_corpus_direct/gc-shadow-repair4b-p-rev1-20260719T132459Z/logs/`.
They are not hash-recorded, so — unlike the WITHHELD precedent in
`data/paddleocr_vl_receipt_extract_20260717T194415Z/` — there is no way to prove a scratch-side
copy is the one that was measured. The decision itself is durable (48 response artifacts,
`decisions.jsonl`, both calibration reports); its wall-clock substrate is not.

Also note an asymmetry inside the tracked set: 84 `*.response.json` but only 36
`*.request.json`, and only four run directories have a loose `artifacts/` tree at all. Three of
them (`…nearmiss-code-n12-20260718Tcheckpoint`, `…n12-calibrated…`,
`…rowid-v5-notes-n12-20260718Tcodex`) carry a matched 12 requests + 12 responses. The P-REV-1
run carries **48 responses and zero requests** — the prompts it actually sent are not in this
repository in any form, tarball included. The three n=24 observation runs have no loose
`artifacts/`; their prompts and responses are inside their `raw_prompt_artifacts.tar.gz`.

## Third-party content

The committed `artifacts/*.response.json` and `*.request.json` embed excerpts of the
`nearmiss-corpus-v1` rows, which are built from public benchmark corpora — C-CRAB / SWE-CARE
patch diffs from open-source Python repositories (SQLFluff, CVAT and others) and CruxEval /
DebugBench snippets. A scan finds one email address, `contact@all-hands.dev` (the OpenHands
project's public contact line, inside a quoted source file), and one credential-shaped hit, the
literal `secrets`/`SECRET_KEY` identifiers inside a quoted `tests/unittests/plugins/secrets_tests.py`
diff. There are no real credentials and no private-individual PII. No files are withheld.

## Integrity

`SHA256SUMS` seals all 180 tracked files, including the `.gitignore` (it is load-bearing
provenance — it is *why* the logs are absent). `README.md` is deliberately not in it
(documentation is not evidence, and hashing it would make every doc edit break the seal),
following the 2026-08-02 precedent in
`data/paddleocr_vl_receipt_extract_20260717T194415Z/`. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/glm52_reviewer_corpus_direct/SHA256SUMS
```

## Known defect, not fixed here

`../../docs/reference/models/model-admission-2026-07-16.md:359-363` cites
`data/glm52_reviewer_corpus_direct/glm52-ccrab-patch-review-rowid-v5-notes-n6-20260718Tcodex/summary.json`,
and the same applies to the `…rowid-n6…` directory. **Neither `summary.json` is tracked** —
both exist only as untracked files in the shared clone. The evidence-durability gate scans
`orchestration/model_registry.yaml`, and these paths are cited from a doc, so the gate does not
see them. Committing them is outside this README's scope; recorded here so the next reader does
not conclude the citation was always dangling.
