# GLM-5.2 UD-IQ2_M — GC-1/2/3 reviewer-capability direct smokes (synthetic, n=3)

Measurement evidence sealed on **2026-09-15** (NIB2-73a residual). These files were committed
to git the same night they were produced (2026-07-18); what was missing was the durability
pair — a README saying what was measured and a `SHA256SUMS` proving the bytes have not
drifted. Nothing was copied or moved; this commit adds documentation and hashes only.

The measured model no longer exists: the `GLM-5.2-UD-IQ2_M` artifact was deleted on
2026-08-31 under the operator KILL ruling
(`epyc-root handoffs/completed/glm51-reap-cpu-evaluation.md:112`), so these runs can never be
repeated.

| | |
|---|---|
| origin | produced in place under `data/glm52_reviewer_capability_direct/` and committed from the shared clone |
| measured (UTC) | 2026-07-18 02:06 – 02:57, from each run's `summary.json` / `plan.json` `generated_at` |
| commits that added it | `ce61f8f4` "Add GLM reviewer capability direct smokes" (2026-07-18 02:37Z) and `fbac66a2` "Add GLM reviewer repair smokes" (2026-07-18 03:07Z) |
| sealed | 2026-09-15 |
| carried | **109 files, 1,117,178 bytes** — 11 run directories (3 dry-run plans, 8 executed), all tracked, nothing untracked |
| kernel | `llama.cpp-experimental` **build 10088 (`d1e5a20eb`)**, `build-hip/bin/llama-server`, read from the 9 committed `logs/*.server.log` files |
| serving | CPU-only GLM, chat/completions, `--reasoning-format deepseek --reasoning off --reasoning-budget 0`, recovered band `p2168` with `indexer_top_k=4096`, `context_length 4096` |

This is the **most completely carried** of the four GLM-5.2 reviewer campaign directories: it
holds prompts, requests, responses *and* server logs (27 `*.prompt.txt`, 27 `*.request.json`,
27 `*.response.json`, 9 `*.server.log`, 11 `plan.json`, 8 `summary.json`). Its sibling
`data/glm52_reviewer_corpus_direct/` gitignores prompts and logs, and
`data/glm52_external_ground_truth_direct/` carries neither.

## What was measured

Three reviewer-capability probes from `scripts/benchmark/glm52_reviewer_capability_direct_runner.py`,
schema `glm52_reviewer_capability_direct.v1`, each on a synthetic **n=3** task set — first as
a smoke, then as a prompt/scorer repair. All runs carry `observation_only: true` in the score
block.

| Run | probe | lanes | result (from `summary.json` `execution.lanes[].score`) |
|---|---|---|---|
| `gc1-strict-if-smoke-20260718Tglm52` | `strict_if` | grammar + free | grammar `emission_rate=1.0`; free lane parsed JSON but was schema-invalid 0/3 |
| `gc1-free-natural-repair-20260718Tglm52` | `strict_if` | free | `emission_rate=0.0` — the *first* repair attempt failed |
| `gc1-free-natural-repair2-20260718Tglm52` | `strict_if` | free | `emission_rate=1.0` — repair closed |
| `gc2-rubric-grammar-smoke-20260718Tglm52` | `rubric_authoring` | grammar | schema-valid 3/3 but shallow: `mean_axis_coverage=0.25`, `mean_composite=0.75` |
| `gc2-grammar-natural-repair-20260718Tglm52` | `rubric_authoring` | grammar | `mean_axis_coverage=1.0`, `mean_composite=1.0`, `mean_grounding_rate=1.0` |
| `gc3-why-smoke-20260718Tglm52` | `why_diagnosis` | free | `n_that_detected=3`, `n_why_matched=0` |
| `gc3-why-natural-repair-20260718Tglm52` | `why_diagnosis` | free | `n_that_detected=3`, `n_why_matched=0` — first repair also failed |
| `gc3-why-natural-repair2-20260718Tglm52` | `why_diagnosis` | free | `n_that_detected=3`, `n_why_matched=3` |
| `dryrun-20260718Tgc1-strict-if-smoke`, `dryrun-20260718Tgc2-rubric-smoke`, `dryrun-20260718Tgc3-why-smoke` | — | — | `mode: dry-run`, plan only, no inference |

**Watch the names.** For GC-1 and GC-3 the *first* natural-prompt repair failed and a second
(`…repair2…`) succeeded; for GC-2 the single `…grammar-natural-repair…` succeeded. The
registry and the handoff cite `gc1-free-natural-repair2`, `gc2-grammar-natural-repair` and
`gc3-why-natural-repair2` — the passing runs. `gc1-free-natural-repair` and
`gc3-why-natural-repair` are the failed first attempts, carried as the record of what did not
work, and not cited anywhere.

## Verdict

Recorded — and it is a *negative* verdict for the role, despite the repairs passing.

- **epyc-root `handoffs/active/glm52-reviewer-capability-gates.md:222`** (roll-up) —
  > **Yes for synthetic observations only**: GC-1r free typed emission `3/3`, GC-2r rubric
  > breadth `mean_composite=1.0`, and GC-3r synthetic why diagnosis `why_match_rate=1.0`.
  > This does not close P-REV-1 or corpus-v1 reviewer admission. ✅ 2026-07-18

  Per-gate rows at `:145` (GC-1), `:149` (GC-2), `:152` (GC-3) and repairs at `:146`, `:150`,
  `:153` — each ends "Observation-only".
- **`docs/reference/models/model-admission-2026-07-16.md:228`** (this repo) —
  > Disposition: GC-1/2/3 smoke execution is complete, but GLM-5.2 is still reviewer-quality
  > blocked. The schema-constrained typed-decision path is viable; rubric-authoring and
  > why-diagnosis need repair or broader claim-grade reruns before any reviewer-role claim […]
- **epyc-root `progress/2026-07/2026-07-18.md:80`** ("GC-1/2/3 smoke execution is closed, but
  GLM is not reviewer-role-ready") and `:126` ("repaired synthetic smokes are positive
  observations, not role admission").

The campaign-level verdict that supersedes all of this is GC-external-1e (2026-07-19): GLM-5.2
is not admitted as the production patch reviewer. See
`../glm52_reviewer_corpus_direct/README.md`.

Owning handoff: epyc-root `handoffs/active/glm52-reviewer-capability-gates.md` (series H6),
row `REV-02` in `handoffs/active/reviewer-control-plane-index.md:14` — now RETARGETED to
GLM-5.3-Flash, so this evidence is historical, never carry-forward state.

## Registry claims this backs

`orchestration/model_registry.yaml`, all under `roles.glm_52_ud_iq2m`. Line numbers are as of
the commit that adds this README; the YAML key paths are the stable reference.

- **L7971** &nbsp;`performance.reviewer_capability_direct_smoke_observation` — the long-form
  claim. Names the three *smoke* summaries (L7976–L7978: `gc1-strict-if-smoke`,
  `gc2-rubric-grammar-smoke`, `gc3-why-smoke`) and the runner. Reports GC-1 grammar 3/3 with
  the free lane failing schema on `blocking.tripwire=null`, GC-2 `mean_axis_coverage=0.25` /
  `mean_composite=0.75`, GC-3 defects detected 3/3 with root cause matched 0/3, first-uncached
  prompt 24.34–24.76 t/s and decode ~2.29–2.41 t/s. Closes with "This is observation-grade
  smoke evidence, not production reviewer admission."
- **L8090** &nbsp;`performance.measured[]` — three rows, protocol
  `reviewer_capability_smoke_observation`, each `role_ready: false`, pointing at the *repair*
  summaries: **L8164** gate `GC-1r` → `gc1-free-natural-repair2` (`emission_rate: 1.0`),
  **L8176** gate `GC-2r` → `gc2-grammar-natural-repair` (`mean_axis_coverage: 1.0`,
  `mean_composite: 1.0`), **L8187** gate `GC-3r` → `gc3-why-natural-repair2`
  (`defect_detection_rate: 1.0`, `why_match_rate: 1.0`). Each carries an explicit
  `limitation` naming the small synthetic n and the still-open gate.
- **L8256** &nbsp;`performance.reviewer_capability_direct` — an evidence list of the same three
  repair `summary.json` files by absolute path.

Two cited numbers are not in the carried files as stated: the registry's
"First uncached prompt rates were 24.34-24.76 t/s, and decode was roughly 2.29-2.41 t/s" is a
range across the three smokes, and the per-lane `elapsed_s` values in `summary.json`
(70.4–548.9 s) are wall-clock, not token rates. UNVERIFIED — whether those t/s figures are
recomputable from the carried `logs/*.server.log` was not checked in this pass; the
`docs/reference/models/model-admission-2026-07-16.md:224-226` table reports the same range.

## Integrity

`SHA256SUMS` seals all 109 tracked files. `README.md` is deliberately not in it (documentation
is not evidence, and hashing it would make every doc edit break the seal), following the
2026-08-02 precedent in `data/paddleocr_vl_receipt_extract_20260717T194415Z/`. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/glm52_reviewer_capability_direct/SHA256SUMS
```

No PII and no credentials: scanned for email addresses, credential-shaped strings and the host
name, zero hits. The task set is synthetic (authored for the probe), so no third-party content
is carried. The files do carry first-party operational detail — local absolute paths under
`/mnt/raid0/llm/`, ports, OS PIDs in `preexisting_processes`, and full llama-server startup
logs.
