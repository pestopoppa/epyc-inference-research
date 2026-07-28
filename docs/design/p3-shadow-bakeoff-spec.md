# P3 Shadow Bake-off — stock-27B vs Fable-Fusion on the GPU shadow lane

**Filed**: 2026-07-28, gpu-serving-tie-in-program task **P3-1** (harness built zero-inference; runs happen in operator windows).
**Program authority**: `epyc-root/handoffs/active/gpu-serving-tie-in-program.md` (ratified decisions D1–D10).
**Lane spec**: `epyc-orchestrator/docs/gpu-shadow-lane.md` (role `coder_escalation_shadow`, port 18100, Steps 0–7 activation choreography).
**Status**: HARNESS READY — no run has happened; every number this spec cites from prior campaigns is an observation.

## What this bake-off is

Once the GPU shadow lane is activated (operator-gated), the bake-off compares two tenants on
TWO separately-scored duties, via the **eval path only — never live /chat** (D3):

| Duty | Shape | Scored by |
|---|---|---|
| **coder** | escalation-shaped SWE-flavored tasks at production sampling (no-think) | SWE: pinned SEARCH/REPLACE converter + swebench harness (`resolved_ids`); LCB: executable code-execution oracle |
| **co-critic** | [candidate solution + typed review request] → typed ReviewDecision verdict | deterministic replay scorer vs executable-oracle gold labels (FA/FR/kappa) |

## Arms

| Key | Model | Serving | sha256 |
|---|---|---|---|
| `stock27b` | `Qwen_Qwen3.6-27B-Q8_0.gguf` (28,665,067,072 B) | shadow lane :18100 | `5927dc06…` (matches the lane-spec pin; re-verified 2026-07-28) |
| `ff27b` | `Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-Q8_0.gguf` (29,787,701,792 B, non-MTP — lane default MTP OFF per D6) | shadow lane :18100 (tenant swap = State-B′ choreography) | `2fff409d…` |
| `a4_control` | `Qwen3.6-35B-A3B-MTP-Q8_0.gguf` (37,801,097,504 B) | **sequential GPU bench window only** — co-residency impossible (37.8 GB); production `coder_escalation` incumbent | `c1283d8b…` |

Full identities (paths, sizes, hashes) are pinned in the manifest.

## Pinned manifests (pairing discipline)

Everything is sha256-pinned in
`artifacts/p3-shadow-bakeoff-20260728/manifest/p3_bakeoff_manifest.json`
(builder/verifier: `scripts/benchmark/p3_bakeoff_manifest.py`; a hash mismatch at run
time fails closed — no silent resampling).

| Task set | Source | n |
|---|---|---|
| coder / `swebench_oracle` | `artifacts/architect-code-eval-20260724/questions_swebench_oracle.json` (reused verbatim — same items as the FG-1 six-arm campaign) | 40 |
| coder / `livecodebench_hard` | `artifacts/architect-code-eval-20260724/questions_livecodebench_hard.json` (reused verbatim) | 53 |
| FG-1 **hard-core tag** | extracted from `fg1_results.json → swe40.unsolved_by_all_six`: the 14/40 SWE instances unsolved by all six FG-1 arms; tagged subset of the SWE 40, **descriptive breakdown only, never a gate** | 14 |
| co-critic / `p3_cocritic_v1` | `artifacts/p3-shadow-bakeoff-20260728/manifest/critic_tasks_v1.json` (built by `p3_bakeoff_critic_build.py`) | 120 (60/60) |

Sampling (identical across arms and duties; production discipline, matches the
2026-07-24 code-eval campaign): chat endpoint, `temperature 0.6, top_p 0.95, top_k 20,
seed 42`, `enable_thinking=false` (no-think protocol), repeats 1, concurrency 1.
Max tokens: SWE 3072 · LCB 4096 · co-critic 1024.

## Capture and scoring (reuse, not reinvention)

Capture runs through the existing **`v7_quality_gate_runner.py`**
(`v7_quality_gate_capture.v4` schema): per-question incremental JSONL persistence,
fingerprinted prompt/response/reasoning, idempotent resume, live-status sidecar for
`capture_integrity_watchdog.py`. The bake-off runner (`p3_bakeoff_runner.py`) only
orchestrates: it verifies manifest pins, emits/executes the exact child-runner
invocations, and **never launches or manages servers** (the lane is assumed up on
`--host/--port` via the operator-gated choreography).

Scoring is deferred to the existing deterministic replay paths:

- **SWE**: `convert_sr_to_patch.py` (pinned in the manifest) → pinned swebench
  harness (`.venv-swebench`, cpuset adapter); the harness report's `resolved_ids`
  is authoritative.
- **LCB**: executable pass@1 via `answer_scoring` code-execution oracle (scored at
  capture; offline re-scorable from the captured rows).
- **Co-critic**: `p3_bakeoff_critic_score.py` — parses typed verdicts from captures
  (fail-closed on schema/fingerprint mismatch) and scores against gold labels.

## Co-critic duty

Task shape: the model receives the original task, a candidate solution, and a typed
review request; it must emit ONE JSON object
`{"decision", "confidence", "blocking": {"tripwire"}}` where `decision` is exactly the
reviewer-control-plane vocabulary (`review_decision.schema.json`):
`approve / request_changes / reject / reject_to_empty / request_evidence / abstain / escalate`.
`confidence` is confidence **in the verdict** (H4 semantics, distinct from advisory
score). Nothing in this vocabulary is invented here.

Scoring (H4 calibration vocabulary): accept-class = `{approve}`; reject-class =
`{reject, reject_to_empty, request_changes}`; non-committal =
`{request_evidence, abstain, escalate}` — the **declared abstention estimand**,
reported as its own rate (with parse-failure rate separate — cross-arm parse gaps are
scorer artifacts until proven otherwise). Committed-only: **FA** (accept on
known-wrong, lower-better), **FR** (reject on known-correct, lower-better), FA/FR
ratio first-class, **Cohen's kappa + prevalence disclosure** (intake-876). Paired
primary metric `verdict_correct` counts non-committal/parse-fail as incorrect
(conservative, stated).

### Co-critic corpus construction (PROPOSAL — mark reviewed before first decision use)

`critic_tasks_v1` mines candidates from **banked** per-question captures of the
2026-07-24 architect code-eval campaign (LCB-hard, arms A3/A4/A1, production
sampling), whose `correct` labels come from the executable code-execution oracle
recorded at capture time:

- eligibility: `finish_reason == "stop"`, non-empty, no request error, no truncation
  (a truncated candidate's label reflects token budget, not quality; an empty
  candidate is degenerate);
- deduplication by response hash; class-balanced 60 known-correct / 60 known-wrong;
  question-stratified round-robin so no item dominates; deterministic (seed 42);
- provenance (source file + sha256 + response hash) embedded per task.

**Known limitations of v1 (why this stays a proposal):** single-oracle gold
(code-execution only — H4's gate-worthy bar is ≥2 oracles or human arbitration);
candidates are same-family Qwen outputs (author/critic family overlap; cross-family
grading is the H4 mandate for closed loops — acceptable for a paired A-vs-B shadow
read, stated here); LCB-only domain. The runner and scorer are **generic over the
task file** — a v2 critic set (e.g. drawn from `nearmiss-corpus-v1` multi-oracle
rows) swaps in by pinning a new file into the manifest, with no code change.

## Statistical plan (stated honestly)

Primary: **paired per-question comparison; exact two-sided McNemar on discordant
pairs**, per duty and per suite (never pooled across duties — the duties are scored
separately by design).

| Suite | n (pairs) | MDE (α=.05, power=.8) |
|---|---|---|
| swebench_oracle | 40 | ~0.20 accuracy at discordant rate 0.20 (FG-1 measured FF-vs-stock discordants 8/40) |
| livecodebench_hard | 53 | ~0.19 at discordant rate 0.25 |
| p3_cocritic_v1 | 120 | ~0.13 at discordant rate 0.25 |

**Consequence**: at these n, only large per-duty quality gaps are resolvable. FG-1
already measured FF-vs-stock SWE quality as statistically tied (+2/−6, p=0.29). The
expected outcome is therefore a quality tie on coder, decided on **secondary axes**:
paired token economics (median completion tokens; tokens/solved — FG-1: FF 1082.6 vs
stock 1506.5, banked observation) and latency/decode telemetry (observation-grade).
The co-critic duty at n=120 is the more discriminating new signal.

## Decision rule feeding P3-2 (tenancy decision package)

Per duty, independently:

1. If exact McNemar p < 0.05 → that duty has a quality winner.
2. Otherwise the duty is a quality tie → rank by token efficiency (paired), then
   latency; report point estimates with the MDE caveat.
3. Co-critic additionally reports FA/FR/kappa per arm; a reviewer-shaped duty winner
   must not have pathological FA (accept-on-wrong) regardless of accuracy.

The P3-2 package to the operator contains: per-duty winner(s) or tie verdicts,
token-efficiency and latency tables, the A4-control comparison (is either 27B tenant
actually competitive with the production incumbent on escalation-shaped work?), and
the co-critic calibration table. **Duties may split across tenants** (one resident
tenant at a time; epoch-based swap via the State-B′ choreography).

## What this bake-off does NOT authorize (D3)

- **No lineup change.** The registry stays frozen; `coder_escalation` stays A4-bound.
- No production traffic on the lane; all runs are forced-role eval-path in operator
  windows. First production traffic requires **P3-3 operator three-gates sign-off**.
- No MEASUREMENT trust-boundary amendment: results are observations under the
  current instruments; any promotion-gating claim needs its own protocol citation.
- The harness never launches/stops servers, never manages processes, never touches
  autopilot.

## Operator-window run recipe

```bash
cd /mnt/raid0/llm/epyc-inference-research
M=artifacts/p3-shadow-bakeoff-20260728/manifest/p3_bakeoff_manifest.json

# 0. Preflight (zero-inference): pins + plan
python3 scripts/benchmark/p3_bakeoff_manifest.py verify --manifest $M
python3 scripts/benchmark/p3_bakeoff_runner.py --manifest $M --arm stock27b   # plan only

# 1. Lane up with tenant stock27b (operator: gpu-shadow-lane.md Steps 0–6), then:
python3 scripts/benchmark/p3_bakeoff_runner.py --manifest $M --arm stock27b \
    --host 127.0.0.1 --port 18100 --run-id <window-id> \
    --execute --i-have-operator-grant

# 2. Tenant swap to ff27b (State-B′), rerun with --arm ff27b (same --run-id).
# 3. A4 control window (sequential; lane torn down), rerun with --arm a4_control
#    and the window's A4 port.
# 4. Score + report (zero-inference; see plan.json post_run for exact commands):
#    SWE: convert_sr_to_patch.py → pinned swebench harness → report resolved_ids
#    critic: p3_bakeoff_critic_score.py per arm
#    paired: p3_bakeoff_report.py per suite per arm-pair
```

## Files

| File | Role |
|---|---|
| `scripts/benchmark/p3_bakeoff_common.py` | verdict parsing (ReviewDecision shape), exact McNemar, kappa, MDE, hashing |
| `scripts/benchmark/p3_bakeoff_critic_build.py` | co-critic corpus builder (proposal v1) |
| `scripts/benchmark/p3_bakeoff_manifest.py` | pinned manifest build/verify |
| `scripts/benchmark/p3_bakeoff_runner.py` | plan-only orchestrator; `--execute --i-have-operator-grant` gated |
| `scripts/benchmark/p3_bakeoff_critic_score.py` | deterministic co-critic replay scorer |
| `scripts/benchmark/p3_bakeoff_report.py` | paired McNemar report per duty/suite |
| `scripts/benchmark/test_p3_bakeoff.py` | 33 tests (all zero-inference) |
| `artifacts/p3-shadow-bakeoff-20260728/manifest/` | pinned manifest + critic tasks + sha256 sidecars |
