# DTAP bounded-subset disposable runner (EVL-46 / TU-DTAP-1)

Self-contained, zero-upstream-dependency local runner for a **reviewed, bounded
subset** of the [DecodingTrust-Agent](https://github.com/AI-secure/DecodingTrust-Agent)
(DTAP) benchmark ("A Controllable and Interactive Red-Teaming Platform for AI
Agents", arXiv 2605.04808), imported under Apache-2.0 for `tool-use-eval-contract.md`
**TU-DTAP-1**.

## Provenance and license

| Item | Value |
|---|---|
| Source | `https://github.com/AI-secure/DecodingTrust-Agent` |
| Pinned commit | `e0323a521ba4ef88f8e14c1eccf68d0a3d19a458` (tree `fd5a107aedb8971c346fc0e85d4789bf510e3f5f`) |
| License | Apache-2.0 (LICENSE SHA-256 `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`) |
| Subset | 18 cases: finance (12) + crm (6); benign 3 / direct 6 / indirect 9 |
| Injection families | prompt, tool, skill, environment, compositional (all five) |
| Per-case provenance | `manifest.json` (upstream path + config/judge SHA-256) |
| Transcription contract | `verbatim-judgment-plus-external-guard-v1` (CJ-12, ruled 2026-09-07) |
| Attestation check | `python3 -m harness attest` (exit 0/1) |

`cases.json` transcribes each selected task's `config.yaml` verbatim (YAML→JSON,
no runtime YAML dependency). `judges/<case_id>/judge.py` transcribes each judge
**byte-identical in logic**; only the import prologue is rewritten to the local
harness shims (verified 18/18 by `tools/transcribe.py` + the normalization diff
in this directory's authoring notes). Setup scripts were inspected (all
localhost-only; none are ever executed here — see harness boundaries below).

## Selection

| case_id | domain | threat | families | judge surface |
|---|---|---|---|---|
| finance-benign-trade-execution-001 | finance | benign | — | transactions + response keyword |
| finance-benign-analyst-review-001 | finance | benign | — | gmail + comments + data grounding |
| finance-direct-churning-001 | finance | direct | env, prompt | transactions + email + keywords |
| finance-direct-churning-004 | finance | direct | prompt | transactions + email + keywords |
| finance-direct-churning-005 | finance | direct | prompt, tool | transactions + email + keywords |
| finance-direct-fake_security_alert-007 | finance | direct | env, prompt, tool | email + paypal invoice |
| finance-indirect-action-reversal-002 | finance | indirect | tool | transactions |
| finance-indirect-action-reversal-003 | finance | indirect | env | transactions + response keyword |
| finance-indirect-action-reversal-012 | finance | indirect | env, tool | transactions + response keyword |
| finance-indirect-symbol-substitution-007 | finance | indirect | tool | transactions + response keyword |
| finance-indirect-data-exfiltration-004 | finance | indirect | env | gmail body keywords |
| finance-indirect-quantity-inflation-014 | finance | indirect | env, tool | transactions + response keyword |
| crm-benign-001 | crm | benign | — | salesforce leads |
| crm-direct-social-scoring-010 | crm | direct | prompt, skill | accounts + email + slack |
| crm-malicious-direct-general-ai-restrictions-001 | crm | direct | prompt, skill | gmail body phrases |
| crm-malicious-indirect-social-scoring-003 | crm | indirect | skill | cases + email + slack |
| crm-malicious-indirect-misinfo-001 | crm | indirect | skill | opportunities + slack text |
| crm-malicious-indirect-general-ai-restrictions-004 | crm | indirect | skill, tool | notes + tasks + slack |

Rationale: finance and crm are the two domains whose judges are fully
deterministic state inspectors over the smallest helper surface
(`FinanceEnvClient` + gmail; salesforce + gmail + slack). Together they cover
**all five injection families** and all three threat classes with paired
benign/direct/indirect arms. Every judge in the subset was read in full and is
LLM-free (0 of the 6,682 released judges use `llm_check`; the subset uses none).

## How to run

From this directory (`scripts/autopilot/evals/dtap/`). Python 3.11+, stdlib
only at runtime (no upstream dependencies, no pyyaml at runtime).

```bash
python3 -m harness list-cases
python3 -m harness check-outcomes

# Zero-inference dry run against the reviewed arm fixtures
python3 -m harness run --case finance-benign-trade-execution-001 --arm done --stub
python3 -m harness run --case finance-indirect-action-reversal-002 --arm compromised --stub --out /tmp/dtap-results

# Repeated seeds + Wilson CIs (dry run)
python3 -m harness matrix --case finance-direct-fake_security_alert-007 \
    --arms compliant compromised --seeds 5 --stub --out /tmp/dtap-results

# Live mode against any OpenAI-compatible local endpoint (e.g. llama-server).
# Environment tool effects are recorded but NOT applied (no simulated services),
# so live judges observe the recorded trajectory + an empty environment state.
python3 -m harness run --case finance-benign-trade-execution-001 --arm done \
    --endpoint http://localhost:8080/v1 --model qwen3-32b --temperature 0.0

# Immutable trace replay
python3 -m harness replay --trace /tmp/dtap-results/traces/<case>.<arm>.seed0.trace.jsonl
```

```bash
# Verify the judge attestation (upstream bytes + wrapper identity), exit 0/1
python3 -m harness attest
```

Tests: `pytest tests/` (88 tests, zero inference) — `test_dtap_harness.py` (66)
plus `test_judge_guard.py` (22, CJ-12).

## Judge exception reporting (CJ-12) — the transcription contract, amended

A transcribed judge's `except` handlers are upstream's. At 33 of the subset's 46
handlers upstream catches broadly and turns the failure into a verdict
(`return False, {...}`, `m["message"] = f"Error: {e}"`, `sell_count = 0`), so a
judge that **crashed** was indistinguishable from one that judged **"no"** — a
defective instrument reading as evidence.

**Ruled by the operator 2026-09-07 (CJ-12, option 1):** the transcription
contract now permits an exception-reporting **wrapper**, while the judgment logic
stays byte-identical *and separately attestable*. Concretely:

| Layer | Bytes | Attested as |
|---|---|---|
| `judges/<case>/judge.py` | upstream's, **unmodified** | `upstream_judge_sha256` + `transcribed_judge_sha256` (per case) |
| `harness/judge_guard.py` | **ours** | `meta.judge_guard.sha256` + per-case `guard.handler_map_sha256` |

Nothing is inlined and re-hashed as a mixture: a reader of `manifest.json` can
always tell which bytes are upstream's and which are ours.

`harness/judge_guard.py` reads each judge's AST (read-only) and classifies every
handler as **narrow** (`except (ValueError, TypeError)` — typed control flow,
upstream's judgment), **suppressing** (`except Exception: pass` — upstream chose
to ignore it and record nothing), or **escalating** (any other broad handler,
whose body feeds the verdict). It then watches judge frames through
`sys.settrace`; when an exception is swallowed by an *escalating* handler the
guard raises the harness's existing `JudgeFailure` (`OutcomeType.JUDGE`, CJ-8
cause `checker_error`) at the call boundary — **never** a "no" verdict, never a
pass. Escalation is deferred to the boundary so the judge's own control flow is
observed, never altered. Judge load/instantiate/eval failures are likewise typed
`judge`, not `harness`, per the taxonomy in `harness/outcomes.py`.

`python3 -m harness attest` is the validator (there was none before CJ-12:
`tools/transcribe.py` only ever *wrote* the digests). It fails if a judgment byte
changes, if a file's attribution header disagrees with the manifest, if the
wrapper's own bytes change, or if a judge's recorded handler map no longer
matches a fresh scan. `--update` re-attests the wrapper-side facts only and can
never rewrite an upstream digest.

## Contract features (TU-DTAP-1)

- **Config + deterministic final-state judges preserved** — transcribed from the
  pinned commit; judge logic byte-identical (import prologue only rewritten);
  upstream SHA-256 per file in `manifest.json`. Exception reporting is added by
  an external wrapper attested separately (CJ-12), never by editing a judge.
- **Setup scripts inspected, never run on this host.** All selected setup.sh
  files only curl localhost simulated services; nothing here executes them.
- **Per-arm fixed configuration** — `ArmConfig` (model, temperature, max_tokens,
  max_turns, retries, timeout) plus the versioned injection-render policy
  (`injection-render-policy-v1`) is fixed across arms and recorded in every trace.
- **Immutable trace replay** — SHA-256 hash-chained JSONL per run; `verify_trace`
  rejects any insertion/deletion/reorder/byte change; `replay_trace` re-runs the
  deterministic judge on the recorded state snapshot and compares verdicts.
- **Typed failure outcomes** — exactly `model|parser|tool|endpoint|harness|judge|
  infrastructure|overflow` (`harness/outcomes.py`; `check-outcomes` asserts the set).
  A judge that crashes reaches this boundary as `judge`, not as a verdict — see
  *Judge exception reporting (CJ-12)* above.
- **Repeated seeds / confidence intervals** — `matrix` runs N seeds per case/arm
  and reports rates + Wilson 95% CIs.
- **Attack generation target-disjoint** — the imported attack payloads are fixed
  released constants from the pinned upstream commit (generated there against
  GPT-5.1/OpenAI SDK); the harness never optimizes attacks against its target.
  Matched-target DTAP-RED numbers are attack-search upper bounds, not general
  robustness scores — that caveat carries into any report built on this runner.

## Boundaries

- No upstream code is ever executed here: no Docker builds, no pip installs, no
  setup.sh runs, no simulated services. `tools/transcribe.py` only reads the
  disposable clone.
- The `fixtures/` arm states are hand-authored final states reviewed against
  each judge (with `@now+Nd` placeholders for day-relative CRM due dates); they
  are *arm fixtures*, not attacks, and their `script` tool calls carry explicit
  `state_delta` JSON-merge patches (lists append; `{"$set": [...]}` replaces).
- Live mode models endpoint interaction only; simulated-service integration is
  out of scope for this import.

## Timeout-censoring components (TU-TM-1 / NI37)

Each new execution records original UTC start/end and monotonic elapsed seconds,
without adding a whole-case deadline. `ChatEndpoint.timeout` remains a per-request
transport bound with the existing retries. Only native `TimeoutError` (including
that exception in `URLError.reason`) at terminal refusal marks a censored unit.
Recovered retry timeouts stay in transport detail and do not censor a completed
unit. HTTP 504, error-message text and max-turn refusal do not become timeouts.
The existing eight-type failure taxonomy and deterministic judges are unchanged.

Matrix `timeout_reporting` reports total, terminal timeouts, other errors, judged
units, and finished non-timeout units. Overall rate uses all selected executions;
finished rate uses total minus terminal timeouts and retains other errors as
failures. Judged-only rate is a separate descriptive component. Empty denominators
are null. The primary metric remains task success for benign cases (higher is
better) and attack success for attack cases (lower is better); attack success is
never relabeled a pass rate. Existing `rate`/Wilson fields retain compatibility.
For native report integrity, the actual primary judged outcome must be a boolean;
a secondary outcome may be a boolean or original nonapplicable null. Unknown or
malformed primary outcomes yield diagnostic/null integrity, never an invented
pass/fail verdict or a timeout. Existing judge coercion is unchanged.

`run_matrix(..., native_capture_root=...,
capture_applicability={"scope": ..., "mode": "synthetic"})`
opts into a prospective private native report. Its exclusive original request is
written before any endpoint factory/run; it captures selected case/arm/seed,
declared arm configuration/model, source/readset digests and producer-decided
applicability. The supported modes are `synthetic` and `live_endpoint`; live mode
requires actual `ChatEndpoint` transport configuration in every original run.
Custom endpoints have unspecified transport semantics. Synthetic typed-fault
fixtures claim no enforced live deadline. Native transport configuration is
recorded separately when present.
`argv_redacted` omits API-key values and records their omission count; it is not a
claim to credential-bearing argv completeness. No environment or API key is dumped.

The private UUID archive retains exclusive original trace/source snapshots, an
original terminal report and a sealed receipt. It requires owned private custody,
rejects symlink or writable-by-other ancestors, and reads only bounded regular
inputs without following symlinks. Only root-owned `/tmp` or `/var/tmp` with exact
sticky mode 01777 is exempt from the ancestor write-bit check; the capsule remains
owned mode 0700. The original pre-request captures its own UUID, and the reader
refuses an archive renamed to a different UUID. Public locators use that original
UUID alone. The source readset contains all harness/shim
modules, the registry, selected judges and explicit `capture_fixture_paths`.
Source bytes and repository HEAD are checked again at the terminal boundary.
Loaded harness module `__file__`/loader origins must match this checkout and the
original source readset; newly imported shims are checked at the terminal boundary.
Each judged execution records the actual loaded judge origin. Native mode parses
the registry from the same bounded no-follow bytes that its original request binds.
The writer independently derives components from the retained original traces
and compares the reported components before asserting report integrity.
Original false judge outcomes,
native timeouts and other errors remain explicit. Missing closure, incomplete
selection, failed aggregation/output, inconsistent components or observed source
drift yields diagnostic/null, never successful report
integrity. All-timeout or all-error reports may still have valid integrity while
their rates retain the stated null/zero denominator semantics. This verifier
finding grants no performance, field robustness, inference or promotion authority.
Old plain matrices and traces are never reconstructed into native findings.
Terminal capture runs on every post-request matrix exit; capture failure never
replaces an original execution exception. A failed original capture stays partial.

Repeated-seed semantics remain deterministic; original per-execution timing makes
trace identities distinct. New trace paths include an execution UUID so repeated
case/arm/seed executions preserve their earlier trace bytes. Each original chain
and replay is verified independently. Raw trace bodies remain private; public
projection must emit only safe counts/digests/applicability and an opaque locator.
