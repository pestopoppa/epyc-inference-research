# UFH-13 thesis runner: orchestrator vs. the strongest model alone

epyc-root handoff: `handoffs/active/thesis-experiment-orchestrator-vs-strongest-model.md` (TE-3 driver,
TE-5 run, TE-6 verdict). Orchestrator side: TE-1, flag `v1_escalation` plus the per-request key
`x_escalation` (epyc-orchestrator `9959e8db`, made opt-in by the follow-up commit).

**Written, not run.** `pilot` and `run` send inference; only the session that owns the window runs them.

## Arms (`arms.py`)

| arm | serves the answer | `/v1` body keys |
|---|---|---|
| A0 | the consultant alone (`architect_general`) | `x_force_role=architect_general`, `x_escalation=off` |
| A1 | frontdoor alone | `x_escalation=off` |
| A2 | frontdoor, escalating to the consultant | `x_escalation=architect_general` |

All arms use client tool mode with `x_show_routing`, temperature 0, seed 42 and max_tokens 16384.
`enable_thinking` is set by each role's registry `chat_template_kwargs`, not by the request body.
A2 names the consultant rather than using `auto`, because `coder_escalation` stays on the 27B after the role swap.

## Suite (`suite.py`)

`data/kernel-v8-candidate/quality-gate/run-20260725T204443Z-fullcontract-both-mode/questions.json` holds
MMLU-Pro 200 and GPQA 195 items. Its sha256 is `1532906b…adb1`, checked on every load, and the loader
refuses any drift. The scorer is `answer_scoring.extract_letter_answer`, the same for every arm; its
sha256 goes into the run manifest.

## Records (`records.py`)

`records.jsonl` gets one fsynced line per (arm, item), written as soon as the item finishes. A rerun
skips every item already on disk. Failures and timeouts are final and count as wrong; they are never
retried. Each record carries:

- the full escalation receipts (`receipts`);
- `escalation_enabled`, `escalation_triggers`, `escalation_to_roles`, `escalation_models`,
  `review_verdicts` and `final_answer_role`;
- `consultant_device_seconds`, `request_device_seconds` and `non_consultant_device_seconds`;
- `cost_problems`.

For A0, the consultant cost is the whole request, and the runner checks that the pinned role served it.
An unmeasured cost is recorded as `None`, never as 0.

## Commands (from the research repo root)

```bash
# zero network
python3 scripts/benchmark/thesis_ufh13/run_thesis.py plan

# SENDS INFERENCE. A2 only on N stratified items; writes pilot_report.json
python3 scripts/benchmark/thesis_ufh13/run_thesis.py pilot 20 \
    --out /mnt/raid0/llm/tmp/ufh13/pilot-<date> --run-id pilot-<date> --seed 7 \
    --orchestrator-commit <sha the live API serves>

# SENDS INFERENCE. A0/A1/A2 interleaved; rerun the same command to resume
python3 scripts/benchmark/thesis_ufh13/run_thesis.py run \
    --out /mnt/raid0/llm/tmp/ufh13/run-<date> --run-id run-<date> --seed 42 \
    --orchestrator-commit <sha>

# zero network
python3 scripts/benchmark/thesis_ufh13/run_thesis.py score --out /mnt/raid0/llm/tmp/ufh13/run-<date>
python3 scripts/benchmark/thesis_ufh13/run_thesis.py pilot-report --out /mnt/raid0/llm/tmp/ufh13/pilot-<date>

# tests (no inference)
python3 -m unittest scripts/benchmark/thesis_ufh13/tests/test_thesis_ufh13.py
```

`--transport opencode` runs each item through headless OpenCode, which is the pre-registered harness.
It passes the prompt on stdin and puts the arm's keys into the plugin's `staticKeys`. The receipts
come from the tap `v1_escalation` events, joined on the OpenCode session id. Two limits apply:

- It has been tested offline only.
- The plugin forwards only `x_*` keys, so it cannot pin seed or temperature. This must be settled
  before the TE-3 freeze.

## Pilot report

The pilot answers one question: does A2 escalate at all? It reports:

- `escalation_rate`, plus `items_by_trigger` for `review_gate`, `review_gate_revision` and `quality_escalation`;
- `review_verdicts`, where `wrong` means a revision and `ok_or_unavailable` means the frontdoor answer was kept;
- `escalation_to_roles`, `escalation_models` and `escalation_not_enabled`;
- consultant seconds.

Pilot items come from the frozen suite. The records are flagged `pilot: true`, and `score` refuses a
pilot run. Seeing A2's outputs on suite items before the freeze is a pre-registration deviation, so
log it as such if the pilot runs before TE-3.

## Scorer (`score.py`)

The pre-registered rule has these parts:

- `G = (Q_A2 − Q_A1)/(Q_A0 − Q_A1)` on pooled accuracy.
- `d = DS_A2/DS_A0` in consultant device-seconds.
- A paired bootstrap over items with 10,000 resamples, stratified by suite, with a recorded seed and
  percentile intervals. Resamples where G is undefined are dropped and counted.

The verdicts are evaluated in this order:

1. REFUTED if A2 is worse than A1 by more than one pooled quantum.
2. NO GAP if the CI of `Q_A0 − Q_A1` includes 0.
3. SUPPORTED if `G ≥ X`, `d ≤ Y` and the lower bound of G is `> d`.
4. REFUTED if the upper bound of G is `< d`.
5. INCONCLUSIVE otherwise.

X = 0.75 and Y = 0.50 are the proposed values; the operator confirms them at TE-0.

`score` writes `score.json` and `belief_measurements.jsonl`, whose rows use the field names of
`scripts/vidya/claim_tuple.py::ClaimTuple`. Their `protocol_id` is empty until a protocol is codified.
No adapter is registered yet (see the epyc-root `scripts/vidya/adapters/README.md` source table).
