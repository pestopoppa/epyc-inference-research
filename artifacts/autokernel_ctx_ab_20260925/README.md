# AutoKernel planner seat: context placement A/B (inline vs variable), 2026-09-25

Root task: INF-78 OAB-9 (`handoffs/active/autokernel-orchestrator-actor-backend.md`), run by main-ak-seat
between DS41 run 9's stop (01:30Z) and run 9b's launch (04:04Z).

**Verdict: INLINE wins both pairs on every cost metric. The campaign keeps the default `inline` mode.**
n = 2 pairs, so this is a direction, not a magnitude.

## Setup

- **Seat**: plain opencode seat (`ActorSeat(bounded=False)`, bare `opencode run --auto`, opencode 1.18.31), the
  campaign default. Only the CONTEXT PLACEMENT varies:
  - `inline`: the whole rendered bundle is in the prompt (79,890 chars, sha256 `a265a03a…`). It is run 8's recorded
    planner prompt plus the `node_profile` section that `render_context` now prints (DS41-C20e fix, research
    `8a9d2a40`).
  - `variable`: the same bundle is written to `actor-context/<call>/` (`sections/`, `json/`, `INDEX.md`,
    `manifest.json`), and the prompt carries only the index (18,055 chars).
- **Code**: research `30631761` (integration lane `lane/ak-planner-integ-20260924`, merged to research main). Both
  arms go through the real `AgentPlanner.propose`, with two patches: the context is pinned to the control bundle,
  and there is one attempt with no retry. Details are in `driver.py`'s docstring.
- **Server**: :8083 Qwen3.8-27B Q8_0 on the MI210, one server generation for all 4 calls (pid 2009477). It ran with
  `-kvu`, `n_ctx_slot` 196,608 and 4 slots, build `b10303-ffc1bac82`.
- **Order**: ABAB (inline, variable, inline, variable), started 01:28:40Z. Pair 3 did not fit the 3.5 h budget: the
  driver needed 71 min and stopped.
- **Workspace**: `/mnt/raid0/llm/tmp/ak-seat-ab/lane`, clean at `ebb68dc55`, the run-8 target the prompt describes.
  No arm edited the lane.

## Results

| metric | inline p1 | inline p2 | variable p1 | variable p2 |
|---|---|---|---|---|
| wall (min) | 37.6 | 28.7 | 54.4 | 33.6 |
| steps | 23 | 24 | 49 | 29 |
| tool calls | 25 | 26 | 54 | 35 |
| tools by name | bash 22, read 3 | bash 21, grep 1, read 4 | bash 35, read 19 | bash 26, grep 1, read 8 |
| decoded tokens | 52.9k | 43.8k | 68.9k | 47.7k |
| first-step context | 40.6k | 40.6k | 19.1k | 19.1k |
| peak context | 104.0k | 91.5k | 163.1k | 118.0k |
| compactions | 0 | 0 | 0 | 0 |
| tool output chars | 65.2k | 33.8k | 204.1k | 136.5k |
| schema-valid | yes | yes | yes | yes |
| reply | abstain | hypothesis `akm-q4k-x4-avx512` | abstain | abstain |
| bundle tool calls | n/a | n/a | 7 | 6 |
| other-slot busy samples | 4 of 75 | 2 of 57 | 1 of 106 | 1 of 66 |

**Variable arm, bundle use.** The arm used the bundle as designed:
- all three required files were read (`03-program_strategy`, `05-node_profile`, `09-inbox`);
- `INDEX.md` was never re-read;
- no greps ran on the bundle.

However, it read **all six file-only sections** in both pairs (`01-target`, `03`, `05`, `07-shared_history`,
`08-serving_observations`, `09`). That is everything the inline prompt carries. On top of that, it explored the
lane SOURCE more than inline did: +29 and +9 tool calls, and 3-4x the tool output.

## Predictions (written before the run, INF-78 §Techniques learned → 1) against results

| prediction | result |
|---|---|
| first-step context 39.5k → ~18.7k | confirmed: 40.6k → 19.1k |
| peak 93.5k → ~79k | refuted: variable's peak was HIGHER (163k/118k vs 104k/92k) |
| 0 compactions instead of 1 | moot: on the `-kvu` 196k slot inline no longer compacts either |
| modest wall gain | refuted: variable was slower in both pairs (+45%, +17%) |

## Caveats

- **n = 2, and the order is fixed.** Inline always ran first within a pair. Both arms were faster in pair 2, which
  looks like a warm prefix cache or drift, so the magnitudes are soft. The direction held in both pairs on wall,
  steps, tool calls, decoded tokens and peak context.
- **Planner only.** Critic acceptance was not measured. Reply quality was not compared: 3 abstains and 1
  hypothesis, all schema-valid.
- **Per-tool latency is unknown.** `opencode export` carries no per-tool or per-step wall time.
- **One model, one seat.** A model that pulls precisely, or a scaffold that caps pulls (INF-78 OAB-12), could
  reverse the result. So could a smaller slot, where inline compacts again. A 3rd pair would tighten the
  magnitude, but it is unlikely to flip a 2/2 result that held on every cost metric.
- **Not comparable to DS41-C20c or run 8**, which ran on the split-KV 98,304-token slot. The two arms here are
  comparable to each other.
- **Contention was negligible**, at 6 of 132 samples (inline) and 2 of 172 samples (variable) with another :8083
  slot busy.

## Files

| file | what |
|---|---|
| `driver.py` | A/B driver (dry-run / run / summarize) |
| `driver-run.log` | the run's console log plus the paired summary |
| `results.jsonl` | one row per call: metrics, server fingerprint, `/slots` samples, bundle access, reply |
| `result-p{1,2}-{inline,variable}.json` | per-call result files |
| `actor-calls.jsonl` | the seat's `actor_call_metrics.v1` + `actor_call.v1` lines for the 4 calls |

These stay on local disk and are **not committed**, because of size:
- `/mnt/raid0/llm/tmp/ak-ctx-ab/actor-replies/`: opencode exports and stdout/stderr per call (2.3 MB);
- `/mnt/raid0/llm/tmp/ak-ctx-ab/actor-context/`: the two variable-arm bundles (564 KB);
- `/mnt/raid0/llm/tmp/ak-ctx-ab/dryrun/`;
- `lane`, a symlink to `/mnt/raid0/llm/tmp/ak-seat-ab/lane`.
