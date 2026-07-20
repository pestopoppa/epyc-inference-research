# GLM-5.2 External JudgeBench-GPT P-REV-1 Live Gate (2026-07-19)

## Run

- Model: `glm_52_ud_iq2m`
- Binary: `/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server`
- Rows: `docs/data/glm52_external_judgebench_gpt_n24_rows_20260719.jsonl`
- Live artifact: `data/glm52_external_ground_truth_direct/glm52-external-judgebench-gpt-n24-p-rev1-20260719T154517Z/`
- Exact-choice rescore artifact: `data/glm52_external_ground_truth_direct/glm52-external-judgebench-gpt-n24-p-rev1-choice-rescore-20260719/`
- Protocol: `p_rev1`
- Attestation: `MEASUREMENT-P-REV1-OPERATOR-APPROVED-20260719`
- Era: `p_rev1_attested`

## Result

The live run completed `24` JudgeBench-GPT pairwise rows (`12` gold A, `12` gold B).

| Scoring view | Correct | Accuracy | Parse failures | Notes |
|---|---:|---:|---:|---|
| Original strict schema score | `15/24` | `62.5%` | `7/24` | Seven responses used 0-100 confidence values despite valid A/B decisions. |
| Exact-choice P-REV-1 rescore | `22/24` | `91.7%` | `0/24` | Scorer now normalizes 0-100 confidence as a warning, not a failed exact-choice decision. |

The exact-choice rescore records `confidence_warning_counts={"confidence_scale_0_100": 7}` and balanced final decisions (`12` A / `12` B). This is positive judge-native evidence for GLM's pairwise preference capability, but it does not clear patch-review admission because the same model already failed decision-grade C-CRAB P-REV-1 (`FA 41.7%`, `FR 25.0%`, `AUC 0.509`). Supersession: SWE-Bench-Verified live execution later completed with `22/24` correct approvals and FR `8.3%`, but that accept-control result still does not clear the C-CRAB hard-negative risk.

## Code Hygiene

The direct runner now defaults `--measurement-protocol p_rev1` runs to `era=p_rev1_attested`, preserves `observation_only=false` in score-only manifests, and carries confidence warnings through live and saved-response summaries. The pairwise prompt now explicitly asks for decimal confidence (`0.0` to `1.0`) to reduce future format warnings.

Follow-up no-inference cleanup fixed the score-only artifact so `summary.json` records `server.not_started=true` and `server.log_file=null` instead of a stale nonexistent server log path. Expanded raw live `artifacts/` and `logs/` remain local/untracked because the repository PII hook flags long digit runs in saved prompts/server logs; the committed decision-grade evidence is the summary, manifest, decisions, and plan.

Focused validation:

```bash
uv run --with pytest pytest -q scripts/benchmark/test_glm52_external_ground_truth_adapter.py scripts/benchmark/test_glm52_external_ground_truth_direct_runner.py scripts/benchmark/test_glm52_swebench_verified_adapter.py
python3 -m py_compile scripts/benchmark/glm52_external_ground_truth_adapter.py scripts/benchmark/glm52_external_ground_truth_direct_runner.py scripts/benchmark/glm52_swebench_verified_adapter.py
```

Result: `24 passed`; compile checks passed.
