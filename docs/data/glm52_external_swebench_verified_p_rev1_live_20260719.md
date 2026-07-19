# GLM-5.2 SWE-Bench-Verified P-REV-1 Live Gate (2026-07-19)

## Run

- Model: `glm_52_ud_iq2m`
- Binary: `/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server`
- Rows: `docs/data/glm52_external_swebench_verified_n24_rows_20260719.jsonl`
- Live artifact: `data/glm52_external_ground_truth_direct/glm52-external-swebench-verified-n24-p-rev1-20260719Tlive/`
- Protocol: `p_rev1`
- Attestation: `MEASUREMENT-P-REV1-OPERATOR-APPROVED-20260719`
- Era: `p_rev1_attested`

## Result

The live CPU-only GLM run completed `24` SWE-Bench-Verified accept-control rows. All rows are known-good patches under the SWE-Bench fail-to-pass/pass-to-pass oracle, so this slice measures false rejects, not false accepts.

| Metric | Value |
|---|---:|
| Correct approvals | `22/24` |
| False rejects | `2/24` |
| FR | `8.3%` |
| Parse failures | `0/24` |
| Approve / reject | `22 / 2` |
| Elapsed | `1965.612s` |
| Median row latency | `69.236s` |
| Max row latency | `196.198s` |
| Median prompt tokens | `1194.5` |
| Max prompt tokens | `3431` |
| Server prompt/decode tail | `20.21 / 2.79 t/s` |

False rejects:

- `glm52-swebench-verified:3ab2791b333e454dcfe6` (`django__django-12663`): GLM rejected with confidence `0.9`, claiming the patch did not address `SimpleLazyObject` integer conversion.
- `glm52-swebench-verified:1ff5b0646b7d0585cb5e` (`pylint-dev__pylint-8898`): GLM rejected with confidence `0.9`, claiming missing test coverage for regex comma handling.

This is positive accept-control evidence and materially better than the C-CRAB P-REV-1 accept side, but it does not clear GLM as a patch-reviewer because the decision-grade C-CRAB hard-negative/accept-control run still failed overall (`FA 41.7%`, `FR 25.0%`, `AUC 0.509`). GLM remains research-only pending a new repair hypothesis or a policy decision that scopes GLM away from hard-negative patch-review.

## Cleanup

The runner stopped the server and recorded `post_processes=[]`. Follow-up checks found no `llama-server`, no AutoPilot process, and no KFD PIDs.

Raw prompt/request/response artifacts and the full server log are local in the live artifact directory. The committed evidence is `plan.json`, `progress.jsonl`, `decisions.jsonl`, `run_manifest.json`, `summary.json`, and this report.
