# Worker General v6 IQK Parity, Full Port Matched 206

Date: 2026-06-28 UTC
Role/port: worker_general / 8072
Scoring: AA Omniscience deterministic F1
Matched questions: 206

| Arm | Run | GGML_IQK | Accuracy | Avg F1 | Hallucination | OI | Avg t/s |
|---|---|---:|---:|---:|---:|---:|---:|
| off | 20260628_042905 | 0 | 0.111650 | 0.236005 | 0.601093 | 0.255279 | 27.7753 |
| on | 20260628_044706 | 1 | 0.111650 | 0.244371 | 0.590164 | 0.260743 | 38.4640 |

Deltas, on minus off:

- Accuracy: +0.000000
- Avg F1: +0.008365
- Hallucination rate: -0.010929
- Omniscience Index: +0.005464
- Avg tokens/sec: +10.6887 (1.3848x)

Paired correctness: both correct 17, IQK-off only 6, IQK-on only 6, both non-correct 177; exact two-sided binomial p=1.000000.

Attestations:

- IQK off: /mnt/raid0/llm/tmp/attest_armB_full_iqk_off_20260628T042726Z.json
- IQK on: /mnt/raid0/llm/tmp/attest_armC_full_iqk_on_20260628T044652Z.json

Notes: both arms were intentionally stopped after exceeding N>=200. The comparison uses only common question IDs. Attestation still warns on unrelated registry gaps, missing AUTOPILOT_TOOL_SENTINELS, and stale llama.cpp GitNexus indexing.
