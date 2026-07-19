# RM-2.fast Reviewer Slate — C-CRAB P-REV-1

Date: 2026-07-19

Protocol: P-REV-1, attestation `MEASUREMENT-P-REV1-OPERATOR-APPROVED-20260719`.

Corpus slice: `docs/data/rm2_reviewer_slate_ccrab_p_rev1_matched_row_ids_20260719.txt`, copied exactly from the GLM P-REV-1 plan (`24` hard accept controls + `24` hard negatives; GC-shadow-repair4b.2c).

All non-GLM arms used `/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server`, MI210 `ROCm0`, `-ngl 99`, `-fa on`, `temperature=0`, JSON schema grammar on the final reviewer call, and the same C-CRAB patch-review prompt family as the GLM run.

| Arm | Artifact | FA | FR | AUC | ECE | Brier | Parse | Median row wall |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GLM-5.2 UD-IQ2_M CPU baseline | `data/glm52_reviewer_corpus_direct/gc-shadow-repair4b-p-rev1-20260719T132459Z` | 41.7% | 25.0% | 0.509 | 0.239 | 0.278 | 0.0% | 121.7s |
| Qwen3.6-27B dense Q8 reviewer | `data/reviewer_model_ablations/rm2-fast-qwen36-27b-q8-ccrab-p-rev1-20260719T162109Z` | 54.2% | 16.7% | 0.503 | 0.316 | 0.329 | 0.0% | 6.2s |
| Qwable-v1 IQ4_XS standalone reviewer | `data/reviewer_model_ablations/rm2-fast-b-qwable-iq4xs-ccrab-p-rev1-20260719T162712Z` | 54.2% | 45.8% | 0.438 | 0.441 | 0.448 | 0.0% | 2.1s |
| Qwen3.6-27B dense Q8 + Qwable IQ4_XS scaffold | `data/reviewer_model_ablations/rm2-fast-b-qwen36-27b-q8-plus-qwable-iq4xs-scaffold-ccrab-p-rev1-20260719T162958Z` | 33.3% | 41.7% | 0.659 | 0.315 | 0.325 | 0.0% | 6.5s |

Verdict:

- Qwen3.6-27B dense Q8 is much faster than GLM, but it over-approves (`FA 54.2%`) and is still random by AUC (`0.503`).
- Qwable standalone is fastest but not a reviewer candidate on this patch-review slice (`AUC 0.438`, `FR 45.8%`).
- Qwen+Qwable scaffold is the only arm with better FA and AUC than GLM (`FA 33.3%`, `AUC 0.659`), but it pays with `FR 41.7%` and lower raw correctness than GLM on the matched 48 rows. It is a repair hypothesis, not a role-ready reviewer.
- No tested small/fast arm cleanly beats GLM as a production reviewer. The reviewer choice remains open; do not decouple v7 on the basis of this slate alone.

Implementation notes:

- `scripts/benchmark/glm52_reviewer_corpus_direct_runner.py` now supports explicit single-GGUF reviewer models, MI210 device/offload flags, and an optional scaffold sidecar while keeping the final reviewer ledger/schema path unchanged.
- The runner now defaults `--era p_rev1_attested` when `--measurement-protocol p_rev1` is selected and no explicit era is supplied.
- Calibration reports must be generated with `--run-manifest <artifact>/run_manifest.json` to carry the decision-grade P-REV-1 stamp.
