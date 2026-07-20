# K11 Gemma4 Free-Form Post-Candidate Gate

Date: 2026-07-20
Artifact: `data/k11_gemma4_determinism/k11_freeform_ud_iq4xs_mtp_n10_post_candidate_20260720T063000Z/`
Build: experimental post-candidate `12a292f0c21d` (`llama-server` reports `10099`)
Status: observation-grade; not frozen-v7 promotion evidence

## Run

Command shape:

```bash
python3 scripts/benchmark/k11_gemma4_determinism_runner.py \
  --execute \
  --runs 10 \
  --max-tokens 1024 \
  --threads 24 \
  --target-model /mnt/raid0/llm/models/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf \
  --draft-model /mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf \
  --spec-type draft-mtp \
  --prompt 'Return exactly 200 repetitions of the word benchmark separated by single spaces. Do not output any other text.' \
  --expected-word benchmark \
  --expected-word-count 200
```

This is the no-schema free-form K11.1 shape. A prior sidecar run in a misleading
`k11_broad_freeform...062300Z` directory used `--schema-task word-array-200`; that
schema-constrained artifact was moved to `/mnt/raid0/llm/tmp/superseded-k11-artifacts/`
and is not used for the K11.1 free-form verdict.

## Result

| Metric | Value |
|---|---:|
| Fresh-server runs | 10 |
| Task pass | 10/10 |
| Unique output hashes | 1 |
| Mean decode | 126.44 t/s |
| Decode range | 124.06-127.51 t/s |
| Draft accepted | 1330/1340 |
| Draft acceptance | 99.25% |

Every run emitted exactly 200 `benchmark` words and zero incorrect words. The
response content hash was stable across all 10 fresh servers.

Cleanup: runner exit code `0`; current post-run checks show no `llama-server`,
AutoPilot, or downloader process and no KFD PIDs.

## Interpretation

The post-candidate UD-IQ4_XS assistant-head MTP lane now passes the previously
open no-stop n=10 exact-count free-form task. This narrows K11.1 further:

- Schema-constrained structured output: already stable.
- No-stop exact-count free-form: now stable on the post-candidate build.
- Stop-string / termination semantics: still not repaired by this run.
- Quality retention against the production ORIG Q4_K_M worker: still a separate
  role-admission gate.

Do not use this as promotion evidence for frozen `6ad45fa3ff`; the measured binary
is post-candidate `12a292f0c21d`.
