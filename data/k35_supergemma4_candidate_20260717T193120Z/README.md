# k35 supergemma4 candidate — 20260717T193120Z

Measurement evidence migrated out of scratch on **2026-08-02**. The master registry
cited these artifacts at their `/mnt/raid0/llm/tmp/` paths, so a routine sweep of that
directory would have left ratified, production-affecting claims with nothing behind
them. Copied byte-for-byte (sha256-verified both ends); the scratch originals were
left in place.

| | |
|---|---|
| scratch origin | `/mnt/raid0/llm/tmp/k35-supergemma4-candidate-20260717T193120Z` |
| measured (file mtimes, UTC) | 2026-07-17 19:31 .. 2026-07-17 19:32 |
| migrated | 2026-08-02 |
| carried | 19 files, 133,077 bytes |

## Registry claims this backs

`orchestration/model_registry.yaml` — these citations resolve to this directory.
The YAML key path is the stable reference; line numbers are as of 2026-08-02.

- **L4641** &nbsp;`roles.supergemma4_26b_mm_q8.performance.k35_vision_observation`
  > k35_vision_observation: "2026-07-17 K35 fixed OCR/chart candidate gate at data/k35_supergemma4_candidate_20260717T193120Z/: SuperGemma4-26B multimodal Q8_0 + F16 projector passed CPU (`4/4`, 25.58-31.76 t/s decode, ~26.4 GiB PSS) and MI210 (`4/4`, 80.35-83.87 t/s decode, ~42% VRAM). Quality-clean...

## Integrity

`SHA256SUMS` lists every carried file, hashed after the copy and compared against the
scratch original. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/k35_supergemma4_candidate_20260717T193120Z/SHA256SUMS
```

