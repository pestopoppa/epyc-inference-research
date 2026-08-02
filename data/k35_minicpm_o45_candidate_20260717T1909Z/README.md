# k35 minicpm o45 candidate — 20260717T1909Z

Measurement evidence migrated out of scratch on **2026-08-02**. The master registry
cited these artifacts at their `/mnt/raid0/llm/tmp/` paths, so a routine sweep of that
directory would have left ratified, production-affecting claims with nothing behind
them. Copied byte-for-byte (sha256-verified both ends); the scratch originals were
left in place.

| | |
|---|---|
| scratch origin | `/mnt/raid0/llm/tmp/k35-minicpm-o45-candidate-20260717T1909Z` |
| measured (file mtimes, UTC) | 2026-07-17 19:08 .. 2026-07-17 19:09 |
| migrated | 2026-08-02 |
| carried | 18 files, 129,012 bytes |

## Registry claims this backs

`orchestration/model_registry.yaml` — these citations resolve to this directory.
The YAML key path is the stable reference; line numbers are as of 2026-08-02.

- **L8986** &nbsp;`roles.minicpm_o_45_local_multimodal.performance.evidence`
  > - data/k35_minicpm_o45_candidate_20260717T1909Z/summary.json

## Integrity

`SHA256SUMS` lists every carried file, hashed after the copy and compared against the
scratch original. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/k35_minicpm_o45_candidate_20260717T1909Z/SHA256SUMS
```

