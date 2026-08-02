# ternary q2 g64 smoke — 20260717T113848Z

Measurement evidence migrated out of scratch on **2026-08-02**. The master registry
cited these artifacts at their `/mnt/raid0/llm/tmp/` paths, so a routine sweep of that
directory would have left ratified, production-affecting claims with nothing behind
them. Copied byte-for-byte (sha256-verified both ends); the scratch originals were
left in place.

| | |
|---|---|
| scratch origin | `/mnt/raid0/llm/tmp/ternary-q2-g64-smoke-20260717T113848Z` |
| measured (file mtimes, UTC) | 2026-07-17 11:38 .. 2026-07-17 11:39 |
| migrated | 2026-08-02 |
| carried | 9 files, 2,035 bytes |

## Registry claims this backs

`orchestration/model_registry.yaml` — these citations resolve to this directory.
The YAML key path is the stable reference; line numbers are as of 2026-08-02.

- **L8301** &nbsp;`roles.ternary_bonsai_27b_q2_g64.performance.runtime_observation`
  > data/ternary_q2_g64_smoke_20260717T113848Z/ loaded
- **L8357** &nbsp;`roles.ternary_bonsai_27b_q2_g64.performance.evidence`
  > - data/ternary_q2_g64_smoke_20260717T113848Z/cpu.stdout
- **L8358** &nbsp;`roles.ternary_bonsai_27b_q2_g64.performance.evidence`
  > - data/ternary_q2_g64_smoke_20260717T113848Z/mi210.stdout
- **L8359** &nbsp;`roles.ternary_bonsai_27b_q2_g64.performance.evidence`
  > - data/ternary_q2_g64_smoke_20260717T113848Z/version.pinned.txt

## Integrity

`SHA256SUMS` lists every carried file, hashed after the copy and compared against the
scratch original. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/ternary_q2_g64_smoke_20260717T113848Z/SHA256SUMS
```

