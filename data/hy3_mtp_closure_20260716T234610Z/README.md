# hy3 mtp closure — 20260716T234610Z

Measurement evidence migrated out of scratch on **2026-08-02**. The master registry
cited these artifacts at their `/mnt/raid0/llm/tmp/` paths, so a routine sweep of that
directory would have left ratified, production-affecting claims with nothing behind
them. Copied byte-for-byte (sha256-verified both ends); the scratch originals were
left in place.

| | |
|---|---|
| scratch origin | `/mnt/raid0/llm/tmp/hy3-mtp-closure-20260716T234610Z` |
| measured (file mtimes, UTC) | 2026-07-16 23:46 .. 2026-07-16 23:50 |
| migrated | 2026-08-02 |
| carried | 29 files, 18,707 bytes |

## Registry claims this backs

_No direct citation resolved to this directory at migration time._

## Integrity

`SHA256SUMS` lists every carried file, hashed after the copy and compared against the
scratch original. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/hy3_mtp_closure_20260716T234610Z/SHA256SUMS
```

