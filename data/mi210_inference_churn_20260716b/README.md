# mi210 inference churn — 20260716b

Measurement evidence migrated out of scratch on **2026-08-02**. The master registry
cited these artifacts at their `/mnt/raid0/llm/tmp/` paths, so a routine sweep of that
directory would have left ratified, production-affecting claims with nothing behind
them. Copied byte-for-byte (sha256-verified both ends); the scratch originals were
left in place.

| | |
|---|---|
| scratch origin | `/mnt/raid0/llm/tmp/mi210-inference-churn-20260716b` |
| measured (file mtimes, UTC) | 2026-07-16 20:29 .. 2026-07-16 20:36 |
| migrated | 2026-08-02 |
| carried | 50 files, 1,778,806 bytes |

## Registry claims this backs

`orchestration/model_registry.yaml` — these citations resolve to this directory.
The YAML key path is the stable reference; line numbers are as of 2026-08-02.

- **L8252** &nbsp;`roles.ternary_bonsai_27b_q2_0.performance.evidence`
  > - data/mi210_inference_churn_20260716b/ternary_bonsai_q2_0_mi210_v7.log
- **L8412** &nbsp;`roles.bonsai_8b_local_orphan.performance.evidence`
  > - data/mi210_inference_churn_20260716b/bonsai_8b_mi210_v7_final.log

## Integrity

`SHA256SUMS` lists every carried file, hashed after the copy and compared against the
scratch original. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/mi210_inference_churn_20260716b/SHA256SUMS
```


## WITHHELD FILES (2026-08-02) — all 13 `*.ps.txt` process captures

Every `*.ps.txt` in this campaign is **deliberately not committed**. They are full `ps`
listings taken before and after each run on a SHARED host, so they capture whatever else
was running: operator usernames and home-directory paths, docker invocations with their
environment, other tenants' command lines. The PII hook flagged them on two independent
rules (a possible AWS access key ID inside a `docker exec ... -e` line, and long digit
runs in crash-handler paths).

Withheld as a CLASS rather than file by file: a process listing from a shared machine
leaks host state by construction, and exempting them one at a time would have meant
re-deciding the same question 13 times and getting it wrong once.

They are pre/post-check captures, not measurement results — nothing in this campaign's
findings rests on them. Recorded hash-and-provenance-only per MEASUREMENT.md §5; hashes
are in the adjacent `*.WITHHELD.sha256` files, originals remain in scratch at
`/mnt/raid0/llm/tmp/mi210-inference-churn-20260716b/`.
