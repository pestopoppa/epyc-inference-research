# np_context study — v8 (2026-07-27)

**What this is.** Raw evidence for the GPU `np` × context batching surface study run on
`production-consolidated-v8`, plus the decision surface rendered from it.

**When measured.** 2026-07-27 (bundle timestamped `20260727`). Its predecessor bundle,
`../np_context_study_20260723/`, holds the earlier 2026-07-23 run of the same study.

**Which claim it backs.** The per-arm `np`/context batching policy for the 122B IQ2 GPU lineup —
specifically that the collapse and `np=2` dip are A1-specific rather than general, so `np` policy is
set per arm rather than globally. It is discovery/decision evidence, not a ratified production claim:
nothing here carries a P-GPU-1 attestation.

**Contents.**

| Path | What |
|---|---|
| `np_context_v8_decision.html` | the rendered decision surface (SHA-256 `816ad5cd…`) |
| `A3_*`, `A4_*`, `Laguna_*` | per-arm run directories: `argv`, `pid`, `server.stdout`, `server.stderr`, result JSON, and per-question JSONL |
| `driver/` | the scripts that produced the runs |
| `logs/` | driver-level logs |
| `*.incomplete-*`, `*.interrupted-*`, `*.invalid-*` | preserved failed/aborted attempts, kept deliberately — they are the record of what did not work and why, and are **not** valid evidence for any claim |
| `SHA256SUMS` | hashes for every file in this bundle except `__pycache__` |

**Durability class.** Carried in git. At 47 MB of text across ~1,700 files this is well inside the
"too large to carry" carve-out in `MEASUREMENT.md` §5, so it is carried rather than recorded
hash-and-provenance-only. `__pycache__` and `*.pyc` are excluded from both the hashes and the repo.

**Provenance note.** `np_context_v8_decision.html` previously lived only at
`/mnt/raid0/llm/tmp/claude-artifacts/` and was cited from that scratch path. Under the
2026-08-02 amendment to `MEASUREMENT.md` §5 — *"Evidence must be DURABLE, not merely hashed"* —
a scratch path may not be the citation of record. The file was copied here on 2026-08-02 and verified
byte-identical to the original; cite this path, not the scratch one.
