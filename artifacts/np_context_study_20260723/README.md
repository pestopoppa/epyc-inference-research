# np_context study — 2026-07-23

**What this is.** Raw evidence for the first run of the GPU `np` × context batching surface study.
Superseded for decision purposes by `../np_context_study_v8_20260727/`, which re-ran the study on
`production-consolidated-v8`; retained because it is the primary record of the earlier run and
because the two together show how the surface moved across the kernel change.

**When measured.** 2026-07-23.

**Which claim it backs.** Nothing currently. It is historical evidence, observation-grade, and carries
no attestation. Read it as the prior state of the `np`/context surface, not as authority for a present
policy — the v8 bundle is the current record.

**Durability class.** Carried in git — 7.2 MB across ~440 text files, well inside the "too large to
carry" carve-out in `MEASUREMENT.md` §5. `SHA256SUMS` covers every file except `__pycache__`.

**Why it was untracked.** This bundle had **zero** files in git until 2026-08-02; its v8 sibling
tracked only its five driver scripts. Both were named in an AutoKernel design handoff as the durable
backing for a decision surface that was itself only in scratch — a hash chain whose every link was
one cleanup away from unverifiable. Tracked on 2026-08-02 to close that.
