# AGG-1.3 predeclared - amendment 1 (2026-11-11, written BEFORE any AGG-1.3 data and before the compute row was published)

Amends `PREDECLARED_AGG13.md` (sha256 `fde0cb0d4bea0fc816e1393b38533c82c766853330d1218d2b0b37c3c5607183`, committed and
pushed as research `81642408c`, **not edited**). Every change below was found while implementing the runner, with no model
call and no AGG-1.3 data in existence; none moves a gate threshold the box states. The amendment's own sha256 is recorded in
the compute row and the run spec.

1. **Bound arithmetic corrected (a slip in the original text).** 3/28 x 1.10 = 0.117857. Calls: 1,359 x 0.117857 =
   **160.17**, so the gate is **<= 160 calls** (unchanged as an integer). Prefill: 1,712,985 x 0.117857 = **201,887.5**, so
   the gate is **<= 201,887 prefill tokens** (the original said 201,892.6, an arithmetic error of 5 tokens). The plain
   3/28 shares (145.6 calls, 183,534 tokens) and the AGG-1.1 pilot comparison (146 calls = 91.2% of the call bound,
   185,976 tokens = 92.1% of the prefill bound) are as written. The runner computes the bound from the box formula, not from
   these figures.
2. **G1 evidence for "re-extracted".** A changed document counts as re-extracted when its L3 stage ran (`l3_ran`, its L3
   facts written by `doc_l3_facts`) and its 8 L4 units ran; a changed document that happened to yield zero L3 facts would
   still count (the original wording "L3 facts written" is read as "L3 extraction run"). A retained document must show
   `l3_ran` false, zero L3 facts, zero L4 units, zero LLM calls. A resumed entry (the job was killed between the store commit
   and its record) re-counts a blob in the job's own `out/cache` as that document's real parse; a blob in a seed cache dir is
   a cache hit and fails G1 for a changed document.
3. **G4's regression check is enforced.** "Reported separately for the 25 retained documents (must equal the AGG-1.1 hits)"
   is a gate input: if the AGG-1.1 `m1` record's per-document `found` counts differ for any retained document, G4 trips even
   when overall M1 >= 0.75 (a retained document's facts changed, which contradicts "skipped by hash").
4. **Retained-document hash guard.** Before `ingest`, the runner compares the cached parse's content hash with the store's
   hash for that slot; a mismatch is a G1 failure raised BEFORE the store is touched (an `ingest` of a different hash would
   replace the retained document and drop its facts). On the real store all 25 caches hash equal to the store (zero-compute
   preflight check `base_hash_check`: 25 equal, 0 unequal, 0 without cache).
5. **Store feature pin.** The supersession ledger (`facts_superseded`, `documents_superseded`, `IngestReceipt.superseded`)
   is orchestrator commit `6bf51c12e606` (store + planner, one commit; the planner part is AGG-1.4's). The job's `orch/`
   is that commit's `src/` tree, sealed by `SOURCE.sha256`.
6. **Smoke run.** `FILLER_SMOKE=1` runs preflight, the 25 hash checks, the first changed document's parse, ingest and L3,
   and ONE real L4 unit (`l4:revenue:MICROSOFT_2022_10K`), then exits 0; the main run resumes from there without repeating
   any unit.
