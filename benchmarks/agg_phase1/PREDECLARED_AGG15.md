# AGG-1.5 - predeclared: mid-size retrieval parity, planner `retrieval` vs `context.search`

Written 2026-11-11 by the AGG-1.5 prep session, **before any AGG-1.5 model call or data exists** (the questions
below are derived by script from the frozen sheet; no model output has been read). Box:
`handoffs/active/knowledge-pipelines.md` AGG-1.5 (the box text is the identity of the task). Separate file from
`agg_phase0/PREDECLARED.md` (sha256 `976a5fe87e3db33d6a5e4547ee4bb6cae324f44d107980342b62f7b6212b8bc9`, untouched, a test pins
it) and from `agg_phase1/PREDECLARED.md` (AGG-1.1). A file cannot carry its own hash: this file's sha256 is recorded in the
AGG-1.5 compute row, the run spec and the runner's `--expect-predeclared-sha` (a mismatch fails preflight, rc 2). Any
mid-run fix needs a dated amendment BEFORE the rerun; one made after AGG-1.5 data exists is POST-DATA and says so in its
heading. Closure scope: the planner `retrieval` path against `context.search` on 4 mid-size FinanceBench FY2022 10-Ks, on
Qwen3.8-27B; it concludes nothing about other models, corpora or sizes, and nothing about dense (L2) retrieval.

## Question

On a document whose size puts it on the planner's `retrieval` path (200k < C <= 1M tokens), does reading pointers from the
store's L1 index (planner path) cost and answer the same as today's `context.search` (fresh per-request index), and does the
store-backed read avoid rebuilding the index per request?

## Corpus and questions (frozen inputs, derived by script, verified without a model)

- 4 documents of the 28 (`corpus.json`, sha256 `bdeb7a978f036a8e2599bc6db752899d5d18bd57f5fbb8ac52a6dd3a1da41f84`) with
  200k < C <= 1M tokens by the Phase 0 store's `n_tokens_est`; all four are Phase 0 documents, so the Phase 0 store is the
  source: JPMorgan 390,496 · PepsiCo 301,525 · American Water Works 258,302 · AES Corporation 231,844 tokens. The other 24
  documents are <= 200k or not in the Phase 0 store and are not used; no new parse, ingest or download.
- **20 single-fact questions**, one statement line each ("what amount is reported for `<line label>` in the consolidated
  financial statements for fiscal 2022, absolute value in USD millions"): JPMorgan 4, PepsiCo 5, American Water Works 5,
  AES 6. Selection rule (code: `parity_m5.select_cells`, salt `agg15-v1`): eligible = cells of `agg_phase1/facts_sheet.csv` with
  status `filled`, confidence `high`, `verified`, and a quote, on those 4 documents (26 cells); per document the first 5 by
  sha256(salt|doc|metric) (JPMorgan has only 4); the remaining slots come from the leftover pool in hash order, at most 6 per
  document. Order is round-robin by document, so the pilot (first 5) touches all four.
- **Golds** are the sheet's cells (Phase 0 cells: two independent readers; every cell also cross-checked by script in
  `facts_sheet_xcheck.json`). Re-verified for these 20 by `parity_m5.py --build-questions` with **no model**: the quote is
  found on its stated page in BOTH pypdfium2 and pypdf text, the gold equals a figure printed in the quote, and the quote is
  present in the stored chunk text of that document (reachable by retrieval). 20 of 20 verified
  (`parity_verification.json`). Only verified questions run; unverified = excluded, never assumed (excluded count: 0).
- `parity_questions.json` sha256 `234b7a23f8f6ff265ed702cf56112c5bde56fd632915a9786388386f0fd26e92`;
  `parity_verification.json` sha256 `a7ca7907abdd5443140da753486275c4877444598b3422b17576f99beba9bf53`.
- Scorer: the Phase 0 deterministic scorer unchanged (`phase0_arms.score_single` on the `FINAL:` line; numeric within
  max(0.5%, half a unit of the last decimal), scale words normalise to USD millions, magnitude compared, so the AES net loss (546)
  matches 546). No FINAL line or an unreadable value counts as not correct.

## Arms (same model loop, only the retrieval backend differs)

Both arms: the same SEARCH / READ / FINAL loop (grammar-constrained one action per call, prompt and caps identical,
<= 12 calls, search k = 8, a READ returns at most 12,000 characters (the Phase 0 `b1_read_chars`; stored chunks average 3.1k characters, so P reads a chunk whole), `id_slot` 0, prompt cache default), on one private
Qwen3.8-27B-Q8_0 + DFlash2 np1 196k server from the v10 GPU kernel store. One question at a time, both arms back to back.

| Arm | Retrieval backend |
|---|---|
| `P-retrieval` | `planner.scoped_facts` scopes the question to its one document, `planner.plan` must return `retrieval` (layers L0 L1 L2; the deterministic size default, the typed planner call stays the Phase 0 stub, no planner model call). Pointers = chunk ids from `IndexRegistry.get_or_build(corpus, 'l1_bm25')` over the stored chunk table of a parity store (one corpus per document, L0 chunks only). The index is keyed by (corpus, content hash, builder, model) and built once per document, then reused for that document's other questions. |
| `B-context.search` | today's flow: a fresh `ContextBundle` per request with the document as one section; `bundle.search` pointers `{section, start, end}`, read with `bundle.get`; the lazy `ContextSearchIndex` builds on the first search of each bundle, i.e. once per request (`context_bundle.py:580-636`). |

Dense retrieval is OFF in both arms (no embedding pool in a private job; the registry reserves `l2_dense` and raises), so
`context.search` runs its documented `lexical` mode and the planner's L2 pointers are not available: the comparison is lexical
L1 against a lexical per-request index. A dense L2 builder would need a dated amendment and a new row; this one does not test it.
Hit lists show a pointer and the first 200 characters of the pointed text in both arms; the READ unit differs (P: a stored chunk,
B: an 800-character span), which is part of what is measured: P reads whole chunks (about 4x the characters of a B span), so the `prefill` component is the one most likely to trip, and a trip there is a finding about the planner's read granularity, filed as such, not a harness defect.

## Gates (M5 parity; any failure -> `phase1_gate_tripped` naming the component; none -> AGG-1.5 DONE)

Tolerances are the Phase 0 `KILL` constants (imported, not copied). Over the 20 paired questions:

| Component | Holds when |
|---|---|
| `accuracy` | \|correct(P) - correct(B)\| <= 1 question |
| `calls` | max over questions of (calls(P) - calls(B)) <= +1 |
| `prefill` | max over questions of (prefill(P) - prefill(B)) <= +2,000 tokens (`timings.prompt_n`, tokens actually prefilled) |
| `no_l3_l4` | the parity store manifest, read after every P question, never lists an L2, L3 or L4 index; every question plans `retrieval` with a single-entity scope on a 200k < C <= 1M document (else stop `plan` with zero model calls) |
| `no_rebuild_per_request` | P-side index builds <= 1 per document over the whole run (4 builds for 20 questions). Reported beside it: B-side builds (expected 20) and the saving in index-build calls and in seconds (sum of B build seconds minus P build seconds, wall-measured around the first search / `get_or_build`) |

Reported, not gated: per-arm medians of calls and prefill, per-question wall, retrieval reachability, `correct` per question.

## Expected effect versus sampling error (ESTIMATES, not results)

n = 20 pairs. Expected accuracy 0.7-0.95 on both arms (ESTIMATE; Phase 0 P-small vs B-today differed by 0 questions on 10).
A 1-question tolerance is 5 pp and the binomial SE at 0.85 is 0.08, so the accuracy component cannot separate arms closer than
about 2 questions: this is a parity check, not an equivalence proof. The index-build saving in seconds is small (a zero-model dry
run on the same text measured about 0.07 s per registry index and about 0.06 s per bundle: ESTIMATE of about 1 s in total); the
call-count saving (4 builds against 20) is exact. A pass supports "the store-backed read costs and answers the same and does not
rebuild per request"; it does not support a speed claim.

## Pilot, checkpoints and stopping rules

1. **Zero compute first:** preflight (hashes, PDF pins, 20/20 verified, PREDECLARED sha256) and the plan + manifest check (every
   question plans `retrieval`, single-entity, no model call, empty forbidden manifest). A bad plan stops with 0 model calls.
2. **Pilot (bounded <= 900 s wall):** questions 1-5 (all four documents) on both arms. Stop on futility when the pilot already
   differs by > 2 questions in accuracy, or by > 2,000 prefill tokens (P minus B) on any question; the run then files
   `phase1_gate_tripped` naming `accuracy` / `prefill`. A pilot wall (sum of the 10 answer walls) > 900 s is a budget stop, not a
   gate: verdict `budget_stop_rebook`; the pilot rate replaces the estimate and the row is re-booked.
3. Otherwise the remaining 15 questions run and the verdict is computed (`parity_m5.score`). Gate components stay futility-only at
   the pilot look (no early GO). Every record is fsynced per answer; the run is resumable per `unit`.
4. Smoke (`FILLER_SMOKE=1`): question 1 on both arms only.

## Budget (ESTIMATE; the pilot replaces it)

About 5,400 s on the 27B GPU window (the box's provisional figure: 20 questions x 2 arms at about 100 s plus index load). The
Phase 0 loops on these same documents took 16-36 s per question at 6-8 calls, so the real figure is probably lower; the 900 s
pilot (10 answers) decides. Server: the Phase 0 np1 196k recipe unchanged (the window is not needed by these small prompts, but
the same recipe keeps kernel, KV type and draft setup identical to Phase 0).

## Pins

- Orchestrator: `9a0ab00d88ffdccab3dd9f25d7f5db26c93b7069` (the AGG-0 pin: planner, store, registry, context bundle at that commit).
- Research: the commit that adds this file (in the run spec; a file cannot carry its own commit).
- Phase 0 store (read-only source, never modified): `/mnt/raid0/llm/tmp/agg0-arms-cpu-20261110-r2/out/store.sqlite`, sha256
  `b5dba42f0398a99b6617e3bd519e3c9db0515b9d0d29e34a25eb103369464323`.
- Independent of AGG-1.1 Step B: it uses no L3 / L4 content and no Phase 1 ingest.
