# NUMA placement and CPU serving characterisation — `P-BENCH-PLACEMENT-1`

Provenance documentation written **2026-09-15** (NIB2-73a). Everything below the campaign
directory is unchanged since **2026-07-30**: twelve commits between 10:32:55 and 21:05:07
UTC that day added all 63 files, and no commit after 2026-07-30 touches any of them
(`git log --follow -- data/numa_placement`). This file exists because the
evidence-durability gate wants a campaign-level `README.md` and `SHA256SUMS`; the
substantive documentation was written by the campaign itself and lives one level down, at
[`20260730-P-BENCH-PLACEMENT-1/README.md`](20260730-P-BENCH-PLACEMENT-1/README.md), which
carries the figure → file map, the units contract and the per-figure rep counts. **Read
that first.** This file adds only what a reader coming from the registry needs and cannot
get there: which claims depend on this tree, which arm each file belongs to, and the three
record defects the inner index does not mention.

| | |
|---|---|
| scratch origin | none — committed straight into the repo, first commit `fcfe0b8c` |
| measured (UTC) | **2026-07-30**, one working day |
| dates taken from | the twelve commit dates (`fcfe0b8c` 10:32:55 → `84067a6e` 21:05:07) and the directory name `20260730-`, which carries a **date only, no time**. Not file mtimes — in a fresh worktree those are checkout times |
| documented | 2026-09-15 |
| carried | **63 files, 237,980 bytes** — the whole campaign, in one subdirectory, nothing withheld |
| structure | a strict script+results pair convention: 17 `*.sh` each with a matching `*_results.txt`, 9 `.py` helpers, 4 `cc_*.log` raw server logs, 3 `.md` |
| kernel | production-consolidated-v8 @ `67a433bf45a8a091d83b4ea0b32ff0735fd51800` (`llama-server --version` → 10107) — `20260730-P-BENCH-PLACEMENT-1/README.md:8-9` |
| host | EPYC 9655, **NPS4**: node0 `0-23,96-119` · node1 `24-47,120-143` · node2 `48-71,144-167` · node3 `72-95,168-191`; `numa_balancing=0`; region-lock held `q0..q3` as `role=bench` — same lines |

## What this directory is — and the label to avoid

This is **the shared attestation directory for protocol `P-BENCH-PLACEMENT-1`**, not "the
defect's evidence" and not "the fix's evidence". Both live here, in the same files,
distinguished by **arm label**, and calling the tree one or the other would mislabel rows
that three registry fields and `MEASUREMENT.md`'s own claim-grammar exemplar depend on:

| arm | what it is |
|---|---|
| `A_prod` — `taskset 0-47,96-143`, no `numactl` | **the defect, measured.** The NPS2-era core list that straddles two of the four real NPS4 nodes with no memory policy at all |
| `B_halfint` — same cores, `numactl --interleave=0,1` | intermediate diagnostic arm |
| `C_fullint` — `taskset 0-95`, `numactl --interleave=all` | **the canonical target configuration**, and the baseline everything later is quoted against |
| `prodopt_results.txt` | the **production-optimal reference** — arm C plus each role's registry `acceleration` recipe. These are the numbers the registry attests |

The defect was ~2x on **exactly two roles**. `handoffs/active/numa-placement-defect-20260730.md:1-6`
corrects its own earlier title in those words — *"The loss is **~2×, not ~3×** … and it is
**two named roles, not 'production CPU inference'**"* — naming `frontdoor` (10.83 → 23.36
tok/s, arm A at 46% of canonical, `20260730-P-BENCH-PLACEMENT-1/highn_results.txt`) and
`ingest_long_context` (12.42 → 22.92, `matrix4_results.txt`). `worker_general` and
`architect_general` were already wired canonically; their matrix rows are counterfactuals,
not live regressions (same handoff, `:36-38`).

**The fix is not in this tree.** That handoff's header still reads *"Status: OPEN —
diagnosis COMPLETE and measured; the **wiring fix is now WRITTEN (uncommitted, not
reloaded)**"* (`:8-9`), and its most recent dated section (2026-09-03) reports that the real
mechanism was found later — INF-70/C7, page cache defeating `--interleave=all` — with that
remedy *merged and awaiting activation*. So arm C is a **validated recipe measured under a
ratified protocol**, not a fix whose sufficiency this evidence proves.

## The one retracted sub-thread — internal, and the index does not mention it

`RETRACTION-ngram-20260730.md:3` — **"Status: RETRACTED, same day, before any recipe
changed."** Two claims committed that evening, `f36483cd` ("ngram-mod composed with
draft-mtp is 2.52x on the 35B at depth") and `bd1f086e` ("2.80x"), were withdrawn at 21:05
by `84067a6e`. The effect was a harness artifact: each cell sent the same prompt r times at
`temperature=0.3, seed=42` against **one live server**, so run 2 hit the prompt cache and
`ngram-mod` — which drafts by matching text already in the context — drafted the model's own
previous answer verbatim (mean accepted draft length `3.58 → 15.88`, acceptance `1.000`).
Corrected on run 1 only, across 16 cells, the gain *"spans **−17.4 % to +2.7 %, centred on
zero**"* and *"**No role should enable `ngram-mod` on this evidence**"* (`:62-67`).

Three things a reader must know about this:

1. **It is entirely internal to this directory.** `84067a6e` changes exactly one file, the
   retraction note; `f36483cd` and `bd1f086e` likewise touch only files here. No sibling
   campaign is implicated.
2. **It touches no placement figure.** Every `A_prod`/`B_halfint`/`C_fullint` arm and every
   `prodopt` arm uses `draft-mtp` or no speculation, and the retraction's own control
   analysis exonerates exactly those: *"All 25 `draft-mtp`-only and `none` cells sit flat
   between 0.92× and 1.08×"* (`:38`), because `draft-mtp` drafts from model weights and a
   warm context cannot help it. The 10 files it invalidates are `ngram_results.txt`,
   `ngram2_results.txt`, `ngram.sh`, `ngram2.sh`, `mkreal.py`, the four
   `cc_frontdoor_q35_p*.log` logs and `ctxcurve.sh`.
3. **The inner index does not list any of them.** `README.md` there was last rewritten at
   18:02 (`825a6139`), *before* the ngram commits (18:15, 18:53) and the retraction
   (21:05). A reader who trusts that index will not learn the retraction exists — which is
   why it is called out here.

The retraction also states four amendments to `P-BENCH-PLACEMENT-1` for context-reading
drafters (`:77-90`). **UNVERIFIED — those four amendments are not in the codified
protocol.** [docs/protocols/numa-placement-measurement-protocol.md](../../docs/protocols/numa-placement-measurement-protocol.md) contains no occurrence
of "ngram", "context-reading" or "drafter"; the amendments exist only as prose in the
retraction note. Reconciling a ratified protocol is a human amendment and is not done here.

## Two further record defects, not covered by the inner caveats section

`20260730-P-BENCH-PLACEMENT-1/README.md:75-97` lists five caveats of its own (wrong gemma
artifact in `modelref_results.txt`, wrong 80B artifact in `ctx80b_results.txt`, fleet
aggregates taken with settings production does not use, a context-curve parser that
mislabels prefill as decode, and varying rep counts). Two more were found on 2026-09-15 and
are recorded here rather than by editing a 2026-07-30 artifact:

- **`matrix2_results.txt` is truncated.** 9 lines, no `DONE` sentinel — the only
  `*_results.txt` here lacking one — ending at a bare
  `##### architect_general_qwen35_122B_Q4KM #####` / `--- A_prod … ---` header with no data
  rows. The inner index (`:59`) says it establishes "placement arms, gemma + 122B"; the
  gemma arms are there (16.37 / 23.43 / 39.03 ± tg128), the **122B half is absent**. Use
  `matrix4_results.txt` for the 122B arms.
- **`quadfleet_results.txt`'s locality column is wrong.** It reports `0.0%` on-own-node for
  *both* the mmap and `--no-mmap` arms, contradicting `locverify_results.txt`, which
  measures the mechanism cleanly (`25.6%` under shared mmap → `100.0%` with `--no-mmap`).
  Its decode figures (mmap 40.91 vs `--no-mmap` 52.13 tok/s) are the usable part.

## Registry claims this backs

`orchestration/model_registry.yaml`. A repo-wide grep for `numa_placement` and
`P-BENCH-PLACEMENT` returns **exactly three lines**, all the same attest string. Key paths
are the stable reference; line numbers are as of 2026-09-15.

- **L1880** &nbsp;`roles.frontdoor.performance.optimized_tps_attest`
  > "[P-BENCH-PLACEMENT-1, n=3, 2026-07-30, attest data/numa_placement/20260730-P-BENCH-PLACEMENT-1/prodopt_results.txt]"

  attesting `optimized_tps: 40.22` (L1879, *"full instance, interleave=all, draft-mtp
  n_max 4"*) → `prodopt_results.txt:17-20`, `q35_PRODSPEC`, `median=40.22 min=40.11
  max=40.29`, acceptance `0.746`. `baseline_tps` is `null` with the comment *"spec-dec-off
  NOT measured for this model. Do not fabricate."* — correct: **there is no `q35_nospec`
  arm in the file.**
- **L5289** &nbsp;`roles.worker_general.performance.optimized_tps_attest` — same string,
  attesting `optimized_tps: 56.86` (L5288) → `prodopt_results.txt:1-4`, `gemma_PRODSPEC`,
  `median=56.86 min=56.52 max=57.19`, acceptance `0.866`; and `baseline_tps: 37.63` (L5287)
  → `:5-8`, `gemma_nospec`, `median=37.63 min=37.18 max=37.78`. Both match to the digit.
- **L5589** &nbsp;`roles.worker_summarize.performance.optimized_tps_attest` — same string,
  attesting `optimized_tps: 40.22` (L5588). **Inherited, not separately measured**: the role
  has no arm in `prodopt_results.txt` and the registry records `speedup:
  shares_frontdoor_gguf`, so the frontdoor figure transfers by identity of the GGUF.

Both L1879 and L5588 sit under a registry comment recording what this evidence *changed*:
*"CORRECTED 2026-07-31. Both rows previously held the same … figure, so `optimized_tps` was
NOT an optimized number — routing and the cost model decided on a value ~40% below the
ratified production optimum."* That correction is the campaign's main production consequence.

**Two of the five `prodopt` arms are in the registry under a different protocol id.** The
122B arms (`q122_PRODSPEC` 24.00, `q122_nospec` 11.30, `prodopt_results.txt:9-16`) appear as
`roles.architect_critic.performance.optimized_tps: 24.00` / `baseline_tps: 11.30`
(L2441-2442) and as `roles.architect_general.performance.superseded_model_history_122b.optimized_tps:
24.00` (L2328) — matching to the digit but attributed to the *published stack measurement
record §01, 2026-07-31*, not to `P-BENCH-PLACEMENT-1`. A future auditor should not read
those as unattested figures; the measurement is here.

Beyond the registry, epyc-root `MEASUREMENT.md:59` registers the protocol (`✅ 2026-07-30`,
grade B) and `MEASUREMENT.md:84` uses this directory as the **canonical claim-grammar
exemplar**, citing `prodopt_results.txt` by path. The executable contract is
[docs/protocols/numa-placement-measurement-protocol.md](../../docs/protocols/numa-placement-measurement-protocol.md) in this repo (`Status: ✅ RATIFIED
2026-07-30`); epyc-root `measurement/protocols/bench-cpu.md` points back to it rather than
restating it. The owning handoff row is `handoffs/active/inference-research-index.md:49`
(INF-43); its open next action is the re-run of the 27 confounded E5 cells audited in
[e5_rederived.md](20260730-P-BENCH-PLACEMENT-1/e5_rederived.md).

## Integrity

`SHA256SUMS` seals all 63 tracked files, including the campaign's own inner
`20260730-P-BENCH-PLACEMENT-1/README.md` and [RETRACTION-ngram-20260730.md](20260730-P-BENCH-PLACEMENT-1/RETRACTION-ngram-20260730.md) — those two are
dated artifacts of the campaign, not descriptions of it, and a reader needs to know they are
the versions the claims were made against. Only this top-level `README.md` is excluded:
documentation is not evidence, and hashing it would make every doc edit break the seal,
following the 2026-08-02 precedent in
`data/paddleocr_vl_receipt_extract_20260717T194415Z/`. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/numa_placement/SHA256SUMS
```

No PII and no credentials: zero email addresses, zero credential-shaped strings, and the
four `cc_*.log` files contain only llama.cpp server telemetry — no prompt or completion text
at all. Prompt corpora were generated to `/mnt/raid0/llm/tmp/` by `mkprompts.py` and
`mkreal.py` and are not carried here. The files do carry first-party operational detail:
~173 local absolute paths under `/mnt/raid0/llm/`, the v8 kernel commit SHA, loopback
`127.0.0.1` with ephemeral bench ports, and the host's NUMA core map. The stock llama.cpp
banner *"CORS is set to allow all origins ('*') and no API key is set"* appears in the logs;
it is a property of a throwaway bench server, not a leaked credential.
