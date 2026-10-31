# Current KB catalog qd-v1 document-token census, 2026-10-31

An enrolled model-free replay of the frozen current catalog found **16,243 of
35,280 rows (46.040%) strictly above 256 model-input tokens**, and 16,283
(46.154%) at or above the cap. Full-token nearest-rank p50/p95/max were
229/1,111/1,929. These are this current cohort, not the 2026-08-23 cohort.

All 35,280 rows had FTS text that matched its catalog row's path, heading,
16-hex text hash and preview; none were unknown. The catalog had 35,376 FTS
rows, of which 96 were orphans outside the declared catalog-row denominator.
Per-row records in `rows.jsonl.gz` contain row ID, original text SHA-256,
status and counts, never text or paths. Compressed and uncompressed SHA-256s
are recorded in `SHA256SUMS`; the uncompressed rows SHA-256 is
`f63732bd80a03f684abdfd89fa437d573f3a383411d0cc505dcc3769550db47f`.

Across known rows, capped model-input tokens total 6,347,102 of 12,378,893
uncapped tokens, **51.274%**. Both totals include the document prefix and
special tokens, exclude padding, and are token counts rather than character
counts. Retained token-offset coverage of document text is 21,058,629 of
41,200,048 tokenizer-covered Unicode codepoints, **51.113%**. The denominator
is uncapped token-covered codepoints, excluding prefix/specials and any text
without token offsets; raw document codepoints total 41,200,048 separately.
These coverage figures are aggregate ratios of summed counts. Higher coverage
means less content omitted by the token cap.

The declared historical `qd-v1` replay uses `[D] ` prepended to original text,
native tokenizer specials, no lowercasing and a 256-token cap; uncapped and
capped tokenizers are independent copies. Snapshot and inputs were pinned
before tokenization by `RUN-MANIFEST.json` and the superseding R6 manifest.
The catalog does not stamp a tokenizer SHA: the pinned current tokenizer is
**not proof of the tokenizer used to produce historical vectors**, nor does
this replay establish behavior of current `qd-v2` code. It neither measures
retrieval quality nor supports adoption; G13 remains open.

The owner applied prospective source enrollment before capture; its note
records registry commit **pending at application**, not an invented commit.
Native short, Unicode, special-token, 255/256/257-boundary and unknown-row
controls passed before capture (2 tests); `controls.stderr` and
`controls-supervisor.json` record a post-capture rerun of the unchanged tests.
The census ran on admitted CPUs 126–127 under the
30–31 mutex with OMP=1: 45.687 seconds, 150,245,376-byte sampled peak owned-tree
RSS, exit 0, children drained. Independent row/aggregate verification passed,
including the 96-orphan count. Sampling may miss RSS transients.
