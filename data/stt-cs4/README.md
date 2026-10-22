# stt-cs4: STT corpus rules, dataset registry, reference protocol (CS-4.STT1, CS-8.STT7 inputs)

Owner tasks: CS-4.STT1 and CS-8.STT7 in epyc-root `handoffs/active/conversation-stack.md`. Style follows
`data/turn-taking-cs45-v1/`. Files here:

| file | what it holds |
|---|---|
| `manifest.schema.json` | schema of one set manifest (ids and hashes only) |
| `registry.yaml` | dataset registry rows: Earnings22-Cleaned-AA, Contextual Earnings-22 |
| `scripts/speech/stt_bakeoff/manifest_validator.py` | refusal gate (`python3 -m scripts.speech.stt_bakeoff.manifest_validator <manifest.json> [--prior <old.json>]`, non-zero on refusal); tests `test_manifest_validator.py` hold the refusal fixtures |

This repository is PUBLIC. Corpus audio, real reference text and private clip content are never committed; only ids, sha256s and labels.

## (a) Labels and the no-pooling rule

Every set carries exactly one label:

- `public-anchor` - LibriSpeech, FLEURS, Earnings22-Cleaned-AA, Contextual Earnings-22. Contaminable by arms trained on public data. Ordering and sanity only.
- `private` - operator-recorded, speakers absent from any public corpus.

Headline gates come only from `private` sets. Public-anchor and private sets are NEVER pooled in any aggregate (`check_no_mixed_strata` refuses), and `cleaned` and `verbatim` reference tracks are never pooled either. A set without a label is refused.
Reason (arXiv:2608.19936, unreplicated preprint): best-WER open models reproduce reference errors, and the behaviour weakens on fresh same-domain speakers, so an i.i.d. split of a public set cannot be the confirmation stratum.

### Where private data lives (DP-A)

**DECIDED 2026-10-22 (operator): DP-A (b).** Private and confirmation strata live in the PRIVATE repo
`https://github.com/pestopoppa/epyc-stt-corpus-private`, cloned at `/mnt/raid0/llm/data/stt-corpus-private`
(config: `private_custody.json`, helper: `scripts/speech/stt_bakeoff/private_custody.py`). That repo is never mirrored
publicly. This repository and epyc-root keep hash manifests only (`utterance_id`, `pcm_sha256`, `reference_text_sha256`,
labels); audio, references and transcripts are never committed here.

The validator still refuses a private set whose `audio_root`, `reference_root`, or any per-utterance path lies inside any
git work tree (walks parents for a `.git` file or directory), with ONE explicit exception: a path inside the verified
private-repo clone (nearest git tree is the clone root and its `.git/config` names the private remote). Public trees, nested
foreign trees, wrong-remote clones and symlinks into public trees stay refused. Sealed per-utterance scoring output
(`custody.seal_scoring_run`) still goes outside every git tree, the private clone included. Manifests carrying raw reference
text or audio bytes are refused. The earlier default (a) (on-disk under `/mnt/raid0/llm/data/` with `SHA256SUMS`) is superseded.

### Rater policy (DP-C)

**DECIDED 2026-10-22 (operator): single labelled rater for now.** Manifests carry `reference_protocol_label` and `rater_count`;
the labels allow a second corrector to be added later without invalidating existing references (a higher `reference_version`
plus an `edit_log` entry).

## (b) Registry

`registry.yaml` records pins, licence notes (CC-BY-SA-4.0 upstream; conflicting card tags noted), known defects and exclusion rules. Internal evaluation only; audio is fetched by pinned revision and never committed. A pin of `null` means "to be pinned at first download".

## (c) Reference transcription guide

Adopted from the Artificial Analysis cleaning conventions. One FROZEN reference text per clip; a cleaned variant is never pooled with a verbatim one.

1. Verbatim: write what was spoken.
2. Restore real repetitions (the speaker said it twice: it is written twice).
3. Drop partial-word fragments.
4. Keep the speaker's grammar as spoken (accented or non-standard grammar is not corrected).
5. Fillers (DP-B default b): scored as ordinary words; the manifest carries `fillers_in_ref` per utterance. Transcribe or omit once, consistently, then freeze.
6. Every edit pass is logged in `edit_log` with its `reference_version`.
7. `reference_text_sha256` changes only through a `reference_version` bump that rescores every arm. The validator (`--prior`) refuses a hash change without a higher version and a logged pass.

## (d) Confirmation reference protocol

1. Model draft (an ASR draft, never an arm under test where avoidable).
2. Independent human correction(s) with adjudication of disagreements.
3. Freeze: `frozen_manifest_sha256` = sha256 of the canonical manifest JSON (sorted keys, compact separators, field itself excluded), recorded BEFORE any arm is scored.
4. Rater count (DP-C): default (b) single-rater, declared with a label (`rater_count: 1`, `reference_protocol_label: "single-rater"`); recommended (a) two independent correctors for en/it with adjudication (`rater_count: 2`, `adjudication` block).
5. A confirmation set (`confirmation: true`) without a frozen manifest hash, or without the rater label and count, is refused; editing it after the freeze changes the hash and is refused.

## (e) Noise variants

MUSAN-style noise variants (10, 5, 0 dB SNR) of the en/it sets are produced for STT scoring only, each as its own derived set with its own manifest (same label as its parent, never pooled with clean or with other labels). Paralinguistic sets still require real recordings; synthetic noise does not stand in for them.

## Fitting screen (CS-8.STT7)

`scripts/speech/stt_bakeoff/fitting_screen.py` builds (never runs) the `benchmark-optimization ref-disagreement` (commit `9b55a563c1356b8fc33ec8199698c54edb7305c1`) and `score_voxpopuli_ref_errors.py` invocations and turns per-item decisions into accept-ref with a Wilson 95% interval. It reports and never gates (`gates: false`); higher accept-ref = more fitting; point estimate strictly above 0.10 labels an arm "public WER not trustworthy; private set only"; it/de/fr are `unmeasured`, never clean.
