# turn-taking-cs45-v1: frozen turn-taking evidence manifest (CS-45)

Owner task: CS-45 in epyc-root handoffs/active/conversation-stack.md. Consumers: CS-22, CS-23, CS-46..CS-49, CS-2.
Measurand note (epyc-root repo): docs/reference/speech/turn-taking-measurands.md.

| file | what it freezes |
|---|---|
| `label_contract.json` | (a) 4-state turn labels, HumDial class names with our own definitions, Badcat-style interrupt classes, talking-to-others / background-speech overlap classes, the self-correction class (annotated, not scored here) |
| `corpus.json`, `SHA256SUMS` | every file's SHA-256, constants of the clip assembly, counts, `advance_eligible_strata = [human_recorded]` |
| `manifest/recordings.jsonl` | source recordings (16 kHz mono PCM16 WAV members of `audio/synth/user.tar` and `audio/synth/assistant.tar`; archives rather than loose files because the repo PII pre-commit hook text-scans every staged blob under 1 MB) with PCM-SHA-256, TTS provenance, ASR word timing |
| `manifest/clips.jsonl`, `manifest/pairs.jsonl` | (c) each pair member = recipe (recording, trim, cut, assistant context) + PCM-SHA-256 of the stereo clip |
| `manifest/splits.json` | selection / confirmation partition (disjoint by speaker, dialogue, source recording, assistant context) |
| `manifest/human_recorded.jsonl`, `manifest/HUMAN_SLOT.md` | the human-recorded stratum slot (EMPTY until CS-4 sets 2/5 are recorded) |
| `external_crosscheck.json` | (d) external material by hash and id only (TurnBench dev; ECHO / HumDial / FDB v1.5 data absent) |
| `pins.json`, `pins/silero_vad_v6.onnx` | (e) the single pinned Silero, FDB / TurnBench commits, ASR, TTS kernel |
| `measurement_era_changes.json`, `scorer_patches/` | (e) every scorer substitution as a measurement-era change, with patch and file hashes |

## How the pairs are built (one recording, matched)

Each pair is ONE rendered user recording (production qwentts.cpp, Base model, speaker-embedding clone of a LibriSpeech
test-clean reference clip, CPU, read-only). Clips are stereo, 16 kHz, ch0 = user, ch1 = assistant context. Only the SOURCE
recordings are stored; a clip is a deterministic function of (recording, recipe), re-materialised and hash-checked by
`tt_materialize.py` / `tt_verify.py`.

- CUT_IC / CUT_WAIT: member b is the same recording cut at a word boundary (cut snapped to a quiet valley near the ASR word boundary), member a runs to its natural end. 10 ms raised-cosine fade at the end of both. Both get exactly 2.0 s of digital silence. The assistant context is identical.
- CTX_BC: the same acknowledgement waveform in both members; a is placed after the assistant finished a question, b over an assistant monologue that keeps going.

## Current build (v2, 2026-10-20)

112 main pairs (CUT_IC 69, CUT_WAIT 13, CTX_BC 30) and 27 reserve pairs, every clip gated by the independent gates (qa/qa_gates_report.json). 37 planned pairs
(of 176 candidates incl. 36 alternate sentences) had no variant that passed and are absent (qa/selection_report.json); nothing was waived.
v1 (fe730c481) is superseded: see qa_evidence_pre_fix/root_causes.json.

## Reserve stratum (backfill only)

`manifest/reserve/` and `audio/reserve/user.tar` hold further synthesized pairs (stratum `reserve`) with the same
selection rules and voice partition as the main set, built so that the human review of the main set cannot push the corpus
below the 100-pair floor. Policy (enforced by `tt_verify.py`, recorded in `corpus.json`): a reserve pair counts ONLY
through `manifest/backfill.json` (a recorded review rejection of a main pair plus a backfill of the same family and split),
never otherwise, and never as evidence for advancing an arm. Reserve ids, dialogues and recordings are disjoint from the
main set; reserve speakers keep their main split; reserve assistant contexts are main recordings reused inside the same split.

## Verify / refuse

```
cd scripts/voice/turn_taking
PY=/mnt/raid0/llm/epyc-orchestrator/.venv/bin/python
$PY -I tt_verify.py verify --corpus ../../../data/turn-taking-cs45-v1      # exit 0 only if all checks pass
$PY -I tt_pins.py check --fdb-checkout <FDB@3e799c45> --tb-checkout <turnbench@38a6f87>
TT_REAL_CORPUS=$PWD/../../../data/turn-taking-cs45-v1 $PY -I -m pytest -q tests/   # refusal (mutation) cases
$PY -I tt_materialize.py --corpus ../../../data/turn-taking-cs45-v1 --out /mnt/raid0/llm/datasets/turntaking-cs45-v1/clips
```

Checks: file digests, PCM-SHA-256 of every recording, every clip re-assembled and re-hashed, pairs are exact splices of one
recording (sample-identical shared span, exactly matched trailing silence), >= 100 English pairs, selection/confirmation
disjointness derived from the rows, strata separation and the advance policy, label-contract conformance, external material
hashes/ids only, pin hashes. Mutation cases (tests/test_tt_corpus.py, tests/test_tt_scorer_patches.py): altered clip
(stale seal and resealed), moved cut, unmatched trailing silence, speaker or dialogue across splits, swapped label, dropped
below 100 pairs, mixed strata, widened advance policy, wrong Silero hash in the pins, swapped Silero file (loader raises
PinMismatch), tampered scorer patch, wrong scorer base hash or commit, external audio copied in.

## Rebuild from renders

tt_plan.py (deterministic job list) -> tt_render_pipeline.py (render variant k under a region-lock build claim; ASR-A cut search) -> `tt_build2.py candidates`
-> tt_qa_gates.py (ASR-B, a different model) -> `tt_build2.py final` (first variant whose clips all pass; nothing waived) -> `tt_verify.py seal`.

## Independent QA gates (v2)

The v1 corpus passed its own QA yet 12 of 20 sampled pairs were defective by ear (root causes: `qa_evidence_pre_fix/root_causes.json`). In v2 admission
rests on `tt_qa_gates.py` (U1 exact word sequence, U2 suffix and cut-point energy, U3 non-empty speech, A1 assistant continues to its natural end),
run with ASR-B = faster-whisper large-v3, which neither places the cuts (ASR-A = faster-whisper large-v3-turbo) nor scores (whisper.cpp large-v3-turbo).
`tt_verify.py` check 13 refuses a corpus whose `qa/qa_gates_report.json` is missing, covers different audio, or contains a failing clip.
