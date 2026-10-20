# Human-recorded stratum: slot, status, and admission rule

**Status (2026-10-20): EMPTY. The human-recorded stratum does not exist yet.** CS-4 sets 2 (paralinguistic,
incl. hesitation) and 5 (multi-turn / interruption scripts) have not been recorded, so
`human_recorded.jsonl` has zero rows. Nothing in this corpus is human speech.

Consequence, fixed by the operator and recorded in `corpus.json` (`advance_eligible_strata`):
an arm in CS-47 / CS-49 may ADVANCE only on `human_recorded` audio. The synthesized stratum can reject or rank
an arm and can never advance one. Until this file has rows, no arm can advance; that is the intended outcome,
not a gap to be papered over.

## Row schema (`human_recorded.jsonl`, one JSON object per line; validated by `tt_verify.py`)

| field | meaning |
|---|---|
| `clip_id` | `tt1-h-NNNN`; unique |
| `stratum` | must be `human_recorded` |
| `speaker_id` | `human-<id>`; never `synth-*`; a speaker belongs to ONE split |
| `dialogue_id` | the scripted dialogue / take group; never shared across splits |
| `split` | `selection` or `confirmation`, assigned per SPEAKER (disjoint by speaker and dialogue) before any arm is scored |
| `label` | one of the four turn states in `label_contract.json` |
| `source_set` | `CS-4:set2` / `CS-4:set5` / other recorded source |
| `path` | repo-relative WAV (`audio/human/...`); 16 kHz stereo PCM16 in the same channel layout as the synthesized clips (ch0 user, ch1 assistant); mono recordings are admitted only with an explicit `assistant_channel: "none"` |
| `pcm` | `{sample_rate, channels, n_samples, pcm_sha256}`; bound to the DECODED samples |
| `gold` | same fields as the synthesized clips (`user_speech_start_s`, `user_speech_end_s`, `eot_gold_s`, `hold_window_s`, ...) |
| `self_correction` | `none` / `repair_in_progress` / `repaired` (+ spans) |
| `pair` | optional: `{pair_id, role}` when a minimal-contrast human pair is recorded |

Admission procedure when recordings land: add the WAV under `audio/human/`, append the row, run
`tt_verify.py seal` then `verify` (the gate refuses a speaker or dialogue in both splits, a synth speaker id in this
stratum, an unbound PCM hash, or a label outside the contract). Human recordings keep their own `speaker_id`
namespace so they can never be pooled with the synthesized voices.
