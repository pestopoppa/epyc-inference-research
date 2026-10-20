# CS-48 barge-in measurement harness

Instrument for CS-23 (policy under test) and CS-49 (live run). It builds nothing in `epyc-orchestrator`: the accepted
`src/voice` `VoiceController` + `CascadeBackend` are imported read-only (`bi_orch.py`, sha256 of every `src/voice/*.py`
file recorded in each `run.json`). No service is contacted; the only agent in this task is a scripted fake.

Interpreter: the orchestrator venv (`numpy scipy onnxruntime pytest`), always `-I`:
`/mnt/raid0/llm/epyc-orchestrator/.venv/bin/python -I -m pytest scripts/voice/barge_in/tests -q`.

| file | role |
|---|---|
| `bi_clock.py` | `RealClock` (live, real threads) and `VirtualClock` (cooperative deterministic scheduler; conformance) |
| `bi_session.py` | stimulus player (20 ms real-time frames), `TimedAudioQueue` (the controller's `AudioQueue`: "queue" hook, playback simulation, recorder), `TimedBackend` (timestamps `cancel_generation/vocoder/orchestrator`), `run_pair` |
| `bi_log.py` | event-log JSONL + validator (sequence, monotonic clock, hook order queue>generation>vocoder>orchestrator, exactly once) |
| `bi_score.py` | integrity gates and the four scorers |
| `bi_fdb.py` | FDB @3e799c4 `get_timing.py` + CS-45 patch (hash-checked), unmodified `eval_pause_handling.py` / `eval_smooth_turn_taking.py` runner |
| `bi_vad.py` | the ONE pinned Silero (CS-45 `tt_silero`; refuses on hash mismatch) |
| `bi_fake.py`, `bi_fixtures.py` | scripted fake agent (real `CascadeBackend`, clock-driven fakes) and LibriSpeech-derived stimuli |
| `bi_manifest.py` | scenario sources: FDB v1.5 sample folders, CS-45 manifest rows (PCM-SHA-256 enforced) |

## Run layout (FDB v1.5, time-synchronous, one sample count, one t0)
`input.wav` (user audio WITH the overlap) -> `output.wav` + `events.jsonl`; `clean_input.wav` (same user audio WITHOUT the
overlap) -> `clean_output.wav` + `clean_events.jsonl` (the FDB v1.5 non-overlap baseline: the model's response to the clean
input, NOT a bleed-free copy of `output.wav`); `run.json` (config echo, sha256 of all six files, controller source hashes).

## Scorers (no cloud judge, no torch, no network)
1. FDB v1.5 `get_timing` (pinned Silero only), BOTH merge-gap readings: `code` user 0.6 s / model 0.5 s, `comment` user 0.3 s / model 0.05 s. `t_stop = model_stop - overlap_start`, `t_resp = model_start - user_end`; the `code` reading is also run through FDB's own `process_folder`.
2. Event-log t_stop / t_resp / per-hook timestamps / chain latency and `respond | resume | continue | silent` labels. **NON-comparable to FDB's GPT-4o labels.**
3. HumDial timing: first-response delay (first model audio at/after user segment end) and timing rejection (a new response's audio began before the segment end), cross-checked on Silero(`output.wav`).
4. FDB v1 `pause_handling` / `smooth_turn_taking` adapter: word timestamps from pinned whisper.cpp (`tt_asr_words.PinnedWhisperCpp`), the unmodified FDB scripts run on a staged folder (only `input.wav/output.wav/output.json/turn_taking.json`: the scripts glob `*output.json`, so a `clean_output.json` would double count). Measurement-era change ME-CS48-2. The FDB smooth-turn script divides by zero when no item takes the turn; that is recorded as `script_error`, not hidden.

## Declared tolerances
Event-log quantities: exact (0 ns) on the virtual clock; real clock: wake-up jitter bound 30 ms on t_stop (test). Silero-derived
t_stop within 0.15 s, t_resp within 0.30 s (32 ms window + 30 ms pad + onset lag); HumDial cross-check 0.20 s. Observed errors are printed by the suite (`-s`).

## Pending / external
* FDB v1.5 stimuli (Google Drive `1DtoxMVO9_Y_nDs2peZtx3pw-U2qYgpd3`, 499 samples): operator manual fetch; `bi_manifest.scenarios_from_fdb15(<root>)` reads a local copy and refuses with "manual fetch pending" when absent.
* CS-45 manifest rows: `scenarios_from_cs45` documents the keys it needs; wire the real schema when CS-45 is committed.
