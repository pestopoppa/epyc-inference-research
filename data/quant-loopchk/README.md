# Loop and truncation secondary gate: preparation

This is offline preparation for QUANT-LOOPCHK-1 / -2, not inference evidence.
The frozen three-prompt pilot is `hard-prompts.json`, SHA-256
`90040cc66140a81319ebe4b99c4c54f12cfa349b5be9a3dd88768101f46a19cf`.
Its three tier-3 items and expected answers are copied from
`benchmarks/prompts/v1/math.yaml` at the research-repo base commit; that suite
was retired for discrimination at >=27B because it saturated on another
quality metric. Three items have extremely coarse resolution and may make this
pilot uninformative; do not claim broad coverage or a calibrated gate from them.

Fix cap at 8192 generated tokens per prompt for both arms, and identical
prompt formatting, sampling, seed, context, and production-store build. Archive
the exact requests, responses, actual finish reasons, effective settings, model
and build identities, and measured decode rate. Cost the window as prompt count
x 8192 / measured decode rate per arm plus 25% and any prefill/load overhead.
The two arms must use the same ordered prompt set. Human-review correctness
against the expected answers independently for every output; wrong is a separate
label and is not automatically inferred from text differences with Q8. Scoring
is an offline step:

```
python3 scripts/quant_loopchk/score.py data/quant-loopchk/hard-prompts.json CAPTURES.json --cap 8192 > result.json
```

CAPTURES is a JSON object with `reference` and `candidate` arrays, one record
per prompt: `id`, full `text` (include reasoning text if emitted separately),
`cap` (8192), `finish_reason` (`stop` or `length`, from the generation API),
and manually reviewed boolean `wrong`. Preserve raw responses separately.
Reject errors, missing records and unknown stop reasons rather than counting
them as completed answers. Counts overlap. The heuristic labels looped if four
consecutive copies of a 3–80-token phrase occur in normalized text (tokens here
are words/punctuation, not model tokens). Inspect all outputs: this heuristic
can miss loops or flag legitimate repetition; record review corrections and
redeclare the detector before measurement if corrections are needed. Length
means cut off even if a plausible answer is present. Do not infer truncation
from text length.

Next to KLD, flag when either candidate loop or cutoff count is strictly greater
than twice the Q8 count; a positive candidate count flags when the reference is
zero. If Q8 cuts off on more than 20% of prompts, stop for futility and
re-declare the cap before collecting a new paired run. Count wrong separately;
this pilot has no prescribed wrong-count threshold. A flag is a secondary gate,
not a KLD replacement. Preserve the per-prompt labels, counts, threshold,
futility decision and all raw evidence. Report an uninformative verdict if the
sample, detector or reference behavior cannot support calibration.

Synthetic fixture only (no model):

```
python3 scripts/quant_loopchk/score.py data/quant-loopchk/hard-prompts.json data/quant-loopchk/synthetic-captures.json --cap 8192
```

The first candidate must be labelled looped and the third cut off. These
invented answers are fixture data, not model results.

## Blocking input for QUANT-LOOPCHK-2

The named reference exists at `/mnt/raid0/llm/models/Qwen3.8-27B-Q8_0.gguf`.
The box does not identify a path or quantization for the incumbent low-bit
Qwen3.8-27B target. No such target was found in the model inventory; DFlash2
low-bit GGUFs are draft models and cannot be substituted. The operator must
identify the exact same-base incumbent and its path, or approve a revised
comparison. Consequently there is no honest exact MODEL-MISSING path or safe
specific download/quant command to state yet. Once identified, record the
exact path and acquisition or quantization command if absent. No generated or
rehearsed row is claimed: its model, runtime, and window cannot be bound until
that input is specified. No row has been published or claimed.
