# IQK matched-pair preparation manifest

`epyc.autokernel.iqk_matched_pair_preparation.v2` is a non-executing campaign
input contract. It publishes two fresh campaign roots atomically and never
builds, benchmarks, acquires a region, or mutates a journal.

V2 adds the required closed `measurement_frame` object:

| recipe | required work field | shape | evidence stage |
|---|---|---|---|
| `t1b.llama_cpu.llama_bench_prefill.v1` | `n_prompt: 512` | `pp512` | `bootstrap` or `heldout_bound` |
| `t1b.llama_cpu.llama_bench_decode.v1` | `n_gen: 128` | `tg128` | `bootstrap` only |

The calibration bundle must name the exact selected recipe and must license the
declared block count. A decode proposal must target exactly the `decode` regime
and `tg128` shape. Decode is intentionally bootstrap-only: its real terminal
journal is projected into the held-out receipt used by a subsequent prefill
pair; it cannot pre-bind the evidence it is intended to produce.

For v2 the physical-envelope template must equal the calibration declaration's
`physical_envelopes.aa_calibration` record except for the
schedule-dependent `measurement_frame_sha256`. The calibration cell uses the
canonical `model.gguf:{pp512|tg128}:aa_calibration` source shape. The publisher
converts that identity to the campaign's canonical `recipe_id:/absolute/model`
unit, preserves every physical fact, and records the source file/envelope
hashes plus a self-hashed conversion receipt in the result. This prevents a
pp512 work/shape envelope from being silently relabelled as tg128.

Both arm objects name fresh campaign/candidate/capture/intervention identities,
an independent native diagnostic source, the evidence stage, an optional
proposal-bound held-out outcome, and a new output directory. Preparation
generates a campaign-local hypothesis store for each arm from its final proposal
and binds the recipe-local regime and shape. The A/A control receives its own
proposal and authorization question rather than inheriting the intervention's.

The producer derives the shared schedule and T0 seeds, physical frame, provider
identity, complete factor vocabulary, and control proposal. The two arms must
differ only on `ggml_iqk`. Existing output paths, mismatched recipe calibration,
noncanonical frames, or a decode `heldout_bound` request refuse before either
output becomes visible.

Legacy `epyc.autokernel.iqk_matched_pair_preparation.v1` manifests remain
accepted with one exact meaning: the historical pp512 prefill frame. New
manifests should use v2 so recipe selection is explicit and hash-bound in the
result.
