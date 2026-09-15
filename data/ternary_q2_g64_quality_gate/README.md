# ternary Bonsai 27B Q2_g64 — strict-output quality gate

Measurement evidence made durable on **2026-09-15** (NIB2-73a). The master registry cited
`ternary_q2_g64_mi210_short_instruction_current_v7_20260718T151711Z/summary.json` from two
places, and it existed only as an UNTRACKED file in the shared clone
`/mnt/raid0/llm/epyc-inference-research` — so the evidence gate passed in that one checkout
and reported `MISSING` in every worktree. Copied byte-for-byte within the same clone
(sha256-verified); the untracked original was left in place.

| | |
|---|---|
| origin | `/mnt/raid0/llm/epyc-inference-research/data/ternary_q2_g64_quality_gate/` (untracked) |
| measured (UTC) | 2026-07-18 15:17:16 |
| made durable | 2026-09-15 |
| carried | 1 file, 1,944 bytes |

## What this evidence says — it is a FAILURE, and that is the point

Schema `bonsai_q1_quality_gate_execute.v1`. Status **`fail`**: passed 0, failed 1, total 1.

The single arm `ternary_q2_g64_mi210_short_instruction` ran on ROCm device 0 under
experimental-v7, with `llama-cli -t 96 -c 2048 -ngl 99 -n 48 --temp 0 --seed 1` against
`/mnt/raid0/llm/models/ternary-bonsai-27b/Ternary-Bonsai-27B-Q2_g64.gguf`. Prompt:

> In exactly six lowercase words, describe why benchmarks need held-out tests.

Generated: `prevents overfitting, ensures generalization, measures true performance.` —
failure reason *"stdout was not exactly six lowercase words"*.

This is the **retry on the refreshed current-v7 binary**, and it failed the same way the
original did. That is the whole evidentiary value: it shows the instruction-format blocker
is a property of the model, not of a stale binary.

## Registry claims this backs

`orchestration/model_registry.yaml` — line numbers as of 2026-09-15.

- **L8768** &nbsp;`roles.ternary_bonsai_27b_q2_g64.performance.quality_observation` — the
  prose citation. The strict-output gate passed only 6 of 8 probes (the short six-word
  instruction failed on both CPU and MI210); this artifact is the MI210 retry that
  "confirm[s] the instruction-format blocker survives the refreshed binary. **This blocks
  any role-ready claim.**"
- **L8826** &nbsp;`roles.ternary_bonsai_27b_q2_g64.performance.evidence` — the same
  artifact in the role's formal evidence list, backing its not-role-ready status.

Negative evidence is the kind most easily lost, because nobody re-runs a failure to get
the number back. Losing this file would not have weakened a performance claim — it would
have quietly removed the *blocker* on one, which is worse.

## What was NOT carried

The run directory holds 16 files / 48,401 bytes. Not carried: `gate.json` (16,849 B) and
`manifest.json` (18,735 B), the harness's full gate and manifest dumps; `commands.sh`
(4,871 B) and `exact_command.sh`; the `arms/` subtree (`result.json`, `stdout.txt`,
`stderr.txt`); `pre_rocm_smi.txt`/`post_rocm_smi.txt`; and the runner/process capture
files. `summary.json` is the cited artifact and it already embeds the full `llama-cli`
command line, the prompt, the generated text and the failure reason — everything needed to
read the claim. The rest remains untracked at the origin path above.

## The sibling run already in this directory

29 files under `ternary_q2_g64_quality_20260717Tcodex/` were already tracked before this
migration — the **original** 8-probe gate across CPU and MI210 arms, whose
`summary.json` and `throughput_observation.json` the registry also cites (those resolved
fine). The artifact carried here is the later **current-v7 MI210 retry** of the one probe
that failed, and it is the half that was untracked.

## Integrity

`SHA256SUMS` seals **30 files** — the 1 carried `summary.json` plus the 29 already
tracked, so it is a campaign-wide integrity record rather than a receipt for one commit.
`README.md` is deliberately not in it (documentation is not evidence, and hashing it would
make every doc edit break the seal), following the 2026-08-02 precedent in
`data/paddleocr_vl_receipt_extract_20260717T194415Z/`. The carried file was hashed after
the copy and compared against the original; verified `OK`. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/ternary_q2_g64_quality_gate/SHA256SUMS
```
