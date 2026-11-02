# Prospective root-owned-source rebuild comparison

The prior result bound the output to executable and HIP bytes but not the
compiler-visible source. This successor compares a new baseline/candidate
build from archive SHA-256 `87e3cdc13b1859a31dfd3ced43c596e6b9653e2194b1ebd0d5eb6772593c04c2`
and patch SHA-256 `4147d2b65bbb416fdee1bf004d5447238933524bf3c087a69dcf6bbeca73b235`,
materialized into root-owned source trees with no group/other write access.
The build runs as node, with before/after complete source inventories and
recorded container toolchain hashes. This does not protect against privileged
root writers or pin every transitive toolchain dependency. The new build and
original retained binary are separate experiments.

After build completion, pin its receipt and all runtime library bytes before
launch. Run exactly FA_CASE 78 and 80, five fresh processes per arm, alternating
arm order. Explicitly set `GGML_CUDA_FA_MASK_SKIP=0` for both arms, after
clearing inherited `GGML_*` variables: this tests the guarded-loop candidate
with mask skipping disabled. Record this setting with each observation. Actual
saved Q/K/V/mask bytes must match each other and the canonical
input bytes preserved in `inputs/case78` and `inputs/case80` from the first
experiment. Capture raw output, rc, process identity, in-window own KFD/VRAM,
and mapped executable and ggml libraries for every run. A missing input,
mapping, positive in-window residency or output makes the corresponding arm
UNKNOWN. No missing observation becomes zero, and an incomplete attempt may
not be silently redrawn.

For each case/arm, report identical versus multiple output byte digests among
the five valid runs; summarize first varying indices and F32 spread separately.
The comparison tests repeatability of these exact binary/input combinations.
Any difference between rebuilt arms is evidence scoped to the single archived
source patch and the recorded build environment, not by itself a universal
kernel or production verdict. Preserve both original and successor evidence.
Write prospective source-side NaN-count observations with native device scope
only where a hashed ROCm device log, matching mapped HIP and positive in-window
own KFD/VRAM samples all establish it. There is no new grading rule or implicit
promotion.
