# Producer-barrier A/B diagnostic result

The predeclared paired diagnostic completed 20 fresh processes, with canonical
input hashes, positive in-window KFD/VRAM samples and mapped ggml evidence in
every run. Both arms produced five distinct output digests for each of cases
78 and 80. Moving the producer exit after the barrier did not restore bitwise
repeatability for these exact builds and inputs; this is not a general kernel
or promotion verdict.

The numerical audit found only finite F32 values. In case 78, 16,384 of
32,768 elements varied within each arm; the maximum observed within-arm
absolute spreads were 0.00275414 for the original guarded candidate and
0.00213913 for the barrier change. In case 80, 4,096 elements varied in each
arm, with maxima 0.000563511 and 0.000680187 respectively. The audit specifies
all observed ranges; these descriptive differences are not a prespecified
threshold, attribution, or proof of improvement.

The original guarded-candidate root build receipt is
`../kvu19a2-repeat-20261102/root-build/build-receipt.json` (SHA-256
`3b327e7b729e02e38ed377847f4ea890d42852aa4e2300f681be07e86742511f`).
The barrier build receipt is `receipt.json` (SHA-256
`1d00db9e5fcf394d59d7548d1269fdfa2333bc3559da4a69e446ad1efc85dafc`).
The candidate source commit is `18d030309a57071c32ffd0db42994ae473d2722a`;
`sync.patch` is the only change from the candidate reproduction commit
`dcad4ce92`. Source snapshots were root-owned and read-only to the build UID,
but root access and the full transitive toolchain were not sealed. Raw results,
producer-written beliefs, and process records are in `native/`; the run's
original path remains in `kvu19a2-sync-live-20261102` in the research worktree.
