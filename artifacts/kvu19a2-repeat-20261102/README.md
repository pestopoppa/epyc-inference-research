# Repeated model-free attention diagnostic — 2026-11-02

Under the committed predeclaration, 20 fresh processes ran: five per arm for
each of FA_CASE 78 (D=576/512) and 80 (D=512/512). All exited zero, all had
own-PID in-window KFD and positive VRAM samples, and all had matching live
ggml mappings. The actual submitted input-byte SHA-256 sets matched across
all ten processes for each case. `inputs/` preserves one exact original input
set per case; `native/` preserves every raw output, log, process sample,
native result and producer-written sidecar.

Baseline outputs were byte-identical across five processes in both cases.
Candidate outputs had five distinct SHA-256s in each case. Post-hoc F32
localization found 16,384/32,768 varying candidate elements in case 78
(maximum within-arm spread 0.0011727512) and 4,096/16,384 in case 80
(maximum 0.0006961077); no baseline element varied. No output was nonfinite.
The exact flattened ranges are in `difference-audit.json`.

The reconstructed source declares output shape `[DV, heads, query]`; under
that layout the varying 512-float blocks correspond to even head indices on
all four query rows in case 78, and even head indices on the final two query
rows in case 80. This coordinate interpretation is a source-based inference:
the original compiler-visible source tree was mutable, and source-to-binary
binding remains unproven. The measurement establishes behavior of the pinned
binaries and inputs, not a cause, correctness tolerance, model quality or
promotion. A next experiment should rebuild baseline and the single candidate
patch from independently immutable inputs, capture its build and libraries,
then repeat these exact inputs with per-head/query tracing.

Native result SHA-256 `56038798a67cc1b34060a81592369bde040628d0c1635dc99f4dfd4286b44afc`;
capture record SHA-256 `46ff49ebe8f83ff6bff31c8e8530d00523f9042c45442cdd71e3fa42033258d0`;
sidecar SHA-256 `491dee9a0c6262aee7253fad651c4aca7aa60c3207113bba06dfbfa8b5d1a5be`.
No source/build warrant is inferred.
