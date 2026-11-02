# Prospective producer-barrier A/B diagnostic

Compare the previously source-bound guarded-loop candidate (A) to the same
candidate plus the producer-barrier change (B), source commit
`18d030309a57071c32ffd0db42994ae473d2722a`; patch and separate immutable
build receipt are adjacent to this plan. B is a diagnostic experiment and does
not modify or promote production. Builds are root-owned and read-only to the
build UID, not immutable against root; the full transitive toolchain is not
pinned. Pin both build receipts, harness, predeclared input hashes and runtime
library bytes before launch, then verify mapped libraries at runtime.

Run cases 78 and 80, five fresh processes per arm, alternating A/B order,
with mask skipping disabled. Actual Q/K/V/mask input bytes must match the
canonical files from the previous experiment. Retain every raw output and
process return, log, mapping and in-window own KFD/VRAM sample. An arm is
UNKNOWN if a run is incomplete or any input, mapping or positive in-window
residency evidence is missing. No missing value is zero, and no failed run is
silently redrawn. Report the distinct output digests in each complete arm and
case, first differing positions and spread. A same-input decrease in observed
variation supports repeatability of these exact builds, not causation or a
production verdict. Capture prospective native claims at write time as before.
