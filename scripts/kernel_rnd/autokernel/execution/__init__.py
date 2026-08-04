"""autokernel.execution — the layer that actually builds and runs things.

Everything above this package DECIDES: the evaluator scores evidence, the
planner proposes, the critic refuses, the release plane promotes. Nothing above
this package may execute, and that separation is deliberate — a module that
decides whether a number is real must not be able to produce the number.

This package is the other side of that line. It builds candidates in
experimental worktrees, runs measurements, and acquires the resource claims those
measurements are required to hold. Its authority is `P-AK-SEARCH-1`
(`measurement/protocols/kernel-research.md`), which permits ranking, retaining,
abandoning, branching and composing candidates *inside experimental worktrees on
the basis of measurements taken on those candidates*, and whose denial 8 states
the limits every module here inherits:

    no name-pattern process check, no signal to any process THE LOOP DID NOT
    LAUNCH, no host reboot, no privileged cache action outside the sanctioned
    path, **no inference run OUTSIDE A HELD CLAIM**.

`cpu_region_claim.py` is what makes the last clause satisfiable rather than
merely prohibitive: before this module there was no way for an agent to *obtain*
a CPU region claim, only to read one. `resource/device_claim.py` is its GPU
sibling and its design reference.
"""
