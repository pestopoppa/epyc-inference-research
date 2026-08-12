"""autokernel.release — the AK5/AK6 release plane (plan, gate, seal, package).

WHY THIS PACKAGE EXISTS
-----------------------
Everything in `autokernel.evaluator` and `autokernel.controller` runs under
`P-AK-SEARCH-1`, whose scope clause says in terms that it *"does NOT apply to T3
or any release gate"*. The release instruments therefore cannot live there, and
`evaluator.api.admit_tier()` refuses `T3`/`T4` by name so that a release-shaped
decision can never be produced under a search protocol. This package is the other
side of that seam.

**The cardinal rule of this plane: AutoKernel never freezes and never cuts over.**
It produces a release PACKAGE that a human executes. A kernel freeze crosses four
human-only trust boundaries (`MEASUREMENT.md:140-142` — the freeze itself, the
era-registry row, the AutoPilot baseline apply, and the pinned-path list), so
there is no such authority to hold, delegate, or flag. Any code path here that
writes a production branch, moves a stable kernel symlink, writes an era-registry
row, or applies an AutoPilot baseline is a defect, not a feature (design §1.3,
§11.2, invariant 5).

Nothing in this package writes any file, starts, stops or signals any process,
builds anything, or runs inference.

Only the read-only plan compiler is bound here. The readiness, T3, packager and
release-local preflight modules are operator-triggered explicit imports; keeping
them out of this package initializer prevents Campaign #1 from acquiring release
authority through an incidental ``release`` import.

Owning design: `epyc-root/handoffs/active/autokernel-research-loop.md`, phase AK5
(§3.2, §10, §11).
"""
from __future__ import annotations

from . import plan

__all__ = ["plan"]
