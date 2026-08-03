"""autokernel — the AutoKernel research-loop substrate (epyc-inference-research).

AutoKernel is the campaign/control plane described by
`epyc-root/handoffs/active/autokernel-research-loop.md`. It proposes, builds, and
measures experimental kernel candidates against a frozen production anchor, and
its release-side job ends at a *release package* — a human executes every freeze
and cutover (§1.3, invariant 5).

This package holds the runtime-owner half of that loop. `schemas.py` is its
single source of truth: every other module (journal, evaluator, planner, release
packager) is written against those versioned contracts and must not invent its
own record shape. Nothing in this package writes to a production kernel tree, and
nothing in it carries freeze or cutover authority.
"""
