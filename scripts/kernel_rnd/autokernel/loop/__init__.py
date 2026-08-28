"""The AutoKernel discovery loop, rebuilt.

Normative specification of the loop's shape:
`docs/guides/agent-workflows/agent-loop-design.md` in epyc-root. If an implementation
here and that block disagree, the block wins until it is deliberately amended.

Custody is NOT rebuilt here. Promotion stays `docs/reference/kernel-freeze-runbook.md`
plus `kernel_freeze_scope.py` -- seven steps, ~100 lines, and they shipped v7, v8 and
v9. This package screens; it never promotes.
"""
