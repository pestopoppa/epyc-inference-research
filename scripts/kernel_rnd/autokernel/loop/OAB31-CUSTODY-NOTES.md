# OAB-31 custody fault: bounded fix and remaining work

This change stops the pool before another author overwrites a candidate when its
inside-iteration disposition cannot be retained or recorded. A sibling hold is
created and synced before recording, and blocks lane reset even if its contents
are incomplete or a dangling symlink. A no-patch result is accepted only for a
pre-author resume rejection whose recorded base still equals HEAD and whose git
status has no tracked, untracked, ignored, or submodule changes. All other
no-patch outcomes fail closed. The hold records the available resume and dispatch
identities for reconciliation; missing values remain missing.

A fatal run-aborted outcome has no resumed_from field and therefore does not
settle that checkpoint; the claim remains in flight. The hold is removed only
after the disposition archive and dispatch-registry operations return. A crash
between these independent writes can leave a valid archive and a held lane or
claim. There is no automatic archive/ledger reconciliation in this patch. Do not
release or replay a claim without classifying whether a commit occurred and
matching the original attempt and retained bytes.

The hold assumes the existing single-owner lane discipline: it fences this
version of reset_to_champion and lives outside the lane's git clean scope; it is
not a lock against an unmodified older process or an external actor deleting its
parent. A failure before the hold is created is still stopped by RunAborted;
the run.py reset path independently retains the dirty lane before reset, but a
separate old or custom reset path is outside this guarantee.

This patch is not completion of OAB-31. The required all-nonfatal-path artifact
coverage and a guard that fails for newly added unrecorded drop sites remain to
be implemented. The existing AST test checks wiring, not this invariant. Dynamic
run.py recorder callers were not resolved by GitNexus; its LOW graph result for
indexed symbols is not proof of the entire blast radius.
