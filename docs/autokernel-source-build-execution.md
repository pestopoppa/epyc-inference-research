# Native source/build execution owner

`SourceBuildExecutionOwner` connects already-selected `build_recipe` work to the
existing concrete actor consumer and controller lifecycle, then binds the accepted
advice to an exact clean detached source snapshot and `BuildPlan`. The build runs as
a controller-owned worker; its nested configure/build processes retain the existing
worktree sandbox and cgroup rules.

The parent pins and rechecks the Python executable, loaded producer callables, module
bytes, admitted plan, deterministic private log/result namespace, terminal stdout,
sealed process receipt, log identity, source snapshot, produced binary and libraries.
Only then does it publish an immutable retention-enrollment artifact. Actor advice,
child stdout and that artifact are records, not grants.

An anchor build retains the serialized v1 preparation contract. A candidate build
uses v2 bytes plus the exact in-memory, identity-keyed source capability issued by
the successful source materializer. The capability binds the original source
preparation, target, controller lifetime, worktree, commit and tree digest; copying
it, parsing its bytes after restart, changing the tree or crossing owners refuses.

This slice does not make source patches, settle the scheduler transition, classify an
artifact as expirable, or authorize deletion. Generic `stage="build"` also does not
close the outstanding exact scheduler/backend/resource binding requirement. A crash
without an existing controller terminal or held-cost receipt remains unresolved and
must not relaunch. Durable enrollment activation is pending the separately reviewed
closed Journal authority event; ordinary and historical candidate integrations remain
unchanged.

