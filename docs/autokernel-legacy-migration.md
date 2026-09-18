# AutoKernel legacy migration snapshots

`autokernel.loop.legacy_migration` imports one explicitly named legacy campaign into a
dedicated, caller-owned artifact directory. It reuses `accumulate.load_bundle(...,
read_only=True)` for accumulator replay, reads actual `ExperimentStore` rows through a
read-only SQLite connection, and publishes through the existing content-addressed
`ArtifactStore` primitive.

The output is historical input only. Its authority block is always
`measurement=unknown_legacy`, `candidate_state=not_created`, `validated=false`, and
`serving_eligible=false`. Recipe, keep, instrument, configuration, source and artifact
facts are retained as recorded or explicitly marked unknown. Migration never initializes
`CandidateTransactions`, never changes the source journal, and never creates a second WAL.

## Operations

Run from the repository with `PYTHONPATH=scripts/kernel_rnd`:

```text
python3 -m autokernel.loop.migration_cli dry-run \
  --import-id old-campaign-1 --campaign-id imported-old-campaign-1 \
  --source /owned/legacy/state --destination /owned/migration/snapshot \
  --source-repo /owned/source/repo --anchor-commit <exact-commit> \
  --config campaign.json --artifact builds/legacy-binary

python3 -m autokernel.loop.migration_cli import <the-same-options>
python3 -m autokernel.loop.migration_cli inspect /owned/migration/snapshot/<locator>.json
```

Dry-run performs the complete bounded read, compatibility, ancestry and frontier checks
without creating the destination or repairing legacy state. Import requires the
destination parent to exist; the destination itself may be absent or an owned mode-0700
directory dedicated to that exact import. The source and repository cannot alias or
contain the destination. Symlinked components and descendants are refused.

Reads default to 64 MiB and 100,000 experiment rows and can be lowered with
`--max-bytes`/`--max-records`; both are strict positive integer bounds and cannot exceed
the installed v1 reader limits. The serialized snapshot must fit the selected byte bound.
The importer freezes and rechecks all named source files,
journal shards, SQLite sidecars, configuration and artifacts. A source that changes or
adds/removes a durable frontier member during assembly is refused rather than mixed.
An active SQLite WAL/SHM is refused until its owner produces a stable checkpoint, so
even dry-run cannot create or negotiate a source-side SQLite file.
Artifacts are identified and hashed; this is not a recursive copy and the source remains
byte-for-byte unchanged.

Publication is content-addressed and deterministic. The same request against the same
frontier returns the same locator. A retry recovers the existing owned stage. A changed
request or replaced source conflicts with the dedicated destination and is refused; use a
new import ID and empty owned destination for a genuinely different historical snapshot.
Before either preview or publication, the writer runs the installed closed v1 parser over
its own output. Publication's dedicated-root check and nested public `ArtifactStore.write`
share `ArtifactStore.exclusive()`, so another instance cannot enter between them.

## Compatibility and rollback inspection

Snapshot v1 declares its accepted reader version and unsupported live-mutation status.
`inspect` verifies the content against its actual `ArtifactStore` locator and produces
only a read-only historical view through the installed fixed v1 parser. Callers cannot
declare additional accepted versions. Unknown fields/newer versions refuse; an
unknown legacy champion-of-record also refuses instead of substituting the anchor. An old
v1 reader can parse a newly emitted v1 snapshot, while a future version must fail closed.
A compatible engine is required before any later admissions; the snapshot itself never
authorizes campaign resume, candidate advancement, serving claims, cleanup, deployment,
or promotion.
