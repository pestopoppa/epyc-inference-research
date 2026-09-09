# Installed ROOT feed source closure

`loop.feed_runtime.InstalledFeedBinding` supplies the concrete ROOT projector to
`FeedRuntimeOwner`, which passes that loaded object to `EvidenceFeed`. The feed
uses the captured adapter's registered `autokernel-unified-arm-measurement`
projector and its captured `claim_tuple.grade`; it adds no grading rule.

## Closed source versions

An installed binding requires an absolute ROOT checkout path, explicit SHA-256
pins, and its owning epoch. The pins identify one of two exact sets:

| Loaded source schema | Required source files | Capability |
| --- | --- | --- |
| `epyc.autokernel.root_feed_projection.v1` | `autokernel_unified_arm.py`, `claim_tuple.py`, `frames.py`, `ledger.py`, `canonical.py`, `lattice.py` | Historical six-file closure; no final-v3 helper |
| `epyc.autokernel.root_feed_projection.v2` | Those six files plus `adapters/autokernel_final_trial.py` | Published parent-final v3 readback |

All paths are relative to `scripts/vidya/`; the arm module is also under
`adapters/`. `ROOT_SOURCES_V1` preserves the six-file set, `ROOT_SOURCES_V2` names
the seven-file set, and `ROOT_SOURCES` is the current seven-file installation
default. These are loader versions, distinct from native capture versions.

The binding does not discover hashes or infer source authority from a checkout
path. An application installs the explicit pin mapping. `root_source_sha256` is
defensively frozen; the loaded object's `source_sha256` and `source_schema`
describe exactly the selected closure. No feed configuration or historical
semantic receipt schema changes are needed to select either source set.

## Captured execution and late imports

Both direct loading and binding construction snapshot a caller-supplied mapping
before validating its closed set and digest syntax. The loader verifies every
selected file with the existing bounded, stable source reader before executing
any source. It compiles those captured bytes, not cached bytecode or a later
read of the same pathname.

The adapter's final-v3 imports are deliberately narrow: `REFERENCE_SCHEMA` and
`validate_final` from `autokernel_final_trial`, and the helper's reverse import
of `autokernel_unified_arm`. These resolve from the retained captured modules,
including after temporary synthetic `sys.modules` entries have been removed.
Unknown relative imports are refused. An ambient cached helper or a subsequent
on-disk change cannot substitute for the captured dependency. Cleanup removes
temporary module entries on successful loading and on import/execution failure.

An explicit six-file binding can still read supported v1/v2 records. A v3 record
requiring the unpinned helper refuses; the loader never borrows that helper from
the filesystem or ambient Python package. Installing seven files is prospective
consumer configuration, not a rewrite of the old six-file source identity.

## Evidence and recovery boundaries

The existing Journal cursor, projection durability, ledger ownership, and
restart rules remain unchanged. Actual diagnostic final-v3 captures can drain
through the installed feed and restart idempotently, but absent scientific
witnesses still produce no measurement tuple or effect finding. Loading the
helper grants no live claim, calibration/control warrant, or scientific pass.

`validation_semantic_adapter.PinnedRootProjection` is a separate, historical
v2-only semantic receipt path. Its fixed source pins, closed receipt grammar,
and compatibility-only authority are not widened by this feed loader. Any
future semantic receipt extension must version and preserve that contract
explicitly; the seven-file feed cannot retroactively promote those receipts.
