# Continuous evidence in the standalone runtime

The installed path is `startup_factory` v2 → `StartupManifest` v2 →
`standalone_inputs.runtime_factory` → `StandaloneRuntime` →
`UnifiedCampaignDriver.tick` → the actual unified planner. V1 snapshot manifests
and factory requests retain their closed fields and behavior.

## Configuration and installed ownership

`epyc.autokernel.startup_factory_request.v2` replaces the v1 `evidence_index`
artifact with an explicit `evidence_feed` object. Its `providers` object names
only `lifecycle` and `readiness`; the feed binding owns any evidence verifiers.
All other factory inputs still come from the sealed campaign export, matching
production export, canonical recipes, and explicit user choices.

An example feed object (paths and epoch are explicit deployment choices):

```json
{
  "schema": "epyc.autokernel.standalone_evidence_feed.v1",
  "binding_id": "installed-autokernel-feed",
  "expected_epoch": "deployment-owned-epoch",
  "source_root": "/path/campaign-store/journal",
  "corpus_root": "/path/native-artifacts",
  "ledger_path": "/path/dedicated-vidya-feed/ledger.jsonl",
  "store_root": "/path/evidence-projection",
  "reader_id": "autokernel-vidya-evidence-v1",
  "max_events": 32,
  "max_bytes": 4194304,
  "max_seconds": 0.25,
  "max_shards": 64,
  "max_projection_entries": 10000
}
```

The source must be the configured controller's `store/journal`, including at
direct runtime composition. Source, corpus, projection, and dedicated ledger
roots must not overlap or traverse symlink aliases. Projection writers cannot
overlap the controller store. The ledger is a single-writer deployment resource;
this connection creates no permission to share an existing external ledger writer.

The application installs `InstalledFeedBinding` in `ProviderRegistry.evidence_feeds`.
It supplies the actual ROOT path, exact hashes of the six local ROOT source files
(`canonical`, `lattice`, `claim_tuple`, `frames`, `ledger`, unified-arm adapter),
and current epoch. The manifest names the binding and asserts its expected epoch;
it cannot create ROOT imports, callables, grades, or provider authority. Configured
experiment-plan epochs must match. This is an immutable per-runtime epoch;
changing it requires a new startup/recovery boundary, not silently rewriting a cache.

The loader reuses the existing pinned semantic loader's bounded stable source
read, compiles those captured bytes without bytecode caches, and routes the fixed
ROOT-local imports to that captured closure. It does not use ambient cached
`sys.modules` implementations. Temporary private registration exists only while
constructing dataclasses; captured module objects/functions retain their own
imports afterward. The installed v2 path never falls back to `_load_vidya`.

The optional finding projector and scope/use/full-result verifiers must be a
complete application-installed set with an explicit support-rule identity.
Without that set, genuine ROOT per-arm tuples still enter the canonical ledger,
but do not become prospective effect findings or ranking authority. No compatible
production effect writer/verifier is invented by this connection.

Factory invocation, using a real v2 request and an existing parent output directory:

```bash
PYTHONPATH=/path/research/scripts/kernel_rnd python3 -m autokernel.loop.startup_factory \
  --request /path/feed-factory-request.json --out-dir /path/new-startup-bundle
PYTHONPATH=/path/research/scripts/kernel_rnd python3 -m autokernel.loop.unified_driver \
  --config /path/new-startup-bundle/startup.json --dry-run
```

The factory also records its selected interpreter/package path and exact dry-run
command. Dry-run opens no feed, SQLite database, controller store, provider, or
model; absent installed identifiers remain unavailable. Listening execution uses
the application-owned `provider_registry` parameter, not an arbitrary CLI import.

## Execution-thread lifetime

The main thread constructs typed configuration, controller, runtime, and a stable
`FeedEvidenceView`. Main-thread controller recovery does not construct the feed.
On the first fresh execution-thread tick, the owner captures ROOT sources, opens
and recovers SQLite/reader leases, then drains one bounded batch before controller
readiness and planning. Every later fresh tick drains on that same thread.

A readiness attempt captures a finite source frontier from its first bounded
read. No planning occurs until that prefix is projected and durable proof is
complete. A later unrelated tail does not move that attempt's target forever;
the next fresh tick captures its own target. Actual projected frontier and owner
generation are bound into the proposal snapshot. Queued invalidations inside the
captured prefix therefore cannot be skipped.

Typed busy, proof-pending, and cooperative projection-deadline failures retain the
owner and use runtime backoff. Corrupt/uncertain durable state refuses admission
and reports recovery required. There is no static snapshot fallback. An exact
pending transaction retry neither drains nor replans its already-issued catalog.

`runtime.run` closes the feed in `finally` on its creating execution thread,
outside the controller mutex. Main-thread `close` requests stop and reports
`shutdown_incomplete` while execution-thread cleanup is outstanding or fails.
Direct tick usage pins its caller thread; another thread cannot drain or close
that SQLite connection. Partially failed construction closes SQLite and both
leases on the creating thread. A clean SIGTERM leaves the campaign terminally
drained: restarting its feed does not authorize resuming that campaign generation.

## Stable retrieval and bounded eviction

The planner consumes one concrete `PlanningEvidence` bundle. Prompt retrieval and
proposal snapshot share one verifier evaluation and evidence generation; callback
mutation is refused. Existing `retrieve` and `proposal_snapshot` APIs remain
available, but combining two independent calls is not the planner's consistency
boundary.

The stable feed view resolves the feed's current internal index for each bundle.
Evictions may rebuild that bounded cache without stranding the planner on an old
object. Before support grading/top-k, relevant dependency and mandatory-signature
keys are checked in the durable SQLite eviction index. Missing relevant evidence
makes retrieval incomplete; the condition is included in result/support digests,
proposal snapshots, and `fences_for(proposal)` for the unchanged pure
`EvidenceIndex.admit_cached` check. Evicting a refutation, invalidation, or
quarantine cannot restore support. Unrelated targets remain usable. Historical
eviction is not a global permanent cap-and-stop policy.

Projection durability precedes the sole owned Journal ACK. Generic
`Journal.commit_cursor` is not a second acknowledgement path. A crash after
projection but before owned ACK leaves the cursor unchanged; recovery checks and
retries that exact row without duplicate canonical frames.

This proves continuous ingestion/planning composition and owner lifecycle, not
scientific eligibility, provider installation, production export repair, selected
profile execution, or the remaining contained source/build authoring feedback
route. The existing accounting strict expected failure remains explicit.
