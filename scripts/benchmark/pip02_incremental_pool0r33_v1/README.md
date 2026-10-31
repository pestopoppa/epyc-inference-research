# PIP-02 incremental Paddle scoring, R29 catalog

This additive package freezes a source-bound English Phase C scorer in a separate namespace. It preserves existing `odl_bench` source paths and their score receipts. The copied Python modules are the static transitive relative-import closure of the scorer and receipt writer; GT, plan, metric and model-config pins are unchanged. The catalog accepts only the exact R28/R29 head producers and exact historical fresh-s1 source map; all original capture, grant, request, response and finish checks remain in force. It never runs inference or reconstructs a missing warrant.

`PIP_SCORE_OUTPUT` must name an existing directory outside this source package. `run_one.py SHARD` writes exclusively there, including a prospective score-start, result and score-receipt; it refuses existing results and source-tree outputs. It uses only files adjacent to its own immutable source path for imports and scorer code. `tests_native.py` is a host-evidence integration test against existing captured records, not a portable unit test or a reason to rerun a producer.

`epoch_watch.py list paddle` discovers input epochs; `prepare paddle SHARD HASH` creates a write-once epoch under `scoring-shards-20261030/paddle-watch-r29/`. Its baselines include successfully reopened old epoch receipts to avoid rescorings of unchanged fully complete shards. Changed or formerly refused records require a new epoch. A prospective scorer still validates every record and selected source itself. The watcher uses only `ready_shards.py` hints requiring all five native finish receipts; hints are not a score, provenance warrant or quality verdict.

`watch.sh` owns a new watcher lock and writes only the new epoch namespace. It verifies the deployed package manifest and each source file before discovery and each invocation, uses the existing R6 CPU admission and bounded prep supervisor, and does not stop or overwrite the old watcher. Start it only from a versioned immutable deployment with `PIP_DEPLOY_MANIFEST_SHA` equal to the SHA-256 of its parent `SHA256SUMS`. The log, service identity and scoring outputs belong outside the deployment. A newly scored epoch is not automatically a canonical project filing; reopen its receipt, merge unique pages with prior verified receipts, and file via the ROOT owner.

Selected producer sources and capture paths remain exact native record inputs and must remain available for receipt replay. Missing, moved, changed or unmatched records are refused; unknown exposed regions are not redrawn. A prior receipt cannot be made durable by editing its paths or inventing an original source.

The watcher uses an unchanged snapshot of the existing ROOT PIP-02 adapter and shared claim tuple solely to replay legacy receipts, independent of removable checkouts. This is the existing measurement projection and grader, not a new rule. Snapshot source bytes are in the package manifest.

R30 version adds only the literal pool2 producer, exact immutable source registry and launcher, original pool-group checks, and its separate watcher output namespace. All native finish, capture, grant, source, and region checks remain in force.

Head recovery version adds exact fresh-s0/s2 origins and a prospective offline assembly receipt for pages 0 and 34. It never invents a historical finish; native model responses, original sources, prospective source inputs, timing, and reconstructed markdown are all independently bound. No grading rule changes.

Tail31 version adds only the exact original pool3 groups 110, 122, 134, 146 producer ID, source registry, launcher, and isolated output layer; existing recovery and origin checks are unchanged.

Pool1 R32 adds the exact original pool1 groups, producer ID, source registry, launcher, and isolated output layer; original native checks remain in force.

Pool0 R33 adds the exact six never-started original groups, producer ID, source registry, launcher, and isolated output layer; original native checks remain in force.
