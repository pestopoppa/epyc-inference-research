# AutoKernel candidate transactions

This opt-in research consumer persists immutable candidate-state transitions without launching a
build, benchmark, verifier, worker, or production promotion. It reuses the campaign controller's
exclusive supervisor lease and the existing AutoKernel journal. There is no second WAL, journal lock,
integration branch, or writer namespace.

Each mutation validates its immutable `candidate_manifest.py` inputs and computes the next state before
writing. It then appends a fsynced `CANDIDATE_TRANSACTION/INTENT` bound to campaign/config/supervisor
identity, the expected prior state digest, complete operation payload, prepared object hashes, and exact
per-repository source refs. Only after that intent may it write content-addressed objects and create
immutable refs beneath `refs/autokernel/candidates/`. A durable `PREPARED` receipt distinguishes refs
the transaction may still create from a completed prepared set whose later absence requires repair.
A fsynced `COMMITTED` event makes the new state
authoritative; `candidate-state.json` is a replaceable derived pointer written afterward.
Immutable objects reuse the accepted measurement-capture `ArtifactStore`: private staging, exact
non-overwriting publication, directory durability, and non-creating verification are shared rather
than reimplemented by this consumer.

Recovery replays pure transitions and verifies event order, prior-state CAS, and the versioned completion
receipt. Historical reconstruction does not probe every archived object or Git ref and therefore does
not relabel their availability as current. Before `PREPARED`, an exact retry may create its still-absent
owned refs from the pinned source objects. After `PREPARED`, missing or externally changed objects/refs
refuse with a named repair; recovery never resets a peer, follows a moved branch, or publishes a
half-prepared state. An interrupted append poisons the current controller incarnation until
restart/replay. Missing or corrupt projection JSON does not override the journal and an identical
explicit retry can repair it. The controller scans the WAL once at startup,
then maintains the candidate event position and projection under its mutex; ordinary operations consume
only newly appended candidate events rather than rereading the WAL or rechecking every historical ref.

The Git adapter consumes commits already created through the accepted source-commit workflow. It does
not stage files or infer that a commit was built, measured, kept, or validated. Ref names derive from
hashed campaign, request, and repository identities, so caller text cannot select another namespace.
Production reference identity is frozen in the candidate state and manifests; no production Git ref is
read or moved by this module. Direct non-derived refs, the canonical frozen `llama.cpp`, `whisper.cpp`,
and `qwentts.cpp` checkouts, and checkouts on `production-consolidated-*` or `production-speech-*`
branches are refused. A distinct experimental linked worktree may share the repository object database
while retaining its own non-production checkout identity.

The existing command remains a non-mutating offline summary:

```text
PYTHONPATH=scripts/kernel_rnd python3 -m autokernel.loop.candidate_cli \
  --manifest candidate.json --row-set rows.json --batch batch.json --state state.json
```

Writes require an explicit subcommand, resolved campaign, owned store, request ID, and every exact source
repository. For example, initialization is:

```text
PYTHONPATH=scripts/kernel_rnd python3 -m autokernel.loop.candidate_cli init \
  --resolved-campaign resolved.json --store /private/campaign-store \
  --request-id initialize-v1 --repo research=/exact/repository/root \
  --manifest candidate.json --state candidate-state-initial.json
```

`integrate`, `start-batch`, `record-row`, and `complete` expose the corresponding pure transitions.
Initialization accepts only a new empty, unvalidated state and a manifest with no accumulated keeps; it
is not a legacy-state import or a way to mint a validated baseline.
There is intentionally no CLI command that advances `validated_candidate`: that API requires a
process-registered trusted row verifier and, when applicable, a trusted LOO verifier. JSON carrying a
`passed` label never supplies that authority. Required CPU/GPU rows and complete LOO obligations remain
those of the frozen batch; optional missing seed rows do not become production vetoes.

Successful validation advancement records a versioned transition receipt after the registered row and
LOO verifiers accept. Restart replay uses a dedicated structural transition reconstruction and that
durable receipt; it never calls the authorization API with synthetic passing callbacks. The reconstructed
pointer is explicitly historical. Current evidence eligibility remains `requires_live_registered_verification`,
and every new advancement still requires the actual registered verifier callbacks.

Legacy accumulator bundles remain readable by their existing owner and are never imported or upgraded
by this consumer. Operational candidate events produce no measurement claims. An engine that does not
support the new event/state schemas must refuse mutation rather than infer an older validated state.

Remaining integration work includes the real measurement verifier/provider registration, worker result
delivery, retention closure for unresolved intent roots, service/dashboard commands, and operator-owned
production promotion. The transaction layer grants no inference, build, broker, admission, deletion, or
production authority and does not complete AKU-05/07/09.
